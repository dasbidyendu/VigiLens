"""
Stateful real-time inference pipeline for VigiLens.

The pipeline owns the per-stream state:

* one YOLO segmentation pass per RGB frame (the mask is reused for IR),
* the model's LSTM hidden state,
* the one-frame prediction latency,
* causal temporal smoothing, and
* the calibrated anomaly threshold.

Frames are OpenCV BGR arrays.  The model's feature extractors perform the
BGR -> RGB conversion internally.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from models.anomaly_model import VigiLensModel
from models.segmentation import YOLOSegmentation
from pipeline.frame_source import MultimodalFrameSource

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Stateful inference pipeline for video anomaly detection."""

    def __init__(self,
                 model: VigiLensModel,
                 segmentation: Optional[YOLOSegmentation],
                 calibration_info: Optional[Dict] = None,
                 smoothing_window: int = 10,
                 smoothing_method: str = "moving_average",
                 threshold: Optional[float] = None,
                 ema_alpha: float = 0.3):
        self.model = model
        self.segmentation = segmentation
        self.calibration_info = calibration_info or {}
        self.smoothing_window = max(1, int(smoothing_window))
        self.smoothing_method = smoothing_method
        self.ema_alpha = float(ema_alpha)

        calibrated = self.calibration_info.get(
            "threshold", self.calibration_info.get("threshold_std", 0.5)
        )
        self.threshold = float(threshold if threshold is not None else calibrated)
        self.default_threshold = float(calibrated)

        self.score_history: deque = deque(maxlen=self.smoothing_window)
        self.ema_score: Optional[float] = None
        self.frame_count = 0
        self.total_inference_time = 0.0
        self.last_mask: Optional[np.ndarray] = None

        logger.info(
            "Inference pipeline initialized: threshold=%.6f smoothing=%s/%d",
            self.threshold, self.smoothing_method, self.smoothing_window,
        )

    def reset_stream(self) -> None:
        """Reset LSTM state, smoothing history and counters for a new video."""
        if hasattr(self.model, "reset_stream"):
            self.model.reset_stream()
        self.score_history.clear()
        self.ema_score = None
        self.frame_count = 0
        self.total_inference_time = 0.0
        self.last_mask = None
        logger.info("Inference stream reset")

    def process_frame(self,
                      rgb_frame: Optional[np.ndarray],
                      ir_frame: Optional[np.ndarray] = None,
                      rgb_mask: Optional[np.ndarray] = None,
                      ir_mask: Optional[np.ndarray] = None) -> Dict:
        """
        Process one synchronized frame pair.

        Args:
            rgb_frame: BGR frame (H, W, 3) or None for IR-only input
            ir_frame: IR frame (H, W) or (H, W, 1) or None
            rgb_mask: Optional precomputed person mask (avoids a second YOLO pass)
            ir_mask: Optional precomputed IR mask

        Returns:
            Dictionary with raw/smoothed scores, decision, masks, prediction,
            and timing information.
        """
        if rgb_frame is None and ir_frame is None:
            raise ValueError("At least one of rgb_frame or ir_frame is required")

        start_time = time.time()
        rgb_available = rgb_frame is not None
        ir_available = ir_frame is not None

        # One segmentation pass.  The resulting mask is reused for IR (resized
        # to the IR resolution), which avoids running YOLO twice per frame.
        if rgb_available and rgb_mask is None and self.segmentation is not None:
            rgb_mask = self.segmentation.extract_person_mask(rgb_frame)

        if ir_available:
            if ir_mask is None and rgb_mask is not None and rgb_available:
                ir_mask = cv2.resize(
                    rgb_mask, (ir_frame.shape[1], ir_frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
        else:
            ir_mask = None

        result = self.model.process_frame(
            rgb_frame,
            ir_frame,
            rgb_mask,
            ir_mask,
            rgb_available,
            ir_available,
        )

        actual_rgb = bool(result.get("rgb_available", rgb_available))
        actual_ir = bool(result.get("ir_available", ir_available))
        raw_score = result.get("anomaly_score")
        score_available = raw_score is not None

        if score_available:
            smoothed_score = self._smooth_score(float(raw_score))
            is_anomalous = smoothed_score > self.threshold
        else:
            # The first frame of a stream has no previous prediction.
            smoothed_score = 0.0
            is_anomalous = False

        inference_time = time.time() - start_time
        self.total_inference_time += inference_time
        self.frame_count += 1
        self.last_mask = rgb_mask

        return {
            "raw_score": float(raw_score) if score_available else 0.0,
            "smoothed_score": float(smoothed_score),
            "score_available": score_available,
            "is_anomalous": bool(is_anomalous),
            "rgb_available": actual_rgb,
            "ir_available": actual_ir,
            "rgb_mask": rgb_mask,
            "ir_mask": ir_mask,
            "prediction": result.get("prediction"),
            "hidden_state": result.get("hidden_state"),
            "inference_time": inference_time,
            "frame_index": self.frame_count - 1,
        }

    def _smooth_score(self, score: float) -> float:
        """Apply causal smoothing to an anomaly score."""
        if self.smoothing_method == "moving_average":
            self.score_history.append(score)
            return float(np.mean(self.score_history))
        if self.smoothing_method == "exponential":
            if self.ema_score is None:
                self.ema_score = score
            else:
                self.ema_score = self.ema_alpha * score + (1.0 - self.ema_alpha) * self.ema_score
            return float(self.ema_score)
        return float(score)

    def process_video(self, frame_source: MultimodalFrameSource) -> Dict:
        """Process every frame from a frame source and return aggregate stats."""
        self.reset_stream()
        results = []
        anomaly_count = 0

        while True:
            pair = frame_source.read()
            if pair is None:
                break
            if isinstance(pair, tuple):
                rgb_frame, ir_frame = pair
            else:
                rgb_frame, ir_frame = pair.visible_frame, pair.ir_frame
            if rgb_frame is None and ir_frame is None:
                break

            result = self.process_frame(rgb_frame, ir_frame)
            results.append(result)
            if result["is_anomalous"]:
                anomaly_count += 1

        avg_time = self.total_inference_time / self.frame_count if self.frame_count else 0.0
        return {
            "total_frames": self.frame_count,
            "anomaly_count": anomaly_count,
            "anomaly_rate": anomaly_count / self.frame_count if self.frame_count else 0.0,
            "avg_inference_time": avg_time,
            "fps": 1.0 / avg_time if avg_time > 0 else 0.0,
            "results": results,
        }

    def get_fps(self) -> float:
        if self.total_inference_time > 0 and self.frame_count > 0:
            return self.frame_count / self.total_inference_time
        return 0.0
