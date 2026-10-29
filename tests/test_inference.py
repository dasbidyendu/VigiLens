"""Tests for the stateful inference pipeline and model streaming state."""

import numpy as np
import torch

from models.anomaly_model import VigiLensModel
from pipeline.inference import InferencePipeline


class FakeModel:
    def __init__(self):
        self.index = 0
        self.reset_calls = 0
        self.last_masks = None

    def reset_stream(self):
        self.index = 0
        self.reset_calls += 1

    def process_frame(self, rgb_frame, ir_frame, rgb_mask, ir_mask, rgb_available, ir_available):
        self.last_masks = (rgb_mask, ir_mask)
        score = None if self.index == 0 else float(self.index)
        self.index += 1
        return {
            "anomaly_score": score,
            "prediction": None,
            "hidden_state": None,
            "rgb_available": rgb_available and rgb_frame is not None,
            "ir_available": ir_available and ir_frame is not None,
        }


class FakeSegmentation:
    def __init__(self):
        self.calls = []

    def extract_person_mask(self, frame):
        self.calls.append(frame.shape)
        return np.ones(frame.shape[:2], dtype=np.uint8)


def test_pipeline_first_frame_has_no_score_and_smoothing_is_causal():
    model = FakeModel()
    segmentation = FakeSegmentation()
    pipeline = InferencePipeline(
        model=model,
        segmentation=segmentation,
        calibration_info={"threshold": 0.5},
        smoothing_window=2,
    )

    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    ir = np.zeros((12, 16), dtype=np.uint8)

    first = pipeline.process_frame(rgb, ir)
    assert first["score_available"] is False
    assert first["raw_score"] == 0.0
    assert first["smoothed_score"] == 0.0
    assert first["is_anomalous"] is False

    second = pipeline.process_frame(rgb, ir)
    assert second["score_available"] is True
    assert second["raw_score"] == 1.0
    assert second["smoothed_score"] == 1.0

    third = pipeline.process_frame(rgb, ir)
    assert third["raw_score"] == 2.0
    assert third["smoothed_score"] == 1.5
    assert third["is_anomalous"] is True

    # One YOLO pass per frame; IR reuses the resized RGB mask (no second pass).
    assert len(segmentation.calls) == 3
    rgb_mask, ir_mask = model.last_masks
    assert rgb_mask.shape == (24, 32)
    assert ir_mask.shape == (12, 16)


def test_pipeline_ir_only_does_not_run_segmentation():
    model = FakeModel()
    segmentation = FakeSegmentation()
    pipeline = InferencePipeline(model=model, segmentation=segmentation)
    ir = np.zeros((12, 16), dtype=np.uint8)

    result = pipeline.process_frame(None, ir)
    assert result["rgb_available"] is False
    assert result["ir_available"] is True
    assert segmentation.calls == []


def test_pipeline_reset_clears_state():
    model = FakeModel()
    pipeline = InferencePipeline(model=model, segmentation=FakeSegmentation(), smoothing_window=3)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    pipeline.process_frame(rgb)
    pipeline.process_frame(rgb)
    assert pipeline.frame_count == 2
    assert len(pipeline.score_history) == 1

    pipeline.reset_stream()
    assert model.reset_calls == 1
    assert pipeline.frame_count == 0
    assert len(pipeline.score_history) == 0


def test_pipeline_process_video_aggregates_results():
    class Source:
        def __init__(self, count):
            self.count = count
            self.index = 0

        def read(self):
            if self.index >= self.count:
                return None, None
            self.index += 1
            return np.zeros((8, 8, 3), dtype=np.uint8), None

    pipeline = InferencePipeline(
        model=FakeModel(), segmentation=None, calibration_info={"threshold": 0.5}
    )
    stats = pipeline.process_video(Source(4))
    assert stats["total_frames"] == 4
    assert stats["anomaly_count"] == 3
    assert stats["anomaly_rate"] == 0.75
    assert len(stats["results"]) == 4
    assert stats["avg_inference_time"] >= 0.0


def test_real_model_streaming_state(monkeypatch):
    model = VigiLensModel(device="cpu", load_extractors=False)
    model.eval()

    def fake_extract(rgb_frame, ir_frame=None, rgb_mask=None, ir_mask=None):
        return torch.randn(1, 2048), torch.randn(1, 2048)

    model.extract_features = fake_extract
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    first = model.process_frame(frame)
    second = model.process_frame(frame)
    assert first["anomaly_score"] is None
    assert second["anomaly_score"] is not None
    assert np.isfinite(second["anomaly_score"])
    assert first["rgb_available"] is True
    assert second["rgb_available"] is True

    model.reset_stream()
    assert model.hidden_state is None
    assert model.previous_prediction is None
