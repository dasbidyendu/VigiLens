"""
Frame preprocessing for visible and IR/thermal inputs.

This module mirrors the preprocessing used by ``models.feature_extractor`` so
that ad-hoc preprocessing and model preprocessing cannot drift apart:

* visible frames are OpenCV BGR -> RGB, resized, scaled to [0, 1], then
  ImageNet-normalized;
* IR frames are single-channel, resized, scaled to [0, 1], replicated to three
  channels, then ImageNet-normalized (the IR encoder is an ImageNet-pretrained
  ResNet50).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _as_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    frame = frame.astype(np.float32)
    if frame.max() <= 1.0:
        frame = frame * 255.0
    return np.clip(frame, 0, 255).astype(np.uint8)


def _ensure_rgb(frame: np.ndarray) -> np.ndarray:
    frame = _as_uint8(frame)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        frame = cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Unsupported visible frame shape: {frame.shape}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _ensure_gray(frame: np.ndarray) -> np.ndarray:
    frame = _as_uint8(frame)
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 1:
        return frame[:, :, 0]
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported IR frame shape: {frame.shape}")


def _resize_mask(mask: Optional[np.ndarray], size: Tuple[int, int]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.shape[:2] != size:
        mask = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.float32)


class FramePreprocessor:
    """Preprocess visible and IR frames for the ResNet50 encoders."""

    def __init__(self,
                 target_size: Tuple[int, int] = (224, 224),
                 normalize: bool = True,
                 ir_as_rgb: bool = True):
        self.target_size = tuple(target_size)
        self.normalize = normalize
        self.ir_as_rgb = ir_as_rgb

    def preprocess_visible(self, frame: np.ndarray,
                           mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Return a normalized (H, W, 3) RGB float32 array."""
        if frame is None:
            raise ValueError("Visible frame is None")
        frame = _ensure_rgb(frame)
        h, w = self.target_size
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = _resize_mask(mask, (h, w))
        if mask is not None:
            frame = frame.astype(np.float32) * mask[:, :, None]
        frame = frame.astype(np.float32) / 255.0
        if self.normalize:
            frame = (frame - IMAGENET_MEAN) / IMAGENET_STD
        return frame.astype(np.float32)

    def preprocess_ir(self, frame: np.ndarray,
                      mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Return a normalized IR array: (H, W, 3) or (H, W, 1)."""
        if frame is None:
            raise ValueError("IR frame is None - IR input unavailable")
        frame = _ensure_gray(frame)
        h, w = self.target_size
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = _resize_mask(mask, (h, w))
        if mask is not None:
            frame = frame.astype(np.float32) * mask
        frame = frame.astype(np.float32) / 255.0
        frame = np.repeat(frame[:, :, None], 3, axis=2)
        if self.normalize:
            frame = (frame - IMAGENET_MEAN) / IMAGENET_STD
        if not self.ir_as_rgb:
            frame = frame[:, :, :1]
        return frame.astype(np.float32)

    def preprocess_pair(self,
                        visible_frame: np.ndarray,
                        ir_frame: Optional[np.ndarray] = None,
                        visible_mask: Optional[np.ndarray] = None,
                        ir_mask: Optional[np.ndarray] = None
                        ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        visible = self.preprocess_visible(visible_frame, visible_mask)
        ir = None
        if ir_frame is not None:
            try:
                ir = self.preprocess_ir(ir_frame, ir_mask)
            except ValueError as exc:
                logger.warning("IR preprocessing failed: %s", exc)
                ir = None
        return visible, ir


def validate_ir_frame(frame: np.ndarray) -> bool:
    """Heuristically check that a frame is plausibly single-channel IR data."""
    if frame is None:
        return False
    if frame.ndim == 3 and frame.shape[2] > 1:
        logger.warning("Frame has multiple channels - unlikely to be pure IR")
        return False
    return True


def align_frames(visible_frame: np.ndarray,
                 ir_frame: np.ndarray,
                 target_size: Tuple[int, int] = (224, 224)) -> Tuple[np.ndarray, np.ndarray]:
    """Resize a visible/IR pair to a common size (no geometric registration)."""
    visible_aligned = cv2.resize(visible_frame, (target_size[1], target_size[0]))
    ir_aligned = cv2.resize(ir_frame, (target_size[1], target_size[0]))
    return visible_aligned, ir_aligned
