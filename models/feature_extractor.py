"""
Feature extraction module using pretrained ResNet50 encoders.

Extracts 2048-dimensional features from RGB and IR frames.
Supports ROI-based extraction using segmentation masks.

Conventions:
- RGB frames are OpenCV BGR uint8 arrays; they are converted to RGB before
  ImageNet normalization.
- IR frames are single-channel (H, W) or (H, W, 1) arrays. They are replicated
  to three channels and normalized with ImageNet statistics, matching the
  RGB-pretrained ResNet50.
"""

import logging
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _as_uint8(frame: np.ndarray) -> np.ndarray:
    """Convert a frame to uint8 without assuming a particular input range."""
    if frame.dtype == np.uint8:
        return frame
    if frame.dtype == np.uint16:
        # 16-bit thermal imagery: scale the observed range to 8-bit.
        frame = frame.astype(np.float32)
        lo, hi = float(frame.min()), float(frame.max())
        if hi > lo:
            frame = (frame - lo) * (255.0 / (hi - lo))
        return np.clip(frame, 0, 255).astype(np.uint8)
    frame = frame.astype(np.float32)
    if frame.max() <= 1.0:
        frame = frame * 255.0
    return np.clip(frame, 0, 255).astype(np.uint8)


def _ensure_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR/greyscale frame to an RGB uint8 frame."""
    frame = _as_uint8(frame)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        frame = cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Unsupported RGB frame shape: {frame.shape}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _ensure_gray(frame: np.ndarray) -> np.ndarray:
    """Convert a frame to a single-channel uint8 IR frame."""
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
    """Resize a binary mask to (height, width) using nearest-neighbour."""
    if mask is None:
        return None
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.shape[:2] != size:
        mask = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.float32)


class ResNet50FeatureExtractor(nn.Module):
    """ResNet50 feature extractor with ImageNet-pretrained weights."""

    def __init__(self, device: str = "cuda"):
        super().__init__()

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load pretrained ResNet50 and remove the classification layer.
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.backbone.eval()
        self.backbone.to(self.device)

        self.feature_dim = 2048

        logger.info(f"ResNet50 feature extractor loaded on {self.device}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from a normalized (B, 3, H, W) tensor."""
        features = self.backbone(x)
        return features.flatten(1)

    def set_trainable(self, trainable: bool = True):
        """Enable or disable gradients for the backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = trainable

    def set_layer4_trainable(self, trainable: bool = True):
        """Unfreeze only the last residual stage (stage-2 fine-tuning)."""
        # The sequential backbone is conv1, bn1, relu, maxpool, layer1..layer4,
        # avgpool.  ``layer4`` is therefore index 7.
        for param in self.backbone[7].parameters():
            param.requires_grad = trainable

    def trainable_parameters(self):
        """Return parameters that currently require gradients."""
        return [p for p in self.backbone.parameters() if p.requires_grad]


class _BaseFeatureExtractor:
    """Shared preprocessing and batched extraction for RGB/IR frames."""

    modality = "base"

    def __init__(self,
                 device: str = "cuda",
                 input_size: Tuple[int, int] = (224, 224),
                 use_amp: bool = True):
        self.device = device
        self.input_size = tuple(input_size)
        self.use_amp = bool(use_amp and torch.cuda.is_available() and str(device).startswith("cuda"))
        self.extractor = ResNet50FeatureExtractor(device)

    def train(self, mode: bool = True):
        """Put the underlying encoder in training mode (for stage-2 tuning)."""
        self.extractor.train(mode)
        return self

    def eval(self):
        """Put the underlying encoder in evaluation mode."""
        self.extractor.eval()
        return self

    def _preprocess_one(self, frame: np.ndarray,
                        mask: Optional[np.ndarray] = None) -> np.ndarray:
        raise NotImplementedError

    def preprocess(self, frame: np.ndarray,
                   mask: Optional[np.ndarray] = None) -> torch.Tensor:
        """Preprocess one frame and return a (1, 3, H, W) tensor."""
        arr = self._preprocess_one(frame, mask)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.extractor.device)

    def extract_batch(self,
                      frames: Sequence[np.ndarray],
                      masks: Optional[Sequence[Optional[np.ndarray]]] = None) -> np.ndarray:
        """Extract (B, 2048) features for a batch of frames."""
        if len(frames) == 0:
            return np.zeros((0, self.extractor.feature_dim), dtype=np.float32)
        if masks is None:
            masks = [None] * len(frames)
        if len(masks) != len(frames):
            raise ValueError("masks must have the same length as frames")

        batch = self.preprocess_batch(frames, masks)
        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=self.use_amp,
            ):
                features = self.extractor(batch)
        return features.detach().float().cpu().numpy().astype(np.float32)

    def preprocess_batch(self,
                         frames: Sequence[np.ndarray],
                         masks: Optional[Sequence[Optional[np.ndarray]]] = None) -> torch.Tensor:
        """Preprocess frames and return a (B, 3, H, W) tensor on the device."""
        if len(frames) == 0:
            return torch.empty((0, 3, *self.input_size), device=self.extractor.device)
        if masks is None:
            masks = [None] * len(frames)
        if len(masks) != len(frames):
            raise ValueError("masks must have the same length as frames")
        tensors = [self.preprocess(frame, mask) for frame, mask in zip(frames, masks)]
        return torch.cat(tensors, dim=0)

    def extract_tensor(self, batch: torch.Tensor) -> torch.Tensor:
        """Run the encoder on a preprocessed batch with gradients enabled."""
        return self.extractor(batch)

    def extract(self, frame: np.ndarray,
                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Extract a (2048,) feature vector for one frame."""
        return self.extract_batch([frame], [mask])[0]


class RGBFeatureExtractor(_BaseFeatureExtractor):
    """RGB frame feature extractor with ROI support."""

    modality = "rgb"

    def _preprocess_one(self, frame: np.ndarray,
                        mask: Optional[np.ndarray] = None) -> np.ndarray:
        frame = _ensure_rgb(frame)
        h, w = self.input_size
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

        mask = _resize_mask(mask, (h, w))
        if mask is not None:
            frame = frame.astype(np.float32) * mask[:, :, None]

        frame = frame.astype(np.float32) / 255.0
        frame = (frame - IMAGENET_MEAN) / IMAGENET_STD
        return frame.astype(np.float32)


class IRFeatureExtractor(_BaseFeatureExtractor):
    """
    IR/thermal feature extractor with ROI support.

    The encoder is an RGB/ImageNet-pretrained ResNet50. IR frames are
    replicated to three channels. This is a known limitation: the encoder was
    not pretrained on thermal data.
    """

    modality = "ir"

    def _preprocess_one(self, frame: np.ndarray,
                        mask: Optional[np.ndarray] = None) -> np.ndarray:
        frame = _ensure_gray(frame)
        h, w = self.input_size
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

        mask = _resize_mask(mask, (h, w))
        if mask is not None:
            frame = frame.astype(np.float32) * mask

        frame = frame.astype(np.float32) / 255.0
        frame = np.repeat(frame[:, :, None], 3, axis=2)
        frame = (frame - IMAGENET_MEAN) / IMAGENET_STD
        return frame.astype(np.float32)
