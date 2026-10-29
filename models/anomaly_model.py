"""
Top-level VigiLens anomaly detection model.

Integrates:
- Feature extraction (RGB/IR ResNet50)
- Modality-aware fusion
- Causal streaming LSTM
- Next-feature prediction

Trains on normal pedestrian behavior only.
Detects anomalies via prediction error.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.feature_extractor import RGBFeatureExtractor, IRFeatureExtractor
from models.fusion import ModalityAwareFusion
from models.temporal_lstm import TemporalModel

logger = logging.getLogger(__name__)


def _to_plain(value):
    """Recursively convert dataclasses/tuples to checkpoint-safe primitives."""
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return {str(k): _to_plain(v) for k, v in vars(value).items()}
    return str(value)


class VigiLensModel(nn.Module):
    """
    Complete VigiLens anomaly detection model.

    Pipeline:
        RGB frame -> YOLO segmentation -> ROI -> ResNet50 -> 2048-D
        IR frame -> RGB-derived ROI -> ResNet50 -> 2048-D
        Fusion -> 512-D
        LSTM -> hidden state -> prediction head -> predicted next 512-D

    Training: normal-only next-feature prediction.
    Inference: prediction error = anomaly score.
    """

    def __init__(self,
                 device: str = "cuda",
                 input_size: Tuple[int, int] = (224, 224),
                 fusion_hidden_dim: int = 256,
                 fusion_output_dim: int = 512,
                 lstm_hidden_size: int = 256,
                 lstm_layers: int = 2,
                 lstm_dropout: float = 0.2,
                 load_extractors: bool = True):
        super().__init__()

        self.device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        self.input_size = input_size
        self.fusion_output_dim = fusion_output_dim
        self.load_extractors = bool(load_extractors)

        # Feature extractors (frozen by default; training only updates fusion
        # and the temporal model unless explicitly unfrozen).  Cached-feature
        # training/evaluation does not need them, so it can skip the ~200 MB
        # ImageNet download and the ResNet50 memory cost.
        if self.load_extractors:
            self.rgb_extractor = RGBFeatureExtractor(self.device, input_size)
            self.ir_extractor = IRFeatureExtractor(self.device, input_size)
        else:
            self.rgb_extractor = None
            self.ir_extractor = None

        # Modality-aware fusion
        self.fusion = ModalityAwareFusion(
            input_dim=2048,
            hidden_dim=fusion_hidden_dim,
            output_dim=fusion_output_dim
        )

        # Temporal model (LSTM + prediction head)
        self.temporal = TemporalModel(
            input_size=fusion_output_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout=lstm_dropout
        )

        # Streaming state
        self.hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.previous_prediction: Optional[torch.Tensor] = None

        logger.info("VigiLens model initialized")

    def extract_features(self,
                         rgb_frame,
                         ir_frame=None,
                         rgb_mask=None,
                         ir_mask=None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Extract features from RGB and IR frames.

        Args:
            rgb_frame: BGR frame (H, W, 3) or None
            ir_frame: IR frame (H, W) or (H, W, 1) or None
            rgb_mask: Binary mask for RGB ROI (H, W) or None
            ir_mask: Binary mask for IR ROI (H, W) or None

        Returns:
            Tuple of (rgb_features, ir_features), each (1, 2048) or None.
        """
        if not self.load_extractors or self.rgb_extractor is None or self.ir_extractor is None:
            raise RuntimeError(
                "This VigiLensModel was created with load_extractors=False and "
                "cannot process raw frames. Use the cached-feature scoring path "
                "or create the model with load_extractors=True."
            )

        rgb_features = None
        if rgb_frame is not None:
            try:
                rgb_np = self.rgb_extractor.extract(rgb_frame, rgb_mask)
                rgb_features = torch.from_numpy(rgb_np).float().unsqueeze(0).to(self.device)
            except Exception as exc:
                logger.error(f"RGB feature extraction failed: {exc}")
                rgb_features = None

        ir_features = None
        if ir_frame is not None:
            try:
                ir_np = self.ir_extractor.extract(ir_frame, ir_mask)
                ir_features = torch.from_numpy(ir_np).float().unsqueeze(0).to(self.device)
            except Exception as exc:
                logger.error(f"IR feature extraction failed: {exc}")
                ir_features = None

        return rgb_features, ir_features

    def fuse_features(self,
                      rgb_features: Optional[torch.Tensor],
                      ir_features: Optional[torch.Tensor],
                      rgb_available: bool = True,
                      ir_available: bool = True) -> torch.Tensor:
        """Fuse features. Gradients flow through the fusion module."""
        return self.fusion(rgb_features, ir_features, rgb_available, ir_available)

    def forward(self,
                fused_features: torch.Tensor,
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the temporal model."""
        prediction, (h, c) = self.temporal(fused_features, hidden_state)
        return prediction, (h, c)

    def reset_stream(self):
        """Reset streaming state (hidden state and previous prediction)."""
        self.hidden_state = None
        self.previous_prediction = None
        logger.debug("Stream state reset")

    def init_hidden(self, batch_size: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state for the temporal model."""
        return self.temporal.init_hidden(batch_size, device)

    def process_frame(self,
                      rgb_frame,
                      ir_frame=None,
                      rgb_mask=None,
                      ir_mask=None,
                      rgb_available: bool = True,
                      ir_available: bool = True) -> Dict:
        """
        Process one frame for streaming inference.

        Returns:
            Dictionary with:
            - prediction: predicted next feature
            - anomaly_score: prediction error from the previous prediction
            - hidden_state: current LSTM hidden state
        """
        with torch.inference_mode():
            rgb_features, ir_features = self.extract_features(
                rgb_frame, ir_frame, rgb_mask, ir_mask
            )

            # Availability must reflect what was actually extracted.
            rgb_available = bool(rgb_available and rgb_features is not None)
            ir_available = bool(ir_available and ir_features is not None)
            if not rgb_available and not ir_available:
                raise ValueError(
                    "No usable modality for this frame: both RGB and IR feature "
                    "extraction failed or both inputs were missing"
                )

            fused = self.fuse_features(
                rgb_features, ir_features, rgb_available, ir_available
            )

            anomaly_score = None
            if self.previous_prediction is not None:
                error = F.mse_loss(self.previous_prediction, fused, reduction="none")
                anomaly_score = error.mean().item()

            prediction, (h, c) = self.forward(fused, self.hidden_state)

            # Detach state to prevent graph growth across frames.
            self.hidden_state = (h.detach(), c.detach())
            self.previous_prediction = prediction.detach()

        return {
            "prediction": prediction,
            "anomaly_score": anomaly_score,
            "hidden_state": (h, c),
            "rgb_available": rgb_available,
            "ir_available": ir_available,
        }

    def training_step(self,
                      fused_features: torch.Tensor,
                      next_fused_features: torch.Tensor,
                      hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                      ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Training step: predict the next feature and compute Smooth L1 loss.

        Args:
            fused_features: Current features (B, 512) or (B, 1, 512)
            next_fused_features: Next features (B, 512) or (B, 1, 512)
            hidden_state: Previous hidden state
        """
        if fused_features.dim() == 3 and fused_features.shape[1] == 1:
            fused_features = fused_features.squeeze(1)
        if next_fused_features.dim() == 3 and next_fused_features.shape[1] == 1:
            next_fused_features = next_fused_features.squeeze(1)

        if fused_features.dim() != 2 or next_fused_features.dim() != 2:
            raise ValueError(
                "training_step expects (B, D) or (B, 1, D) tensors; got "
                f"{tuple(fused_features.shape)} and {tuple(next_fused_features.shape)}"
            )

        prediction, (h, c) = self.forward(fused_features, hidden_state)
        loss = F.smooth_l1_loss(prediction, next_fused_features)
        return loss, (h, c)

    def save_checkpoint(self,
                        path: str,
                        epoch: int,
                        loss: float,
                        config: Dict,
                        optimizer_state: Optional[Dict] = None,
                        extra: Optional[Dict] = None,
                        include_encoders: bool = False):
        """
        Save a checkpoint with plain-dict metadata (safe to torch.load).

        ``include_encoders`` is used only for stage-2 online fine-tuning; the
        normal cached-feature checkpoint contains just fusion + temporal
        weights and is therefore small.
        """
        checkpoint = {
            "epoch": int(epoch),
            "model_state_dict": self.state_dict(),
            "loss": float(loss),
            "config": _to_plain(config),
        }
        if optimizer_state is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state
        if extra:
            checkpoint["extra"] = _to_plain(extra)
        if include_encoders:
            if self.rgb_extractor is None or self.ir_extractor is None:
                raise RuntimeError(
                    "include_encoders=True requires a model created with "
                    "load_extractors=True"
                )
            checkpoint["encoder_state_dict"] = {
                "rgb": self.rgb_extractor.extractor.state_dict(),
                "ir": self.ir_extractor.extractor.state_dict(),
            }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str, strict: bool = True) -> Dict:
        """Load a checkpoint. Falls back to weights_only=False for old files."""
        try:
            checkpoint = torch.load(
                path, map_location=self.device, weights_only=True
            )
        except Exception:
            # Older checkpoints may contain dataclass config objects.
            checkpoint = torch.load(
                path, map_location=self.device, weights_only=False
            )

        state = checkpoint.get("model_state_dict", checkpoint)
        try:
            self.load_state_dict(state, strict=strict)
        except RuntimeError as exc:
            if not strict:
                raise
            logger.warning(
                "Strict checkpoint load failed (%s); retrying with strict=False", exc
            )
            self.load_state_dict(state, strict=False)

        encoders = checkpoint.get("encoder_state_dict")
        if encoders:
            if self.rgb_extractor is None or self.ir_extractor is None:
                logger.info(
                    "Checkpoint contains fine-tuned encoder weights, but this "
                    "model was created with load_extractors=False; skipping them."
                )
            else:
                try:
                    self.rgb_extractor.extractor.load_state_dict(encoders["rgb"])
                    self.ir_extractor.extractor.load_state_dict(encoders["ir"])
                    logger.info("Loaded fine-tuned encoder weights from checkpoint")
                except (KeyError, RuntimeError) as exc:
                    logger.warning("Could not load encoder weights: %s", exc)

        logger.info(f"Checkpoint loaded from {path}")
        return {
            "epoch": checkpoint.get("epoch"),
            "loss": checkpoint.get("loss"),
            "config": checkpoint.get("config"),
            "extra": checkpoint.get("extra"),
        }
