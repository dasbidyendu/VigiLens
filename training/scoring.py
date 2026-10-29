"""
Shared scoring utilities for cached features.

Training, validation, calibration and offline evaluation all need the same
definition of an anomaly score.  Keeping it in one module prevents the three
paths from drifting apart.

Score definition
----------------
For frame ``t`` the temporal model predicts the fused feature of frame
``t + 1``.  The anomaly score assigned to frame ``t + 1`` is the mean squared
error between that prediction and the actual fused feature at ``t + 1``.
Frame 0 therefore has no score and is assigned ``0.0``.  This is exactly the
one-frame latency used by the streaming inference pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def load_cache(cache_path: Union[str, Path], map_location: str = "cpu") -> Dict[str, Any]:
    """Load a feature-cache file with a safe fallback for old checkpoints."""
    path = Path(cache_path)
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


def metadata_path(cache_path: Union[str, Path]) -> Path:
    """Return the JSON sidecar path for a ``.pt`` cache file."""
    return Path(cache_path).with_suffix(".json")


def read_cache_metadata(cache_path: Union[str, Path]) -> Dict[str, Any]:
    """Read cache metadata from its sidecar, returning an empty dict if absent."""
    path = metadata_path(cache_path)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read cache metadata %s: %s", path, exc)
        return {}


def _as_mask(value, batch_size: int, device: torch.device) -> torch.Tensor:
    if value is None:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)
    if isinstance(value, bool):
        return torch.full((batch_size,), value, dtype=torch.bool, device=device)
    mask = torch.as_tensor(value, device=device).bool().reshape(-1)
    if mask.numel() == 1:
        mask = mask.expand(batch_size)
    if mask.numel() != batch_size:
        raise ValueError(f"Availability mask has {mask.numel()} elements, expected {batch_size}")
    return mask


def fuse_features(model,
                  rgb_features: Optional[torch.Tensor],
                  ir_features: Optional[torch.Tensor],
                  rgb_available=True,
                  ir_available=True) -> torch.Tensor:
    """Fuse a batch of frame-level features through ``model.fusion``."""
    if rgb_features is None and ir_features is None:
        raise ValueError("At least one modality is required")
    batch_size = rgb_features.shape[0] if rgb_features is not None else ir_features.shape[0]
    device = rgb_features.device if rgb_features is not None else ir_features.device
    rgb_mask = _as_mask(rgb_available, batch_size, device)
    ir_mask = _as_mask(ir_available, batch_size, device)
    return model.fusion(rgb_features, ir_features, rgb_mask, ir_mask)


def fuse_sequence(model,
                  rgb_sequence: torch.Tensor,
                  ir_sequence: Optional[torch.Tensor] = None,
                  rgb_available=True,
                  ir_available=True) -> torch.Tensor:
    """
    Fuse a batch of feature sequences.

    Args:
        rgb_sequence: (B, T, 2048)
        ir_sequence: (B, T, 2048) or None
        rgb_available: bool or (B,) tensor
        ir_available: bool or (B,) tensor

    Returns:
        (B, T, fusion_output_dim)
    """
    if rgb_sequence.dim() != 3:
        raise ValueError(f"rgb_sequence must be (B, T, D); got {tuple(rgb_sequence.shape)}")
    batch_size, sequence_length, _ = rgb_sequence.shape
    rgb_flat = rgb_sequence.reshape(batch_size * sequence_length, -1)

    ir_flat = None
    if ir_sequence is not None:
        if ir_sequence.shape[:2] != rgb_sequence.shape[:2]:
            raise ValueError(
                "IR sequence shape must match RGB sequence shape; got "
                f"{tuple(ir_sequence.shape)} and {tuple(rgb_sequence.shape)}"
            )
        ir_flat = ir_sequence.reshape(batch_size * sequence_length, -1)

    rgb_mask = _as_mask(rgb_available, batch_size, rgb_sequence.device)
    ir_mask = _as_mask(ir_available, batch_size, rgb_sequence.device)

    # A modality that is unavailable for every sample can be omitted entirely.
    if not bool(rgb_mask.any()):
        rgb_flat = None
    if not bool(ir_mask.any()):
        ir_flat = None

    fused = fuse_features(
        model,
        rgb_flat,
        ir_flat,
        rgb_mask.repeat_interleave(sequence_length),
        ir_mask.repeat_interleave(sequence_length),
    )
    return fused.view(batch_size, sequence_length, -1)


def predict_fused_sequence(model,
                           fused: torch.Tensor,
                           hidden_state=None,
                           chunk_size: int = 256
                           ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Run the temporal model over ``(B, T, D)``, optionally in chunks."""
    if fused.dim() != 3:
        raise ValueError(f"fused must be (B, T, D); got {tuple(fused.shape)}")
    if fused.shape[1] <= chunk_size:
        return model.temporal.forward_sequence(fused, hidden_state)

    predictions = []
    state = hidden_state
    for start in range(0, fused.shape[1], chunk_size):
        chunk = fused[:, start:start + chunk_size]
        prediction, state = model.temporal.forward_sequence(chunk, state)
        predictions.append(prediction)
    return torch.cat(predictions, dim=1), state


def causal_moving_average(scores: np.ndarray, window: int) -> np.ndarray:
    """Causal moving average that only uses the current and past scores."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    window = max(1, int(window))
    if window == 1 or scores.size == 0:
        return scores.astype(np.float32)
    cumulative = np.concatenate(([0.0], np.cumsum(scores)))
    result = np.empty_like(scores)
    for index in range(scores.size):
        start = max(0, index + 1 - window)
        result[index] = (cumulative[index + 1] - cumulative[start]) / (index + 1 - start)
    return result.astype(np.float32)


def score_fused_sequence(model,
                         fused: torch.Tensor,
                         smoothing_window: int = 1,
                         chunk_size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-frame prediction errors and their causal smoothed version.

    Args:
        model: Object with ``fusion`` and ``temporal`` modules
        fused: (N, D) fused features for one video
        smoothing_window: Causal moving-average window (1 disables smoothing)
        chunk_size: Temporal chunk size

    Returns:
        (raw_scores, smoothed_scores), both float32 arrays of length N
    """
    if fused.dim() != 2:
        raise ValueError(f"fused must be (N, D); got {tuple(fused.shape)}")

    total = fused.shape[0]
    raw = np.zeros(total, dtype=np.float32)
    if total == 0:
        return raw, raw

    state = None
    previous_prediction = None
    offset = 0
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = fused[start:end].unsqueeze(0)  # (1, C, D)
        predictions, state = model.temporal.forward_sequence(chunk, state)
        predictions = predictions.squeeze(0)  # (C, D)
        length = end - start

        if length > 1:
            errors = F.mse_loss(predictions[:-1], fused[start + 1:end], reduction="none").mean(dim=1)
            raw[start + 1:end] = errors.detach().cpu().numpy()
        if previous_prediction is not None:
            raw[start] = F.mse_loss(
                previous_prediction, fused[start], reduction="none"
            ).mean().item()
        previous_prediction = predictions[-1].detach()
        offset = end

    smoothed = causal_moving_average(raw, smoothing_window)
    return raw, smoothed


@torch.inference_mode()
def score_cache_file(model,
                     cache_path: Union[str, Path],
                     smoothing_window: int = 1,
                     chunk_size: int = 256,
                     device: Optional[Union[str, torch.device]] = None) -> Dict[str, Any]:
    """
    Score every frame in a cached feature file.

    Returns a dictionary with raw/smoothed scores, labels (if the cache or its
    sidecar contains them), and metadata used by the evaluation report.
    """
    if hasattr(model, "eval"):
        model.eval()
    if hasattr(model, "fusion"):
        model.fusion.eval()
    if hasattr(model, "temporal"):
        model.temporal.eval()

    data = load_cache(cache_path)
    metadata = read_cache_metadata(cache_path)

    rgb = data["rgb_features"].float()
    ir = data.get("ir_features")
    has_ir = bool(data.get("has_ir", ir is not None)) and ir is not None
    if ir is not None:
        ir = ir.float()

    target_device = torch.device(device) if device is not None else _model_device(model)
    rgb = rgb.to(target_device)
    if ir is not None:
        ir = ir.to(target_device)

    # Fuse the whole video in chunks, then score it.  For typical clips this
    # is a few MB; the chunking keeps very long evaluation videos bounded.
    fused_chunks = []
    for start in range(0, rgb.shape[0], chunk_size):
        end = start + chunk_size
        rgb_chunk = rgb[start:end]
        ir_chunk = ir[start:end] if ir is not None else None
        fused_chunks.append(
            fuse_features(model, rgb_chunk, ir_chunk, True, has_ir)
        )
    fused = torch.cat(fused_chunks, dim=0) if fused_chunks else rgb.new_zeros((0, 512))

    raw, smoothed = score_fused_sequence(model, fused, smoothing_window, chunk_size)

    labels = data.get("labels")
    if labels is None:
        labels = metadata.get("labels")
    if labels is not None:
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if labels.size != raw.size:
            logger.warning(
                "Label length %d does not match score length %d for %s; ignoring labels",
                labels.size, raw.size, cache_path,
            )
            labels = None

    return {
        "sequence_id": data.get("sequence_id", metadata.get("sequence_id", Path(cache_path).stem)),
        "raw_scores": raw,
        "smoothed_scores": smoothed,
        "labels": labels,
        "num_frames": int(raw.size),
        "has_ir": has_ir,
        "fps": data.get("fps", metadata.get("fps")),
        "cache_path": str(cache_path),
    }


def _model_device(model) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cpu")


def prediction_errors(model,
                      cache_paths,
                      smoothing_window: int = 1,
                      chunk_size: int = 256,
                      device: Optional[Union[str, torch.device]] = None) -> np.ndarray:
    """Collect non-zero prediction errors from one or more cache files."""
    errors = []
    for path in cache_paths:
        result = score_cache_file(
            model, path, smoothing_window=1, chunk_size=chunk_size, device=device
        )
        # Frame 0 has no prediction and its placeholder zero is not a real
        # normal-data error; exclude it from calibration statistics.
        errors.append(result["raw_scores"][1:])
    if not errors:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(errors).astype(np.float32)
