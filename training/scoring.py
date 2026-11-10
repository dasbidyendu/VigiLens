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

Context window
--------------
Training sees windows of ``sequence_length`` frames that each start from a
zero LSTM state.  Scoring therefore predicts frame ``t`` from at most the
previous ``model.context_length`` (= ``sequence_length - 1``) frames, again
from a zero state, instead of carrying the LSTM state through a whole video
(a horizon the model never saw during training).  The streaming pipeline in
``VigiLensModel.process_frame`` uses the same rule, so offline scores equal
the scores the app computes.

Smoothing
---------
Smoothing is applied to frames ``1..N-1`` only (frame 0 has no score), which
is what the streaming pipeline does.
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
    """Load a feature-cache file (tensors and plain Python values only)."""
    return torch.load(Path(cache_path), map_location=map_location, weights_only=True)


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
    index = np.arange(scores.size)
    start = np.maximum(0, index + 1 - window)
    result = (cumulative[index + 1] - cumulative[start]) / (index + 1 - start)
    return result.astype(np.float32)


def causal_ema(scores: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average seeded with the first score (as in the app)."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    result = np.empty_like(scores)
    state = None
    for index, value in enumerate(scores):
        state = value if state is None else alpha * value + (1.0 - alpha) * state
        result[index] = state
    return result.astype(np.float32)


def smooth_scores(raw: np.ndarray,
                  method: str = "moving_average",
                  window: int = 1,
                  alpha: float = 0.3) -> np.ndarray:
    """
    Smooth a per-video raw score array exactly like ``InferencePipeline``.

    ``raw[0]`` is the placeholder for the unscored first frame: it is left at
    0 and excluded from the smoothing history.
    """
    raw = np.asarray(raw, dtype=np.float32).reshape(-1)
    smoothed = np.zeros_like(raw)
    if raw.size <= 1:
        return smoothed
    if method == "exponential":
        smoothed[1:] = causal_ema(raw[1:], alpha)
    elif method == "moving_average":
        smoothed[1:] = causal_moving_average(raw[1:], window)
    else:
        smoothed[1:] = raw[1:]
    return smoothed


def _context_length(model, context_length: Optional[int]) -> Optional[int]:
    if context_length is not None:
        return int(context_length) or None
    value = getattr(model, "context_length", None)
    return int(value) if value else None


def _score_windowed(model, fused: torch.Tensor, window: int, chunk_size: int) -> np.ndarray:
    """Predict frame t from frames [t-window, t) with a zero initial state."""
    total = fused.shape[0]
    raw = np.zeros(total, dtype=np.float32)
    if total < 2:
        return raw

    # Head: frames 1..min(window, total-1).  A single zero-state pass over the
    # first ``window`` frames gives every shorter context in one go.
    head_len = min(window, total)
    head_predictions, _ = model.temporal.forward_sequence(fused[:head_len].unsqueeze(0))
    head_predictions = head_predictions.squeeze(0)
    head_targets = min(head_len, total - 1)
    errors = F.mse_loss(
        head_predictions[:head_targets].float(), fused[1:head_targets + 1].float(), reduction="none"
    ).mean(dim=1)
    raw[1:head_targets + 1] = errors.detach().cpu().numpy()

    # Tail: frames window+1..total-1, each from a full window of context.
    offsets = torch.arange(window, device=fused.device)
    batch = max(1, int(chunk_size))
    for first in range(window + 1, total, batch):
        targets_idx = torch.arange(first, min(first + batch, total), device=fused.device)
        windows = fused[(targets_idx - window).unsqueeze(1) + offsets]  # (M, W, D)
        predictions, _ = model.temporal.forward_sequence(windows)
        errors = F.mse_loss(
            predictions[:, -1].float(), fused[targets_idx].float(), reduction="none"
        ).mean(dim=1)
        raw[targets_idx.cpu().numpy()] = errors.detach().cpu().numpy()
    return raw


def _score_stateful(model, fused: torch.Tensor, chunk_size: int) -> np.ndarray:
    """Legacy scoring: one LSTM state carried through the whole video."""
    total = fused.shape[0]
    raw = np.zeros(total, dtype=np.float32)
    state = None
    previous_prediction = None
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = fused[start:end].unsqueeze(0)  # (1, C, D)
        predictions, state = model.temporal.forward_sequence(chunk, state)
        predictions = predictions.squeeze(0)  # (C, D)
        if end - start > 1:
            errors = F.mse_loss(predictions[:-1], fused[start + 1:end], reduction="none").mean(dim=1)
            raw[start + 1:end] = errors.detach().cpu().numpy()
        if previous_prediction is not None:
            raw[start] = F.mse_loss(
                previous_prediction, fused[start], reduction="none"
            ).mean().item()
        previous_prediction = predictions[-1].detach()
    return raw


def score_fused_sequence(model,
                         fused: torch.Tensor,
                         smoothing_window: int = 1,
                         chunk_size: int = 256,
                         context_length: Optional[int] = None,
                         smoothing_method: str = "moving_average",
                         ema_alpha: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-frame prediction errors and their causal smoothed version.

    Args:
        model: Object with ``fusion`` and ``temporal`` modules
        fused: (N, D) fused features for one video
        smoothing_window: Causal moving-average window (1 disables smoothing)
        chunk_size: Batch/chunk size for the temporal model
        context_length: Frames of context per prediction; defaults to
            ``model.context_length``.  ``0`` selects legacy stateful scoring.
        smoothing_method: ``"moving_average"`` or ``"exponential"``
        ema_alpha: EMA coefficient for ``"exponential"``

    Returns:
        (raw_scores, smoothed_scores), both float32 arrays of length N
    """
    if fused.dim() != 2:
        raise ValueError(f"fused must be (N, D); got {tuple(fused.shape)}")

    total = fused.shape[0]
    if total == 0:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty

    window = _context_length(model, context_length)
    if window:
        raw = _score_windowed(model, fused, window, chunk_size)
    else:
        raw = _score_stateful(model, fused, chunk_size)

    smoothed = smooth_scores(raw, smoothing_method, smoothing_window, ema_alpha)
    return raw, smoothed


@torch.inference_mode()
def score_cache_file(model,
                     cache_path: Union[str, Path],
                     smoothing_window: int = 1,
                     chunk_size: int = 256,
                     device: Optional[Union[str, torch.device]] = None,
                     smoothing_method: str = "moving_average",
                     ema_alpha: float = 0.3) -> Dict[str, Any]:
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

    raw, smoothed = score_fused_sequence(
        model, fused, smoothing_window, chunk_size,
        smoothing_method=smoothing_method, ema_alpha=ema_alpha,
    )

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
        "frame_indices": data.get("frame_indices", metadata.get("frame_indices")),
        "frame_stride": int(data.get("frame_stride", metadata.get("frame_stride", 1)) or 1),
        "start_frame": int(data.get("start_frame", metadata.get("start_frame", 0)) or 0),
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


def raw_score_sequences(model,
                        cache_paths,
                        chunk_size: int = 256,
                        device: Optional[Union[str, torch.device]] = None):
    """Per-video raw score arrays (frame 0 included as the 0.0 placeholder)."""
    sequences = []
    for path in cache_paths:
        result = score_cache_file(model, path, smoothing_window=1, chunk_size=chunk_size, device=device)
        sequences.append(result["raw_scores"])
    return sequences


def prediction_errors(model,
                      cache_paths,
                      smoothing_window: int = 1,
                      chunk_size: int = 256,
                      device: Optional[Union[str, torch.device]] = None,
                      smoothing_method: str = "moving_average",
                      ema_alpha: float = 0.3) -> np.ndarray:
    """
    Collect per-frame errors from cache files, smoothed like the app would.

    Frame 0 of each video has no prediction; its placeholder zero is not a
    real normal-data error and is excluded.
    """
    errors = []
    for raw in raw_score_sequences(model, cache_paths, chunk_size=chunk_size, device=device):
        smoothed = smooth_scores(raw, smoothing_method, smoothing_window, ema_alpha)
        errors.append(smoothed[1:])
    if not errors:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(errors).astype(np.float32)
