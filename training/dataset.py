"""
Datasets for VigiLens normal-only training.

Two dataset implementations are provided:

1. ``NormalVideoDataset`` reads video files directly. It is convenient for
   small smoke tests but re-decodes frames and recomputes CNN features every
   epoch, so it is slow for real training.
2. ``CachedFeatureDataset`` reads precomputed per-frame RGB/IR features. This
   is the recommended path for KAIST/Colab training.
"""

import json
import logging
import random
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert a decoded OpenCV frame to 3-channel uint8."""
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 1:
        return cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def _to_gray(frame: np.ndarray) -> np.ndarray:
    """Convert a decoded OpenCV frame to single-channel uint8."""
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 1:
        return frame[:, :, 0]
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported frame shape: {frame.shape}")


class NormalVideoDataset(Dataset):
    """Normal-only video dataset for next-feature prediction training."""

    def __init__(self,
                 video_paths: List[Path],
                 ir_video_paths: Optional[List[Path]] = None,
                 sequence_length: int = 16,
                 sequence_stride: int = 1,
                 rgb_dropout_prob: float = 0.15,
                 ir_dropout_prob: float = 0.15,
                 input_size: Tuple[int, int] = (224, 224)):
        self.video_paths = list(video_paths)
        if ir_video_paths:
            if len(ir_video_paths) != len(video_paths):
                raise ValueError("video_paths and ir_video_paths must have equal length")
            self.ir_video_paths = list(ir_video_paths)
        else:
            self.ir_video_paths = [None] * len(video_paths)

        self.sequence_length = int(sequence_length)
        self.sequence_stride = max(1, int(sequence_stride))
        self.rgb_dropout_prob = float(rgb_dropout_prob)
        self.ir_dropout_prob = float(ir_dropout_prob)
        self.input_size = tuple(input_size)

        self.video_info = []
        for rgb_path, ir_path in zip(self.video_paths, self.ir_video_paths):
            cap = cv2.VideoCapture(str(rgb_path))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            cap.release()

            ir_frames = 0
            if ir_path is not None and Path(ir_path).exists():
                cap_ir = cv2.VideoCapture(str(ir_path))
                ir_frames = int(cap_ir.get(cv2.CAP_PROP_FRAME_COUNT))
                cap_ir.release()

            self.video_info.append({
                "rgb_path": Path(rgb_path),
                "ir_path": Path(ir_path) if ir_path is not None else None,
                "frame_count": frame_count,
                "ir_frame_count": ir_frames,
                "fps": fps,
                "sequence_id": Path(rgb_path).stem,
            })

        logger.info(
            f"NormalVideoDataset: {len(video_paths)} videos, "
            f"seq_len={sequence_length}, stride={self.sequence_stride}"
        )

    def __len__(self) -> int:
        total = 0
        for info in self.video_info:
            if info["frame_count"] >= self.sequence_length:
                total += (info["frame_count"] - self.sequence_length) // self.sequence_stride + 1
        return total

    def _locate(self, idx: int) -> Tuple[int, int]:
        """Map a flat sequence index to (video_index, start_frame)."""
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")

        cumulative = 0
        for i, info in enumerate(self.video_info):
            if info["frame_count"] < self.sequence_length:
                continue
            count = (info["frame_count"] - self.sequence_length) // self.sequence_stride + 1
            if idx < cumulative + count:
                return i, (idx - cumulative) * self.sequence_stride
            cumulative += count
        raise IndexError(f"Index {idx} out of range")

    def __getitem__(self, idx: int) -> Dict:
        video_idx, start = self._locate(idx)
        info = self.video_info[video_idx]

        rgb_sequence, ir_sequence, has_ir = self._load_sequence(
            info["rgb_path"], info["ir_path"], start, self.sequence_length
        )

        rgb_available = random.random() > self.rgb_dropout_prob
        ir_available = has_ir and (random.random() > self.ir_dropout_prob)
        if not rgb_available and not ir_available:
            rgb_available = True

        return {
            "rgb_sequence": rgb_sequence,
            "ir_sequence": ir_sequence,
            "rgb_available": rgb_available,
            "ir_available": ir_available,
            "sequence_id": info["sequence_id"],
        }

    def _load_sequence(self,
                       rgb_path: Path,
                       ir_path: Optional[Path],
                       start_frame: int,
                       length: int) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Load exactly ``length`` frames, padding by repeating the last frame."""
        cap_rgb = cv2.VideoCapture(str(rgb_path))
        if not cap_rgb.isOpened():
            raise ValueError(f"Cannot open RGB video: {rgb_path}")

        cap_ir = None
        if ir_path is not None and Path(ir_path).exists():
            cap_ir = cv2.VideoCapture(str(ir_path))
            if not cap_ir.isOpened():
                logger.warning(f"Cannot open IR video: {ir_path}. Using RGB only.")
                cap_ir = None

        cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        if cap_ir is not None:
            cap_ir.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        h, w = self.input_size
        rgb_frames: List[np.ndarray] = []
        ir_frames: List[np.ndarray] = []

        for _ in range(length):
            ret_rgb, frame_rgb = cap_rgb.read()
            if not ret_rgb:
                break
            frame_rgb = _to_rgb(frame_rgb)
            rgb_frames.append(cv2.resize(frame_rgb, (w, h), interpolation=cv2.INTER_LINEAR))

            if cap_ir is not None:
                ret_ir, frame_ir = cap_ir.read()
                if ret_ir:
                    frame_ir = _to_gray(frame_ir)
                    ir_frames.append(
                        cv2.resize(frame_ir, (w, h), interpolation=cv2.INTER_LINEAR)
                    )

        cap_rgb.release()
        if cap_ir is not None:
            cap_ir.release()

        if not rgb_frames:
            raise ValueError(f"No readable frames in RGB video: {rgb_path}")

        # Pad RGB by repeating the last frame.
        while len(rgb_frames) < length:
            rgb_frames.append(rgb_frames[-1].copy())

        # Pad IR by repeating its last frame; if no IR, use zeros.
        has_ir = len(ir_frames) > 0
        if has_ir:
            while len(ir_frames) < length:
                ir_frames.append(ir_frames[-1].copy())
        else:
            ir_frames = [np.zeros((h, w), dtype=np.uint8) for _ in range(length)]

        rgb_sequence = np.stack(rgb_frames[:length]).astype(np.float32) / 255.0
        ir_sequence = np.stack(ir_frames[:length]).astype(np.float32) / 255.0
        return rgb_sequence, ir_sequence, has_ir


class CachedFeatureDataset(Dataset):
    """
    Dataset over precomputed per-frame RGB/IR features.

    Each cache file is a ``.pt`` dictionary with:
        - ``rgb_features``: (N, 2048) float tensor
        - ``ir_features``: (N, 2048) float tensor or None
        - ``sequence_id``: str
        - ``has_ir``: bool

    This avoids running YOLO + ResNet50 on every epoch.
    """

    def __init__(self,
                 cache_paths: Sequence[Path],
                 sequence_length: int = 16,
                 sequence_stride: int = 1,
                 rgb_dropout_prob: float = 0.15,
                 ir_dropout_prob: float = 0.15,
                 max_cached_videos: int = 4):
        self.cache_paths = [Path(p) for p in cache_paths]
        if not self.cache_paths:
            raise ValueError("cache_paths is empty")
        self.sequence_length = int(sequence_length)
        self.sequence_stride = max(1, int(sequence_stride))
        self.rgb_dropout_prob = float(rgb_dropout_prob)
        self.ir_dropout_prob = float(ir_dropout_prob)
        self.max_cached_videos = max(1, int(max_cached_videos))

        self.index: List[Tuple[int, int]] = []
        self.metadata: List[Dict] = []
        self._cache: "OrderedDict[int, Dict]" = OrderedDict()

        for cache_idx, path in enumerate(self.cache_paths):
            # Prefer the JSON sidecar: it lets a 100k-frame cache be indexed
            # without loading the feature tensors into RAM.
            sidecar = path.with_suffix(".json")
            meta = None
            if sidecar.exists():
                try:
                    with open(sidecar, "r", encoding="utf-8") as handle:
                        sidecar_data = json.load(handle)
                    meta = {
                        "path": str(path),
                        "sequence_id": sidecar_data.get("sequence_id", path.stem),
                        "num_frames": int(sidecar_data["num_frames"]),
                        "has_ir": bool(sidecar_data.get("has_ir", False)),
                    }
                except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
                    logger.warning("Invalid cache sidecar %s: %s", sidecar, exc)
                    meta = None

            if meta is None:
                data = self._load_cache(cache_idx)
                meta = {
                    "path": str(path),
                    "sequence_id": data.get("sequence_id", path.stem),
                    "num_frames": int(data["rgb_features"].shape[0]),
                    "has_ir": bool(data.get("has_ir", data.get("ir_features") is not None)),
                }

            self.metadata.append(meta)
            n_frames = meta["num_frames"]
            if n_frames >= self.sequence_length:
                starts = range(
                    0, n_frames - self.sequence_length + 1, self.sequence_stride
                )
                self.index.extend((cache_idx, start) for start in starts)

        if not self.index:
            raise ValueError(
                f"No sequences of length {self.sequence_length} in {len(self.cache_paths)} cache files"
            )
        logger.info(
            f"CachedFeatureDataset: {len(self.cache_paths)} videos, "
            f"{len(self.index)} sequences, seq_len={self.sequence_length}"
        )

    def _load_cache(self, cache_idx: int) -> Dict:
        if cache_idx in self._cache:
            self._cache.move_to_end(cache_idx)
            return self._cache[cache_idx]

        try:
            data = torch.load(self.cache_paths[cache_idx], map_location="cpu", weights_only=True)
        except Exception:
            data = torch.load(self.cache_paths[cache_idx], map_location="cpu", weights_only=False)
        if "rgb_features" not in data:
            raise ValueError(f"Cache file missing rgb_features: {self.cache_paths[cache_idx]}")
        self._cache[cache_idx] = data
        while len(self._cache) > self.max_cached_videos:
            self._cache.popitem(last=False)
        return data

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict:
        cache_idx, start = self.index[idx]
        data = self._load_cache(cache_idx)
        end = start + self.sequence_length

        rgb_sequence = data["rgb_features"][start:end].float()
        ir_features = data.get("ir_features")
        has_ir = bool(data.get("has_ir", ir_features is not None)) and ir_features is not None

        if has_ir:
            ir_sequence = ir_features[start:end].float()
        else:
            ir_sequence = torch.zeros(
                self.sequence_length, rgb_sequence.shape[-1], dtype=torch.float32
            )

        rgb_available = random.random() > self.rgb_dropout_prob
        ir_available = has_ir and (random.random() > self.ir_dropout_prob)
        if not rgb_available and not ir_available:
            rgb_available = True

        return {
            "rgb_sequence": rgb_sequence,
            "ir_sequence": ir_sequence,
            "rgb_available": rgb_available,
            "ir_available": ir_available,
            "sequence_id": data.get("sequence_id", self.cache_paths[cache_idx].stem),
        }


def split_videos_by_video(video_paths: List[Path],
                          ir_video_paths: Optional[List[Path]],
                          train_ratio: float = 0.8,
                          seed: int = 42) -> Tuple[List[Path], List[Path], Optional[List[Path]], Optional[List[Path]]]:
    """Split videos (not frames) into train/validation sets."""
    num_videos = len(video_paths)
    if num_videos == 0:
        raise ValueError("video_paths is empty")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")

    num_train = max(1, min(num_videos - 1, int(round(num_videos * train_ratio))))
    indices = list(range(num_videos))
    random.Random(seed).shuffle(indices)

    train_indices = sorted(indices[:num_train])
    val_indices = sorted(indices[num_train:])

    train_rgb = [video_paths[i] for i in train_indices]
    val_rgb = [video_paths[i] for i in val_indices]

    train_ir = [ir_video_paths[i] for i in train_indices] if ir_video_paths else None
    val_ir = [ir_video_paths[i] for i in val_indices] if ir_video_paths else None

    logger.info(f"Video split: {len(train_rgb)} train, {len(val_rgb)} val")
    return train_rgb, val_rgb, train_ir, val_ir
