"""
Precompute RGB/IR features once, then train fusion + LSTM on the cached
features.

Running YOLO and two ResNet50 encoders on every epoch is the main bottleneck in
VigiLens training.  This script performs that work once per video and stores:

    <sequence>.pt      rgb_features (N, 2048), ir_features (N, 2048) or None,
                       labels (N,) when available, fps, frame indices, ...
    <sequence>.json    small sidecar with counts/metadata (used by the dataset
                       without loading the tensors)

Typical use::

    python -m training.precompute --data-dir Data --output cache --device cuda
    python -m training.precompute --manifest Evaluation/manifest.json \
        --output cache/eval --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from tqdm import tqdm

from training.utils import (
    IMAGE_EXTENSIONS,
    discover_videos,
    ensure_dir,
    extract_sequence_id,
    pair_videos_by_sequence_id,
    safe_stem,
)

logger = logging.getLogger(__name__)


class LabelResolver:
    """Resolve a frame index to 0/1 from a manifest entry."""

    def __init__(self, spec: Optional[Dict[str, Any]], base_dir: Path):
        self.spec = spec or {}
        self.labels: Optional[np.ndarray] = None
        self.segments: List[Tuple[int, int]] = []
        self.constant: Optional[int] = None

        label_file = self.spec.get("frame_labels")
        if label_file:
            path = Path(label_file)
            if not path.is_absolute():
                path = base_dir / path
            self.labels = _load_label_file(path)
        elif self.spec.get("segments"):
            for segment in self.spec["segments"]:
                if len(segment) != 2:
                    raise ValueError(f"Invalid segment {segment}; expected [start, end]")
                start, end = int(segment[0]), int(segment[1])
                self.segments.append((start, end))
        elif "label" in self.spec:
            self.constant = _label_to_int(self.spec["label"])

    def __call__(self, frame_index: int) -> int:
        if self.labels is not None:
            if 0 <= frame_index < self.labels.size:
                return int(self.labels[frame_index])
            return 0
        if self.segments:
            return int(any(start <= frame_index < end for start, end in self.segments))
        return int(self.constant or 0)


def _label_to_int(value: Any) -> int:
    if isinstance(value, str):
        return 1 if value.strip().lower() in {"1", "anomaly", "abnormal", "positive", "true"} else 0
    return 1 if int(value) else 0


def _load_label_file(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        labels = np.load(path)
    else:
        values = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                token = line.strip()
                if not token or token.startswith("#"):
                    continue
                # Accept "0", "1", "normal", "anomaly", or "frame,label".
                token = token.split(",")[-1].strip()
                values.append(_label_to_int(token))
        labels = np.asarray(values, dtype=np.int64)
    return np.asarray(labels, dtype=np.int64).reshape(-1)


def _resize_mask(mask: Optional[np.ndarray], frame: np.ndarray) -> Optional[np.ndarray]:
    if mask is None:
        return None
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def _open_capture(path: Path) -> Optional[cv2.VideoCapture]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return None
    return capture


def precompute_pair(rgb_path: Union[str, Path],
                    ir_path: Optional[Union[str, Path]],
                    output_dir: Union[str, Path],
                    segmentation=None,
                    rgb_extractor=None,
                    ir_extractor=None,
                    batch_size: int = 16,
                    max_frames: Optional[int] = None,
                    frame_stride: int = 1,
                    start_frame: int = 0,
                    sequence_id: Optional[str] = None,
                    label_spec: Optional[Dict[str, Any]] = None,
                    label_base_dir: Optional[Union[str, Path]] = None,
                    save_half: bool = False,
                    overwrite: bool = False) -> Optional[Path]:
    """
    Extract and save features for one RGB/IR video pair.

    Returns the cache path, or ``None`` when the cache already exists and
    ``overwrite`` is false.
    """
    if rgb_extractor is None or ir_extractor is None:
        raise ValueError("rgb_extractor and ir_extractor are required")

    rgb_path = Path(rgb_path)
    ir_path = Path(ir_path) if ir_path is not None else None
    output_dir = ensure_dir(Path(output_dir))
    sequence_id = sequence_id or extract_sequence_id(rgb_path)
    cache_path = output_dir / f"{safe_stem(sequence_id)}.pt"
    sidecar_path = cache_path.with_suffix(".json")
    if cache_path.exists() and not overwrite:
        logger.info("Skipping existing cache: %s", cache_path)
        return cache_path

    rgb_cap = _open_capture(rgb_path)
    if rgb_cap is None:
        raise ValueError(f"Cannot open RGB video: {rgb_path}")

    ir_cap = _open_capture(ir_path) if ir_path is not None else None
    if ir_path is not None and ir_cap is None:
        logger.warning("Cannot open IR video %s; using RGB only", ir_path)

    fps = float(rgb_cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 20.0

    resolver = LabelResolver(label_spec, Path(label_base_dir or "."))
    frame_stride = max(1, int(frame_stride))
    start_frame = max(0, int(start_frame))
    max_frames = int(max_frames) if max_frames is not None else None

    rgb_features: List[np.ndarray] = []
    ir_features: List[np.ndarray] = []
    frame_indices: List[int] = []
    labels: List[int] = []
    has_ir = ir_cap is not None
    input_index = 0
    batch_frames_rgb: List[np.ndarray] = []
    batch_frames_ir: List[np.ndarray] = []
    batch_indices: List[int] = []
    batch_labels: List[int] = []

    def flush_batch() -> None:
        nonlocal has_ir
        if not batch_frames_rgb:
            return
        masks = None
        if segmentation is not None:
            masks = segmentation.extract_person_masks(batch_frames_rgb)

        rgb_feats = rgb_extractor.extract_batch(batch_frames_rgb, masks)
        rgb_features.extend(list(rgb_feats))

        if has_ir and batch_frames_ir:
            ir_masks = []
            for rgb_frame, ir_frame, mask in zip(batch_frames_rgb, batch_frames_ir, masks or [None] * len(batch_frames_rgb)):
                ir_masks.append(_resize_mask(mask, ir_frame))
            ir_feats = ir_extractor.extract_batch(batch_frames_ir, ir_masks)
            ir_features.extend(list(ir_feats))
        elif has_ir:
            has_ir = False

        frame_indices.extend(batch_indices)
        labels.extend(batch_labels)
        batch_frames_rgb.clear()
        batch_frames_ir.clear()
        batch_indices.clear()
        batch_labels.clear()

    try:
        while True:
            if max_frames is not None and len(frame_indices) + len(batch_indices) >= max_frames:
                break

            ret_rgb, rgb_frame = rgb_cap.read()
            if not ret_rgb:
                break

            ret_ir, ir_frame = (True, None)
            if ir_cap is not None:
                ret_ir, ir_frame = ir_cap.read()

            current_index = input_index
            input_index += 1
            if current_index < start_frame:
                continue
            if (current_index - start_frame) % frame_stride != 0:
                continue

            if ir_cap is not None and not ret_ir:
                logger.warning("IR video ended at frame %d; truncating pair %s", current_index, sequence_id)
                has_ir = False
                ir_cap.release()
                ir_cap = None
                break

            batch_frames_rgb.append(rgb_frame)
            if has_ir and ir_frame is not None:
                batch_frames_ir.append(ir_frame)
            batch_indices.append(current_index)
            batch_labels.append(resolver(current_index))

            if len(batch_frames_rgb) >= max(1, int(batch_size)):
                flush_batch()
    finally:
        rgb_cap.release()
        if ir_cap is not None:
            ir_cap.release()

    flush_batch()

    if not rgb_features:
        logger.warning("No frames were processed for %s", sequence_id)
        return None

    rgb_tensor = torch.from_numpy(np.stack(rgb_features).astype(np.float32))
    ir_tensor = None
    if has_ir and ir_features and len(ir_features) == len(rgb_features):
        ir_tensor = torch.from_numpy(np.stack(ir_features).astype(np.float32))
    else:
        has_ir = False

    if save_half:
        rgb_tensor = rgb_tensor.half()
        if ir_tensor is not None:
            ir_tensor = ir_tensor.half()

    data = {
        "rgb_features": rgb_tensor,
        "ir_features": ir_tensor,
        "has_ir": bool(has_ir),
        "sequence_id": sequence_id,
        "fps": float(fps),
        "frame_indices": frame_indices,
        "labels": torch.tensor(labels, dtype=torch.int64) if labels else None,
        "source_rgb": str(rgb_path),
        "source_ir": str(ir_path) if ir_path is not None else None,
        "frame_stride": frame_stride,
        "start_frame": start_frame,
        "dtype": "float16" if save_half else "float32",
    }
    torch.save(data, cache_path)

    sidecar = {
        "sequence_id": sequence_id,
        "num_frames": int(rgb_tensor.shape[0]),
        "has_ir": bool(has_ir),
        "fps": float(fps),
        "source_rgb": str(rgb_path),
        "source_ir": str(ir_path) if ir_path is not None else None,
        "frame_stride": frame_stride,
        "start_frame": start_frame,
        "dtype": data["dtype"],
    }
    with open(sidecar_path, "w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, indent=2)

    logger.info(
        "Cached %s: %d frames, has_ir=%s -> %s",
        sequence_id, rgb_tensor.shape[0], has_ir, cache_path,
    )
    return cache_path


def _resolve_path(value: Union[str, Path], base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path)


def precompute_manifest(manifest_path: Union[str, Path],
                        output_dir: Union[str, Path],
                        segmentation=None,
                        rgb_extractor=None,
                        ir_extractor=None,
                        batch_size: int = 16,
                        max_frames: Optional[int] = None,
                        frame_stride: int = 1,
                        save_half: bool = False,
                        overwrite: bool = False,
                        limit: Optional[int] = None) -> List[Path]:
    """Precompute caches for every video in an evaluation manifest."""
    manifest_path = Path(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    base_dir = manifest_path.parent
    entries = manifest.get("videos", manifest) if isinstance(manifest, dict) else manifest
    if not isinstance(entries, list):
        raise ValueError("Manifest must contain a 'videos' list")

    if limit is not None:
        entries = entries[:limit]

    cache_paths = []
    for entry in tqdm(entries, desc="Precomputing manifest"):
        if "rgb" not in entry:
            raise ValueError(f"Manifest entry has no 'rgb' field: {entry}")
        rgb_path = _resolve_path(entry["rgb"], base_dir)
        ir_path = _resolve_path(entry["ir"], base_dir) if entry.get("ir") else None
        sequence_id = entry.get("sequence_id") or extract_sequence_id(rgb_path)
        cache_path = precompute_pair(
            rgb_path,
            ir_path,
            output_dir,
            segmentation=segmentation,
            rgb_extractor=rgb_extractor,
            ir_extractor=ir_extractor,
            batch_size=batch_size,
            max_frames=max_frames,
            frame_stride=frame_stride,
            sequence_id=sequence_id,
            label_spec=entry,
            label_base_dir=base_dir,
            save_half=save_half,
            overwrite=overwrite,
        )
        if cache_path is not None:
            cache_paths.append(cache_path)

    mapping = {path.stem: str(path) for path in cache_paths}
    mapping_path = Path(output_dir) / "cache_manifest.json"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mapping_path, "w", encoding="utf-8") as handle:
        json.dump(mapping, handle, indent=2)
    logger.info("Wrote cache mapping to %s", mapping_path)
    return cache_paths


def precompute_data_dir(data_dir: Union[str, Path],
                        output_dir: Union[str, Path],
                        splits: Sequence[str] = ("train", "val"),
                        segmentation=None,
                        rgb_extractor=None,
                        ir_extractor=None,
                        batch_size: int = 16,
                        max_frames: Optional[int] = None,
                        frame_stride: int = 1,
                        save_half: bool = False,
                        overwrite: bool = False) -> List[Path]:
    """Precompute caches for ``Data/<split>/rgb`` and ``Data/<split>/ir``."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    cache_paths = []

    for split in splits:
        split_rgb_dir = data_dir / split / "rgb"
        if not split_rgb_dir.exists():
            logger.warning("Skipping split %s: %s does not exist", split, split_rgb_dir)
            continue

        rgb_videos = discover_videos(data_dir, split, "rgb", required=True)
        ir_videos = discover_videos(data_dir, split, "ir", required=False)
        if ir_videos:
            pairs = pair_videos_by_sequence_id(rgb_videos, ir_videos)
        else:
            logger.warning("No IR videos for split %s; using RGB-only caches", split)
            pairs = [(path, None) for path in rgb_videos]

        split_output = ensure_dir(output_dir / split)
        for rgb_path, ir_path in tqdm(pairs, desc=f"Precomputing {split}"):
            cache_path = precompute_pair(
                rgb_path,
                ir_path,
                split_output,
                segmentation=segmentation,
                rgb_extractor=rgb_extractor,
                ir_extractor=ir_extractor,
                batch_size=batch_size,
                max_frames=max_frames,
                frame_stride=frame_stride,
                save_half=save_half,
                overwrite=overwrite,
            )
            if cache_path is not None:
                cache_paths.append(cache_path)

    return cache_paths


def build_extractors(device: str, use_amp: bool = True):
    from models.feature_extractor import IRFeatureExtractor, RGBFeatureExtractor

    return (
        RGBFeatureExtractor(device=device, use_amp=use_amp),
        IRFeatureExtractor(device=device, use_amp=use_amp),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute VigiLens RGB/IR features")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-dir", help="Dataset root containing <split>/rgb and <split>/ir")
    source.add_argument("--manifest", help="Evaluation manifest JSON")
    parser.add_argument("--output", required=True, help="Cache output directory")
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-half", action="store_true", help="Store float16 features")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-segmentation", action="store_true", help="Use full frames (no YOLO)")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    segmentation = None
    if not args.no_segmentation:
        from models.segmentation import YOLOSegmentation

        segmentation = YOLOSegmentation(device=args.device)

    rgb_extractor, ir_extractor = build_extractors(args.device, use_amp=not args.no_amp)

    if args.manifest:
        paths = precompute_manifest(
            args.manifest,
            args.output,
            segmentation=segmentation,
            rgb_extractor=rgb_extractor,
            ir_extractor=ir_extractor,
            batch_size=args.batch_size,
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
            save_half=args.save_half,
            overwrite=args.overwrite,
            limit=args.limit,
        )
    else:
        paths = precompute_data_dir(
            args.data_dir,
            args.output,
            splits=[split.strip() for split in args.splits.split(",") if split.strip()],
            segmentation=segmentation,
            rgb_extractor=rgb_extractor,
            ir_extractor=ir_extractor,
            batch_size=args.batch_size,
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
            save_half=args.save_half,
            overwrite=args.overwrite,
        )

    print(json.dumps({"caches": [str(path) for path in paths], "count": len(paths)}, indent=2))


if __name__ == "__main__":
    main()
