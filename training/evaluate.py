"""
Offline frame-level evaluation for VigiLens.

The evaluation manifest is intentionally simple and dataset-agnostic::

    {
      "name": "my_eval_set",
      "videos": [
        {
          "sequence_id": "scene001",
          "rgb": "rgb/scene001_rgb.mp4",
          "ir": "ir/scene001_ir.mp4",
          "label": "anomaly",
          "segments": [[120, 260]],
          "cache": "cache/scene001.pt"
        },
        {
          "sequence_id": "scene002",
          "rgb": "rgb/scene002_rgb.mp4",
          "label": "normal"
        }
      ]
    }

Labels can be supplied in three ways (in priority order):

1. ``frame_labels``: a ``.txt`` file with one 0/1 (or normal/anomaly) per line,
   or a ``.npy`` array.
2. ``segments``: a list of ``[start, end)`` anomalous frame intervals.
3. ``label``: ``"normal"`` / ``"anomaly"`` for the whole video.

If none are present, the labels stored in the precomputed cache are used.

Metrics are frame-level ROC-AUC, average precision, precision/recall/F1 at a
threshold, FPR, and FPR at 95% TPR.  A normal-only manifest is valid: AUC/AP
are reported as NaN and the false-positive rate is still meaningful.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from training.metrics import compute_binary_metrics, compute_video_metrics
from training.scoring import read_cache_metadata, score_cache_file
from training.utils import safe_stem

logger = logging.getLogger(__name__)


def _resolve(base_dir: Path, value: Union[str, Path]) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path)


def _label_to_int(value: Any) -> int:
    if isinstance(value, str):
        return 1 if value.strip().lower() in {"1", "anomaly", "abnormal", "positive", "true"} else 0
    return 1 if int(value) else 0


def _load_label_file(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        values = np.load(path)
    else:
        values = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                token = line.strip()
                if not token or token.startswith("#"):
                    continue
                token = token.split(",")[-1].strip()
                values.append(_label_to_int(token))
        values = np.asarray(values, dtype=np.int64)
    return np.asarray(values, dtype=np.int64).reshape(-1)


def _align_labels(labels: np.ndarray, num_frames: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels.size == num_frames:
        return labels
    if labels.size > num_frames:
        logger.warning("Truncating %d labels to %d frames", labels.size, num_frames)
        return labels[:num_frames]
    logger.warning("Padding %d labels to %d frames with zeros", labels.size, num_frames)
    return np.pad(labels, (0, num_frames - labels.size), constant_values=0)


def labels_for_entry(entry: Dict[str, Any],
                     num_frames: int,
                     base_dir: Path,
                     cache_labels: Optional[np.ndarray] = None) -> np.ndarray:
    """Resolve frame labels for one manifest entry."""
    if entry.get("frame_labels"):
        path = _resolve(base_dir, entry["frame_labels"])
        labels = _load_label_file(path)
        return _align_labels(labels, num_frames)

    if "segments" in entry:
        labels = np.zeros(num_frames, dtype=np.int64)
        for segment in entry.get("segments") or []:
            if len(segment) != 2:
                raise ValueError(f"Invalid segment {segment}; expected [start, end]")
            start = max(0, int(segment[0]))
            end = min(num_frames, int(segment[1]))
            if end > start:
                labels[start:end] = 1
        return labels

    if "label" in entry:
        return np.full(num_frames, _label_to_int(entry["label"]), dtype=np.int64)

    if cache_labels is not None:
        return _align_labels(cache_labels, num_frames)

    return np.zeros(num_frames, dtype=np.int64)


def resolve_cache_path(entry: Dict[str, Any],
                       base_dir: Path,
                       cache_dir: Optional[Path],
                       cache_mapping: Optional[Dict[str, str]] = None) -> Path:
    """Find the precomputed cache for a manifest entry."""
    candidates = []
    if entry.get("cache"):
        candidates.append(_resolve(base_dir, entry["cache"]))
    sequence_id = entry.get("sequence_id")
    if cache_mapping and sequence_id and sequence_id in cache_mapping:
        candidates.append(Path(cache_mapping[sequence_id]))
    if cache_dir is not None:
        if sequence_id:
            candidates.append(cache_dir / f"{safe_stem(sequence_id)}.pt")
            candidates.append(cache_dir / f"{sequence_id}.pt")
        if entry.get("rgb"):
            candidates.append(cache_dir / f"{safe_stem(Path(entry['rgb']).stem)}.pt")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find a feature cache for entry "
        f"{entry.get('sequence_id', entry.get('rgb', '<unknown>'))}. "
        "Run `python -m training.precompute --manifest ... --output cache/eval` "
        "or pass --cache-dir."
    )


def evaluate_manifest(model,
                      manifest_path: Union[str, Path],
                      cache_dir: Optional[Union[str, Path]] = None,
                      output_dir: Union[str, Path] = "Evaluation/results",
                      calibration_path: Optional[Union[str, Path]] = None,
                      threshold: Optional[float] = None,
                      smoothing_window: int = 1,
                      chunk_size: int = 256,
                      device: Optional[Union[str, torch.device]] = None,
                      write_csv: bool = True) -> Dict[str, Any]:
    """Score a manifest and write per-frame scores plus a JSON summary."""
    manifest_path = Path(manifest_path)
    base_dir = manifest_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    entries = manifest.get("videos", manifest) if isinstance(manifest, dict) else manifest
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest must contain a non-empty 'videos' list")

    cache_root = Path(cache_dir) if cache_dir is not None else None
    cache_mapping = {}
    if cache_root is not None:
        mapping_path = cache_root / "cache_manifest.json"
        if mapping_path.exists():
            with open(mapping_path, "r", encoding="utf-8") as handle:
                cache_mapping = json.load(handle)

    calibration = None
    if calibration_path is not None and Path(calibration_path).exists():
        with open(calibration_path, "r", encoding="utf-8") as handle:
            calibration = json.load(handle)
        if threshold is None:
            threshold = calibration.get("threshold", calibration.get("threshold_std"))
    if threshold is None:
        threshold = 0.5

    if hasattr(model, "eval"):
        model.eval()
    if hasattr(model, "fusion"):
        model.fusion.eval()
    if hasattr(model, "temporal"):
        model.temporal.eval()

    all_raw: List[np.ndarray] = []
    all_smoothed: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    per_video: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []

    for entry in entries:
        cache_path = resolve_cache_path(entry, base_dir, cache_root, cache_mapping)
        result = score_cache_file(
            model,
            cache_path,
            smoothing_window=smoothing_window,
            chunk_size=chunk_size,
            device=device,
        )
        labels = labels_for_entry(
            entry, result["num_frames"], base_dir, result.get("labels")
        )
        sequence_id = entry.get("sequence_id", result["sequence_id"])

        all_raw.append(result["raw_scores"])
        all_smoothed.append(result["smoothed_scores"])
        all_labels.append(labels)

        video_label = int(labels.any())
        per_video.append({
            "sequence_id": sequence_id,
            "cache": str(cache_path),
            "num_frames": int(result["num_frames"]),
            "label": video_label,
            "max_raw_score": float(np.max(result["raw_scores"])) if result["raw_scores"].size else 0.0,
            "max_smoothed_score": float(np.max(result["smoothed_scores"])) if result["smoothed_scores"].size else 0.0,
            "mean_raw_score": float(np.mean(result["raw_scores"])) if result["raw_scores"].size else 0.0,
            "has_ir": bool(result["has_ir"]),
        })

        for frame_index in range(result["num_frames"]):
            csv_rows.append({
                "sequence_id": sequence_id,
                "frame_index": frame_index,
                "raw_score": float(result["raw_scores"][frame_index]),
                "smoothed_score": float(result["smoothed_scores"][frame_index]),
                "label": int(labels[frame_index]),
            })

    raw_scores = np.concatenate(all_raw)
    smoothed_scores = np.concatenate(all_smoothed)
    labels = np.concatenate(all_labels)
    primary_scores = smoothed_scores if smoothing_window > 1 else raw_scores
    score_type = "smoothed" if smoothing_window > 1 else "raw"

    metrics = compute_binary_metrics(labels, primary_scores, threshold=threshold)
    metrics_raw = compute_binary_metrics(labels, raw_scores, threshold=threshold)
    metrics_smoothed = compute_binary_metrics(labels, smoothed_scores, threshold=threshold)

    # Best-F1 threshold is useful when no calibration file was supplied.  The
    # metrics above use the requested/calibrated threshold; expose the
    # data-driven alternative separately.
    best_metrics = compute_binary_metrics(labels, primary_scores, threshold=None)
    video_metrics = compute_video_metrics(
        [video["label"] for video in per_video],
        [video["max_smoothed_score"] if smoothing_window > 1 else video["max_raw_score"] for video in per_video],
    )

    summary = {
        "manifest": str(manifest_path),
        "cache_dir": str(cache_root) if cache_root is not None else None,
        "num_videos": len(per_video),
        "num_frames": int(labels.size),
        "score_type": score_type,
        "smoothing_window": int(smoothing_window),
        "threshold": float(threshold),
        "calibration": calibration,
        "metrics": metrics,
        "metrics_raw": metrics_raw,
        "metrics_smoothed": metrics_smoothed,
        "metrics_at_best_f1_threshold": best_metrics,
        "video_metrics": video_metrics,
        "per_video": per_video,
        "notes": (
            "AUC/AP are NaN when the evaluation set contains only one class. "
            "FPR is still valid for a normal-only smoke test."
        ),
    }

    scores_csv = output_dir / "scores.csv"
    results_json = output_dir / "results.json"
    if write_csv:
        with open(scores_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["sequence_id", "frame_index", "raw_score", "smoothed_score", "label"],
            )
            writer.writeheader()
            writer.writerows(csv_rows)
    with open(results_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("Evaluation results written to %s", results_json)
    return summary


def _print_summary(summary: Dict[str, Any]) -> None:
    metrics = summary["metrics"]
    print("\nVigiLens evaluation")
    print("=" * 60)
    print(f"Videos: {summary['num_videos']}  Frames: {summary['num_frames']}")
    print(f"Score type: {summary['score_type']}  Threshold: {summary['threshold']:.6f}")
    print(f"AUC: {metrics['auc']:.4f}" if np.isfinite(metrics["auc"]) else "AUC: n/a (single class)")
    print(f"AP:  {metrics['ap']:.4f}" if np.isfinite(metrics["ap"]) else "AP:  n/a (single class)")
    print(
        f"Precision: {metrics['precision']:.4f}  Recall: {metrics['recall']:.4f}  "
        f"F1: {metrics['f1']:.4f}"
    )
    print(
        f"FPR: {metrics['fpr']:.4f}  FPR@95TPR: {metrics['fpr_at_95tpr']:.4f}"
        if np.isfinite(metrics["fpr_at_95tpr"])
        else f"FPR: {metrics['fpr']:.4f}  FPR@95TPR: n/a"
    )
    print(f"Best-F1 threshold: {summary['metrics_at_best_f1_threshold']['best_f1_threshold']:.6f}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VigiLens on a labeled manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--smoothing-window", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--output-dir", default="Evaluation/results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from models.anomaly_model import VigiLensModel

    model = VigiLensModel(device=args.device, load_extractors=False)
    model.load_checkpoint(args.checkpoint)
    model.to(args.device)

    summary = evaluate_manifest(
        model,
        args.manifest,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        calibration_path=args.calibration,
        threshold=args.threshold,
        smoothing_window=args.smoothing_window,
        chunk_size=args.chunk_size,
        device=args.device,
        write_csv=not args.no_csv,
    )
    _print_summary(summary)


if __name__ == "__main__":
    main()
