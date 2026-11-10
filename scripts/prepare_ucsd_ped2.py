"""
Prepare UCSD Ped2 (a labeled pedestrian-walkway anomaly benchmark) for VigiLens.

UCSD Ped2 is greyscale/visible-only: VigiLens runs in RGB-only mode on it
(the fusion module's RGB branch).  It is used here because it has frame-level
anomaly labels (bikes, carts, skaters on a pedestrian walkway), so AUC/AP can
be measured.  Expected layout (the official ``UCSD_Anomaly_Dataset.v1p2``)::

    UCSDped2/
      Train/Train001/001.tif ...        normal only
      Test/Test001/001.tif ...          contains anomalies
      Test/Test001_gt/001.bmp ...       optional pixel masks
      UCSDped2.m                        optional, has  gt_frame = [61:180];

Outputs::

    <output>/train/rgb/ped2_train001_rgb.mp4 ...   (normal, for training)
    <output>/val/rgb/ped2_train015_rgb.mp4 ...     (held-out normal, calibration)
    <eval-root>/rgb/ped2_test001.mp4 ...
    <eval-root>/labels/ped2_test001.txt            one 0/1 per frame
    <eval-root>/manifest.json                      for training.precompute/evaluate

Frame labels come from ``UCSDped2.m`` (1-based inclusive ``gt_frame`` ranges)
when present, otherwise from the ``TestXXX_gt`` masks (frame anomalous if its
mask has any foreground pixel).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.utils import IMAGE_EXTENSIONS, ensure_dir, list_images, prepare_split_dirs


def find_ped2_root(path: Path) -> Path:
    path = Path(path)
    if (path / "Train").is_dir() and (path / "Test").is_dir():
        return path
    for candidate in path.rglob("UCSDped2"):
        if (candidate / "Train").is_dir() and (candidate / "Test").is_dir():
            return candidate
    raise SystemExit(f"Could not find UCSDped2/Train and UCSDped2/Test under {path}")


def _clip_dirs(split_dir: Path, prefix: str) -> List[Path]:
    pattern = re.compile(rf"^{prefix}\d+$")
    return sorted(p for p in split_dir.iterdir() if p.is_dir() and pattern.match(p.name))


def frames_to_video(frame_dir: Path, out_path: Path, fps: float) -> int:
    images = list_images(frame_dir, IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f"No frames in {frame_dir}")
    first = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError(f"Cannot read {images[0]}")
    height, width = first.shape[:2]
    ensure_dir(out_path.parent)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Could not open {out_path} for writing")
    try:
        for image_path in images:
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Cannot read {image_path}")
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    finally:
        writer.release()
    return len(images)


def parse_gt_frames(m_file: Path) -> Dict[int, List[tuple]]:
    """Parse ``TestVideoFile{end+1}.gt_frame = [61:180];`` lines (1-based)."""
    ranges: Dict[int, List[tuple]] = {}
    index = 0
    for line in m_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = re.search(r"gt_frame\s*=\s*\[([^\]]*)\]", line)
        if not match:
            continue
        index += 1
        spans = []
        for start, end in re.findall(r"(\d+)\s*:\s*(\d+)", match.group(1)):
            spans.append((int(start), int(end)))
        ranges[index] = spans
    return ranges


def labels_from_masks(gt_dir: Path, num_frames: int) -> Optional[np.ndarray]:
    masks = list_images(gt_dir, IMAGE_EXTENSIONS) if gt_dir.is_dir() else []
    if not masks:
        return None
    labels = np.zeros(num_frames, dtype=np.int64)
    for index, mask_path in enumerate(masks[:num_frames]):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        labels[index] = int(mask is not None and bool((mask > 0).any()))
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare UCSD Ped2 for VigiLens")
    parser.add_argument("--input", required=True, help="Folder containing UCSDped2/")
    parser.add_argument("--output", default="Data_ped2", help="Training data root")
    parser.add_argument("--eval-root", default="Evaluation/ped2", help="Evaluation output root")
    parser.add_argument("--val-videos", type=int, default=2,
                        help="Hold out the last N normal training videos for calibration")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--clean", action="store_true",
                        help="Delete existing <output>/train and <output>/val folders first")
    args = parser.parse_args()

    root = find_ped2_root(Path(args.input))
    output = Path(args.output)
    eval_root = Path(args.eval_root)
    prepare_split_dirs(output, ("train", "val"), args.clean)

    train_clips = _clip_dirs(root / "Train", "Train")
    if len(train_clips) <= args.val_videos:
        raise SystemExit(f"Only {len(train_clips)} training clips; lower --val-videos")
    val_clips = train_clips[-args.val_videos:] if args.val_videos > 0 else []
    for clip in train_clips:
        split = "val" if clip in val_clips else "train"
        out = output / split / "rgb" / f"ped2_{clip.name.lower()}_rgb.mp4"
        count = frames_to_video(clip, out, args.fps)
        print(f"{split:5s} {clip.name}: {count} frames")

    m_files = list(root.glob("*.m"))
    gt_ranges = parse_gt_frames(m_files[0]) if m_files else {}
    entries = []
    for number, clip in enumerate(_clip_dirs(root / "Test", "Test"), start=1):
        sequence_id = f"ped2_{clip.name.lower()}"
        out = eval_root / "rgb" / f"{sequence_id}.mp4"
        count = frames_to_video(clip, out, args.fps)

        labels = None
        if number in gt_ranges:
            labels = np.zeros(count, dtype=np.int64)
            for start, end in gt_ranges[number]:
                labels[max(0, start - 1):min(count, end)] = 1
        if labels is None:
            labels = labels_from_masks(clip.parent / f"{clip.name}_gt", count)
        if labels is None:
            raise SystemExit(f"No ground truth for {clip.name}: need UCSDped2.m or {clip.name}_gt/")

        label_path = eval_root / "labels" / f"{sequence_id}.txt"
        ensure_dir(label_path.parent)
        label_path.write_text("\n".join(str(int(v)) for v in labels) + "\n", encoding="utf-8")
        entries.append({
            "sequence_id": sequence_id,
            "rgb": str(out.resolve()),
            "ir": None,
            "label": "anomaly" if labels.any() else "normal",
            "frame_labels": str(label_path.resolve()),
        })
        print(f"test  {clip.name}: {count} frames, {int(labels.sum())} anomalous")

    manifest_path = eval_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump({"name": "ucsd_ped2", "videos": entries}, handle, indent=2)
    print(json.dumps({"train_clips": len(train_clips) - len(val_clips), "val_clips": len(val_clips),
                      "test_clips": len(entries), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
