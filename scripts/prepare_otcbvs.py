"""
Prepare OTCBVS Dataset 10 (or a similar visible/thermal collection) for
VigiLens.

The script is deliberately tolerant of layouts.  It discovers:

* video pairs whose filenames contain visible/color/rgb and thermal/ir/lwir
  markers (for example ``Video_1.avi`` + ``Video_1_thermal.avi``);
* paired image directories (``visible/`` + ``thermal/``, ``color/`` + ``lwir/``,
  etc.);
* already-paired files with ``_rgb`` / ``_ir`` suffixes.

Everything is converted to MP4 under ``Data/train`` and ``Data/val`` with a
consistent ``<sequence>_rgb.mp4`` / ``<sequence>_ir.mp4`` naming convention.

Example::

    python scripts/prepare_otcbvs.py --input /content/OTCBVS --output Data
    python scripts/prepare_otcbvs.py --input ... --max-frames 300 --all-train
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.utils import (
    IMAGE_EXTENSIONS,
    ensure_dir,
    list_images,
    natural_key,
    safe_stem,
)

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".webm", ".m4v"}
VISIBLE_WORDS = ("visible", "color", "colour", "rgb", "day")
IR_WORDS = ("thermal", "infrared", "lwir", "ir", "night")
MODALITY_WORDS = VISIBLE_WORDS + IR_WORDS


def _classify(name: str) -> Optional[str]:
    lowered = name.lower()
    if any(word in lowered for word in IR_WORDS):
        return "ir"
    if any(word in lowered for word in VISIBLE_WORDS):
        return "rgb"
    return None


def _pair_key(name: str) -> str:
    lowered = name.lower()
    for word in sorted(MODALITY_WORDS, key=len, reverse=True):
        lowered = lowered.replace(word, "")
    return re.sub(r"[^a-z0-9]+", "", lowered)


def _is_video(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def discover_video_pairs(root: Path) -> List[Tuple[Path, Path]]:
    videos = [path for path in root.rglob("*") if _is_video(path)]
    rgb_videos = [path for path in videos if _classify(path.stem) == "rgb"]
    ir_videos = [path for path in videos if _classify(path.stem) == "ir"]

    ir_by_key: Dict[str, Path] = {}
    for path in ir_videos:
        ir_by_key.setdefault(_pair_key(path.stem), path)

    pairs = []
    used_ir = set()
    for rgb_path in rgb_videos:
        key = _pair_key(rgb_path.stem)
        ir_path = ir_by_key.get(key)
        if ir_path is not None and ir_path not in used_ir:
            pairs.append((rgb_path, ir_path))
            used_ir.add(ir_path)

    # Fallback: files with matching stems but no modality keyword in one name.
    if not pairs:
        rgb_by_key = {_pair_key(path.stem): path for path in videos}
        for path in videos:
            key = _pair_key(path.stem)
            other = rgb_by_key.get(key)
            if other is not None and other != path and _classify(path.stem) == "ir":
                pairs.append((other, path))

    return pairs


def discover_directory_pairs(root: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    for directory in root.rglob("*"):
        if not directory.is_dir():
            continue
        name = directory.name.lower()
        if not any(word in name for word in VISIBLE_WORDS):
            continue
        parent = directory.parent
        candidates = []
        for sibling in parent.iterdir():
            if not sibling.is_dir() or sibling == directory:
                continue
            sibling_name = sibling.name.lower()
            if any(word in sibling_name for word in IR_WORDS):
                candidates.append(sibling)
        if candidates:
            pairs.append((directory, candidates[0]))
    return pairs


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    frame = frame.astype(np.float32)
    low, high = float(frame.min()), float(frame.max())
    if high > low:
        frame = (frame - low) * (255.0 / (high - low))
    return np.clip(frame, 0, 255).astype(np.uint8)


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    image = _to_uint8(image)
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    return image


def _write_pair_from_images(visible_dir: Path,
                            ir_dir: Path,
                            out_rgb: Path,
                            out_ir: Path,
                            fps: float,
                            max_frames: Optional[int]) -> int:
    visible_images = list_images(visible_dir)
    ir_images = list_images(ir_dir)
    count = min(len(visible_images), len(ir_images))
    if max_frames is not None:
        count = min(count, max_frames)
    if count == 0:
        raise ValueError(f"No image pairs in {visible_dir} and {ir_dir}")

    first_rgb = _read_image(visible_images[0])
    first_ir = _read_image(ir_images[0])
    rgb_h, rgb_w = first_rgb.shape[:2]
    ir_h, ir_w = first_ir.shape[:2]

    ensure_dir(out_rgb.parent)
    ensure_dir(out_ir.parent)
    rgb_writer = cv2.VideoWriter(str(out_rgb), cv2.VideoWriter_fourcc(*"mp4v"), fps, (rgb_w, rgb_h))
    ir_writer = cv2.VideoWriter(
        str(out_ir), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ir_w, ir_h), isColor=False
    )
    if not rgb_writer.isOpened() or not ir_writer.isOpened():
        raise RuntimeError(f"Could not open output videos: {out_rgb} / {out_ir}")

    try:
        for index in range(count):
            rgb = _read_image(visible_images[index])
            ir = _read_image(ir_images[index])
            if rgb.ndim == 2:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
            if rgb.shape[:2] != (rgb_h, rgb_w):
                rgb = cv2.resize(rgb, (rgb_w, rgb_h))
            if ir.ndim == 3:
                ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
            if ir.shape[:2] != (ir_h, ir_w):
                ir = cv2.resize(ir, (ir_w, ir_h))
            rgb_writer.write(rgb)
            ir_writer.write(ir)
    finally:
        rgb_writer.release()
        ir_writer.release()
    return count


def _write_pair_from_videos(rgb_path: Path,
                            ir_path: Path,
                            out_rgb: Path,
                            out_ir: Path,
                            fps: Optional[float],
                            max_frames: Optional[int]) -> int:
    rgb_cap = cv2.VideoCapture(str(rgb_path))
    ir_cap = cv2.VideoCapture(str(ir_path))
    if not rgb_cap.isOpened() or not ir_cap.isOpened():
        rgb_cap.release()
        ir_cap.release()
        raise ValueError(f"Could not open video pair: {rgb_path}, {ir_path}")

    source_fps = float(rgb_cap.get(cv2.CAP_PROP_FPS)) or 20.0
    output_fps = float(fps) if fps else source_fps
    rgb_w = int(rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rgb_h = int(rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ir_w = int(ir_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ir_h = int(ir_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ensure_dir(out_rgb.parent)
    ensure_dir(out_ir.parent)
    rgb_writer = cv2.VideoWriter(str(out_rgb), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (rgb_w, rgb_h))
    ir_writer = cv2.VideoWriter(
        str(out_ir), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (ir_w, ir_h), isColor=False
    )
    if not rgb_writer.isOpened() or not ir_writer.isOpened():
        rgb_cap.release()
        ir_cap.release()
        raise RuntimeError(f"Could not open output videos: {out_rgb} / {out_ir}")

    count = 0
    try:
        while True:
            if max_frames is not None and count >= max_frames:
                break
            ret_rgb, rgb = rgb_cap.read()
            ret_ir, ir = ir_cap.read()
            if not ret_rgb or not ret_ir:
                break
            if rgb.shape[1] != rgb_w or rgb.shape[0] != rgb_h:
                rgb = cv2.resize(rgb, (rgb_w, rgb_h))
            if ir.ndim == 3:
                ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
            if ir.shape[1] != ir_w or ir.shape[0] != ir_h:
                ir = cv2.resize(ir, (ir_w, ir_h))
            rgb_writer.write(rgb)
            ir_writer.write(ir)
            count += 1
    finally:
        rgb_cap.release()
        ir_cap.release()
        rgb_writer.release()
        ir_writer.release()
    return count


def split_pairs(pairs: Sequence[Tuple[Path, Path]],
                val_ratio: float,
                seed: int) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    pairs = list(pairs)
    if not pairs:
        return [], []
    if val_ratio <= 0 or len(pairs) == 1:
        return pairs, []
    shuffled = pairs[:]
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_count = min(val_count, len(shuffled) - 1)
    return shuffled[val_count:], shuffled[:val_count]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OTCBVS Dataset 10 for VigiLens")
    parser.add_argument("--input", required=True, help="OTCBVS root directory")
    parser.add_argument("--output", default="Data")
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--all-train", action="store_true", help="Put every pair in train")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", default=None, help="Optional pairs.json output")
    args = parser.parse_args()

    root = Path(args.input)
    if not root.exists():
        raise SystemExit(f"Input directory not found: {root}")

    pairs = discover_video_pairs(root)
    if not pairs:
        pairs = discover_directory_pairs(root)
    if not pairs:
        print("No visible/thermal pairs found. Top-level contents:")
        for path in list(root.rglob("*"))[:40]:
            print(" ", path)
        raise SystemExit(
            "Could not discover pairs. Ensure filenames contain visible/color/rgb "
            "and thermal/ir/lwir markers, or that paired image directories exist."
        )

    train_pairs, val_pairs = (pairs, []) if args.all_train else split_pairs(pairs, args.val_ratio, args.seed)
    output = Path(args.output)
    records = []

    for split, split_pairs_list in (("train", train_pairs), ("val", val_pairs)):
        for rgb_path, ir_path in split_pairs_list:
            sequence_id = safe_stem(rgb_path.stem if rgb_path.is_file() else rgb_path.parent.name)
            out_rgb = output / split / "rgb" / f"{sequence_id}_rgb.mp4"
            out_ir = output / split / "ir" / f"{sequence_id}_ir.mp4"
            try:
                if rgb_path.is_dir():
                    count = _write_pair_from_images(
                        rgb_path, ir_path, out_rgb, out_ir, args.fps, args.max_frames
                    )
                else:
                    count = _write_pair_from_videos(
                        rgb_path, ir_path, out_rgb, out_ir, args.fps, args.max_frames
                    )
            except Exception as exc:
                print(f"Skipping {sequence_id}: {exc}")
                continue
            records.append({
                "sequence_id": sequence_id,
                "split": split,
                "frames": count,
                "rgb": str(out_rgb),
                "ir": str(out_ir),
                "source_rgb": str(rgb_path),
                "source_ir": str(ir_path),
            })
            print(f"{split:5s} {sequence_id}: {count} frames -> {out_rgb.name}")

    manifest_path = Path(args.manifest) if args.manifest else output / "pairs.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump({"dataset": "OTCBVS", "pairs": records}, handle, indent=2)
    print(json.dumps({"pairs": len(records), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
