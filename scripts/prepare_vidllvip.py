"""
Prepare VidLLVIP dataset for VigiLens.

VidLLVIP dataset structure:
    dataset/
        vi/  (visible videos)
        ir/  (infrared videos)

Files with identical names in vi/ and ir/ represent corresponding visible and infrared clips.

This script:
- Discovers RGB/IR pairs by exact filename matching
- Validates video readability and metadata
- Creates deterministic train/val/test splits at the source-video level
- Converts to MP4 under Data/train, Data/val, Data/test with consistent naming
- Reports missing files and data leakage warnings

Usage:
    python scripts/prepare_vidllvip.py --input /path/to/VidLLVIP --output Data --val-ratio 0.2 --test-ratio 0.1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_REPO_ROOT))

from training.utils import ensure_dir, safe_stem

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".webm", ".m4v"}


def _is_video(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def discover_vidllvip_pairs(root: Path) -> List[Tuple[Path, Path]]:
    """
    Discover RGB/IR pairs in VidLLVIP dataset structure.
    
    VidLLVIP uses exact filename matching between vi/ and ir/ directories.
    """
    vi_dir = root / "vi"
    ir_dir = root / "ir"
    
    if not vi_dir.exists():
        raise FileNotFoundError(f"Visible directory not found: {vi_dir}")
    if not ir_dir.exists():
        raise FileNotFoundError(f"IR directory not found: {ir_dir}")
    
    vi_videos = [path for path in vi_dir.iterdir() if _is_video(path)]
    ir_videos = [path for path in ir_dir.iterdir() if _is_video(path)]
    
    # Create mapping from filename to IR path
    ir_by_name = {path.name: path for path in ir_videos}
    
    pairs = []
    missing_ir = []
    missing_vi = []
    
    for vi_path in vi_videos:
        ir_path = ir_by_name.get(vi_path.name)
        if ir_path is not None:
            pairs.append((vi_path, ir_path))
        else:
            missing_ir.append(vi_path.name)
    
    # Check for IR videos without corresponding VI
    vi_names = {path.name for path in vi_videos}
    for ir_path in ir_videos:
        if ir_path.name not in vi_names:
            missing_vi.append(ir_path.name)
    
    if missing_ir:
        print(f"Warning: {len(missing_ir)} visible videos without IR pair (will be skipped):")
        for name in missing_ir[:10]:
            print(f"  - {name}")
        if len(missing_ir) > 10:
            print(f"  ... and {len(missing_ir) - 10} more")
    
    if missing_vi:
        print(f"Warning: {len(missing_vi)} IR videos without visible pair (will be skipped):")
        for name in missing_vi[:10]:
            print(f"  - {name}")
        if len(missing_vi) > 10:
            print(f"  ... and {len(missing_vi) - 10} more")
    
    return pairs


def extract_source_id(filename: str) -> Optional[str]:
    """
    Attempt to extract source video ID from VidLLVIP filename.
    
    VidLLVIP clips are derived from a smaller set of source videos.
    We need to identify clips from the same source to prevent data leakage.
    
    VidLLVIP naming convention varies; this is a best-effort heuristic.
    Common patterns:
    - clip_001_0001.mp4 -> source: clip_001
    - scene01_clip01.mp4 -> source: scene01
    - 001_0001.mp4 -> source: 001
    """
    # Remove extension
    name = Path(filename).stem
    
    # Try pattern: prefix_number_number (extract prefix_number)
    # This handles cases like clip_001_0001 -> clip_001
    match = re.match(r'^(.+_\d+)_(\d+)$', name)
    if match:
        return match.group(1)
    
    # Try pattern: prefix_number (extract prefix)
    # This handles cases like scene01_clip01 -> scene01
    match = re.match(r'^(.+)_\d+$', name)
    if match:
        return match.group(1)
    
    # Pattern: number_number (extract first number)
    match = re.match(r'^(\d+)_(\d+)$', name)
    if match:
        return match.group(1)
    
    # If no pattern matches, use the full filename as source ID
    # This means each clip is treated as independent (no leakage prevention)
    return name


def validate_video_pair(rgb_path: Path, ir_path: Path) -> Dict:
    """Validate video pair and return metadata."""
    rgb_cap = cv2.VideoCapture(str(rgb_path))
    ir_cap = cv2.VideoCapture(str(ir_path))
    
    if not rgb_cap.isOpened() or not ir_cap.isOpened():
        rgb_cap.release()
        ir_cap.release()
        raise ValueError(f"Cannot open video pair: {rgb_path}, {ir_path}")
    
    rgb_frame_count = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ir_frame_count = int(ir_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rgb_fps = rgb_cap.get(cv2.CAP_PROP_FPS) or 20.0
    ir_fps = ir_cap.get(cv2.CAP_PROP_FPS) or 20.0
    rgb_w = int(rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rgb_h = int(rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ir_w = int(ir_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ir_h = int(ir_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    rgb_cap.release()
    ir_cap.release()
    
    return {
        "rgb_frames": rgb_frame_count,
        "ir_frames": ir_frame_count,
        "rgb_fps": rgb_fps,
        "ir_fps": ir_fps,
        "rgb_resolution": (rgb_w, rgb_h),
        "ir_resolution": (ir_w, ir_h),
    }


def split_by_source(pairs: Sequence[Tuple[Path, Path]],
                   val_ratio: float,
                   test_ratio: float,
                   seed: int) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """
    Split pairs by source video ID to prevent data leakage.
    
    Clips from the same source video must not appear in different splits.
    """
    # Group pairs by source ID
    source_to_pairs: Dict[str, List[Tuple[Path, Path]]] = {}
    for rgb_path, ir_path in pairs:
        source_id = extract_source_id(rgb_path.name)
        if source_id not in source_to_pairs:
            source_to_pairs[source_id] = []
        source_to_pairs[source_id].append((rgb_path, ir_path))
    
    # Get list of unique source IDs
    source_ids = list(source_to_pairs.keys())
    
    # Shuffle sources
    random.Random(seed).shuffle(source_ids)
    
    # Calculate split counts
    n_sources = len(source_ids)
    n_test = max(1, int(round(n_sources * test_ratio))) if test_ratio > 0 else 0
    n_val = max(1, int(round(n_sources * val_ratio))) if val_ratio > 0 else 0
    n_train = n_sources - n_val - n_test
    
    if n_train <= 0:
        print(f"Warning: Too few sources ({n_sources}) for train/val/test split.")
        print(f"  Requested: test={test_ratio}, val={val_ratio}")
        print(f"  Result: train={n_train}, val={n_val}, test={n_test}")
        print(f"  Falling back to clip-level splitting (may cause data leakage)")
        # Fall back to clip-level split
        return split_by_clip(pairs, val_ratio, test_ratio, seed)
    
    # Split sources
    test_sources = source_ids[:n_test]
    val_sources = source_ids[n_test:n_test + n_val]
    train_sources = source_ids[n_test + n_val:]
    
    # Assign pairs to splits
    train_pairs = []
    val_pairs = []
    test_pairs = []
    
    for source_id in train_sources:
        train_pairs.extend(source_to_pairs[source_id])
    for source_id in val_sources:
        val_pairs.extend(source_to_pairs[source_id])
    for source_id in test_sources:
        test_pairs.extend(source_to_pairs[source_id])
    
    print(f"Source-level split: {len(train_sources)} train sources, {len(val_sources)} val sources, {len(test_sources)} test sources")
    print(f"  -> {len(train_pairs)} train clips, {len(val_pairs)} val clips, {len(test_pairs)} test clips")
    
    return train_pairs, val_pairs, test_pairs


def split_by_clip(pairs: Sequence[Tuple[Path, Path]],
                 val_ratio: float,
                 test_ratio: float,
                 seed: int) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """
    Fallback: split by clip (may cause data leakage).
    """
    pairs_list = list(pairs)
    random.Random(seed).shuffle(pairs_list)
    
    n_pairs = len(pairs_list)
    n_test = max(1, int(round(n_pairs * test_ratio))) if test_ratio > 0 else 0
    n_val = max(1, int(round(n_pairs * val_ratio))) if val_ratio > 0 else 0
    n_train = n_pairs - n_val - n_test
    
    if n_train <= 0:
        raise ValueError(f"Too few pairs ({n_pairs}) for requested splits")
    
    test_pairs = pairs_list[:n_test]
    val_pairs = pairs_list[n_test:n_test + n_val]
    train_pairs = pairs_list[n_test + n_val:]
    
    print(f"Clip-level split (WARNING: may cause data leakage):")
    print(f"  -> {len(train_pairs)} train clips, {len(val_pairs)} val clips, {len(test_pairs)} test clips")
    
    return train_pairs, val_pairs, test_pairs


def convert_pair(rgb_path: Path,
                ir_path: Path,
                out_rgb: Path,
                out_ir: Path,
                fps: Optional[float] = None,
                max_frames: Optional[int] = None) -> int:
    """Convert a video pair to MP4 format."""
    rgb_cap = cv2.VideoCapture(str(rgb_path))
    ir_cap = cv2.VideoCapture(str(ir_path))
    
    if not rgb_cap.isOpened() or not ir_cap.isOpened():
        rgb_cap.release()
        ir_cap.release()
        raise ValueError(f"Cannot open video pair: {rgb_path}, {ir_path}")
    
    source_fps = float(rgb_cap.get(cv2.CAP_PROP_FPS)) or 20.0
    output_fps = float(fps) if fps else source_fps
    rgb_w = int(rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rgb_h = int(rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ir_w = int(ir_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ir_h = int(ir_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    ensure_dir(out_rgb.parent)
    ensure_dir(out_ir.parent)
    
    rgb_writer = cv2.VideoWriter(str(out_rgb), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (rgb_w, rgb_h))
    ir_writer = cv2.VideoWriter(str(out_ir), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (ir_w, ir_h), isColor=False)
    
    if not rgb_writer.isOpened() or not ir_writer.isOpened():
        rgb_cap.release()
        ir_cap.release()
        rgb_writer.release()
        ir_writer.release()
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
            
            # Ensure IR is grayscale
            if ir.ndim == 3:
                ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
            
            rgb_writer.write(rgb)
            ir_writer.write(ir)
            count += 1
    finally:
        rgb_cap.release()
        ir_cap.release()
        rgb_writer.release()
        ir_writer.release()
    
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare VidLLVIP dataset for VigiLens")
    parser.add_argument("--input", required=True, help="VidLLVIP dataset root directory (contains vi/ and ir/)")
    parser.add_argument("--output", default="Data", help="Output directory for prepared data")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio")
    parser.add_argument("--all-train", action="store_true", help="Put all pairs in train (no val/test)")
    parser.add_argument("--fps", type=float, default=None, help="Output FPS (default: use source FPS)")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames per clip")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--manifest", default=None, help="Optional manifest output path")
    args = parser.parse_args()
    
    root = Path(args.input)
    if not root.exists():
        raise SystemExit(f"Input directory not found: {root}")
    
    print(f"Discovering VidLLVIP pairs in {root}...")
    pairs = discover_vidllvip_pairs(root)
    
    if not pairs:
        raise SystemExit(
            "No RGB/IR pairs found. Ensure the dataset has vi/ and ir/ directories "
            "with matching video filenames."
        )
    
    print(f"Found {len(pairs)} valid RGB/IR pairs")
    
    # Split data
    if args.all_train:
        train_pairs, val_pairs, test_pairs = pairs, [], []
        print("All pairs assigned to training (--all-train)")
    else:
        train_pairs, val_pairs, test_pairs = split_by_source(
            pairs, args.val_ratio, args.test_ratio, args.seed
        )
    
    output = Path(args.output)
    records = []
    
    for split, split_pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        if not split_pairs:
            print(f"Skipping {split} (no pairs)")
            continue
        
        split_dir = output / split
        for rgb_path, ir_path in split_pairs:
            sequence_id = safe_stem(rgb_path.stem)
            out_rgb = split_dir / "rgb" / f"{sequence_id}_rgb.mp4"
            out_ir = split_dir / "ir" / f"{sequence_id}_ir.mp4"
            
            try:
                metadata = validate_video_pair(rgb_path, ir_path)
                count = convert_pair(rgb_path, ir_path, out_rgb, out_ir, args.fps, args.max_frames)
            except Exception as exc:
                print(f"Skipping {sequence_id}: {exc}")
                continue
            
            source_id = extract_source_id(rgb_path.name)
            records.append({
                "sequence_id": sequence_id,
                "source_id": source_id,
                "split": split,
                "frames": count,
                "rgb": str(out_rgb),
                "ir": str(out_ir),
                "source_rgb": str(rgb_path),
                "source_ir": str(ir_path),
                "metadata": metadata,
            })
            print(f"{split:5s} {sequence_id} (source: {source_id}): {count} frames")
    
    # Write manifest
    manifest_path = Path(args.manifest) if args.manifest else output / "vidllvip_pairs.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump({
            "dataset": "VidLLVIP",
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
            "split_method": "source_level",
            "pairs": records,
        }, handle, indent=2)
    
    print(f"\nSummary:")
    print(f"  Total pairs: {len(pairs)}")
    print(f"  Train: {len(train_pairs)}")
    print(f"  Val: {len(val_pairs)}")
    print(f"  Test: {len(test_pairs)}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
