"""
Prepare the KAIST Multispectral Pedestrian dataset for VigiLens.

KAIST ships as synchronized image sequences::

    set00/V000/visible/*.jpg
    set00/V000/lwir/*.jpg

This script converts each pair to ``<sequence>_rgb.mp4`` / ``<sequence>_ir.mp4``
under ``Data/train`` and ``Data/val``.  Use ``--max-frames`` for a quick
preview and ``--sets`` to select a subset (for example ``--sets 00``).

Note: KAIST is a pedestrian detection dataset, not a labeled anomaly-detection
benchmark.  It is useful as a large normal-pedestrian corpus for VigiLens
training, but anomaly metrics should be computed on a dataset with frame-level
anomaly labels (or on custom labeled clips).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.utils import ensure_dir, safe_stem
from prepare_otcbvs import _write_pair_from_images, split_pairs


def discover_sequences(root: Path) -> List[Tuple[Path, Path, str]]:
    sequences = []
    for visible_dir in root.rglob("visible"):
        if not visible_dir.is_dir():
            continue
        lwir_dir = visible_dir.parent / "lwir"
        if not lwir_dir.is_dir():
            # Some mirrors use ``thermal`` or ``ir`` instead of ``lwir``.
            for name in ("thermal", "ir", "infrared"):
                candidate = visible_dir.parent / name
                if candidate.is_dir():
                    lwir_dir = candidate
                    break
        if not lwir_dir.is_dir():
            continue

        relative = visible_dir.parent.relative_to(root)
        set_match = re.search(r"set\d+", str(relative).lower())
        set_name = set_match.group(0) if set_match else relative.parts[0] if relative.parts else "set00"
        sequence_id = safe_stem("_".join(relative.parts))
        sequences.append((visible_dir, lwir_dir, sequence_id))
    return sorted(sequences, key=lambda item: item[2])


def _set_number(sequence_id: str) -> Optional[int]:
    match = re.search(r"set(\d+)", sequence_id.lower())
    return int(match.group(1)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare KAIST RGB/IR sequences for VigiLens")
    parser.add_argument("--input", required=True, help="KAIST root containing set00/, set01/, ...")
    parser.add_argument("--output", default="Data")
    parser.add_argument("--sets", default=None, help="Comma-separated set numbers, e.g. 00,01")
    parser.add_argument("--val-sets", default=None, help="Comma-separated set numbers for validation")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--all-train", action="store_true")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    root = Path(args.input)
    if not root.exists():
        raise SystemExit(f"Input directory not found: {root}")

    sequences = discover_sequences(root)
    if not sequences:
        raise SystemExit(
            "No visible/lwir image directories found. Expected "
            "set00/V000/visible and set00/V000/lwir."
        )

    if args.sets:
        allowed = {int(value.strip()) for value in args.sets.split(",") if value.strip()}
        sequences = [item for item in sequences if _set_number(item[2]) in allowed]
    if not sequences:
        raise SystemExit("No sequences matched --sets")

    if args.all_train:
        train_sequences, val_sequences = sequences, []
    elif args.val_sets:
        val_sets = {int(value.strip()) for value in args.val_sets.split(",") if value.strip()}
        train_sequences = [item for item in sequences if _set_number(item[2]) not in val_sets]
        val_sequences = [item for item in sequences if _set_number(item[2]) in val_sets]
    else:
        # split_pairs expects (rgb, ir) tuples; keep sequence ids separately.
        train_pairs, val_pairs = split_pairs(
            [(item[0], item[1]) for item in sequences], args.val_ratio, args.seed
        )
        pair_to_id = {(item[0], item[1]): item[2] for item in sequences}
        train_sequences = [(rgb, ir, pair_to_id[(rgb, ir)]) for rgb, ir in train_pairs]
        val_sequences = [(rgb, ir, pair_to_id[(rgb, ir)]) for rgb, ir in val_pairs]

    output = Path(args.output)
    records = []
    for split, split_sequences in (("train", train_sequences), ("val", val_sequences)):
        for visible_dir, ir_dir, sequence_id in split_sequences:
            out_rgb = output / split / "rgb" / f"{sequence_id}_rgb.mp4"
            out_ir = output / split / "ir" / f"{sequence_id}_ir.mp4"
            try:
                count = _write_pair_from_images(
                    visible_dir, ir_dir, out_rgb, out_ir, args.fps, args.max_frames
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
                "source_rgb": str(visible_dir),
                "source_ir": str(ir_dir),
            })
            print(f"{split:5s} {sequence_id}: {count} frames -> {out_rgb.name}")

    manifest_path = Path(args.manifest) if args.manifest else output / "kaist_pairs.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump({"dataset": "KAIST", "pairs": records}, handle, indent=2)
    print(json.dumps({
        "train": len(train_sequences),
        "val": len(val_sequences),
        "manifest": str(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
