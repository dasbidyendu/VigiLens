"""
Build an evaluation manifest from a simple folder layout::

    Evaluation/
      normal/
        rgb/*.mp4          (or flat *.mp4)
        ir/*.mp4           (optional)
      anomaly/
        rgb/*.mp4
        ir/*.mp4
      labels/              (optional frame labels, <sequence>.txt)
      segments.json        (optional {"sequence": [[start, end], ...]})

This is intentionally small; any existing labeled dataset can also be described
with the manifest format directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.utils import (
    DEFAULT_VIDEO_EXTENSIONS,
    extract_sequence_id,
    natural_key,
    pair_videos_by_sequence_id,
)


def _videos(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(
        (path for path in directory.rglob("*")
         if path.is_file() and path.suffix.lower() in DEFAULT_VIDEO_EXTENSIONS),
        key=lambda path: natural_key(str(path)),
    )


def _discover_split(root: Path, split_dir: Path, label: str) -> List[Dict]:
    rgb_dir = split_dir / "rgb"
    ir_dir = split_dir / "ir"
    if rgb_dir.exists():
        rgb_videos = _videos(rgb_dir)
        ir_videos = _videos(ir_dir)
        if ir_videos:
            pairs = pair_videos_by_sequence_id(rgb_videos, ir_videos)
        else:
            pairs = [(path, None) for path in rgb_videos]
    else:
        pairs = [(path, None) for path in _videos(split_dir)]

    entries = []
    for rgb_path, ir_path in pairs:
        sequence_id = extract_sequence_id(rgb_path)
        entries.append({
            "sequence_id": sequence_id,
            "rgb": str(rgb_path.resolve()),
            "ir": str(ir_path.resolve()) if ir_path is not None else None,
            "label": label,
        })
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a VigiLens evaluation manifest")
    parser.add_argument("--root", default="Evaluation")
    parser.add_argument("--output", default=None)
    parser.add_argument("--normal-dir", default="normal")
    parser.add_argument("--anomaly-dir", default="anomaly")
    parser.add_argument("--segments", default=None, help="JSON: {sequence_id: [[start,end], ...]}")
    parser.add_argument("--labels-dir", default="labels")
    parser.add_argument("--anomaly-all", action="store_true", help="Label entire anomaly videos as anomalous")
    args = parser.parse_args()

    root = Path(args.root)
    normal_dir = root / args.normal_dir
    anomaly_dir = root / args.anomaly_dir
    if not normal_dir.exists() and not anomaly_dir.exists():
        raise SystemExit(f"Neither {normal_dir} nor {anomaly_dir} exists")

    entries = _discover_split(root, normal_dir, "normal")
    entries += _discover_split(root, anomaly_dir, "anomaly")

    segments = {}
    if args.segments:
        with open(args.segments, "r", encoding="utf-8") as handle:
            segments = json.load(handle)

    labels_dir = root / args.labels_dir
    for entry in entries:
        sequence_id = entry["sequence_id"]
        label_file = labels_dir / f"{sequence_id}.txt"
        if label_file.exists():
            entry["frame_labels"] = str(label_file.resolve())
        if entry["label"] == "anomaly":
            if sequence_id in segments:
                entry["segments"] = segments[sequence_id]
            elif args.anomaly_all:
                entry["segments"] = []

    output = Path(args.output) if args.output else root / "manifest.json"
    manifest = {
        "name": root.name,
        "videos": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps({"manifest": str(output), "videos": len(entries)}, indent=2))


if __name__ == "__main__":
    main()
