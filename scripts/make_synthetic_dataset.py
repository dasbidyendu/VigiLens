"""
Generate a tiny synthetic RGB/IR dataset for an end-to-end smoke test.

The synthetic "pedestrians" are moving rectangles.  They are intentionally
simple: YOLO will usually detect no person, so the smoke test exercises the
full-frame feature path.  That is fine for verifying the plumbing (precompute
-> train -> calibrate -> evaluate) without downloading a real dataset.

The evaluation split contains both normal clips and clips whose second half is
anomalous, so frame-level AUC/AP/F1 can be demonstrated immediately.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def _parse_size(value: str):
    width, height = value.lower().split("x")
    return int(width), int(height)


def _draw_frame(width: int,
                height: int,
                t: int,
                frames: int,
                phase: float,
                anomalous: bool,
                rng: np.random.Generator):
    rgb = np.full((height, width, 3), 22, dtype=np.uint8)
    ir = np.full((height, width), 12, dtype=np.uint8)

    # Background texture.
    for _ in range(8):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        cv2.circle(rgb, (x, y), int(rng.integers(4, 18)), (40, 40, 40), -1)

    margin = max(12, width // 8)
    speed = 1.0
    if anomalous and t >= frames // 2:
        speed = 4.0
    x = margin + int((width - 2 * margin) * ((t * speed / max(frames, 1) + phase) % 1.0))
    y = height // 2 + int((height // 5) * math.sin(2 * math.pi * t / max(frames, 1) + phase))
    w = max(10, width // 10)
    h = max(20, height // 4)

    color = (60, 170, 240) if not (anomalous and t >= frames // 2) else (40, 40, 230)
    cv2.rectangle(rgb, (x, y), (min(width - 1, x + w), min(height - 1, y + h)), color, -1)
    cv2.rectangle(ir, (x, y), (min(width - 1, x + w), min(height - 1, y + h)), 235, -1)

    # A little sensor noise.
    rgb = np.clip(rgb.astype(np.int16) + rng.integers(-6, 7, rgb.shape), 0, 255).astype(np.uint8)
    ir = np.clip(ir.astype(np.int16) + rng.integers(-4, 5, ir.shape), 0, 255).astype(np.uint8)
    return rgb, ir


def _write_video(path: Path,
                 frames: int,
                 width: int,
                 height: int,
                 fps: float,
                 phase: float,
                 anomalous: bool,
                 seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")
    rng = np.random.default_rng(seed)
    try:
        for t in range(frames):
            rgb, _ = _draw_frame(width, height, t, frames, phase, anomalous, rng)
            writer.write(rgb)
    finally:
        writer.release()


def _write_ir_video(path: Path,
                    frames: int,
                    width: int,
                    height: int,
                    fps: float,
                    phase: float,
                    anomalous: bool,
                    seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), isColor=False
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")
    rng = np.random.default_rng(seed + 1)
    try:
        for t in range(frames):
            _, ir = _draw_frame(width, height, t, frames, phase, anomalous, rng)
            writer.write(ir)
    finally:
        writer.release()


def _make_pair(root: Path,
               name: str,
               frames: int,
               width: int,
               height: int,
               fps: float,
               phase: float,
               anomalous: bool,
               seed: int) -> dict:
    rgb_path = (root / "rgb" / f"{name}_rgb.mp4").resolve()
    ir_path = (root / "ir" / f"{name}_ir.mp4").resolve()
    _write_video(rgb_path, frames, width, height, fps, phase, anomalous, seed)
    _write_ir_video(ir_path, frames, width, height, fps, phase, anomalous, seed)
    entry = {
        "sequence_id": name,
        "rgb": str(rgb_path),
        "ir": str(ir_path),
        "label": "anomaly" if anomalous else "normal",
    }
    if anomalous:
        entry["segments"] = [[frames // 2, frames]]
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a synthetic VigiLens smoke dataset")
    parser.add_argument("--output", default="Data")
    parser.add_argument("--eval-dir", default="Evaluation")
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--size", default="320x240")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--num-train", type=int, default=4)
    parser.add_argument("--num-val", type=int, default=2)
    parser.add_argument("--num-normal-eval", type=int, default=2)
    parser.add_argument("--num-anomaly", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    width, height = _parse_size(args.size)
    output = Path(args.output)
    eval_dir = Path(args.eval_dir)
    if output.exists() and not args.overwrite:
        existing = list(output.rglob("*.mp4"))
        if existing:
            raise SystemExit(f"{output} already contains videos; pass --overwrite to replace")
    if eval_dir.exists() and not args.overwrite:
        existing = list(eval_dir.rglob("*.mp4"))
        if existing:
            raise SystemExit(f"{eval_dir} already contains videos; pass --overwrite to replace")

    for split, count in (("train", args.num_train), ("val", args.num_val)):
        for index in range(count):
            _make_pair(
                output / split, f"{split}_{index:03d}", args.frames,
                width, height, args.fps, phase=index * 0.17,
                anomalous=False, seed=args.seed + index,
            )

    entries = []
    for index in range(args.num_normal_eval):
        entries.append(_make_pair(
            eval_dir / "normal", f"normal_{index:03d}", args.frames,
            width, height, args.fps, phase=0.1 + index * 0.13,
            anomalous=False, seed=args.seed + 100 + index,
        ))
    for index in range(args.num_anomaly):
        entries.append(_make_pair(
            eval_dir / "anomaly", f"anomaly_{index:03d}", args.frames,
            width, height, args.fps, phase=0.2 + index * 0.11,
            anomalous=True, seed=args.seed + 200 + index,
        ))

    manifest = {"name": "synthetic_smoke", "videos": entries}
    manifest_path = eval_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(json.dumps({
        "output": str(output),
        "eval_dir": str(eval_dir),
        "manifest": str(manifest_path),
        "train_videos": args.num_train,
        "val_videos": args.num_val,
        "eval_videos": len(entries),
    }, indent=2))


if __name__ == "__main__":
    main()
