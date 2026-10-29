"""Shared fixtures for the VigiLens test suite.

The tests never download YOLO or ResNet50 weights.  Cached-feature tests use
``VigiLensModel(load_extractors=False)`` and tiny random feature tensors.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.anomaly_model import VigiLensModel


@pytest.fixture
def tiny_model() -> VigiLensModel:
    """A fusion + LSTM model with no ResNet encoders."""
    torch.manual_seed(0)
    model = VigiLensModel(device="cpu", load_extractors=False)
    model.eval()
    return model


@pytest.fixture
def make_cache(tmp_path):
    """Write a small feature cache plus its JSON sidecar."""

    def _make(path,
              num_frames: int = 12,
              has_ir: bool = True,
              labels: Optional[list] = None,
              seed: int = 0,
              sequence_id: Optional[str] = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        generator = torch.Generator().manual_seed(seed)
        rgb = torch.randn(num_frames, 2048, generator=generator, dtype=torch.float32)
        ir = None
        if has_ir:
            ir = torch.randn(num_frames, 2048, generator=generator, dtype=torch.float32)
        label_tensor = None
        if labels is not None:
            label_tensor = torch.tensor(labels, dtype=torch.int64)
        sequence_id = sequence_id or path.stem
        data = {
            "rgb_features": rgb,
            "ir_features": ir,
            "has_ir": bool(has_ir and ir is not None),
            "sequence_id": sequence_id,
            "fps": 20.0,
            "frame_indices": list(range(num_frames)),
            "labels": label_tensor,
            "source_rgb": f"{sequence_id}_rgb.mp4",
            "source_ir": f"{sequence_id}_ir.mp4" if has_ir else None,
            "dtype": "float32",
        }
        torch.save(data, path)
        sidecar = {
            "sequence_id": sequence_id,
            "num_frames": int(num_frames),
            "has_ir": bool(data["has_ir"]),
            "fps": 20.0,
            "source_rgb": data["source_rgb"],
            "source_ir": data["source_ir"],
            "dtype": "float32",
        }
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as handle:
            json.dump(sidecar, handle, indent=2)
        return path

    return _make


@pytest.fixture
def make_video(tmp_path):
    """Write a tiny MJPG AVI with a solid colour per frame."""

    def _make(path, num_frames: int = 6, size=(64, 48), color=(10, 20, 30), grayscale=False):
        import cv2

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        width, height = size
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            10.0,
            (width, height),
            isColor=not grayscale,
        )
        assert writer.isOpened(), f"Could not open VideoWriter for {path}"
        try:
            for index in range(num_frames):
                if grayscale:
                    frame = np.full((height, width), 20 + index * 10, dtype=np.uint8)
                else:
                    frame = np.full((height, width, 3), color, dtype=np.uint8)
                    frame[:, :, 0] = np.clip(10 + index * 5, 0, 255)
                writer.write(frame)
        finally:
            writer.release()
        return path

    return _make
