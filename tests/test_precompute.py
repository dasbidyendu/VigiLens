"""Tests for one-time feature precomputation."""

import json
from pathlib import Path

import numpy as np
import torch

from training.precompute import (
    LabelResolver,
    precompute_data_dir,
    precompute_manifest,
    precompute_pair,
)


class FakeExtractor:
    """Stand-in for RGB/IRFeatureExtractor; no ResNet download."""

    def __init__(self, dim=2048, seed=0):
        self.dim = dim
        self.rng = np.random.default_rng(seed)

    def extract_batch(self, frames, masks=None):
        return self.rng.normal(size=(len(frames), self.dim)).astype(np.float32)


def test_label_resolver_segments_constant_and_file(tmp_path):
    resolver = LabelResolver({"segments": [[2, 4]]}, tmp_path)
    assert [resolver(i) for i in range(6)] == [0, 0, 1, 1, 0, 0]

    constant = LabelResolver({"label": "anomaly"}, tmp_path)
    assert constant(0) == 1

    labels_file = tmp_path / "labels.txt"
    labels_file.write_text("0\nnormal\n1\nanomaly\n", encoding="utf-8")
    from_file = LabelResolver({"frame_labels": str(labels_file)}, tmp_path)
    assert [from_file(i) for i in range(4)] == [0, 0, 1, 1]


def test_precompute_pair_writes_cache_and_sidecar(make_video, tmp_path):
    rgb = make_video(tmp_path / "seq_rgb.avi", num_frames=6)
    ir = make_video(tmp_path / "seq_ir.avi", num_frames=6, grayscale=True)
    output = tmp_path / "cache"

    path = precompute_pair(
        rgb,
        ir,
        output,
        rgb_extractor=FakeExtractor(seed=1),
        ir_extractor=FakeExtractor(seed=2),
        batch_size=2,
        sequence_id="seq",
        label_spec={"segments": [[2, 4]]},
    )

    assert path is not None and path.exists()
    assert path.with_suffix(".json").exists()
    data = torch.load(path, map_location="cpu", weights_only=True)
    assert data["rgb_features"].shape == (6, 2048)
    assert data["ir_features"].shape == (6, 2048)
    assert data["has_ir"] is True
    assert data["sequence_id"] == "seq"
    assert data["labels"].tolist() == [0, 0, 1, 1, 0, 0]
    assert data["frame_indices"] == list(range(6))

    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["num_frames"] == 6
    assert sidecar["has_ir"] is True


def test_precompute_pair_skips_existing_unless_overwrite(make_video, tmp_path):
    rgb = make_video(tmp_path / "seq_rgb.avi", num_frames=3)
    output = tmp_path / "cache"
    kwargs = dict(
        rgb_extractor=FakeExtractor(),
        ir_extractor=FakeExtractor(),
        sequence_id="seq",
    )
    first = precompute_pair(rgb, None, output, **kwargs)
    mtime = first.stat().st_mtime_ns
    second = precompute_pair(rgb, None, output, **kwargs)
    assert second == first
    assert second.stat().st_mtime_ns == mtime


def test_precompute_pair_truncates_when_ir_is_shorter(make_video, tmp_path):
    rgb = make_video(tmp_path / "seq_rgb.avi", num_frames=8)
    ir = make_video(tmp_path / "seq_ir.avi", num_frames=4, grayscale=True)
    path = precompute_pair(
        rgb,
        ir,
        tmp_path / "cache",
        rgb_extractor=FakeExtractor(),
        ir_extractor=FakeExtractor(),
        batch_size=1,
        sequence_id="seq",
    )
    data = torch.load(path, map_location="cpu", weights_only=True)
    assert data["rgb_features"].shape[0] == 4
    assert data["has_ir"] is False
    assert data["ir_features"] is None


def test_precompute_manifest_and_data_dir(make_video, tmp_path):
    rgb = make_video(tmp_path / "manifest_rgb.avi", num_frames=5)
    ir = make_video(tmp_path / "manifest_ir.avi", num_frames=5, grayscale=True)
    manifest = {
        "videos": [
            {
                "sequence_id": "manifest",
                "rgb": str(rgb),
                "ir": str(ir),
                "segments": [[1, 3]],
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    paths = precompute_manifest(
        manifest_path,
        tmp_path / "cache" / "eval",
        rgb_extractor=FakeExtractor(),
        ir_extractor=FakeExtractor(),
        batch_size=2,
    )
    assert len(paths) == 1
    assert (tmp_path / "cache" / "eval" / "cache_manifest.json").exists()
    data = torch.load(paths[0], map_location="cpu", weights_only=True)
    assert data["labels"].tolist() == [0, 1, 1, 0, 0]

    data_dir = tmp_path / "Data"
    (data_dir / "train" / "rgb").mkdir(parents=True)
    (data_dir / "train" / "ir").mkdir(parents=True)
    train_rgb = make_video(data_dir / "train" / "rgb" / "scene_rgb.avi", num_frames=4)
    train_ir = make_video(data_dir / "train" / "ir" / "scene_ir.avi", num_frames=4, grayscale=True)
    cache_paths = precompute_data_dir(
        data_dir,
        tmp_path / "cache" / "data",
        splits=["train"],
        rgb_extractor=FakeExtractor(),
        ir_extractor=FakeExtractor(),
        batch_size=2,
    )
    assert len(cache_paths) >= 1
    assert train_rgb.exists() and train_ir.exists()
