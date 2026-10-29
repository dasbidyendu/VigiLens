"""Tests for the video and cached-feature datasets."""

from pathlib import Path

import numpy as np
import torch

from training.dataset import (
    CachedFeatureDataset,
    NormalVideoDataset,
    split_videos_by_video,
)
from training.utils import extract_sequence_id, pair_videos_by_sequence_id


def test_cached_dataset_indexes_sidecars(make_cache, tmp_path):
    first = make_cache(tmp_path / "cache" / "a.pt", num_frames=10, has_ir=True)
    second = make_cache(tmp_path / "cache" / "b.pt", num_frames=6, has_ir=False, seed=1)

    dataset = CachedFeatureDataset(
        [first, second],
        sequence_length=4,
        sequence_stride=2,
        rgb_dropout_prob=0.0,
        ir_dropout_prob=0.0,
    )

    # 10 frames -> starts 0,2,4,6; 6 frames -> starts 0,2
    assert len(dataset) == 6
    assert [meta["num_frames"] for meta in dataset.metadata] == [10, 6]
    assert dataset.metadata[0]["has_ir"] is True
    assert dataset.metadata[1]["has_ir"] is False

    item = dataset[0]
    assert item["rgb_sequence"].shape == (4, 2048)
    assert item["ir_sequence"].shape == (4, 2048)
    assert item["rgb_available"] is True
    assert item["ir_available"] is True

    no_ir = dataset[-1]
    assert no_ir["ir_available"] is False
    assert torch.allclose(no_ir["ir_sequence"], torch.zeros_like(no_ir["ir_sequence"]))


def test_cached_dataset_loads_without_sidecar(make_cache, tmp_path):
    path = make_cache(tmp_path / "c.pt", num_frames=5, has_ir=True)
    path.with_suffix(".json").unlink()

    dataset = CachedFeatureDataset([path], sequence_length=5, sequence_stride=1)
    assert len(dataset) == 1
    assert dataset.metadata[0]["num_frames"] == 5
    assert dataset[0]["rgb_sequence"].shape == (5, 2048)


def test_cached_dataset_modality_dropout_never_removes_both(make_cache, tmp_path):
    path = make_cache(tmp_path / "d.pt", num_frames=6, has_ir=True)
    dataset = CachedFeatureDataset(
        [path],
        sequence_length=3,
        rgb_dropout_prob=1.0,
        ir_dropout_prob=1.0,
    )
    item = dataset[0]
    assert item["rgb_available"] is True
    assert item["ir_available"] is False


def test_normal_video_dataset_pads_and_handles_missing_ir(make_video, tmp_path):
    rgb = make_video(tmp_path / "rgb.avi", num_frames=5, size=(64, 48))
    dataset = NormalVideoDataset(
        [rgb], ir_video_paths=None, sequence_length=3, sequence_stride=1,
        rgb_dropout_prob=0.0, ir_dropout_prob=0.0,
    )
    assert len(dataset) == 3
    item = dataset[0]
    assert item["rgb_sequence"].shape == (3, 224, 224, 3)
    assert item["ir_sequence"].shape == (3, 224, 224)
    assert item["ir_available"] is False
    assert item["rgb_sequence"].dtype == np.float32
    assert 0.0 <= float(item["rgb_sequence"].min()) <= float(item["rgb_sequence"].max()) <= 1.0


def test_split_videos_by_video_is_deterministic(tmp_path):
    videos = [tmp_path / f"scene{i:02d}_rgb.mp4" for i in range(10)]
    train_a, val_a, _, _ = split_videos_by_video(videos, None, train_ratio=0.7, seed=42)
    train_b, val_b, _, _ = split_videos_by_video(videos, None, train_ratio=0.7, seed=42)

    assert train_a == train_b
    assert val_a == val_b
    assert len(train_a) == 7
    assert len(val_a) == 3
    assert set(train_a).isdisjoint(val_a)
    assert set(train_a) | set(val_a) == set(videos)


def test_sequence_id_extraction_and_pairing(tmp_path):
    assert extract_sequence_id(tmp_path / "scene001_rgb.mp4") == "scene001"
    assert extract_sequence_id(tmp_path / "scene001_thermal.avi") == "scene001"
    assert extract_sequence_id(tmp_path / "scene001.mp4") == "scene001"

    rgb = [tmp_path / "b_rgb.mp4", tmp_path / "a_rgb.mp4"]
    ir = [tmp_path / "a_ir.mp4", tmp_path / "b_ir.mp4"]
    pairs = pair_videos_by_sequence_id(rgb, ir)
    assert [pair[0].name for pair in pairs] == ["a_rgb.mp4", "b_rgb.mp4"]
    assert [pair[1].name for pair in pairs] == ["a_ir.mp4", "b_ir.mp4"]


def test_pairing_rejects_duplicate_sequence_ids(tmp_path):
    rgb = [tmp_path / "a_rgb.mp4", tmp_path / "sub" / "a_rgb.mp4"]
    ir = [tmp_path / "a_ir.mp4"]
    try:
        pair_videos_by_sequence_id(rgb, ir)
    except ValueError as exc:
        assert "Duplicate RGB sequence ID" in str(exc)
    else:
        raise AssertionError("Duplicate sequence IDs should raise")
