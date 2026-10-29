"""Tests for cached training, validation, and checkpointing."""

import json

import torch

from models.anomaly_model import VigiLensModel
from training.dataset import CachedFeatureDataset
from training.train import (
    build_cached_datasets,
    discover_caches,
    train_cached,
)
from training.validate import validate_cached


def _dataset(paths, sequence_length=4):
    return CachedFeatureDataset(
        paths,
        sequence_length=sequence_length,
        sequence_stride=1,
        rgb_dropout_prob=0.0,
        ir_dropout_prob=0.0,
    )


def test_validate_cached_returns_real_errors(tiny_model, make_cache, tmp_path):
    first = make_cache(tmp_path / "a.pt", num_frames=12, has_ir=True)
    second = make_cache(tmp_path / "b.pt", num_frames=8, has_ir=False, seed=1)
    result = validate_cached(
        tiny_model,
        [first, second],
        device="cpu",
        sequence_length=4,
        batch_size=2,
        amp=False,
    )

    # 12 frames -> 9 sequences, 8 frames -> 5 sequences, 3 errors each.
    assert result["num_frames"] == (9 + 5) * 3
    assert result["num_batches"] == 7
    assert result["errors"].shape == (42,)
    assert torch.isfinite(torch.tensor(result["val_loss"]))


def test_build_cached_datasets_and_discover(tiny_model, make_cache, tmp_path):
    cache_dir = tmp_path / "cache"
    make_cache(cache_dir / "train" / "a.pt", num_frames=10, has_ir=True)
    make_cache(cache_dir / "val" / "b.pt", num_frames=8, has_ir=True, seed=1)

    train_dataset, val_dataset = build_cached_datasets(
        cache_dir, sequence_length=4, sequence_stride=1,
        rgb_dropout_prob=0.0, ir_dropout_prob=0.0,
    )
    assert len(train_dataset) == 7
    assert len(val_dataset) == 5
    assert len(discover_caches(cache_dir)) == 2


def test_train_cached_one_epoch_writes_checkpoints(tiny_model, make_cache, tmp_path):
    train_cache = make_cache(tmp_path / "cache" / "train" / "a.pt", num_frames=12, has_ir=True)
    val_cache = make_cache(tmp_path / "cache" / "val" / "b.pt", num_frames=10, has_ir=True, seed=1)
    train_dataset = _dataset([train_cache], sequence_length=4)
    val_dataset = _dataset([val_cache], sequence_length=4)
    checkpoint_dir = tmp_path / "checkpoints"

    result = train_cached(
        tiny_model,
        train_dataset,
        val_dataset,
        device="cpu",
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        amp=False,
        checkpoint_dir=checkpoint_dir,
        config_dict={"test": True},
    )

    assert result["history"][0]["epoch"] == 1
    assert (checkpoint_dir / "best_model.pth").exists()
    assert (checkpoint_dir / "last_model.pth").exists()
    assert (checkpoint_dir / "training_history.json").exists()
    with open(checkpoint_dir / "training_history.json", "r", encoding="utf-8") as handle:
        history = json.load(handle)
    assert len(history) == 1

    fresh = VigiLensModel(device="cpu", load_extractors=False)
    metadata = fresh.load_checkpoint(checkpoint_dir / "best_model.pth")
    assert metadata["epoch"] == 1
    assert metadata["config"] == {"test": True}
    for key, value in tiny_model.state_dict().items():
        assert torch.allclose(fresh.state_dict()[key], value)


def test_checkpoint_round_trip_without_extractors(tmp_path):
    model = VigiLensModel(device="cpu", load_extractors=False)
    assert model.rgb_extractor is None
    assert model.ir_extractor is None

    path = tmp_path / "checkpoint.pth"
    model.save_checkpoint(str(path), epoch=3, loss=0.25, config={"device": "cpu"})
    loaded = VigiLensModel(device="cpu", load_extractors=False)
    metadata = loaded.load_checkpoint(str(path))

    assert metadata["epoch"] == 3
    assert metadata["loss"] == 0.25
    for key, value in model.state_dict().items():
        assert torch.allclose(loaded.state_dict()[key], value)

    try:
        model.save_checkpoint(str(path), 1, 0.1, {}, include_encoders=True)
    except RuntimeError as exc:
        assert "load_extractors=True" in str(exc)
    else:
        raise AssertionError("include_encoders should require extractors")
