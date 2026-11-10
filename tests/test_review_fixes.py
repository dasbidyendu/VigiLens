"""Regression tests for the issues found in the code review."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from models.anomaly_model import VigiLensModel
from models.mask_stabilizer import MaskStabilizer
from training.calibration import AnomalyCalibrator, calibrate_from_caches, threshold_for
from training.dataset import CachedFeatureDataset, VideoBlockSampler
from training.scoring import fuse_features, score_fused_sequence, smooth_scores
from training.train import _param_groups, build_cached_datasets


# ---------------------------------------------------------------------------
# Scoring: offline == streaming, and matches the training window
# ---------------------------------------------------------------------------

def _feature_stream(num_frames, seed=0):
    generator = torch.Generator().manual_seed(seed)
    rgb = torch.randn(num_frames, 2048, generator=generator)
    ir = torch.randn(num_frames, 2048, generator=generator)
    return rgb, ir


@pytest.mark.parametrize("num_frames", [3, 15, 16, 40])
def test_offline_scores_equal_streaming_scores(tiny_model, num_frames):
    rgb, ir = _feature_stream(num_frames)
    with torch.inference_mode():
        fused = fuse_features(tiny_model, rgb, ir, True, True)
        raw_offline, _ = score_fused_sequence(tiny_model, fused, chunk_size=7)

    frames = iter(range(num_frames))

    def fake_extract(rgb_frame, ir_frame=None, rgb_mask=None, ir_mask=None):
        index = next(frames)
        return rgb[index:index + 1], ir[index:index + 1]

    tiny_model.extract_features = fake_extract
    tiny_model.reset_stream()
    streamed = []
    for _ in range(num_frames):
        result = tiny_model.process_frame(np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4), np.uint8))
        streamed.append(0.0 if result["anomaly_score"] is None else result["anomaly_score"])

    np.testing.assert_allclose(raw_offline, np.asarray(streamed, dtype=np.float32), atol=1e-5)


def test_score_only_depends_on_context_window(tiny_model):
    window = tiny_model.context_length
    torch.manual_seed(1)
    fused = torch.randn(window + 10, 512)
    perturbed = fused.clone()
    perturbed[0] += 5.0  # frame 0 is outside the window of every frame > window

    with torch.inference_mode():
        raw, _ = score_fused_sequence(tiny_model, fused)
        raw_perturbed, _ = score_fused_sequence(tiny_model, perturbed)
    np.testing.assert_allclose(raw[window + 1:], raw_perturbed[window + 1:], atol=1e-6)
    assert not np.allclose(raw[1:window + 1], raw_perturbed[1:window + 1])


def test_context_length_follows_checkpoint_sequence_length(tmp_path):
    model = VigiLensModel(device="cpu", load_extractors=False)
    path = tmp_path / "ckpt.pth"
    model.save_checkpoint(str(path), 1, 0.1, {"training": {"sequence_length": 9}})
    loaded = VigiLensModel(device="cpu", load_extractors=False)
    loaded.load_checkpoint(str(path))
    assert loaded.context_length == 8


def test_smoothing_ignores_unscored_first_frame():
    raw = np.array([0.0, 2.0, 4.0, 6.0], dtype=np.float32)
    np.testing.assert_allclose(smooth_scores(raw, "moving_average", 2), [0.0, 2.0, 3.0, 5.0])
    ema = smooth_scores(raw, "exponential", alpha=0.5)
    np.testing.assert_allclose(ema, [0.0, 2.0, 3.0, 4.5])


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def test_strict_load_does_not_silently_accept_mismatched_checkpoint(tmp_path):
    small = VigiLensModel(device="cpu", load_extractors=False, fusion_output_dim=256)
    path = tmp_path / "small.pth"
    small.save_checkpoint(str(path), 1, 0.1, {})
    model = VigiLensModel(device="cpu", load_extractors=False)
    with pytest.raises(RuntimeError):
        model.load_checkpoint(str(path))


class PickledConfig:  # an arbitrary class: not loadable with weights_only=True
    pass


def test_pickled_checkpoint_is_refused_by_default(tmp_path):
    path = tmp_path / "pickled.pth"
    model = VigiLensModel(device="cpu", load_extractors=False)
    torch.save({"model_state_dict": model.state_dict(), "config": PickledConfig()}, path)
    with pytest.raises(RuntimeError, match="weights_only"):
        VigiLensModel(device="cpu", load_extractors=False).load_checkpoint(str(path))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def test_layer_norm_and_bias_get_no_weight_decay(tiny_model):
    groups = _param_groups(tiny_model.parameters(), 1e-2)
    no_decay = next(group for group in groups if group["weight_decay"] == 0.0)
    norm_gain = tiny_model.fusion.fusion_layer[2].weight
    assert any(param is norm_gain for param in no_decay["params"])
    decayed = next(group for group in groups if group["weight_decay"] > 0.0)
    assert all(param.ndim > 1 for param in decayed["params"])


def test_cached_datasets_never_use_test_or_eval_caches(make_cache, tmp_path):
    cache = tmp_path / "cache"
    for name in ("a", "b", "c", "d"):
        make_cache(cache / f"{name}.pt", num_frames=8)
    make_cache(cache / "test" / "anomalous.pt", num_frames=8)
    make_cache(cache / "eval" / "anomalous2.pt", num_frames=8)

    train, val = build_cached_datasets(cache, sequence_length=4, rgb_dropout_prob=0.0,
                                       ir_dropout_prob=0.0, val_fraction=0.25)
    names = {Path(p).name for p in train.cache_paths} | {Path(p).name for p in val.cache_paths}
    assert names == {"a.pt", "b.pt", "c.pt", "d.pt"}
    assert not set(train.cache_paths) & set(val.cache_paths)
    assert len(val.cache_paths) == 1


def test_video_block_sampler_covers_every_window_once(make_cache, tmp_path):
    paths = [make_cache(tmp_path / f"v{i}.pt", num_frames=6 + i, seed=i) for i in range(5)]
    dataset = CachedFeatureDataset(paths, sequence_length=4, max_cached_videos=2,
                                   rgb_dropout_prob=0.0, ir_dropout_prob=0.0)
    sampler = VideoBlockSampler(dataset, seed=3)
    order = list(sampler)
    assert sorted(order) == list(range(len(dataset)))
    # Windows of at most two videos are interleaved at any time.
    videos = [dataset.index[i][0] for i in order]
    first_block = set(videos[:len([v for v in videos if v in set(videos[:1])])])
    assert len(first_block) <= 2
    assert list(sampler) != order  # reshuffled next epoch


def test_resume_restores_scheduler_and_best_loss(tiny_model, make_cache, tmp_path):
    from training.train import train_cached

    train_cache = make_cache(tmp_path / "train" / "a.pt", num_frames=12)
    val_cache = make_cache(tmp_path / "val" / "b.pt", num_frames=10, seed=1)
    dataset = lambda p: CachedFeatureDataset([p], sequence_length=4, rgb_dropout_prob=0.0, ir_dropout_prob=0.0)
    checkpoints = tmp_path / "ckpt"
    first = train_cached(tiny_model, dataset(train_cache), dataset(val_cache), "cpu", epochs=2,
                         batch_size=2, amp=False, checkpoint_dir=checkpoints)
    best = min(entry["val_loss"] for entry in first["history"])

    model = VigiLensModel(device="cpu", load_extractors=False)
    second = train_cached(model, dataset(train_cache), dataset(val_cache), "cpu", epochs=3,
                          batch_size=2, amp=False, checkpoint_dir=checkpoints,
                          resume=checkpoints / "last_model.pth")
    assert [entry["epoch"] for entry in second["history"]] == [1, 2, 3]
    assert second["best_val_loss"] <= best


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def test_calibration_has_threshold_per_smoothing_setting(tiny_model, make_cache, tmp_path):
    caches = [make_cache(tmp_path / f"n{i}.pt", num_frames=20, seed=i) for i in range(2)]
    checkpoint = tmp_path / "best.pth"
    tiny_model.save_checkpoint(str(checkpoint), 1, 0.1, {})
    info = calibrate_from_caches(tiny_model, caches, output_path=tmp_path / "cal.json",
                                 checkpoint_path=checkpoint)

    saved = json.loads((tmp_path / "cal.json").read_text(encoding="utf-8"))
    assert len(saved["checkpoint_sha256"]) == 64
    assert saved["num_samples"] == 2 * 19
    raw, source = threshold_for(saved, "moving_average", 1)
    assert source == "matched" and raw == pytest.approx(info["threshold_raw"])
    smooth10, source10 = threshold_for(saved, "moving_average", 10)
    assert source10 == "matched"
    # Smoothing shrinks the spread of normal scores, so the threshold moves.
    assert smooth10 != pytest.approx(raw)
    assert threshold_for(saved, "exponential", ema_alpha=0.3)[1] == "matched"


def test_threshold_for_legacy_and_missing_calibration():
    assert threshold_for({"threshold": 0.9}, "moving_average", 10) == (0.9, "raw")
    assert threshold_for(None, "moving_average", 10, fallback=0.5) == (0.5, "default")


# ---------------------------------------------------------------------------
# Segmentation / preprocessing
# ---------------------------------------------------------------------------

def test_mask_stabilizer_bridges_short_gaps_only():
    mask = np.ones((2, 2), np.uint8)
    stabilizer = MaskStabilizer(hold_frames=2)
    out = stabilizer.update_many([mask, None, None, None, mask])
    assert out[1] is mask and out[2] is mask
    assert out[3] is None
    assert out[4] is mask


def test_resnet_is_frozen_and_batchnorm_stays_in_eval(monkeypatch):
    import torchvision

    from models import feature_extractor

    # Avoid the ImageNet download: same architecture, random weights.
    original = torchvision.models.resnet50
    monkeypatch.setattr(feature_extractor.models, "resnet50",
                        lambda weights=None: original(weights=None))
    extractor = feature_extractor.ResNet50FeatureExtractor("cpu")
    assert not any(p.requires_grad for p in extractor.parameters())

    extractor.set_layer4_trainable(True)
    trainable = {id(p) for p in extractor.trainable_parameters()}
    layer4 = {id(p) for p in extractor.backbone[7].parameters()}
    assert trainable == layer4

    extractor.train(True)
    batchnorms = [m for m in extractor.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert batchnorms and not any(m.training for m in batchnorms)


def test_preprocessing_module_reuses_extractor_helpers():
    from models import feature_extractor
    from pipeline import preprocessing

    assert preprocessing._as_uint8 is feature_extractor._as_uint8
    frame16 = np.array([[0, 1000], [2000, 4000]], dtype=np.uint16)
    assert preprocessing._as_uint8(frame16).max() == 255


# ---------------------------------------------------------------------------
# Dataset preparation scripts
# ---------------------------------------------------------------------------

def test_otcbvs_classification_uses_whole_tokens():
    from scripts.prepare_otcbvs import _classify, _pair_key

    assert _classify("stairs_visible_01") == "rgb"
    assert _classify("first_thermal_01") == "ir"
    assert _classify("seq1_IR") == "ir"
    assert _classify("night_walk") is None
    assert _pair_key("seq1_day_visible") == _pair_key("seq1_day_thermal")
    assert _pair_key("seq1_day_visible") != _pair_key("seq1_night_thermal")


def test_prepare_split_dirs_refuses_stale_output(tmp_path):
    from training.utils import prepare_split_dirs

    stale = tmp_path / "Data" / "val" / "rgb" / "old_rgb.mp4"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"x")
    keep = tmp_path / "Data" / "notes.txt"
    keep.write_text("keep", encoding="utf-8")

    with pytest.raises(SystemExit):
        prepare_split_dirs(tmp_path / "Data", ("train", "val"), clean=False)
    prepare_split_dirs(tmp_path / "Data", ("train", "val"), clean=True)
    assert not stale.exists()
    assert keep.exists()


def test_vidllvip_source_regex_and_split_method(tmp_path):
    from scripts import prepare_vidllvip as prep

    assert prep.extract_source_id("cam3_take7.mp4", r"^(cam\d+)_") == "cam3"
    pairs = [(Path(f"clip_{i:03d}.mp4"), Path(f"clip_{i:03d}.mp4")) for i in range(10)]
    # Every name maps to source "clip": a leak-free split is impossible.
    with pytest.raises(SystemExit):
        prep.split_by_source(pairs, 0.2, 0.1, 0, allow_clip_split=False)
    prep.split_by_source(pairs, 0.2, 0.1, 0, allow_clip_split=True)
    assert prep.LAST_SPLIT_METHOD == "clip_level"
