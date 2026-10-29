"""Tests for threshold calibration."""

import json
from pathlib import Path

import numpy as np

from training.calibration import AnomalyCalibrator, calibrate_model


def test_compute_threshold_statistics(tiny_model):
    calibrator = AnomalyCalibrator(tiny_model, k_std=1.0, percentile=75.0)
    info = calibrator.compute_threshold([1.0, 2.0, 3.0, 4.0])

    assert info["mean_error"] == 2.5
    assert info["std_error"] == np.std([1.0, 2.0, 3.0, 4.0])
    assert info["threshold_std"] == 2.5 + np.std([1.0, 2.0, 3.0, 4.0])
    assert info["threshold"] == info["threshold_std"]
    assert info["threshold_percentile"] == 3.25
    assert info["num_samples"] == 4
    assert info["method"] == "std"


def test_compute_threshold_rejects_empty_or_non_finite(tiny_model):
    calibrator = AnomalyCalibrator(tiny_model)
    for values in ([], [float("nan"), float("inf")]):
        try:
            calibrator.compute_threshold(values)
        except ValueError as exc:
            assert "no finite" in str(exc)
        else:
            raise AssertionError("Invalid calibration data should raise")


def test_save_and_load_calibration(tmp_path):
    path = tmp_path / "calibration.json"
    info = {"threshold": 0.25, "threshold_std": 0.25, "k_std": 3.0}
    AnomalyCalibrator.save_calibration(info, path)
    loaded = AnomalyCalibrator.load_calibration(path)
    assert loaded == info
    with open(path, "r", encoding="utf-8") as handle:
        assert json.load(handle)["threshold"] == 0.25


def test_collect_errors_from_caches_excludes_frame_zero(tiny_model, make_cache, tmp_path):
    first = make_cache(tmp_path / "a.pt", num_frames=10, has_ir=True)
    second = make_cache(tmp_path / "b.pt", num_frames=6, has_ir=False, seed=1)

    calibrator = AnomalyCalibrator(tiny_model)
    errors = calibrator.collect_errors_from_caches([first, second])
    assert len(errors) == 9 + 5
    assert np.isfinite(errors).all()


def test_collect_errors_rejects_raw_videos(tiny_model, tmp_path):
    calibrator = AnomalyCalibrator(tiny_model)
    try:
        calibrator.collect_errors([tmp_path / "video.mp4"])
    except ValueError as exc:
        assert "precomputed" in str(exc).lower()
    else:
        raise AssertionError("Raw videos should be rejected for calibration")


def test_calibrate_model_writes_json(tiny_model, make_cache, tmp_path):
    cache = make_cache(tmp_path / "cache" / "val.pt", num_frames=8, has_ir=True)
    checkpoint_dir = tmp_path / "checkpoints"

    info = calibrate_model(
        tiny_model,
        [cache],
        device="cpu",
        checkpoint_dir=str(checkpoint_dir),
        k_std=2.0,
        percentile=90.0,
    )

    output = checkpoint_dir / "calibration.json"
    assert output.exists()
    saved = AnomalyCalibrator.load_calibration(output)
    assert saved == info
    assert saved["num_samples"] == 7
    assert saved["k_std"] == 2.0
