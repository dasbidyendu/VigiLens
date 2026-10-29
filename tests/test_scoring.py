"""Tests for the shared scoring path."""

import numpy as np
import torch

from training.scoring import (
    causal_moving_average,
    fuse_features,
    fuse_sequence,
    prediction_errors,
    score_cache_file,
    score_fused_sequence,
)


def test_causal_moving_average_is_causal():
    scores = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    np.testing.assert_allclose(causal_moving_average(scores, 2), [1.0, 1.5, 2.5, 3.5])
    np.testing.assert_allclose(causal_moving_average(scores, 3), [1.0, 1.5, 2.0, 3.0])
    np.testing.assert_allclose(causal_moving_average(scores, 1), scores)


def test_fuse_sequence_matches_per_frame_fusion(tiny_model):
    torch.manual_seed(0)
    rgb = torch.randn(2, 4, 2048)
    ir = torch.randn(2, 4, 2048)

    sequence = fuse_sequence(tiny_model, rgb, ir, True, True)
    frame = fuse_features(tiny_model, rgb[0, 0].unsqueeze(0), ir[0, 0].unsqueeze(0), True, True)

    assert sequence.shape == (2, 4, 512)
    assert torch.allclose(sequence[0, 0], frame[0], atol=1e-5)


def test_score_cache_file_starts_at_zero_and_smooths_causally(tiny_model, make_cache, tmp_path):
    path = make_cache(tmp_path / "score.pt", num_frames=8, has_ir=True, labels=[0] * 8)
    result = score_cache_file(tiny_model, path, smoothing_window=3)

    raw = result["raw_scores"]
    smoothed = result["smoothed_scores"]
    assert raw.shape == (8,)
    assert raw[0] == 0.0
    assert np.isfinite(raw).all()
    np.testing.assert_allclose(smoothed, causal_moving_average(raw, 3), rtol=1e-6)
    assert result["labels"].tolist() == [0] * 8
    assert result["num_frames"] == 8
    assert result["has_ir"] is True


def test_score_fused_sequence_chunking_matches_single_pass(tiny_model):
    torch.manual_seed(0)
    fused = torch.randn(10, 512)
    raw_single, smooth_single = score_fused_sequence(tiny_model, fused, smoothing_window=1, chunk_size=64)
    raw_chunked, smooth_chunked = score_fused_sequence(tiny_model, fused, smoothing_window=1, chunk_size=3)

    np.testing.assert_allclose(raw_single, raw_chunked, atol=1e-5)
    np.testing.assert_allclose(smooth_single, smooth_chunked, atol=1e-5)


def test_prediction_errors_exclude_first_frame(tiny_model, make_cache, tmp_path):
    first = make_cache(tmp_path / "a.pt", num_frames=10, has_ir=True)
    second = make_cache(tmp_path / "b.pt", num_frames=6, has_ir=False, seed=1)
    errors = prediction_errors(tiny_model, [first, second])
    assert errors.shape == (9 + 5,)
    assert np.isfinite(errors).all()
