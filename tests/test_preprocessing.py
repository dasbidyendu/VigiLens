"""Tests for shared visible/IR preprocessing."""

import numpy as np

from pipeline.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    FramePreprocessor,
    align_frames,
    validate_ir_frame,
)


def test_visible_preprocessing_converts_bgr_to_rgb():
    preprocessor = FramePreprocessor(target_size=(4, 4), normalize=True)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[:, :] = (10, 20, 30)  # BGR

    output = preprocessor.preprocess_visible(frame)
    assert output.shape == (4, 4, 3)
    expected = (np.array([30, 20, 10], dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    np.testing.assert_allclose(output[0, 0], expected, rtol=1e-6)


def test_ir_preprocessing_replicates_grayscale_to_three_channels():
    preprocessor = FramePreprocessor(target_size=(4, 4), normalize=True)
    frame = np.full((8, 8), 128, dtype=np.uint8)

    output = preprocessor.preprocess_ir(frame)
    assert output.shape == (4, 4, 3)
    pixel = 128.0 / 255.0
    expected = [
        (pixel - float(IMAGENET_MEAN[c])) / float(IMAGENET_STD[c]) for c in range(3)
    ]
    for c in range(3):
        np.testing.assert_allclose(output[:, :, c], np.full_like(output[:, :, c], expected[c]), rtol=1e-5)

def test_mask_zeroes_background_before_normalization():
    preprocessor = FramePreprocessor(target_size=(4, 4), normalize=True)
    frame = np.full((8, 8, 3), 255, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1

    output = preprocessor.preprocess_visible(frame, mask)
    foreground = (1.0 - IMAGENET_MEAN) / IMAGENET_STD
    background = (0.0 - IMAGENET_MEAN) / IMAGENET_STD
    np.testing.assert_allclose(output[2, 2], foreground, rtol=1e-6)
    np.testing.assert_allclose(output[0, 0], background, rtol=1e-6)


def test_preprocess_pair_handles_missing_ir():
    preprocessor = FramePreprocessor(target_size=(4, 4))
    visible = np.zeros((8, 8, 3), dtype=np.uint8)
    processed_visible, processed_ir = preprocessor.preprocess_pair(visible, None)
    assert processed_visible.shape == (4, 4, 3)
    assert processed_ir is None


def test_ir_validation_and_alignment():
    assert validate_ir_frame(np.zeros((4, 4), dtype=np.uint8)) is True
    assert validate_ir_frame(np.zeros((4, 4, 3), dtype=np.uint8)) is False

    visible = np.zeros((10, 20, 3), dtype=np.uint8)
    ir = np.zeros((5, 8), dtype=np.uint8)
    aligned_visible, aligned_ir = align_frames(visible, ir, target_size=(4, 6))
    assert aligned_visible.shape == (4, 6, 3)
    assert aligned_ir.shape == (4, 6)
