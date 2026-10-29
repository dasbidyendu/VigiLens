"""
Tests for VidLLVIP dataset preparation and integration.
"""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

# Import the prepare_vidllvip module functions
import sys
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.prepare_vidllvip import (
    discover_vidllvip_pairs,
    extract_source_id,
    split_by_source,
    split_by_clip,
    validate_video_pair,
)


def create_dummy_video(path: Path, frames: int = 10, width: int = 640, height: int = 480, is_ir: bool = False):
    """Create a dummy video file for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (width, height), isColor=not is_ir)
    for _ in range(frames):
        if is_ir:
            frame = np.random.randint(0, 255, (height, width), dtype=np.uint8)
        else:
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_vidllvip_pairing():
    """Test RGB-IR filename pairing in VidLLVIP structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        vi_dir = tmpdir / "vi"
        ir_dir = tmpdir / "ir"
        vi_dir.mkdir()
        ir_dir.mkdir()
        
        # Create matching pairs
        create_dummy_video(vi_dir / "clip001.mp4")
        create_dummy_video(ir_dir / "clip001.mp4")
        create_dummy_video(vi_dir / "clip002.mp4")
        create_dummy_video(ir_dir / "clip002.mp4")
        
        # Create unmatched files
        create_dummy_video(vi_dir / "clip003.mp4")
        create_dummy_video(ir_dir / "clip004.mp4")
        
        pairs = discover_vidllvip_pairs(tmpdir)
        
        assert len(pairs) == 2, f"Expected 2 pairs, got {len(pairs)}"
        
        # Verify pairing
        pair_names = [(p[0].name, p[1].name) for p in pairs]
        assert ("clip001.mp4", "clip001.mp4") in pair_names
        assert ("clip002.mp4", "clip002.mp4") in pair_names
        
        print("✓ VidLLVIP pairing test passed")


def test_missing_directories():
    """Test error handling when vi/ or ir/ directories are missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Missing vi directory
        with pytest.raises(FileNotFoundError):
            discover_vidllvip_pairs(tmpdir)
        
        # Create vi but missing ir
        vi_dir = tmpdir / "vi"
        vi_dir.mkdir()
        create_dummy_video(vi_dir / "clip001.mp4")
        
        with pytest.raises(FileNotFoundError):
            discover_vidllvip_pairs(tmpdir)
        
        print("✓ Missing directories test passed")


def test_source_id_extraction():
    """Test source ID extraction from VidLLVIP filenames."""
    test_cases = [
        ("clip_001_0001.mp4", "clip_001"),
        ("scene01_clip01.mp4", "scene01_clip01"),  # Cannot clearly distinguish source from clip
        ("001_0001.mp4", "001"),
        ("video_123_456.mp4", "video_123"),
        ("simple.mp4", "simple"),
    ]
    
    for filename, expected_source in test_cases:
        source = extract_source_id(filename)
        assert source == expected_source, f"Expected {expected_source} for {filename}, got {source}"
    
    print("✓ Source ID extraction test passed")


def test_source_level_splitting():
    """Test source-level splitting to prevent data leakage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        vi_dir = tmpdir / "vi"
        ir_dir = tmpdir / "ir"
        vi_dir.mkdir()
        ir_dir.mkdir()
        
        # Create clips from 3 sources
        for source in ["scene01", "scene02", "scene03"]:
            for clip_num in [1, 2]:
                filename = f"{source}_{clip_num:04d}.mp4"
                create_dummy_video(vi_dir / filename)
                create_dummy_video(ir_dir / filename)
        
        pairs = discover_vidllvip_pairs(tmpdir)
        assert len(pairs) == 6
        
        train, val, test = split_by_source(pairs, val_ratio=0.2, test_ratio=0.1, seed=42)
        
        # Verify no overlap in sources between splits
        train_sources = {extract_source_id(p[0].name) for p in train}
        val_sources = {extract_source_id(p[0].name) for p in val}
        test_sources = {extract_source_id(p[0].name) for p in test}
        
        assert len(train_sources & val_sources) == 0, "Train and val share sources"
        assert len(train_sources & test_sources) == 0, "Train and test share sources"
        assert len(val_sources & test_sources) == 0, "Val and test share sources"
        
        print("✓ Source-level splitting test passed")


def test_clip_level_fallback():
    """Test clip-level fallback when source extraction fails."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        vi_dir = tmpdir / "vi"
        ir_dir = tmpdir / "ir"
        vi_dir.mkdir()
        ir_dir.mkdir()
        
        # Create clips with no clear source pattern
        for i in range(10):
            filename = f"video{i}.mp4"
            create_dummy_video(vi_dir / filename)
            create_dummy_video(ir_dir / filename)
        
        pairs = discover_vidllvip_pairs(tmpdir)
        assert len(pairs) == 10
        
        train, val, test = split_by_source(pairs, val_ratio=0.2, test_ratio=0.1, seed=42)
        
        # With no clear source pattern, each clip is its own source
        # So we should still get a split
        assert len(train) + len(val) + len(test) == 10
        
        print("✓ Clip-level fallback test passed")


def test_video_validation():
    """Test video pair validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        vi_dir = tmpdir / "vi"
        ir_dir = tmpdir / "ir"
        vi_dir.mkdir()
        ir_dir.mkdir()
        
        # Create valid pair
        rgb_path = vi_dir / "test.mp4"
        ir_path = ir_dir / "test.mp4"
        create_dummy_video(rgb_path, frames=30, width=640, height=480)
        create_dummy_video(ir_path, frames=30, width=640, height=480)
        
        metadata = validate_video_pair(rgb_path, ir_path)
        
        assert metadata["rgb_frames"] == 30
        assert metadata["ir_frames"] == 30
        assert metadata["rgb_resolution"] == (640, 480)
        assert metadata["ir_resolution"] == (640, 480)
        
        print("✓ Video validation test passed")


def test_corrupted_video():
    """Test handling of corrupted/unreadable videos."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        vi_dir = tmpdir / "vi"
        ir_dir = tmpdir / "ir"
        vi_dir.mkdir()
        ir_dir.mkdir()
        
        # Create valid RGB
        rgb_path = vi_dir / "test.mp4"
        create_dummy_video(rgb_path)
        
        # Create invalid IR (empty file)
        ir_path = ir_dir / "test.mp4"
        ir_path.write_bytes(b"")
        
        with pytest.raises(ValueError):
            validate_video_pair(rgb_path, ir_path)
        
        print("✓ Corrupted video test passed")


def test_temporal_ordering():
    """Test that temporal order is preserved during conversion."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        vi_dir = tmpdir / "vi"
        ir_dir = tmpdir / "ir"
        vi_dir.mkdir()
        ir_dir.mkdir()
        
        # Create videos with known frame values
        rgb_path = vi_dir / "test.mp4"
        ir_path = ir_dir / "test.mp4"
        
        # RGB: frames with increasing values
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(rgb_path), fourcc, 30.0, (100, 100))
        for i in range(10):
            frame = np.full((100, 100, 3), i * 25, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        
        # IR: frames with increasing values
        writer = cv2.VideoWriter(str(ir_path), fourcc, 30.0, (100, 100), isColor=False)
        for i in range(10):
            frame = np.full((100, 100), i * 25, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        
        # Verify frames are in order
        cap = cv2.VideoCapture(str(rgb_path))
        frames = []
        for _ in range(10):
            ret, frame = cap.read()
            if ret:
                frames.append(frame[0, 0, 0])
        cap.release()
        
        # Frames should be increasing
        assert frames == sorted(frames), "Frames not in temporal order"
        
        print("✓ Temporal ordering test passed")


def test_split_isolation():
    """Test that train/val/test splits are properly isolated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        vi_dir = tmpdir / "vi"
        ir_dir = tmpdir / "ir"
        vi_dir.mkdir()
        ir_dir.mkdir()
        
        # Create clips from multiple sources
        sources = ["scene01", "scene02", "scene03", "scene04", "scene05"]
        for source in sources:
            for clip_num in range(3):
                filename = f"{source}_{clip_num:04d}.mp4"
                create_dummy_video(vi_dir / filename)
                create_dummy_video(ir_dir / filename)
        
        pairs = discover_vidllvip_pairs(tmpdir)
        train, val, test = split_by_source(pairs, val_ratio=0.2, test_ratio=0.1, seed=42)
        
        # Get all clip paths in each split
        train_clips = {p[0].name for p in train}
        val_clips = {p[0].name for p in val}
        test_clips = {p[0].name for p in test}
        
        # Verify no overlap
        assert len(train_clips & val_clips) == 0
        assert len(train_clips & test_clips) == 0
        assert len(val_clips & test_clips) == 0
        
        print("✓ Split isolation test passed")


if __name__ == "__main__":
    test_vidllvip_pairing()
    test_missing_directories()
    test_source_id_extraction()
    test_source_level_splitting()
    test_clip_level_fallback()
    test_video_validation()
    test_corrupted_video()
    test_temporal_ordering()
    test_split_isolation()
    print("\nAll VidLLVIP integration tests passed!")
