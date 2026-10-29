"""Tests for video/directory frame sources and synchronization."""

import cv2
import numpy as np

from pipeline.frame_source import (
    DirectoryFrameSource,
    SingleVideoSource,
    VideoFrameSource,
    create_frame_source,
)
from pipeline.synchronization import FrameSynchronizer, SynchronizedFrameSource


def _read_all(source):
    frames = []
    while True:
        visible, ir = source.read()
        if visible is None and ir is None:
            break
        frames.append((visible, ir))
    return frames


def test_video_frame_source_reads_all_visible_frames_and_resets(make_video, tmp_path):
    rgb_path = make_video(tmp_path / "rgb.avi", num_frames=6, size=(64, 48))
    ir_path = make_video(tmp_path / "ir.avi", num_frames=6, size=(64, 48), grayscale=True)
    source = VideoFrameSource(rgb_path, ir_path)
    try:
        frames = _read_all(source)
        assert len(frames) == 6
        assert all(visible is not None and ir is not None for visible, ir in frames)
        assert frames[0][0].shape == (48, 64, 3)
        assert frames[0][1].ndim in (2, 3)
        assert frames[0][1].shape[0] == 48

        source.reset()
        first_again = source.read()
        assert first_again[0] is not None
        assert first_again[1] is not None
    finally:
        source.release()


def test_single_video_source_has_no_ir(make_video, tmp_path):
    rgb_path = make_video(tmp_path / "rgb.avi", num_frames=3)
    source = create_frame_source(rgb_path)
    try:
        assert isinstance(source, SingleVideoSource)
        frames = _read_all(source)
        assert len(frames) == 3
        assert all(ir is None for _, ir in frames)
        source.reset()
        assert source.read()[0] is not None
    finally:
        source.release()


def test_directory_frame_source_natural_sorting_and_reset(tmp_path):
    visible_dir = tmp_path / "visible"
    ir_dir = tmp_path / "ir"
    visible_dir.mkdir()
    ir_dir.mkdir()

    for name in ["frame10.png", "frame2.png", "frame1.png"]:
        cv2.imwrite(str(visible_dir / name), np.full((16, 16, 3), 40, dtype=np.uint8))
        cv2.imwrite(str(ir_dir / name), np.full((16, 16), 80, dtype=np.uint8))

    source = DirectoryFrameSource(visible_dir, ir_dir)
    assert [path.name for path in source.visible_frames] == ["frame1.png", "frame2.png", "frame10.png"]
    assert source.read()[0] is not None
    source.reset()
    assert source.current_index == 0
    frames = _read_all(source)
    assert len(frames) == 3
    assert all(ir is not None for _, ir in frames)


def test_synchronized_frame_source_wraps_pairs(make_video, tmp_path):
    rgb_path = make_video(tmp_path / "rgb.avi", num_frames=4)
    ir_path = make_video(tmp_path / "ir.avi", num_frames=4, grayscale=True)
    base = VideoFrameSource(rgb_path, ir_path)
    source = SynchronizedFrameSource(base)
    try:
        first = source.read()
        second = source.read()
        assert first is not None and second is not None
        assert first.frame_index == 0
        assert second.frame_index == 1
        assert first.visible_frame is not None
        assert first.ir_frame is not None
        source.reset()
        assert source.read().frame_index == 0
    finally:
        source.release()


def test_frame_synchronizer_maps_fps_and_never_drops_visible():
    synchronizer = FrameSynchronizer(visible_fps=30.0, ir_fps=10.0)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    ir = np.zeros((4, 4), dtype=np.uint8)

    for index in range(6):
        paired = synchronizer.sync_frames(frame, ir)
        assert paired is not None
        assert paired.frame_index == index
        assert paired.visible_frame is frame
    assert synchronizer.expected_ir_index(3) == 1
    synchronizer.reset()
    assert synchronizer.visible_frame_index == 0
