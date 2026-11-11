"""Webcam frame source and the app's webcam mode (no real camera needed)."""

from pathlib import Path

import cv2
import numpy as np
import pytest


class FakeCapture:
    def __init__(self, index, backend=None, frames=5, opened=True):
        self.index = index
        self.remaining = frames
        self.opened = opened
        self.released = False
        self.props = {cv2.CAP_PROP_FPS: 30.0, cv2.CAP_PROP_FRAME_WIDTH: 64,
                      cv2.CAP_PROP_FRAME_HEIGHT: 48}

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def get(self, prop):
        return self.props.get(prop, 0)

    def read(self):
        if self.remaining <= 0:
            return False, None
        self.remaining -= 1
        return True, np.zeros((48, 64, 3), np.uint8)

    def release(self):
        self.released = True


def test_camera_source_reads_until_budget(monkeypatch):
    from pipeline import frame_source as fs

    captures = []
    monkeypatch.setattr(fs.cv2, "VideoCapture",
                        lambda i, b=None: captures.append(FakeCapture(i, b, frames=10)) or captures[-1])
    source = fs.CameraFrameSource(0, max_frames=3)
    frames = []
    while True:
        rgb, ir = source.read()
        if rgb is None:
            break
        assert ir is None
        frames.append(rgb)
    assert len(frames) == 3
    assert source.ir_available is False and source.fps == 30.0
    source.release()
    assert captures[-1].released


def test_camera_source_stops_when_camera_does(monkeypatch):
    from pipeline import frame_source as fs

    monkeypatch.setattr(fs.cv2, "VideoCapture", lambda i, b=None: FakeCapture(i, b, frames=2))
    source = fs.CameraFrameSource(1)
    assert source.read()[0] is not None
    assert source.read()[0] is not None
    assert source.read() == (None, None)


def test_camera_source_raises_when_no_camera(monkeypatch):
    from pipeline import frame_source as fs

    monkeypatch.setattr(fs.cv2, "VideoCapture", lambda i, b=None: FakeCapture(i, b, opened=False))
    with pytest.raises(RuntimeError, match="camera 3"):
        fs.CameraFrameSource(3)


def test_camera_source_unknown_fps_falls_back(monkeypatch):
    from pipeline import frame_source as fs

    def capture(i, b=None):
        cap = FakeCapture(i, b)
        cap.props[cv2.CAP_PROP_FPS] = -1.0
        return cap

    monkeypatch.setattr(fs.cv2, "VideoCapture", capture)
    assert fs.CameraFrameSource(0).fps == 30.0


def test_app_webcam_mode_renders_without_starting():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=120).run()
    assert not app.exception
    app.radio[0].set_value("Webcam").run()
    assert not app.exception
    labels = [button.label for button in app.button]
    assert "Start webcam" in labels and "Stop" in labels
    assert any("Start webcam" in info.value for info in app.info)
