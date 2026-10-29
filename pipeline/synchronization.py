"""
Frame synchronization helpers for visible and IR/thermal inputs.

OpenCV does not expose capture timestamps in a portable way, so the default
strategy is frame-index/FPS mapping: the IR frame nearest in time to each
visible frame is selected.  The visible stream is never dropped.

``VideoFrameSource`` already performs this mapping while reading.  The classes
here are useful when a caller has its own two streams and wants a common
``PairedFrame`` interface, or when timestamps are available from an external
source.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PairedFrame:
    """A synchronized visible/IR frame pair."""

    visible_frame: np.ndarray
    ir_frame: Optional[np.ndarray]
    frame_index: int
    visible_timestamp: Optional[float] = None
    ir_timestamp: Optional[float] = None


class FrameSynchronizer:
    """
    Map visible frame indices to the nearest IR frame index.

    The synchronizer is intentionally non-dropping: a mismatch is resolved by
    choosing the nearest IR frame, not by discarding the visible frame.  This
    matters for streaming inference, where every visible frame must produce a
    score.
    """

    def __init__(self,
                 visible_fps: float,
                 ir_fps: Optional[float] = None,
                 sync_tolerance: float = 0.1):
        self.visible_fps = float(visible_fps) if visible_fps and visible_fps > 0 else 30.0
        self.ir_fps = float(ir_fps) if ir_fps and ir_fps > 0 else None
        self.sync_tolerance = float(sync_tolerance)
        self.visible_frame_index = 0
        self.ir_frame_index = 0
        self.ir_available = self.ir_fps is not None

        if self.ir_available:
            self.fps_ratio = self.visible_fps / self.ir_fps
            logger.info(
                "Frame synchronizer: visible_fps=%.2f, ir_fps=%.2f, ratio=%.3f",
                self.visible_fps, self.ir_fps, self.fps_ratio,
            )
        else:
            self.fps_ratio = None
            logger.info("Frame synchronizer: IR unavailable - visible only mode")

    def expected_ir_index(self, visible_index: Optional[int] = None) -> Optional[int]:
        """Return the IR frame index nearest to a visible frame index."""
        if not self.ir_available:
            return None
        if visible_index is None:
            visible_index = self.visible_frame_index
        return int(round(visible_index * self.ir_fps / self.visible_fps))

    def sync_frames(self,
                    visible_frame: np.ndarray,
                    ir_frame: Optional[np.ndarray],
                    visible_timestamp: Optional[float] = None,
                    ir_timestamp: Optional[float] = None) -> Optional[PairedFrame]:
        """Pair the supplied frames without dropping the visible frame."""
        if visible_frame is None:
            return None

        if visible_timestamp is not None and ir_timestamp is not None:
            difference = abs(visible_timestamp - ir_timestamp)
            if difference > self.sync_tolerance:
                logger.debug(
                    "Timestamp difference %.3fs exceeds tolerance %.3fs; pairing nearest frames",
                    difference, self.sync_tolerance,
                )

        paired = PairedFrame(
            visible_frame=visible_frame,
            ir_frame=ir_frame if self.ir_available else None,
            frame_index=self.visible_frame_index,
            visible_timestamp=visible_timestamp,
            ir_timestamp=ir_timestamp,
        )
        self.visible_frame_index += 1
        if self.ir_available and ir_frame is not None:
            self.ir_frame_index += 1
        return paired

    def reset(self) -> None:
        self.visible_frame_index = 0
        self.ir_frame_index = 0
        logger.info("Frame synchronizer reset")


class SynchronizedFrameSource:
    """
    Wrap a frame source and expose ``PairedFrame`` objects.

    ``VideoFrameSource``/``DirectoryFrameSource`` already return synchronized
    tuples; this wrapper only adds the ``PairedFrame`` container and bookkeeping.
    """

    def __init__(self, frame_source, sync_tolerance: float = 0.1):
        self.frame_source = frame_source
        self.visible_fps = getattr(frame_source, "visible_fps", 30.0)
        self.ir_fps = getattr(frame_source, "ir_fps", None)
        self.synchronizer = FrameSynchronizer(
            visible_fps=self.visible_fps,
            ir_fps=self.ir_fps,
            sync_tolerance=sync_tolerance,
        )
        logger.info("Synchronized frame source initialized")

    def read(self) -> Optional[PairedFrame]:
        pair = self.frame_source.read()
        if pair is None:
            return None
        visible_frame, ir_frame = pair
        if visible_frame is None and ir_frame is None:
            return None
        return self.synchronizer.sync_frames(visible_frame, ir_frame)

    def reset(self) -> None:
        if hasattr(self.frame_source, "reset"):
            self.frame_source.reset()
        self.synchronizer.reset()

    def release(self) -> None:
        self.frame_source.release()
        self.synchronizer.reset()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
