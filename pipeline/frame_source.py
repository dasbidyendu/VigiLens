"""
Frame source module for handling visible and IR/thermal video inputs.

This module provides abstractions for reading synchronized visible and IR frames
from various sources including:
- Separate video files (visible.mp4, thermal.mp4)
- Paired frame directories
- Live camera inputs

Convention: every frame returned by these sources is an OpenCV **BGR** array
(or a single-channel IR array).  The model's feature extractors convert BGR to
RGB internally, so callers must not swap channels twice.
"""

import re
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _natural_key(path: Path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(path.name))]


class MultimodalFrameSource:
    """
    Abstract base class for multimodal frame sources.
    
    Handles synchronized reading of visible and IR frames from various sources.
    """
    
    def __init__(self, visible_source: Union[str, Path], ir_source: Optional[Union[str, Path]] = None):
        """
        Initialize the frame source.
        
        Args:
            visible_source: Path to visible video file or directory
            ir_source: Path to IR video file or directory (None if IR unavailable)
        """
        self.visible_source = Path(visible_source)
        self.ir_source = Path(ir_source) if ir_source else None
        self.ir_available = ir_source is not None
        
        if not self.visible_source.exists():
            raise FileNotFoundError(f"Visible source not found: {visible_source}")
        
        if self.ir_source and not self.ir_source.exists():
            logger.warning(f"IR source not found: {ir_source}. IR will be unavailable.")
            self.ir_available = False
    
    def read(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Read a single synchronized frame pair.
        
        Returns:
            Tuple of (visible_frame, ir_frame). Either can be None if unavailable.
        """
        raise NotImplementedError("Subclasses must implement read()")
    
    def release(self):
        """Release resources."""
        raise NotImplementedError("Subclasses must implement release()")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class VideoFrameSource(MultimodalFrameSource):
    """
    Frame source for synchronized video files.
    
    Reads from separate visible and IR video files with frame synchronization.
    """
    
    def __init__(self, visible_video: Union[str, Path], ir_video: Optional[Union[str, Path]] = None):
        """
        Initialize video frame source.

        Args:
            visible_video: Path to visible video file (mp4, avi, mov)
            ir_video: Path to IR/thermal video file (optional)
        """
        super().__init__(visible_video, ir_video)

        # Open visible video
        self.visible_cap = cv2.VideoCapture(str(self.visible_source))
        if not self.visible_cap.isOpened():
            raise RuntimeError(f"Could not open visible video: {visible_video}")

        # Open IR video if available
        self.ir_cap = None
        if self.ir_available:
            self.ir_cap = cv2.VideoCapture(str(self.ir_source))
            if not self.ir_cap.isOpened():
                logger.warning(f"Could not open IR video: {ir_video}. IR will be unavailable.")
                self.ir_available = False
        
        # Get video properties
        self.visible_fps = float(self.visible_cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.visible_frame_count = int(self.visible_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.visible_width = int(self.visible_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.visible_height = int(self.visible_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.visible_index = 0
        self.ir_index = 0
        self.last_ir_frame = None

        if self.ir_available:
            self.ir_fps = float(self.ir_cap.get(cv2.CAP_PROP_FPS)) or self.visible_fps
            self.ir_frame_count = int(self.ir_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.ir_width = int(self.ir_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.ir_height = int(self.ir_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(f"Visible: {self.visible_width}x{self.visible_height} @ {self.visible_fps:.2f} FPS")
            logger.info(f"IR: {self.ir_width}x{self.ir_height} @ {self.ir_fps:.2f} FPS")
        else:
            self.ir_fps = None
            logger.info(f"Visible: {self.visible_width}x{self.visible_height} @ {self.visible_fps:.2f} FPS")
            logger.warning("IR input unavailable")

    def read(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Read the next visible frame and its temporally nearest IR frame.

        The visible stream is never dropped.  If the IR stream has a higher
        frame rate, extra IR frames are skipped; if it has a lower frame rate,
        the most recent IR frame is reused.  This keeps every visible frame in
        the output while respecting the two time bases.
        """
        ret_visible, visible_frame = self.visible_cap.read()
        if not ret_visible:
            return None, None

        ir_frame = None
        if self.ir_available and self.ir_cap is not None:
            target_ir_index = int(round(
                self.visible_index * (self.ir_fps or self.visible_fps) / max(self.visible_fps, 1e-6)
            ))

            while self.ir_index <= target_ir_index:
                ret_ir, candidate = self.ir_cap.read()
                if not ret_ir:
                    logger.warning("IR video ended before visible video")
                    self.ir_available = False
                    self.ir_cap.release()
                    self.ir_cap = None
                    self.last_ir_frame = None
                    break
                self.last_ir_frame = candidate
                self.ir_index += 1

            if self.ir_available:
                ir_frame = self.last_ir_frame

        self.visible_index += 1
        return visible_frame, ir_frame

    def reset(self):
        """Rewind both videos to the first frame."""
        self.visible_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.visible_index = 0
        if self.ir_cap is not None:
            self.ir_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.ir_index = 0
        self.last_ir_frame = None

    def release(self):
        """Release video capture resources."""
        if self.visible_cap is not None:
            self.visible_cap.release()
        if self.ir_cap is not None:
            self.ir_cap.release()


class DirectoryFrameSource(MultimodalFrameSource):
    """
    Frame source for paired frame directories.
    
    Reads from directories containing synchronized frame sequences:
    data/
        visible/
            000001.jpg
            000002.jpg
            ...
        ir/
            000001.png
            000002.png
            ...
    """
    
    def __init__(self, visible_dir: Union[str, Path], ir_dir: Optional[Union[str, Path]] = None):
        """
        Initialize directory frame source.
        
        Args:
            visible_dir: Path to directory containing visible frames
            ir_dir: Path to directory containing IR frames (optional)
        """
        super().__init__(visible_dir, ir_dir)
        
        # Get naturally sorted frame lists (frame2 before frame10).
        self.visible_frames = sorted(
            (f for f in self.visible_source.iterdir()
             if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS),
            key=_natural_key,
        )

        if not self.visible_frames:
            raise ValueError(f"No visible frames found in {visible_dir}")

        self.current_index = 0
        self.visible_fps = 30.0
        self.ir_fps = 30.0 if self.ir_available else None

        if self.ir_available:
            self.ir_frames = sorted(
                (f for f in self.ir_source.iterdir()
                 if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS),
                key=_natural_key,
            )

            if not self.ir_frames:
                logger.warning(f"No IR frames found in {ir_dir}. IR will be unavailable.")
                self.ir_available = False
                self.ir_fps = None
            elif len(self.ir_frames) != len(self.visible_frames):
                logger.warning(
                    "Frame count mismatch: %d visible vs %d IR; using %d pairs",
                    len(self.visible_frames), len(self.ir_frames),
                    min(len(self.visible_frames), len(self.ir_frames)),
                )

            logger.info(f"Found {len(self.visible_frames)} visible frames")
            if self.ir_available:
                logger.info(f"Found {len(self.ir_frames)} IR frames")
        else:
            logger.info(f"Found {len(self.visible_frames)} visible frames")
            logger.warning("IR input unavailable")
    
    def read(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Read synchronized frames from directories.
        
        Returns:
            (visible_frame, ir_frame) where ir_frame is None if unavailable
        """
        if self.current_index >= len(self.visible_frames):
            return None, None
        
        # Read visible frame
        visible_frame = cv2.imread(str(self.visible_frames[self.current_index]))
        if visible_frame is None:
            logger.error(f"Failed to read visible frame: {self.visible_frames[self.current_index]}")
            return None, None
        
        # Read IR frame if available
        ir_frame = None
        if self.ir_available and self.current_index < len(self.ir_frames):
            ir_frame = cv2.imread(str(self.ir_frames[self.current_index]), cv2.IMREAD_UNCHANGED)
            if ir_frame is None:
                logger.warning(f"Failed to read IR frame: {self.ir_frames[self.current_index]}")
                ir_frame = None
        
        self.current_index += 1
        return visible_frame, ir_frame

    def reset(self):
        """Rewind to the first frame pair."""
        self.current_index = 0

    def release(self):
        """Release resources (no-op for directory source)."""
        pass


class SingleVideoSource(MultimodalFrameSource):
    """
    Frame source for single video (visible only).
    
    Used when only visible video is available. IR is reported as unavailable.
    """
    
    def __init__(self, visible_video: Union[str, Path]):
        """
        Initialize single video source.
        
        Args:
            visible_video: Path to visible video file
        """
        super().__init__(visible_video, None)
        
        self.cap = cv2.VideoCapture(str(self.visible_source))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {visible_video}")
        
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.visible_fps = self.fps
        self.ir_fps = None
        self.visible_index = 0

        logger.info(f"Visible only: {self.width}x{self.height} @ {self.fps:.2f} FPS")
        logger.warning("IR input unavailable - only visible video provided")
    
    def read(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Read frame from single video.
        
        Returns:
            (visible_frame, None) - IR is always None
        """
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        return frame, None

    def reset(self):
        """Rewind to the first frame."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self):
        """Release video capture resources."""
        if self.cap is not None:
            self.cap.release()


def create_frame_source(visible_source: Union[str, Path], 
                       ir_source: Optional[Union[str, Path]] = None) -> MultimodalFrameSource:
    """
    Factory function to create appropriate frame source based on input types.
    
    Args:
        visible_source: Path to visible video file or directory
        ir_source: Path to IR video file or directory (optional)
    
    Returns:
        Appropriate MultimodalFrameSource subclass instance
    """
    visible_path = Path(visible_source)
    
    # Determine if source is directory or file
    if visible_path.is_dir():
        # Directory-based source
        ir_path = Path(ir_source) if ir_source else None
        return DirectoryFrameSource(visible_path, ir_path)
    else:
        # Video file source
        if ir_source is None:
            return SingleVideoSource(visible_path)
        else:
            return VideoFrameSource(visible_path, ir_source)
