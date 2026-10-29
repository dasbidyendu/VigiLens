"""Small shared helpers for dataset discovery and pairing."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DEFAULT_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

_MODALITY_SUFFIXES = (
    "_rgb", "_ir", "-rgb", "-ir", "_visible", "_thermal",
    "_color", "_lwir", "-visible", "-thermal", "-color", "-lwir",
)


def natural_key(value: str):
    """Sort key that orders ``frame2`` before ``frame10``."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(value))]


def extract_sequence_id(video_path: Path) -> str:
    """Remove a modality suffix and the extension from a video filename."""
    stem = Path(video_path).stem
    lowered = stem.lower()
    for suffix in _MODALITY_SUFFIXES:
        if lowered.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def discover_videos(data_dir: Path,
                    split: str,
                    modality: str,
                    extensions: Optional[Sequence[str]] = None,
                    required: bool = True) -> List[Path]:
    """Discover video files in ``data_dir/split/modality``."""
    video_dir = Path(data_dir) / split / modality
    if not video_dir.exists():
        if required:
            raise ValueError(f"Video directory not found: {video_dir}")
        logger.warning("Video directory not found: %s. %s modality disabled.", video_dir, modality.upper())
        return []

    allowed = {ext.lower() for ext in (extensions or DEFAULT_VIDEO_EXTENSIONS)}
    videos = sorted(
        (path for path in video_dir.rglob("*")
         if path.is_file() and path.suffix.lower() in allowed),
        key=lambda path: natural_key(str(path)),
    )
    if not videos:
        if required:
            raise ValueError(f"No videos found in {video_dir} with extensions {sorted(allowed)}")
        logger.warning("No videos found in %s. %s modality disabled.", video_dir, modality.upper())
    return videos


def pair_videos_by_sequence_id(rgb_videos: Sequence[Path],
                               ir_videos: Sequence[Path]) -> List[Tuple[Path, Path]]:
    """Pair RGB and IR videos by their modality-stripped sequence ID."""
    rgb_dict = {}
    for rgb_path in rgb_videos:
        sequence_id = extract_sequence_id(rgb_path)
        if sequence_id in rgb_dict:
            raise ValueError(
                f"Duplicate RGB sequence ID: {sequence_id} "
                f"({rgb_dict[sequence_id]} and {rgb_path})"
            )
        rgb_dict[sequence_id] = Path(rgb_path)

    ir_dict = {}
    for ir_path in ir_videos:
        sequence_id = extract_sequence_id(ir_path)
        if sequence_id in ir_dict:
            raise ValueError(
                f"Duplicate IR sequence ID: {sequence_id} "
                f"({ir_dict[sequence_id]} and {ir_path})"
            )
        ir_dict[sequence_id] = Path(ir_path)

    common_ids = set(rgb_dict) & set(ir_dict)
    unmatched_rgb = set(rgb_dict) - common_ids
    unmatched_ir = set(ir_dict) - common_ids
    if unmatched_rgb:
        logger.warning("Unmatched RGB videos: %s", sorted(unmatched_rgb))
    if unmatched_ir:
        logger.warning("Unmatched IR videos: %s", sorted(unmatched_ir))
    if not common_ids:
        raise ValueError("No matching RGB/IR pairs found. Check filename conventions.")

    pairs = [(rgb_dict[sequence_id], ir_dict[sequence_id]) for sequence_id in sorted(common_ids)]
    logger.info("Paired %d RGB/IR video pairs by sequence ID", len(pairs))
    return pairs


def list_images(directory: Path,
                extensions: Sequence[str] = IMAGE_EXTENSIONS) -> List[Path]:
    allowed = {ext.lower() for ext in extensions}
    return sorted(
        (path for path in Path(directory).iterdir()
         if path.is_file() and path.suffix.lower() in allowed),
        key=lambda path: natural_key(path.name),
    )


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_stem(value: str) -> str:
    """Make a filesystem-safe identifier from a sequence name."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return cleaned or "sequence"
