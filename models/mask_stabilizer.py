"""
Temporal stabilisation of per-frame person masks.

YOLO occasionally misses a person for a frame or two.  Without a mask the
encoders see the full frame, so a single missed detection makes the feature
vector jump and the prediction error spike: a false alarm caused by the
detector, not by the scene.  ``MaskStabilizer`` re-uses the most recent mask
for up to ``hold_frames`` consecutive misses before falling back to the full
frame.

The same stabiliser is used by feature precomputation and by streaming
inference so cached training features and live features follow one rule.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


class MaskStabilizer:
    """Hold the last person mask across short detection gaps."""

    def __init__(self, hold_frames: int = 5):
        self.hold_frames = max(0, int(hold_frames))
        self._last_mask: Optional[np.ndarray] = None
        self._misses = 0

    def reset(self) -> None:
        self._last_mask = None
        self._misses = 0

    def update(self, mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Return the mask to use for this frame (``None`` = full frame)."""
        if mask is not None:
            self._last_mask = mask
            self._misses = 0
            return mask
        if self._last_mask is not None and self._misses < self.hold_frames:
            self._misses += 1
            return self._last_mask
        self._last_mask = None
        return None

    def update_many(self, masks: Sequence[Optional[np.ndarray]]) -> List[Optional[np.ndarray]]:
        """Stabilise consecutive frames of one video, in order."""
        return [self.update(mask) for mask in masks]
