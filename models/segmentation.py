"""
YOLO segmentation module for pedestrian ROI extraction.

Uses YOLOv8n-seg for lightweight instance segmentation.
Extracts pedestrian masks for feature extraction ROI.

Ultralytics returns masks at the model input resolution unless
``retina_masks=True`` is used. This module always resizes masks to the
original frame size, so downstream code can safely do ``frame * mask``.
"""

import logging
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class YOLOSegmentation:
    """
    YOLO segmentation wrapper for pedestrian ROI extraction.

    Uses YOLOv8n-seg (segmentation variant) for instance segmentation.
    Focuses on the 'person' class for pedestrian detection.
    """

    def __init__(self,
                 model_path: str = "yolov8n-seg.pt",
                 device: str = "cuda",
                 conf: float = 0.25,
                 retina_masks: bool = True):
        """
        Initialize YOLO segmentation model.

        Args:
            model_path: Path to YOLO segmentation model weights
            device: Device to run inference on ('cuda' or 'cpu')
            conf: Confidence threshold for detections
            retina_masks: Return masks at original image resolution
        """
        self.device = device if (device == "cpu" or _cuda_available()) else "cpu"
        if self.device != device:
            logger.warning(f"Requested device {device} is unavailable; using {self.device}")
        self.conf = conf
        self.retina_masks = retina_masks

        self.model = YOLO(model_path)
        try:
            self.model.to(self.device)
        except Exception as exc:  # pragma: no cover - depends on installed torch
            logger.warning(f"Could not move YOLO model to {self.device}: {exc}")
            self.device = "cpu"
            self.model.to("cpu")

        self.person_class_id = 0  # COCO 'person' class ID

        logger.info(f"YOLO segmentation loaded: {model_path} on {self.device}")

    def extract_person_masks(self,
                             frames: Sequence[np.ndarray]) -> List[Optional[np.ndarray]]:
        """
        Extract pedestrian masks for a batch of frames.

        Args:
            frames: Sequence of RGB/BGR frames (H, W, 3)

        Returns:
            List of binary masks (H, W) or None when no person is detected.
            Masks are always at the input frame resolution.
        """
        if not frames:
            return []

        results = self.model(
            list(frames),
            verbose=False,
            conf=self.conf,
            retina_masks=self.retina_masks,
            device=self.device,
        )

        masks_out: List[Optional[np.ndarray]] = []
        for frame, result in zip(frames, results):
            frame_h, frame_w = frame.shape[:2]
            if result is None or result.masks is None or result.boxes is None:
                masks_out.append(None)
                continue

            person_indices = [
                i for i, cls in enumerate(result.boxes.cls)
                if int(cls) == self.person_class_id
            ]
            if not person_indices:
                masks_out.append(None)
                continue

            masks = result.masks.data.cpu().numpy()  # (N, H_mask, W_mask)
            person_masks = masks[person_indices]

            combined = np.zeros(person_masks.shape[1:], dtype=np.uint8)
            for mask in person_masks:
                combined = np.maximum(combined, (mask > 0.5).astype(np.uint8))

            if combined.shape[:2] != (frame_h, frame_w):
                combined = cv2.resize(
                    combined, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST
                )

            masks_out.append(combined if combined.any() else None)

        return masks_out

    def extract_person_mask(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract a pedestrian mask for one frame.

        Args:
            frame: Input BGR frame (H, W, 3)

        Returns:
            Binary mask (H, W) where 1 = pedestrian region, 0 = background.
            Returns None if no person is detected.
        """
        return self.extract_person_masks([frame])[0]

    def extract_largest_person_mask(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract a mask for the largest detected pedestrian.

        Args:
            frame: Input BGR frame (H, W, 3)

        Returns:
            Binary mask (H, W) for the largest pedestrian, or None.
        """
        results = self.model(
            frame,
            verbose=False,
            conf=self.conf,
            retina_masks=self.retina_masks,
            device=self.device,
        )

        if not results or results[0].masks is None or results[0].boxes is None:
            return None

        result = results[0]
        person_indices = [
            i for i, cls in enumerate(result.boxes.cls)
            if int(cls) == self.person_class_id
        ]
        if not person_indices:
            return None

        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()

        largest_idx = None
        max_area = 0.0
        for idx in person_indices:
            x1, y1, x2, y2 = boxes[idx]
            area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
            if area > max_area:
                max_area = area
                largest_idx = idx

        if largest_idx is None:
            return None

        mask = (masks[largest_idx] > 0.5).astype(np.uint8)
        frame_h, frame_w = frame.shape[:2]
        if mask.shape[:2] != (frame_h, frame_w):
            mask = cv2.resize(mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
        return mask if mask.any() else None

    def apply_mask_to_frame(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply a binary mask to a frame, zeroing out the background."""
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(
                mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        mask_expanded = mask[:, :, np.newaxis]
        masked = frame * mask_expanded
        return masked.astype(np.uint8)

    def get_masked_roi(self,
                       frame: np.ndarray,
                       mask: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Get a masked frame and the bounding box of the mask ROI."""
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(
                mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST
            )

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return frame, (0, 0, frame.shape[1], frame.shape[0])

        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        masked = self.apply_mask_to_frame(frame, mask)
        return masked, (int(x1), int(y1), int(x2), int(y2))


def _cuda_available() -> bool:
    """Check CUDA without importing torch at module import time."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
