"""Manual letterbox and bounding-box coordinate adjustment."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def letterbox(
    frame: np.ndarray,
    target_size: int = 640,
    pad_color: int = 114,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize while preserving aspect ratio and add centered gray padding."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame deve ter shape (altura, largura, 3)")
    if target_size < 1:
        raise ValueError("target_size deve ser positivo")

    height, width = frame.shape[:2]
    scale = min(target_size / height, target_size / width)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    pad_w = (target_size - new_width) // 2
    pad_h = (target_size - new_height) // 2
    output = np.full(
        (target_size, target_size, 3), pad_color, dtype=frame.dtype
    )
    output[pad_h:pad_h + new_height, pad_w:pad_w + new_width] = resized
    return output, scale, (pad_w, pad_h)


def adjust_bboxes(
    boxes_xyxy: np.ndarray,
    scale: float,
    pad_w: int,
    pad_h: int,
) -> np.ndarray:
    """Map xyxy boxes from letterboxed coordinates to original coordinates."""
    boxes = np.asarray(boxes_xyxy).copy().astype(float)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes_xyxy deve ter shape (N, 4)")
    if scale <= 0:
        raise ValueError("scale deve ser positivo")
    boxes[:, [0, 2]] -= pad_w
    boxes[:, [1, 3]] -= pad_h
    boxes /= scale
    return boxes
