"""Reusable image preprocessing for the YOLO inference pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from preprocessing.utils.letterbox import letterbox


@dataclass(frozen=True)
class PreprocessConfig:
    """Immutable configuration for one preprocessing pipeline."""

    infer_size: int = 320
    convert_rgb: bool = True
    use_letterbox: bool = True
    gaussian_blur: bool = False
    gaussian_ksize: int = 3
    gaussian_sigma: float = 0.8
    median_blur: bool = False
    median_ksize: int = 3
    clahe: bool = False
    clahe_clip: float = 2.0
    clahe_tile: int = 8
    clahe_space: str = "lab"
    normalize: bool = False


@dataclass(frozen=True)
class PreprocessResult:
    """Processed frame plus geometry metadata for box restoration."""

    frame: np.ndarray
    scale: float = 1.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    pad_w: int = 0
    pad_h: int = 0
    orig_size: Tuple[int, int] = (0, 0)


class Preprocessor:
    """Stateless, configurable preprocessing for one BGR frame at a time."""

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.cfg = config or PreprocessConfig()
        self._validate_config()
        self._clahe = (
            cv2.createCLAHE(
                clipLimit=self.cfg.clahe_clip,
                tileGridSize=(self.cfg.clahe_tile, self.cfg.clahe_tile),
            )
            if self.cfg.clahe
            else None
        )

    def _validate_config(self) -> None:
        if self.cfg.infer_size < 1:
            raise ValueError("infer_size deve ser positivo")
        for name, enabled, kernel in (
            ("gaussian_ksize", self.cfg.gaussian_blur, self.cfg.gaussian_ksize),
            ("median_ksize", self.cfg.median_blur, self.cfg.median_ksize),
        ):
            if enabled and (kernel < 3 or kernel % 2 == 0):
                raise ValueError(f"{name} deve ser ímpar e >= 3")
        if self.cfg.clahe and self.cfg.clahe_space not in {"lab", "hsv"}:
            raise ValueError("clahe_space deve ser 'lab' ou 'hsv'")
        if self.cfg.clahe and self.cfg.clahe_tile < 1:
            raise ValueError("clahe_tile deve ser positivo")

    def process(self, frame: np.ndarray) -> PreprocessResult:
        """Process one BGR HWC frame and return image plus geometry metadata."""
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame deve ter shape (altura, largura, 3)")
        orig_h, orig_w = frame.shape[:2]
        if orig_h < 1 or orig_w < 1:
            raise ValueError("frame não pode ser vazio")

        out = frame.copy()
        if self.cfg.clahe:
            out = self._apply_clahe(out)
        if self.cfg.gaussian_blur:
            out = cv2.GaussianBlur(
                out,
                (self.cfg.gaussian_ksize, self.cfg.gaussian_ksize),
                sigmaX=self.cfg.gaussian_sigma,
            )
        elif self.cfg.median_blur:
            out = cv2.medianBlur(out, self.cfg.median_ksize)
        if self.cfg.convert_rgb:
            out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

        if self.cfg.use_letterbox:
            out, scale, (pad_w, pad_h) = letterbox(out, self.cfg.infer_size)
            scale_x = scale_y = scale
        else:
            out = cv2.resize(
                out,
                (self.cfg.infer_size, self.cfg.infer_size),
                interpolation=cv2.INTER_LINEAR,
            )
            scale_x = self.cfg.infer_size / orig_w
            scale_y = self.cfg.infer_size / orig_h
            scale = min(scale_x, scale_y)
            pad_w = pad_h = 0

        if self.cfg.normalize:
            out = out.astype(np.float32) / 255.0

        return PreprocessResult(
            frame=out,
            scale=scale,
            scale_x=scale_x,
            scale_y=scale_y,
            pad_w=pad_w,
            pad_h=pad_h,
            orig_size=(orig_h, orig_w),
        )

    def adjust_boxes(
        self, boxes_xyxy: np.ndarray, result: PreprocessResult
    ) -> np.ndarray:
        """Map boxes from processed coordinates back to original coordinates."""
        boxes = np.asarray(boxes_xyxy).copy().astype(float)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("boxes_xyxy deve ter shape (N, 4)")
        if result.scale_x <= 0 or result.scale_y <= 0:
            raise ValueError("escalas inválidas no resultado")
        boxes[:, [0, 2]] -= result.pad_w
        boxes[:, [1, 3]] -= result.pad_h
        boxes[:, [0, 2]] /= result.scale_x
        boxes[:, [1, 3]] /= result.scale_y
        return boxes

    def _apply_clahe(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self._clahe is None:
            return frame_bgr
        if self.cfg.clahe_space == "lab":
            converted = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
            first, second, third = cv2.split(converted)
            first = self._clahe.apply(first)
            return cv2.cvtColor(
                cv2.merge([first, second, third]), cv2.COLOR_LAB2BGR
            )
        converted = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        first, second, third = cv2.split(converted)
        third = self._clahe.apply(third)
        return cv2.cvtColor(cv2.merge([first, second, third]), cv2.COLOR_HSV2BGR)


CONFIG_DEFAULT = PreprocessConfig(
    infer_size=320,
    convert_rgb=True,
    use_letterbox=True,
)

CONFIG_LOW_LIGHT = PreprocessConfig(
    infer_size=320,
    convert_rgb=True,
    use_letterbox=True,
    clahe=True,
    clahe_clip=2.0,
    clahe_tile=8,
    clahe_space="lab",
)

CONFIG_HIGH_QUALITY = PreprocessConfig(
    infer_size=640,
    convert_rgb=True,
    use_letterbox=True,
)
