#!/usr/bin/env python3
"""E2: compare naive square resize with aspect-preserving letterbox."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, ".")
from preprocessing.utils.evaluate import evaluate_pipeline
from preprocessing.utils.letterbox import adjust_bboxes, letterbox

TARGET = 416


def preproc_naive_resize(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (TARGET, TARGET), interpolation=cv2.INTER_LINEAR)


def preproc_letterbox(frame: np.ndarray) -> np.ndarray:
    frame_lb, _, _ = letterbox(frame, target_size=TARGET)
    return frame_lb


def demo_bbox_adjustment() -> None:
    images = sorted(Path("dataset/exports/epi-v1/valid/images").glob("*.jpg"))
    if not images:
        raise FileNotFoundError("nenhuma imagem .jpg no split valid")
    image_path = images[0]
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"não foi possível decodificar: {image_path}")

    orig_h, orig_w = frame.shape[:2]
    frame_lb, scale, (pad_w, pad_h) = letterbox(frame, target_size=TARGET)
    bbox_lb = np.array([[60, 90, 200, 310]], dtype=float)
    bbox_orig = adjust_bboxes(bbox_lb, scale, pad_w, pad_h)

    print(f"  Frame original: {orig_w}x{orig_h}")
    print(
        f"  Frame letterboxed: {TARGET}x{TARGET} "
        f"(scale={scale:.4f}, pad_w={pad_w}, pad_h={pad_h})"
    )
    print(f"  Bbox letterboxed: {bbox_lb[0].astype(int).tolist()}")
    print(f"  Bbox original: {bbox_orig[0].astype(int).tolist()}")

    boxed_lb = frame_lb.copy()
    boxed_orig = frame.copy()
    cv2.rectangle(
        boxed_lb, tuple(bbox_lb[0, :2].astype(int)),
        tuple(bbox_lb[0, 2:].astype(int)), (0, 255, 0), 2
    )
    cv2.rectangle(
        boxed_orig, tuple(bbox_orig[0, :2].astype(int)),
        tuple(bbox_orig[0, 2:].astype(int)), (0, 255, 0), 2
    )
    output_dir = Path("preprocessing/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "e2_bbox_letterboxed.jpg"), boxed_lb)
    cv2.imwrite(str(output_dir / "e2_bbox_original.jpg"), boxed_orig)


def main() -> int:
    print("=" * 65)
    print(" E2: Resize Ingênuo vs Letterbox")
    print("=" * 65)
    results = [
        evaluate_pipeline(None, "E2-baseline"),
        evaluate_pipeline(
            preproc_naive_resize, "E2-A: resize ingênuo", label_mode="stretch"
        ),
        evaluate_pipeline(
            preproc_letterbox, "E2-B: letterbox", label_mode="letterbox"
        ),
    ]

    print("\n--- Demonstração de ajuste de coordenadas ---")
    demo_bbox_adjustment()
    print("\n--- Resumo E2 ---")
    baseline = results[0]["map50"]
    for result in results[1:]:
        delta = result["map50"] - baseline
        print(
            f"  {result['label']:30s} "
            f"mAP@0.5={result['map50']:.4f} delta={delta:+.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
