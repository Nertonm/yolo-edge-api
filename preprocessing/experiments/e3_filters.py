#!/usr/bin/env python3
"""E3: impacto de filtros de suavizacao na deteccao de EPIs."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, ".")
from preprocessing.utils.evaluate import evaluate_pipeline


def preproc_rgb_only(frame: np.ndarray) -> np.ndarray:
    """Apenas BGR para RGB, sem filtro."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def preproc_gauss_33(frame: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(preproc_rgb_only(frame), (3, 3), sigmaX=0.8)


def preproc_gauss_55(frame: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(preproc_rgb_only(frame), (5, 5), sigmaX=1.5)


def preproc_gauss_77(frame: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(preproc_rgb_only(frame), (7, 7), sigmaX=2.0)


def preproc_median_3(frame: np.ndarray) -> np.ndarray:
    return cv2.medianBlur(preproc_rgb_only(frame), 3)


def benchmark_filter_cost(n_frames: int = 200) -> None:
    """Mede o custo medio por frame em uma imagem 640x480."""
    test = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    filters = [
        ("cvtColor apenas", lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2RGB)),
        ("GaussianBlur 3x3", lambda f: cv2.GaussianBlur(f, (3, 3), 0.8)),
        ("GaussianBlur 5x5", lambda f: cv2.GaussianBlur(f, (5, 5), 1.5)),
        ("GaussianBlur 7x7", lambda f: cv2.GaussianBlur(f, (7, 7), 2.0)),
        ("medianBlur k=3", lambda f: cv2.medianBlur(f, 3)),
        ("bilateralFilter", lambda f: cv2.bilateralFilter(f, 9, 75, 75)),
    ]
    print(f"\n--- Custo por filtro ({n_frames} frames 640x480) ---")
    for name, function in filters:
        start = time.perf_counter()
        for _ in range(n_frames):
            function(test)
        elapsed = (time.perf_counter() - start) / n_frames * 1000
        print(f"  {name:22s}: {elapsed:.2f} ms/frame")


def main() -> int:
    print("=" * 65)
    print(" E3: Filtragem: Gaussiano vs Mediana")
    print("=" * 65)
    print("[INFO] Mesmo modelo, dataset e split val em todas as variantes.")

    results = [
        evaluate_pipeline(None, "E3-baseline"),
        evaluate_pipeline(preproc_rgb_only, "E3-A: RGB apenas (sem filtro)"),
        evaluate_pipeline(preproc_gauss_33, "E3-B: GaussianBlur 3x3"),
        evaluate_pipeline(preproc_gauss_55, "E3-C: GaussianBlur 5x5"),
        evaluate_pipeline(preproc_gauss_77, "E3-D: GaussianBlur 7x7"),
        evaluate_pipeline(preproc_median_3, "E3-E: medianBlur k=3"),
    ]
    benchmark_filter_cost()

    print("\n--- Resumo E3 ---")
    baseline = results[0]["map50"]
    for result in results[1:]:
        delta = result["map50"] - baseline
        print(
            f"  {result['label']:35s} "
            f"mAP@0.5={result['map50']:.4f} delta={delta:+.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
