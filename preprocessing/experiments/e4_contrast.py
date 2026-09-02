#!/usr/bin/env python3
"""E4: equalizacao global versus CLAHE em imagens subexpostas."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, ".")
import preprocessing.utils.evaluate as evaluate_module
from preprocessing.utils.evaluate import evaluate_pipeline

DATASET_DARK = "dataset/exports/epi-v1-dark/data.yaml"


def rgb_only(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def equalize_hist_hsv(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    value = cv2.equalizeHist(value)
    return cv2.cvtColor(cv2.merge([hue, saturation, value]), cv2.COLOR_HSV2RGB)


def equalize_hist_lab(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    lightness = cv2.equalizeHist(lightness)
    return cv2.cvtColor(cv2.merge([lightness, channel_a, channel_b]), cv2.COLOR_LAB2RGB)


def clahe_hsv(frame, clip=2.0, tile=8):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    value = clahe.apply(value)
    return cv2.cvtColor(cv2.merge([hue, saturation, value]), cv2.COLOR_HSV2RGB)


def clahe_lab(frame, clip=2.0, tile=8):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    lightness = clahe.apply(lightness)
    return cv2.cvtColor(cv2.merge([lightness, channel_a, channel_b]), cv2.COLOR_LAB2RGB)


def main() -> int:
    dark_path = Path(DATASET_DARK)
    if not dark_path.exists():
        raise FileNotFoundError(f"dataset escuro ausente: {dark_path}")

    print("=" * 65)
    print(" E4: Equalização de Contraste em Imagens Subexpostas")
    print("=" * 65)
    print("[INFO] Dataset: epi-v1-dark, gamma=2.2, split val")

    original_dataset = evaluate_module.DATASET_YAML
    evaluate_module.DATASET_YAML = DATASET_DARK
    try:
        results = [
            evaluate_pipeline(rgb_only, "E4-A: RGB apenas (ilum. ruim)"),
            evaluate_pipeline(equalize_hist_hsv, "E4-B: equalizeHist (HSV/V)"),
            evaluate_pipeline(equalize_hist_lab, "E4-C: equalizeHist (LAB/L)"),
            evaluate_pipeline(clahe_hsv, "E4-D: CLAHE clip=2 tile=8 (HSV)"),
            evaluate_pipeline(clahe_lab, "E4-E: CLAHE clip=2 tile=8 (LAB)"),
            evaluate_pipeline(
                lambda frame: clahe_lab(frame, clip=4.0, tile=8),
                "E4-F: CLAHE clip=4 tile=8 (LAB)",
            ),
        ]
    finally:
        evaluate_module.DATASET_YAML = original_dataset

    print("\n--- Resumo E4 (dataset escurecido) ---")
    baseline = results[0]["map50"]
    for result in results[1:]:
        delta = result["map50"] - baseline
        print(
            f"  {result['label']:38s} "
            f"mAP@0.5={result['map50']:.4f} delta={delta:+.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
