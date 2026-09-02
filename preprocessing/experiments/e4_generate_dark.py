#!/usr/bin/env python3
"""Gera imagens subexpostas para o experimento E4."""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

SRC = Path("dataset/exports/epi-v1/valid")
DEST = Path("dataset/exports/epi-v1-dark/valid")
GAMMA = 2.2


def main() -> int:
    src_images = sorted(SRC.joinpath("images").glob("*.jpg"))
    src_labels = sorted(SRC.joinpath("labels").glob("*.txt"))
    if not src_images:
        raise FileNotFoundError(f"nenhuma imagem encontrada em {SRC / 'images'}")

    dest_images = DEST / "images"
    dest_labels = DEST / "labels"
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    for label in src_labels:
        shutil.copy2(label, dest_labels / label.name)

    table = np.array(
        [((value / 255.0) ** GAMMA) * 255 for value in range(256)],
        dtype=np.uint8,
    )
    generated = 0
    for image_path in src_images:
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"imagem inválida: {image_path}")
        dark = cv2.LUT(frame, table)
        if not cv2.imwrite(str(dest_images / image_path.name), dark):
            raise OSError(f"falha ao salvar: {image_path.name}")
        generated += 1

    print(f"Geradas {generated} imagens escuras em {dest_images}")
    print(f"Labels copiadas: {len(src_labels)}")
    print(f"Gamma aplicado: {GAMMA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
