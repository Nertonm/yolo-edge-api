#!/usr/bin/env python3
"""Gera comparativo visual BGR versus RGB para inspeção."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comparativo visual BGR/RGB")
    parser.add_argument(
        "--image",
        default=None,
        help="Imagem de entrada; por padrão usa a primeira imagem de valid",
    )
    parser.add_argument(
        "--output-dir",
        default="preprocessing/outputs",
        help="Diretório de saída",
    )
    return parser.parse_args()


def find_image(image_arg: str | None) -> Path:
    if image_arg:
        path = Path(image_arg)
        if not path.is_file():
            raise FileNotFoundError(f"imagem não encontrada: {path}")
        return path

    root = Path("dataset/exports/epi-v1/valid/images")
    images = sorted(root.glob("*.jpg")) + sorted(root.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"nenhuma imagem encontrada em {root}")
    return images[0]


def main() -> int:
    args = parse_args()
    image_path = find_image(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise ValueError(f"não foi possível decodificar: {image_path}")

    # OpenCV lê BGR. Matplotlib espera RGB.
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Arquivo correto e arquivo com canais trocados de propósito.
    correct_path = output_dir / "e1_rgb_correct.png"
    swapped_path = output_dir / "e1_bgr_as_rgb_incorrect.png"
    side_by_side_path = output_dir / "e1_bgr_rgb_comparison.png"

    cv2.imwrite(str(correct_path), frame_bgr)
    # Aqui o array RGB é entregue ao writer como se fosse BGR.
    cv2.imwrite(str(swapped_path), frame_rgb)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(frame_bgr)
    axes[0].set_title("BGR interpretado como RGB (incorreto)")
    axes[1].imshow(frame_rgb)
    axes[1].set_title("BGR convertido para RGB (correto)")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(image_path.name)
    figure.tight_layout()
    figure.savefig(side_by_side_path, dpi=150)
    plt.close(figure)

    print(f"Imagem de entrada: {image_path}")
    print(f"Arquivo correto: {correct_path}")
    print(f"Arquivo com canais trocados: {swapped_path}")
    print(f"Comparativo lado a lado: {side_by_side_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
