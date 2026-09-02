#!/usr/bin/env python3
"""E1: impacto da conversao BGR/RGB no pipeline de avaliacao.

Reproduz as tres variantes descritas na Aula 5. A avaliacao compartilhada
salva os frames em disco antes de chamar model.val(); por isso este experimento
serve como comparacao do caminho de serializacao, mas nao isola perfeitamente
a ordem de canais em memoria. A limitacao fica registrada no output.
"""
import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
from preprocessing.utils.evaluate import evaluate_pipeline


def preproc_bgr_raw(frame: np.ndarray) -> np.ndarray:
    """Retorna o frame BGR sem conversao, como no enunciado."""
    return frame


def preproc_rgb_correct(frame: np.ndarray) -> np.ndarray:
    """Converte BGR para RGB com OpenCV."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def preproc_rgb_flip(frame: np.ndarray) -> np.ndarray:
    """Converte BGR para RGB invertendo o ultimo eixo do NumPy."""
    return frame[:, :, ::-1]


def main() -> int:
    print("=" * 65)
    print(" E1: Impacto da Conversao de Espaco de Cor")
    print("=" * 65)
    print("[INFO] Mesmo modelo, mesmo split val e mesmo dataset epi-v1.")
    print("[LIMITACAO] evaluate_pipeline serializa e relê as imagens; o delta")
    print("[LIMITACAO] não isola uma chamada model(frame) BGR/RGB em memória.")

    results = [
        evaluate_pipeline(None, "E1-baseline (Ultralytics padrão)"),
        evaluate_pipeline(preproc_bgr_raw, "E1-A: BGR sem conversão"),
        evaluate_pipeline(preproc_rgb_correct, "E1-B: BGR para RGB (cvtColor)"),
        evaluate_pipeline(preproc_rgb_flip, "E1-C: BGR para RGB (NumPy flip)"),
    ]

    print("\n--- Resumo E1 ---")
    baseline_map = results[0]["map50"]
    for result in results[1:]:
        delta = result["map50"] - baseline_map
        sign = "+" if delta >= 0 else ""
        print(
            f"  {result['label']:38s} "
            f"mAP@0.5={result['map50']:.4f} delta={sign}{delta:.4f}"
        )
    print("[INFO] Interprete o resultado junto da limitacao acima.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
