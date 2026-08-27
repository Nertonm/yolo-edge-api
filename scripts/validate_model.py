"""
scripts/validate_model.py
Quality gate: bloqueia o deploy se o mAP@0.5 estiver abaixo do limiar.
Uso: python scripts/validate_model.py [--threshold 0.50]
"""
import argparse
import sys
from pathlib import Path

# Limiar padrão de qualidade
DEFAULT_THRESHOLD = 0.50


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/yolov8n.pt")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--dataset",
        default=None,
        help="Caminho para o YAML do dataset de validacao (opcional)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERRO] Modelo nao encontrado: {model_path}")
        sys.exit(1)

    from ultralytics import YOLO
    model = YOLO(str(model_path))

    if args.dataset:
        print(f"[INFO] Validando com dataset: {args.dataset}")
        metrics = model.val(data=args.dataset, split="val", verbose=False)
    else:
        # Validacao rapida com COCO128 (dataset embutido no ultralytics)
        print("[INFO] Validando com COCO128 (dataset padrao)")
        metrics = model.val(data="coco128.yaml", split="val", verbose=False)

    map50 = float(metrics.box.map50)
    print(f"[INFO] mAP@0.5 = {map50:.4f}  |  Limiar: {args.threshold:.4f}")
    if map50 < args.threshold:
        print("[FALHA] mAP abaixo do limiar. Deploy bloqueado.")
        sys.exit(1)
    print("[OK] Quality gate aprovado. Deploy autorizado.")


if __name__ == "__main__":
    main()
