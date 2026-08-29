"""
scripts/inspect_dataset.py
Valida a integridade e balanceamento de um dataset no formato YOLOv8.
Uso: python scripts/inspect_dataset.py --dataset dataset/exports/epi-v1/data.yaml
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Caminho para data.yaml")
    p.add_argument("--min-per-class", type=int, default=30,
                   help="Mínimo de instâncias por classe no split de treino")
    return p.parse_args()


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def count_labels(labels_dir: Path) -> tuple:
    """Conta instâncias por classe em todos os arquivos de label e o nº de
    imagens sem label. Exemplo de saída: (counts, missing)."""
    counts = defaultdict(int)
    missing = 0
    images_dir = labels_dir.parent / "images"
    for img_path in images_dir.glob("*"):
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            missing += 1
            continue
        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cls = int(line.split()[0])
                counts[cls] += 1
    return dict(counts), missing


def main():
    args   = parse_args()
    cfg    = load_yaml(args.dataset)
    base   = Path(args.dataset).parent
    names  = cfg.get("names", [])
    nc     = cfg.get("nc", len(names))

    print(f"\n{'='*55}")
    print(f" Inspeção do Dataset: {Path(args.dataset).parent.name}")
    print(f"{'='*55}")
    print(f" Classes ({nc}): {names}")

    issues = 0
    total_imgs = 0

    for split in ["train", "valid", "test"]:
        labels_dir = base / split / "labels"
        if not labels_dir.exists():
            print(f"  [{split}] AVISO: diretório não encontrado")
            continue

        counts, missing = count_labels(labels_dir)
        total = sum(counts.values())
        imgs  = len(list((base / split / "images").glob("*")))
        total_imgs += imgs

        print(f"\n  [{split.upper()}]  {imgs} imagens  |  {total} anotações  |  {missing} sem label")
        if missing:
            issues += 1
        for cls_id, cls_name in enumerate(names):
            n = counts.get(cls_id, 0)
            bar = '█' * min(int(n / max(total, 1) * 30), 30)
            warn = "  <- ABAIXO DO MINIMO" if split == "train" and n < args.min_per_class else ""
            print(f"    {cls_name:15s} {n:5d}  {bar}{warn}")
            if warn:
                issues += 1

    print(f"\n{'='*55}")
    print(f" Total de imagens: {total_imgs}")
    if issues:
        print(f" {issues} problema(s) encontrado(s). Revise antes de treinar.")
        sys.exit(1)
    else:
        print(" Dataset aprovado para treinamento.")


if __name__ == "__main__":
    main()