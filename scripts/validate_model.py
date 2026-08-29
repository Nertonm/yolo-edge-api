"""
scripts/validate_model.py
Quality gate: bloqueia o deploy se o mAP@0.5 estiver abaixo do limiar.
Uso: python scripts/validate_model.py [--threshold 0.60]

Padrao: valida contra o dataset EPI da Aula 4 (dataset/exports/epi-v1/data.yaml).
Fallback para COCO128 apenas se o dataset EPI nao existir.
"""
import argparse
import sys
import tempfile
from pathlib import Path

# O modelo .pt foi salvo sem allowlist de globals; o torch 2.6+ bloqueia o
# weights_only=True por padrao. O modelo e artefato proprio (Aula 3), entao
# liberamos o weights_only=False como default, mesmo padrao dos scripts v1/v2/v3.
import torch as _torch
_orig_torch_load = _torch.load


def _load_allow_weights_only(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


_torch.load = _load_allow_weights_only

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

# Limiar default conforme a aula (quality gate 0.60)
DEFAULT_THRESHOLD = 0.60
# Dataset EPI da Aula 4 (ponteiro versionado via DVC)
DATASET_YAML = "dataset/exports/epi-v1/data.yaml"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/yolov8n.pt")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--dataset",
        default=None,
        help="Caminho para o YAML do dataset de validacao (default: dataset EPI)",
    )
    return parser.parse_args()


def resolve_dataset_yaml(yaml_path):
    """Rewrites train/val/test/path to ABSOLUTE paths based on the yaml's own
    location. O ultralytics resolve paths relativos contra o datasets_dir do
    settings, nao contra o diretorio do yaml; com paths absolutos funciona
    igual na acerola e no CI (qualquer checkout dir). Retorna um yaml
    temporario com as paths absolutas e caminho do arquivo."""
    p = Path(yaml_path)
    if not p.is_absolute():
        p = Path.cwd() / p
    p = p.resolve()
    if yaml is None:
        raise SystemExit("[ERRO] modulo 'yaml' ausente para resolver dataset")
    cfg = yaml.safe_load(p.read_text())
    base = p.parent
    cfg["path"] = str(base)
    for key in ("train", "val", "test"):
        v = cfg.get(key)
        if v:
            cfg[key] = str((base / v).resolve())
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False,
        prefix="validate_model_data_",
    )
    yaml.safe_dump(cfg, tmp, sort_keys=False)
    tmp.close()
    return tmp.name


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERRO] Modelo nao encontrado: {model_path}")
        sys.exit(1)

    from ultralytics import YOLO
    model = YOLO(str(model_path))

    dataset = args.dataset
    if dataset is None and Path(DATASET_YAML).exists():
        dataset = DATASET_YAML
        print(f"[INFO] Usando dataset EPI padrao: {dataset}")

    if dataset:
        dataset = resolve_dataset_yaml(dataset)
        print(f"[INFO] Validando com dataset: {dataset}")
        metrics = model.val(data=dataset, split="val", verbose=False)
    else:
        # Validacao rapida com COCO128 (dataset embutido no ultralytics)
        print("[INFO] Validando com COCO128 (fallback, dataset EPI ausente)")
        metrics = model.val(data="coco128.yaml", split="val", verbose=False)

    map50 = float(metrics.box.map50)
    print(f"[INFO] mAP@0.5 = {map50:.4f}  |  Limiar: {args.threshold:.4f}")
    if map50 < args.threshold:
        print("[FALHA] mAP abaixo do limiar. Deploy bloqueado.")
        sys.exit(1)
    print("[OK] Quality gate aprovado. Deploy autorizado.")


if __name__ == "__main__":
    main()