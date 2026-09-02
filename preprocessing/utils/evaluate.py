"""
preprocessing/utils/evaluate.py
Avalia o mAP@0.5 de um pipeline de pré-processamento no dataset epi-v1.
Recebe uma função de pré-processamento e retorna as métricas.
"""
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import cv2
from ultralytics import YOLO
import shutil
import yaml
import torch

# Patch para contornar o aviso/erro de weights_only no torch.load em versões recentes
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Resolve project data from this module's location instead of the current
# working directory. Ultralytics resolves relative dataset paths against its
# global datasets directory, which otherwise makes this YAML point at a
# non-existent ``.../datasets/dataset/...`` path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_YAML = PROJECT_ROOT / "dataset/exports/epi-v1/data.yaml"
MODEL_PATH = "models/yolov8n.pt"


def resolve_dataset_yaml(yaml_path: str | Path) -> str:
    """Return a temporary yaml with absolute paths so Ultralytics does not
    prepend the global datasets_dir from ~/.config/Ultralytics/settings.yaml.
    """
    p = Path(yaml_path)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    p = p.resolve()

    if not p.exists():
        raise FileNotFoundError(f"Dataset config not found: {p}")

    cfg = yaml.safe_load(p.read_text()) or {}
    base = p.parent.resolve()
    cfg["path"] = str(base)

    for key in ("train", "val", "test"):
        v = cfg.get(key)
        if not v:
            continue
        value = Path(v)
        if not value.is_absolute():
            value = (base / value).resolve()
        cfg[key] = str(value)

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        delete=False,
        prefix="eval_dataset_",
    )
    yaml.safe_dump(cfg, tmp, sort_keys=False)
    tmp.close()
    return tmp.name


def transform_yolo_label_text(
    label_text: str,
    orig_shape: tuple[int, int],
    output_shape: tuple[int, int],
    mode: str = "copy",
) -> str:
    """Transform normalized YOLO boxes for stretch or centered letterbox."""
    if mode == "copy":
        return label_text
    if mode not in {"stretch", "letterbox"}:
        raise ValueError(f"modo de label desconhecido: {mode}")

    orig_h, orig_w = orig_shape
    out_h, out_w = output_shape
    if orig_h <= 0 or orig_w <= 0 or out_h <= 0 or out_w <= 0:
        raise ValueError("dimensões inválidas para transformar labels")

    if mode == "letterbox":
        scale = min(out_w / orig_w, out_h / orig_h)
        pad_x = (out_w - orig_w * scale) / 2.0
        pad_y = (out_h - orig_h * scale) / 2.0
    else:
        scale_x = out_w / orig_w
        scale_y = out_h / orig_h
        scale = None
        pad_x = pad_y = 0.0

    transformed = []
    for line in label_text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = parts[0]
        xc, yc, bw, bh = map(float, parts[1:5])
        x1 = (xc - bw / 2.0) * orig_w
        y1 = (yc - bh / 2.0) * orig_h
        x2 = (xc + bw / 2.0) * orig_w
        y2 = (yc + bh / 2.0) * orig_h
        if mode == "letterbox":
            x1, x2 = x1 * scale + pad_x, x2 * scale + pad_x
            y1, y2 = y1 * scale + pad_y, y2 * scale + pad_y
        else:
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y
        new_xc = ((x1 + x2) / 2.0) / out_w
        new_yc = ((y1 + y2) / 2.0) / out_h
        new_bw = (x2 - x1) / out_w
        new_bh = (y2 - y1) / out_h
        transformed.append(
            f"{cls_id} {new_xc:.10f} {new_yc:.10f} "
            f"{new_bw:.10f} {new_bh:.10f}"
        )
    return "\n".join(transformed) + ("\n" if transformed else "")


def evaluate_pipeline(
    preprocess_fn: Optional[Callable] = None,
    label: str = "baseline",
    split: str = "val",
    verbose: bool = False,
    classes: Optional[list[int]] = None,
    label_mode: str = "copy",
) -> dict:
    """
    Avalia o mAP@0.5 do modelo com uma função de pré-processamento opcional.
    
    Args:
        preprocess_fn: função que recebe um frame BGR NumPy e retorna
            um frame transformado (qualquer formato aceito pelo YOLO).
            Se None, usa o comportamento padrão da Ultralytics.
        label: nome para identificar este experimento no log.
        split: split do dataset a avaliar ('val' ou 'test').
        
    Returns:
        dict com map50, map50_95, tempo médio de pré-processamento em ms.
        
    Nota sobre E1 (espaço de cor): esta função salva os frames pré-processados
    em disco e aponta o model.val() para eles. O model.val() sempre relê o
    arquivo e aplica sua própria conversão BGR->RGB internamente -- por isso
    ela mede corretamente transformações espaciais/de intensidade (resize,
    blur, CLAHE), mas não consegue medir o efeito de inverter a ordem dos
    canais (E1): o resultado do E1 aparece com delta ~0 aqui, de propósito.
    A verificação do E1 é visual, via e1_visualize.py.
    """
    model = YOLO(MODEL_PATH)
    
    if preprocess_fn is None:
        # Avaliação padrão : sem pré-processamento customizado
        val_kwargs = {
            "data": resolve_dataset_yaml(DATASET_YAML),
            "split": split,
            "verbose": verbose,
        }
        if classes is not None:
            val_kwargs["classes"] = classes
        metrics = model.val(**val_kwargs)
        preproc_ms = 0.0
    else:
        # Roboflow exporta a pasta de validação como "valid/", mas a chave
        # do data.yaml (e o split= do Ultralytics) usa "val" -- traduz aqui
        split_dirname = {"val": "valid", "test": "test", "train": "train"}.get(split, split)
        dataset_dir = Path(DATASET_YAML).parent
        
        src_images_dir = dataset_dir / split_dirname / "images"
        src_labels_dir = dataset_dir / split_dirname / "labels"
        
        images = sorted(src_images_dir.glob("*.jpg")) + sorted(src_images_dir.glob("*.png"))
        
        # Diretório temporário com as imagens JÁ pré-processadas + os mesmos
        # labels -- o model.val() só sabe ler do disco, então o resultado de
        # preprocess_fn precisa ser gravado antes de avaliar (esse era o bug:
        # antes, os frames processados eram calculados e descartados)
        safe_label = "".join(c if c.isalnum() else "_" for c in label)
        tmp_root = Path("preprocessing/outputs/_tmp_eval") / safe_label
        
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
            
        tmp_images_dir = tmp_root / "images"
        tmp_labels_dir = tmp_root / "labels"
        tmp_images_dir.mkdir(parents=True, exist_ok=True)
        tmp_labels_dir.mkdir(parents=True, exist_ok=True)
        
        preproc_times = []
        
        for img_path in images:
            frame = cv2.imread(str(img_path))
            if frame is None:
                raise ValueError(f"imagem inválida: {img_path}")
            t0 = time.perf_counter()
            frame_proc = preprocess_fn(frame)
            preproc_times.append((time.perf_counter() - t0) * 1000)
            
            cv2.imwrite(str(tmp_images_dir / img_path.name), frame_proc)
            
            label_src = src_labels_dir / f"{img_path.stem}.txt"
            if label_src.exists():
                label_text = transform_yolo_label_text(
                    label_src.read_text(),
                    frame.shape[:2],
                    frame_proc.shape[:2],
                    label_mode,
                )
                (tmp_labels_dir / label_src.name).write_text(label_text)
                
        # data.yaml temporário apontando para as imagens já processadas
        with open(DATASET_YAML) as f:
            base_cfg = yaml.safe_load(f)
            
        tmp_yaml_cfg = {
            "path": str(tmp_root.resolve()),
            "train": "images",
            "val": "images",
            "test": "images",
            "names": base_cfg["names"],
        }
        
        tmp_yaml = tmp_root / "data.yaml"
        with open(tmp_yaml, "w") as f:
            yaml.safe_dump(tmp_yaml_cfg, f)
            
        # Roda inferência nas imagens JÁ processadas (não mais nas originais)
        val_kwargs = {"data": str(tmp_yaml), "split": "val", "verbose": verbose}
        if classes is not None:
            val_kwargs["classes"] = classes
        metrics = model.val(**val_kwargs)
        preproc_ms = float(np.mean(preproc_times)) if preproc_times else 0.0
        
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    
    print(f"[{label:30s}] mAP@0.5={map50:.4f} mAP@0.5:0.95={map50_95:.4f} preproc={preproc_ms:.1f}ms")
    
    return {
        "label": label,
        "map50": map50,
        "map50_95": map50_95,
        "preproc_ms": preproc_ms
    }
