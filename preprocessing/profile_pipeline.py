#!/usr/bin/env python3
"""Profile the Aula 5 camera-to-detection pipeline.

Measures capture, preprocessing, inference, postprocessing, and OSD separately.
The default is 50 samples, matching the Aula 5 profiling exercise.
"""
from __future__ import annotations

import argparse
import contextlib
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

# The project model was saved with a torch version whose safe loader rejects
# some legacy globals. This is the same compatibility shim used elsewhere.
_original_torch_load = torch.load


def _torch_load_legacy_compatible(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


# Keep the model-loading behavior consistent with the existing pipeline.
torch.load = _torch_load_legacy_compatible

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stream.capture_frames import RpicamCapture  # noqa: E402
from ultralytics import YOLO  # noqa: E402


STAGES = ("capture", "preproc", "infer", "postproc", "osd")


class Tee:
    """Sends the same text to the terminal and to a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()



def open_run_log(log_dir: Path) -> tuple[object, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"profile_{stamp}.log"
    return path.open("w", encoding="utf-8"), path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profiling do pipeline câmera -> YOLO conforme a Aula 5"
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--infer-size", type=int, default=416)
    parser.add_argument("--samples", type=int, default=50,
                        help="Número de amostras (padrão: 50)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Frames descartados para aquecer o modelo")
    parser.add_argument("--model", default="models/yolov8n.pt")
    parser.add_argument("--log-dir", default="preprocessing/outputs/profiling",
                        help="Diretório dos logs (padrão: preprocessing/outputs/profiling)")
    return parser.parse_args()


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def profile(args: argparse.Namespace) -> int:
    if args.samples < 1:
        raise ValueError("--samples deve ser >= 1")
    if args.infer_size < 32:
        raise ValueError("--infer-size deve ser >= 32")

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"modelo não encontrado: {model_path}")

    print(f"[INFO] modelo: {model_path}")
    print(f"[INFO] câmera: device={args.device} {args.width}x{args.height} @ {args.fps} FPS")
    print(f"[INFO] inferência: {args.infer_size}x{args.infer_size}")
    print(f"[INFO] amostras: {args.samples} (+ {args.warmup} warmup)")

    model = YOLO(str(model_path))
    cap = RpicamCapture(args.device, args.width, args.height, args.fps)
    times = {stage: [] for stage in STAGES}
    processed = 0

    try:
        for _ in range(args.warmup):
            cap.read()

        while processed < args.samples:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            times["capture"].append(elapsed_ms(t0))
            if not ret or frame is None:
                raise RuntimeError("frame inválido durante o profiling")

            t1 = time.perf_counter()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            infer_frame = cv2.resize(
                frame_rgb, (args.infer_size, args.infer_size),
                interpolation=cv2.INTER_LINEAR,
            )
            times["preproc"].append(elapsed_ms(t1))

            t2 = time.perf_counter()
            results = model(infer_frame, verbose=False)
            times["infer"].append(elapsed_ms(t2))

            t3 = time.perf_counter()
            boxes = results[0].boxes
            box_data = [box.xyxy[0].tolist() for box in boxes]
            times["postproc"].append(elapsed_ms(t3))

            t4 = time.perf_counter()
            annotated = frame.copy()
            scale_x = args.width / args.infer_size
            scale_y = args.height / args.infer_size
            for coords in box_data:
                x1, y1, x2, y2 = coords
                cv2.rectangle(
                    annotated,
                    (int(x1 * scale_x), int(y1 * scale_y)),
                    (int(x2 * scale_x), int(y2 * scale_y)),
                    (0, 255, 0),
                    2,
                )
            times["osd"].append(elapsed_ms(t4))
            processed += 1

    finally:
        cap.release()

    means = {stage: float(np.mean(values)) for stage, values in times.items()}
    total = sum(means.values())
    if total <= 0:
        raise RuntimeError("tempo total inválido")

    print("\n=== PROFILING DO PIPELINE ===")
    print(f"{'Etapa':12s} {'Média (ms)':>12s} {'% do total':>12s}")
    print("-------------------------------------")
    for stage in STAGES:
        print(f"{stage:12s} {means[stage]:12.1f} {means[stage] / total * 100:11.1f}%")
    print("-------------------------------------")
    print(f"{'TOTAL':12s} {total:12.1f} {'100.0':>11s}%")
    print(f"FPS estimado: {1000.0 / total:.1f}")
    print(f"Amostras válidas: {processed}")
    return 0


def main() -> int:
    args = parse_args()
    log_file, log_path = open_run_log(Path(args.log_dir))
    stdout = Tee(sys.stdout, log_file)
    stderr = Tee(sys.stderr, log_file)
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            print(f"[INFO] log: {log_path}")
            return profile(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[ERRO] {exc}", file=stderr)
        return 1
    finally:
        log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
