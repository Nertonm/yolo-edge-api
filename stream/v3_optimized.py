#!/usr/bin/env python3
"""V3 otimizada: threading, frame skip, OSD e gravação AVI."""
import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

# A wheel OpenCV/Qt instalado no Pi fornece XCB, não o plugin Wayland.
# Em sessão gráfica com DISPLAY, force XCB salvo override explícito.
if os.environ.get("DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
    os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from preprocessing.preprocessor import PreprocessConfig, Preprocessor

_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class OptimizedCamera:
    """Captura MJPEG em thread e mantém somente o frame mais recente."""

    def __init__(self, device, width, height, fps=30):
        self._cmd = [
            "rpicam-vid", "-t", "0", "-n", "--codec", "mjpeg",
            "--camera", str(device), "--width", str(width),
            "--height", str(height), "--framerate", str(fps), "-o", "-",
        ]
        self._proc = None
        self._capture_error = None
        self._buffer = queue.Queue(maxsize=1)
        self._running = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self.frames_in = 0
        self.frames_out = 0
        self.frames_dropped = 0

    def start(self):
        self._running.set()
        self._proc = subprocess.Popen(
            self._cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        self._thread.start()
        print(f"[Camera] rpicam-vid pid={self._proc.pid}, buffer=1")

    def _loop(self):
        raw = b""
        while self._running.is_set():
            chunk = self._proc.stdout.read(4096)
            if not chunk:
                code = self._proc.poll()
                self._capture_error = (
                    "rpicam-vid encerrou"
                    if code in (None, 0)
                    else f"rpicam-vid falhou com exit_code={code}"
                )
                self._running.clear()
                return
            raw += chunk
            end = raw.rfind(b"\xff\xd9")
            if end < 0:
                continue
            start = raw.rfind(b"\xff\xd8", 0, end)
            if start < 0:
                continue
            frame = cv2.imdecode(
                np.frombuffer(raw[start:end + 2], dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            raw = raw[end + 2:]
            if frame is None:
                continue
            self.frames_in += 1
            if self._buffer.full():
                self._buffer.get_nowait()
                self.frames_dropped += 1
            self._buffer.put(frame)

    def read(self, timeout=2.0):
        try:
            frame = self._buffer.get(timeout=timeout)
            self.frames_out += 1
            return frame
        except queue.Empty:
            if not self._running.is_set():
                raise RuntimeError(self._capture_error or "captura encerrada")
            return None

    def stop(self):
        self._running.clear()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread.is_alive():
            self._thread.join(timeout=2)
        print(
            f"[Camera] encerrada: capturados={self.frames_in}, "
            f"entregues={self.frames_out}, descartados={self.frames_dropped}"
        )


class RealtimeDetector:
    """Executa YOLO a cada N frames e mantém o último OSD/detecção."""

    def __init__(self, model_path, conf, infer_every, infer_size):
        if infer_every < 1:
            raise ValueError("infer_every deve ser >= 1")
        if infer_size < 32:
            raise ValueError("infer_size deve ser >= 32")
        print(f"[YOLO] modelo={model_path}, infer_every={infer_every}, size={infer_size}")
        self.model = YOLO(model_path)
        self.conf = conf
        self.infer_every = infer_every
        self.infer_size = infer_size
        self.preprocessor = Preprocessor(
            PreprocessConfig(infer_size=infer_size)
        )
        self.frame_idx = 0
        self.last_boxes = []
        self.last_infer_ms = 0.0
        self.fps_window = deque(maxlen=30)
        self.last_tick = time.perf_counter()

    def process(self, frame):
        self.frame_idx += 1
        now = time.perf_counter()
        self.fps_window.append(now - self.last_tick)
        self.last_tick = now

        infer_frame = self.frame_idx % self.infer_every == 0
        if infer_frame:
            preproc_result = self.preprocessor.process(frame)
            t0 = time.perf_counter()
            results = self.model(
                preproc_result.frame, conf=self.conf, verbose=False
            )
            self.last_infer_ms = (time.perf_counter() - t0) * 1000
            self.last_boxes = []
            for result in results:
                for box in result.boxes:
                    bbox_lb = box.xyxy[0].cpu().numpy().reshape(1, 4)
                    x1, y1, x2, y2 = self.preprocessor.adjust_boxes(
                        bbox_lb, preproc_result
                    )[0]
                    self.last_boxes.append((
                        self.model.names[int(box.cls[0])],
                        float(box.conf[0]),
                        int(x1), int(y1), int(x2), int(y2),
                    ))

        output = frame.copy()
        for label, confidence, x1, y1, x2, y2 in self.last_boxes:
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            caption = f"{label} {confidence:.0%}"
            cv2.putText(output, caption, (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        fps = len(self.fps_window) / sum(self.fps_window) if self.fps_window else 0
        for i, line in enumerate((
            f"FPS: {fps:.1f}",
            f"Infer: {self.last_infer_ms:.0f}ms",
            f"Det: {len(self.last_boxes)}",
            f"Frame: {self.frame_idx}",
        )):
            cv2.putText(output, line, (10, 28 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 255, 255) if infer_frame else (200, 200, 200), 2)
        return output


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--model", default="models/yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--infer-every", type=int, default=3)
    p.add_argument("--infer-size", type=int, default=320)
    p.add_argument("--frames", type=int, default=0,
                   help="Frames; 0 mantém execução até q/Ctrl+C")
    p.add_argument("--output", default=None, help="Arquivo AVI anotado")
    p.add_argument("--no-display", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    camera = OptimizedCamera(args.device, args.width, args.height, args.fps)
    detector = RealtimeDetector(args.model, args.conf, args.infer_every, args.infer_size)
    writer = None
    if args.output:
        writer = cv2.VideoWriter(
            args.output, cv2.VideoWriter_fourcc(*"XVID"), args.fps,
            (args.width, args.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"não foi possível abrir saída AVI: {args.output}")
        print(f"[INFO] gravando: {args.output}")

    processed = 0
    camera.start()
    time.sleep(0.5)
    t_start = time.perf_counter()
    try:
        while args.frames == 0 or processed < args.frames:
            frame = camera.read()
            if frame is None:
                print("[AVISO] timeout na leitura")
                continue
            annotated = detector.process(frame)
            processed += 1
            if writer:
                writer.write(annotated)
            if not args.no_display:
                cv2.imshow("YOLO V3 : q encerra", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.stop()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    print("\n" + "=" * 58)
    print("RELATÓRIO : V3 frame skip + OSD")
    print("=" * 58)
    total_time = time.perf_counter() - t_start
    print(f"  Frames processados   : {processed}")
    print(f"  Tempo total          : {total_time:.1f} s")
    print(f"  FPS médio sustentado : {processed / total_time:.1f} FPS")
    print(f"  Frames capturados    : {camera.frames_in}")
    print(f"  Frames descartados   : {camera.frames_dropped}")
    print(f"  Inferências          : {detector.frame_idx // detector.infer_every}")
    print("=" * 58)


if __name__ == "__main__":
    main()
