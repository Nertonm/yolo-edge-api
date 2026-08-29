#!/usr/bin/env python3
"""
stream/capture_frames.py
Captura frames do stream de câmera e os salva em dataset/raw/.
Reutiliza a lógica de captura do pipeline de streaming em tempo real..

Uso:
  python stream/capture_frames.py --source 0 --total 200 --interval 1.5
  python stream/capture_frames.py --source mjpeg --url http://localhost:5001/stream
"""
import argparse
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

OUTPUT_DIR = Path("dataset/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_snapshot(url: str):
    """Busca um único frame JPEG via HTTP - sem streaming, sem buffer acumulado."""
    with urllib.request.urlopen(url, timeout=5) as resp:
        data = resp.read()
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return frame is not None, frame


def parse_args():
    p = argparse.ArgumentParser(description="Captura de frames para dataset")
    p.add_argument("--source",   default="0",
                   help="Índice da câmera (0, 1) ou 'mjpeg' para stream HTTP")
    p.add_argument("--url",      default="http://localhost:5001/stream",
                   help="URL do stream MJPEG (usado quando --source mjpeg)")
    p.add_argument("--total",    type=int, default=200,
                   help="Total de frames a capturar")
    p.add_argument("--interval", type=float, default=1.5,
                   help="Intervalo entre capturas em segundos")
    p.add_argument("--width",    type=int, default=640)
    p.add_argument("--height",   type=int, default=480)
    p.add_argument("--manual", action="store_true",
                   help="Captura manual: pressione ENTER a cada frame, em vez de intervalo automático")
    p.add_argument("--snapshot-url", default="http://localhost:5001/snapshot",
                   help="URL do endpoint /snapshot (usado apenas com --source mjpeg --manual)")
    p.add_argument("--lap-threshold", type=float, default=15.0,
                   help="Variância mínima do Laplaciano p/ aceitar frame (nitidez). "
                        "Aula 4 usa 80; na cena real desta câmera a variância fica 15-20, "
                        "então o default aqui é 15 para não descartar tudo.")
    return p.parse_args()


class RpicamCapture:
    """
    Captura frames da câmera CSI via rpicam-vid (MJPEG), com interface
    compatível com cv2.VideoCapture (.read(), .isOpened(), .release()).

    cv2.VideoCapture() não suporta libcamera (câmeras CSI no Raspberry Pi
    OS Bookworm/Trixie) - só funciona com V4L2 nativo (webcams USB).
    """
    def __init__(self, device: int, width: int, height: int, fps: int = 15):
        cmd = [
            "rpicam-vid", "-t", "0", "-n", "--codec", "mjpeg",
            "--camera", str(device),
            "--width", str(width), "--height", str(height),
            "--framerate", str(fps),
            "-o", "-",
        ]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._buf = b""
        self._error = None

    def read(self):
        while True:
            start = self._buf.find(b"\xff\xd8")
            end = self._buf.find(b"\xff\xd9", start + 2) if start != -1 else -1
            if start != -1 and end != -1:
                jpg = self._buf[start:end + 2]
                self._buf = self._buf[end + 2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                return frame is not None, frame
            chunk = self._proc.stdout.read(4096)
            if not chunk:
                code = self._proc.poll()
                self._error = ("rpicam-vid encerrou" if code in (None, 0)
                               else f"rpicam-vid falhou com exit_code={code}")
                raise RuntimeError(self._error)
            self._buf += chunk

    def isOpened(self):
        return self._proc.poll() is None

    def release(self):
        self._proc.terminate()
        try:
            self._proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._proc.kill()


def open_capture(args):
    """Abre a câmera local (via rpicam-vid) ou o stream MJPEG configurado no pipeline de streaming."""
    if args.source == "mjpeg":
        # Lê o stream MJPEG servido pelo mjpeg_server.py
        cap = cv2.VideoCapture(args.url)
    else:
        # Câmera CSI/USB local: cv2.VideoCapture não suporta libcamera,
        # por isso usamos rpicam-vid via subprocesso (mesmo padrão do v1/v2/v3)
        cap = RpicamCapture(int(args.source), args.width, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a fonte: {args.source}")
    return cap


def is_sharp_enough(frame: np.ndarray, threshold: float = 15.0) -> bool:
    """
    Descarta frames borrados usando a variância do Laplaciano.
    Um frame nítido tem alta variância nas bordas detectadas.
    Default acompanha --lap-threshold da CLI (15); a cena real desta câmera
    tem variância 15-20, e o threshold 80 da apostila descartaria tudo aqui.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score >= threshold


def main():
    args = parse_args()
    cap  = open_capture(args)

    saved   = 0
    skipped = 0
    last_saved = 0.0

    print(f"[INFO] Iniciando captura: {args.total} frames | intervalo: {args.interval}s")
    print(f"[INFO] Salvando em: {OUTPUT_DIR.resolve()}")
    print("[INFO] Pressione Ctrl+C para encerrar antecipadamente.")

    try:
        while saved < args.total:
            if args.manual:
                input(f"  [{saved:>3}/{args.total}] Pressione ENTER para capturar...")
                if args.source == "mjpeg":
                    # Requisição HTTP avulsa: sempre o frame atual, sem fila
                    ret, frame = fetch_snapshot(args.snapshot_url)
                else:
                    # Câmera direta: descarta o buffer acumulado durante a espera
                    flush_until = time.time() + 0.5
                    while time.time() < flush_until:
                        cap.read()
                    ret, frame = cap.read()
            else:
                # Aguarda o intervalo desde a última captura com sleep em vez de
                # spin de re-leitura (que decodificava frames à toa).
                remaining = args.interval - (time.time() - last_saved)
                if remaining > 0:
                    time.sleep(remaining)
                    # descarta frames acumulados no buffer durante a espera
                    flush_until = time.time() + 0.3
                    while time.time() < flush_until:
                        cap.read()
                ret, frame = cap.read()

            if not ret:
                print("[AVISO] Frame inválido, tentando novamente...")
                time.sleep(0.1)
                continue

            # Descarta frames borrados automaticamente
            if not is_sharp_enough(frame, args.lap_threshold):
                skipped += 1
                if args.manual:
                    print("  [AVISO] Frame borrado, descartado - tente de novo.")
                continue

            # Nome com timestamp para evitar colisões
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]
            path = OUTPUT_DIR / f"frame_{ts}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1
            last_saved = time.time()
            end_char = "\n" if args.manual else "\r"
            print(f"  [{saved:>3}/{args.total}] Salvo: {path.name} ",
                  f"(descartados: {skipped})", end=end_char)

    except KeyboardInterrupt:
        print("\n[INFO] Captura interrompida pelo usuário.")
    except RuntimeError as e:
        print(f"\n[ERRO] {e}")
    finally:
        cap.release()
        print(f"\n[OK] {saved} frames salvos em {OUTPUT_DIR}")
        print(f"[OK] {skipped} frames borrados descartados automaticamente")


if __name__ == "__main__":
    main()