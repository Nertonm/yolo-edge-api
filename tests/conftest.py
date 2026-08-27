"""Resolve os imports de app.main (e de schemas/model dentro de main) ao
rodar pytest a partir da raiz do projeto yolo-edge-api."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))