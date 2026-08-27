"""Testes da API YOLO Inference (Aula 3 — Edge AI).

Smoke tests (/health, /metrics), unit tests (_decode_image) e integration
tests (/predict, /predict/batch) com a imagem de referência zidane.jpg.
"""
import base64
import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

try:
    from app.main import _decode_image, app  # host: rodando da raiz do projeto
except ImportError:  # container (cwd=/app): módulos na raiz
    from main import _decode_image, app

ASSETS = Path(__file__).resolve().parent / "assets"

client = TestClient(app)


# ── SMOKE TESTS — saúde e observabilidade ─────────────────────────
class TestSmokeEndpoints:
    def test_health_endpoint_returns_fields(self):
        """Endpoint /health deve retornar status, model_loaded e model_name."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_name" in data

    def test_metrics_endpoint_accessible(self):
        """Endpoint /metrics deve estar acessível."""
        resp = client.get("/metrics")
        assert resp.status_code == 200


# ── UNIT TESTS — funções isoladas ─────────────────────────────────
class TestDecodeImage:
    def _make_b64_image(self, width=32, height=32, fmt="JPEG"):
        img = Image.new("RGB", (width, height), color=(128, 64, 192))
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        return base64.b64encode(buf.getvalue()).decode()

    def test_returns_numpy_array(self):
        result = _decode_image(self._make_b64_image())
        assert isinstance(result, np.ndarray)

    def test_correct_shape(self):
        result = _decode_image(self._make_b64_image(64, 48))
        assert result.shape == (48, 64, 3)

    def test_png_format(self):
        result = _decode_image(self._make_b64_image(fmt="PNG"))
        assert result.shape[2] == 3

    def test_invalid_base64_raises(self):
        with pytest.raises(Exception):
            _decode_image("dado_invalido_nao_e_base64")


# ── INTEGRATION TESTS — fluxo completo de inferência ─────────────
class TestPredictEndpoint:
    @pytest.fixture
    def zidane_b64(self):
        img_path = ASSETS / "zidane.jpg"
        return base64.b64encode(img_path.read_bytes()).decode()

    def test_predict_returns_200(self, zidane_b64):
        resp = client.post("/predict", json={
            "image_base64": zidane_b64,
            "confidence": 0.3,
        })
        assert resp.status_code == 200

    def test_predict_detects_at_least_one_object(self, zidane_b64):
        """A imagem zidane.jpg deve produzir ao menos 1 detecção com conf >= 0.3."""
        data = client.post("/predict", json={
            "image_base64": zidane_b64,
            "confidence": 0.3,
        }).json()
        assert len(data["detections"]) >= 1

    def test_predict_detections_is_list(self, zidane_b64):
        data = client.post("/predict", json={
            "image_base64": zidane_b64,
            "confidence": 0.3,
        }).json()
        assert isinstance(data["detections"], list)

    def test_predict_response_schema(self, zidane_b64):
        """Resposta deve conter todos os campos do schema PredictResponse."""
        data = client.post("/predict", json={
            "image_base64": zidane_b64,
            "confidence": 0.3,
        }).json()
        assert "detections" in data
        assert "inference_ms" in data
        assert "model_used" in data
        assert "image_width" in data
        assert "image_height" in data
        assert data["inference_ms"] > 0

    def test_predict_detection_fields(self, zidane_b64):
        """Cada detecção deve ter label, confidence e bbox válidos."""
        data = client.post("/predict", json={
            "image_base64": zidane_b64,
            "confidence": 0.3,
        }).json()
        for det in data["detections"]:
            assert isinstance(det["label"], str)
            assert 0.0 <= det["confidence"] <= 1.0
            assert len(det["bbox"]) == 4

    def test_predict_missing_input_returns_422(self):
        """Requisição sem imagem deve retornar HTTP 422."""
        resp = client.post("/predict", json={"confidence": 0.3})
        assert resp.status_code == 422


# ── BATCH ENDPOINT ────────────────────────────────────────────────
class TestBatchEndpoint:
    @pytest.fixture
    def two_images_b64(self):
        img_path = ASSETS / "zidane.jpg"
        b64 = base64.b64encode(img_path.read_bytes()).decode()
        return [b64, b64]  # mesma imagem duas vezes para simplificar

    def test_batch_returns_correct_count(self, two_images_b64):
        data = client.post("/predict/batch", json={
            "images_base64": two_images_b64,
            "confidence": 0.3,
        }).json()
        assert len(data["results"]) == 2

    def test_batch_total_ms_is_positive(self, two_images_b64):
        data = client.post("/predict/batch", json={
            "images_base64": two_images_b64,
            "confidence": 0.3,
        }).json()
        assert data["total_inference_ms"] > 0