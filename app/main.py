import base64
import io
import json
import time

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Response
from model import get_default_model_name, load_model
from PIL import Image
from preprocessing.preprocessor import CONFIG_DEFAULT, Preprocessor
from schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    Detection,
    HealthResponse,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
)

app = FastAPI(
    title="YOLO Inference API",
    description="API REST para inferência com YOLOv8 no Raspberry Pi 5",
    version="1.0.0",
)

# -- Métricas simples em memória -----------------------------
_metrics = {"total": 0, "success": 0, "total_ms": 0.0}
_preprocessor = Preprocessor(CONFIG_DEFAULT)


def log_event(event: str, **fields) -> None:
    """Emite um evento JSON de uma linha para o log do container."""
    payload = {"event": event, **fields}
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


def _decode_image(image_base64: str) -> np.ndarray:
    """Converte base64 → numpy array RGB."""
    raw = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(img)


def _load_image_from_request(request: PredictRequest) -> np.ndarray:
    """Lê a imagem a partir de Base64 ou URL pública sempre em RGB."""
    if not request.image_base64 and not request.image_url:
        raise HTTPException(
            status_code=422,
            detail="Forneça image_base64 ou image_url."
        )
    if request.image_base64:
        return _decode_image(request.image_base64)
    else:
        resp = httpx.get(request.image_url, timeout=15.0, follow_redirects=True)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        return np.array(img)


def _run_inference(image_np: np.ndarray, model_name: str, confidence: float) -> PredictResponse:
    model = load_model(model_name)
    frame_bgr = np.ascontiguousarray(image_np[:, :, ::-1])
    preproc_result = _preprocessor.process(frame_bgr)
    t0 = time.perf_counter()
    results = model(preproc_result.frame, conf=confidence, verbose=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    detections = []
    for result in results:
        for box in result.boxes:
            bbox_lb = box.xyxy[0].cpu().numpy().reshape(1, 4)
            bbox_orig = _preprocessor.adjust_boxes(bbox_lb, preproc_result)[0]
            cls_id = int(box.cls[0].item())
            conf_val = float(box.conf[0].item())
            detections.append(Detection(
                label=model.names[cls_id],
                confidence=round(conf_val, 4),
                bbox=[round(float(c), 2) for c in bbox_orig],
            ))
    h, w = image_np.shape[:2]
    return PredictResponse(
        detections=detections,
        inference_ms=round(elapsed_ms, 2),
        model_used=model_name,
        image_width=w,
        image_height=h,
    )


# -- Endpoints -----------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health_check():
    model_name = get_default_model_name()
    try:
        load_model(model_name)
        loaded = True
    except Exception:
        loaded = False
    log_event("health_check", status="ok", model_loaded=loaded, model_name=model_name)
    return HealthResponse(status="ok", model_loaded=loaded, model_name=model_name)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    _metrics["total"] += 1
    try:
        img = _load_image_from_request(request)
        result = _run_inference(img, request.model_name, request.confidence)
        _metrics["success"] += 1
        _metrics["total_ms"] += result.inference_ms
        log_event("prediction", endpoint="/predict", status="ok", detections=len(result.detections), inference_ms=result.inference_ms, model_name=result.model_used)
        return result
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/image", responses={200: {"content": {"image/jpeg": {}}}})
def predict_image(request: PredictRequest):
    """Executa a inferência e retorna a imagem anotada em JPEG com cores 100% calibradas em RGB."""
    _metrics["total"] += 1
    try:
        # Reutiliza o mesmo caminho pre-processado do endpoint JSON.
        img_rgb = _load_image_from_request(request)
        result = _run_inference(img_rgb, request.model_name, request.confidence)
        _metrics["success"] += 1
        _metrics["total_ms"] += result.inference_ms
        log_event(
            "prediction",
            endpoint="/predict/image",
            status="ok",
            detections=len(result.detections),
            inference_ms=result.inference_ms,
            model_name=result.model_used,
        )
        annotated = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        for detection in result.detections:
            x1, y1, x2, y2 = map(int, detection.bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"{detection.label} {detection.confidence:.0%}",
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )
        annotated_pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        buffer = io.BytesIO()
        annotated_pil.save(buffer, format="JPEG", quality=95)
        return Response(content=buffer.getvalue(), media_type="image/jpeg")
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest):
    t_total = time.perf_counter()
    results = []
    for img_b64 in request.images_base64:
        img = _decode_image(img_b64)
        results.append(_run_inference(img, request.model_name, request.confidence))
    total_ms = (time.perf_counter() - t_total) * 1000
    log_event("prediction_batch", endpoint="/predict/batch", status="ok", count=len(results), total_inference_ms=round(total_ms, 2), model_name=request.model_name)
    return BatchPredictResponse(results=results, total_inference_ms=round(total_ms, 2))


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    avg = (_metrics["total_ms"] / _metrics["success"] if _metrics["success"] > 0 else 0.0)
    return MetricsResponse(
        total_requests=_metrics["total"],
        successful_requests=_metrics["success"],
        avg_inference_ms=round(avg, 2),
    )