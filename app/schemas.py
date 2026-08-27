from pydantic import BaseModel, Field
from typing import List, Optional


class PredictRequest(BaseModel):
    image_base64: Optional[str] = Field(
        None,
        description="Imagem PNG/JPG codificada em base64"
    )
    image_url: Optional[str] = Field(
        None,
        description="URL pública acessível a partir do container"
    )
    confidence: float = Field(0.25, ge=0.0, le=1.0,
        description="Limiar mínimo de confiança (0–1)")
    model_name: str = Field("yolov8n.pt",
        description="Nome do arquivo de pesos dentro de /app/models/")


class Detection(BaseModel):
    label: str
    confidence: float
    bbox: List[float]   # [x1, y1, x2, y2] em pixels


class PredictResponse(BaseModel):
    detections: List[Detection]
    inference_ms: float
    model_used: str
    image_width: int
    image_height: int


class BatchPredictRequest(BaseModel):
    images_base64: List[str]
    confidence: float = 0.25
    model_name: str = "yolov8n.pt"


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    total_inference_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str


class MetricsResponse(BaseModel):
    total_requests: int
    successful_requests: int
    avg_inference_ms: float