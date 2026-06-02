import os
import io
import time
import asyncio
import logging
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

# Observabilidad
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_fastapi_instrumentator import Instrumentator

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.model import PneumoniaCNN
from api.database import engine, Base, SessionLocal
from api.models import AuditoriaDiagnostico, ModelRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- OpenTelemetry setup (envia trazas a Grafana Cloud si OTEL_ENDPOINT esta configurado) ---
_tracer_provider = TracerProvider()
_otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
if _otlp_endpoint:
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        _tracer_provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=_otlp_endpoint))
        )
        logger.info(f"[OTEL] Trazas enviando a {_otlp_endpoint}")
    except Exception as e:
        logger.warning(f"[OTEL] No se pudo iniciar exporter: {e}")
trace.set_tracer_provider(_tracer_provider)
tracer = trace.get_tracer("jhm-pneumonia-api")

# --- Rate limiter (10 predicciones/minuto por IP) ---
_TESTING = os.getenv("TESTING", "false").lower() == "true"
limiter = Limiter(key_func=get_remote_address, enabled=not _TESTING)

# --- Modelo CNN cargado una sola vez al iniciar ---
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "pneumonia_model.pth")
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")

torch.set_num_threads(1)
_cnn = PneumoniaCNN()
_cnn.load_state_dict(torch.load(_MODEL_PATH, map_location=_DEVICE, weights_only=True))
_cnn.eval()
_cnn.to(_DEVICE)

_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])
_CLASSES = ["NORMAL", "PNEUMONIA"]

# --- Tablas en DB ---
try:
    Base.metadata.create_all(bind=engine)
    logger.info("[DB] Tablas verificadas/creadas")
except Exception as e:
    logger.warning(f"[DB] Advertencia al crear tablas: {e}")

# --- App FastAPI ---
app = FastAPI(
    title="JHM Pneumonia Detection API",
    description="CNN de deteccion de neumonia con observabilidad completa",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        os.getenv("FRONTEND_URL", ""),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus: expone /metrics automaticamente
Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_instrument_requests_inprogress=True,
    inprogress_labels=True,
).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

# OpenTelemetry: instrumenta todas las rutas automaticamente
FastAPIInstrumentor.instrument_app(app)


# ─── Helpers ────────────────────────────────────────────────────────────────

def _run_inference(image_bytes: bytes) -> tuple[str, float]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _transform(image).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        outputs = _cnn(tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, idx = torch.max(probs, 1)
    return _CLASSES[idx.item()], round(confidence.item() * 100, 2)


def _save_audit(filename: str, prediction: str, confidence: float, latency_ms: float):
    try:
        db = SessionLocal()
        try:
            db.add(AuditoriaDiagnostico(
                nombre_imagen=filename,
                resultado=prediction,
                confianza=confidence,
                latencia_ms=latency_ms,
                model_version=_MODEL_VERSION,
            ))
            db.commit()
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"[DB] Error guardando auditoria: {e}")


# ─── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/", tags=["Sistema"])
def read_root():
    return {
        "status": "online",
        "model": "JHM-6-CNN",
        "version": "2.0.0",
        "model_version": _MODEL_VERSION,
        "device": str(_DEVICE),
    }


@app.get("/health", tags=["Sistema"])
def health_check():
    """Health check detallado para load balancers y uptime monitors."""
    db_ok = False
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    status = "healthy" if db_ok else "degraded"
    return JSONResponse(
        status_code=200 if db_ok else 503,
        content={
            "status": status,
            "checks": {
                "model_loaded": True,
                "database": "connected" if db_ok else "disconnected",
                "device": str(_DEVICE),
                "model_version": _MODEL_VERSION,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


@app.get("/model/info", tags=["MLOps"])
def model_info():
    """Info del modelo activo en produccion (MLOps - Model Registry)."""
    try:
        db = SessionLocal()
        active = db.query(ModelRegistry).filter(ModelRegistry.is_active == True).first()
        db.close()
        if active:
            return {
                "active_model": {
                    "version": active.version,
                    "accuracy": active.accuracy,
                    "deployed_at": active.deployed_at.isoformat(),
                    "description": active.description,
                }
            }
    except Exception:
        pass
    return {"active_model": {"version": _MODEL_VERSION, "accuracy": 0.974, "source": "env"}}


@app.post("/predict", tags=["Diagnostico"])
@limiter.limit("10/minute")
async def predict(request: Request, file: UploadFile = File(...)):
    """
    Diagnostico de neumonia por radiografia.
    Rate limit: 10 predicciones/minuto por IP.
    """
    with tracer.start_as_current_span("predict") as span:
        t_start = time.perf_counter()
        span.set_attribute("file.name", file.filename or "unknown")

        contents = await file.read()

        loop = asyncio.get_event_loop()
        prediction, confidence_pct = await loop.run_in_executor(None, _run_inference, contents)

        latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

        span.set_attribute("prediction.result", prediction)
        span.set_attribute("prediction.confidence", confidence_pct)
        span.set_attribute("prediction.latency_ms", latency_ms)

        logger.info(f"PREDICT | {file.filename} | {prediction} | {confidence_pct}% | {latency_ms}ms")

        await loop.run_in_executor(
            None, _save_audit, file.filename, prediction, confidence_pct, latency_ms
        )

        return {
            "filename": file.filename,
            "prediction": prediction,
            "confidence": confidence_pct,
            "latency_ms": latency_ms,
            "model_version": _MODEL_VERSION,
            "status": "success",
        }
