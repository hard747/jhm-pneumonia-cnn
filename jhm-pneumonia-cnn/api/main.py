import os
import io
import asyncio
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.model import PneumoniaCNN
from api.database import engine, Base, SessionLocal
import api.models
from api.models import AuditoriaDiagnostico 

# --- Modelo cargado una sola vez al iniciar ---
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "pneumonia_model.pth")
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

try:
    import api.models # Forzamos que se cargue la estructura justo aquí
    Base.metadata.create_all(bind=engine)
    print("[DB] ¡Intento de creación de tablas ejecutado con éxito!")
except Exception as e:
    print(f"[DB] Advertencia al crear tablas: {e}")

app = FastAPI(title="JHM Pneumonia Detection API", version="2.0.0")

# 🔌 CONFIGURACIÓN DE CORS EN PRODUCCIÓN CORREGIDA
# Añadimos tu URL de Vercel para romper el bloqueo de seguridad en internet
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", 
        "http://127.0.0.1:5173", 
        "http://localhost:5174", 
        "http://127.0.0.1:5174",
        "https://jhm-pneumonia-cnn.vercel.app"  # 🚀 ¡Permiso concedido a Vercel en la nube!
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _run_inference(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _transform(image).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        outputs = _cnn(tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, idx = torch.max(probs, 1)
    return _CLASSES[idx.item()], round(confidence.item() * 100, 2)


def _save_audit(filename: str, prediction: str, confidence: float):
    try:
        db = SessionLocal()
        try:
            db.add(AuditoriaDiagnostico(
                nombre_imagen=filename,
                resultado=prediction,
                confianza=confidence,
            ))
            db.commit()
        finally:
            db.close()
    except Exception as e:
        print(f"[DB] Error guardando auditoría: {e}")


@app.get("/")
def read_root():
    return {"status": "online", "model": "JHM-6-CNN", "version": "2.0.0", "device": str(_DEVICE)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    loop = asyncio.get_event_loop()
    prediction, confidence_pct = await loop.run_in_executor(None, _run_inference, contents)

    await loop.run_in_executor(None, _save_audit, file.filename, prediction, confidence_pct)

    return {
        "filename": file.filename,
        "prediction": prediction,
        "confidence": confidence_pct,
        "status": "success",
    }