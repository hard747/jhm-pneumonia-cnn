import pytest
from fastapi.testclient import TestClient
from io import BytesIO
from PIL import Image
import os
import sys

# Forzar base de datos en memoria RAM y añadir rutas
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from api.main import app 

client = TestClient(app)

def create_dummy_image():
    """Genera una imagen en memoria simulando una radiografía PNG real"""
    file = BytesIO()
    image = Image.new('RGB', size=(224, 224), color=(150, 150, 150))
    image.save(file, 'PNG')
    file.name = 'test_radiografia.png'
    file.seek(0)
    return file

def test_api_root_healthy():
    """INTEGRATION TEST: Verifica que la API responda al ping base"""
    response = client.get("/")
    assert response.status_code == 200

def test_full_prediction_flow():
    """
    INTEGRATION TEST: Simula el flujo completo de un diagnóstico médico.
    FastAPI -> PyTorch Inference -> SQLite Audit Log.
    """
    dummy_image = create_dummy_image()
    
    response = client.post(
        "/predict",
        files={"file": (dummy_image.name, dummy_image, "image/png")}
    )
    
    # Validar códigos e integridad de la respuesta del sistema integrado
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert data["prediction"] in ["NORMAL", "PNEUMONIA"]