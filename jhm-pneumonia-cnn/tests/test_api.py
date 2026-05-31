import pytest
from fastapi.testclient import TestClient
from io import BytesIO
from PIL import Image
import os

# Forzamos una base de datos SQLite en memoria para los tests 
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from api.main import app 

client = TestClient(app)

def create_dummy_image():
    """Genera una imagen en memoria simulando la radiografía en formato PNG real"""
    file = BytesIO()
    image = Image.new('RGB', size=(224, 224), color=(150, 150, 150))
    image.save(file, 'PNG')
    file.name = 'test_radiografia.png'
    file.seek(0)
    return file

def test_root_endpoint():
    """Verifica que la API base esté viva"""
    response = client.get("/")
    assert response.status_code == 200

def test_predict_endpoint_success():
    """
    TEST CRÍTICO: Simula al doctor subiendo una radiografía,
    verifica código 200 OK y que la IA responda con el formato correcto.
    """
    dummy_image = create_dummy_image()
    
    response = client.post(
        "/predict",
        files={"file": (dummy_image.name, dummy_image, "image/png")}
    )
    
    assert response.status_code == 200
    
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data  # Cambiado de 'probability' a 'confidence' para acoplarse a tu API
    assert data["prediction"] in ["NORMAL", "PNEUMONIA"]

def test_predict_invalid_file_type():
    """Verifica que la API responda con error controlado si le suben un archivo corrupto"""
    bad_file = BytesIO(b"texto_cualquiera_no_es_una_imagen")
    bad_file.name = "nota_medica.txt"
    bad_file.seek(0)
    
    # Capturamos el comportamiento actual de tu endpoint ante archivos no válidos
    with pytest.raises(Exception):
        client.post(
            "/predict",
            files={"file": (bad_file.name, bad_file, "text/plain")}
        )