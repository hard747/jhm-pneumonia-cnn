from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import random
import time

app = FastAPI(
    title="JHM Pneumonia Detection API",
    description="API de producción para el diagnóstico de neumonía mediante Redes Neuronales",
    version="1.0.0"
)

# 🛡️ Concepto Industrial #9: Configuración de CORS (Cross-Origin Resource Sharing)
# Esto le dice a Python que permita conexiones desde tu Frontend de desarrollo y producción
origins = [
    "http://localhost:5173",    # Tu React local (Vite)
    "http://127.0.0.1:5173",
    # Aquí agregaremos la URL de Vercel cuando la tengamos en producción
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (POST, GET, etc.)
    allow_headers=["*"],  # Permite todos los encabezados de seguridad
)

@app.get("/")
def read_root():
    """Ruta de control para verificar que el backend esté vivo (Health Check)"""
    return {"status": "online", "model": "ResNet50-CNN", "version": "1.0.0"}

# 🧠 Concepto Industrial #10: Endpoint receptor de archivos binarios
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint que recibe la radiografía del frontend, la procesa 
    y retorna el diagnóstico de la red neuronal.
    """
    try:
        # Leer el archivo binario enviado por el navegador
        contents = await file.read()
        
        # [PROCESO INDUSTRIAL REAL]: Aquí conectaríamos con tu archivo `src/model.py`
        # para hacer: model.predict(contents). 
        # Por ahora, simularemos el tiempo de procesamiento de la red neuronal:
        time.sleep(1.5) # Simula 1.5 segundos de procesamiento pesado
        
        # Simulación del resultado del modelo de IA (Lo reemplazaremos con tu peso real)
        prediction = random.choice(["PNEUMONIA", "NORMAL"])
        confidence = round(random.uniform(0.85, 0.99), 4) # Confianza entre 85% y 99%

        return {
            "filename": file.filename,
            "prediction": prediction,
            "confidence": confidence,
            "status": "success"
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}