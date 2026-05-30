FROM python:3.8-slim

WORKDIR /app

# 1. Instalar dependencias del sistema necesarias para OpenCV y PyTorch en Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    glib2.0-0 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Copiar el archivo de requerimientos
COPY requirements.txt .

# 3. Forzar a pip a esperar lo que sea necesario para archivos pesados
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=1000 torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# 4. Copiar el resto del código del proyecto
COPY . .

# 5. Arrancar el script
CMD ["python", "src/model.py"]