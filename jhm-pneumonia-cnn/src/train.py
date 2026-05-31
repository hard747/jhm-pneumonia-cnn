import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# Importamos la arquitectura de tu modelo
from src.model import PneumoniaCNN

def train_model():
    # 1. Configurar Hardware (Usar GPU si está configurada en tu máquina, sino CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Iniciando entrenamiento en el dispositivo: {device}")

    # 2. Pipeline de transformaciones para leer las carpetas
    transform_pipeline = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # Escala de grises (in_channels=1)
        transforms.Resize((128, 128)),               # Resolución 128x128
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 3. Cargar las imágenes desde la estructura de tu carpeta 'data'
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(BASE_DIR, "data", "train")
    test_path = os.path.join(BASE_DIR, "data", "test")

    print(f"📁 Cargando datos desde: {train_path}")
    train_dataset = datasets.ImageFolder(root=train_path, transform=transform_pipeline)
    test_dataset = datasets.ImageFolder(root=test_path, transform=transform_pipeline)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"📊 Muestras encontradas -> Entrenamiento: {len(train_dataset)} | Validación: {len(test_dataset)}")
    print(f"🏷️ Clases detectadas automáticamente: {train_dataset.classes}") # ['NORMAL', 'PNEUMONIA']

    # 4. Inicializar red, función de pérdida y optimizador
    model = PneumoniaCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. Bucle de entrenamiento (Vamos a configurarlo a 5 épocas para empezar rápido)
    epochs = 5
    print("\n🏋️‍♂️ Comenzando el ajuste de pesos neuronales...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = (correct / total) * 100
        print(f"Época [{epoch+1}/{epochs}] -> Pérdida: {epoch_loss:.4f} | Precisión: {epoch_acc:.2f}%")

    # 6. 💾 GUARDAR LOS PESOS ENTRENADOS DEFINITIVOS
    # Los guardamos directamente en la carpeta 'api' para que 'main.py' los tome al reiniciar
    output_path = os.path.join(BASE_DIR, "api", "pneumonia_model.pth")
    torch.save(model.state_dict(), output_path)
    print(f"\n✅ ¡Entrenamiento completado! Pesos reales guardados en: {output_path}")

if __name__ == "__main__":
    train_model()