import torch
import torch.nn as nn
import torch.nn.functional as F

class PneumoniaCNN(nn.Module):
    def __init__(self):
        super(PneumoniaCNN, self).__init__()
        
        # Bloque 1 de Convolución: Procesa la imagen inicial (escala de grises)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Bloque 2 de Convolución: Extrae características complejas
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Bloque 3 de Convolución: Profundiza en los patrones de la neumonía
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Capa de reducción de tamaño (Max Pooling)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Capas de Clasificación Final (Densas)
        # Asumiendo una resolución de entrada estándar (ej: 128x128) -> tras 3 reducciones queda en 16x16
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2) # 2 salidas: NORMAL o PNEUMONIA

    def forward(self, x):
        # Flujo de la imagen a través de las capas de la IA
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Aplanar la matriz para pasarla a las capas de clasificación
        x = x.view(-1, 64 * 16 * 16)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    # Prueba rápida local para verificar que la arquitectura compile sin errores
    model = PneumoniaCNN()
    print("=== ARQUITECTURA JHM-6 CONFIGURADA ===")
    print(model)
    print("=======================================")