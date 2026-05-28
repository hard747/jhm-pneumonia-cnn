import os

def configurar_dataset():
    print("=== INICIANDO CONFIGURACIÓN DE DATASET MÉDICO (JHM-7) ===")
    
    # 1. Definir la estructura de carpetas estándar para Deep Learning con PyTorch
    subpastas = [
        "data/train/NORMAL",
        "data/train/PNEUMONIA",
        "data/test/NORMAL",
        "data/test/PNEUMONIA"
    ]
    
    for pasta in subpastas:
        os.makedirs(pasta, exist_ok=True)
    print("✔ Estructura de directorios /data creada correctamente.")

    # 2. Archivos placeholder para validar el pipeline de carga de imágenes de prova
    for pasta in subpastas:
        with open(os.path.join(pasta, "placeholder.txt"), "w") as f:
            f.write("Amostra de imagem médica homologada para teste da CNN.")
            
    print("✔ Archivos de verificación generados en los directorios médicos.")
    print("=========================================================")
    print("Infraestructura de datos lista para el entrenamiento de la red neuronal.")

if __name__ == "__main__":
    configurar_dataset()