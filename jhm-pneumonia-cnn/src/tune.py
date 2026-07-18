"""
Busqueda de hiperparametros para PneumoniaCNN -- version 2.

La v1 de este script separaba un 15% de data/train/ como validacion. Esa
validacion dio resultados excelentes (98% accuracy) para una configuracion
que luego resulto ser PEOR que el baseline original al medirla contra
data/test/ real (71.6% vs 80.9% de accuracy, recall NORMAL cayendo de 53%
a 25%). La causa: data/train/ y data/test/ en este dataset publico (Kaggle
Chest X-Ray Images) vienen de lotes/fuentes distintas con caracteristicas
de imagen algo diferentes -- un train/test distribution shift documentado
por varios usuarios del dataset. Validar con una porcion de train/ no es
representativo de como el modelo se comporta en test/.

Fix: en vez de separar la validacion desde train/, se separa desde el
propio test/ (que sabemos que representa la distribucion real de uso).
test/ (624 imagenes) se divide en:
  - tuning_val (40%, ~250 imgs): usado SOLO para elegir la configuracion
  - holdout_final (60%, ~374 imgs): jamas visto durante la busqueda,
    usado unicamente para el reporte final en evaluate.py

train/ se usa completo (sin recortar) para entrenar cada configuracion.

Configuraciones (mas conservadoras que v1, dado lo aprendido):
  - baseline_5ep:      exactamente la receta original de train.py (lr=0.001,
                       5 epocas) -- sirve de control, debe reproducir ~80%
  - augmentation_5ep:  misma receta + augmentation (regularizacion, puede
                       generalizar mejor al shift train/test)
  - augmentation_10ep: augmentation con mas epocas

Uso: python -m src.tune
"""
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.model import PneumoniaCNN

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
TEST_DIR = os.path.join(BASE_DIR, "data", "test")
BATCH_SIZE = 32
SEED = 42

EVAL_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

AUG_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=7),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

CONFIGS = [
    {"name": "baseline_5ep",     "lr": 0.001, "augmentation": False, "epochs": 5},
    {"name": "augmentation_5ep", "lr": 0.001, "augmentation": True,  "epochs": 5},
    {"name": "augmentation_10ep","lr": 0.001, "augmentation": True,  "epochs": 10},
]


def make_test_split():
    plain_test = datasets.ImageFolder(root=TEST_DIR, transform=EVAL_TRANSFORM)
    targets = [s[1] for s in plain_test.samples]
    tuning_idx, holdout_idx = train_test_split(
        list(range(len(plain_test))), test_size=0.6, stratify=targets, random_state=SEED
    )
    return plain_test, tuning_idx, holdout_idx


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def eval_split(model, loader, device):
    model.eval()
    preds, labels_all = [], []
    for images, labels in loader:
        images = images.to(device)
        out = model(images)
        preds.extend(torch.argmax(out, dim=1).cpu().numpy().tolist())
        labels_all.extend(labels.numpy().tolist())
    return labels_all, preds


def run_config(cfg, val_loader, device):
    transform = AUG_TRANSFORM if cfg["augmentation"] else EVAL_TRANSFORM
    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = PneumoniaCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss()

    t0 = time.time()
    for _ in range(cfg["epochs"]):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
    elapsed = time.time() - t0

    labels, preds = eval_split(model, val_loader, device)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    recall_pneumonia = recall_score(labels, preds, pos_label=1, zero_division=0)
    recall_normal = recall_score(labels, preds, pos_label=0, zero_division=0)
    accuracy = float(np.mean(np.array(labels) == np.array(preds)))

    metrics = {
        "config": cfg["name"],
        "lr": cfg["lr"],
        "augmentation": cfg["augmentation"],
        "epochs": cfg["epochs"],
        "tuning_val_accuracy": round(accuracy, 4),
        "tuning_val_f1_macro": round(float(f1_macro), 4),
        "tuning_val_recall_pneumonia": round(float(recall_pneumonia), 4),
        "tuning_val_recall_normal": round(float(recall_normal), 4),
        "train_seconds": round(elapsed, 1),
    }
    return model, metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    plain_test, tuning_idx, holdout_idx = make_test_split()
    print(f"test/ dividido -- tuning_val: {len(tuning_idx)} | holdout_final (jamas usado aqui): {len(holdout_idx)}")

    tuning_loader = DataLoader(Subset(plain_test, tuning_idx), batch_size=BATCH_SIZE, shuffle=False)

    results = []
    best_model = None
    best_metrics = None
    for cfg in CONFIGS:
        print(f"\n--- Config: {cfg['name']} ({cfg}) ---")
        model, metrics = run_config(cfg, tuning_loader, device)
        print(metrics)
        results.append(metrics)
        if best_metrics is None or metrics["tuning_val_f1_macro"] > best_metrics["tuning_val_f1_macro"]:
            best_model, best_metrics = model, metrics

    out_csv = os.path.join(BASE_DIR, "outputs", "hyperparameter_search.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nComparacion guardada en {out_csv}")
    print(f"\n=== Mejor config por F1 macro en tuning_val (porcion de test/, nunca en holdout): "
          f"{best_metrics['config']} ===")

    # Guarda el mejor modelo encontrado en un archivo separado -- NO reemplaza
    # el modelo de produccion todavia. Eso se decide comparando contra
    # holdout_idx (el 60% de test/ jamas visto ni en tuning ni en train).
    candidate_path = os.path.join(BASE_DIR, "outputs", "candidate_model.pth")
    torch.save(best_model.state_dict(), candidate_path)
    print(f"Modelo candidato guardado en {candidate_path} (no reemplaza produccion todavia)")

    with open(os.path.join(BASE_DIR, "outputs", "best_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "best_config": best_metrics,
            "search_results": results,
            "holdout_idx_count": len(holdout_idx),
            "note": "candidate_model.pth debe compararse contra el modelo de produccion en el holdout final antes de reemplazarlo",
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
