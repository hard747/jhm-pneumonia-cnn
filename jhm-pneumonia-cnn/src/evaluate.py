"""
Evalua un checkpoint entrenado contra el conjunto de test (nunca usado para
entrenar ni para seleccionar hiperparametros), reportando las metricas
clinicamente relevantes para un clasificador de neumonia por radiografia:

- Recall (sensibilidad) de la clase PNEUMONIA es la metrica primaria: un
  falso negativo (paciente con neumonia clasificado como sano) es el error
  mas costoso en un sistema de apoyo diagnostico, mucho mas que un falso
  positivo (que solo implica una revision manual adicional).
- Accuracy, precision, F1 y matriz de confusion se guardan tambien para dar
  el panorama completo.

Uso: python -m src.evaluate [--checkpoint api/pneumonia_model.pth]
"""
import argparse
import json
import os

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.model import PneumoniaCNN

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["NORMAL", "PNEUMONIA"]

TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())
    return all_labels, all_preds


def build_report(labels, preds) -> dict:
    acc = accuracy_score(labels, preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    return {
        "n_samples": len(labels),
        "accuracy": round(float(acc), 4),
        "per_class": {
            CLASSES[i]: {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }
            for i in range(2)
        },
        "confusion_matrix": {
            "labels": CLASSES,
            "matrix": cm.tolist(),
        },
        "primary_metric": {
            "name": "recall_PNEUMONIA",
            "value": round(float(recall[1]), 4),
            "why": "en deteccion de neumonia, un falso negativo (recall bajo) es el error clinicamente mas caro",
        },
    }


def plot_confusion_matrix(cm, out_path):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASSES)
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusion -- test set")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                     color="white" if cm[i][j] > cm.max() / 2 else "black", fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join(BASE_DIR, "api", "pneumonia_model.pth"))
    parser.add_argument("--out", default=os.path.join(BASE_DIR, "outputs", "evaluation_report.json"))
    parser.add_argument("--figure", default=os.path.join(BASE_DIR, "reports", "figures", "confusion_matrix.png"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    test_dataset = datasets.ImageFolder(root=os.path.join(BASE_DIR, "data", "test"), transform=TRANSFORM)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"Test set: {len(test_dataset)} imagenes -- clases {test_dataset.classes}")

    model = PneumoniaCNN().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))

    labels, preds = evaluate(model, test_loader, device)
    report = build_report(labels, preds)
    report["checkpoint"] = os.path.relpath(args.checkpoint, BASE_DIR)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    os.makedirs(os.path.dirname(args.figure), exist_ok=True)
    import numpy as np
    plot_confusion_matrix(np.array(report["confusion_matrix"]["matrix"]), args.figure)

    print("\n=== REPORTE DE EVALUACION (test set, nunca visto en entrenamiento) ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nGuardado en {args.out} y {args.figure}")

    print("\n--- classification_report (sklearn) ---")
    print(classification_report(labels, preds, target_names=CLASSES, zero_division=0))


if __name__ == "__main__":
    main()
