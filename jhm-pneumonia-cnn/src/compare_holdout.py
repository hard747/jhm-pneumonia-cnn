"""
Compara el modelo de produccion actual contra el modelo candidato generado
por src/tune.py, ambos medidos sobre holdout_final: el 60% de data/test/
que src/tune.py nunca uso (ni para entrenar, ni para elegir configuracion).
Es la unica comparacion justa antes de decidir si el candidato reemplaza
a produccion.

Uso: python -m src.compare_holdout
"""
import json
import os

import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.model import PneumoniaCNN

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(BASE_DIR, "data", "test")
SEED = 42

EVAL_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


@torch.no_grad()
def run(model, loader, device):
    model.eval()
    preds, labels_all = [], []
    for images, labels in loader:
        images = images.to(device)
        out = model(images)
        preds.extend(torch.argmax(out, dim=1).cpu().numpy().tolist())
        labels_all.extend(labels.numpy().tolist())
    return labels_all, preds


def metrics_for(checkpoint_path, holdout_loader, device):
    model = PneumoniaCNN().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    labels, preds = run(model, holdout_loader, device)
    return {
        "checkpoint": os.path.basename(checkpoint_path),
        "accuracy": round(float(np.mean(np.array(labels) == np.array(preds))), 4),
        "f1_macro": round(float(f1_score(labels, preds, average="macro", zero_division=0)), 4),
        "recall_pneumonia": round(float(recall_score(labels, preds, pos_label=1, zero_division=0)), 4),
        "recall_normal": round(float(recall_score(labels, preds, pos_label=0, zero_division=0)), 4),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plain_test = datasets.ImageFolder(root=TEST_DIR, transform=EVAL_TRANSFORM)
    targets = [s[1] for s in plain_test.samples]
    _, holdout_idx = train_test_split(
        list(range(len(plain_test))), test_size=0.6, stratify=targets, random_state=SEED
    )
    holdout_loader = DataLoader(Subset(plain_test, holdout_idx), batch_size=32, shuffle=False)
    print(f"Holdout final: {len(holdout_idx)} imagenes (nunca vistas por src/tune.py)")

    production_path = os.path.join(BASE_DIR, "api", "pneumonia_model.pth")
    candidate_path = os.path.join(BASE_DIR, "outputs", "candidate_model.pth")

    prod_metrics = metrics_for(production_path, holdout_loader, device)
    cand_metrics = metrics_for(candidate_path, holdout_loader, device)

    print("\n=== PRODUCCION (actual) ===")
    print(json.dumps(prod_metrics, indent=2, ensure_ascii=False))
    print("\n=== CANDIDATO (de src/tune.py) ===")
    print(json.dumps(cand_metrics, indent=2, ensure_ascii=False))

    winner = "candidato" if cand_metrics["f1_macro"] > prod_metrics["f1_macro"] else "produccion (sin cambios)"
    print(f"\n=== Decision (por F1 macro en holdout final): {winner} ===")

    with open(os.path.join(BASE_DIR, "outputs", "holdout_comparison.json"), "w", encoding="utf-8") as f:
        json.dump({"production": prod_metrics, "candidate": cand_metrics, "winner": winner}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
