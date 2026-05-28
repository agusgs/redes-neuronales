"""Analiza los seeds problemáticos del último experimento canonical.

Re-corre exactamente los seeds 8 y 9 (los outliers) y muestra qué clases
fallaron específicamente, para entender el patrón de inestabilidad.
"""
from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data import CANONICAL_DIR, LSA16_NAMES, list_samples  # noqa: E402
from src.experiment import run_single  # noqa: E402
from src.model import get_model_spec  # noqa: E402
from src.train import TrainConfig  # noqa: E402


def analyze_seed(samples, seed: int) -> None:
    print(f"\n{'='*64}")
    print(f"  Seed = {seed}")
    print(f"{'='*64}")
    spec = get_model_spec("lenet")
    result = run_single(samples, seed=seed, test_size=0.10,
                        config=TrainConfig(epochs=20),
                        device=torch.device("cpu"),
                        spec=spec, verbose=False)
    print(f"Test accuracy: {result.test_acc*100:.2f}%")

    labels = np.array(result.labels)
    preds = np.array(result.preds)

    # Per-class accuracy
    print(f"\nAccuracy por clase:")
    for cls in range(16):
        mask = labels == cls
        if mask.sum() == 0:
            continue
        cls_acc = (preds[mask] == cls).mean()
        n_correct = (preds[mask] == cls).sum()
        n_total = mask.sum()
        marker = " ← FALLA" if cls_acc < 0.8 else ""
        print(f"  {cls:2d} {LSA16_NAMES[cls]:12s}: {cls_acc*100:6.2f}%  ({n_correct}/{n_total}){marker}")

    # Qué clases se confundieron con cuáles
    errors = labels != preds
    if errors.any():
        print(f"\nConfusiones (label → predicción):")
        for l, p in zip(labels[errors], preds[errors]):
            print(f"  {LSA16_NAMES[l]} ({l}) → {LSA16_NAMES[p]} ({p})")


def main() -> None:
    samples = list_samples(CANONICAL_DIR)
    # Seeds buenos para comparar + seeds malos
    for seed in [0, 8, 9]:
        analyze_seed(samples, seed)


if __name__ == "__main__":
    main()
