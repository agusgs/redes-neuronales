"""Experimentos LOSO (Leave-One-Subject-Out) sobre CNNs clásicas.

Evalúa la arquitectura seleccionada entrenando sobre 9 sujetos y testeando sobre 1,
repitiendo el proceso para los 10 sujetos. Esto evita el Data Leakage y mide la
verdadera capacidad de generalización del modelo.

Ejemplos:
    # Correr LOSO sobre la versión canónica
    python experiments/03_loso_resnet.py --mode canonical

    # Correr LOSO sobre la versión raw
    python experiments/03_loso_resnet.py --mode raw
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import CANONICAL_DIR, RAW_DIR, SEGMENTED_DIR, list_samples, loso_split
from src.experiment import _make_loaders
from src.model import MODEL_REGISTRY, get_model_spec
from src.train import TrainConfig, evaluate, train_model

OUT_DIR = PROJECT_ROOT / "outputs" / "03_loso_resnet"
RESULTS_JSONL = OUT_DIR / "results.jsonl"


def _append_jsonl(record: dict) -> None:
    RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def run_loso(
    image_dir: str,
    name: str,
    model: str = "resnet18",
    config: TrainConfig | None = None,
    device: torch.device | None = None,
    augment: bool = False,
    note: str = "",
) -> None:
    spec = get_model_spec(model)
    if config is None:
        config = TrainConfig(epochs=spec.default_epochs, lr=spec.default_lr)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = list_samples(image_dir)
    if not samples:
        raise FileNotFoundError(f"No PNG files found in {image_dir!r}")

    # Verificar cuántos sujetos hay
    subjects = sorted(list(set(s.subject for s in samples)))
    print(f"\n============================================================")
    print(f" Iniciando LOSO Cross-Validation")
    print(f" Dataset: {name}")
    print(f" Modelo : {model}")
    print(f" Sujetos: {len(subjects)} encontrados {subjects}")
    print(f" Augment: {augment}")
    print(f" Device : {device}")
    print(f"============================================================\n")

    fold_results = []
    
    t0 = time.time()

    for test_subject in subjects:
        print(f"--- Fold {test_subject}/{len(subjects)} (Test Sujeto {test_subject}) ---")
        
        # Split estricto por sujeto
        train_samples, test_samples = loso_split(samples, test_subject)
        print(f"    Train: {len(train_samples)} imágenes | Test: {len(test_samples)} imágenes")
        
        train_loader, test_loader = _make_loaders(
            train_samples, test_samples,
            batch_size=config.batch_size, augment=augment, spec=spec
        )

        # Semilla fija para inicialización de red y augmentation para que los folds sean deterministas
        # excepto por los datos de entrada
        torch.manual_seed(42 + test_subject)
        np.random.seed(42 + test_subject)

        net = spec.factory().to(device)
        
        # Entrenar
        train_model(net, train_loader, test_loader, config, device, verbose=False)
        
        # Evaluar
        test_acc, labels, preds = evaluate(net, test_loader, device)
        print(f"    -> Accuracy (Sujeto {test_subject}): {test_acc*100:.2f}%\n")
        
        fold_results.append({
            "subject": test_subject,
            "test_acc": test_acc
        })

    elapsed = time.time() - t0
    accuracies = np.array([f["test_acc"] for f in fold_results])
    mean_acc = float(accuracies.mean())
    std_acc = float(accuracies.std(ddof=0))
    
    print("============================================================")
    print(f" RESUMEN LOSO - {name}")
    print("============================================================")
    for f in fold_results:
        print(f"  Sujeto {f['subject']:2d}: {f['test_acc']*100:.2f}%")
    print("-" * 60)
    print(f"  Media Global: {mean_acc*100:.2f}%")
    print(f"  Std Dev     : {std_acc*100:.2f}%")
    print(f"  Tiempo total: {elapsed/60:.1f} minutos")
    print("============================================================\n")

    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": f"{name} [LOSO]",
        "image_dir": image_dir,
        "model": model,
        "epochs": config.epochs,
        "lr": config.lr,
        "augment": augment,
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "folds": fold_results,
        "note": note,
        "elapsed_min": round(elapsed / 60, 2),
    }
    _append_jsonl(record)
    print(f"✓ Resultados guardados en {RESULTS_JSONL}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["segmented", "raw", "canonical"], required=True,
        help="Qué dataset usar."
    )
    parser.add_argument(
        "--model", choices=list(MODEL_REGISTRY), default="resnet18",
        help="Arquitectura (default: resnet18)."
    )
    parser.add_argument("--epochs", type=int, default=None,
                        help="Épocas (default: el del modelo elegido).")
    parser.add_argument("--augment", action="store_true",
                        help="Aplicar data augmentation.")
    parser.add_argument("--note", default="", help="Anotación para el log.")
    args = parser.parse_args()

    dirs = {
        "raw": RAW_DIR,
        "segmented": SEGMENTED_DIR,
        "canonical": CANONICAL_DIR
    }
    names = {
        "raw": "Raw (full image)",
        "segmented": "Segmented RGB",
        "canonical": "Canonical aligned"
    }
    
    spec = get_model_spec(args.model)
    epochs = args.epochs if args.epochs is not None else spec.default_epochs
    config = TrainConfig(epochs=epochs, lr=spec.default_lr)

    run_loso(
        image_dir=dirs[args.mode],
        name=names[args.mode],
        model=args.model,
        config=config,
        augment=args.augment,
        note=args.note
    )


if __name__ == "__main__":
    main()
