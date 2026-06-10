"""Runner de experimentos sobre LSA16 con la metodología del paper Quiroga et al. 2017.

Corre stratified randomized subsampling sobre uno o varios datasets
(raw / segmented / canonical), con LeNet (default, fiel al paper) o
modelos modernos (ResNet18 con transfer learning).

Resultados se loguean en ``outputs/01_baseline/results.jsonl`` y en una
tabla markdown legible.

Ejemplos:
    # Réplica del paper (LeNet, segmented)
    python experiments/01_baseline_paper.py --mode segmented --n-runs 3

    # Réplica del paper con preprocesamiento canónico
    python experiments/01_baseline_paper.py --mode canonical --n-runs 10

    # Mejora: ResNet18 con transfer learning
    python experiments/01_baseline_paper.py --mode canonical --model resnet18 --n-runs 5

    # Augmentation
    python experiments/01_baseline_paper.py --mode canonical --augment --n-runs 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment import (  # noqa: E402
    ExperimentResult, run_canonical, run_raw, run_segmented,
)
from src.model import MODEL_REGISTRY  # noqa: E402
from src.train import TrainConfig  # noqa: E402

OUT_DIR = PROJECT_ROOT / "outputs" / "01_baseline"
RESULTS_JSONL = OUT_DIR / "results.jsonl"
RESULTS_MD = OUT_DIR / "experiments_log.md"

# Referencia del paper SOLO aplica al modelo LeNet (el paper usa esa arquitectura).
# Modelos modernos (ResNet18, etc.) son contribución nuestra — no hay benchmark del paper.
DATASET_PAPER_REFERENCE = {
    "Segmented RGB": 96.18,
    "Raw (full image)": 83.54,
    "Canonical aligned": 96.18,
}


def _paper_ref(result_name: str, model: str) -> float | None:
    if model != "lenet":
        return None
    for ds_name, ref in DATASET_PAPER_REFERENCE.items():
        if ds_name in result_name:
            return ref
    return None


def _append_jsonl(record: dict) -> None:
    RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _append_markdown(result: ExperimentResult, config: TrainConfig, note: str,
                     augment: bool, model: str) -> None:
    ref = _paper_ref(result.name, model)
    aug = "✓" if augment else "—"
    paper_cell = f"{ref:.2f}%" if ref is not None else "—"
    gap_cell = f"{ref - result.mean*100:+.2f}pp" if ref is not None else "—"
    row = (
        f"| {time.strftime('%Y-%m-%d %H:%M')} | {result.name} | {aug} | "
        f"{len(result.runs)} | {config.epochs} | "
        f"{result.mean*100:.2f}% | {result.std*100:.2f}% | "
        f"{paper_cell} | {gap_cell} | {note or '—'} |\n"
    )
    header = (
        "| Fecha | Experimento | Aug | Runs | Épocas | Media | Std | Paper | Gap | Nota |\n"
        "|---|---|:-:|---:|---:|---:|---:|---:|---:|---|\n"
    )
    if not RESULTS_MD.exists():
        RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_MD.write_text(
            "# Bitácora de experimentos — Baseline paper\n\n"
            "Resultados de la réplica del paper Quiroga et al. 2017 sobre LSA16.\n\n"
            + header,
            encoding="utf-8",
        )
    with RESULTS_MD.open("a", encoding="utf-8") as f:
        f.write(row)


def _log_result(result: ExperimentResult, config: TrainConfig, note: str,
                augment: bool, model: str) -> None:
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": result.name,
        "image_dir": result.image_dir,
        "model": model,
        "n_runs": len(result.runs),
        "epochs": config.epochs,
        "lr": config.lr,
        "test_size": result.test_size,
        "augment": augment,
        "mean_acc": result.mean,
        "std_acc": result.std,
        "paper_reference": _paper_ref(result.name, model),
        "per_run": [{"seed": r.seed, "test_acc": r.test_acc} for r in result.runs],
        "note": note,
    }
    _append_jsonl(record)
    _append_markdown(result, config, note, augment, model)


def _print_summary(result: ExperimentResult, model: str) -> None:
    ref = _paper_ref(result.name, model)
    print()
    print("=" * 60)
    print(f"  RESUMEN — {result.name}")
    print("=" * 60)
    for i, r in enumerate(result.runs, 1):
        print(f"  Run {i} (seed={r.seed}): {r.test_acc*100:.2f}%")
    print(f"  {'-'*40}")
    print(f"  Media:  {result.mean*100:.2f}%")
    print(f"  Std:    {result.std*100:.2f}%")
    if ref is not None:
        print(f"  Paper:  {ref:.2f}%")
        print(f"  Gap:    {(ref - result.mean*100):+.2f}pp")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["segmented", "raw", "canonical", "all"], default="segmented",
        help="Qué dataset(s) usar."
    )
    parser.add_argument(
        "--model", choices=list(MODEL_REGISTRY), default="lenet",
        help="Arquitectura. 'lenet' = paper. 'resnet18' = transfer learning."
    )
    parser.add_argument("--n-runs", type=int, default=3,
                        help="Número de subsampling runs (default: 3).")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Épocas por run (default: el del modelo elegido).")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: el del modelo elegido).")
    parser.add_argument("--note", default="",
                        help="Anotación libre que se guarda en el log.")
    parser.add_argument("--augment", action="store_true",
                        help="Aplicar data augmentation (rot ±10°, translate ±10%%, color jitter).")
    parser.add_argument("--seed-base", type=int, default=0,
                        help="Seed inicial. Runs usan seed_base, seed_base+1, ... (default: 0).")
    parser.add_argument("--device", default=None,
                        help="Forzar un dispositivo (ej: 'cpu', 'mps', 'cuda').")
    args = parser.parse_args()

    spec = MODEL_REGISTRY[args.model]
    epochs = args.epochs if args.epochs is not None else spec.default_epochs
    lr = args.lr if args.lr is not None else spec.default_lr
    config = TrainConfig(epochs=epochs, lr=lr)

    modes = ["raw", "segmented", "canonical"] if args.mode == "all" else [args.mode]
    runners = {"raw": run_raw, "segmented": run_segmented, "canonical": run_canonical}

    device_obj = torch.device(args.device) if args.device else None

    for mode in modes:
        try:
            result = runners[mode](
                n_runs=args.n_runs, config=config, model=args.model,
                augment=args.augment, seed_base=args.seed_base, device=device_obj,
            )
        except FileNotFoundError as e:
            print(f"\n⚠️  No se pudo correr {mode}: {e}")
            print(f"   (¿quizá falta generar el dataset? Ver scripts/preprocess_canonical.py)")
            continue
        _print_summary(result, args.model)
        _log_result(result, config, args.note, args.augment, args.model)
        print(f"\n✓ Resultados guardados en:")
        print(f"   {RESULTS_JSONL}")
        print(f"   {RESULTS_MD}")


if __name__ == "__main__":
    main()
