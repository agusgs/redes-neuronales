"""Replicación del protocolo de validación del paper.

Quiroga et al. 2017 (Sec. 3.2 Methodology):
"we averaged 100 runs of stratified randomized subsampling cross-validation
to estimate the test set accuracy."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import (
    CANONICAL_DIR, LSA16Dataset, RAW_DIR, SEGMENTED_DIR, Sample,
    build_transform, list_samples, stratified_split,
)
from src.model import ModelSpec, get_model_spec
from src.train import History, TrainConfig, evaluate, train_model


@dataclass
class RunResult:
    seed: int
    test_acc: float
    history: History
    labels: List[int]
    preds: List[int]


@dataclass
class ExperimentResult:
    name: str
    image_dir: str
    test_size: float
    runs: List[RunResult] = field(default_factory=list)

    @property
    def accuracies(self) -> np.ndarray:
        return np.array([r.test_acc for r in self.runs])

    @property
    def mean(self) -> float:
        return float(self.accuracies.mean()) if self.runs else float("nan")

    @property
    def std(self) -> float:
        return float(self.accuracies.std(ddof=0)) if self.runs else float("nan")


def _make_loaders(
    train_samples,
    test_samples,
    batch_size: int,
    augment: bool,
    spec: ModelSpec,
):
    train_tx = build_transform(train=True, augment=augment,
                               size=spec.input_size, mean=spec.mean, std=spec.std)
    test_tx = build_transform(train=False, size=spec.input_size,
                              mean=spec.mean, std=spec.std)
    train_ds = LSA16Dataset(train_samples, transform=train_tx)
    test_ds = LSA16Dataset(test_samples, transform=test_tx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def run_single(
    samples: List[Sample],
    seed: int,
    test_size: float,
    config: TrainConfig,
    device: torch.device,
    spec: ModelSpec,
    verbose: bool = False,
    augment: bool = False,
) -> RunResult:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_samples, test_samples = stratified_split(samples, test_size=test_size, seed=seed)
    train_loader, test_loader = _make_loaders(train_samples, test_samples,
                                              config.batch_size, augment=augment, spec=spec)

    model = spec.factory().to(device)
    history = train_model(model, train_loader, test_loader, config, device, verbose=verbose)
    test_acc, labels, preds = evaluate(model, test_loader, device)

    return RunResult(seed=seed, test_acc=test_acc, history=history, labels=labels, preds=preds)


def run_experiment(
    image_dir: str,
    name: str,
    model: str = "lenet",
    n_runs: int = 10,
    test_size: float = 0.10,
    config: TrainConfig | None = None,
    device: torch.device | None = None,
    seed_base: int = 0,
    verbose: bool = False,
    augment: bool = False,
) -> ExperimentResult:
    spec = get_model_spec(model)
    # Si no se pasa config, usar los defaults del modelo.
    if config is None:
        config = TrainConfig(epochs=spec.default_epochs, lr=spec.default_lr)
    # MPS (Apple Silicon GPU) tiene un bug con BatchNorm + Adam en PyTorch 2.12:
    # el modelo entrena bien pero al evaluar predice una sola clase. CPU es estable
    # y suficiente para LSA16 (~110s por corrida de 20 épocas).
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = list_samples(image_dir)
    if not samples:
        raise FileNotFoundError(f"No PNG files found in {image_dir!r}")

    full_name = f"{name} [{spec.name}]"
    result = ExperimentResult(name=full_name, image_dir=image_dir, test_size=test_size)
    aug_tag = " | augment=ON" if augment else ""
    print(f"[{full_name}] dir={image_dir} | n_samples={len(samples)} | "
          f"n_runs={n_runs} | epochs={config.epochs} | lr={config.lr} | "
          f"input={spec.input_size} | device={device}{aug_tag}")
    for i in range(n_runs):
        seed = seed_base + i
        run = run_single(samples, seed=seed, test_size=test_size,
                         config=config, device=device, spec=spec,
                         verbose=verbose, augment=augment)
        result.runs.append(run)
        accs = result.accuracies
        print(f"  run {i+1:>3}/{n_runs} (seed={seed}): "
              f"test_acc={run.test_acc*100:.2f}%  "
              f"running_mean={accs.mean()*100:.2f}% ± {accs.std()*100:.2f}%")
    return result


def run_segmented(n_runs: int = 10, **kwargs) -> ExperimentResult:
    return run_experiment(SEGMENTED_DIR, name="Segmented RGB", n_runs=n_runs, **kwargs)


def run_raw(n_runs: int = 10, **kwargs) -> ExperimentResult:
    return run_experiment(RAW_DIR, name="Raw (full image)", n_runs=n_runs, **kwargs)


def run_canonical(n_runs: int = 10, **kwargs) -> ExperimentResult:
    return run_experiment(CANONICAL_DIR, name="Canonical aligned", n_runs=n_runs, **kwargs)
