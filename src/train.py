"""Loop de entrenamiento y evaluación.

Hiperparámetros del paper (Quiroga et al. 2017, Sec. 3.2):
- Adam, lr=0.0007
- LeNet: 20 épocas
- CrossEntropy
- Sin weight decay
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 0.0007
    batch_size: int = 32


@dataclass
class History:
    train_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)


def _run_epoch(model, loader, criterion, optimizer, device):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, correct, n = 0.0, 0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for X, y in loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if is_train:
                optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * X.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += y.size(0)
    return total_loss / n, correct / n


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
    verbose: bool = False,
) -> History:
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    history = History()

    for epoch in range(config.epochs):
        tl, ta = _run_epoch(model, train_loader, criterion, optimizer, device)
        vl, va = _run_epoch(model, val_loader, criterion, None, device)
        history.train_loss.append(tl)
        history.train_acc.append(ta)
        history.val_loss.append(vl)
        history.val_acc.append(va)
        if verbose:
            print(f"  epoch {epoch+1:>2}/{config.epochs}  "
                  f"train_loss={tl:.4f}  train_acc={ta:.4f}  "
                  f"val_loss={vl:.4f}  val_acc={va:.4f}")

    return history


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, list[int], list[int]]:
    model.eval()
    correct, n = 0, 0
    all_preds: list[int] = []
    all_labels: list[int] = []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        preds = model(X).argmax(1)
        correct += (preds == y).sum().item()
        n += y.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
    return correct / n, all_labels, all_preds
