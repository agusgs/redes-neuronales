"""Modelos disponibles para LSA16.

- ``LeNetLSA16``: réplica fiel del paper Quiroga et al. 2017.
- ``resnet18_transfer``: ResNet-18 preentrenada en ImageNet, fine-tuning end-to-end
  con la cabeza FC reemplazada para 16 clases.

Para agregar un modelo nuevo:
    1. Definir su factory (devuelve nn.Module).
    2. Agregarlo al dict ``MODEL_REGISTRY`` con su ``ModelSpec`` (input size,
       normalización ImageNet o paper, lr/epochs defaults).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torchvision.models as tvm


# ─── Constantes ──────────────────────────────────────────────────────────────
NUM_CLASSES = 16

PAPER_MEAN = (0.5, 0.5, 0.5)
PAPER_STD = (0.5, 0.5, 0.5)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ModelSpec:
    """Especificación completa de un modelo para entrenamiento en LSA16."""
    name: str
    factory: Callable[[], nn.Module]
    input_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    default_lr: float
    default_epochs: int


# ─── LeNet (paper) ───────────────────────────────────────────────────────────

def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.ELU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.BatchNorm2d(out_ch),
    )


class LeNetLSA16(nn.Module):
    """LeNet fiel al paper Quiroga et al. 2017 (Sec 3.2).

    4 conv blocks (32, 64, 128, 256) con BN post-MaxPool y ELU, FC 512, 16 logits.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, input_size: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            _conv_block(128, 256),
        )
        flat = 256 * (input_size // 16) * (input_size // 16)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ─── ResNet-18 transfer learning ─────────────────────────────────────────────

def resnet18_transfer(num_classes: int = NUM_CLASSES) -> nn.Module:
    """ResNet-18 preentrenada en ImageNet con la última FC reemplazada.

    Fine-tuning end-to-end (todos los pesos entrenables). Para freezing parcial
    habría que setear ``requires_grad=False`` en los layers deseados.
    """
    model = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ─── Registry ────────────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, ModelSpec] = {
    "lenet": ModelSpec(
        name="LeNetLSA16",
        factory=lambda: LeNetLSA16(),
        input_size=128,
        mean=PAPER_MEAN,
        std=PAPER_STD,
        default_lr=0.0007,
        default_epochs=20,
    ),
    "resnet18": ModelSpec(
        name="ResNet18-transfer",
        factory=lambda: resnet18_transfer(),
        input_size=224,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        # lr más baja: el backbone ya está entrenado, no queremos romperlo.
        default_lr=1e-4,
        # ResNet18 tiene muchísimos más parámetros que LeNet — usamos un poco
        # más de épocas pero con lr chica el sobreajuste se controla.
        default_epochs=20,
    ),
}


def get_model_spec(name: str) -> ModelSpec:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Modelo desconocido: {name!r}. Disponibles: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]
