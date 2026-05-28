"""Dataset, transforms y splits para LSA16.

Convención de filenames: ``{clase}_{sujeto}_{repeticion}.png`` (clase 1-indexed).
"""
from __future__ import annotations

import os
import glob
import re
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset
import torchvision.transforms as T


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "lsa16_raw")
SEGMENTED_DIR = os.path.join(PROJECT_ROOT, "lsa16_segmented_right_hand")
CANONICAL_DIR = os.path.join(PROJECT_ROOT, "lsa16_segmented_canonical")

IMG_SIZE = 128
NUM_CLASSES = 16

LSA16_NAMES = [
    "Five", "Four", "Horns", "Curve", "Fingers", "Double", "Hook", "Index",
    "L", "Flat", "Mitten", "Beak", "Thumb", "Fist", "Telephone", "V",
]

_FNAME_RE = re.compile(r"^(\d+)_(\d+)_(\d+)\.png$")


@dataclass(frozen=True)
class Sample:
    path: str
    label: int      # 0-indexed
    subject: int    # 1..10
    repetition: int # 1..5


def list_samples(image_dir: str) -> List[Sample]:
    samples: List[Sample] = []
    for path in sorted(glob.glob(os.path.join(image_dir, "*.png"))):
        m = _FNAME_RE.match(os.path.basename(path))
        if not m:
            continue
        cls, subj, rep = (int(g) for g in m.groups())
        samples.append(Sample(path=path, label=cls - 1, subject=subj, repetition=rep))
    return samples


def build_transform(
    train: bool,
    augment: bool = False,
    size: int = IMG_SIZE,
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> T.Compose:
    """Transform para LSA16.

    Defaults fieles al paper: resize a 128x128, normalize a [-1, 1].
    Para modelos preentrenados en ImageNet pasar ``size=224, mean=IMAGENET_MEAN,
    std=IMAGENET_STD`` (ver ``src.model.IMAGENET_MEAN/STD``).

    Con ``augment=True`` y ``train=True`` agrega rotación pequeña, traslación
    y color jitter ligero. NO incluye flip horizontal (cambia la quiralidad).
    """
    ops: list = []
    if train and augment:
        # Pequeño jitter geométrico — las manos ya están aproximadamente
        # alineadas (canonical), así que sólo perturbaciones leves.
        ops.append(T.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            fill=0,  # fondo negro
        ))
        ops.append(T.ColorJitter(brightness=0.15, contrast=0.15))
    ops += [
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=list(mean), std=list(std)),
    ]
    return T.Compose(ops)


class LSA16Dataset(Dataset):
    """Dataset PyTorch sobre una lista explícita de Samples."""

    def __init__(self, samples: Sequence[Sample], transform: T.Compose | None = None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, s.label


def stratified_split(
    samples: Sequence[Sample],
    test_size: float = 0.10,
    seed: int = 0,
) -> tuple[list[Sample], list[Sample]]:
    """Stratified randomized subsampling — el protocolo del paper.

    Estratifica por clase, NO separa por sujeto. Una imagen del sujeto 3
    puede caer en train mientras otra del mismo sujeto cae en test.
    """
    labels = np.array([s.label for s in samples])
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.zeros(len(samples)), labels))
    train = [samples[i] for i in train_idx]
    test = [samples[i] for i in test_idx]
    return train, test
