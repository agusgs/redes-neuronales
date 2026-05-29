"""Genera el dataset en formato YOLO desde lsa16_raw + lsa16_segmented_right_hand.

Estructura producida:
    dataset_yolo/
    ├── dataset.yaml        # configuración Ultralytics
    ├── images/
    │   ├── train/   (70%)
    │   ├── val/     (15%)
    │   └── test/    (15%)
    └── labels/
        ├── train/
        ├── val/
        └── test/

Cada imagen .png va con su .txt correspondiente. Formato de cada .txt:
    <class_id> <xc> <yc> <w> <h>     (todo normalizado [0,1], class 0-indexed)

Una sola línea por archivo (una mano por imagen).
"""
from __future__ import annotations

import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import LSA16_NAMES, RAW_DIR, SEGMENTED_DIR, list_samples  # noqa: E402
from src.yolo_annotations import detect_right_hand_bbox  # noqa: E402


OUT_ROOT = PROJECT_ROOT / "dataset_yolo"

# Splits: 70/15/15 estratificado por clase, partición fija (seed=0)
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SPLIT_SEED = 0


def _clean_output_dir() -> None:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    for split in ("train", "val", "test"):
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


def _stratified_split(samples) -> dict:
    """Asigna cada sample a 'train', 'val' o 'test' estratificado por clase."""
    n = len(samples)
    labels = np.array([s.label for s in samples])

    # Primero apartamos test (15%), después val (15% del 85% restante = 17.6%).
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=SPLIT_SEED)
    train_val_idx, test_idx = next(sss1.split(np.zeros(n), labels))

    val_frac_within = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac_within,
                                  random_state=SPLIT_SEED)
    train_idx_inner, val_idx_inner = next(
        sss2.split(np.zeros(len(train_val_idx)), labels[train_val_idx])
    )
    train_idx = train_val_idx[train_idx_inner]
    val_idx = train_val_idx[val_idx_inner]

    assignment = {}
    for i in train_idx: assignment[i] = "train"
    for i in val_idx:   assignment[i] = "val"
    for i in test_idx:  assignment[i] = "test"
    return assignment


def _write_dataset_yaml(class_names: list[str]) -> None:
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    yaml_content = (
        f"# Dataset LSA16 en formato Ultralytics YOLO\n"
        f"# Generado por scripts/generate_yolo_dataset.py\n\n"
        f"path: {OUT_ROOT.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  images/test\n\n"
        f"nc: {len(class_names)}\n"
        f"names:\n{names_block}\n"
    )
    (OUT_ROOT / "dataset.yaml").write_text(yaml_content, encoding="utf-8")


def main() -> None:
    _clean_output_dir()

    samples = list_samples(RAW_DIR)
    if not samples:
        raise FileNotFoundError(f"Sin imágenes en {RAW_DIR}")
    print(f"Encontradas {len(samples)} imágenes raw.")

    assignment = _stratified_split(samples)

    counts = {"train": Counter(), "val": Counter(), "test": Counter()}
    failures: list[tuple[str, str]] = []
    processed = 0

    for i, sample in enumerate(samples):
        split = assignment[i]
        fname = Path(sample.path).name

        raw_bgr = cv2.imread(sample.path)
        seg_path = Path(SEGMENTED_DIR) / fname
        seg_bgr = cv2.imread(str(seg_path))
        if raw_bgr is None or seg_bgr is None:
            failures.append((fname, "no se pudo leer"))
            continue

        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        seg_rgb = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2RGB)
        bbox = detect_right_hand_bbox(raw_rgb, seg_rgb)
        if bbox is None:
            failures.append((fname, "no se detectó bbox"))
            continue

        h_raw, w_raw = raw_bgr.shape[:2]
        xc, yc, ww, hh = bbox.to_yolo(w_raw, h_raw)

        # Copiar imagen
        shutil.copy(sample.path, OUT_ROOT / "images" / split / fname)
        # Escribir anotación
        label_fname = fname.replace(".png", ".txt")
        label_path = OUT_ROOT / "labels" / split / label_fname
        label_path.write_text(f"{sample.label} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n", encoding="utf-8")

        counts[split][sample.label] += 1
        processed += 1
        if processed % 100 == 0:
            print(f"  {processed}/{len(samples)} procesadas")

    _write_dataset_yaml(LSA16_NAMES)

    # Resumen
    print()
    print("=" * 64)
    print(f"  Procesadas:   {processed}/{len(samples)}")
    print(f"  Fallas:       {len(failures)}")
    print("=" * 64)
    for split in ("train", "val", "test"):
        total = sum(counts[split].values())
        print(f"  {split:>5}: {total:>4} imágenes  ({total/processed*100:.1f}%)")
    print("=" * 64)
    print("  Distribución por clase (train / val / test):")
    for cls in range(16):
        t, v, te = counts['train'][cls], counts['val'][cls], counts['test'][cls]
        print(f"    {cls:2d} {LSA16_NAMES[cls]:12s}: {t:>3} / {v:>3} / {te:>3}")
    print("=" * 64)
    if failures:
        print(f"\n  Imágenes con error ({len(failures)}):")
        for fname, reason in failures[:10]:
            print(f"    {fname}: {reason}")
        if len(failures) > 10:
            print(f"    ... ({len(failures) - 10} más)")
    print(f"\n  ✓ Dataset listo en {OUT_ROOT}")
    print(f"    YAML: {OUT_ROOT / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
