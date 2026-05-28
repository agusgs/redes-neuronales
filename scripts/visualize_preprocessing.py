"""Muestra side-by-side original vs canonical_align para verificar el pipeline.

Selecciona muestras de 4 clases × 5 sujetos. Si la fila inferior queda con
manos consistentemente verticales y dedos arriba, el preprocesamiento anda.
"""
from __future__ import annotations
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.data import LSA16_NAMES, SEGMENTED_DIR
from src.preprocessing import canonical_align


def load_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main() -> None:
    classes = [1, 5, 9, 14]
    subjects = [1, 2, 3, 4, 5]

    fig, axes = plt.subplots(2 * len(classes), len(subjects), figsize=(15, 24))
    fig.suptitle("Canonical alignment — original (arriba) vs procesado (abajo) por par",
                 fontsize=13, fontweight="bold")

    for ri, cls in enumerate(classes):
        for c, subj in enumerate(subjects):
            fname = f"{cls}_{subj}_1.png"
            path = os.path.join(SEGMENTED_DIR, fname)
            if not os.path.exists(path):
                continue
            original = load_rgb(path)
            processed = canonical_align(original, target_size=128)

            row_orig = 2 * ri
            row_proc = 2 * ri + 1
            axes[row_orig][c].imshow(original)
            axes[row_orig][c].set_title(
                f"cls={cls} ({LSA16_NAMES[cls-1]}) subj={subj}\n{original.shape[1]}×{original.shape[0]}",
                fontsize=9,
            )
            axes[row_orig][c].axis("off")

            axes[row_proc][c].imshow(processed)
            axes[row_proc][c].set_title(f"procesado 128×128", fontsize=9, color="green")
            axes[row_proc][c].axis("off")

    out_path = os.path.join(PROJECT_ROOT, "outputs", "01_baseline", "figures",
                            "preprocessing_comparison.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
