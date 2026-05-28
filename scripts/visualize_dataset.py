"""Visualiza muestras del dataset segmentado para chequear si las manos
están en orientación canónica (vertical) o en orientaciones variadas.

Si están en orientaciones variadas, falta el preprocesamiento que el paper
aplica (rotación + corrección de inversión).
"""
from __future__ import annotations
import os
import sys

import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.data import LSA16_NAMES, SEGMENTED_DIR


def main() -> None:
    # 4 clases, 5 sujetos por clase, para ver variabilidad por intérprete
    classes = [1, 5, 9, 14]   # Five, Double, Flat, Telephone
    subjects = [1, 2, 3, 4, 5]

    fig, axes = plt.subplots(len(classes), len(subjects), figsize=(15, 12))
    fig.suptitle(
        "Dataset segmented_right_hand — 4 clases × 5 sujetos\n"
        "¿Están las manos rotadas a orientación canónica?",
        fontsize=13, fontweight="bold",
    )

    for r, cls in enumerate(classes):
        for c, subj in enumerate(subjects):
            fname = f"{cls}_{subj}_1.png"  # primer ejemplo de cada (clase, sujeto)
            path = os.path.join(SEGMENTED_DIR, fname)
            ax = axes[r][c]
            if os.path.exists(path):
                img = Image.open(path)
                ax.imshow(img)
                ax.set_title(f"cls={cls} ({LSA16_NAMES[cls-1]})\nsubj={subj} {img.size}",
                             fontsize=9)
            else:
                ax.set_title(f"{fname} (no existe)", fontsize=9, color="red")
            ax.axis("off")

    out_path = os.path.join(PROJECT_ROOT, "outputs", "01_baseline", "figures",
                            "dataset_inspection.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
