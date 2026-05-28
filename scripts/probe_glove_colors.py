"""Sondea los rangos HSV reales de los guantes rojo y magenta en lsa16_raw/.

Muestra los histogramas de Hue/Sat/Val sobre píxeles saturados (los guantes)
y permite ajustar los rangos con datos antes de implementar la detección.

Usa la máscara de lsa16_segmented_right_hand como "ground truth" del rojo
(es la mano derecha, que está enguantada con rojo) y por exclusión asume
que el otro componente saturado es magenta.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import RAW_DIR  # noqa: E402

# Muestra de 6 imágenes diversas
SAMPLE_FILES = [
    "1_1_1.png", "5_3_2.png", "9_5_3.png",
    "12_2_4.png", "14_4_5.png", "16_8_1.png",
]


def main() -> None:
    fig, axes = plt.subplots(len(SAMPLE_FILES), 3, figsize=(14, 4 * len(SAMPLE_FILES)))
    fig.suptitle(
        "Sondeo de colores de guantes en lsa16_raw/\n"
        "Izquierda: raw | Centro: máscara de píxeles saturados (S>100, V>50) | Derecha: histograma de Hue de esos píxeles",
        fontsize=11, fontweight="bold",
    )

    all_hues = []

    for r, fname in enumerate(SAMPLE_FILES):
        path = Path(RAW_DIR) / fname
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Máscara de "píxeles saturados" — candidatos a guante
        sat_mask = ((s > 100) & (v > 50)).astype(np.uint8) * 255

        # Hue de esos píxeles
        hues_here = h[sat_mask > 0]
        all_hues.append(hues_here)

        axes[r][0].imshow(rgb)
        axes[r][0].set_title(f"{fname}", fontsize=9)
        axes[r][0].axis("off")

        axes[r][1].imshow(sat_mask, cmap="gray")
        axes[r][1].set_title(f"saturados (n={sat_mask.sum()//255})", fontsize=9)
        axes[r][1].axis("off")

        axes[r][2].hist(hues_here, bins=180, range=(0, 180), color="purple")
        axes[r][2].axvspan(0, 10, alpha=0.2, color="red", label="rojo bajo")
        axes[r][2].axvspan(170, 180, alpha=0.2, color="red", label="rojo alto")
        axes[r][2].axvspan(140, 170, alpha=0.2, color="magenta", label="magenta")
        axes[r][2].set_xlabel("Hue (0-180)")
        axes[r][2].set_ylabel("count")
        axes[r][2].legend(fontsize=7)

    out_path = PROJECT_ROOT / "outputs" / "02_yolo" / "figures" / "probe_glove_colors.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # Estadísticas globales
    all_hues_flat = np.concatenate(all_hues)
    print(f"\nEstadísticas sobre {len(all_hues_flat):,} píxeles saturados:")
    print(f"  Hue min/p5/median/p95/max: "
          f"{all_hues_flat.min()}/{np.percentile(all_hues_flat, 5):.0f}/"
          f"{np.percentile(all_hues_flat, 50):.0f}/"
          f"{np.percentile(all_hues_flat, 95):.0f}/{all_hues_flat.max()}")
    print(f"  Distribución por bandas:")
    for lo, hi, name in [(0, 15, "rojo bajo"), (140, 170, "magenta"), (170, 180, "rojo alto")]:
        count = ((all_hues_flat >= lo) & (all_hues_flat <= hi)).sum()
        pct = 100 * count / len(all_hues_flat)
        print(f"    Hue {lo:>3}-{hi:>3} ({name:11s}): {count:>8,} px  ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
