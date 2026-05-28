"""Sondeo v2: compara HSV de la mano DERECHA (vía segmented) vs el resto
de los píxeles saturados (la otra mano) para determinar los colores reales
de cada guante.

Estrategia:
1. Para cada sample image, cargar raw y segmented.
2. La segmented tiene la mano derecha aislada. Sus pixels son el guante derecho.
3. En el raw, los pixels saturados que NO matchean el color del guante derecho
   son la mano izquierda.

Devolvemos los rangos de Hue de cada guante con confianza.
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

from src.data import RAW_DIR, SEGMENTED_DIR  # noqa: E402

SAMPLE_FILES = [
    "1_1_1.png", "5_3_2.png", "9_5_3.png",
    "12_2_4.png", "14_4_5.png", "16_8_1.png",
]


def main() -> None:
    fig, axes = plt.subplots(len(SAMPLE_FILES), 4, figsize=(18, 4 * len(SAMPLE_FILES)))
    fig.suptitle(
        "Sondeo de color por mano: segmented (mano derecha) vs raw saturado\n"
        "Col 1: raw | Col 2: segmented_right_hand | Col 3: hist Hue de la derecha | "
        "Col 4: hist Hue de los saturados del raw (incluye ambas manos)",
        fontsize=11, fontweight="bold",
    )

    hues_derecha = []
    hues_raw_total = []

    for r, fname in enumerate(SAMPLE_FILES):
        raw_bgr = cv2.imread(str(Path(RAW_DIR) / fname))
        seg_bgr = cv2.imread(str(Path(SEGMENTED_DIR) / fname))
        if raw_bgr is None or seg_bgr is None:
            continue

        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        seg_rgb = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2RGB)

        # Hue de la mano derecha (vía segmented, donde hay pixels no negros)
        seg_hsv = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2HSV)
        seg_gray = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2GRAY)
        seg_mask = seg_gray > 10
        hues_d = seg_hsv[..., 0][seg_mask]
        sats_d = seg_hsv[..., 1][seg_mask]
        # Filtramos solo los píxeles saturados (sin shadows)
        hues_d = hues_d[sats_d > 100]
        hues_derecha.append(hues_d)

        # Hue total de saturados en el raw
        raw_hsv = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2HSV)
        sat_mask = (raw_hsv[..., 1] > 100) & (raw_hsv[..., 2] > 50)
        hues_t = raw_hsv[..., 0][sat_mask]
        hues_raw_total.append(hues_t)

        axes[r][0].imshow(raw_rgb)
        axes[r][0].set_title(f"raw — {fname}", fontsize=9)
        axes[r][0].axis("off")

        axes[r][1].imshow(seg_rgb)
        axes[r][1].set_title(f"segmented_right_hand (mano derecha aislada)", fontsize=9)
        axes[r][1].axis("off")

        axes[r][2].hist(hues_d, bins=180, range=(0, 180), color="#E74C3C")
        axes[r][2].set_title(f"Hue de la mano DERECHA (n={len(hues_d):,})", fontsize=9)
        axes[r][2].set_xlabel("Hue")

        axes[r][3].hist(hues_t, bins=180, range=(0, 180), color="purple")
        axes[r][3].set_title(f"Hue de TODOS los saturados raw (n={len(hues_t):,})", fontsize=9)
        axes[r][3].set_xlabel("Hue")

    out_path = PROJECT_ROOT / "outputs" / "02_yolo" / "figures" / "probe_glove_colors_v2.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # Estadísticas consolidadas
    h_der = np.concatenate(hues_derecha)
    h_tot = np.concatenate(hues_raw_total)
    print(f"\n=== Mano DERECHA (rojo, vía segmented) ===")
    print(f"  Hue p5/p50/p95: "
          f"{np.percentile(h_der, 5):.0f} / {np.percentile(h_der, 50):.0f} / "
          f"{np.percentile(h_der, 95):.0f}")
    print(f"  Hue ∈ [0,10]∪[170,180]: "
          f"{((h_der <= 10) | (h_der >= 170)).mean()*100:.1f}%")

    print(f"\n=== TODOS los saturados del raw (ambas manos) ===")
    print(f"  Hue p5/p50/p95: "
          f"{np.percentile(h_tot, 5):.0f} / {np.percentile(h_tot, 50):.0f} / "
          f"{np.percentile(h_tot, 95):.0f}")

    # Por bandas
    print(f"\n=== Distribución por banda de Hue ===")
    bands = [(0, 10, "rojo"), (10, 40, "naranja"), (40, 80, "amarillo/verde"),
             (80, 110, "cian"), (110, 140, "azul"), (140, 170, "magenta"),
             (170, 180, "rojo wrap")]
    print(f"  {'Banda':<22} {'Derecha':>10} {'Raw total':>10}")
    for lo, hi, name in bands:
        d_pct = ((h_der >= lo) & (h_der < hi)).mean() * 100
        t_pct = ((h_tot >= lo) & (h_tot < hi)).mean() * 100
        marker = "  ← derecha" if d_pct > 15 else ("  ← otra mano" if t_pct - d_pct > 10 else "")
        print(f"  {name:>4} ({lo:>3}-{hi:>3})       {d_pct:>9.1f}% {t_pct:>9.1f}%{marker}")


if __name__ == "__main__":
    main()
