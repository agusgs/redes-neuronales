"""Detecta anomalías en las bboxes generadas comparando contra el tamaño
de la imagen segmentada (que ES la mano derecha aislada del raw).

Lógica: si nuestra bbox tiene dimensiones muy distintas a las del segmented,
probablemente detectó otra región (cara, mano izquierda, brazo) y no la mano.

Métrica:
    ratio_area = area_bbox / area_segmented
    ratio_w    = ancho_bbox / ancho_segmented
    ratio_h    = alto_bbox / alto_segmented

Esperamos los tres cerca de 1.0. Flagueamos cualquier imagen donde
estos ratios estén fuera del rango [0.5, 2.0].
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import LSA16_NAMES, RAW_DIR, SEGMENTED_DIR, list_samples  # noqa: E402
from src.yolo_annotations import detect_right_hand_bbox  # noqa: E402


# Rangos de "normalidad"
RATIO_MIN = 0.5
RATIO_MAX = 2.0


def main() -> None:
    samples = list_samples(RAW_DIR)
    print(f"Validando {len(samples)} imágenes...")

    anomalies = []
    ratios_all = []

    for s in samples:
        fname = Path(s.path).name
        raw_bgr = cv2.imread(s.path)
        seg_bgr = cv2.imread(str(Path(SEGMENTED_DIR) / fname))
        if raw_bgr is None or seg_bgr is None:
            continue
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        seg_rgb = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2RGB)
        bbox = detect_right_hand_bbox(raw_rgb, seg_rgb)
        if bbox is None:
            anomalies.append({"fname": fname, "reason": "bbox None"})
            continue

        seg_h, seg_w = seg_bgr.shape[:2]
        rw = bbox.w / seg_w
        rh = bbox.h / seg_h
        ra = (bbox.w * bbox.h) / (seg_w * seg_h)
        ratios_all.append((fname, rw, rh, ra, bbox))

        out_of_range = (
            rw < RATIO_MIN or rw > RATIO_MAX or
            rh < RATIO_MIN or rh > RATIO_MAX or
            ra < RATIO_MIN or ra > RATIO_MAX
        )
        if out_of_range:
            anomalies.append({
                "fname": fname, "rw": rw, "rh": rh, "ra": ra, "bbox": bbox,
            })

    # Estadísticas globales
    rws = np.array([r[1] for r in ratios_all])
    rhs = np.array([r[2] for r in ratios_all])
    ras = np.array([r[3] for r in ratios_all])
    print(f"\n=== Estadísticas (sobre {len(ratios_all)} con bbox válida) ===")
    print(f"  ratio_w  p5/p50/p95: {np.percentile(rws, 5):.2f} / "
          f"{np.percentile(rws, 50):.2f} / {np.percentile(rws, 95):.2f}")
    print(f"  ratio_h  p5/p50/p95: {np.percentile(rhs, 5):.2f} / "
          f"{np.percentile(rhs, 50):.2f} / {np.percentile(rhs, 95):.2f}")
    print(f"  ratio_a  p5/p50/p95: {np.percentile(ras, 5):.2f} / "
          f"{np.percentile(ras, 50):.2f} / {np.percentile(ras, 95):.2f}")

    print(f"\n=== Anomalías ===")
    print(f"  {len(anomalies)} de {len(samples)} ({len(anomalies)/len(samples)*100:.1f}%)")

    if not anomalies:
        print("  ✓ Sin anomalías")
        return

    # Listar las primeras 20
    print(f"\n  Top 20 anomalías:")
    for a in anomalies[:20]:
        if "bbox" in a and isinstance(a.get("rw"), float):
            print(f"    {a['fname']:14s} rw={a['rw']:.2f} rh={a['rh']:.2f} ra={a['ra']:.2f}")
        else:
            print(f"    {a['fname']:14s} {a.get('reason', '?')}")

    # Visualizar hasta 16 anomalías
    show = [a for a in anomalies if "bbox" in a][:16]
    if show:
        n_cols = 4
        n_rows = (len(show) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), squeeze=False)
        fig.suptitle(f"Anomalías detectadas: bbox muy distinto del tamaño de segmented "
                     f"({len(anomalies)} total, mostrando {len(show)})",
                     fontsize=12, fontweight="bold")
        for ax, a in zip(axes.flat, show):
            raw = cv2.imread(str(Path(RAW_DIR) / a["fname"]))
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            bbox = a["bbox"]
            ax.imshow(raw)
            rect = patches.Rectangle((bbox.x, bbox.y), bbox.w, bbox.h,
                                     linewidth=2.5, edgecolor="#FF1744", facecolor="none")
            ax.add_patch(rect)
            cls = int(a["fname"].split("_")[0])
            ax.set_title(f"{a['fname']} ({LSA16_NAMES[cls-1]})\n"
                         f"rw={a['rw']:.2f} rh={a['rh']:.2f} ra={a['ra']:.2f}",
                         fontsize=9)
            ax.axis("off")
        for ax in axes.flat[len(show):]:
            ax.axis("off")
        out_path = PROJECT_ROOT / "outputs" / "02_yolo" / "figures" / "bbox_anomalies.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=110, bbox_inches="tight")
        print(f"\n  ✓ Visualización: {out_path}")


if __name__ == "__main__":
    main()
