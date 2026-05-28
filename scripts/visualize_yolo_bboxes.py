"""Dibuja la bbox detectada de la mano derecha sobre raw para 16 ejemplos
(uno por clase) y guarda la grilla. Verificación visual antes de generar
las 800 anotaciones.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import LSA16_NAMES, RAW_DIR, SEGMENTED_DIR  # noqa: E402
from src.yolo_annotations import detect_right_hand_bbox  # noqa: E402


def load_rgb(path: Path):
    bgr = cv2.imread(str(path))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None


def main() -> None:
    # Una muestra por clase, mezclando sujetos para ver variabilidad
    samples = [(cls, ((cls * 3) % 10) + 1, 1) for cls in range(1, 17)]

    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    fig.suptitle(
        "Bbox detectada (verde) de la mano derecha sobre raw — 1 ejemplo por clase\n"
        "Color usado para detectar = Hue dominante del segmented (referencia per-imagen)",
        fontsize=12, fontweight="bold",
    )

    ok, fail = 0, 0
    for ax, (cls, subj, rep) in zip(axes.flat, samples):
        fname = f"{cls}_{subj}_{rep}.png"
        raw = load_rgb(Path(RAW_DIR) / fname)
        seg = load_rgb(Path(SEGMENTED_DIR) / fname)
        if raw is None or seg is None:
            ax.set_title(f"{fname}\n(no existe)", fontsize=9, color="red")
            ax.axis("off")
            continue

        bbox = detect_right_hand_bbox(raw, seg)
        ax.imshow(raw)
        ax.set_title(f"cls={cls} ({LSA16_NAMES[cls-1]}) subj={subj}", fontsize=10, fontweight="bold")
        ax.axis("off")

        if bbox is None:
            ax.text(0.5, 0.5, "FALLA", transform=ax.transAxes,
                    ha="center", va="center", fontsize=20, color="red", fontweight="bold")
            fail += 1
            continue

        rect = patches.Rectangle(
            (bbox.x, bbox.y), bbox.w, bbox.h,
            linewidth=2.5, edgecolor="#00FF41", facecolor="none",
        )
        ax.add_patch(rect)
        h_raw, w_raw = raw.shape[:2]
        xc, yc, ww, hh = bbox.to_yolo(w_raw, h_raw)
        ax.set_xlabel(
            f"yolo: xc={xc:.2f} yc={yc:.2f} w={ww:.2f} h={hh:.2f}",
            fontsize=7, fontfamily="monospace", color="#005500",
        )
        ax.axes.xaxis.set_visible(True)
        ax.set_xticks([])
        ok += 1

    out_path = PROJECT_ROOT / "outputs" / "02_yolo" / "figures" / "bboxes_per_class.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved: {out_path}")
    print(f"OK: {ok} | Falla: {fail}")


if __name__ == "__main__":
    main()
