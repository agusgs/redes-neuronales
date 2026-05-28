"""Visualización de 32 muestras aleatorias estratificadas para detectar
visualmente bboxes que caen en cara, otra mano u otra región incorrecta.

Cada muestra se compara con su segmented (que ES la mano derecha real).
Si la bbox no contiene aproximadamente los mismos colores que el segmented
→ anomalía visual.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import LSA16_NAMES, RAW_DIR, SEGMENTED_DIR, list_samples  # noqa: E402
from src.yolo_annotations import detect_right_hand_bbox  # noqa: E402


N_SAMPLES = 32
SEED = 42


def main() -> None:
    random.seed(SEED)
    samples = list_samples(RAW_DIR)
    # Estratificado: 2 por clase
    by_class: dict = {}
    for s in samples:
        by_class.setdefault(s.label, []).append(s)
    picked = []
    for label, group in by_class.items():
        picked.extend(random.sample(group, min(2, len(group))))
    random.shuffle(picked)
    picked = picked[:N_SAMPLES]

    fig, axes = plt.subplots(N_SAMPLES // 4, 8, figsize=(24, 4 * (N_SAMPLES // 4)))
    fig.suptitle(
        f"Validación: {N_SAMPLES} muestras aleatorias (2 por clase)\n"
        f"Cada par: raw+bbox (izq) y segmented_right_hand (der). "
        f"La bbox debe enmarcar la misma mano que aparece en el segmented.",
        fontsize=13, fontweight="bold",
    )

    for i, sample in enumerate(picked):
        fname = Path(sample.path).name
        raw_bgr = cv2.imread(sample.path)
        seg_bgr = cv2.imread(str(Path(SEGMENTED_DIR) / fname))
        if raw_bgr is None or seg_bgr is None:
            continue
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        seg_rgb = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2RGB)
        bbox = detect_right_hand_bbox(raw_rgb, seg_rgb)

        row = i // 4
        col_raw = (i % 4) * 2
        col_seg = col_raw + 1

        ax_raw = axes[row][col_raw]
        ax_seg = axes[row][col_seg]
        ax_raw.imshow(raw_rgb)
        cls_name = LSA16_NAMES[sample.label]
        ax_raw.set_title(f"{fname}\n{cls_name}", fontsize=8, fontweight="bold")
        ax_raw.axis("off")

        if bbox is not None:
            rect = patches.Rectangle(
                (bbox.x, bbox.y), bbox.w, bbox.h,
                linewidth=2.5, edgecolor="#00FF41", facecolor="none",
            )
            ax_raw.add_patch(rect)
        else:
            ax_raw.text(0.5, 0.5, "BBOX = None", transform=ax_raw.transAxes,
                        ha="center", va="center", color="red", fontsize=14, fontweight="bold")

        ax_seg.imshow(seg_rgb)
        ax_seg.set_title("segmented (referencia)", fontsize=8, color="gray")
        ax_seg.axis("off")

    out_path = PROJECT_ROOT / "outputs" / "02_yolo" / "figures" / "bboxes_random_32.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
