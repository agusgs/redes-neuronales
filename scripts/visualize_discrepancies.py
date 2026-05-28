"""Visualiza casos donde template matching y detección por color discrepan.
Permite decidir cuál de los dos métodos es correcto.

Verde = template matching (la mano segmented "encajada" en raw)
Rojo  = detección por color (lo que veníamos usando)
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import LSA16_NAMES, RAW_DIR, SEGMENTED_DIR, list_samples  # noqa: E402
from src.yolo_annotations import detect_right_hand_bbox, detect_right_hand_bbox_by_color  # noqa: E402


def iou(a, b) -> float:
    x1 = max(a.x, b.x); y1 = max(a.y, b.y)
    x2 = min(a.x + a.w, b.x + b.w); y2 = min(a.y + a.h, b.y + b.h)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    samples = list_samples(RAW_DIR)
    discrepancies = []
    for s in samples:
        fname = Path(s.path).name
        raw_bgr = cv2.imread(s.path)
        seg_bgr = cv2.imread(str(Path(SEGMENTED_DIR) / fname))
        if raw_bgr is None or seg_bgr is None:
            continue
        raw = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        seg = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2RGB)
        bb_tm = detect_right_hand_bbox(raw, seg)
        bb_co = detect_right_hand_bbox_by_color(raw, seg)
        if bb_tm is None or bb_co is None:
            continue
        if iou(bb_tm, bb_co) < 0.3:
            discrepancies.append((s, bb_tm, bb_co))

    print(f"Discrepancias encontradas: {len(discrepancies)}")
    show = discrepancies[:16]
    n_cols = 4
    n_rows = (len(show) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), squeeze=False)
    fig.suptitle(
        f"Discrepancias: template matching (verde) vs detección por color (rojo)\n"
        f"({len(show)} de {len(discrepancies)} total)",
        fontsize=12, fontweight="bold",
    )

    for ax, (sample, bb_tm, bb_co) in zip(axes.flat, show):
        fname = Path(sample.path).name
        raw_bgr = cv2.imread(sample.path)
        raw = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(raw)
        ax.add_patch(patches.Rectangle((bb_tm.x, bb_tm.y), bb_tm.w, bb_tm.h,
                                       linewidth=3, edgecolor="#00FF41", facecolor="none",
                                       label="template"))
        ax.add_patch(patches.Rectangle((bb_co.x, bb_co.y), bb_co.w, bb_co.h,
                                       linewidth=2.5, edgecolor="#FF1744", facecolor="none",
                                       linestyle="--", label="color"))
        ax.set_title(f"{fname} ({LSA16_NAMES[sample.label]})", fontsize=9, fontweight="bold")
        ax.axis("off")
    for ax in axes.flat[len(show):]:
        ax.axis("off")

    out_path = PROJECT_ROOT / "outputs" / "02_yolo" / "figures" / "discrepancies.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
