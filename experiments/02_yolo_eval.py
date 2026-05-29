"""Evaluación de un modelo YOLO entrenado sobre LSA16.

Calcula métricas comparables con el baseline CNN:
- Classification accuracy: por imagen, tomar la clase del top-1 detection y
  comparar con ground truth. Es directamente comparable con la accuracy del CNN.
- Average IoU sobre los matches: cuán bien localiza la mano cuando la detecta.
- Detection rate: % de imágenes donde se detectó al menos un objeto.
- mAP@50, mAP@50-95: métricas de detección estándar.
- Matriz de confusión 16×16.

Salidas:
- outputs/02_yolo/eval_<run-name>.json (machine-readable)
- outputs/02_yolo/figures/confusion_<run-name>.png
- outputs/02_yolo/figures/iou_distribution_<run-name>.png

Ejemplo:
    python experiments/02_yolo_eval.py --weights outputs/02_yolo/yolov8s_20260522_2030/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src.data import LSA16_NAMES  # noqa: E402


DATASET_YAML = PROJECT_ROOT / "dataset_yolo" / "dataset.yaml"
TEST_IMG_DIR = PROJECT_ROOT / "dataset_yolo" / "images" / "test"
TEST_LBL_DIR = PROJECT_ROOT / "dataset_yolo" / "labels" / "test"


def _read_gt_label(label_path: Path) -> tuple[int, tuple[float, float, float, float]] | None:
    """Lee un .txt YOLO con UN sola línea y devuelve (class_id, (xc, yc, w, h))."""
    if not label_path.exists():
        return None
    line = label_path.read_text().strip().splitlines()[0]
    parts = line.split()
    cls = int(parts[0])
    xc, yc, w, h = map(float, parts[1:5])
    return cls, (xc, yc, w, h)


def _yolo_to_xyxy(box_norm, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    xc, yc, w, h = box_norm
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return x1, y1, x2, y2


def _iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True,
                        help="Ruta a best.pt (ej: outputs/02_yolo/yolov8s_.../weights/best.pt)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Umbral de confianza para considerar una detección.")
    parser.add_argument("--iou-match-threshold", type=float, default=0.5,
                        help="IoU mínimo para considerar que una detección matchea la GT bbox.")
    args = parser.parse_args()

    weights_path = Path(args.weights).resolve()
    run_name = weights_path.parent.parent.name
    # Output dir = el padre de la carpeta del run (ej. outputs/02_yolo-gpu/)
    # Esto hace que la evaluación quede pegada a donde están los pesos.
    output_dir = weights_path.parent.parent.parent
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Import después de validar args
    from ultralytics import YOLO

    print(f"Cargando pesos: {weights_path}")
    model = YOLO(str(weights_path))

    # 1) Métricas de detección estándar via Ultralytics
    print(f"\nCorriendo Ultralytics val sobre split=test...")
    val_metrics = model.val(data=str(DATASET_YAML), split="test",
                            device=args.device, verbose=False)

    # 2) Inferencia imagen por imagen para calcular accuracy + IoU comparable con CNN
    print(f"\nCorriendo inferencia sobre {TEST_IMG_DIR}/")
    test_imgs = sorted(TEST_IMG_DIR.glob("*.png"))
    results = model.predict(source=[str(p) for p in test_imgs],
                            conf=args.conf, device=args.device, verbose=False)

    y_true, y_pred = [], []
    iou_matches = []
    detection_rate = 0
    misses = 0  # imágenes donde el modelo no detectó nada
    misses_files = []
    mismatches = []  # (fname, gt_cls, pred_cls) cuando hay clasificación incorrecta

    for img_path, res in zip(test_imgs, results):
        gt = _read_gt_label(TEST_LBL_DIR / img_path.with_suffix(".txt").name)
        if gt is None:
            continue
        gt_cls, gt_box_norm = gt
        gt_xyxy = _yolo_to_xyxy(gt_box_norm, res.orig_shape[1], res.orig_shape[0])

        if res.boxes is None or len(res.boxes) == 0:
            misses += 1
            misses_files.append({"file": img_path.name, "gt": LSA16_NAMES[gt_cls]})
            y_true.append(gt_cls)
            y_pred.append(-1)  # sin detección
            continue

        detection_rate += 1
        # Top-1 detection por confianza
        confs = res.boxes.conf.cpu().numpy()
        top_idx = int(np.argmax(confs))
        pred_cls = int(res.boxes.cls[top_idx].cpu().item())
        pred_xyxy = res.boxes.xyxy[top_idx].cpu().numpy().tolist()
        iou = _iou_xyxy(pred_xyxy, gt_xyxy)

        y_true.append(gt_cls)
        y_pred.append(pred_cls)
        iou_matches.append(iou)
        if pred_cls != gt_cls:
            mismatches.append((img_path.name, gt_cls, pred_cls))

    # Stats
    n_total = len(y_true)
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n_total if n_total else 0
    accuracy_detected_only = sum(1 for t, p in zip(y_true, y_pred) if t == p and p != -1) / max(1, detection_rate)
    mean_iou = float(np.mean(iou_matches)) if iou_matches else 0.0
    median_iou = float(np.median(iou_matches)) if iou_matches else 0.0

    print(f"\n{'='*64}")
    print(f"  Evaluación — {run_name}")
    print(f"{'='*64}")
    print(f"  Imágenes test           : {n_total}")
    print(f"  Detección rate          : {detection_rate}/{n_total} ({detection_rate/n_total*100:.1f}%)")
    print(f"  Sin detección (miss)    : {misses}")
    print(f"  Classification accuracy : {accuracy*100:.2f}%  (sobre todas)")
    print(f"    → solo detected only  : {accuracy_detected_only*100:.2f}%")
    print(f"  IoU promedio            : {mean_iou:.4f}")
    print(f"  IoU mediana             : {median_iou:.4f}")
    print(f"  mAP@50                  : {val_metrics.box.map50:.4f}")
    print(f"  mAP@50-95               : {val_metrics.box.map:.4f}")
    print(f"  Precision (Ultralytics) : {val_metrics.box.mp:.4f}")
    print(f"  Recall (Ultralytics)    : {val_metrics.box.mr:.4f}")
    print(f"{'='*64}")

    # Per-class report (excluyendo -1)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    detected_mask = y_pred_arr != -1
    if detected_mask.any():
        print(f"\nClassification report (solo imágenes con detección):")
        print(classification_report(
            y_true_arr[detected_mask], y_pred_arr[detected_mask],
            target_names=LSA16_NAMES, digits=3, zero_division=0,
        ))

    # Confusion matrix
    cm = confusion_matrix(y_true_arr[detected_mask], y_pred_arr[detected_mask],
                          labels=list(range(16)))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LSA16_NAMES, yticklabels=LSA16_NAMES, ax=ax)
    ax.set_title(f"Matriz de confusión — {run_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicción"); ax.set_ylabel("Ground truth")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = fig_dir / f"confusion_{run_name}.png"
    plt.savefig(cm_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Matriz de confusión: {cm_path}")

    # IoU histogram
    if iou_matches:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(iou_matches, bins=30, range=(0, 1), color="#3498DB", alpha=0.85)
        ax.axvline(mean_iou, color="black", ls="--", label=f"mean={mean_iou:.3f}")
        ax.axvline(median_iou, color="red", ls=":", label=f"median={median_iou:.3f}")
        ax.set_xlabel("IoU entre detección top-1 y ground truth bbox")
        ax.set_ylabel("# imágenes")
        ax.set_title(f"Distribución de IoU — {run_name}", fontweight="bold")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        iou_path = fig_dir / f"iou_distribution_{run_name}.png"
        plt.savefig(iou_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"✓ Distribución de IoU: {iou_path}")

    # Persistir JSON
    eval_record = {
        "run_name": run_name,
        "weights": str(weights_path),
        "n_test": n_total,
        "detection_rate": detection_rate / n_total,
        "misses": misses,
        "misses_files": misses_files,
        "classification_accuracy": accuracy,
        "classification_accuracy_detected_only": accuracy_detected_only,
        "mean_iou": mean_iou,
        "median_iou": median_iou,
        "mAP50": float(val_metrics.box.map50),
        "mAP50_95": float(val_metrics.box.map),
        "precision": float(val_metrics.box.mp),
        "recall": float(val_metrics.box.mr),
        "mismatches": [{"file": fn, "gt": LSA16_NAMES[gt], "pred": LSA16_NAMES[pr] if pr >= 0 else "no_detect"}
                       for fn, gt, pr in mismatches],
    }
    eval_path = output_dir / f"eval_{run_name}.json"
    eval_path.write_text(json.dumps(eval_record, indent=2), encoding="utf-8")
    print(f"✓ Eval completo: {eval_path}")


if __name__ == "__main__":
    main()
