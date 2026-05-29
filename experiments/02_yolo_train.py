"""Entrena YOLOv8 sobre el dataset LSA16 generado en dataset_yolo/.

Logueo:
- Métricas estándar de Ultralytics (mAP@50, mAP@50-95, precision, recall) en
  outputs/02_yolo/<run-name>/results.csv y curvas en PNG.
- Pesos del mejor modelo en outputs/02_yolo/<run-name>/weights/best.pt.
- Resumen final en outputs/02_yolo/results.jsonl (append-only, machine-readable)
  con un registro por run (timestamp, modelo, hiperparámetros, métricas).

Ejemplos:
    # Entrenamiento default (yolov8s, 50 épocas, imgsz=640)
    python experiments/02_yolo_train.py

    # Más rápido para iterar
    python experiments/02_yolo_train.py --model yolov8n.pt --imgsz 480 --epochs 30

    # Más preciso
    python experiments/02_yolo_train.py --model yolov8m.pt --epochs 80
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASET_YAML = PROJECT_ROOT / "dataset_yolo" / "dataset.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "02_yolo"
RESULTS_JSONL = OUTPUT_DIR / "results.jsonl"


def append_jsonl(record: dict) -> None:
    RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolov8s.pt",
                        help="Modelo base preentrenado en COCO (yolov8n/s/m/l/x.pt)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Tamaño de entrada. 640 es default Ultralytics. "
                             "320-480 acelera el entrenamiento.")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default=None,
                        help="Nombre del run (subcarpeta en outputs/02_yolo). "
                             "Si se omite, genera uno con timestamp.")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping: épocas sin mejora antes de cortar.")
    parser.add_argument("--device", default="cpu",
                        help="cpu|mps|0,1,... — usar 'cpu' por defecto, MPS rompe BN.")
    parser.add_argument("--note", default="", help="Anotación libre para el log.")
    args = parser.parse_args()

    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"No existe {DATASET_YAML}. Generá el dataset primero:\n"
            f"  python scripts/generate_yolo_dataset.py"
        )

    # Import después de validar — ultralytics tarda en cargar
    from ultralytics import YOLO

    run_name = args.name or f"{Path(args.model).stem}_{time.strftime('%Y%m%d_%H%M')}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=" * 60)
    print(f"  Entrenando YOLO sobre LSA16")
    print(f"=" * 60)
    print(f"  Modelo base   : {args.model}")
    print(f"  Dataset       : {DATASET_YAML}")
    print(f"  Épocas        : {args.epochs} (patience={args.patience})")
    print(f"  Imagen size   : {args.imgsz}")
    print(f"  Batch size    : {args.batch}")
    print(f"  Device        : {args.device}")
    print(f"  Output        : {OUTPUT_DIR / run_name}")
    print(f"=" * 60)
    print()

    model = YOLO(args.model)
    t0 = time.time()

    # Ultralytics arguments:
    # - fliplr=0.0: NO flip horizontal (cambiaría la quiralidad de la mano).
    # - flipud=0.0: NO flip vertical (no tiene sentido para handshapes).
    # - degrees, translate, scale: jitter geométrico ligero.
    # - hsv_*: jitter de color leve.
    # - mosaic=0.0: deshabilitar mosaic (junta varias imágenes en una; no aporta
    #   acá porque cada raw tiene UNA mano centrada y mosaic distorsionaría la
    #   relación espacial que YOLO necesita aprender).
    results = model.train(
        data=str(DATASET_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(OUTPUT_DIR),
        name=run_name,
        patience=args.patience,
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.0,
        degrees=10.0,
        translate=0.1,
        scale=0.1,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        exist_ok=False,
        verbose=True,
    )

    elapsed = time.time() - t0
    print(f"\n✓ Training terminado en {elapsed/60:.1f} min")

    # Validation final
    metrics = model.val(data=str(DATASET_YAML), split="test", device=args.device)
    print(f"\nMétricas en TEST:")
    print(f"  mAP@50      : {metrics.box.map50:.4f}")
    print(f"  mAP@50-95   : {metrics.box.map:.4f}")
    print(f"  Precision   : {metrics.box.mp:.4f}")
    print(f"  Recall      : {metrics.box.mr:.4f}")

    # Log a JSONL
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_name": run_name,
        "model": args.model,
        "epochs_target": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "elapsed_min": round(elapsed / 60, 2),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "note": args.note,
    }
    append_jsonl(record)
    print(f"\n✓ Registro guardado en {RESULTS_JSONL}")
    print(f"✓ Pesos del mejor modelo: {OUTPUT_DIR / run_name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
