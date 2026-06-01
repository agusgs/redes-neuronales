"""Script para ejecutar LOSO en YOLOv8.

Genera de forma dinámica los archivos .txt y .yaml requeridos por Ultralytics 
para cada fold (sujeto excluido), sin duplicar las imágenes físicamente.

Ejemplo:
    python experiments/04_yolo_loso.py --model yolov8s --epochs 100
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ultralytics se importa localmente para que el script no falle si no está instalado
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


OUT_DIR = PROJECT_ROOT / "outputs" / "04_yolo_loso"
RESULTS_JSONL = OUT_DIR / "results.jsonl"
YOLO_IMAGES_DIR = PROJECT_ROOT / "dataset_yolo" / "images"

# Clases en el orden exacto
NAMES = [
    "Five", "Four", "Horns", "Curve", "Fingers", "Double", "Hook", "Index",
    "L", "Flat", "Mitten", "Beak", "Thumb", "Fist", "Telephone", "V",
]


def _append_jsonl(record: dict) -> None:
    RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def setup_fold_yaml(test_subject: int, all_images: list[Path], fold_dir: Path) -> Path:
    """Crea los txt con paths absolutos y el yaml para un fold específico."""
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    train_txt = fold_dir / "train.txt"
    val_txt = fold_dir / "val.txt"
    
    train_paths = []
    val_paths = []
    
    for img_path in all_images:
        # El formato es {class}_{subject}_{rep}.png
        # Ej: 14_10_003.png -> subject es el index 1
        parts = img_path.stem.split("_")
        if len(parts) >= 3:
            subject = int(parts[1])
            if subject == test_subject:
                val_paths.append(str(img_path.absolute()))
            else:
                train_paths.append(str(img_path.absolute()))

    train_txt.write_text("\n".join(train_paths))
    val_txt.write_text("\n".join(val_paths))
    
    yaml_path = fold_dir / "dataset.yaml"
    yaml_content = f"""
path: {str(fold_dir.absolute())}
train: {str(train_txt.absolute())}
val: {str(val_txt.absolute())}

names:
"""
    for i, name in enumerate(NAMES):
        yaml_content += f"  {i}: {name}\n"
        
    yaml_path.write_text(yaml_content)
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolov8s", help="Modelo YOLO base (ej: yolov8s o yolov8m)")
    parser.add_argument("--epochs", type=int, default=100, help="Épocas por fold")
    parser.add_argument("--batch", type=int, default=32, help="Batch size (bajar a 16 si usas yolov8m)")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolución de entrada")
    parser.add_argument("--device", default="0", help="GPU device (0) o 'cpu'")
    parser.add_argument("--note", default="", help="Nota para el log")
    args = parser.parse_args()

    if YOLO is None:
        print("Error: ultralytics no está instalado. Ejecuta: pip install ultralytics")
        sys.exit(1)

    all_images = list(YOLO_IMAGES_DIR.rglob("*.png")) + list(YOLO_IMAGES_DIR.rglob("*.jpg"))
    if not all_images:
        print(f"No se encontraron imágenes en {YOLO_IMAGES_DIR}")
        sys.exit(1)

    # Buscar sujetos únicos
    subjects = set()
    for p in all_images:
        parts = p.stem.split("_")
        if len(parts) >= 3:
            subjects.add(int(parts[1]))
    subjects = sorted(list(subjects))

    print(f"\n============================================================")
    print(f" Iniciando YOLOv8 LOSO Cross-Validation")
    print(f" Modelo : {args.model}")
    print(f" Sujetos: {len(subjects)} encontrados {subjects}")
    print(f"============================================================\n")

    fold_results = []
    t0 = time.time()

    for test_subject in subjects:
        print(f"\n{'='*60}")
        print(f" FOLD: Testeando Sujeto {test_subject}")
        print(f"{'='*60}\n")
        
        fold_dir = OUT_DIR / f"fold_configs" / f"subject_{test_subject}"
        yaml_path = setup_fold_yaml(test_subject, all_images, fold_dir)
        
        # Iniciar YOLO
        model = YOLO(f"{args.model}.pt")
        
        # Entrenar
        # Parámetros estrictos para LSA16 (sin flip, sin mosaic)
        results = model.train(
            data=str(yaml_path),
            epochs=args.epochs,
            patience=20,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project=str(OUT_DIR),
            name=f"run_subject_{test_subject}",
            exist_ok=True,
            # Augmentations bloqueadas
            fliplr=0.0,
            flipud=0.0,
            mosaic=0.0,
            # Augmentations permitidas
            degrees=10.0,
            translate=0.1,
            scale=0.1
        )
        
        # Extraer métricas de validación del objeto results devuelto por model.train
        # metrics.box.map50 es el mAP@50
        val_map50 = float(results.box.map50)
        
        # Para accuracy de clasificación, YOLO lo integra en fitness, pero map50 es el estándar de detección
        print(f"\n>>> Fold Sujeto {test_subject} completado. mAP@50 = {val_map50:.4f}\n")
        
        fold_results.append({
            "subject": test_subject,
            "test_map50": val_map50
        })

    elapsed = time.time() - t0
    maps = np.array([f["test_map50"] for f in fold_results])
    mean_map = float(maps.mean())
    std_map = float(maps.std(ddof=0))
    
    print("\n============================================================")
    print(f" RESUMEN YOLO LOSO - {args.model}")
    print("============================================================")
    for f in fold_results:
        print(f"  Sujeto {f['subject']:2d}: mAP@50 = {f['test_map50']:.4f}")
    print("-" * 60)
    print(f"  Media Global: {mean_map:.4f}")
    print(f"  Std Dev     : {std_map:.4f}")
    print(f"  Tiempo total: {elapsed/60:.1f} minutos")
    print("============================================================\n")

    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": f"YOLOv8 LOSO",
        "model": args.model,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "mean_map50": mean_map,
        "std_map50": std_map,
        "folds": fold_results,
        "note": args.note,
        "elapsed_min": round(elapsed / 60, 2),
    }
    _append_jsonl(record)
    print(f"✓ Resultados guardados en {RESULTS_JSONL}")


if __name__ == "__main__":
    import numpy as np  # Import local inside main for the metrics calculation
    main()
