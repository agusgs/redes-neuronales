# Reconocimiento Robusto de la Lengua de Señas Argentina (LSA)

**Trabajo de Investigación — Redes Neuronales 2026-1**
Universidad Nacional de Quilmes — Departamento de Ciencia y Tecnología

**Autores:** Julián Santiago González Avendaño, Agustín García Smith
**Directora:** Ing. Roxana Martínez

---

## Resumen

Este trabajo evalúa dos paradigmas para el reconocimiento de configuraciones manuales (*handshapes*) de la Lengua de Señas Argentina sobre el dataset LSA16:

1. **CNN clásica con preprocesamiento manual** (réplica + mejora del paper Quiroga et al. 2017).
2. **YOLOv8 sobre imagen completa sin preprocesamiento** (contribución original).

**Resultado principal**: en el escenario realista (imagen completa 640×480 sin recortar), LeNet colapsa al 22% mientras YOLO alcanza 89.74% — una diferencia de +67 puntos porcentuales que valida el cambio de paradigma.

Lectura completa: ver `Trabajo_LSA16.pdf`.

---

## Estructura del paquete

```
trabajo-parcial/
├── README.md                          ← este archivo
├── Trabajo_LSA16.pdf                  ← documento principal (LEER ESTO)
├── Trabajo_LSA16.tex                  ← fuente LaTeX del documento
├── Propuesta_LSA16.tex                ← propuesta original
├── paper-lsa16-cnn.pdf                ← paper de referencia (Quiroga 2017)
├── paper-lsa16.pdf                    ← paper del dataset (Ronchetti 2016)
│
├── requirements.txt                   ← dependencias Python
├── .tool-versions                     ← versión de Python para asdf
│
├── src/                               ← código del pipeline (módulos reusables)
│   ├── data.py                          Dataset PyTorch + transforms + splits
│   ├── model.py                         LeNetLSA16 + ResNet-18 transfer + registry
│   ├── train.py                         Loop de entrenamiento
│   ├── experiment.py                    Stratified subsampling N runs
│   ├── preprocessing.py                 Alineamiento canónico (PCA + flip)
│   └── yolo_annotations.py              Generación de bboxes vía template matching
│
├── experiments/                       ← scripts ejecutables
│   ├── 01_baseline_paper.py             Entrena CNN (LeNet o ResNet-18)
│   ├── 02_yolo_train.py                 Entrena YOLOv8
│   └── 02_yolo_eval.py                  Evalúa YOLO con métricas comparables al CNN
│
├── scripts/                           ← utilidades
│   ├── preprocess_canonical.py          Genera dataset canónico desde segmented
│   ├── generate_yolo_dataset.py         Genera dataset_yolo/ con bboxes
│   ├── visualize_*.py                   Visualizaciones de validación
│   ├── analyze_failures.py              Análisis de seeds problemáticos
│   ├── probe_glove_colors*.py           Sondeo HSV del color de guantes
│   └── build_notebook_01.py             Generador del notebook
│
├── notebooks/
│   └── 01_baseline_paper_replication.ipynb  ← notebook técnico (narrativa + figuras)
│
├── lsa16_raw/                         ← 800 imágenes raw 640×480 (dataset original)
├── lsa16_segmented_right_hand/        ← 800 imágenes segmentadas (dataset original)
│
└── outputs/                           ← resultados experimentales
    ├── 01_baseline/
    │   ├── results.jsonl                Log machine-readable (1 línea por experimento)
    │   ├── experiments_log.md           Log human-readable (tabla markdown)
    │   ├── notebook_notes.md            Notas internas del proceso
    │   └── figures/                     PNG generados (preprocessing, etc.)
    └── 02_yolo-gpu/
        ├── results.jsonl                Log de runs YOLO
        ├── eval_*.json                  Métricas de evaluación
        ├── figures/                     Confusion matrices, IoU distributions, bboxes
        ├── yolov8s_gpu/                 Run principal (88 épocas, GPU)
        │   ├── results.csv              Métricas por época
        │   ├── results.png              Curvas de entrenamiento
        │   └── weights/best.pt          ★ MODELO ENTRENADO ★
        └── yolov8m_imgsz960-2/          Ablación con modelo más grande
            └── weights/best.pt
```

**Nota**: los directorios `lsa16_segmented_canonical/` y `dataset_yolo/` **no se incluyen** porque se generan automáticamente con los scripts de preprocesamiento (ver más abajo).

---

## Instrucciones de reproducción

### Requisitos

- **Python 3.12.5** (recomendado vía [asdf](https://asdf-vm.com) que lee `.tool-versions`, o cualquier distribución equivalente). En macOS con asdf, ver más abajo.
- **~3 GB de espacio libre** (datasets + entornos virtuales + datasets derivados).
- **CPU** suficiente para CNN. **GPU NVIDIA con CUDA** recomendada para YOLO (CPU es ~20× más lento pero funciona).
- En macOS: PyTorch+MPS (GPU de Apple Silicon) tiene un bug con BatchNorm que rompe el entrenamiento. Forzamos CPU en macOS.

### 1. Crear el entorno virtual e instalar dependencias

```bash
# Crear venv y activarlo
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# o:  .venv\Scripts\activate  # Windows

# Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt
```

**Nota sobre asdf en macOS**: si tu Python fue instalado con asdf, asegurate de que tenga soporte para `lzma`. Si `import lzma` falla, reinstalá Python con:

```bash
brew install xz openssl readline sqlite zlib bzip2
export LDFLAGS="-L$(brew --prefix xz)/lib -L$(brew --prefix openssl@3)/lib -L$(brew --prefix readline)/lib -L$(brew --prefix sqlite)/lib -L$(brew --prefix zlib)/lib -L$(brew --prefix bzip2)/lib"
export CPPFLAGS="-I$(brew --prefix xz)/include -I$(brew --prefix openssl@3)/include -I$(brew --prefix readline)/include -I$(brew --prefix sqlite)/include -I$(brew --prefix zlib)/include -I$(brew --prefix bzip2)/include"
asdf install python 3.12.5
```

### 2. Generar los datasets derivados

```bash
# Genera lsa16_segmented_canonical/ (mano alineada canónicamente, ~1 min)
python scripts/preprocess_canonical.py

# Genera dataset_yolo/ con bboxes en formato YOLO (~1 min)
python scripts/generate_yolo_dataset.py
```

### 3. Reproducir experimentos CNN

```bash
# Réplica del paper LeNet sobre canonical (~20 min en CPU)
python experiments/01_baseline_paper.py --mode canonical --n-runs 10 \
    --note "réplica paper"

# Test de data augmentation (~25 min)
python experiments/01_baseline_paper.py --mode canonical --augment --n-runs 10 \
    --note "canonical + augmentation"

# Transfer learning con ResNet-18 (~90 min en CPU, ~15 min en GPU)
python experiments/01_baseline_paper.py --mode canonical --model resnet18 --n-runs 10 \
    --note "ResNet18 transfer"

# Baseline raw (colapso esperado, ~25 min)
python experiments/01_baseline_paper.py --mode raw --n-runs 10 \
    --note "LeNet sobre raw"
```

Cada corrida se loguea en `outputs/01_baseline/results.jsonl` y `experiments_log.md`.

### 4. Reproducir experimentos YOLO

```bash
# Entrenamiento principal: YOLOv8s sobre raw, transfer learning desde COCO
# ~10 min en GPU NVIDIA, ~3-5 hs en CPU
python experiments/02_yolo_train.py \
    --model yolov8s.pt --imgsz 640 --batch 32 \
    --device 0 \    # ← cambiar a 'cpu' si no hay GPU
    --epochs 100 --name yolov8s_principal

# Ablación con modelo más grande (~40 min en GPU)
python experiments/02_yolo_train.py \
    --model yolov8m.pt --imgsz 960 --batch 16 \
    --device 0 \
    --epochs 100 --name yolov8m_ablation

# Evaluación de cada run sobre el split de test
python experiments/02_yolo_eval.py \
    --weights outputs/02_yolo/yolov8s_principal/weights/best.pt
```

**La primera vez que se ejecuta YOLO**, descarga automáticamente los pesos preentrenados de COCO (`yolov8s.pt` y/o `yolov8m.pt`, ~22-52MB cada uno).

### 5. Inferencia con el modelo ya entrenado (sin reentrenar)

Si solo querés usar el modelo entrenado que viene en el paquete:

```bash
python experiments/02_yolo_eval.py \
    --weights outputs/02_yolo-gpu/yolov8s_gpu/weights/best.pt
```

Esto genera:
- `outputs/02_yolo-gpu/eval_yolov8s_gpu.json` (métricas detalladas)
- `outputs/02_yolo-gpu/figures/confusion_yolov8s_gpu.png` (matriz de confusión)
- `outputs/02_yolo-gpu/figures/iou_distribution_yolov8s_gpu.png`

### 6. Generar el notebook técnico

El notebook se genera programáticamente desde un script Python:

```bash
python scripts/build_notebook_01.py
jupyter notebook notebooks/01_baseline_paper_replication.ipynb
```

El notebook lee directamente de los archivos JSON/CSV de outputs/, así que refleja automáticamente cualquier nuevo experimento que se haya corrido.

### 7. Compilar el documento LaTeX

```bash
# Con Tectonic (recomendado, instala el package que falte automáticamente)
tectonic Trabajo_LSA16.tex

# O con pdflatex (requiere bibliography manual)
pdflatex Trabajo_LSA16.tex
pdflatex Trabajo_LSA16.tex  # 2da pasada para resolver referencias
```

---

## Resultados obtenidos (resumen)

Detalle completo en `Trabajo_LSA16.pdf` (sección 7). Resumen ejecutivo:

| Modelo | Input | Accuracy |
|---|---|---:|
| LeNet (paper, 100 runs) | mano segmentada + canonical 128×128 | 96.18% |
| **LeNet (nuestra réplica, 10 runs)** | mano segmentada + canonical 128×128 | **91.25% ± 7.83%** |
| LeNet + augmentation (10 runs) | mano segmentada + canonical 128×128 | 91.62% ± 6.71% |
| **ResNet-18 transfer (10 runs)** | mano segmentada + canonical 128×128 | **97.62% ± 1.63%** |
| LeNet sobre raw (10 runs) | imagen completa 640×480 | 22.12% ± 5.76% |
| **YOLOv8s sobre raw** | imagen completa 640×480 | **89.74%** |
| YOLOv8m @ 960 sobre raw | imagen completa 640×480 | 90.60% |

---

## Reportar problemas

Si al reproducir algún experimento obtenés resultados muy distintos a los reportados, es probable que sea por:

- **Versión de PyTorch distinta a 2.5.1**: pueden cambiar comportamientos numéricos sutiles. Sugerimos respetar `requirements.txt`.
- **MPS habilitado en macOS**: bloqueado por código; si tu setup lo activa, fuerza `--device cpu`.
- **Datasets derivados sin regenerar**: si modificaste `src/preprocessing.py` o `src/yolo_annotations.py`, hay que regenerar los datasets antes de reentrenar.
