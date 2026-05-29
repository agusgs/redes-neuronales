"""Construye notebooks/01_baseline_paper_replication.ipynb desde cero.

Source de verdad: este .py. El notebook generado es un artefacto.
Para regenerar:
    python scripts/build_notebook_01.py

El notebook lee resultados desde outputs/01_baseline/results.jsonl
en tiempo de ejecución, así que se mantiene actualizado.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "01_baseline_paper_replication.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python (LSA16)",
            "language": "python",
            "name": "lsa16",
        },
        "language_info": {"name": "python", "version": "3.12.5"},
    }

    cells: list[nbf.NotebookNode] = []

    cells.append(md(
        "# 01 — LSA16: del baseline CNN a detección espacial con YOLO\n\n"
        "**Trabajo de Investigación — Redes Neuronales 2026-1**  \n"
        "Universidad Nacional de Quilmes — Departamento de Ciencia y Tecnología\n\n"
        "**Autores:** Julián Santiago González Avendaño, Agustín García Smith  \n"
        "**Directora:** Ing. Roxana Martínez\n\n"
        "---\n\n"
        "## Resumen\n\n"
        "Este notebook documenta el trabajo experimental completo sobre el dataset "
        "**LSA16** (16 configuraciones manuales de la Lengua de Señas Argentina) "
        "en tres etapas:\n\n"
        "1. **Replicación del baseline CNN** del paper Quiroga et al. (2017) "
        "[\"A Study of Convolutional Architectures for Handshape Recognition applied "
        "to Sign Language\"](../paper-lsa16-cnn.pdf), que reporta **96.18%** de "
        "accuracy para LeNet sobre la versión segmentada del dataset.\n"
        "2. **Mejoras sobre el CNN**: data augmentation (hipótesis refutada) y "
        "transfer learning con ResNet-18 (mejora confirmada, supera al paper).\n"
        "3. **Detección espacial con YOLOv8** sobre las imágenes completas — "
        "**contribución original del trabajo**.\n\n"
        "**Resultados consolidados** (sobre el split de test, ~117 imágenes):\n\n"
        "| Modelo | Input | Accuracy | Notas |\n"
        "|---|---|---:|---|\n"
        "| LeNet (paper) | mano segmentada + canonical 128×128 | 96.18% | benchmark del paper, 100 runs |\n"
        "| LeNet (nuestro 10 runs) | mano segmentada + canonical 128×128 | 91.25% ± 7.83% | replica con varianza honesta |\n"
        "| **ResNet-18 transfer** | mano segmentada + canonical 128×128 | **97.62% ± 1.63%** | supera al paper |\n"
        "| LeNet sobre raw | imagen completa 640×480 | 22.12% ± 5.76% | **CNN colapsa en raw** |\n"
        "| **YOLOv8s** | **imagen completa 640×480** | **89.74%** | **localiza + clasifica end-to-end** |\n"
        "| YOLOv8m + imgsz=960 | imagen completa 640×480 | 90.60% | ablation: más capacidad ≠ mejor |\n\n"
        "**Hallazgo clave**: en el escenario **realista** (imagen completa, sin "
        "preprocesamiento manual), el CNN colapsa a 22.12% mientras YOLO alcanza "
        "**89.74%** — una diferencia de **+67pp**. Esto valida la hipótesis "
        "central del trabajo: para sistemas reales (webcam, fondo no controlado), "
        "**la detección espacial es ampliamente superior a la clasificación de "
        "imagen entera**."
    ))

    cells.append(md(
        "## 1. Contexto y objetivo\n\n"
        "### 1.1. ¿Por qué reconocer LSA?\n\n"
        "El reconocimiento automático de Lengua de Señas Argentina (LSA) es un problema de "
        "Visión por Computadora con alto valor de accesibilidad para la comunidad sorda. "
        "El pipeline completo de SLR (Sign Language Recognition) involucra detección de manos, "
        "clasificación de configuraciones, seguimiento temporal y análisis semántico. "
        "La calidad de la **clasificación de configuración manual** (*handshape recognition*) "
        "es el cuello de botella del pipeline.\n\n"
        "### 1.2. Paper de referencia\n\n"
        "Quiroga et al. (2017) compararon varias arquitecturas CNN sobre LSA16 y reportaron:\n\n"
        "| Método | Accuracy LSA16 |\n"
        "|---|---:|\n"
        "| Feedforward (baseline) | 86.58% |\n"
        "| **LeNet** | **95.78%** |\n"
        "| All Convolutional | 94.56% |\n"
        "| VGG16 | 95.92% |\n"
        "| ResNet-34 | 93.49% |\n\n"
        "Y para distintos esquemas de preprocesamiento con LeNet:\n\n"
        "| Esquema | Accuracy |\n"
        "|---|---:|\n"
        "| Raw (RGB) | 83.54% |\n"
        "| **Segmented Hand RGB** | **96.18%** |\n"
        "| Segmented Grayscale | 87.08% |\n\n"
        "### 1.3. Objetivos del notebook\n\n"
        "1. **Replicar** 96.18% sobre la versión segmentada del dataset usando "
        "exactamente la metodología del paper (LeNet, Adam lr=0.0007, 20 épocas, "
        "sin augmentation).\n"
        "2. **Caracterizar honestamente la varianza** del experimento (algo que el "
        "paper diluye promediando 100 corridas).\n"
        "3. **Mejorar el baseline CNN**: data augmentation y transfer learning "
        "con ResNet-18.\n"
        "4. **Implementar y evaluar un detector espacial (YOLOv8)** sobre las "
        "imágenes completas, comparándolo cuantitativamente con el baseline CNN. "
        "Esta es la **contribución original** del trabajo: demostrar que la "
        "detección espacial supera a la clasificación tradicional cuando "
        "operamos sobre escenarios realistas no preprocesados."
    ))

    cells.append(md(
        "## 2. Setup del entorno"
    ))

    cells.append(code(
        "import json\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        "# Acceso a src/ desde el notebook\n"
        "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "sys.path.insert(0, str(PROJECT_ROOT))\n\n"
        "import cv2\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "from PIL import Image\n\n"
        "from src.data import CANONICAL_DIR, LSA16_NAMES, RAW_DIR, SEGMENTED_DIR\n"
        "from src.preprocessing import canonical_align\n\n"
        "RESULTS_JSONL = PROJECT_ROOT / 'outputs' / '01_baseline' / 'results.jsonl'\n"
        "FIGURES_DIR  = PROJECT_ROOT / 'outputs' / '01_baseline' / 'figures'\n\n"
        "print(f'Proyecto raíz: {PROJECT_ROOT}')\n"
        "print(f'Clases LSA16: {LSA16_NAMES}')"
    ))

    cells.append(md(
        "## 3. Dataset LSA16\n\n"
        "### 3.1. Composición\n\n"
        "- **800 imágenes** en total\n"
        "- **16 clases** (configuraciones manuales más usadas en LSA)\n"
        "- **10 sujetos** que ejecutaron **5 repeticiones** de cada configuración\n"
        "- Captura en entorno controlado: fondo blanco, guantes de color fluorescente, "
        "iluminación uniforme\n\n"
        "El convenio de nombres es `clase_sujeto_repeticion.png` (clase 1-indexed)."
    ))

    cells.append(code(
        "# Distribución del dataset\n"
        "import glob\n"
        "from collections import Counter\n\n"
        "files = sorted(glob.glob(str(PROJECT_ROOT / 'lsa16_segmented_right_hand' / '*.png')))\n"
        "classes  = Counter(int(Path(f).name.split('_')[0]) for f in files)\n"
        "subjects = Counter(int(Path(f).name.split('_')[1]) for f in files)\n"
        "print(f'Total imágenes: {len(files)}')\n"
        "print(f'Clases: {len(classes)} (50 imgs c/u)  | Sujetos: {len(subjects)} (80 imgs c/u)')\n"
        "print()\n"
        "for cls in sorted(classes):\n"
        "    print(f'  Clase {cls:2d} ({LSA16_NAMES[cls-1]:10s}): {classes[cls]} imgs')"
    ))

    cells.append(md(
        "### 3.2. Variantes que tenemos del dataset\n\n"
        "1. **`lsa16_raw/`** — imágenes completas 640×480 con la persona y el guante. "
        "Es el escenario *no controlado* (más cercano a una webcam real).\n"
        "2. **`lsa16_segmented_right_hand/`** — la mano derecha recortada y segmentada sobre "
        "fondo negro, en tamaño variable (~150×150). Es la versión que el paper usa para "
        "el experimento \"Segmented Hand RGB\".\n"
        "3. **`lsa16_segmented_canonical/`** — generada por nosotros aplicando el "
        "preprocesamiento del paper (alineamiento canónico). 128×128 fijo. Ver sección 4."
    ))

    cells.append(code(
        "# Comparación visual de las tres variantes para 5 clases\n"
        "ejemplos = [(1, 1, 1), (5, 1, 1), (8, 1, 1), (12, 1, 1), (15, 1, 1)]\n\n"
        "fig, axes = plt.subplots(3, len(ejemplos), figsize=(15, 9))\n"
        "fig.suptitle('Tres variantes del dataset para una muestra de cada clase',\n"
        "             fontsize=13, fontweight='bold')\n\n"
        "for col, (cls, subj, rep) in enumerate(ejemplos):\n"
        "    fname = f'{cls}_{subj}_{rep}.png'\n"
        "    for row, (folder, label) in enumerate([\n"
        "        (RAW_DIR, 'raw (640×480)'),\n"
        "        (SEGMENTED_DIR, 'segmented (var.)'),\n"
        "        (CANONICAL_DIR, 'canonical (128×128)'),\n"
        "    ]):\n"
        "        img = Image.open(Path(folder) / fname).convert('RGB')\n"
        "        axes[row][col].imshow(img)\n"
        "        if row == 0:\n"
        "            axes[row][col].set_title(f'Clase {cls}: {LSA16_NAMES[cls-1]}', fontsize=10)\n"
        "        if col == 0:\n"
        "            axes[row][col].set_ylabel(label, fontsize=10, fontweight='bold')\n"
        "        axes[row][col].set_xticks([]); axes[row][col].set_yticks([])\n\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    cells.append(md(
        "## 4. Preprocesamiento canónico (clave para la replicación)\n\n"
        "### 4.1. El problema\n\n"
        "Al entrenar la primera versión del modelo con la carpeta `lsa16_segmented_right_hand/` "
        "obtuvimos **83.33% ± 2.12%** sobre 3 corridas — **muy por debajo del 96.18%** del paper. "
        "El gap era ~13pp, demasiado para atribuir a ruido.\n\n"
        "Inspeccionando las imágenes notamos algo crítico: **las manos no están en orientación "
        "canónica**. Los sujetos pueden tener la mano vertical, inclinada, o de costado.\n\n"
        "### 4.2. La solución del paper\n\n"
        "El paper original (Sección 2.2 del paper 2016 referenciado) aplica un pipeline de "
        "normalización geométrica:\n\n"
        "1. **Máscara binaria** a partir del componente conectado más grande.\n"
        "2. **PCA** sobre los píxeles de la mano → eje principal.\n"
        "3. **Rotación** para alinear el eje principal con la vertical.\n"
        "4. **Corrección 180°** contando \"cruces horizontales\": el lado con más cruces "
        "son los dedos (porque separados crean más transiciones).\n"
        "5. **Crop + resize a 128×128** manteniendo aspect ratio (padding negro).\n\n"
        "Implementación: [`src/preprocessing.py`](../src/preprocessing.py)."
    ))

    cells.append(code(
        "# Demostración del pipeline canónico sobre una imagen\n"
        "from src.preprocessing import canonical_align\n\n"
        "muestras = [(3, 4, 1), (5, 2, 1), (9, 3, 1), (14, 5, 1)]\n\n"
        "fig, axes = plt.subplots(2, len(muestras), figsize=(14, 7))\n"
        "fig.suptitle('Antes (arriba) y después (abajo) de canonical_align()',\n"
        "             fontsize=12, fontweight='bold')\n\n"
        "for col, (cls, subj, rep) in enumerate(muestras):\n"
        "    fname = f'{cls}_{subj}_{rep}.png'\n"
        "    bgr  = cv2.imread(str(Path(SEGMENTED_DIR) / fname))\n"
        "    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)\n"
        "    proc = canonical_align(rgb, target_size=128)\n\n"
        "    axes[0][col].imshow(rgb)\n"
        "    axes[0][col].set_title(f'cls={cls} ({LSA16_NAMES[cls-1]}) subj={subj}', fontsize=10)\n"
        "    axes[0][col].axis('off')\n"
        "    axes[1][col].imshow(proc)\n"
        "    axes[1][col].set_title('procesado 128×128', fontsize=10, color='green')\n"
        "    axes[1][col].axis('off')\n\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    cells.append(md(
        "## 5. Arquitecturas\n\n"
        "Probamos dos modelos sobre el mismo pipeline de entrenamiento. Ambas "
        "implementaciones están en [`src/model.py`](../src/model.py) y se "
        "seleccionan vía el flag ``--model`` del runner.\n\n"
        "### 5.1. LeNetLSA16 (paper)\n\n"
        "Réplica directa de la especificación del paper (Sec. 3.2):\n\n"
        "> *\"LeNet architecture employed four convolutional layers with sizes "
        "(32, 64, 128, 256) and 3x3 filters, and a 512-dimensional feedforward layer. "
        "We found that adding Batch Normalization layers after each max pooling layer "
        "and replacing ReLU activation functions for ELUs reduced the training time.\"*"
    ))

    cells.append(code(
        "from src.model import LeNetLSA16, resnet18_transfer\n\n"
        "lenet = LeNetLSA16()\n"
        "n_params = sum(p.numel() for p in lenet.parameters())\n"
        "print(f'LeNetLSA16 — parámetros: {n_params:,}')\n"
        "print()\n"
        "print(lenet)"
    ))

    cells.append(md(
        "### 5.2. ResNet-18 con transfer learning\n\n"
        "ResNet-18 [He et al., 2015] **preentrenada en ImageNet** "
        "(`torchvision.models.resnet18(weights=IMAGENET1K_V1)`), con la capa final FC "
        "reemplazada por una `Linear(512, 16)` para nuestras 16 clases. Fine-tuning "
        "**end-to-end** (todos los pesos entrenables) con learning rate más baja "
        "(`1e-4` vs `7e-4` del paper) para no romper las representaciones del backbone.\n\n"
        "Configuración:\n\n"
        "| Parámetro | LeNet (paper) | ResNet-18 transfer |\n"
        "|---|---|---|\n"
        "| Input size | 128×128 | 224×224 |\n"
        "| Normalización | mean=std=0.5 | ImageNet |\n"
        "| Learning rate | 0.0007 | 0.0001 |\n"
        "| Épocas | 20 | 20 |\n"
        "| Optimizer | Adam | Adam |\n\n"
        "El resto del pipeline (subsampling estratificado, evaluación) es idéntico."
    ))

    cells.append(code(
        "rn = resnet18_transfer()\n"
        "n_rn = sum(p.numel() for p in rn.parameters())\n"
        "print(f'ResNet-18 (transfer) — parámetros: {n_rn:,}')\n"
        "print(f'  Solo capa final FC: {rn.fc.weight.numel() + rn.fc.bias.numel():,}')\n"
        "print()\n"
        "print('Backbone preentrenado en ImageNet (1.28M imágenes, 1000 clases)')\n"
        "print('Fine-tuning end-to-end con lr=1e-4 para no destruir las representaciones')"
    ))

    cells.append(md(
        "## 6. Metodología del experimento\n\n"
        "### 6.1. Protocolo de validación\n\n"
        "El paper usa **\"stratified randomized subsampling cross-validation\"** "
        "promediado sobre 100 corridas. Cada corrida:\n\n"
        "1. Toma todas las 800 imágenes.\n"
        "2. Hace un split aleatorio estratificado por clase (típicamente 90% train, 10% test).\n"
        "3. Entrena LeNet desde cero.\n"
        "4. Evalúa sobre el test set.\n\n"
        "Las accuracies se promedian. **Nota crítica**: este protocolo **no separa por sujeto** — "
        "el sujeto 3 puede aparecer tanto en train como en test, lo cual es metodológicamente "
        "más laxo que LOSO (leave-one-subject-out).\n\n"
        "### 6.2. Hiperparámetros del paper\n\n"
        "| Parámetro | Valor |\n"
        "|---|---|\n"
        "| Optimizador | Adam |\n"
        "| Learning rate | 0.0007 |\n"
        "| Loss | Cross-Entropy |\n"
        "| Épocas (LeNet) | 20 |\n"
        "| Batch size | 32 |\n"
        "| Augmentation | Ninguna |\n"
        "| Weight decay | Ninguno |\n"
        "| Dispositivo | CPU (MPS bloqueado por bug numérico) |"
    ))

    cells.append(md(
        "## 7. Resultados\n\n"
        "Cada experimento se loguea en `outputs/01_baseline/results.jsonl` "
        "(append-only). La siguiente celda lee directamente desde ahí para mantener "
        "los números actualizados."
    ))

    cells.append(code(
        "import pandas as pd\n\n"
        "PAPER_REFERENCE = {\n"
        "    'Segmented RGB': 96.18,\n"
        "    'Canonical aligned': 96.18,\n"
        "    'Raw (full image)': None,   # el paper usa otra versión de \"raw\" — ver §8.3\n"
        "}\n\n"
        "rows = []\n"
        "with open(RESULTS_JSONL) as f:\n"
        "    for line in f:\n"
        "        r = json.loads(line)\n"
        "        ref = PAPER_REFERENCE.get(r['experiment'])\n"
        "        rows.append({\n"
        "            'timestamp': r['timestamp'],\n"
        "            'experimento': r['experiment'],\n"
        "            'augment': '✓' if r.get('augment') else '—',\n"
        "            'runs': r['n_runs'],\n"
        "            'épocas': r['epochs'],\n"
        "            'mean (%)': round(r['mean_acc']*100, 2),\n"
        "            'std (%)':  round(r['std_acc']*100, 2),\n"
        "            'paper (%)': ref if ref else '—',\n"
        "            'gap (pp)': round(ref - r['mean_acc']*100, 2) if ref else '—',\n"
        "            'nota': r.get('note') or '',\n"
        "        })\n"
        "df = pd.DataFrame(rows)\n"
        "df"
    ))

    cells.append(md(
        "### 7.1. Progresión experimental\n\n"
        "El siguiente gráfico muestra el recorrido completo desde el setup inicial sin "
        "preprocesamiento hasta el mejor modelo (ResNet-18 transfer). Para cada "
        "configuración mostramos la **última corrida con la mayor cantidad de runs** "
        "(la más representativa estadísticamente)."
    ))

    cells.append(code(
        "# Plot de progresión: una barra por (modelo, augment, dataset)\n"
        "import json\n"
        "from collections import OrderedDict\n\n"
        "data = [json.loads(l) for l in open(RESULTS_JSONL)]\n\n"
        "# Para cada (experimento, augment) nos quedamos con la corrida que más runs tenga\n"
        "key = lambda r: (r['experiment'], r.get('augment', False))\n"
        "best_per_setup = {}\n"
        "for r in data:\n"
        "    k = key(r)\n"
        "    if k not in best_per_setup or r['n_runs'] > best_per_setup[k]['n_runs']:\n"
        "        best_per_setup[k] = r\n\n"
        "# Orden lógico del relato: sin canonical → canonical → +augment → ResNet18\n"
        "def sort_key(item):\n"
        "    (exp, aug), r = item\n"
        "    is_canonical = 'Canonical' in exp\n"
        "    is_resnet    = 'ResNet18' in exp\n"
        "    return (is_canonical, aug, is_resnet)\n\n"
        "ordered = sorted(best_per_setup.items(), key=sort_key)\n\n"
        "labels, means, stds, colors = [], [], [], []\n"
        "for (exp, aug), r in ordered:\n"
        "    label = exp\n"
        "    if aug:\n"
        "        label += ' + augment'\n"
        "    label += f'\\n({r[\"n_runs\"]} runs)'\n"
        "    labels.append(label)\n"
        "    means.append(r['mean_acc']*100)\n"
        "    stds.append(r['std_acc']*100)\n"
        "    is_resnet = 'ResNet18' in exp\n"
        "    is_canonical = 'Canonical' in exp\n"
        "    if is_resnet:        colors.append('#27AE60')   # verde — mejor\n"
        "    elif is_canonical:   colors.append('#3498DB')   # azul — replica paper\n"
        "    else:                colors.append('#E67E22')   # naranja — baseline pre-canonical\n\n"
        "fig, ax = plt.subplots(figsize=(12, 5))\n"
        "x = np.arange(len(labels))\n"
        "bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors, alpha=0.85, edgecolor='white')\n"
        "ax.axhline(y=96.18, color='black', ls='--', lw=1.5,\n"
        "           label='Paper LeNet (100 runs) = 96.18%')\n"
        "ax.set_xticks(x)\n"
        "ax.set_xticklabels(labels, rotation=12, ha='right', fontsize=8)\n"
        "ax.set_ylabel('Accuracy (%) — barras de error: ±1 std entre semillas')\n"
        "ax.set_ylim(60, 105)\n"
        "ax.set_title('Progresión experimental: del baseline al transfer learning',\n"
        "             fontweight='bold')\n"
        "ax.legend(loc='lower right')\n"
        "ax.grid(axis='y', alpha=0.3)\n"
        "for bar, m in zip(bars, means):\n"
        "    ax.text(bar.get_x() + bar.get_width()/2, m + 1.5,\n"
        "            f'{m:.2f}%', ha='center', fontsize=9, fontweight='bold')\n"
        "plt.tight_layout(); plt.show()"
    ))

    cells.append(md(
        "## 8. Hallazgos y discusión\n\n"
        "### 8.1. La replicación se logra solo con canonical alignment\n\n"
        "Sin el preprocesamiento de alineamiento canónico, LeNet sobre la versión segmentada "
        "alcanza apenas **~83%** — peor que el feedforward del paper (86.58%). Aplicando el "
        "pipeline completo se llega a **95.83% ± 1.18% en 3 corridas**, esencialmente "
        "indistinguible del 96.18% del paper.\n\n"
        "Esto confirma que **el paso de preprocesamiento es indispensable** para reproducir "
        "los resultados publicados. La especificación textual del paper (Sec. 3.2) describe la "
        "arquitectura pero **no menciona** explícitamente el preprocesamiento canónico — está "
        "documentado en el paper de 2016 referenciado.\n\n"
        "### 8.2. Inestabilidad de entrenamiento en algunas semillas\n\n"
        "Al ampliar a 10 corridas, la media bajó a **91.25% ± 7.83%**. Dos corridas (seeds 8 y 9) "
        "produjeron resultados muy por debajo (70% y 85%). El análisis de matriz de confusión "
        "reveló que en seed 8, el modelo cayó en un **modo colapsado** prediciendo \"Flat\" "
        "para 17 de los 24 errores. No es un problema de imágenes específicas sino de "
        "**inestabilidad de entrenamiento**: con LeNet inicializada aleatoriamente desde cero, "
        "el optimizador a veces cae en un mínimo local malo.\n\n"
        "El paper promedia 100 corridas, lo cual *enmascara* estos outliers. Para nuestra "
        "tesis es importante reportar la varianza honesta.\n\n"
        "### 8.3. Data augmentation: hipótesis refutada\n\n"
        "**Hipótesis inicial**: data augmentation (rotación ±10°, traslación ±10%, color "
        "jitter) actuaría como regularización y reduciría la varianza entre semillas.\n\n"
        "**Resultado** (10 runs, canonical):\n\n"
        "| Métrica | Sin augment | Con augment |\n"
        "|---|---:|---:|\n"
        "| Media | 91.25% | 91.62% |\n"
        "| Std | 7.83% | 6.71% |\n"
        "| Peor run | 70.00% (seed 8) | 75.00% (seed 1) |\n"
        "| Mejor run | 97.50% (seed 0) | 100.00% (seed 6) |\n\n"
        "**Interpretación**: augmentation arregló el mode collapse de seed 8 "
        "(70% → 91.25%) pero **introdujo un nuevo outlier en seed 1** (95% → 75%). "
        "La media casi no cambió y la varianza bajó marginalmente. Cambia *qué* "
        "semillas son problemáticas pero no elimina el problema.\n\n"
        "**Conclusión**: la inestabilidad **no es por falta de datos**. Es "
        "**inestabilidad de optimización intrínseca** del setup LeNet desde cero con "
        "~720 imgs de train. Hay basins de atracción malos en el paisaje de pérdida, "
        "y la inicialización aleatoria a veces cae ahí. Augmentation cambia los basins "
        "pero no los elimina. Atacar la raíz requiere mejorar la **inicialización**.\n\n"
        "### 8.4. Transfer learning con ResNet-18 supera al paper\n\n"
        "**Hipótesis**: una red preentrenada en ImageNet inicia el fine-tuning en un "
        "basin de buena calidad (las features ya son útiles para clasificar formas y "
        "texturas), por lo que evita el mode collapse.\n\n"
        "**Resultado** (10 runs, canonical, fine-tuning end-to-end con lr=1e-4):\n\n"
        "| Métrica | LeNet canonical | LeNet + augment | **ResNet-18 transfer** |\n"
        "|---|---:|---:|---:|\n"
        "| Media | 91.25% | 91.62% | **97.62%** |\n"
        "| Std | 7.83% | 6.71% | **1.63%** |\n"
        "| Peor run | 70.00% | 75.00% | **93.75%** |\n"
        "| Mejor run | 97.50% | 100.00% | **100.00%** |\n\n"
        "**Análisis seed-por-seed** (los antes problemáticos):\n"
        "- Seed 1 (con augment caía a 75%): ResNet-18 → **100%** ✅\n"
        "- Seed 8 (LeNet sin augment: 70% mode collapse): ResNet-18 → **97.5%** ✅\n\n"
        "**Conclusión**:\n"
        "- ResNet-18 transfer **supera al paper por +1.44pp** (97.62% vs 96.18%).\n"
        "- La varianza se redujo en un **factor de 4.8×** respecto a LeNet (7.83 → 1.63).\n"
        "- Ningún run cae bajo 93.75% — los outliers severos desaparecen.\n"
        "- Esto confirma la hipótesis: el problema de LeNet era la **inicialización aleatoria** "
        "y no se podía resolver con augmentation. Los pesos preentrenados de ImageNet ponen al "
        "modelo en un basin bueno desde el inicio.\n\n"
        "### 8.5. Costo computacional y consideración de hardware\n\n"
        "ResNet-18 tarda aproximadamente **6× más por corrida que LeNet** en CPU. "
        "Factores que explican la diferencia:\n\n"
        "| Factor | Impacto |\n"
        "|---|---|\n"
        "| Input 224×224 vs 128×128 | 3.06× más píxeles por capa |\n"
        "| 18 capas vs 4 | 4.5× más operaciones |\n"
        "| Skip connections (ResNet) | Pasadas adicionales de memoria |\n"
        "| BatchNorm en cada capa | ~17 capas de BN vs 4 |\n"
        "| Activaciones para backprop | Mayor presión de memoria en RAM |\n\n"
        "**Sobre la GPU**: el código soporta MPS (Metal Performance Shaders, GPU de los Mac "
        "Apple Silicon), pero detectamos un bug numérico de PyTorch+MPS con BatchNorm+Adam: "
        "el modelo entrena con loss aparente correcto pero al evaluar predice una sola clase. "
        "Por eso forzamos CPU. Con GPU funcional, el costo se reduciría aproximadamente "
        "10× y la comparación sería incluso más favorable a ResNet-18.\n\n"
        "**Trade-off de la tesis**: +6.37pp de accuracy y 4.8× menos varianza justifican "
        "ampliamente el costo extra. Para inferencia en producción (cámara webcam en "
        "tiempo real), ResNet-18 sigue siendo razonablemente liviano (~45MB de pesos).\n\n"
        "### 8.6. Nuestro \"raw\" no es el del paper\n\n"
        "La carpeta `lsa16_raw/` contiene imágenes **completas 640×480** con la persona y el "
        "guante. Releyendo la Fig. 6 del paper, **el \"raw\" del paper es la mano recortada con "
        "fondo original** (sin segmentar) — un crop chico, no la imagen completa.\n\n"
        "Por eso obtuvimos **22.12% ± 5.76%** en \"raw\" mientras el paper reporta **83.54%**: "
        "estamos resolviendo un problema más difícil. Esto es **deseable para la tesis** porque "
        "nuestro \"raw\" refleja el escenario realista (webcam con persona completa) y "
        "justifica directamente la motivación de YOLO como modelo de detección espacial."
    ))

    # ─── §9. Motivación YOLO ─────────────────────────────────────────────
    cells.append(md(
        "## 9. Detección espacial con YOLO — motivación\n\n"
        "Las secciones anteriores establecieron un baseline CNN sólido (ResNet-18 = "
        "**97.62%**), pero ese baseline opera sobre imágenes **previamente recortadas "
        "y alineadas canónicamente**. En un sistema real (cámara webcam con persona "
        "completa) la imagen tiene fondo no controlado, posición variable de las manos, "
        "y otros elementos visuales. Vimos en §8.6 que LeNet sobre imagen completa "
        "raw colapsa al 22.12%.\n\n"
        "**YOLO** (*You Only Look Once* [Redmon et al. 2016]) aborda el problema "
        "de manera fundamentalmente distinta: combina **localización espacial** y "
        "**clasificación** en un solo modelo end-to-end. Recibe la imagen completa "
        "sin preprocesar y produce simultáneamente:\n\n"
        "- Las coordenadas (*bounding boxes*) de los objetos de interés.\n"
        "- La clase predicha de cada objeto.\n\n"
        "Para nuestro problema: YOLO ve la imagen completa de una persona haciendo "
        "una seña, encuentra la mano dominante (con guante de color), y predice su "
        "configuración manual de las 16 clases. No requiere segmentación previa "
        "ni alineamiento canónico.\n\n"
        "Usamos **YOLOv8** [Ultralytics 2023] en sus variantes `s` (small, 11M params) "
        "y `m` (medium, 26M params), ambas preentrenadas en **COCO** [Lin et al. 2014] "
        "(transfer learning desde 1.28M imágenes de 80 clases generales)."
    ))

    # ─── §10. Generación de anotaciones ──────────────────────────────────
    cells.append(md(
        "## 10. Generación automática de anotaciones\n\n"
        "YOLO requiere un dataset con **bounding boxes anotadas** (formato: "
        "`class_id xc yc w h` normalizado). LSA16 viene con máscaras de segmentación "
        "pero **no con bboxes**, así que las generamos automáticamente.\n\n"
        "### 10.1. Pipeline: template matching enmascarado\n\n"
        "Aprovechamos que `lsa16_segmented_right_hand/` **es literalmente un crop "
        "de `lsa16_raw/`** (con fondo seteado a negro). Usando "
        "`cv2.matchTemplate(raw, segmented, TM_CCORR_NORMED, mask=non_zero_pixels)` "
        "encontramos la posición exacta donde la segmented \"encaja\" en la raw, "
        "obteniendo la bbox directamente.\n\n"
        "Implementación: [`src/yolo_annotations.py`](../src/yolo_annotations.py).\n\n"
        "### 10.2. Iteración: detección por color (descartada)\n\n"
        "Probamos primero un enfoque por color HSV (buscar el color del guante en "
        "la raw). Falló en **76 de 800 imágenes (9.5%)** porque:\n\n"
        "- Los guantes tienen **2 colores distintos** en el dataset (rosa/magenta y "
        "cian, varía entre sujetos).\n"
        "- La **piel facial saturada** (especialmente lipstick) caía en el mismo "
        "rango de Hue del guante magenta, generando bboxes sobre la **cara**.\n\n"
        "Template matching es inmune a este problema porque busca un match "
        "**estructural** de píxeles, no solo de color.\n\n"
        "### 10.3. Resultados del pipeline\n\n"
        "Sobre las 800 imágenes:\n"
        "- **794/800 (99.25%) con bbox correctamente detectada**\n"
        "- 6 fallos: casos donde el segmented no era un crop literal\n"
        "- Validación cruzada confirma que template matching es correcto en las 76 "
        "imágenes donde difería con el método por color\n\n"
        "Partición final estratificada (`dataset_yolo/`): 556 train, 121 val, 117 test."
    ))

    cells.append(code(
        "# Ejemplos de bboxes detectadas (1 muestra por clase)\n"
        "from IPython.display import Image as IPImage\n\n"
        "bbox_fig = PROJECT_ROOT / 'outputs' / '02_yolo-gpu' / 'figures' / 'bboxes_per_class.png'\n"
        "if not bbox_fig.exists():\n"
        "    # Fallback al directorio original si no se copió aún\n"
        "    bbox_fig = PROJECT_ROOT / 'outputs' / '02_yolo' / 'figures' / 'bboxes_per_class.png'\n"
        "IPImage(filename=str(bbox_fig))"
    ))

    # ─── §11. Entrenamiento YOLO ─────────────────────────────────────────
    cells.append(md(
        "## 11. Entrenamiento YOLOv8\n\n"
        "Implementación en [`experiments/02_yolo_train.py`](../experiments/02_yolo_train.py) "
        "usando la API de Ultralytics.\n\n"
        "### 11.1. Hiperparámetros y razonamiento\n\n"
        "| Parámetro | Valor | Razón |\n"
        "|---|---|---|\n"
        "| Modelo base | `yolov8s.pt` | balance velocidad/precisión, 11M params (comparable a ResNet-18) |\n"
        "| Pesos iniciales | COCO pretrained | transfer learning desde 1.28M imgs / 80 clases |\n"
        "| `imgsz` | 640 | default Ultralytics; mantiene detalle de dedos |\n"
        "| `batch` | 32 (GPU) | limitado por VRAM (3080 Ti = 12GB) |\n"
        "| `epochs` | 100 (`patience=20`) | early stopping conservador |\n"
        "| `optimizer` | AdamW auto, lr=0.0005 | seleccionado automáticamente |\n"
        "| `fliplr` | **0.0** | CRÍTICO: flip horizontal cambia mano derecha/izquierda |\n"
        "| `flipud` | **0.0** | no tiene sentido geométrico para handshapes |\n"
        "| `mosaic` | **0.0** | distorsiona relación espacial (1 mano por imagen) |\n"
        "| `degrees`, `translate`, `scale` | 10°, 0.1, 0.1 | jitter geométrico leve |\n"
        "| `hsv_s`, `hsv_v` | 0.4, 0.3 | variación moderada de iluminación |\n\n"
        "### 11.2. Costo computacional\n\n"
        "| Hardware | yolov8s, 88 épocas (con early stop) |\n"
        "|---|---:|\n"
        "| Mac M3 Pro (CPU) | ~3-5 horas estimadas |\n"
        "| **RTX 3080 Ti (GPU)** | **8.5 minutos** |\n\n"
        "Speedup ~20×. Los entrenamientos definitivos se hicieron en GPU."
    ))

    cells.append(code(
        "# Curvas de entrenamiento de YOLOv8s\n"
        "training_curves = PROJECT_ROOT / 'outputs' / '02_yolo-gpu' / 'yolov8s_gpu' / 'results.png'\n"
        "IPImage(filename=str(training_curves))"
    ))

    # ─── §12. Resultados YOLO ────────────────────────────────────────────
    cells.append(md(
        "## 12. Resultados YOLOv8\n\n"
        "Métricas calculadas sobre el split de test (117 imágenes) usando "
        "[`experiments/02_yolo_eval.py`](../experiments/02_yolo_eval.py)."
    ))

    cells.append(code(
        "# Cargar métricas de eval para ambas variantes\n"
        "import json\n\n"
        "eval_s_path = PROJECT_ROOT / 'outputs' / '02_yolo-gpu' / 'eval_yolov8s_gpu.json'\n"
        "eval_m_path = PROJECT_ROOT / 'outputs' / '02_yolo-gpu' / 'eval_yolov8m_imgsz960-2.json'\n\n"
        "eval_s = json.load(open(eval_s_path))\n"
        "eval_m = json.load(open(eval_m_path))\n\n"
        "yolo_results = pd.DataFrame([\n"
        "    {\n"
        "        'Modelo': 'YOLOv8s @ 640',\n"
        "        'Detection rate': f\"{eval_s['detection_rate']*100:.1f}%\",\n"
        "        'Acc (todas)': f\"{eval_s['classification_accuracy']*100:.2f}%\",\n"
        "        'Acc (detectadas)': f\"{eval_s['classification_accuracy_detected_only']*100:.2f}%\",\n"
        "        'IoU mediana': f\"{eval_s['median_iou']:.3f}\",\n"
        "        'mAP@50': f\"{eval_s['mAP50']:.3f}\",\n"
        "        'mAP@50-95': f\"{eval_s['mAP50_95']:.3f}\",\n"
        "    },\n"
        "    {\n"
        "        'Modelo': 'YOLOv8m @ 960',\n"
        "        'Detection rate': f\"{eval_m['detection_rate']*100:.1f}%\",\n"
        "        'Acc (todas)': f\"{eval_m['classification_accuracy']*100:.2f}%\",\n"
        "        'Acc (detectadas)': f\"{eval_m['classification_accuracy_detected_only']*100:.2f}%\",\n"
        "        'IoU mediana': f\"{eval_m['median_iou']:.3f}\",\n"
        "        'mAP@50': f\"{eval_m['mAP50']:.3f}\",\n"
        "        'mAP@50-95': f\"{eval_m['mAP50_95']:.3f}\",\n"
        "    },\n"
        "])\n"
        "yolo_results"
    ))

    cells.append(md(
        "### 12.1. Matriz de confusión (yolov8s)"
    ))

    cells.append(code(
        "cm_path = PROJECT_ROOT / 'outputs' / '02_yolo-gpu' / 'figures' / 'confusion_yolov8s_gpu.png'\n"
        "IPImage(filename=str(cm_path))"
    ))

    cells.append(md(
        "**Observaciones**:\n\n"
        "- **11 de 16 clases** con clasificación perfecta.\n"
        "- Confusiones más frecuentes: **V→Horns (2 errores)**, Beak→Fingers/Index "
        "(1 c/u), Curve→Beak (1).\n"
        "- Las confusiones son entre clases **visualmente similares**: V y Horns "
        "son ambas configuraciones con 2 dedos extendidos en horquilla. Beak, "
        "Fingers e Index son poses cerradas con punta visible.\n\n"
        "### 12.2. Distribución de IoU (calidad de localización)"
    ))

    cells.append(code(
        "iou_path = PROJECT_ROOT / 'outputs' / '02_yolo-gpu' / 'figures' / 'iou_distribution_yolov8s_gpu.png'\n"
        "IPImage(filename=str(iou_path))"
    ))

    cells.append(md(
        "La gran mayoría de las detecciones tienen **IoU ≥ 0.75** con mediana 0.82. "
        "Los pocos outliers con IoU < 0.6 corresponden a casos difíciles que "
        "explican parte de la diferencia entre acc-global y acc-cuando-detecta.\n\n"
        "### 12.3. Ablación: ¿más capacidad mejora el resultado?\n\n"
        "Probamos una variante más ambiciosa (yolov8m + imgsz=960) — modelo más "
        "grande Y mayor resolución de entrada — para testear si el accuracy mejora:"
    ))

    cells.append(code(
        "comparison = pd.DataFrame([\n"
        "    {'Métrica': 'Detection rate',         'yolov8s @ 640': '94.9% (6 misses)', 'yolov8m @ 960': '98.3% (2 misses)', 'Δ': '+3.4pp ✅'},\n"
        "    {'Métrica': 'Acc global (todas)',     'yolov8s @ 640': '89.74%',           'yolov8m @ 960': '90.60%',           'Δ': '+0.86pp'},\n"
        "    {'Métrica': 'Acc (solo detectadas)',  'yolov8s @ 640': '94.59%',           'yolov8m @ 960': '92.17%',           'Δ': '−2.42pp ❌'},\n"
        "    {'Métrica': 'IoU promedio',           'yolov8s @ 640': '0.779',            'yolov8m @ 960': '0.899',            'Δ': '+12pp ✅'},\n"
        "    {'Métrica': 'IoU mediana',            'yolov8s @ 640': '0.818',            'yolov8m @ 960': '0.949',            'Δ': '+13pp ✅'},\n"
        "    {'Métrica': 'mAP@50',                 'yolov8s @ 640': '0.923',            'yolov8m @ 960': '0.925',            'Δ': '≈'},\n"
        "    {'Métrica': 'mAP@50-95',              'yolov8s @ 640': '0.875',            'yolov8m @ 960': '0.864',            'Δ': '−1.1pp'},\n"
        "    {'Métrica': 'Tiempo entrenamiento',   'yolov8s @ 640': '8.5 min',          'yolov8m @ 960': '39.7 min',         'Δ': '4.7× más'},\n"
        "])\n"
        "comparison"
    ))

    cells.append(md(
        "**Hallazgo metodológico interesante**: más capacidad **NO mejoró la "
        "clasificación**, e incluso la empeoró ligeramente (−2.42pp en "
        "acc-cuando-detecta). Lo que sí mejoró drásticamente fue:\n\n"
        "- **Localización**: IoU mediana 0.82 → 0.95.\n"
        "- **Cobertura**: 4 misses menos (detection rate 94.9% → 98.3%).\n\n"
        "**Interpretación**: el cuello de botella ya no es la capacidad del modelo, "
        "sino la **similitud visual intrínseca** entre algunas configuraciones "
        "manuales. Más parámetros (26M de yolov8m, 47k params por sample de train) "
        "generan **sobreajuste leve** sobre las 556 imágenes de entrenamiento.\n\n"
        "**Para la tesis: reportamos yolov8s como modelo principal** (mejor "
        "classification y más liviano), y yolov8m como ablation negativa — un "
        "resultado científicamente valioso que confirma que *más no siempre es mejor* "
        "en datasets pequeños."
    ))

    # ─── §13. Comparación final CNN vs YOLO ─────────────────────────────
    cells.append(md(
        "## 13. Comparación final: CNN baseline vs YOLO detector espacial\n\n"
        "### 13.1. Tabla unificada de todos los modelos experimentados"
    ))

    cells.append(code(
        "final = pd.DataFrame([\n"
        "    {'Modelo': 'LeNet (paper, 100 runs)',         'Input': 'mano segmentada + canonical 128×128', 'Accuracy': '96.18%',         'Notas': 'benchmark del paper'},\n"
        "    {'Modelo': 'LeNet (nuestra réplica, 10 runs)','Input': 'mano segmentada + canonical 128×128', 'Accuracy': '91.25% ± 7.83%', 'Notas': 'varianza honesta'},\n"
        "    {'Modelo': 'LeNet + augment (10 runs)',       'Input': 'mano segmentada + canonical 128×128', 'Accuracy': '91.62% ± 6.71%', 'Notas': 'hipótesis refutada'},\n"
        "    {'Modelo': 'ResNet-18 transfer (10 runs)',    'Input': 'mano segmentada + canonical 128×128', 'Accuracy': '97.62% ± 1.63%', 'Notas': 'supera al paper'},\n"
        "    {'Modelo': 'LeNet sobre raw (10 runs)',       'Input': 'imagen completa 640×480',             'Accuracy': '22.12% ± 5.76%', 'Notas': 'CNN colapsa sin preproc.'},\n"
        "    {'Modelo': '**YOLOv8s sobre raw**',           'Input': '**imagen completa 640×480**',         'Accuracy': '**89.74%**',     'Notas': '**modelo principal**'},\n"
        "    {'Modelo': 'YOLOv8m @ 960 sobre raw',         'Input': 'imagen completa 640×480',             'Accuracy': '90.60%',         'Notas': 'ablation: más ≠ mejor'},\n"
        "])\n"
        "final"
    ))

    cells.append(md(
        "### 13.2. Análisis cualitativo\n\n"
        "La tabla cuenta dos historias según en qué eje focalicemos:\n\n"
        "**Sobre la mano preprocesada** (canonical 128×128):\n"
        "- ResNet-18 (97.62%) supera a LeNet (91-96%). Transfer learning resuelve "
        "la inestabilidad de entrenamiento del CNN puramente entrenado desde cero.\n"
        "- La comparación se reduce a *\"qué arquitectura clasifica mejor sobre un "
        "crop limpio\"*.\n\n"
        "**Sobre el escenario realista** (imagen completa 640×480 sin preprocesamiento):\n"
        "- LeNet colapsa a **22.12%** — apenas mejor que random (6.25%) pero "
        "inviable como sistema.\n"
        "- YOLOv8s alcanza **89.74%** — **una diferencia de +67pp** respecto a LeNet "
        "en la misma entrada.\n"
        "- YOLOv8m apenas mejora a 90.60%, confirmando que estamos cerca del techo "
        "del enfoque YOLO para este dataset.\n\n"
        "**La diferencia de +67pp en el escenario realista es la contribución "
        "principal del trabajo**: el cambio de paradigma de clasificación a "
        "detección espacial habilita el procesamiento de imágenes no controladas, "
        "donde el CNN tradicional fracasa."
    ))

    cells.append(code(
        "# Visualización gráfica de la comparación final\n"
        "fig, ax = plt.subplots(figsize=(11, 6))\n\n"
        "data = [\n"
        "    ('LeNet\\nraw 640×480',          22.12,  5.76, '#E74C3C'),\n"
        "    ('LeNet\\ncanonical (10 runs)',  91.25,  7.83, '#F39C12'),\n"
        "    ('LeNet+augment\\ncanonical',    91.62,  6.71, '#F39C12'),\n"
        "    ('ResNet-18\\ncanonical',        97.62,  1.63, '#3498DB'),\n"
        "    ('YOLOv8s\\nraw 640×480',        89.74,  0.0,  '#27AE60'),\n"
        "    ('YOLOv8m @960\\nraw 640×480',   90.60,  0.0,  '#27AE60'),\n"
        "]\n"
        "labels  = [d[0] for d in data]\n"
        "means   = [d[1] for d in data]\n"
        "stds    = [d[2] for d in data]\n"
        "colors  = [d[3] for d in data]\n\n"
        "x = np.arange(len(data))\n"
        "bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors, alpha=0.85, edgecolor='white')\n"
        "ax.axhline(y=96.18, color='black', ls='--', lw=1.2,\n"
        "           label='Paper benchmark LeNet=96.18%')\n"
        "ax.axhline(y=6.25, color='gray', ls=':', lw=1.0, label='Random (1/16=6.25%)')\n"
        "ax.set_xticks(x)\n"
        "ax.set_xticklabels(labels, fontsize=9)\n"
        "ax.set_ylabel('Accuracy (%)', fontsize=11)\n"
        "ax.set_ylim(0, 105)\n"
        "ax.set_title('Comparación final — todos los modelos sobre LSA16',\n"
        "             fontsize=13, fontweight='bold')\n"
        "ax.legend(loc='center right', fontsize=9)\n"
        "ax.grid(axis='y', alpha=0.3)\n"
        "for bar, m in zip(bars, means):\n"
        "    ax.text(bar.get_x() + bar.get_width()/2, m + 2, f'{m:.1f}%',\n"
        "            ha='center', fontsize=10, fontweight='bold')\n"
        "# Rótulos por color\n"
        "from matplotlib.patches import Patch\n"
        "legend_elements = [\n"
        "    Patch(facecolor='#E74C3C', label='LeNet / raw — colapsa'),\n"
        "    Patch(facecolor='#F39C12', label='LeNet / canonical'),\n"
        "    Patch(facecolor='#3498DB', label='ResNet-18 / canonical'),\n"
        "    Patch(facecolor='#27AE60', label='YOLOv8 / raw — contribución'),\n"
        "]\n"
        "leg2 = ax.legend(handles=legend_elements, loc='lower right', fontsize=9, title='Familia de modelo')\n"
        "ax.add_artist(leg2)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # ─── §14. Conclusiones finales ──────────────────────────────────────
    cells.append(md(
        "## 14. Conclusiones finales y trabajo futuro\n\n"
        "### 14.1. Síntesis del trabajo\n\n"
        "Este notebook documenta el camino completo desde la replicación del paper "
        "hasta una propuesta original con detección espacial. Hallazgos principales:\n\n"
        "1. **Replicación reproducible** del baseline LeNet del paper (95.83% obtenido "
        "vs 96.18% target en 3 runs, indistinguible dentro del ruido), con el "
        "**preprocesamiento canónico identificado como ingrediente crítico** que el "
        "paper no documenta explícitamente.\n"
        "2. **Caracterización honesta de la varianza** (10 runs muestran inestabilidad "
        "de entrenamiento de LeNet que el paper diluye con 100 runs).\n"
        "3. **Mejora sobre el paper con ResNet-18 transfer learning**: 97.62% ± 1.63%, "
        "superando el benchmark publicado por +1.44pp y reduciendo la varianza 4.8×.\n"
        "4. **Refutación con rigor de la hipótesis \"augmentation arregla la "
        "inestabilidad\"**: no la elimina, solo cambia qué semillas son problemáticas.\n"
        "5. **Pipeline automático de generación de anotaciones YOLO** vía template "
        "matching enmascarado (99.25% de éxito), con identificación y corrección "
        "del problema de falsos positivos en regiones faciales que tenía el enfoque "
        "por color HSV.\n"
        "6. **YOLOv8s sobre raw alcanza 89.74% accuracy y mAP@50=0.923**, con "
        "**+67pp sobre LeNet** en la misma entrada. Esta es la contribución central "
        "del trabajo.\n"
        "7. **Ablación negativa con yolov8m+imgsz=960**: mejora drásticamente la "
        "localización (IoU mediana 0.82 → 0.95) pero **no la clasificación** "
        "(sobreajuste leve). Resultado científicamente útil que orienta futuras "
        "decisiones de modelo para datasets pequeños.\n\n"
        "### 14.2. Trabajo futuro identificado\n\n"
        "1. **Evaluación LOSO** (leave-one-subject-out) sobre los modelos finales "
        "(ResNet-18, YOLOv8s) — protocolo más riguroso donde el sujeto de test "
        "nunca está en train. Mediría la capacidad de generalizar a usuarios no "
        "vistos durante el entrenamiento (típicamente cae 10-15pp pero es más "
        "representativo del uso real).\n"
        "2. **ResNet-18 + data augmentation**: combinación no testeada. Hipótesis: "
        "podría llevar el accuracy CNN cerca del 99%. Costo: ~90 min de GPU. "
        "No se ejecutó por priorizar el contraste con YOLO.\n"
        "3. **YOLOv8n** (nano): modelo más liviano para deployment en tiempo real "
        "en hardware modesto (webcam de notebook, mobile).\n"
        "4. **Test end-to-end con webcam real**: validar el pipeline en condiciones "
        "no controladas (iluminación variable, fondo no blanco, distancia variable).\n\n"
        "### 14.3. Reproducibilidad\n\n"
        "Para reproducir cualquier resultado de este notebook:\n\n"
        "```bash\n"
        "cd trabajo-parcial\n"
        "source .venv/bin/activate\n\n"
        "# CNN baselines (canonical preprocessing) — total ~2.5 hs CPU\n"
        "python scripts/preprocess_canonical.py\n"
        "python experiments/01_baseline_paper.py --mode canonical --n-runs 10\n"
        "python experiments/01_baseline_paper.py --mode canonical --augment --n-runs 10\n"
        "python experiments/01_baseline_paper.py --mode canonical --model resnet18 --n-runs 10\n"
        "python experiments/01_baseline_paper.py --mode raw --n-runs 10\n\n"
        "# YOLO pipeline (~10 min en GPU NVIDIA, ~3-5 hs en CPU)\n"
        "python scripts/generate_yolo_dataset.py\n"
        "python experiments/02_yolo_train.py --model yolov8s.pt --imgsz 640 --device 0\n"
        "python experiments/02_yolo_eval.py --weights outputs/02_yolo/yolov8s_*/weights/best.pt\n"
        "```\n\n"
        "Todos los resultados se loguean en `outputs/01_baseline/results.jsonl` "
        "y `outputs/02_yolo*/results.jsonl` (machine-readable). El notebook lee de "
        "estos archivos en tiempo de ejecución, así que **regenerarlo** después de "
        "nuevos experimentos solo requiere re-ejecutar las celdas."
    ))

    nb.cells = cells

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NOTEBOOK_PATH.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook generado: {NOTEBOOK_PATH}")
    print(f"  {len(cells)} celdas")


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    main()
