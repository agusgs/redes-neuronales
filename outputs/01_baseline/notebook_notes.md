# Notas de hallazgos para actualizar el notebook

Este archivo registra hallazgos experimentales a medida que avanzamos.
Cuando actualicemos `scripts/build_notebook_01.py` (o creemos un notebook
02), tomar de acá los puntos para narrar.

---

## Sesión 2026-05-22

### Hallazgo 1: el preprocesamiento canónico es el ingrediente clave
- Sin canonical: 83.33% ± 2.12% (3 runs)
- Con canonical: 95.83% ± 1.18% (3 runs) → matchea el 96.18% del paper
- Cierre de gap: 12.5pp gracias al alineamiento PCA + flip 180°.
- Lección: el paper menciona la arquitectura pero el preprocesamiento está
  en el paper 2016 referenciado, no en el 2017. Sin él, LeNet va peor que
  el feedforward del paper (86.58%).

### Hallazgo 2: inestabilidad de entrenamiento aparece con más runs
- Con 10 runs (canonical, sin augment): 91.25% ± **7.83%**
- Seed 8 cayó a 70% (mode collapse — modelo predice "Flat" para todo).
- Seed 9 cayó a 85%.
- Los otros 8 seeds están en rango 91–97%.
- El paper promedia 100 runs lo cual diluye estos outliers.
- Conclusión: con LeNet + Adam + lr=0.0007 + 20 épocas + ~720 imgs, hay
  basins de optimización malos. La inicialización aleatoria a veces cae ahí.

### Hallazgo 3: data augmentation NO arregla la inestabilidad
- Con augment (10 runs, mismos seeds): 91.62% ± 6.71%
- Media casi igual (+0.37pp), varianza bajó marginalmente (-1.12pp).
- ✅ Arregló mode collapse seed 8 (70% → 91.25%).
- ❌ Introdujo nuevo outlier seed 1 (95% → 75%).
- Cambia QUÉ seeds son problemáticos, no si los hay.
- **Implicación**: el problema es inestabilidad de optimización intrínseca,
  no falta de datos. Hay que atacar la inicialización, no aumentar el
  training set virtual.

### Hallazgo 4: Transfer learning con ResNet18 resuelve la inestabilidad
- Canonical + ResNet18 transfer (10 runs): **97.62% ± 1.63%**
- vs LeNet canonical (10 runs): 91.25% ± 7.83% → mejora de +6.37pp en media, std 4.8× menor.
- vs paper LeNet (100 runs): 96.18% → **superamos al paper por +1.44pp**.
- Seeds problemáticos arreglados:
  - Seed 1 (LeNet+augment lo rompía a 75%): ResNet18 → **100%**.
  - Seed 8 (LeNet sin augment: 70% mode collapse): ResNet18 → **97.5%**.
- Peor run: 93.75% (vs 70% de LeNet). No hay outliers.
- **Conclusión**: la inestabilidad de LeNet era inicialización. ImageNet pretrained
  pone al modelo en un basin bueno desde el inicio.

**Costo computacional**: ResNet18 tarda ~6× más por corrida que LeNet en CPU.
Razones: input 224×224 (3× más píxeles), 18 capas vs 4, skip connections,
~17 capas de BatchNorm. En GPU sería ~10× más rápido pero MPS sigue bloqueado.
Trade-off de la tesis: +6pp accuracy y 5× menos varianza justifican el costo.

### Hallazgo 5: nuestro "raw" no es el del paper
- `lsa16_raw/` = imágenes completas 640×480 con persona entera.
- Paper's "raw" = mano recortada SIN segmentar fondo (Fig. 6 del paper).
- Por eso obtuvimos 22.12% vs su 83.54% — son problemas distintos.
- **Esto es bueno para la tesis**: nuestro raw representa el escenario real
  (webcam con persona completa) y justifica directamente la motivación de
  YOLO como modelo de detección espacial.

---

## Trabajo futuro identificado (para mencionar en tesis / nb futuros)

- **ResNet-18 + data augmentation**: combinación no testeada. Hipótesis: podría
  llevar el accuracy cerca del 99% y reducir aún más la varianza, especialmente
  el peor run (seed 7 con 93.75%). Costo: ~90 min de CPU. No se ejecutó por
  priorizar el contraste contra YOLO (contribución original). Es trabajo futuro
  natural si se quiere maximizar el baseline.

- **LOSO sobre todos los modelos finales**: protocolo de evaluación más riguroso
  donde el sujeto de test nunca está en train. Ejecutar al final del trabajo
  sobre todos los modelos (LeNet, LeNet+aug, ResNet18, YOLO) para producir
  la tabla comparativa final de la tesis. Costo: varias horas. Resultado
  esperado: caída de 10-15pp en accuracy respecto al subsampling, pero números
  más representativos del uso real (cámara con persona desconocida).

- **Matriz de confusión del seed 8 (sin augment)**: agregar visualización
  específica que muestre el mode collapse a "Flat". Es evidencia visual fuerte
  del fenómeno descrito en §8.2.

## Diseño del experimento YOLO (decisiones de modelado)

**Decisión: Opción A — un solo bounding box por imagen, sobre la mano derecha.**

### Contexto del dataset

Las imágenes raw (640×480) contienen una persona completa con dos guantes
fluorescentes: **rojo (mano derecha)** y **magenta (mano izquierda)**. El fondo
es blanco y la ropa negra. Pero **las 16 clases de LSA16 corresponden únicamente
a la configuración de la mano derecha** (lo confirma el nombre de la carpeta
de segmentación: `lsa16_segmented_right_hand`).

En lengua de señas existe una mano *dominante* (que define la "configuración" o
handshape) y otra *no-dominante* (apoyo, simetría o posición neutral). El dataset
solo anotó la mano dominante.

### Alternativas consideradas

| Opción | Qué detectar | Por qué descartada/elegida |
|---|---|---|
| **A. 1 bbox: mano derecha** | Solo el guante rojo + su handshape | **Elegida**. Match exacto con la tarea del CNN, comparación limpia, propuesta lo sugiere ("la mano" en singular), tenemos ground truth de clase. |
| B. 2 bboxes: ambas manos | Roja y magenta separadas | Descartada. No tenemos ground truth de clase para la izquierda. Tendríamos que inventar la etiqueta o ignorarla, complicando el modelo. |
| C. 1 bbox cubriendo ambas | Región general de señas | Descartada. Pierde precisión, no aprovecha la información de cuál mano es la relevante. |

### Pipeline de generación automática de anotaciones

**Hallazgo durante la implementación**: probamos dos enfoques distintos.

**Enfoque 1 (descartado): thresholding HSV por color del guante.**
Idea: encontrar en la raw los píxeles del color del guante derecho (color que
varía entre imágenes — algunos sujetos llevan rosa/magenta, otros cian).
Tomamos el color dominante del segmented como referencia per-imagen, después
threshold en HSV con tolerancia ± 15°.

Problema descubierto: en **76 de 800 imágenes (9.5%) el bbox cae en la cara**
porque los labios, mejillas o regiones de piel saturada caen dentro del rango
de Hue del guante (especialmente magenta vs piel rojiza). El método de color
no distingue mano vs cara cuando comparten Hue dominante.

**Enfoque 2 (adoptado): template matching enmascarado.**
La imagen segmentada **es literalmente la mano derecha del raw recortada**
con el fondo seteado a negro. Usamos `cv2.matchTemplate(raw, segmented,
TM_CCORR_NORMED, mask=non_zero_pixels)` para encontrar la posición *exacta*
donde la segmented "encaja" en el raw. Es inmune a confusiones con cara o
piel porque busca un match estructural de píxeles, no solo de color.

Resultado: 794/800 imágenes con bbox correctamente detectada (99.25%). Las 6
fallas son casos donde la segmented parece haber sido procesada levemente y
no es un crop literal del raw.

**Validación**: visualizamos 16 muestras de las 76 discrepancias entre los
dos métodos (`outputs/02_yolo/figures/discrepancies.png`) y confirmamos
visualmente que template matching es correcto y color es incorrecto en todos
esos casos.

Lección metodológica: cuando un método tiene ground truth indirecto disponible
(como la segmented siendo un crop del raw), template matching es preferible
a heurísticas de color/threshold que pueden generar false positives.

### Split del dataset YOLO

70% train / 15% val / 15% test, estratificado por clase. Una sola partición
fija (no múltiples runs como el CNN baseline) porque YOLO se entrena una sola
vez y se evalúa con métricas de detección (mAP, IoU) sobre val/test.

---

## Discusión sobre YOLO vs CNN para el documento

### Por qué ResNet18 (97.62%) > YOLO (89.74%) es esperable y NO es una limitación del trabajo

Las dos métricas no son directamente comparables porque resuelven problemas distintos:

- **ResNet18** recibe la mano YA recortada Y alineada canónicamente (input 128×128).
  Es un problema de pura **clasificación** sobre una imagen preprocesada de manera
  manual y elaborada.
- **YOLO** recibe la imagen completa 640×480 raw, con la persona entera,
  fondo no preprocesado, y debe **localizar + clasificar** end-to-end.

El argumento central de la tesis NO es "YOLO supera a CNN en accuracy", sino
**"YOLO funciona en el escenario realista donde CNN colapsa"**:

| Escenario | LeNet/CNN | YOLO |
|---|---:|---:|
| Mano segmentada + alineada (preprocesamiento manual) | 97.62% | n/a |
| **Imagen raw 640×480 (escenario realista)** | **22.12%** ❌ | **89.74%** ✅ |

YOLO le saca **+67pp** a LeNet en el escenario realista. Esa es la diferencia
operativa: ResNet18 requiere pipeline de visión clásica (segmentación + PCA +
flip + crop) antes de funcionar; YOLO no necesita nada.

### Análisis de qué limita YOLO actualmente (1ra corrida yolov8s + imgsz=640)

Mirando la evaluación:
- **6 imágenes (5.1%) sin detección alguna** → estos cuentan como error.
- **Cuando detecta, clasifica al 94.59%** — muy cerca de LeNet canonical.

Conclusión: el principal lever para mejorar YOLO no es la cabeza de
clasificación, sino la **capacidad de detección** sobre los casos difíciles.
Esto sugiere que pueden ayudar:
- Mayor resolución de entrada (`imgsz=960`) para captar manos pequeñas o
  lejanas del frame.
- Modelo más grande (`yolov8m`, `yolov8l`) con más capacidad de discriminar
  poses ambiguas.

### Confusiones inter-clase más frecuentes (yolov8s)

- **V → Horns** (2 errores): ambas señas usan 2 dedos extendidos en V/horquilla.
  Diferenciación dependía del ángulo entre dedos.
- **Beak → Fingers**, **Beak → Index** (1 c/u): poses cerradas con punta visible.
- **Curve → Beak** (1): poses semi-cerradas.
- **Fingers → Double** (1).

Esto refleja la similitud visual intrínseca entre algunas configuraciones manuales
del LSA — un humano no entrenado también las confundiría.

### Hiperparámetros de YOLO: trade-offs explorados

Documentar para la tesis el razonamiento detrás de los hiperparámetros:

| Parámetro | Valor elegido | Razón |
|---|---|---|
| `imgsz` | 640 (CPU) / 960 (GPU ambicioso) | Más resolución = mejor detalle de dedos, pero ~3× más cómputo |
| `batch` | 16 (CPU) / 32 (GPU 12GB VRAM) | Más batch = gradiente más estable, pero limitado por VRAM |
| `fliplr` | 0.0 | Crítico: flip horizontal cambia mano derecha/izquierda → distorsiona el ground truth |
| `flipud` | 0.0 | No tiene sentido geométrico para handshapes |
| `mosaic` | 0.0 | Mosaic junta 4 imgs en 1; con UNA mano por imagen distorsiona la relación espacial |
| `degrees` | 10.0 | Pequeño jitter de rotación; las manos del dataset no están perfectamente verticales |
| `translate` | 0.1 | Jitter espacial leve para robustez a posición |
| `hsv_s`, `hsv_v` | 0.4, 0.3 | Variación moderada de saturación/brillo para distintas condiciones de iluminación |
| `patience` | 20 | Early stopping conservador; suficiente para detectar plateau |

### Sobre el costo computacional (relevante para escalabilidad)

| Hardware | Setup | Tiempo de entrenamiento |
|---|---|---|
| Mac M3 Pro (CPU) | yolov8s, imgsz=480, batch=16, 50 épocas | ~2-3 horas estimadas (cancelado) |
| RTX 3080 Ti (GPU) | yolov8s, imgsz=640, batch=32, 88 épocas (con early stop) | **8.5 min** |

**El speedup de GPU es ~20×** para este modelo en este dataset. Esto es relevante
para discutir en la tesis: en producción (cámara real time) se necesita GPU, pero
el entrenamiento es accesible incluso en hardware modesto.

### Sobre tunear más vs aceptar el resultado

Punto importante para la honestidad metodológica del documento:

- **Tunear más YOLO probablemente lleve a ~92-95%** (con yolov8m + imgsz=960
  + augmentation refinada). No al 97.62% de ResNet18, porque las tareas
  son fundamentalmente distintas.
- **No es necesario** que YOLO supere a ResNet18 para que la tesis sea sólida.
  La comparación útil es CNN-sobre-raw vs YOLO-sobre-raw (22% vs 89-95%).
- Decidimos correr **una** corrida adicional ambiciosa para caracterizar el
  techo realista de YOLO. Si llega a 93-95%, lo reportamos como mejora.
  Si no aporta, también es información (sabemos donde está el límite).

---

## Resultados YOLO — yolov8m + imgsz=960 (segunda corrida, 2026-05-22)

**Hipótesis a testear**: ¿más capacidad de modelo (yolov8m, 26M params) + mayor
resolución (imgsz=960) reduce los misses y mejora la accuracy global?

**Setup**: yolov8m.pt, imgsz=960, batch=16, 100 épocas completadas (sin early
stop), RTX 3080 Ti, 39.7 min.

### Comparación lado a lado

| Métrica | yolov8s @ 640 | yolov8m @ 960 | Δ |
|---|---:|---:|---:|
| **Accuracy global** (todas las test) | 89.74% | **90.60%** | +0.86pp ✅ |
| Accuracy (solo cuando detecta) | 94.59% | 92.17% | -2.42pp ❌ |
| **Detection rate** | 94.9% (6 misses) | **98.3%** (2 misses) | **+3.4pp** ✅✅ |
| **IoU promedio** | 0.779 | **0.899** | +12pp ✅✅✅ |
| **IoU mediana** | 0.818 | **0.949** | +13pp ✅✅✅ |
| mAP@50 | 0.923 | 0.925 | ≈ |
| mAP@50-95 | 0.875 | 0.864 | -1.1pp |
| Precision | 0.880 | 0.907 | +2.7pp ✅ |
| Recall | 0.894 | 0.869 | -2.5pp ❌ |
| Tiempo entrenamiento (GPU) | 8.5 min | 39.7 min | 4.7× más |

### Hallazgo metodológico importante para la tesis

**Más capacidad ≠ mejor en datasets chicos.**

- **Mejoraron drásticamente**:
  - Detection rate (4 misses menos, de 6 → 2): mayor resolución captura manos
    pequeñas o lejanas que el modelo s no veía.
  - IoU promedio (+12pp): las bboxes están ahora casi pegadas a la mano real
    (mediana 0.95). yolov8m + 960 px tiene muchísimo más detalle para regresión
    de coordenadas precisas.
- **Empeoró ligeramente**:
  - Accuracy de clasificación CUANDO detecta: bajó 2.4pp.
  - El modelo más grande **sobreajusta levemente** sobre 556 imgs de train
    (26M params para 556 ejemplos ≈ 47k params por sample — muchísimo).
- **Neto**: +0.86pp global, costo computacional 4.7× mayor.

### Implicación para la tesis

> *"En un dataset pequeño y bien-estructurado como LSA16, aumentar la capacidad
> del modelo (yolov8s → yolov8m) y la resolución (640 → 960) mejora la
> **localización** sustancialmente (IoU promedio +12pp) pero **no mejora la
> clasificación**, e incluso la empeora ligeramente por sobreajuste. El cuello
> de botella ya no es la capacidad del modelo, sino la similitud visual
> intrínseca entre algunas configuraciones manuales (V vs Horns, Beak vs Index,
> L vs Horns)."*

### Recomendación: cuál usar como modelo "final" para la tesis

- **Si lo que importa es classification accuracy**: yolov8s @ 640 (94.59% sobre
  detectadas vs 92.17% del m), entrenable en 8.5 min, menos sobreajuste.
- **Si lo que importa es localización + cobertura**: yolov8m @ 960 (IoU mediana
  0.95, detection rate 98.3%).
- **Para producción real (webcam)**: probablemente yolov8s — más rápido en
  inferencia, accuracy comparable, menos pesado en disco/RAM.

Para la tesis reportaríamos **yolov8s como el modelo principal** y mencionaríamos
yolov8m como ablation de "qué pasa con más capacidad" — con el resultado de que
no mejora classification, lo que aporta una conclusión negativa interesante.

### Confusiones de yolov8m

| Confusión | Cantidad |
|---|:-:|
| Beak → Double | 2 |
| Beak → Fingers | 1 |
| L → Horns | 1 |
| Curve → Flat | 1 |
| Double → Flat | 1 |
| Four → V | 1 |
| Fingers → Fist | 1 |
| Thumb → Fist | 1 |

Distinto patrón al yolov8s pero con número de errores similar (8 vs 6). Las
confusiones siguen siendo entre clases visualmente cercanas.

---

## Resultados YOLO — yolov8s baseline (2026-05-22)

**YOLOv8s preentrenada en COCO, fine-tuned sobre LSA16 raw (640×480 imágenes
completas con persona y dos guantes). Entrenamiento en RTX 3080 Ti, 88 épocas
(cortado manualmente, no por early stopping).**

| Métrica | Valor |
|---|---:|
| Detection rate | 94.9% (111/117) |
| Classification accuracy (todas las test) | 89.74% |
| Classification accuracy (solo detectadas) | 94.59% |
| IoU promedio | 0.779 |
| IoU mediana | 0.818 |
| mAP@50 | 0.923 |
| mAP@50-95 | 0.875 |
| Precision | 0.880 |
| Recall | 0.894 |

**Comparación clave para la tesis**:

| Modelo | Input | Acc | Comentario |
|---|---|---:|---|
| LeNet (paper) | mano segmentada + canonical 128×128 | 96.18% | requiere preprocesamiento pesado |
| LeNet canonical (nuestro 10 runs) | mano segmentada + canonical 128×128 | 91.25% ± 7.83% | inestable entre semillas |
| ResNet18 transfer | mano segmentada + canonical 128×128 | 97.62% ± 1.63% | mejor CNN, supera al paper |
| LeNet sobre raw (paper "raw" ≠ nuestro raw) | imagen completa 640×480 | 22.12% | falla en escenario realista |
| **YOLOv8s sobre raw** | **imagen completa 640×480** | **89.74%** | **localiza + clasifica end-to-end** |

**Lectura para la tesis**: en accuracy directa el ResNet18 gana, pero opera
sobre una mano YA recortada y alineada (preprocesamiento manual elaborado).
YOLO, sobre la imagen completa sin ningún preprocesamiento (el escenario
de webcam real), alcanza 89.74% — un orden de magnitud mejor que LeNet sobre
raw (22%). Esta es exactamente la motivación del trabajo: en escenarios
del mundo real, detección espacial (YOLO) es ampliamente superior a
clasificación de imagen entera.

**Confusiones más frecuentes** (matriz de confusión `outputs/02_yolo-gpu/figures/confusion_yolov8s_gpu.png`):
- V → Horns (2 casos): ambos gestos involucran dos dedos extendidos.
- Beak → Fingers, Beak → Index (1 caso c/u): poses cerradas con punta visible.
- Curve → Beak (1 caso), Fingers → Double (1 caso): poses semi-cerradas.

**Tiempo de entrenamiento**:
- RTX 3080 Ti, imgsz=640, batch=32, 88 épocas: **8.5 min** total.
- Mac M3 Pro CPU (intento previo, cancelado): ~5 min por época → 7+ hs estimadas.

---

## Resultados consolidados (estado al 2026-05-22)

| Modelo | Canonical | Augment | Runs | Media | Std | Peor | Mejor |
|---|:-:|:-:|:-:|---:|---:|---:|---:|
| LeNet | ✗ | ✗ | 3  | 83.33% | 2.12% | 81.25% | 86.25% |
| LeNet | ✓ | ✗ | 3  | 95.83% | 1.18% | 95.00% | 97.50% |
| LeNet | ✓ | ✗ | 10 | 91.25% | 7.83% | 70.00% | 97.50% |
| LeNet | ✓ | ✓ | 10 | 91.62% | 6.71% | 75.00% | 100.00% |
| ResNet-18 | ✓ | ✗ | 10 | **97.62%** | **1.63%** | 93.75% | 100.00% |
| LeNet | full image (raw) | ✗ | 10 | 22.12% | 5.76% | — | — |

Paper de referencia (Quiroga et al. 2017, LeNet, 100 runs): **96.18%**.
