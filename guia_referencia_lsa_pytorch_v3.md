# Guía de Referencia y Especificación Técnica: Reconocimiento de Configuraciones Manuales en LSA con PyTorch (V3)

Este documento sirve como marco teórico, estado del arte y especificación técnica para el desarrollo de un proyecto de investigación académica en el área de Visión por Computadora y Redes Neuronales. El objetivo es el reconocimiento automático de configuraciones manuales (*handshapes*) de la Lengua de Señas Argentina (LSA) utilizando el dataset LSA16 como base.

---

## 1. Contexto Lingüístico y Definición del Problema
El reconocimiento automático de la lengua de señas (SLR) es un problema complejo que abarca el procesamiento de imágenes, sistemas inteligentes y lingüística. La traducción completa de una seña involucra un flujo secuencial de tareas:
1. Localización y segmentación de las manos del intérprete.
2. Reconocimiento y clasificación de la forma o configuración estática de las manos (*Handshape Recognition*).
3. Seguimiento temporal (*Tracking*) para detectar las trayectorias y movimientos.
4. Análisis de rasgos no manuales (expresiones faciales, movimiento de labios).
5. Asignación de significado semántico y traducción al lenguaje escrito/hablado.

La calidad del reconocimiento de la configuración manual es el factor más crítico para el éxito de un sistema de traducción general. Este proyecto delimita el alcance exclusivamente a la clasificación estática de la forma de la mano, omitiendo las restricciones temporales del video para garantizar la viabilidad técnica en el marco de una asignatura cuatrimestral.

---

## 2. Interpretación de Imágenes y Preprocesamiento Avanzado

### 2.1. Referencia de Origen de las Imágenes
Las imágenes utilizadas corresponden al dataset **LSA16**, detallado originalmente en la **Sección 2.1 del paper de 2016 ("Methods - Argentinian Sign Language Handshapes Database")** y retomado en la **Sección 3.1 del paper de 2017 ("Experiments and Results - Datasets")**. 
* **Composición:** Contiene un total de 800 imágenes. Participaron 10 sujetos experimentales, ejecutando 5 repeticiones de 16 configuraciones manuales diferentes (las más utilizadas en el léxico de la LSA).
* **Entorno controlado:** Para aislar el problema de la variación del tono de piel y simplificar la segmentación, los sujetos vistieron ropa negra, se ubicaron frente a un fondo blanco con iluminación controlada y utilizaron **guantes de colores fluorescentes**.

### 2.2. Flujo de Preprocesamiento de Imágenes (Sección 2.2, Paper 2016)
Antes de alimentar cualquier modelo de clasificación, las imágenes pasan por un pipeline estricto de normalización geométrica para remover ruido de traslación, escala y rotación:
1. **Máscara de Segmentación:** Se determina el componente conectado más grande en la imagen para obtener la máscara de segmentación limpia.
2. **Cálculo de Inclinación:** Se calculan los ejes principales de los píxeles de la mano para determinar el ángulo de inclinación $\theta$.
3. **Orientación Canónica:** Se rota la imagen por $-\theta$ para colocarla en una orientación vertical estándar.
4. **Corrección de Inversión (180°):** El sistema cuenta el número de cruces posibles de líneas horizontales para estimar la posición de los dedos y voltear la imagen si quedó apuntando hacia abajo.
5. **Redimensión y Centrado:** La imagen resultante se re-muestrea a un tamaño fijo de **128x128 píxeles** sin alterar su relación de aspecto original.

---

## 3. Diccionario y Significado de Clases (Labels)
Para que el modelo tenga utilidad en inferencia y sea interpretable por humanos, a continuación se detalla el mapeo exacto de las 16 clases estructurales de LSA16 (nombre de clase en inglés según el dataset original, y su configuración física equivalente):

* **0 - Five:** Mano abierta, cinco dedos.
* **1 - Four:** Cuatro dedos.
* **2 - Horns:** Cuernos / Gesto de rock.
* **3 - Curve:** Mano curva.
* **4 - Fingers together:** Dedos juntos / "Montoncito".
* **5 - Double:** Doble.
* **6 - Hook:** Gancho.
* **7 - Index:** Dedo índice.
* **8 - L:** Forma de letra L.
* **9 - Flat Hand:** Mano plana.
* **10 - Mitten:** Manopla / Dedos juntos rectos.
* **11 - Beak:** Pico.
* **12 - Thumb:** Pulgar arriba.
* **13 - Fist:** Puño cerrado.
* **14 - Telephone:** Teléfono / Gesto de "Shaka".
* **15 - V:** Forma de V / Gesto de paz.

*Nota técnica:* Estas formas estáticas no son necesariamente palabras completas o letras aisladas del abecedario, sino los bloques de construcción estructurales a los que, en la vida real, se les suma movimiento para formar una seña con sentido semántico completo.

---

## 4. Estado del Arte y Baselines Académicos

### 4.1. Baseline Convolucional: Redes Profundas (Sección 3.2, Paper 2017)
Línea base fundamental del proyecto sobre imágenes pre-segmentadas a color (RGB):

| Método de Clasificación | Tipo de Datos / Descriptor | Precisión en LSA16 (%) | Complejidad Estructural |
| :--- | :--- | :---: | :--- |
| **ProbSom (2016)** | Transformada de Radon (32x32) | 92.30% | Baja (Modelo estadístico competitivo) |
| **LeNet (1998)** | Segmentación RGB (4 capas conv) | 95.78% | Baja-Media (Liviana, rápida de entrenar) |
| **AllConvolutional (2014)**| Segmentación RGB (Sin Max-Pooling) | 94.56% | Media |
| **VGG16 (2014)** | Segmentación RGB (Filtros 3x3) | **95.92%** | Muy Alta (~138M parámetros, costosa) |
| **ResNet-34 (2015)** | Segmentación RGB (Bloques residuales)| 93.49% | Media-Alta (Con conexiones shortcut) |

### 4.2. Impacto Empírico del Preprocesamiento (Sección 3.3, Paper 2017)
Demostración de la importancia de segmentar la mano:

| Esquema de Preprocesamiento | Precisión LeNet (%) | Implicación Técnica |
| :--- | :---: | :--- |
| **Raw (RGB Original con Fondo)** | 83.54% | El modelo sufre ruido por el entorno. |
| **Segmented Hand (RGB a Color)** | **96.18%** | **Rendimiento óptimo.** Preserva texturas internas y bordes. |
| **Grayscale (Escala de Grises)** | 87.08% | La pérdida de color degrada detección de falanges. |

---

## 5. Diseño de la Arquitectura Propuesta en PyTorch

### 5.1. Bloque 1: Réplica del Baseline Histórico (LeNet Modificada)
```python
import torch
import torch.nn as nn

class LeNetLSA16(nn.Module):
    def __init__(self, num_classes=16):
        super(LeNetLSA16, self).__init__()
        # Entrada esperada: [3, 128, 128]
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2), # -> [32, 64, 64]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2), # -> [64, 32, 32]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2), # -> [128, 16, 16]

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2)  # -> [256, 8, 8]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

### 5.2. Diccionario de Inferencia
Implementación del mapeo de clases para el script de evaluación/predicción:

```python
LSA16_CLASSES = {
    0: "Five (Mano abierta, cinco dedos)",
    1: "Four (Cuatro dedos)",
    2: "Horns (Cuernos / Gesto de rock)",
    3: "Curve (Mano curva)",
    4: "Fingers together (Dedos juntos / Montoncito)",
    5: "Double (Doble)",
    6: "Hook (Gancho)",
    7: "Index (Dedo índice)",
    8: "L (Forma de letra L)",
    9: "Flat Hand (Mano plana)",
    10: "Mitten (Manopla / Dedos juntos rectos)",
    11: "Beak (Pico)",
    12: "Thumb (Pulgar arriba)",
    13: "Fist (Puño cerrado)",
    14: "Telephone (Teléfono / Gesto de Shaka)",
    15: "V (Forma de V / Gesto de paz)"
}
```

---

## 6. Metodología de Validación y Parámetros
* **Optimizador:** ADAM con *learning rate* fija de `0.0007`.
* **Esquema de Validación:** Múltiples iteraciones de validación cruzada estratificada por submuestreo aleatorio (90% entrenamiento, 10% testeo).

---

## 7. Instrucciones de Prompting para Desarrollo Asistido por CLI
1. *"Generá un script de PyTorch utilizando la clase LeNetLSA16 y el diccionario LSA16_CLASSES especificados en las secciones 5.1 y 5.2. Usá el optimizador ADAM con LR=0.0007."*
2. *"Escribí el bucle de validación cruzada y al imprimir las predicciones, traducilas usando el diccionario LSA16_CLASSES para que el log sea legible."*
