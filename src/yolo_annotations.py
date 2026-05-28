"""Generación automática de anotaciones YOLO desde el dataset LSA16.

Estrategia (ver outputs/01_baseline/notebook_notes.md §"Diseño YOLO"):
1. Para cada imagen raw, se usa la versión segmentada (`lsa16_segmented_right_hand/`)
   como referencia per-imagen del color de la mano derecha.
2. Se thresholdea la raw por el Hue dominante de la segmented ± tolerancia.
3. Componente conectado más grande → bbox de la mano derecha.

Por qué per-imagen: descubrimos sondeando que el color de la mano derecha NO
es constante en el dataset. En la mayoría de imágenes es rosa/magenta
(Hue ~175) pero en algunas es cian (Hue ~90). Usando la segmented como
referencia el pipeline se adapta automáticamente.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# Tolerancia en grados de Hue alrededor del color dominante del segmented.
# 15° captura sombras y variaciones de iluminación sin invadir el otro guante.
HUE_TOLERANCE = 15
# Saturación y Value mínimos: filtran fondo blanco, ropa negra, sombras.
MIN_SATURATION = 100
MIN_VALUE = 50
# Píxeles mínimos en el componente conectado para considerarlo válido.
MIN_COMPONENT_AREA = 200


@dataclass(frozen=True)
class BBox:
    """Bounding box en coordenadas absolutas de píxel."""
    x: int       # left
    y: int       # top
    w: int       # width
    h: int       # height

    def to_yolo(self, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        """Formato YOLO normalizado: (xc, yc, w, h) en [0, 1]."""
        xc = (self.x + self.w / 2) / img_w
        yc = (self.y + self.h / 2) / img_h
        return xc, yc, self.w / img_w, self.h / img_h


def _dominant_hue(rgb: np.ndarray, min_sat: int = MIN_SATURATION) -> float | None:
    """Hue dominante (mediana) de los píxeles saturados de una imagen RGB.

    Devuelve None si no hay píxeles saturados suficientes.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    sat_mask = (s > min_sat) & (v > MIN_VALUE)
    if sat_mask.sum() < 100:
        return None
    return float(np.median(h[sat_mask]))


def _hue_band_mask(hue_channel: np.ndarray, target_hue: float, tolerance: int) -> np.ndarray:
    """Máscara booleana de píxeles cuyo Hue ∈ [target-tol, target+tol], con wrap-around.

    El espacio de Hue de OpenCV es [0, 180) circular: 0 y 180 son el mismo color (rojo).
    """
    lo = target_hue - tolerance
    hi = target_hue + tolerance
    if lo < 0:
        return (hue_channel >= (lo + 180)) | (hue_channel <= hi)
    if hi >= 180:
        return (hue_channel >= lo) | (hue_channel <= (hi - 180))
    return (hue_channel >= lo) & (hue_channel <= hi)


def detect_right_hand_bbox_by_color(
    raw_rgb: np.ndarray,
    segmented_rgb: np.ndarray,
    hue_tolerance: int = HUE_TOLERANCE,
) -> BBox | None:
    """Detección por color (fallback). Ver detect_right_hand_bbox como método principal.

    Detecta el bounding box buscando en la raw los píxeles cuyo Hue matchea
    el color dominante de la segmented. Puede confundirse con regiones del
    fondo o piel que tengan colores similares.
    """
    target_hue = _dominant_hue(segmented_rgb)
    if target_hue is None:
        return None

    hsv = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    mask = (
        _hue_band_mask(h, target_hue, hue_tolerance)
        & (s > MIN_SATURATION)
        & (v > MIN_VALUE)
    ).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    if stats[largest_idx, cv2.CC_STAT_AREA] < MIN_COMPONENT_AREA:
        return None

    x = int(stats[largest_idx, cv2.CC_STAT_LEFT])
    y = int(stats[largest_idx, cv2.CC_STAT_TOP])
    w = int(stats[largest_idx, cv2.CC_STAT_WIDTH])
    height = int(stats[largest_idx, cv2.CC_STAT_HEIGHT])
    return BBox(x=x, y=y, w=w, h=height)


def detect_right_hand_bbox(
    raw_rgb: np.ndarray,
    segmented_rgb: np.ndarray,
) -> BBox | None:
    """Localiza la mano derecha en la raw vía template matching enmascarado.

    La imagen segmentada ES una copia recortada de la mano derecha del raw
    (con el fondo seteado a negro). Usando cv2.matchTemplate con una máscara
    de los píxeles no-negros del segmented, encontramos la posición *exacta*
    donde el segmented "encaja" en la raw. Es inmune a confusiones con piel
    o fondo (a diferencia del método por color).

    Si el match es de baja calidad (por ej. el segmented fue rotado y no
    se corresponde literalmente al raw), retorna None.
    """
    raw_bgr = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)
    tpl_bgr = cv2.cvtColor(segmented_rgb, cv2.COLOR_RGB2BGR)

    # Máscara: dónde el segmented es no-negro
    gray = cv2.cvtColor(segmented_rgb, cv2.COLOR_RGB2GRAY)
    mask = (gray > 10).astype(np.uint8) * 255
    if mask.sum() == 0:
        return None

    tpl_h, tpl_w = tpl_bgr.shape[:2]
    raw_h, raw_w = raw_bgr.shape[:2]
    if tpl_h > raw_h or tpl_w > raw_w:
        return None

    # TM_CCORR_NORMED con mask: producto interno normalizado solo en los pixels
    # válidos del template. Score ∈ [0, 1], 1 = match perfecto.
    res = cv2.matchTemplate(raw_bgr, tpl_bgr, cv2.TM_CCORR_NORMED, mask=mask)
    # matchTemplate puede devolver inf/nan en pixels donde la normalización falla
    res = np.where(np.isfinite(res), res, -1.0)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    if max_val < 0.95:  # match débil → probablemente el segmented fue procesado
        return None

    x, y = int(max_loc[0]), int(max_loc[1])
    return BBox(x=x, y=y, w=tpl_w, h=tpl_h)
