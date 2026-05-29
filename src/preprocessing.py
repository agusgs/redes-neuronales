"""Canonical alignment del paper Quiroga et al. (Sec 2.2 del paper LSA16 2016).

Pipeline:
1. Binary mask (cualquier pixel no negro = mano).
2. Componente conectado más grande para limpiar ruido.
3. PCA sobre los píxeles de la mano → eje principal.
4. Rotar la imagen para alinear el eje principal con la vertical.
5. Detectar inversión 180° contando cruces horizontales: el lado con más cruces
   son los dedos (separados → muchas transiciones background/mano).
6. Crop al bbox de la mano + resize a 128x128 manteniendo aspect ratio (padding negro).
"""
from __future__ import annotations

import numpy as np
import cv2


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Devuelve una máscara con solo la componente conectada de mayor área."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    # stats[0] es el background, ignoramos
    largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest_idx).astype(np.uint8) * 255


def _principal_axis_angle_deg(mask: np.ndarray) -> float:
    """Ángulo del eje principal de la mano respecto al eje X (en grados)."""
    ys, xs = np.where(mask > 0)
    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, -1]  # eigenvector con mayor eigenvalue
    return float(np.degrees(np.arctan2(principal[1], principal[0])))


def _count_crossings(mask_strip: np.ndarray) -> int:
    """Cuenta transiciones background↔mano a lo largo de filas horizontales."""
    binary = (mask_strip > 0).astype(np.int32)
    diff = np.abs(np.diff(binary, axis=1))
    return int(diff.sum())

def _fingers_at_bottom(mask: np.ndarray) -> bool:
    """True si los dedos están en la mitad inferior de la mano.
    
    Heurística Combinada: 
    1. Transiciones: Dedos separados generan más cruces horizontales. (Falla en 1 solo dedo).
    2. Centro de Masa: La palma tiene más concentración de píxeles. (Falla en dedos en gancho).
    Solo rotamos la imagen 180 grados si AMBAS heurísticas coinciden en que 
    los dedos están apuntando hacia abajo, logrando robustez en todas las clases.
    """
    ys, _ = np.where(mask > 0)
    if len(ys) == 0:
        return False
        
    y_min, y_max = ys.min(), ys.max()
    y_mid = (y_min + y_max) // 2
    
    # 1. Heurística del Centro de Masa
    y_mid_box = (y_min + y_max) / 2.0
    y_com = ys.mean()
    com_flips = y_com < y_mid_box
    
    # 2. Heurística de Transiciones (Paper Original)
    upper_crossings = _count_crossings(mask[y_min:y_mid])
    lower_crossings = _count_crossings(mask[y_mid:y_max + 1])
    cross_flips = lower_crossings > upper_crossings
    
    return com_flips and cross_flips



def _aspect_preserving_resize(img: np.ndarray, target_size: int) -> np.ndarray:
    """Resize manteniendo aspect ratio + padding negro hasta target_size × target_size."""
    h, w = img.shape[:2]
    if h >= w:
        new_h = target_size
        new_w = max(1, int(round(w * target_size / h)))
    else:
        new_w = target_size
        new_h = max(1, int(round(h * target_size / w)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_size, target_size, img.shape[2] if img.ndim == 3 else 1),
                      dtype=img.dtype)
    if img.ndim == 2:
        canvas = canvas[..., 0]
    pad_y = (target_size - new_h) // 2
    pad_x = (target_size - new_w) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas


def canonical_align(image: np.ndarray, target_size: int = 128, mask_threshold: int = 10) -> np.ndarray:
    """Aplica el pipeline de alineamiento canónico al segmented RGB image.

    Args:
        image: ndarray RGB (uint8) — imagen segmentada (mano sobre fondo negro).
        target_size: tamaño final del lado mayor.
        mask_threshold: pixeles con intensidad > este valor cuentan como mano.

    Returns:
        ndarray RGB (uint8) de target_size × target_size, mano vertical y centrada.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image, got shape {image.shape}")

    # 1. Máscara binaria
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, mask_threshold, 255, cv2.THRESH_BINARY)

    # 2. Componente conectado más grande
    mask = _largest_connected_component(mask)
    if mask.sum() == 0:
        return _aspect_preserving_resize(image, target_size)

    # 3-4. PCA y rotación
    angle = _principal_axis_angle_deg(mask)
    # Queremos el eje principal vertical → angle final 90° (o -90°). Rotamos por (angle - 90)
    # con cv2.getRotationMatrix2D, ángulo positivo = sentido antihorario.
    rotation_deg = angle - 90.0
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, rotation_deg, scale=1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderValue=(0, 0, 0))
    rotated_mask = cv2.warpAffine(mask, M, (w, h), borderValue=0)

    # 5. Corrección 180° si los dedos quedaron abajo
    if _fingers_at_bottom(rotated_mask):
        rotated = cv2.rotate(rotated, cv2.ROTATE_180)
        rotated_mask = cv2.rotate(rotated_mask, cv2.ROTATE_180)

    # 6. Crop al bbox + resize manteniendo aspect ratio
    ys, xs = np.where(rotated_mask > 0)
    if len(ys) == 0:
        return _aspect_preserving_resize(rotated, target_size)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    cropped = rotated[y0:y1 + 1, x0:x1 + 1]
    return _aspect_preserving_resize(cropped, target_size)
