"""Aplica canonical_align a todas las imágenes de lsa16_segmented_right_hand/
y guarda los resultados en lsa16_segmented_canonical/.

Después de correr este script, el modelo puede entrenarse contra el dataset
canónico (con manos alineadas verticalmente) en vez del original.
"""
from __future__ import annotations

import os
import sys
import time

import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.data import SEGMENTED_DIR  # noqa: E402
from src.preprocessing import canonical_align  # noqa: E402

CANONICAL_DIR = os.path.join(PROJECT_ROOT, "lsa16_segmented_canonical")


def main() -> None:
    os.makedirs(CANONICAL_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(SEGMENTED_DIR) if f.endswith(".png"))
    print(f"Procesando {len(files)} imágenes desde {SEGMENTED_DIR}")
    print(f"Destino: {CANONICAL_DIR}")

    t0 = time.time()
    ok, fail = 0, 0
    for i, fname in enumerate(files, 1):
        src_path = os.path.join(SEGMENTED_DIR, fname)
        dst_path = os.path.join(CANONICAL_DIR, fname)
        bgr = cv2.imread(src_path)
        if bgr is None:
            print(f"  ⚠️  No se pudo leer {fname}")
            fail += 1
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        try:
            processed = canonical_align(rgb, target_size=128)
        except Exception as e:
            print(f"  ⚠️  Error procesando {fname}: {e}")
            fail += 1
            continue
        out_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
        cv2.imwrite(dst_path, out_bgr)
        ok += 1
        if i % 100 == 0:
            print(f"  {i}/{len(files)} procesadas ({time.time()-t0:.1f}s)")

    print(f"\n✓ Listo. {ok} ok | {fail} fallos | {time.time()-t0:.1f}s total")


if __name__ == "__main__":
    main()
