# Bitácora de experimentos — Baseline paper

Resultados de la réplica del paper Quiroga et al. 2017 sobre LSA16.

| Fecha | Dataset | Aug | Runs | Épocas | Media | Std | Paper | Gap | Nota |
|---|---|:-:|---:|---:|---:|---:|---:|---:|---|
| 2026-05-22 16:40 | Segmented RGB | — | 3 | 20 | 83.33% | 2.12% | 96.18% | +12.85pp | baseline limpio sin Dropout |
| 2026-05-22 16:46 | Canonical aligned | — | 3 | 20 | 95.83% | 1.18% | 96.18% | +0.35pp | post-canonical alignment |
| 2026-05-22 17:24 | Canonical aligned | 10 | 20 | 91.25% | 7.83% | 96.18% | +4.93pp | réplica fiel del paper, 10 runs |
| 2026-05-22 17:40 | Raw (full image) | 10 | 20 | 22.12% | 5.76% | 83.54% | +61.42pp | réplica raw del paper |
| 2026-05-22 18:22 | Canonical aligned | ✓ | 10 | 20 | 91.62% | 6.71% | 96.18% | +4.56pp | canonical + augmentation |
| 2026-05-22 20:13 | Canonical aligned [ResNet18-transfer] | — | 10 | 20 | 97.62% | 1.63% | — | — | ResNet18 transfer learning |
