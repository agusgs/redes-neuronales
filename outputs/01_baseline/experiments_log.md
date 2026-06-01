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
| 2026-05-28 09:54 | Canonical aligned [LeNetLSA16] | � | 10 | 20 | 93.25% | 1.87% | 96.18% | +2.93pp | r�plica paper |
| 2026-05-28 10:19 | Canonical aligned [LeNetLSA16] | ✓ | 10 | 20 | 92.25% | 3.48% | 96.18% | +3.93pp | canonical + augmentation |
| 2026-05-28 10:26 | Canonical aligned [ResNet18-transfer] | — | 10 | 20 | 97.75% | 1.56% | — | — | ResNet18 transfer |
| 2026-05-28 10:53 | Raw (full image) [LeNetLSA16] | — | 10 | 20 | 23.12% | 5.60% | 83.54% | +60.42pp | LeNet sobre raw |
| 2026-05-29 13:03 | Canonical aligned [LeNetLSA16] | — | 10 | 20 | 94.00% | 2.73% | 96.18% | +2.18pp | — |
| 2026-05-29 13:09 | Canonical aligned [LeNetLSA16] | ✓ | 10 | 20 | 94.50% | 3.32% | 96.18% | +1.68pp | — |
| 2026-05-29 13:18 | Canonical aligned [ResNet18-transfer] | — | 10 | 20 | 97.38% | 1.63% | — | — | — |
| 2026-05-29 13:27 | Canonical aligned [ResNet18-transfer] | ✓ | 10 | 20 | 98.00% | 1.50% | — | — | — |
| 2026-05-29 17:44 | Segmented RGB [LeNetLSA16] | — | 10 | 20 | 83.12% | 2.45% | 96.18% | +13.06pp | — |
| 2026-05-30 08:36 | Raw (full image) [ResNet18-transfer] | — | 10 | 20 | 77.00% | 4.34% | — | — | — |
