# Provisional Structure10k Distribution Audit

## Core quality metrics

| Metric | Structure10k | C04 Hard10k |
|---|---:|---:|
| Unique instances | 10,000 | 10,000 |
| Strict-region instances | 7,781 (77.81%) | 729 |
| TSPLIB48 neighborhood coverage | 47/48 | 45/48 |
| 3×3 JS divergence | 0.000000 | 0.568365 |
| Effective occupied cells | 6.306 | 2.592 |
| Largest cell share | 22.92% | 72.54% |
| FFT/NN-CV auxiliary JS | 0.4754 | 0.4552 |
| HGB predicted log1p hardness mean | 0.06312 | 0.10773 |

Structure10k improves substantially over the full E1 cumulative-pool JS of 0.3206 and the original Hard10k. This is expected from the exact coarse quota constraint and is not evidence of generalization beyond the two frozen selection features.

## Diversity and provenance

- Lineages: 945 observed; effective count 755.1; largest lineage 48 (0.48%).
- Direct parents/families: 5,433 observed; effective count 4437.6; maximum 9 (0.09%).
- Origins: `{"E1_new_offspring": 8758, "c04_baseline": 1242}`.
- Known formal-training overlap: 0.
- Coordinate validity/hash verification: PASS.

The auxiliary FFT/NN-CV JS is diagnostic only. FFT uses raw-count occupancy and remains node-count sensitive across the variable-size TSPLIB48 reference, so it cannot override the frozen two-feature audit.

## Uncovered reference neighborhoods

- `eil51`: nearest distance 0.184621 vs strict radius 0.095956.

The full provisional coordinate bundle is `provisional_structure10k.npz`, SHA-256 `d93417d90b1b91815f0124b1ee8be31e003abf07afd1c3527cc34fbbbe45e231`. It contains no tours or Oracle labels.
