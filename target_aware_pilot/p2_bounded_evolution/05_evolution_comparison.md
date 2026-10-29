# E0/E1 Evolution Comparison

| Metric | G0 shared | E0 G1 | E0 G2 | E1 G1 | E1 G2 |
|---|---:|---:|---:|---:|---:|
| Sliced Wasserstein | 0.689199 | 0.651905 | 0.619057 | 0.390737 | 0.312119 |
| 3x3 JS | 0.543551 | 0.516382 | 0.488987 | 0.335041 | 0.179518 |
| Strict target fraction | 7.07% | 7.02% | 8.02% | 21.49% | 48.13% |
| Coverage | 45/48 | 45/48 | 45/48 | 45/48 | 45/48 |
| Effective lineages | 531.4 | 507.1 | 491.2 | 603.6 | 596.3 |
| Auxiliary JS | 0.498164 | 0.502925 | 0.513651 | 0.509750 | 0.484463 |

Raw offspring, survivor population, and bounded archive are reported as separate object types in `02_generation_metrics.csv`; archive retention is identical across arms and is not substituted for the raw-generation comparison.

E1's lower G2 parent→offspring improvement rate does not contradict population adaptation: E1 parents already lie closer to the target support. The causal comparison is the survivor trajectory and next-generation raw distribution under equal mutation budgets, not improvement rate alone.
