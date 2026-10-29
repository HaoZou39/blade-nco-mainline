# E0/E1 Evolution Comparison

| Metric | G0 shared | E0 G1 | E0 G2 | E1 G1 | E1 G2 |
|---|---:|---:|---:|---:|---:|
| Sliced Wasserstein | 0.689199 | 0.658552 | 0.618302 | 0.394931 | 0.321713 |
| 3x3 JS | 0.543551 | 0.511530 | 0.479869 | 0.329345 | 0.177498 |
| Strict target fraction | 7.07% | 7.34% | 7.49% | 21.76% | 48.59% |
| Coverage | 45/48 | 44/48 | 46/48 | 45/48 | 46/48 |
| Effective lineages | 531.4 | 510.3 | 490.2 | 606.8 | 598.4 |
| Auxiliary JS | 0.498164 | 0.505008 | 0.513555 | 0.511847 | 0.488114 |

Raw offspring, survivor population, and bounded archive are reported as separate object types in `02_generation_metrics.csv`; archive retention is identical across arms and is not substituted for the raw-generation comparison.

E1's lower G2 parent→offspring improvement rate does not contradict population adaptation: E1 parents already lie closer to the target support. The causal comparison is the survivor trajectory and next-generation raw distribution under equal mutation budgets, not improvement rate alone.
