# E0/E1 Evolution Comparison

| Metric | G0 shared | E0 G1 | E0 G2 | E1 G1 | E1 G2 |
|---|---:|---:|---:|---:|---:|
| Sliced Wasserstein | 0.689199 | 0.653592 | 0.615636 | 0.393064 | 0.316128 |
| 3x3 JS | 0.543551 | 0.517417 | 0.486933 | 0.334291 | 0.178785 |
| Strict target fraction | 7.07% | 7.37% | 7.81% | 20.96% | 46.96% |
| Coverage | 45/48 | 45/48 | 45/48 | 45/48 | 45/48 |
| Effective lineages | 531.4 | 503.3 | 478.8 | 604.8 | 589.7 |
| Auxiliary JS | 0.498164 | 0.506255 | 0.512938 | 0.514473 | 0.487887 |

Raw offspring, survivor population, and bounded archive are reported as separate object types in `02_generation_metrics.csv`; archive retention is identical across arms and is not substituted for the raw-generation comparison.

E1's lower G2 parent→offspring improvement rate does not contradict population adaptation: E1 parents already lie closer to the target support. The causal comparison is the survivor trajectory and next-generation raw distribution under equal mutation budgets, not improvement rate alone.
