# M4 Hardness Distribution

Pooled thresholds fixed before comparison: P90=0.12987216, P95=0.15288566. Positive E1−E0 means E1 is harder.

| replicate | arm | mean | median | p90 | p95 | hard_tail_p90_fraction | hard_tail_p95_fraction |
|---|---|---|---|---|---|---|---|
| Original | E0 | 0.088504771 | 0.081039043 | 0.14701572 | 0.17105768 | 0.162 | 0.0829 |
| Original | E1 | 0.061719647 | 0.0558689 | 0.10503534 | 0.12166122 | 0.0363 | 0.015 |
| S1 | E0 | 0.089937069 | 0.083062066 | 0.14759683 | 0.17178971 | 0.1649 | 0.0863 |
| S1 | E1 | 0.061585923 | 0.055748988 | 0.10457759 | 0.12314509 | 0.0393 | 0.0143 |
| S2 | E0 | 0.088875069 | 0.08186009 | 0.14796679 | 0.17319977 | 0.1632 | 0.0879 |
| S2 | E1 | 0.061028451 | 0.055396109 | 0.10277126 | 0.12033733 | 0.0343 | 0.0136 |

## E1 minus E0

| replicate | mean | median | p90 | p95 | hard_tail_p90_fraction | hard_tail_p95_fraction | wasserstein | rank_biserial_E1_vs_E0 |
|---|---|---|---|---|---|---|---|---|
| Original | -0.026785124 | -0.025170143 | -0.041980381 | -0.049396461 | -0.1257 | -0.0679 | 0.026785124 | -0.38006281 |
| S1 | -0.028351146 | -0.027313078 | -0.043019243 | -0.048644627 | -0.1256 | -0.072 | 0.028351786 | -0.40251929 |
| S2 | -0.027846618 | -0.026463981 | -0.045195531 | -0.052862441 | -0.1289 | -0.0743 | 0.027846618 | -0.38934994 |

Instance-level tests and bootstrap intervals are descriptive conditional on each frozen population; 10k instances are not 10k evolution replicates. Structure/hardness correlations are in `logs/hardness_structure_correlations.csv` and cannot by themselves establish training value.
