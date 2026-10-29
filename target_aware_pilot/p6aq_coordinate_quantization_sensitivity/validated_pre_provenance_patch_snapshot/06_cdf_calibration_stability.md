# Uniform CDF Calibration Stability

The split is exactly P6-A's pre-existing within-grid `calibration_index` parity: even versus odd rows. No favorable split was selected. Both target and candidate raw/theoretical features are transformed with the same half in each counterfactual; the official full calibration remains unchanged.

| role | p50 | p90 | p99 | maximum | min_spearman |
|---|---|---|---|---|---|
| candidate | 0.012767 | 0.213566 | 0.230363 | 0.234522 | 1.000000 |
| target | 0.058934 | 0.205640 | 0.426513 | 0.527156 | 0.985997 |

Split-even versus split-odd survivor retention is 0.9656 (Jaccard 0.9335); official-phi SWD between their selected populations is 0.008309. Tail continuation means clipping is exactly zero. Candidate n=100 transforms preserve within-feature ranks almost perfectly because each half transform is monotone; cross-n TSPLIB ranks can move more.
