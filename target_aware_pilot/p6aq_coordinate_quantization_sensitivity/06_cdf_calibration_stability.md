# Uniform CDF Calibration Stability

The split is exactly P6-A's pre-existing within-grid `calibration_index` parity: even versus odd rows. No favorable split was selected. Both target and candidate raw/theoretical features are transformed with the same half in each counterfactual; the official full calibration remains unchanged.

## Overall absolute score difference

| role | n | p50_abs_difference | p90_abs_difference | p95_abs_difference | p99_abs_difference | max_abs_difference |
|---|---|---|---|---|---|---|
| candidate | 597650 | 0.011252 | 0.128451 | 0.178019 | 0.220774 | 0.234522 |
| target | 240 | 0.058082 | 0.172832 | 0.210194 | 0.291620 | 0.527156 |

## Per-feature diagnostics

| role | feature | p50_abs_difference | p90_abs_difference | p99_abs_difference | max_abs_difference | spearman_split_scores |
|---|---|---|---|---|---|---|
| candidate | spn_sum_of_lowest_edge_values | 0.002023 | 0.002299 | 0.008012 | 0.199434 | 1.000000 |
| candidate | spn_distance_min | 0.000638 | 0.049530 | 0.119410 | 0.179936 | 1.000000 |
| candidate | spn_distance_varcoeff | 0.014283 | 0.034083 | 0.135575 | 0.211943 | 1.000000 |
| candidate | spn_nng_3_strong_components_varcoeff | 0.126729 | 0.213566 | 0.230363 | 0.234522 | 1.000000 |
| candidate | spn_mst_dists_sd | 0.015067 | 0.024596 | 0.056441 | 0.208138 | 1.000000 |
| target | spn_sum_of_lowest_edge_values | 0.059786 | 0.152774 | 0.184090 | 0.194707 | 0.996092 |
| target | spn_distance_min | 0.047314 | 0.205640 | 0.426513 | 0.527156 | 0.995224 |
| target | spn_distance_varcoeff | 0.084729 | 0.178948 | 0.292892 | 0.341933 | 0.985997 |
| target | spn_nng_3_strong_components_varcoeff | 0.064358 | 0.186637 | 0.240217 | 0.251946 | 0.995332 |
| target | spn_mst_dists_sd | 0.051614 | 0.131794 | 0.231878 | 0.258145 | 0.996743 |

Split-even versus split-odd survivor retention is 0.9656 (Jaccard 0.9335); official-phi SWD between their selected populations is 0.008309 and energy distance is 0.001381. The even and odd selections retain 0.9794 and 0.9750 of the official selection. Tail continuation means clipping is exactly zero. Candidate n=100 transforms preserve within-feature ranks almost perfectly because each half transform is monotone; cross-n TSPLIB ranks can move more.
