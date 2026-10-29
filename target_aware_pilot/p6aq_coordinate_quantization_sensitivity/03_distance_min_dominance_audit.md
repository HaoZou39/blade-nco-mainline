# distance_min Dominance Audit

| analysis | feature | component_total | contribution_share |
|---|---|---|---|
| AllCandidatesNearest | spn_sum_of_lowest_edge_values | 9871.613960 | 0.013326 |
| AllCandidatesNearest | spn_distance_min | 620245.278012 | 0.837276 |
| AllCandidatesNearest | spn_distance_varcoeff | 39010.481971 | 0.052661 |
| AllCandidatesNearest | spn_nng_3_strong_components_varcoeff | 52555.250173 | 0.070945 |
| AllCandidatesNearest | spn_mst_dists_sd | 19106.479549 | 0.025792 |
| Guided10kExactOT | spn_sum_of_lowest_edge_values | 37983.103635 | 0.244005 |
| Guided10kExactOT | spn_distance_min | 40516.035179 | 0.260277 |
| Guided10kExactOT | spn_distance_varcoeff | 8365.008918 | 0.053737 |
| Guided10kExactOT | spn_nng_3_strong_components_varcoeff | 3577.105145 | 0.022979 |
| Guided10kExactOT | spn_mst_dists_sd | 65223.999589 | 0.419002 |
| Random10kExactOT | spn_sum_of_lowest_edge_values | 59225.926370 | 0.228958 |
| Random10kExactOT | spn_distance_min | 90332.090050 | 0.349209 |
| Random10kExactOT | spn_distance_varcoeff | 13495.419270 | 0.052171 |
| Random10kExactOT | spn_nng_3_strong_components_varcoeff | 4021.808647 | 0.015548 |
| Random10kExactOT | spn_mst_dists_sd | 91600.924017 | 0.354114 |

The published 83.7% is the share of squared distance after assigning every one of 119,530 candidates to its individually nearest target, without capacity constraints or survivor selection. In the actual Guided10k exact OT objective, `distance_min` contributes 26.03%; Random10k contributes 34.92%. The squared Euclidean objective is exactly additive over all five transformed dimensions, including the frozen finite score for the NNG singleton semantic atom.

## distance_min distribution

| group | representation | median | iqr | p01 | p05 | p95 | p99 |
|---|---|---|---|---|---|---|---|
| TSPLIB48 | raw | 0.004971 | 0.006879 | 0.000070 | 0.000557 | 0.023417 | 0.038765 |
| TSPLIB48 | theory | 1.111329 | 1.592765 | 0.019958 | 0.136410 | 5.060836 | 6.541567 |
| Candidates119530 | raw | 0.000393 | 0.000590 | 0.000021 | 0.000053 | 0.002362 | 0.006510 |
| Candidates119530 | theory | 0.028861 | 0.043038 | 0.001530 | 0.003894 | 0.173381 | 0.474927 |
| Random10k | raw | 0.000394 | 0.000602 | 0.000021 | 0.000051 | 0.002346 | 0.006650 |
| Random10k | theory | 0.029219 | 0.044376 | 0.001546 | 0.003684 | 0.169975 | 0.481273 |
| Guided10k | raw | 0.002734 | 0.002568 | 0.000273 | 0.000819 | 0.009249 | 0.014776 |
| Guided10k | theory | 0.198549 | 0.188097 | 0.019790 | 0.059067 | 0.672590 | 1.074453 |

The stored TSPLIB48 raw median is 0.004971, versus 0.000393 for the frozen synthetic bank (12.65×). Raw/theory/phi quantiles and tail-extrapolation diagnostics for every Top5 feature and population are in `cache/feature_distribution_statistics.csv`. The transform uses tail continuation and therefore has zero clipping by construction.
