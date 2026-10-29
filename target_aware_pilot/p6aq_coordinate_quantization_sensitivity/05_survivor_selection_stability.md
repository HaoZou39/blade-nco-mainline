# Survivor10k Selection Stability

| amplitude_a | retention_min | retention_median | retention_max | jaccard_min | cost_cross_max_change | swd_membership_max | energy_membership_max |
|---|---|---|---|---|---|---|---|
| 0.050000 | 0.979700 | 0.981900 | 0.989400 | 0.960208 | 0.009323 | 0.006999 | 0.002126 |
| 0.250000 | 0.978800 | 0.981900 | 0.987300 | 0.958480 | 0.009755 | 0.006837 | 0.002243 |
| 0.500000 | 0.973500 | 0.979500 | 0.983600 | 0.948368 | 0.010468 | 0.010026 | 0.002181 |

## Target Top5 phi sensitivity

| amplitude_a | feature | median | max |
|---|---|---|---|
| 0.050000 | spn_distance_min | 0.000482 | 0.420014 |
| 0.050000 | spn_distance_varcoeff | 0.000001 | 0.007753 |
| 0.050000 | spn_mst_dists_sd | 0.000022 | 0.034281 |
| 0.050000 | spn_nng_3_strong_components_varcoeff | 0.000000 | 2.782575 |
| 0.050000 | spn_sum_of_lowest_edge_values | 0.000014 | 0.007817 |
| 0.250000 | spn_distance_min | 0.002466 | 1.378355 |
| 0.250000 | spn_distance_varcoeff | 0.000007 | 0.064760 |
| 0.250000 | spn_mst_dists_sd | 0.000110 | 0.189679 |
| 0.250000 | spn_nng_3_strong_components_varcoeff | 0.000373 | 2.782575 |
| 0.250000 | spn_sum_of_lowest_edge_values | 0.000068 | 0.068528 |
| 0.500000 | spn_distance_min | 0.005681 | 2.325903 |
| 0.500000 | spn_distance_varcoeff | 0.000014 | 0.125489 |
| 0.500000 | spn_mst_dists_sd | 0.000270 | 0.357399 |
| 0.500000 | spn_nng_3_strong_components_varcoeff | 0.008090 | 2.782575 |
| 0.500000 | spn_sum_of_lowest_edge_values | 0.000136 | 0.188362 |

Every row uses all 119,530 candidates, exact HiGHS transport, unchanged 208/209 quotas, and 10,000 unique survivors. Cross-cost columns in `05_selection_stability.csv` re-solve assignment for fixed memberships and therefore separate target movement from membership adaptation. Hash changes and population distribution changes are reported independently. NNG SCC-CV has the largest worst-case pointwise score jumps; `distance_min` is the next most responsive dimension. These maxima are localized—the population-level and membership metrics above are the relevant propagation test.
