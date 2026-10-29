# Top5 Feature Definition Audit

The installed `salesperson` 1.0.0 implementation called by the formal extractor was inspected and a targeted extractor was numerically cross-checked against persisted C04 F40 raw values (maximum discrepancy at floating precision). Coordinates enter as ordinary Euclidean doubles; TSPLIB `EUC_2D` rounded edge weights and tours are not used.

| rank | feature | M4 permutation rho drop | mathematical definition | implementation | X->aX | node effect | undefined/degenerate |
|---|---|---|---|---|---|---|---|
| 1 | spn_sum_of_lowest_edge_values | 0.3654310687 | sum of the n smallest values in {d_ij:i<j} | Euclidean double, not TSPLIB EUC_2D rounding | degree 1 | depends on n order statistics | finite n>=2; duplicates can create zeros |
| 2 | spn_distance_min | 0.0569902381 | min_{i<j} d_ij | Euclidean double | degree 1 | extreme over n(n-1)/2 pairs | zero with duplicate nodes |
| 3 | spn_distance_varcoeff | 0.0373547894 | sampleSD({d_ij}) / mean({d_ij}) | R sample SD | invariant | finite-sample dependence | undefined if mean=0 |
| 4 | spn_nng_3_strong_components_varcoeff | 0.0202735079 | sampleSD({|C_l|}) / mean({|C_l|}), directed 3-NNG strong SCCs | cccd::nng kd-tree; igraph strong components | invariant except ties | k fixed; SCC law changes with n | R sd undefined for exactly one SCC |
| 5 | spn_mst_dists_sd | 0.0200599365 | sampleSD of n-1 Prim MST edge lengths | salesperson spantree; R sample SD | degree 1 | edge scale and sample size depend on n | undefined for <2 MST edges |

All five are invariant to node permutation for generic coordinates. MST/3-NNG tie resolution can make exact-tie cases implementation-order sensitive; continuous C04 coordinates avoid this, while the audited implementation/version remains frozen.

Direct scale/permutation fixtures: maximum theoretical absolute difference `3.592e-14`, maximum final-phi difference `2.086e-13`. Full diagnostics are in `diagnostics/scale_permutation_invariance_tests.csv`.

Cost: pairwise distance materialization and the MST are at least quadratic in n in this implementation; the 3-NNG uses a kd-tree. At TSP100 the frozen historical full-F40 extraction is the relevant measured budget.
