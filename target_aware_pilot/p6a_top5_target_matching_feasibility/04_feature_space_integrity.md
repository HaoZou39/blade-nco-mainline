# Feature-Space Integrity

- All final inputs finite: target `True`, candidates `True`.
- Finite score ranges: target [-3.3660134333603384, 4.388488041807082]; candidate [-2.857934317263118, 4.0689411987560415].
- Exact duplicated rows: target 0, candidate 0.
- Effective correlation dimension: target 3.459/5; candidates 2.839/5.
- Maximum absolute pairwise Spearman: target 0.732; candidates 0.746.
- Target nearest-pair distance: min 0.534419, median per-target nearest 1.143523.
- Nearest-target squared-distance contribution shares: `{"spn_distance_min": 0.8372764595797831, "spn_distance_varcoeff": 0.052660712445141124, "spn_mst_dists_sd": 0.02579206342858953, "spn_nng_3_strong_components_varcoeff": 0.07094495574167159, "spn_sum_of_lowest_edge_values": 0.013325808804827458}`. No HGB-importance weighting is applied.

The full Top5 is therefore usable as a complete finite representation; no feature was deleted and there is no fallback to Top2. Correlation and effective dimension show redundancy but not a degenerate space.
