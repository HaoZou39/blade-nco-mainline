# TSPLIB48 Target Distribution Audit

Each of the 48 development instances has equal empirical authority. The target is the joint set of 48 five-dimensional transformed points—not five independently resampled marginals. No threshold, 70/30 split, performance weight, TSPLIB22 datum, or BQ gap enters it.

| feature | min | q25 | median | q75 | max |
|---|---|---|---|---|---|
| spn_sum_of_lowest_edge_values | -3.36601 | -2.60113 | -1.86610 | 1.13084 | 2.11444 |
| spn_distance_min | -1.84449 | -0.10388 | 1.32384 | 1.99401 | 3.11570 |
| spn_distance_varcoeff | -1.61291 | 1.94625 | 2.53756 | 3.20387 | 4.38849 |
| spn_nng_3_strong_components_varcoeff | -2.05744 | -1.22349 | -0.18172 | 0.64292 | 1.85142 |
| spn_mst_dists_sd | -3.01099 | -1.90136 | -0.87260 | 2.36860 | 3.27667 |

There are 0 exact duplicate five-dimensional rows. Median per-target nearest distance is 1.1435; median mean-five-nearest distance is 1.6821. Per-instance local geometry and NNG singleton states are in `diagnostics/target_local_geometry.csv`. The small n=48 empirical target is intentionally discrete; capacities implement its empirical masses without fitting a fine grid or KDE.
