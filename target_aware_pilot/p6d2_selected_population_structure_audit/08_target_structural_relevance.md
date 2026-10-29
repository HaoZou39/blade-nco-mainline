# Target Structural Relevance

Exact Top5 OT/SWD/energy are recomputed with the frozen P6-A representation and equal TSPLIB48 reference quotas. Original70 proximity uses aspect-preserving coordinates, the formal F40 scaler and only the 28 F40 fields finite for all 70 instances; no missing value is imputed, and exclusions are recorded in `manifests/original70_f40_missingness.csv`. Raw size-dependent coordinates are not compared. TSPLIB22 is descriptive only. QD4 remains interpretation-limited because 70.8% of TSPLIB48 references clip on at least one axis and only 29 niches are directly occupied.

QD4 clip audit:

| feature | spearman_with_n_nodes | clip_zero_fraction | clip_one_fraction | transformed_min | transformed_max |
|---|---|---|---|---|---|
| spn_sum_of_lowest_edge_values | 0.531258 | 0 | 0.4791667 | -0.5901225 | 2.67231 |
| spn_distance_varcoeff | 0.0496645 | 0.04166667 | 0.02083333 | -0.9369803 | 2.771764 |
| spn_mst_depth_mean | 0.8273975 | 0.04166667 | 0.375 | -1.358191 | 3.420494 |
| spn_mst_dists_median | -0.7615404 | 0.2916667 | 0.0625 | -1.56943 | 2.060632 |

Population target metrics:

| record_type | replicate | arm | reference | top5_ot_total | top5_ot_mean | top5_swd | top5_energy | nearest_reference_mean | d0_top5_ot_mean | ot_reproduced | assigned_n | nearest_candidate_distance | assigned_cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| population | Original | A | ALL | 227049.3 | 22.70493 | 1.590281 | 3.95096 | 2.351117 | 22.70493 | True | NA | NA | NA |
| population | Original | B | ALL | 144779.9 | 14.47799 | 1.226971 | 2.170935 | 1.991298 | 14.47799 | True | NA | NA | NA |
| population | Original | C | ALL | 147945.4 | 14.79454 | 1.240011 | 2.237638 | 2.009132 | 14.79454 | True | NA | NA | NA |
| population | S1 | A | ALL | 227988.4 | 22.79884 | 1.59261 | 3.876302 | 2.349764 | 22.79884 | True | NA | NA | NA |
| population | S1 | B | ALL | 146000 | 14.6 | 1.234385 | 2.208423 | 2.000473 | 14.6 | True | NA | NA | NA |
| population | S1 | C | ALL | 148923.4 | 14.89234 | 1.24626 | 2.280887 | 2.016465 | 14.89234 | True | NA | NA | NA |
| population | S2 | A | ALL | 229390.8 | 22.93908 | 1.597184 | 3.891862 | 2.350137 | 22.93908 | True | NA | NA | NA |
| population | S2 | B | ALL | 145772.3 | 14.57723 | 1.231915 | 2.110259 | 1.996717 | 14.57723 | True | NA | NA | NA |
| population | S2 | C | ALL | 148134.7 | 14.81347 | 1.241117 | 2.163683 | 2.009451 | 14.81347 | True | NA | NA | NA |

F40 proximity summary:

| replicate | arm | split | f40_nn1 | f40_nn5_mean |
|---|---|---|---|---|
| Original | B | TSPLIB22 | 9.607141 | 9.781365 |
| Original | B | TSPLIB48 | 6.284937 | 6.525109 |
| Original | C | TSPLIB22 | 9.600317 | 9.789642 |
| Original | C | TSPLIB48 | 6.337715 | 6.583768 |
| S1 | B | TSPLIB22 | 9.652861 | 9.877221 |
| S1 | B | TSPLIB48 | 6.309456 | 6.569712 |
| S1 | C | TSPLIB22 | 9.588303 | 9.82271 |
| S1 | C | TSPLIB48 | 6.324428 | 6.570199 |
| S2 | B | TSPLIB22 | 9.516532 | 9.782231 |
| S2 | B | TSPLIB48 | 6.336103 | 6.5809 |
| S2 | C | TSPLIB22 | 9.487686 | 9.770091 |
| S2 | C | TSPLIB48 | 6.246198 | 6.550804 |
