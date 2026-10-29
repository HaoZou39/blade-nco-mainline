# EvoReal Generator Breakdown

Each generator is evaluated with ten fixed, uniform, no-replacement 10k subsamples (the common supported size; no replacement).

| dataset | repeats | ot_mean_cost_mean | ot_mean_cost_sd | sliced_wasserstein_mean | nearest_mean_mean | nearest_reference_effective_mean |
|---|---|---|---|---|---|---|
| EvoReal_S1 | 10 | 20.264687 | 0.087350 | 1.396709 | 1.844004 | 5.188941 |
| EvoReal_S2 | 10 | 11.581284 | 0.047084 | 0.887126 | 1.832770 | 15.503777 |
| EvoReal_S3 | 10 | 7.985711 | 0.022838 | 0.827738 | 1.680032 | 13.124796 |

Transformed Top5 summaries (full generator groups; raw counterparts are preserved in `cache/evoreal_generator_raw_transformed_summary.csv`):

| generator | feature | mean | median | q25 | q75 |
|---|---|---|---|---|---|
| S1 | spn_sum_of_lowest_edge_values | -2.449307 | -2.658282 | -2.745063 | -2.382092 |
| S1 | spn_distance_min | -0.640428 | -1.125618 | -1.956709 | 0.249968 |
| S1 | spn_distance_varcoeff | 3.125445 | 3.126229 | 2.759763 | 3.505381 |
| S1 | spn_nng_3_strong_components_varcoeff | -0.641434 | -0.588920 | -1.362629 | 0.088305 |
| S1 | spn_mst_dists_sd | 2.518082 | 2.843314 | 2.459563 | 3.005682 |
| S2 | spn_sum_of_lowest_edge_values | -0.132827 | -0.301020 | -2.106326 | 1.838027 |
| S2 | spn_distance_min | 0.716929 | 0.687272 | -0.601269 | 2.055749 |
| S2 | spn_distance_varcoeff | 0.496079 | 0.328202 | -0.866652 | 2.076432 |
| S2 | spn_nng_3_strong_components_varcoeff | -0.828533 | -1.263736 | -1.956400 | 0.278008 |
| S2 | spn_mst_dists_sd | -0.095438 | -0.353671 | -2.037936 | 1.951593 |
| S3 | spn_sum_of_lowest_edge_values | -1.477620 | -2.100693 | -2.306791 | -0.970494 |
| S3 | spn_distance_min | -0.389367 | -0.445281 | -0.902352 | 0.069006 |
| S3 | spn_distance_varcoeff | 1.346219 | 1.827845 | 0.471695 | 2.347259 |
| S3 | spn_nng_3_strong_components_varcoeff | -0.071537 | -0.078915 | -0.674524 | 0.458320 |
| S3 | spn_mst_dists_sd | 0.603864 | 0.711934 | -0.344738 | 1.722270 |

Pairwise structural overlap diagnostics:

| generator_a | generator_b | top5_swd | nearest_reference_js | shared_references |
|---|---|---|---|---|
| S1 | S2 | 1.730457 | 0.519013 | 33 |
| S1 | S3 | 1.094201 | 0.459859 | 26 |
| S2 | S3 | 0.904331 | 0.322593 | 31 |

The generators share only 26–33 of 48 nearest-reference supports pairwise and have large assignment-mass JS values, so they cover materially different, partly overlapping structural regions. PCA is visualization only.
