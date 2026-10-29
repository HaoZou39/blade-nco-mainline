# P6-G1 Generation Dynamics

## Observed trajectory

| dataset | n | coverage_48 | coverage_17 | median_population_to_reference_distance | pr107_nearest_distance | pr144_nearest_distance |
|---|---|---|---|---|---|---|
| P03 | 10000 | 23 | 14 | 0.62635 | 1.0122 | 0.36576 |
| Original_G1_RAW | 109571 | 25 | 15 | 0.60173 | 0.9552 | 0.19804 |
| Original_E0_G2_RAW | 109603 | 24 | 14 | 0.58223 | 1.0016 | 0.31945 |
| Original_E1_G2_RAW | 109778 | 25 | 15 | 0.34758 | 0.89307 | 0.18758 |
| S1_G1_RAW | 109541 | 24 | 14 | 0.60138 | 0.98784 | 0.28857 |
| S1_E0_G2_RAW | 109574 | 22 | 14 | 0.58535 | 1.126 | 0.31991 |
| S1_E1_G2_RAW | 109751 | 24 | 14 | 0.35086 | 0.97856 | 0.27106 |
| S2_G1_RAW | 109570 | 25 | 15 | 0.60107 | 0.92773 | 0.29836 |
| S2_E0_G2_RAW | 109619 | 24 | 14 | 0.58252 | 1.1642 | 0.30596 |
| S2_E1_G2_RAW | 109762 | 25 | 15 | 0.35352 | 0.9182 | 0.19445 |

P03 starts at 23/48 neighborhoods. Full G1 RAW reaches only 24–25/48 and full G2 RAW 22–25/48 across the three historical seeds/branches. The 75–150 control remains 14/17 at P03, at most 15/17 for historical RAW, versus 17/17 for EvoReal. Thus the full reference gap is not solely a sample-count comparison or solely caused by very large TSPLIB sizes.

G1 survivor selection changes the parent distribution supplied to G2. E0 and E1 therefore have different G2 RAW distributions even with the same operator family; historical observations cannot separate parent-distribution effects from branch random streams causally. E0/E1 union nevertheless retains every neighborhood ever reached in these histories, so the persistent 15-reference gap remains a generation/reachability observation.

The E1 target-aware survivor branch consistently shifts density toward the reference support: across seeds, G2-RAW median population-to-reference distance is about 0.35 for E1 versus 0.58 for E0, and formal-F40 Wasserstein/IQR is about 0.395 versus 0.477. This is a strong selection association, but it does not create a new globally reachable neighborhood beyond the historical RAW union and therefore does not overturn the primary generation bottleneck.

## Exact parent→offspring movement

| replicate | arm | generation | n | f40_delta_median | f40_delta_p90 | f40_delta_p99 | fft_nn_delta_median | fft_nn_delta_p90 | nearest_ref_improved_fraction | new_neighborhood_instance_n | pr107_improved_fraction | pr144_improved_fraction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Original | SHARED | 1 | 109571 | 0.42434 | 0.74332 | 1.0963 | 0.17631 | 0.59914 | 0.52546 | 5982 | 0.4628 | 0.46181 |
| Original | E0 | 2 | 109603 | 0.41968 | 0.73942 | 1.097 | 0.1785 | 0.60726 | 0.52021 | 6393 | 0.47066 | 0.46962 |
| Original | E1 | 2 | 109778 | 0.4624 | 0.85095 | 1.2389 | 0.1735 | 0.6408 | 0.4431 | 6705 | 0.39555 | 0.39608 |
| S1 | SHARED | 1 | 109541 | 0.42295 | 0.74314 | 1.0836 | 0.17572 | 0.59542 | 0.52911 | 6002 | 0.46254 | 0.46159 |
| S1 | E0 | 2 | 109574 | 0.41835 | 0.73865 | 1.0974 | 0.17827 | 0.6067 | 0.52093 | 6336 | 0.47205 | 0.47095 |
| S1 | E1 | 2 | 109751 | 0.46065 | 0.85134 | 1.2511 | 0.17178 | 0.6397 | 0.44546 | 6867 | 0.39613 | 0.39683 |
| S2 | SHARED | 1 | 109570 | 0.42448 | 0.74507 | 1.0935 | 0.17756 | 0.60593 | 0.52574 | 6024 | 0.46055 | 0.45951 |
| S2 | E0 | 2 | 109619 | 0.41751 | 0.73508 | 1.089 | 0.17886 | 0.60808 | 0.5165 | 6407 | 0.47091 | 0.46964 |
| S2 | E1 | 2 | 109762 | 0.46095 | 0.8507 | 1.2511 | 0.17227 | 0.63693 | 0.44736 | 6993 | 0.39891 | 0.39907 |

The distributions include a dominant small-movement body and operator-dependent long tails. A positive move toward a reference is common, but a positive one-step change does not imply entry into its frozen neighborhood.

## Operator behavior (descriptive, not causal)

| operator_id | calls | accepted | f40_median | f40_p95 | fft_median | improved | new_neighborhood | pr107_best | pr107_min_child | pr144_best | pr144_min_child |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear_projection_mutation | 90000 | 88033 | 0.60712 | 0.94655 | 0.28855 | 0.47189 | 5051 | 2.9125 | 1.289 | 2.8935 | 0.35625 |
| axis_projection_mutation | 90000 | 90000 | 0.5965 | 0.93613 | 0.29852 | 0.46987 | 5364 | 3.0093 | 1.2682 | 3.0049 | 0.29144 |
| expansion_mutation | 90000 | 89030 | 0.54866 | 0.94169 | 0.21297 | 0.46221 | 4054 | 2.156 | 1.1374 | 2.1513 | 0.25943 |
| rotation_mutation | 90000 | 90000 | 0.5169 | 0.87014 | 0.2058 | 0.50579 | 5086 | 2.5411 | 0.92762 | 2.5393 | 0.3066 |
| cluster_mutation | 90000 | 90000 | 0.51575 | 0.87414 | 0.29333 | 0.49207 | 5638 | 3.312 | 1.1588 | 3.312 | 0.29605 |
| grid_mutation | 90000 | 90000 | 0.45991 | 1.0428 | 0.23501 | 0.61701 | 16460 | 3.0505 | 0.89307 | 2.9807 | 0.21226 |
| uniform_mutation | 90000 | 90000 | 0.36886 | 0.76316 | 0.14317 | 0.50816 | 4133 | 2.2718 | 0.97479 | 2.2713 | 0.27031 |
| compression_mutation | 90000 | 89902 | 0.34888 | 0.7573 | 0.16286 | 0.47982 | 3799 | 2.2944 | 1.0259 | 2.2873 | 0.26727 |
| normal_mutation | 90000 | 90000 | 0.29003 | 0.68834 | 0.058766 | 0.5026 | 2406 | 1.9912 | 0.97775 | 1.9824 | 0.21025 |
| explosion_mutation | 90000 | 89804 | 0.28155 | 0.7096 | 0.082103 | 0.48041 | 2682 | 2.3786 | 1.0494 | 2.3751 | 0.22122 |
| implosion_mutation | 90000 | 90000 | 0.19782 | 0.55964 | 0.11231 | 0.47742 | 3036 | 1.4269 | 0.94774 | 1.4253 | 0.18758 |

Operators acted on different historical parents and sometimes underwent repair. These statistics identify movement scale and observed contributions, not an operator causal ranking.

## Fixed-size robustness

For every eligible population, 10k and 50k unique samples were drawn with 30 fixed deterministic repeats. `tables/fixed_sample_robustness.csv` reports mean absolute SMD, quantile-approximated marginal Wasserstein/IQR, and frozen FFT–NN coverage. Full RAW remains the historical reachability upper bound; matched samples are the fair density comparison.

| dataset | sample_n | coverage_48_mean | coverage_48_p025 | coverage_48_p975 | coverage_17_mean | mean_abs_smd_mean | mean_quantile_wasserstein_over_evoreal_iqr_mean |
|---|---|---|---|---|---|---|---|
| Original_E0_G2_RAW | 10000 | 22.433 | 21 | 24 | 14 | 0.47851 | 0.88085 |
| Original_E0_G2_RAW | 50000 | 23.7 | 22.725 | 24 | 14 | 0.47774 | 0.89084 |
| Original_E1_G2_RAW | 10000 | 24.1 | 23 | 25 | 14.267 | 0.18778 | 0.63275 |
| Original_E1_G2_RAW | 50000 | 24.733 | 24 | 25 | 14.733 | 0.18694 | 0.63885 |
| Original_G1_RAW | 10000 | 22.933 | 22 | 24.275 | 14.2 | 0.53107 | 0.9013 |
| Original_G1_RAW | 50000 | 24.3 | 23 | 25 | 14.4 | 0.52305 | 0.9065 |
| P03 | 10000 | 23 | 23 | 23 | 14 | 0.58192 | 0.92171 |
| S1_E0_G2_RAW | 10000 | 21.933 | 21 | 22 | 14 | 0.48178 | 0.88481 |
| S1_E0_G2_RAW | 50000 | 22 | 22 | 22 | 14 | 0.48248 | 0.89213 |
| S1_E1_G2_RAW | 10000 | 23.6 | 23 | 24 | 14 | 0.18582 | 0.63902 |
| S1_E1_G2_RAW | 50000 | 24 | 24 | 24 | 14 | 0.18597 | 0.6422 |
| S1_G1_RAW | 10000 | 22.633 | 22 | 24 | 14 | 0.52249 | 0.90044 |
| S1_G1_RAW | 50000 | 23.533 | 23 | 24 | 14 | 0.52341 | 0.90708 |
| S2_E0_G2_RAW | 10000 | 22.233 | 21.725 | 24 | 14 | 0.47081 | 0.88729 |
| S2_E0_G2_RAW | 50000 | 23.333 | 22 | 24 | 14 | 0.47112 | 0.88806 |
| S2_E1_G2_RAW | 10000 | 23.933 | 22 | 25 | 14.4 | 0.18414 | 0.6403 |
| S2_E1_G2_RAW | 50000 | 24.867 | 24 | 25 | 14.867 | 0.18311 | 0.64349 |
| S2_G1_RAW | 10000 | 22.633 | 22 | 24 | 14.1 | 0.52683 | 0.90485 |
| S2_G1_RAW | 50000 | 23.9 | 22.725 | 25 | 14.5 | 0.52427 | 0.90583 |
