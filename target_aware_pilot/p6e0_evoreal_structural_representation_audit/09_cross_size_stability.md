# Cross-Size Stability

TSPLIB48 FFT-vs-N Spearman `0.7972`; NN-CV-vs-N `0.1566`. Fixed subsampling results below show that the raw-count FFT scale and threshold classification are not fully stable across N. Therefore TSP100 and broad-size TSPLIB values cannot be treated as directly interchangeable. A predeclared broad comparable band (75<=N<=150) contains only `17` references and is group-imbalanced; its results are diagnostic only. The apparent raw target gap cannot be cleanly separated from size confounding.

| record | dataset | feature | parameter | value | n |
|---|---|---|---|---|---|
| reference_correlation | TSPLIB48 | fft_energy | N | 0.797240 | 48 |
| reference_correlation | TSPLIB48 | fft_energy | logN | 0.797240 | 48 |
| reference_correlation | TSPLIB48 | nn_ratio | N | 0.156601 | 48 |
| reference_correlation | TSPLIB48 | nn_ratio | logN | 0.156601 | 48 |
| size_bin | TSPLIB48 | fft_energy | <100 | 4.398719 | 2 |
| size_bin | TSPLIB48 | nn_ratio | <100 | 0.255503 | 2 |
| size_bin | TSPLIB48 | fft_energy | 100-199 | 11.401573 | 20 |
| size_bin | TSPLIB48 | nn_ratio | 100-199 | 0.558677 | 20 |
| size_bin | TSPLIB48 | fft_energy | 200-499 | 24.915548 | 15 |
| size_bin | TSPLIB48 | nn_ratio | 200-499 | 0.533648 | 15 |
| size_bin | TSPLIB48 | fft_energy | 500-999 | 40.100684 | 6 |
| size_bin | TSPLIB48 | nn_ratio | 500-999 | 0.522501 | 6 |
| size_bin | TSPLIB48 | fft_energy | >=1000 | 103.106321 | 5 |
| size_bin | TSPLIB48 | nn_ratio | >=1000 | 0.719265 | 5 |
| subsample_stability | EvoReal_subsample50 | fft_energy | median_subsample_over_original | 0.588460 | 300 |
| subsample_stability | EvoReal_subsample50 | nn_ratio | median_abs_change | 0.110763 | 300 |
| subsample_stability | EvoReal_subsample50 | approx_type | classification_agreement | 0.690000 | 300 |
| subsample_stability | EvoReal_subsample75 | fft_energy | median_subsample_over_original | 0.804848 | 300 |
| subsample_stability | EvoReal_subsample75 | nn_ratio | median_abs_change | 0.049180 | 300 |
| subsample_stability | EvoReal_subsample75 | approx_type | classification_agreement | 0.810000 | 300 |
| subsample_stability | TSPLIB48_subsample100 | fft_energy | median_subsample_over_original | 0.594459 | 460 |
| subsample_stability | TSPLIB48_subsample100 | nn_ratio | median_abs_change | 0.065454 | 460 |
| subsample_stability | TSPLIB48_subsample100 | approx_type | classification_agreement | 0.628261 | 460 |
| comparable_size_summary | TSPLIB48 | reference_count | 75<=N<=150 | 17.000000 | 17 |

## Comparable-size diagnostic

| dataset | reference_band | reference_n | population_n | reference_neighborhood_coverage_fraction | population_within_band_reference_radius_fraction | reference_to_population_nearest_median | sliced_wasserstein |
|---|---|---|---|---|---|---|---|
| B_Original | 75<=N<=150 | 17 | 10000 | 0.882353 | 0.332500 | 0.004156 | 0.852755 |
| C_Original | 75<=N<=150 | 17 | 10000 | 0.882353 | 0.321400 | 0.003718 | 0.865452 |
| B_S1 | 75<=N<=150 | 17 | 10000 | 0.823529 | 0.339900 | 0.005447 | 0.851539 |
| C_S1 | 75<=N<=150 | 17 | 10000 | 0.823529 | 0.331300 | 0.005448 | 0.863142 |
| B_S2 | 75<=N<=150 | 17 | 10000 | 0.823529 | 0.334500 | 0.004419 | 0.855116 |
| C_S2 | 75<=N<=150 | 17 | 10000 | 0.882353 | 0.329100 | 0.004585 | 0.861138 |
| EvoReal50k | 75<=N<=150 | 17 | 50000 | 1.000000 | 0.362680 | 0.002164 | 0.712618 |
