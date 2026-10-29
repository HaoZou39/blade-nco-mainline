# Representation Comparison

Frozen QD4 clipping is reproduced: `70.8%` of TSPLIB48 touches at least one boundary. This is a map-normalization issue, not proof that FFT–NN is superior. Highest absolute Full-F40/Top5 correlations are shown below. NN-CV overlaps local nearest-neighbor statistics. FFT_APPROX is only partly complementary: its correlations with the frozen Top5 are modest on TSPLIB48, but its maximum Full-F40 correlation is high and substantially reflects node-size/distance-scale confounding.

| scope | target_feature | comparison_feature | n | spearman |
|---|---|---|---|---|
| TSPLIB48 | fft_energy | spn_mst_dists_median | 48 | -0.927703 |
| C_Original | fft_energy | spn_sum_of_lowest_edge_values | 10000 | -0.902003 |
| C_S1 | fft_energy | spn_sum_of_lowest_edge_values | 10000 | -0.899073 |
| B_Original | fft_energy | spn_sum_of_lowest_edge_values | 10000 | -0.898145 |
| B_S1 | fft_energy | spn_sum_of_lowest_edge_values | 10000 | -0.896631 |
| C_S2 | fft_energy | spn_sum_of_lowest_edge_values | 10000 | -0.893840 |
| B_S2 | fft_energy | spn_sum_of_lowest_edge_values | 10000 | -0.890363 |
| B_S2 | nn_ratio | spn_nearest_neighbour_median | 10000 | -0.887553 |
| B_Original | nn_ratio | spn_nearest_neighbour_median | 10000 | -0.886934 |
| B_S1 | nn_ratio | spn_nearest_neighbour_median | 10000 | -0.884745 |
| C_S2 | nn_ratio | spn_nearest_neighbour_median | 10000 | -0.880754 |
| C_S1 | nn_ratio | spn_nearest_neighbour_median | 10000 | -0.880207 |
| C_Original | nn_ratio | spn_nearest_neighbour_median | 10000 | -0.879813 |
| TSPLIB48 | nn_ratio | phi__spn_sum_of_lowest_edge_values | 48 | -0.761507 |
