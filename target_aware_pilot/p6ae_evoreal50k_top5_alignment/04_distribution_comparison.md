# Equal-size Distribution Comparison

Lower is better for all distance columns. EvoReal summary retains all 20 predeclared subsamples; fixed datasets have one complete-set observation. Exact OT measures global mass matching, while nearest-reference distances measure local proximity only.

| dataset | repeats | ot_mean_cost_mean | ot_mean_cost_sd | sliced_wasserstein_mean | nearest_mean_mean | nearest_q95_mean | old2_swd_mean | old2_coarse_js_mean |
|---|---|---|---|---|---|---|---|---|
| EvoReal50k | 20 | 8.689244 | 0.082434 | 0.796259 | 1.783936 | 2.702028 | 0.572843 | 0.122457 |
| Uniform_D0_10k | 1 | 13.518932 | 0.000000 | 1.163749 | 1.468647 | 2.088874 | 0.724125 | 0.473287 |
| P03_Population10k | 1 | 26.610826 | 0.000000 | 1.733078 | 2.484621 | 2.767002 | 0.689199 | 0.543551 |
| P5_S1_E1_ArchiveStructure10k | 1 | 21.257236 | 0.000000 | 1.537516 | 2.362092 | 2.816769 | 0.239681 | 0.065106 |
| P5_S1_E1_G2_Population10k | 1 | 22.384872 | 0.000000 | 1.583299 | 2.383103 | 2.790738 | 0.321713 | 0.177498 |
| P5_S2_E1_ArchiveStructure10k | 1 | 21.239718 | 0.000000 | 1.537917 | 2.358574 | 2.812851 | 0.242425 | 0.068075 |
| P5_S2_E1_G2_Population10k | 1 | 22.363505 | 0.000000 | 1.583710 | 2.381329 | 2.792498 | 0.316128 | 0.178785 |
| P5_Original_E1_ArchiveStructure10k | 1 | 21.082490 | 0.000000 | 1.531629 | 2.353414 | 2.813123 | 0.239921 | 0.069735 |
| P5_Original_E1_G2_Population10k | 1 | 22.175368 | 0.000000 | 1.576463 | 2.375783 | 2.792226 | 0.312119 | 0.179518 |

EvoReal mean OT 8.689244 versus Uniform 13.518932, P03 26.610826, and the mean of all three P5 E1 archive replicates 21.193148. No 10k subset was optimized.
