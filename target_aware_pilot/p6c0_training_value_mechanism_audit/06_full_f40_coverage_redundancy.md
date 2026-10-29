# Full F40 Coverage and Structural Redundancy

All 40 frozen HGB40 inputs were computed with the formal salesperson extractor and frozen no-clipping asinh scaler. Raw values are exactly reconstructed as `center + scale*sinh(z)` for marginal summaries. No uncalibrated raw-feature Euclidean distance is used. The F40 NN diagnostic uses the calibrated transformed 40-vector and is explicitly a structural proxy.

| replicate | is_top5 | mean_std_ratio | median_std_ratio | features_compressed_below_0_8 | mean_abs_median_shift |
|---|---|---|---|---|---|
| Original | False | 1.0492196 | 0.98597561 | 2 | 0.39153715 |
| Original | True | 1.433033 | 1.4284961 | 0 | 1.4577941 |
| S1 | False | 1.0553574 | 0.99791591 | 2 | 0.38138853 |
| S1 | True | 1.4595601 | 1.3918808 | 0 | 1.4449154 |
| S2 | False | 1.0545418 | 0.99844667 | 2 | 0.37622044 |
| S2 | True | 1.4306235 | 1.4160042 | 0 | 1.4466867 |

## Population redundancy and lineage

| replicate | arm | unique_coordinates | effective_lineages_shannon | maximum_lineage_share | parent_or_P0_fraction | G2_offspring_fraction | top5_nn_median | f40_nn_median | top5_near_duplicate_fraction_p01 | top5_near_duplicate_fraction_p05 | top5_near_duplicate_fraction_p10 | f40_near_duplicate_fraction_p01 | f40_near_duplicate_fraction_p05 | f40_near_duplicate_fraction_p10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Original | E0 | 10000 | 484.65504 | 0.0163 | 0.0822 | 0.9178 | 0.05614569 | 2.5263364 | 0.0074 | 0.0782 | 0.1706 | 0.0008 | 0.0099 | 0.0265 |
| Original | E1 | 10000 | 225.17802 | 0.0416 | 0.0982 | 0.9018 | 0.19781116 | 2.2062786 | 0.0101 | 0.0215 | 0.0279 | 0.0188 | 0.09 | 0.168 |
| S1 | E0 | 10000 | 487.00643 | 0.0136 | 0.084 | 0.916 | 0.055049585 | 2.5297372 | 0.01 | 0.0769 | 0.1741 | 0.0022 | 0.012 | 0.029 |
| S1 | E1 | 10000 | 215.7916 | 0.0426 | 0.1002 | 0.8998 | 0.19575871 | 2.1762196 | 0.0119 | 0.0221 | 0.0284 | 0.0189 | 0.0882 | 0.179 |
| S2 | E0 | 10000 | 476.48737 | 0.0142 | 0.0814 | 0.9186 | 0.055000972 | 2.5182425 | 0.0103 | 0.0815 | 0.1709 | 0.0028 | 0.0128 | 0.031 |
| S2 | E1 | 10000 | 221.62495 | 0.041 | 0.0985 | 0.9015 | 0.19768315 | 2.2109433 | 0.0104 | 0.0198 | 0.0281 | 0.0166 | 0.0872 | 0.1665 |

Near-duplicate thresholds are the pooled six-population P1/P5/P10 values and all three are shown. Feature similarity is not gradient similarity. `TRAINING_SIGNAL_REDUNDANCY_NOT_DIRECTLY_ESTABLISHED`: no historical per-instance gradients or gradient-direction logs exist.
