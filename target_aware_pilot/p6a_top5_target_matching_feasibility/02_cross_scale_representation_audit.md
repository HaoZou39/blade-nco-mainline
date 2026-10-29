# Cross-Scale Representation Audit

## Evaluated representations

| approach | target_missing | max_abs_uniform_n_rho | max_abs_target_n_rho |
|---|---|---|---|
| coordinate_scale_only | 2 | 0.9857 | 0.6548 |
| raw_F40_definition | 2 | 0.9862 | 0.6152 |
| theoretical_node_and_scale | 2 | 0.6192 | 0.5444 |
| uniform_conditional_tail_score | 0 | 0.0024 | 0.4611 |

The frozen HGB arcsinh transform is retained as provenance but is not a physical scale/node normalization. The selected matching representation is built in two explicit stages:

1. Preserve raw Top5, then form dimensionless companions: `sum_low/(D sqrt(n))`, `n min_distance/D`, raw pairwise-distance CV, raw NNG SCC-size CV, and `sqrt(n) MST_SD/D`, where D is the bounding-box diagonal.
2. Map each companion through an independent Uniform([0,1]^2) conditional empirical CDF score. Within calibration support, unique values are linearly interpolated in logit plotting-position space; outside support, the logit score continues with the frozen Uniform IQR and is compressed by `asinh(score/2)`. Calibration-grid scores are interpolated in log(n). This avoids empirical-CDF endpoint saturation while making five unweighted coordinates commensurate without HGB importance weights.

Calibration contains 2,304 independent instances over 10 n-grid values; coordinate hash `7bbdfc38fdb3ced18a804b465fe7fd5382edda17a2a23ac98982ce4d4d9fbbe7`. It estimates nuisance transforms only. The target empirical distribution remains the 48 transformed TSPLIB points and never calibrates its own axes.

NNG resolution: raw NA is preserved and labelled `single SCC`. In the representation it is an ordered semantic atom below finite cross-component CV values, mapped to the midpoint of that atom's Uniform conditional probability mass before the same score transform. This is not zero imputation and does not delete the feature. Two TSPLIB48 rows use this atom; C04 has none.

The theoretical multipliers describe leading random-geometric scaling, not exact universal laws. Conditional Uniform scores address residual finite-n effects; calibration interpolation, tail continuation, and finite Monte Carlo size remain the main representation uncertainty.

Even/odd split-half calibration sensitivity on the 48 targets is descriptive, not a confidence interval: median absolute score difference 0.0581, p95 0.2102, maximum 0.5272 on `pr1002` / `spn_distance_min`. The full frozen calibration—not either half—defines matching.
