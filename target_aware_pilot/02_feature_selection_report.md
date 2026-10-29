# P0 Feature Selection Report

## Surrogate validity

The audited C04 HGB40 is correctly conditioned on M4. On the independently labelled, acquisition-shifted C04 Hard10k validation set it achieves:

| Metric | Value |
|---|---:|
| Spearman(predicted, observed `log1p_LBQ`) | 0.4127 |
| MAE in `log1p_LBQ` | 0.03248 |
| Observed top-decile recall from predicted top decile | 0.307 |
| Mean-hardness enrichment, predicted top decile vs all | 1.353x |

This is useful but moderate ranking signal. For comparison only, the in-sample calibration Spearman is 0.8921 and MAE is 0.01283; those training-fit values are not used as generalization evidence.

Scikit-learn's `HistGradientBoostingRegressor` does not expose a public `feature_importances_`. Therefore `01_m4_hgb40_feature_importance.csv` reports this explicitly and includes two auditable alternatives:

1. repeated permutation importance on the independent Hard10k validation labels; and
2. a clearly labelled internal split-gain/count proxy extracted from the persisted trees.

The latter is diagnostic rather than a public model importance API.

## Ranking and stability

The leading permutation ranks are:

| Rank | Feature | Spearman drop | Bootstrap Top-5 frequency | Comparability note |
|---:|---|---:|---:|---|
| 1 | `spn_sum_of_lowest_edge_values` | 0.36543 | 1.000 | raw distance sum; strongly scale/node dependent |
| 2 | `spn_distance_min` | 0.05699 | 1.000 | raw distance; scale/node dependent |
| 3 | `spn_distance_varcoeff` | 0.03735 | 1.000 | dimensionless and finite on all 48 references |
| 4 | `spn_nng_3_strong_components_varcoeff` | 0.02027 | 0.767 | undefined on 2/48 references and normalization-sensitive |
| 5 | `spn_mst_dists_sd` | 0.02006 | 0.500 | raw distance dispersion; scale and size dependent |
| 6 | `spn_fraction_shorter_mean_distance` | 0.01944 | 0.567 | dimensionless and finite on all 48 references |

The top-ranked raw-distance features cannot be compared fairly across TSPLIB sizes and coordinate conventions. They remain in the audit table but are rejected for target-density matching.

## Selected features

### 1. `spn_distance_varcoeff`

- Permutation rank 3 and bootstrap Top-5 frequency 1.0.
- Coefficient of variation of all pairwise distances: a global, dimensionless description of distance dispersion and spatial shape.
- Exactly invariant to uniform coordinate scaling in the audit and finite on all TSPLIB48 instances.
- Spearman correlation with `log(n)` on TSPLIB48 is 0.0497, providing no evidence that node count drives the observed target distribution.

### 2. `spn_fraction_shorter_mean_distance`

- Permutation rank 6 and bootstrap Top-5 frequency 0.567.
- Fraction of pairwise distances below their mean: a dimensionless summary of distance-distribution balance/skew and spatial heterogeneity.
- Exactly invariant to uniform coordinate scaling and finite on all TSPLIB48 instances.
- Spearman correlation with `log(n)` on TSPLIB48 is -0.0951.
- Its correlation with `spn_distance_varcoeff` in the candidate pool is only 0.1225, so it adds complementary information.

## Why permutation rank 6 was selected

The choice was determined before interpreting target coverage and was not changed to make TSPLIB easier to match. Rank 4 is undefined for `eil51` and `rat195` after the formal aspect-preserving normalization and showed unexpected normalization sensitivity. Rank 5 is a scale-dependent MST standard deviation and has a TSPLIB48 Spearman correlation of -0.615 with `log(n)`. Rank 6 is the nearest stable, finite, dimensionless, and nonredundant alternative.

## Correlation and normalization checks

- The raw Top-5 Spearman correlation matrix is provided in `p0_top5_spearman_correlation.csv`. The top two raw-scale features correlate at 0.6503, reinforcing that simply taking ranks 1 and 2 would be redundant as well as non-comparable.
- The two selected features are unchanged by native-coordinate vs aspect-preserving normalized-coordinate extraction up to numerical precision (maximum absolute differences approximately `2.3e-14` and `0`).
- Per-axis min-max normalization is not the formal pipeline and changes geometry. It produces median absolute changes of 0.534 and 0.623 in transformed feature space, so it is shown only as a sensitivity warning and is not used for matching.

## P0 conclusion

The M4-conditioned HGB40 has sufficient, though not strong, held-label ranking skill to support an exploratory structural audit. Two comparable and interpretable features can be selected without using TSPLIB held-out data. P0 therefore proceeds to P1 with the limitations above recorded explicitly.
