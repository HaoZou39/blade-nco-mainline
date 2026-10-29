# M0 vs M4 HGB40 Feature Importance

## Comparability boundary

- M0 audit source: C00 HGB40, conditioned on M0, calibration `all_archive` with 500 rows.
- M4 audit source: C04 HGB40, conditioned on M4, calibration `canonical_sort_uniform_random10k` with 10,000 rows.
- F40 names/order hash is identical: `19c2778c207a82641d6fab4290d8ccc198c63aa4280550b1c54e60cb5b70c8d7`; HGB config hash is also identical: `40d519462c3f7268ebb6deb3d35acbdaf0723fac766e792605ba8e39d0de8b48`.
- The importance algorithm is identical in this audit: eight independent-set permutations, Spearman-drop primary score, 30 row bootstraps, fixed audit seeds, plus the same private persisted-tree split-gain diagnostic.
- The validation datasets are **not the same**: each cycle uses its own surrogate-selected, newly labelled Hard10k. Both are label-held-out but acquisition-shifted. C00 also has only 500 calibration rows whereas C04 has 10,000. Therefore raw drop differences and rank movement are descriptive, not a causal estimate of M0→M4 adaptation.

## M4 Top10 with M0 comparison

| rank m4 | feature | spearman drop m4 | bootstrap top5 frequency m4 | rank m0 | spearman drop m0 | bootstrap top5 frequency m0 | rank change m4 minus m0 |
|---|---|---|---|---|---|---|---|
| 1 | spn_sum_of_lowest_edge_values | 0.365431 | 1.000 | 1 | 0.471162 | 1.000 | +0 |
| 2 | spn_distance_min | 0.056990 | 1.000 | 2 | 0.047508 | 1.000 | +0 |
| 3 | spn_distance_varcoeff | 0.037355 | 1.000 | 35 | -0.005108 | 0.000 | -32 |
| 4 | spn_nng_3_strong_components_varcoeff | 0.020274 | 0.767 | 31 | -0.003832 | 0.000 | -27 |
| 5 | spn_mst_dists_sd | 0.020060 | 0.500 | 16 | 0.000604 | 0.100 | -11 |
| 6 | spn_fraction_shorter_mean_distance | 0.019439 | 0.567 | 4 | 0.015452 | 1.000 | +2 |
| 7 | spn_mst_dists_span | 0.016596 | 0.033 | 3 | 0.024897 | 1.000 | +4 |
| 8 | spn_hull_dists_varcoeff | 0.015525 | 0.067 | 11 | 0.002716 | 0.033 | -3 |
| 9 | spn_mst_depth_mean | 0.014588 | 0.067 | 32 | -0.004116 | 0.000 | -23 |
| 10 | spn_hull_dists_median | 0.013420 | 0.000 | 7 | 0.003507 | 0.100 | +3 |

## M0 Top10 with M4 comparison

| rank m0 | feature | spearman drop m0 | bootstrap top5 frequency m0 | rank m4 | spearman drop m4 | bootstrap top5 frequency m4 | rank change m4 minus m0 |
|---|---|---|---|---|---|---|---|
| 1 | spn_sum_of_lowest_edge_values | 0.471162 | 1.000 | 1 | 0.365431 | 1.000 | +0 |
| 2 | spn_distance_min | 0.047508 | 1.000 | 2 | 0.056990 | 1.000 | +0 |
| 3 | spn_mst_dists_span | 0.024897 | 1.000 | 7 | 0.016596 | 0.033 | +4 |
| 4 | spn_fraction_shorter_mean_distance | 0.015452 | 1.000 | 6 | 0.019439 | 0.567 | +2 |
| 5 | spn_nearest_neighbour_median | 0.005152 | 0.333 | 38 | 0.002407 | 0.000 | +33 |
| 6 | spn_centroid_sd | 0.005098 | 0.333 | 11 | 0.012902 | 0.000 | +5 |
| 7 | spn_hull_dists_median | 0.003507 | 0.100 | 10 | 0.013420 | 0.000 | +3 |
| 8 | spn_hull_edges_mean | 0.003284 | 0.033 | 33 | 0.003899 | 0.000 | +25 |
| 9 | spn_angles_cos_varcoeff | 0.003119 | 0.067 | 40 | 0.001999 | 0.000 | +31 |
| 10 | spn_mst_depth_skew | 0.003008 | 0.000 | 35 | 0.003454 | 0.000 | +25 |

## Consistency result

- Top-5 overlap: **2/5**; Top-10 overlap: **5/10**.
- Across all F40 ranks, Spearman rank correlation is **0.144**.
- Validation skill differs (M0 selected-set rho 0.424, M4 0.413), but the validation distributions also differ.
- Conclusion: importance is **partially consistent, not invariant**. Agreement is summarized by overlap/rank correlation; disagreement cannot be assigned solely to solver adaptation.

## Frozen two-feature ranks

| feature | rank m0 | spearman drop m0 | bootstrap top5 frequency m0 | rank m4 | spearman drop m4 | bootstrap top5 frequency m4 |
|---|---|---|---|---|---|---|
| spn_distance_varcoeff | 35 | -0.005108 | 0.000 | 3 | 0.037355 | 1.000 |
| spn_fraction_shorter_mean_distance | 4 | 0.015452 | 1.000 | 6 | 0.019439 | 0.567 |

The old structural pair was not Top1/Top2. P0 first preserved the full M4 ranking, then applied a cross-size matching criterion: finite on all TSPLIB48, dimensionless under uniform scaling, interpretable, and nonredundant. It chose M4 ranks 3 and 6. This was a defensible *matching-space* decision, but it must not be reinterpreted as HGB40 itself selecting those exact two features.

The legacy GP4/QD4 feature set is not used as M0 importance evidence.
