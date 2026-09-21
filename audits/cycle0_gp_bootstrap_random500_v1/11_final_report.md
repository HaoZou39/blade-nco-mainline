# Cycle-0 GP bootstrap audit: Old-Centroid435 vs Random500

## 1. Scientific question
Can one uniform Random500 from frozen O0 retain at least 90% of historical centroid435 hard-region acquisition enrichment?

## 2. Frozen protocol
M0, N=100, O0=109,785, QD4/CVT/P1-P99, canonical GP4 2000x60, seeds 2026-2030, teacher-forced H0, and five-formula median were frozen. No external benchmark, mutation, BQ update, GP retuning, or formal-run write occurred.

## 3. Exact arms
- OLD_CENTROID435: one centroid-nearest member per reached niche; exact historical ordered-manifest match.
- RANDOM500: coordinate-hash sorted O0 plus the first 500 indices of one `default_rng(2026).permutation`; no stratification or reroll.

## 4. Shared P0
Canonical P0 began with 1267; Old overlap excluded=0, Random overlap excluded=5, leaving 1262 identical evaluation instances and zero leakage.

## 5. Primary and secondary results
| Metric | Old-Centroid435 | Random500 | Random / Old |
|---|---|---|---|
| Calibration N | 435.000000 | 500.000000 | 1.149425 |
| ENR5 | 1.169212 | 1.087214 | 0.929869 |
| ENR10 (PRIMARY) | 1.172575 | 1.102534 | 0.940267 |
| ENR20 | 1.128254 | 1.127279 | 0.999136 |
| P0 Spearman | 0.119007 | 0.090948 | 0.764221 |
| Represented niches | 435.000000 | 112.000000 | 0.257471 |
| GP runtime (s) | 129.154813 | 123.621940 | 0.957161 |

ENR10 is **PRIMARY**. `R_ENR=0.940267`, `Delta_ENR10=-0.070041`. Random500 retains **94.03%** against the preset 90% gate. ENR5/ENR20, tail H0, prediction distributions, and diagnostic P0 Spearman are in `08_secondary_metrics.csv`; they do not gate the verdict.

## 6. Calibration structure
| arm | calibration_n | represented_niches | effective_niche_number | niche_entropy_nats | operator_coverage | H0_mean | H0_std | H0_median | H0_q95 |
|---|---|---|---|---|---|---|---|---|---|
| OLD_CENTROID435 | 435 | 435 | 435.000000 | 6.075346 | 11 | 0.068171 | 0.054103 | 0.056410 | 0.175307 |
| RANDOM500 | 500 | 112 | 71.985980 | 4.276471 | 11 | 0.046107 | 0.031482 | 0.039562 | 0.098511 |

Structure is diagnostic; niche coverage is not treated as an independently validated calibration benefit.

## 7. Cost
| arm | calibration_n | expert_tour_cache_hits | new_concorde_calls | current_H0_refreshes_this_audit | labeling_wall_seconds_this_audit | gp_wall_seconds |
|---|---|---|---|---|---|---|
| OLD_CENTROID435 | 435 | 435 | 0 | 0 | 0.000000 | 129.154813 |
| RANDOM500 | 500 | 49 | 451 | 500 | 220.825354 | 123.621940 |

## 8. Integrity
All 20 checks: **PASS**. All ten seed predictions finite/nonconstant: **True**. Optional full-O0 scoring used zero new labels and did not affect the gate.

## 9. Verdict
**C0_RANDOM500_BOOTSTRAP_SUPPORTED**

## 10. Exact Cycle-0 rule
Cycle 0: GP0 <- Random500(O0). Cycle s >= 1: B_s <- Random500(A_s, 500); refresh current L_BQ only for B_s; GP_s <- GP4(B_s).

This audit addresses GP0 initialization only. It does not claim Random500 is a more accurate regressor, centroid sampling is generally useless, or QD is unnecessary. QD remains the evolutionary parent-diversity mechanism.
