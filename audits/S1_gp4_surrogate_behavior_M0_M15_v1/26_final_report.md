# S1 GP4 surrogate behavior audit — offspring-only M0–M15

## Scope
This audit reuses the completed historical offspring-only Full run `blade_v2_full_seed2026`.
Primary horizon is C00–C14 (solvers M0–M14, snapshots GP0–GP14, final solver M15).
Corrected Parent+Offspring Union cycles are not in the primary analysis.
Historical GP predictors are loaded from saved joblibs; nothing is refit.
Current hardness on sealed P0 uses cached expert tours and teacher-forced L_BQ. NEW CONCORDE CALLS = 0.

## Historical run identity
- Status: **UNAMBIGUOUS**
- Run: `blade_v2_full_seed2026`
- Architecture: offspring-only; candidate selection uses O_s, not P∪O.
- M0 sha256 match: True
- M15 sha256 match: True
- C00–C14 CYCLE_COMPLETE present; C15 leftover excluded.
- Matches `summary/full_main_experiment` M0–M15 8-set trajectory.

## GP protocol inventory
Calibration is Random500 with current-hardness refresh on all 500 rows for every C00–C14.
C00 source is Random500 from O0; C01–C14 source is Random500 from archive A_s.
Protocol source change across the run: True (C00 O0 vs later A_s). This is reported, not normalized away.
GP seeds 2026–2030, population 2000, 60 generations, SPERF, median ensemble, four QD4 features, target log1p_LBQ.
See `03_gp_protocol_inventory.csv` and `04_gp_formula_inventory.csv`.
Unique formula hashes across 75 seed-formulas: 75. Mean node count: 8.92.

## P0 leakage
Raw sealed P0 N = 1267. Overlap with GP calibration is hash-based (see `05_p0_leakage_by_cycle.csv`).
P0_common N = 1234 (secondary matched panel; primary metrics use per-cycle clean P0).

## Metric families
Archive-fit Spearman is a training diagnostic only. Independent P0 metrics are generalization. Selected-T metrics are acquisition-conditioned and are not independent validation.
Archive-fit Spearman trajectory: 0.224, 0.384, 0.304, 0.320, 0.240, 0.240, 0.269, 0.177, 0.302, 0.308, 0.275, 0.247, 0.337, 0.266, 0.317

## Independent P0 hard-region localization
ENR10 trajectory: 0.983, 0.916, 1.113, 1.099, 0.995, 1.058, 1.060, 1.072, 1.070, 1.069, 1.085, 1.092, 1.094, 1.007, 1.146
Early median ENR10=0.9951, Spearman=0.0483, Top10 Recall=0.0945, tail10 rho=0.0164
Mid median ENR10=1.0694, Spearman=0.0158, Top10 Recall=0.1349, tail10 rho=0.0211
Late median ENR10=1.0919, Spearman=0.0324, Top10 Recall=0.1496, tail10 rho=0.0488
ENR10>1 means the predicted Top10% of sealed P0 is harder than the panel mean under the current solver. R² is not used as a success threshold.

## Fine-tail ranking
rho_tail10 is Spearman inside GP's own predicted Top10% on clean P0. selected-tail10 rho is the same idea inside historically selected T_s (conditioned on selection).
Median P0 tail10 rho = 0.0211. Median selected tail10 rho = 0.1222. Median within-niche centered rho = 0.1786.

## Between vs within niche on T_s
Median between-niche rho = -0.1000.
Median within-centered rho = 0.1786.
Sample support is in `13_between_within_niche.csv`. Variance fractions are in `14_hardness_variance_decomposition.csv`.

## Actual acquisition T_s
T_s is the historical global-GP Top10k of O_s. Additional QD-floor mixing is not the offspring-only Full rule; component=`global_gp` is verified per cycle.
T mean exact H trajectory: [0.07803, 0.08953, 0.09523, 0.09571, 0.09650, 0.12341, 0.11455, 0.13932, 0.14906, 0.18128, 0.15018, 0.12185, 0.28376, 0.17261, 0.16362]
Selected-set rho trajectory: [0.548, 0.179, 0.210, 0.093, 0.071, 0.261, 0.098, 0.369, 0.136, 0.236, 0.028, -0.018, 0.457, 0.025, 0.126]
No new Oracle labels were created for full-candidate ground truth. Random500_A_s is a GP calibration draw, not a fair O_s random acquisition control, and is not pooled as such.

## Structural concentration
GP Top10k occupied niches C00=212 C14=5.
N_eff C00=83.5 C14=3.2.
Largest niche share C00=0.0414 C14=0.4051.
Adjacent-cycle Top10 hotspot Jaccard median = 0.4500.

## GP score landscape
See `16_score_distribution.csv` for mean/std/percentiles, margin_10, and the rank-10000 vs 10001 threshold gap. Score variance is not treated as a proxy for structural concentration.

## Seed / formula stability
Median seed-to-seed Top10 Jaccard on P0 (per-cycle medians, then median across cycles) = 0.3514.
Median Spearman(disagreement, |error|) = -0.0130.
This is diagnostic only; uncertainty-aware acquisition is not implemented here.

## Feature usage (descriptive)
See `20_feature_formula_usage.csv`.

```
                      feature  n_formulas_using  n_cycles_any_formula_uses  n_cycles_majority_formulas_use
spn_sum_of_lowest_edge_values                46                         10                              10
        spn_distance_varcoeff                35                          8                               7
           spn_mst_depth_mean                34                         11                               6
         spn_mst_dists_median                33                         10                               9
```
Occurrence frequency is not causal importance.

## TEMPORAL TRANSFER DIAGNOSTIC ONLY
Current GP median ENR10 = 1.0704.
Lag-1 GP median ENR10 = 1.0635.
Frozen GP0 late (C10–C14) median ENR10 = 0.9201.
This is not the formal Dynamic-vs-Frozen-GP ablation.

## Current-hardness drift on P0
Median adjacent Spearman(H_s, H_s+1) = 0.9046.
Median adjacent Top10 hardness Jaccard = 0.5728.
This separates moving-target solver drift from surrogate failure.

## Cycle-index descriptive trends
- P0_ENR10: Spearman vs cycle = 0.4214
- P0_Spearman: Spearman vs cycle = -0.0679
- P0_tail10_rho: Spearman vs cycle = 0.3786
- GP_Top10k_N_eff: Spearman vs cycle = -0.9393
- GP_Top10k_niches: Spearman vs cycle = -0.9240
- largest_niche_share: Spearman vs cycle = 0.8714
- seed_disagreement: Spearman vs cycle = 0.7036
These are descriptive associations, not claims that time causes the changes.

## Interpretation matrix
- AXIS A hard-region localization: **REGION_LOCALIZATION_INTERMITTENT**
- AXIS B fine-tail ranking: **FINE_TAIL_RANKING_WEAK**
- AXIS C structural concentration: **SELECTION_CONCENTRATION_SEVERE**
- Future direction (evidence only, no new model trained): **COARSE_GP_PLUS_LOCAL_RERANKER_WORTH_TESTING**

## Negative claims (explicitly not made)
- Low R² is not taken to mean GP is useless.
- High archive-fit Spearman is not taken to mean GP generalizes.
- Concentration is not taken to prove QD is necessary or that GP is wrong.
- No claim that HGB/RF/neural ensembles will improve downstream BQ.
- Better regression accuracy is not equated with better acquisition.
- Selected-T correlation is not called independent generalization.
- External 8-set benchmark tables do not control surrogate conclusions.

## Required questions

### Q1. Across C00–C14, does GP4 consistently enrich true hard instances on independent P0?
ENR10 exceeds 1.0 in 12/15 cycles. Classification: REGION_LOCALIZATION_INTERMITTENT.

### Q2. How large is ENR10, and how does it change from early to late?
Early median ENR10=0.9951; mid=1.0694; late=1.0919.

### Q3. Is global Spearman informative about acquisition quality?
Only weakly. Acquisition uses Top10% selection; ENR10 and Top10 recall are the operational quantities. Global Spearman mixes easy bulk with the tail and can move independently of ENR10. Archive-fit Spearman is not an independent check.

### Q4. Can GP rank instances inside its own predicted hard tail?
Median P0 rho_tail10=0.0211; median selected-tail10 rho=0.1222. Classification: FINE_TAIL_RANKING_WEAK.

### Q5. Is the main signal between structural regions rather than within regions?
Median between-niche rho=-0.1000; median within-centered rho=0.1786.

### Q6. Does the GP-selected Top10k become structurally concentrated over time?
Occupied niches 212 → 5; N_eff 83.5 → 3.2; largest share 0.0414 → 0.4051. Classification: SELECTION_CONCENTRATION_SEVERE.

### Q7. Does concentration increase while hardness enrichment remains positive?
Late median ENR10=1.0919 with concentration class SELECTION_CONCENTRATION_SEVERE. Compare ENR10 trajectory against occupied-niches / N_eff in the master table rather than treating concentration as proof of model error.

### Q8. Are GP evolutionary seeds/formulas stable with one another?
Median seed Top10 Jaccard = 0.3514. Unique formulas=75/75. Ensemble ENR10 minus median single-seed ENR10 is in `18_seed_stability.csv`.

### Q9. Can GP ensemble disagreement identify prediction-error regions?
Median Spearman(disagreement, |error|) = -0.0130. Quartile error ratios are in `19_disagreement_error_analysis.csv`. Diagnostic only.

### Q10. How quickly do historical GP models become stale as the solver changes?
Current median ENR10=1.0704; lag-1=1.0635; frozen GP0 late=0.9201. TEMPORAL TRANSFER DIAGNOSTIC ONLY.

### Q11. Dominant limitation?
Localization=REGION_LOCALIZATION_INTERMITTENT; fine-tail=FINE_TAIL_RANKING_WEAK; concentration=SELECTION_CONCENTRATION_SEVERE. The dominant limitation is the combination of those three axes, not a single number.

### Q12. What should a future stronger surrogate / ensemble attempt to improve?
Recommended direction: **COARSE_GP_PLUS_LOCAL_RERANKER_WORTH_TESTING**. No stronger model was trained. If localization persists and tail ranking is weak, a cheap GP first stage plus local reranker on the reduced hard subset is the evidence-aligned next test. If ENR10 itself collapses, a global replacement is the evidence-aligned next test. Neither is launched here.

## Stop
Do not launch any stronger surrogate or ensemble experiment from this audit. Stop for human review.
