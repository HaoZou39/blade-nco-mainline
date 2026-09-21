# S1 protocol

Mode: OFFLINE / READ-ONLY SURROGATE AUDIT.
Scope: offspring-only Full trajectory C00–C14 (M0–M15 / GP0–GP14).
Union-corrected cycles are excluded from primary analysis.
No BQ/GP/RF/HGB/XGBoost/LightGBM/neural training. No mutation. No Concorde.

## Historical run
- FULL_ID: `blade_v2_full_seed2026`
- Identity: UNAMBIGUOUS
- Architecture: OFFSPRING-ONLY
- Candidate pool: O_s (not P_{s-1} UNION O_s)
- M0 sha256: `6826d7a71b30d882e4673a26ced162b7391157795931efe230f4d019292bba15` match=True
- M15 sha256: `82077aa03ab1c98a713bd92adc7797b7d1947c6a7ce3e62f01a33121524c6668` match=True
- C15 leftover directories exist without CYCLE_COMPLETE and are excluded.

## GP4 protocol (historical, not assumed)
- Features (GP4 / QD4): `spn_sum_of_lowest_edge_values`, `spn_distance_varcoeff`, `spn_mst_depth_mean`, `spn_mst_dists_median`
- Target: `log1p_LBQ` (current-solver teacher-forced hardness)
- Ensemble: five seeds 2026–2030, SPERF formula selection, median aggregation
- Population 2000 × 60 generations, Huber fitness
- Calibration: Random500 (C00 from O0; later cycles from archive A_s) with current-hardness refresh N=500
- If a cycle differed, it is recorded in `03_gp_protocol_inventory.csv` rather than normalized away.

## Solver / GP alignment
GP_s is evaluated relative to M_s. Analysis stops at GP14 / C14. M15 does not imply GP15.

## P0 panel
- Canonical sealed E1/E2 membership file: `D:\NCO\BLADE-NCO-mainline\runs\audits\gp4_calibration_sample_size_v1\P0_MEMBERSHIP_AND_FEATURES.csv`
- Raw N = 1267
- Membership hash = `5ac913183f59f30d39f4f36b21bea1fc62ccc06b2d7fce4a98c75af87a5003a4`
- Per-cycle clean panel: P0 minus GP_s calibration hashes (instance_hash identity).
- P0_common (union of all C00–C14 calibration excluded): N=1234, used_as_secondary=True

## Current hardness
- H_s(X) = L_BQ(X, M_s) via canonical teacher-forced hardness
- Expert tours reused from cycle-0 initialization validation
- NEW CONCORDE CALLS = 0

## Metric families (kept separate)
1. ARCHIVE-FIT: GP_s vs current hardness on calibration B_s
2. INDEPENDENT P0: GP_s vs H_s on P0_clean_s
3. SELECTED-T: GP_s vs exact H_s on historically selected T_s (acquisition-conditioned)

## Primary metric
ENR10 on independent P0. Regression R²/Spearman are secondary and do not override acquisition metrics.

## Temporal transfer
Lag-1 and frozen-GP0 comparisons are labeled TEMPORAL TRANSFER DIAGNOSTIC ONLY.
