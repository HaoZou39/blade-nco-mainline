# S1.1 protocol

Mode: OFFLINE / READ-ONLY / NO TRAINING.

Source: `D:\NCO\BLADE-NCO-mainline\audits\S1_gp4_surrogate_behavior_M0_M15_v1` (offspring-only `blade_v2_full_seed2026`, C00–C14).

This audit does not train BQ, GP, HGB/RF/XGBoost, or any new surrogate.
It does not run mutation or Concorde. Historical runs and S1 artifacts are not modified.
NEW CONCORDE CALLS = 0.

## Panel

Per-cycle `P0_clean_s` is reconstructed from the same S1 definition:
sealed P0 membership minus GP_s calibration hashes (`instance_hash`).
Current hardness `H_s` is read from S1 `06_p0_current_hardness.parquet`.
Historical GP_s predictions are replayed from saved formulas, then checked against S1 ENR.

k_q = max(1, round(q * N)) with the same rounding as S1.

## Estimators

- GP ENR: predicted Top-q% by historical GP_s score.
- Oracle ENR: true Top-q% by exact H_s. Analysis-only perfect-ranking ceiling. Not a deployable policy.
- Random ENR: 10000 subsets of size k_q, sampled without replacement.
  seed = 2026 + cycle_index * 100 + q_code (5/10/20).

## Capture rate

CAPTURE_RATE_q = (GP_ENR − RandomMean) / (Oracle_ENR − RandomMean).
Negative values are preserved. Extremely small denominators are marked
`CAPTURE_RATE_UNDEFINED_LOW_HEADROOM`.

Primary metric: ENR10.
