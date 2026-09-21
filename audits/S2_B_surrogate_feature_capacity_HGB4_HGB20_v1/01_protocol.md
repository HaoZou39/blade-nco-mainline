# S2-B protocol — surrogate feature capacity (HGB4 vs HGB20)

Offline / controlled feature-set comparison under identical Random500 calibration.
Historical run: `blade_v2_full_seed2026`. Cycles: C00–C14 (M0–M14).

## Scientific question

Does expanding the structural representation from 4 to the frozen canonical 20-feature
set allow the **same** HGB family, **same** 500 labels, and **same** HGB_CFG to recover
materially more hard-tail signal on leakage-clean P0?

## Held fixed

- historical B_s membership, |B_s|=500
- current solver M_s and exact L_BQ labels
- target log1p(L_BQ)
- HGB_CFG from `src/final_run/science.py`
- seeds 2026–2030, prediction = five-seed median
- S1 P0_clean_s and S1.1 Random/Oracle ceilings

## Feature sets

- HGB4: QD4 / GP4 four features (S2-A reused)
- HGB20: `configs/hgb20_features.yaml` (identical to cycle0 `gp20_features.json`)
- GP4: historical reference only; not retrained

HGB20 features are solver-free structural salesperson quantities, transformed with the
frozen `formal_raw_scaler.json` arcsinh map. No P0 refit. No invented features.

## Not done in S2-B

No HGB tuning, no feature selection/ablation, no Full94, no RF, no ensemble,
no sample-size change, no Concorde, no BQ, no mutation, no production GP change,
no automatic S2-C launch.

## Primary metric

CaptureRate10 = (Model_ENR10 − RandomMean_ENR10) / (Oracle_ENR10 − RandomMean_ENR10)
Primary contrast: DeltaCapture10 = CaptureRate10(HGB20) − CaptureRate10(HGB4).

