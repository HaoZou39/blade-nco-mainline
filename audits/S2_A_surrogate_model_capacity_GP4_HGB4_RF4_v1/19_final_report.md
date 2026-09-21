# S2-A surrogate model capacity audit — GP4 vs HGB4

Offline controlled comparison. Model class only. RF4 not run.

Config SHA256: `c15fed1a8cce3ef9679d124bd91af636aa71e294358459c3a27536b5794f83ae`
Historical run: `blade_v2_full_seed2026`
HGB4: canonical `HGB_CFG` from `src/final_run/science.py`; not modified; no search.
RF4: `NOT_RUN_RF_CONFIG_NOT_FROZEN` (human Option 3, 2026-09-21).

## Primary late-phase medians (C10–C14)

- GP4 ENR10=1.0919 CaptureRate10=5.90% Top10Recall=14.96% Tail10Rho=0.0488 niches=5.0 N_eff=3.86
- HGB4 ENR10=1.0691 CaptureRate10=4.27% Top10Recall=11.81% Tail10Rho=-0.0573 niches=15.0 N_eff=6.14
- Oracle ENR10=2.5636

## Q1–Q12

**Q1.** Under the same Random500 and GP4 features, does HGB4 outperform GP4? **NO**
Paired CaptureRate10 HGB4−GP4: mean=-0.0242, median=-0.0249, 4/15 cycles positive, bootstrap 95% CI [-0.0568, 0.0092] (seed=2026, 10000 repeats).

**Q2.** Does RF4 outperform GP4? **NOT_RUN_RF_CONFIG_NOT_FROZEN**

**Q3.** Largest Oracle ENR10 capture among 4-feature models: **GP4** (late median CaptureRate10 GP4=5.90%, HGB4=4.27%).

**Q4.** Best 4-feature model vs GP4, late CaptureRate10: -1.63 percentage points (HGB4−GP4). Trajectory mean delta=-2.42 pp.

**Q5.** Top10 Recall late median change: -3.15 percentage points. Trajectory mean HGB4−GP4=-1.69 pp, CI [-3.89, 0.64] pp.

**Q6.** Fine-tail ranking (late median tail10 rho): GP4=0.0488, HGB4=-0.0573.

**Q7.** Structural concentration (late median Top10k niches / N_eff / largest share): GP4 5.0 / 3.86 / 0.3360; HGB4 15.0 / 6.14 / 0.2748.

**Q8.** Phase CaptureRate10 deltas (HGB4−GP4 medians): EARLY=0.0446 MID=-0.0453 LATE=-0.0163. HGB better every cycle: False.

**Q9.** Complementarity: observed. Mean unique true-hard recovery GP-only=12.60, HGB-only=10.47. Ensemble not built.

**Q10.** Scoring ~100k candidates, median ms: GP4=20.25, HGB4=35794.82. Median O_s wall-clock s: GP4=0.0222, HGB4=39.2608. Median ensemble fit s: GP4=82.85 (historical), HGB4=16.18.

**Q11.** Is GP model class itself clearly limiting? **UNCLEAR** under a frozen 4-feature Random500 design.

**Q12.** Next experiment: **S2-B feature-capacity audit**. Do not launch automatically. Case B.

## Verdict

`MODEL_CLASS_CAPACITY_GAIN_MIXED`

This is an offline surrogate conclusion only. It does not authorize changing the production GP surrogate.

## Complementarity flag

ENSEMBLE_COMPLEMENTARITY_OBSERVED

## Notes

- RF4 fields: `NOT_RUN_RF_CONFIG_NOT_FROZEN`.
- No S2-B / S2-C / ensemble launch.
- No Concorde, no BQ training, no mutation, no production pipeline edit.
- Late ENR10 gain HGB4−GP4 = -0.0227.
