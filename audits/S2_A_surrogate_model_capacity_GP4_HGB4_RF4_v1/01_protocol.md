# S2-A protocol

Mode: OFFLINE / CONTROLLED SURROGATE COMPARISON.

Scope: C00–C14 of `blade_v2_full_seed2026`. Model class only.
Fixed: Random500 B_s, current H_s, GP4 four features, target log1p(L_BQ), S1 P0_clean_s.

Historical GP4 is reused, not refit.

HGB4 hyperparameters are the canonical hardness-regression `HGB_CFG` from
`src/final_run/science.py` (`train_hgb_controls`). The config is not modified.
No HGB hyperparameter search.

RF4 is **not run**. Human confirmation 2026-09-21 Option 3: drop RF4.
RF-related fields are marked `NOT_RUN_RF_CONFIG_NOT_FROZEN`.

No extra features. No sample-size expansion. No Concorde. No BQ training.
No mutation. No production pipeline change. Do not launch S2-B.
