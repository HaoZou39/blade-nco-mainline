# MODEL_CONFIG_CONFIRMATION_NEEDED

Status: `S2_A_IMPLEMENTATION_READY_MODEL_CONFIG_CONFIRMATION_NEEDED`

S2-A implementation is in place. **HGB4/RF4 training has not started.**

## What is resolved

### HGB4 — canonical hardness-regression config

Source: `src/final_run/science.py` symbol `HGB_CFG`

Used by: `train_hgb_controls` for HGB4 on QD4 features with `y = log1p(L_BQ)`.
Same dict is in `scripts/run_cycle0_models_and_evaluation.py`.

```
learning_rate: 0.05
max_iter: 500
max_leaf_nodes: 31
max_depth: null
min_samples_leaf: 20
l2_regularization: 1.0e-4
early_stopping: false
random_state: SEED   # 2026–2030 for the five-seed median
```

No scaling. This is a solver-hardness surrogate control, not a classification task.

Config file: `configs/surrogate_s2a_gp4_hgb4_rf4.yaml`
Config SHA256: `5589a61149a741ab86efb23e057c04095e6201df8f996a176f317b301bd5cb38`

## What is unresolved

### RF4 — no canonical hardness-regression config

Repository search found:

- `HistGradientBoostingRegressor` hardness controls: yes
- `RandomForestRegressor` hardness surrogate: **none**
- `RandomForestClassifier` in EvoReal structure-alignment audits: **unrelated classification**; not reusable under S2-A §7

Inventing RF hyperparameters would violate S2-A (no silent invention, no search).

## Candidate RF4 configurations (not selected)

These are the smallest reasonable candidates for human confirmation.
None has been trained.

```yaml
A_sklearn_regressor_defaults:
  n_estimators: 100
  max_depth: null
  max_features: 1.0
  min_samples_leaf: 1
  max_leaf_nodes: null
  bootstrap: true
  n_jobs: 1
  note: sklearn RandomForestRegressor defaults except random_state=SEED
B_hgb_capacity_analog:
  n_estimators: 500
  max_depth: null
  max_features: 1.0
  min_samples_leaf: 20
  max_leaf_nodes: 31
  bootstrap: true
  n_jobs: 1
  note: capacity analog of canonical HGB_CFG (not itself a historical RF config)
C_moderate_regularization:
  n_estimators: 300
  max_depth: null
  max_features: sqrt
  min_samples_leaf: 5
  max_leaf_nodes: null
  bootstrap: true
  n_jobs: 1
  note: smaller/regularized forest; also not a historical hardness config
```

Shared across any chosen candidate:

- `random_state` in {2026, 2027, 2028, 2029, 2030}
- five-seed median prediction
- no scaling
- `n_jobs: 1` (deterministic)

## Source integrity already checked

- historical run `blade_v2_full_seed2026`
- B_s N=500 unique for C00–C14
- P0_clean_s hash identity vs S1
- B_s ∩ P0_clean_s empty

## Required human decision

Confirm one of:

1. RF candidate A, B, or C above
2. another frozen RF config specified in writing
3. drop RF4 from S2-A and run GP4 vs HGB4 only

Do not request a search. After confirmation, S2-A training can resume from this framework.

STOP before HGB/RF training.
