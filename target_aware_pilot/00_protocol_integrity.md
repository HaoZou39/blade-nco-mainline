# P0/P1 Protocol Integrity Audit

## Scope and isolation

- This audit is restricted to P0/P1. It did not generate a new training set, start BQ fine-tuning, or modify the formal A0 implementation.
- All new analysis code, tables, and figures are contained in `target_aware_pilot/`.
- The analysis reused frozen artifacts from the completed five-cycle A0 run and performed CPU-only feature extraction/statistics. No Oracle or GPU workload was launched.
- The held-out TSPLIB22 were used only to verify manifest identity, uniqueness, and disjointness. Their coordinates/features were not loaded into feature selection, distribution estimation, parameter selection, or acquisition analysis.

## M4 checkpoint integrity

- Checkpoint: `runs/BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1/A0_FULL_BASE/checkpoints/M4_E256.tsp.best`
- SHA-256: `c749311bd2a461084d0cbcfd383a4c0a1fc1268762b72e5f7220952f108f8190`
- Size: 12,470,052 bytes.
- The checkpoint loads successfully on CPU and contains 113 finite tensors / 3,107,539 parameters.
- Stored metadata identifies cycle 3, arm `A0 Full`, epoch 256, 10,000 hard unique instances, and no anchor mixture.

## Correct M4-conditioned HGB40

The matching surrogate is the C04 model, not the C03 model:

- C04 `model_before` is M4 and its recorded input checkpoint SHA-256 exactly matches the M4 hash above.
- C04 recomputed solver-relative hardness with M4 on the frozen 10,000-instance calibration sample before fitting HGB40.
- C03 is M3-conditioned and is the surrogate used to acquire data that subsequently trained M4; it must not be interpreted as M4-conditioned.
- Five persisted C04 HGB40 replicas are present for seeds 2026–2030. Their predictions are numerically identical, consistent with the frozen HGB setup having no stochastic row/feature subsampling.

## Calibration and F40 integrity

- Calibration count: exactly 10,000.
- Policy: `canonical_sort_uniform_random10k`; sample seed: 202604.
- Calibration membership SHA-256: `76026e7436d54e9dc15db8c84f948cb579aa584598e66fe786a0569270f75f25`.
- F40 feature names and order were read from frozen `09_feature_subset_manifests.json`; recorded F40 manifest hash: `19c2778c207a82641d6fab4290d8ccc198c63aa4280550b1c54e60cb5b70c8d7`.
- Features were computed with the repository's frozen `salesperson` extractor. HGB inputs use the frozen per-feature transform `arcsinh((raw - calibration_median) / calibration_scale)` with the recorded missing-value handling.
- Prediction target: `log1p_LBQ`; early stopping is disabled. Training parameters and persisted model hashes are recorded in the existing C04 manifests and were not altered.

## Validation boundary

- The C04 selected Hard10k has fresh M4 hardness labels that were not used to fit HGB40, so it is used as an independent post-ranking validation set.
- It is not IID: it was selected by the same surrogate from the candidate pool. Accordingly, its Spearman/MAE/tail metrics are reported as acquisition-shifted validation evidence, not as an unbiased population generalization estimate.
- Training-fit metrics are retained only for context and are not interpreted as validation performance.

## TSPLIB split integrity

- The exact EvoReal development lists were recovered from the repository's corrected author files: 19 S1 + 12 S2 + 17 S3 = 48 unique instances.
- Every listed instance was matched by name and exact coordinate array to the frozen OriginalTSPLIB70 assets.
- The held-out set is the exact 22-instance complement. Development and held-out names are unique and disjoint.
- No random 48/22 reconstruction was used.

## Acquisition artifacts used

- FOGA pool: C04 Parent+Child union, 119,530 unique instances.
- BLADE Hard10k: C04 pure global HGB ranking output (`06_selection/T_selected.parquet`), 10,000 unique instances.
- The subsequent QD parent update is not substituted for the requested original Hard10k acquisition.

## Reproducibility

The complete audit driver is `target_aware_pilot/run_p0_p1_audit.py`. It reads frozen artifacts and writes only within this directory. Supporting validation, scale-dependence, missingness, and normalization-sensitivity tables are retained alongside the required deliverables.
