# HGB40 Provenance Audit

**Gate: PASS for frozen inference.** The recovered formal C04 ensemble is conditioned on formal M4 SHA256 `c749311bd2a461084d0cbcfd383a4c0a1fc1268762b72e5f7220952f108f8190`. C04 refreshed 10,000 archive examples with M4 teacher-forced LBQ/CE and fitted five `HistGradientBoostingRegressor` models (seeds 2026–2030) to `log1p_LBQ`; deployment aggregation is the median and this audit applies `expm1` exactly once. There is no output calibrator. The payload F40 names/order equal the frozen manifest byte-for-byte. Parameters were fixed, `early_stopping=false`, and no P6-B/C0/C1/C2 labels were used for fitting or model selection.

The historical model has no internal held-out validation split: its manifest reports training fit only. The later C04 Hard10k labels were obtained after deployment and were not used to refit or choose a seed. Exact coordinate leakage into P6-B unions is small, but root overlap is very high because both experiments descend from the same evolutionary ecosystem. Consequently full replay and exact-hash-safe results are descriptive; only the small no-training-root subset is called strict root-safe. No geometric near-duplicate threshold was pre-registered, so near-duplicate leakage is not asserted from feature distance.

| record_type | replicate | evaluation_n | exact_coordinate_overlap_n | exact_coordinate_overlap_fraction | training_lineage_root_overlap_n | training_lineage_root_overlap_fraction | strict_root_safe_n | postfit_validation_coordinate_overlap_n |
|---|---|---|---|---|---|---|---|---|
| replicate | Original | 19929 | 34 | 0.001706057 | 19644 | 0.9856992 | 285 | 21 |
| replicate | S1 | 19937 | 33 | 0.001655214 | 19588 | 0.9824949 | 349 | 15 |
| replicate | S2 | 19927 | 27 | 0.001354946 | 19582 | 0.9826868 | 345 | 17 |
| global_unique | GlobalUnique | 59678 | 68 | 0.001139448 | 58701 | 0.9836288 | 977 | 39 |

Cross-replicate reuse and all hashes are in the CSV/manifests.
