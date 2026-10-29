# Runtime and Resource Audit

| status | current_materialization_pass_wall_s | cumulative_wall_status | peak_rss_mb_current_pass | cpu_only | gpu_used | mutation | concorde | hgb_training | bq_training | selection_rows | candidate_rows | exact_ot_evaluations_total | exact_ot_solves_current_pass | exact_ot_cache_reused_current_pass | python | numpy | pandas | scipy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| COMPLETE | 66.18831 | not summed across interrupted report pass and deterministic resume | 1087.805 | True | False | False | False | False | False | 150000 | 59793 | 15 | 0 | True | 3.10.12 | 2.2.6 | 2.3.3 | 1.15.3 |

CPU-only frozen inference reuse, quota selection, exact Top5 OT and diagnostics. The first pass completed all 15 exact OT evaluations and then stopped on a report-column naming error; the repaired deterministic pass reused the validated structural CSV. Therefore only current-pass wall time is reported and no falsely precise cumulative time is synthesized. No GPU, mutation, evolution, Concorde, HGB/BQ training, or benchmark evaluation.
