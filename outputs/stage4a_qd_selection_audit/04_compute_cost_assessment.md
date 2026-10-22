# Stage 4A compute-cost assessment

## Exact work units

| cycle | new_current_checkpoint_labels_needed | missing_B_with_reusable_tour_n | missing_B_without_reusable_tour_n | BQ_label_wall_s_point_estimate | Concorde_CPU_s_median_runtime_extrapolation | Concorde_CPU_s_P90_runtime_extrapolation | idealized_8_worker_Concorde_wall_s_median | idealized_8_worker_Concorde_wall_s_P90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C00 | 41 | 9 | 32 | 0.925 | 16.076 | 31.720 | 2.009 | 3.965 |
| C03 | 765 | 313 | 452 | 17.266 | 227.067 | 448.049 | 28.383 | 56.006 |
| C09 | 862 | 318 | 544 | 19.455 | 273.284 | 539.245 | 34.160 | 67.406 |
| ALL_C00_C09 | 6904 | 2952 | 3952 | 155.821 | 1985.327 | 3917.453 | 248.166 | 489.682 |

For the proposed C00+C03+C09 study, the exact requirement is **1,668** new same-checkpoint LBQ labels. Of these, **640** already have a verified reusable optimal tour in the audited cache scope; **1,028** need both a new Concorde tour and BQ labeling. Across all ten cycles, the corresponding counts are **6,904**, **2,952**, and **3,952**.

## Measured timing anchors

- Existing S6-AE archive-refresh measurements total 2042.559 seconds for 90,500 same-checkpoint hardness evaluations with tours already available: 0.022570 wall-seconds per instance (44.31 instances/second). This is the point estimate used above.
- Verified S6-AE ledger rows record a median Concorde solver runtime of 0.502360 seconds and P90 of 0.991258 seconds per instance.
- The implementation calls Concorde with 8 workers. The table's 8-worker wall figures are idealized divisions of CPU-time extrapolations; they are **unverified estimates**, exclude scheduling/I/O/start-up effects, and must not be treated as measured future wall time.
- Historical combined Oracle-plus-hardness wall times and physical new-call counts are shown below. They are descriptive benchmarks only because the wall time also includes cache lookup, tour assembly, and 10,000 BQ hardness evaluations.

| cycle | new_concorde_calls | oracle_and_hardness_wall_s |
| --- | --- | --- |
| C00 | 0 | 235.053 |
| C01 | 1552 | 338.980 |
| C02 | 3933 | 556.926 |
| C03 | 4346 | 594.635 |
| C04 | 4677 | 635.927 |
| C05 | 4777 | 668.338 |
| C06 | 4540 | 607.941 |
| C07 | 4864 | 631.753 |
| C08 | 4853 | 670.756 |
| C09 | 5320 | 714.192 |

No GPU reservation, future hardware speed, or unseen Concorde failure rate is assumed. Exact cost becomes known only if Stage 4B is separately authorized and run.
