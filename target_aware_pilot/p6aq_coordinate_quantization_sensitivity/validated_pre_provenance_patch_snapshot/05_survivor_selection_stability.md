# Survivor10k Selection Stability

| amplitude_a | retention_min | retention_median | retention_max | jaccard_min | cost_cross_max_change | swd_membership_max | energy_membership_max |
|---|---|---|---|---|---|---|---|
| 0.050000 | 0.979700 | 0.981900 | 0.989400 | 0.960208 | 0.009323 | 0.006999 | 0.002126 |
| 0.250000 | 0.978800 | 0.981900 | 0.987300 | 0.958480 | 0.009755 | 0.006837 | 0.002243 |
| 0.500000 | 0.973500 | 0.979500 | 0.983600 | 0.948368 | 0.010468 | 0.010026 | 0.002181 |

Every row uses all 119,530 candidates, exact HiGHS transport, unchanged 208/209 quotas, and 10,000 unique survivors. Cross-cost columns in `05_selection_stability.csv` re-solve assignment for fixed memberships and therefore separate target movement from membership adaptation. Hash changes and population distribution changes are reported independently.
