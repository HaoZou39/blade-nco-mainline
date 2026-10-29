# Bottleneck Decomposition

Quota loss is exactly zero in all seeds. C reaches the exact quota capacity in every seed; B does so except for one all48/matched reference in S2. These differences are structural opportunity counts, not BQ training-value or gap losses.

| replicate | scope | reference_total | C_pool | C_free10k | C_quota10k | C_B | C_C | Generation_Unreachable | Quota_Loss | Selection_Loss_B | Selection_Loss_C | capacity_exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Original | all48 | 48 | 25 | 25 | 25 | 25 | 25 | 23 | 0 | 0 | 0 | True |
| Original | matched_75_150 | 17 | 15 | 15 | 15 | 15 | 15 | 2 | 0 | 0 | 0 | True |
| S1 | all48 | 48 | 24 | 24 | 24 | 24 | 24 | 24 | 0 | 0 | 0 | True |
| S1 | matched_75_150 | 17 | 14 | 14 | 14 | 14 | 14 | 3 | 0 | 0 | 0 | True |
| S2 | all48 | 48 | 25 | 25 | 25 | 24 | 25 | 23 | 0 | 1 | 0 | True |
| S2 | matched_75_150 | 17 | 15 | 15 | 15 | 14 | 15 | 2 | 0 | 1 | 0 | True |
