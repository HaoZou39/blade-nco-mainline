# Frozen Evaluation Pool

Each row in the CSV is a unique coordinate within one replicate. C0/C1 are boolean memberships, not extra observations. Global deduplication leaves 59,678 coordinate hashes from 59,793 replicate-level unique rows.

| replicate | memberships_E0_plus_E1 | unique_union_n | E0_only | E1_only | shared | strict_root_safe_n | C0_membership | Oracle_C1_membership | F40_complete |
|---|---|---|---|---|---|---|---|---|---|
| Original | 20000 | 19929 | 9929 | 9929 | 71 | 285 | 10000 | 10000 | True |
| S1 | 20000 | 19937 | 9937 | 9937 | 63 | 349 | 10000 | 10000 | True |
| S2 | 20000 | 19927 | 9927 | 9927 | 73 | 345 | 10000 | 10000 | True |
