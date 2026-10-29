# Input Integrity

All B/C members are unique members of their corresponding frozen union. B and C have identical per-niche counts and exactly reproduce frozen lambda=.5 quotas. All P6-E0 metrics reproduce within 1e-12. Explicit lineage IDs are absent; immutable direct-parent hash is reported as a lineage proxy and lineage_root separately.

| replicate | arm | selected_n | all_members_in_union | quota_matches_lambda05 | max_abs_quota_difference |
|---|---|---|---|---|---|
| Original | B | 10000 | True | True | 0 |
| Original | C | 10000 | True | True | 0 |
| S1 | B | 10000 | True | True | 0 |
| S1 | C | 10000 | True | True | 0 |
| S2 | B | 10000 | True | True | 0 |
| S2 | C | 10000 | True | True | 0 |
