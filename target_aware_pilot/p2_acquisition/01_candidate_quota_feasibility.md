# Structure10k Candidate and Quota Feasibility

## Integrity boundary

- E1 expanded pool: 229,148 unique, finite, N=100 coordinate arrays. Every SHA-256 instance ID was reproduced from canonical float64 coordinate bytes.
- Known formal training union C00–C04: 44,998 unique. The E1 overlap (18,670) was excluded before acquisition.
- C00–C03 exclusion is required to avoid reuse from M1–M4. C04 was additionally excluded so the provisional set also has zero overlap with the completed original-M5 Hard10k comparator.
- Eligible after conservative exclusion: 210,478 unique candidates.
- M0 is an imported canonical checkpoint and contains no per-instance pretraining membership in the repository; exact collision auditing against its external pretraining stream is therefore unavailable. No M0 training instances are knowingly reused.
- Held-out TSPLIB22 coordinates/features were never loaded. Only the frozen development TSPLIB48 reference was used.

## Frozen acquisition rule

The 10,000 budget is assigned by largest-remainder quotas from the frozen TSPLIB48 3×3 cell counts. Within each cell, quota is divided across the cell's development references. Each reference receives 12 nearest anchors, followed by deterministic hash-diverse candidates ordered through strict, transition (`r < d <= 2r`), and boundary fallback tiers. Global lineage and direct-parent caps are enforced and relaxed only by the recorded fixed schedule.

HGB hardness is absent from this rule and was computed only after `selection_freeze.json` was written.

| Cell | Target refs | Target quota | Eligible | Strict eligible | Selected | Deficit |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 11 | 2,292 | 17,153 | 4,520 | 2,292 | 0 |
| 1 | 5 | 1,042 | 3,820 | 2,212 | 1,042 | 0 |
| 2 | 0 | 0 | 12 | 0 | 0 | 0 |
| 3 | 1 | 208 | 40,953 | 3,133 | 208 | 0 |
| 4 | 10 | 2,083 | 18,297 | 8,403 | 2,083 | 0 |
| 5 | 5 | 1,042 | 1,161 | 842 | 1,042 | 0 |
| 6 | 4 | 833 | 79,866 | 10,534 | 833 | 0 |
| 7 | 1 | 208 | 27,871 | 3,029 | 208 | 0 |
| 8 | 11 | 2,292 | 15,773 | 6,023 | 2,292 | 0 |

All cell quotas are feasible and exactly filled. Cell 5 requires 256 selected non-strict/transition candidates because its strict capacity is below its 1,042 target quota. At reference level, `eil51` remains the only development target with zero strict-neighborhood candidates in the expanded pool.

The secondary per-reference subquotas are exact for 48/48 references; the maximum absolute deviation is 0. The single same-cell coverage repair deliberately moves one assignment slot so that `ts225` is covered without altering any primary cell quota. Full details are in `support_reference_quota_fulfillment.csv`.
