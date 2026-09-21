# Final BLADE-NCO QD Resolution and Hard-Population Budget Audit

## Decision

Audit A: **K1000_RESOLUTION_ROBUST** (6/6 robust cycles; deep stability PASS).

Audit B: **C10_REQUIRED**. The smallest internally saturated per-niche depth is **c=10**. Therefore the development recommendation is K=1000, c=10, mu=10000, lambda=110000.

No sealed external benchmark was evaluated or used for selection. This remains a development conclusion; nothing was copied to or launched from `BLADE-NCO-mainline`.

## Table A — QD resolution sensitivity

| K | c | mu | robust_cycles | common_grid_entropy | GP4_quality_ratio | lineage_diversity | classification |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 500 | 20 | 10000 | 0 | 0.94966 | 1 | 0.978645 | COMPARATOR |
| 1000 | 10 | 10000 | 6 | 0.963776 | 0.996153 | 0.978995 | ROBUST_MIDDLE_RESOLUTION |
| 2000 | 5 | 10000 | 0 | 0.993094 | 0.992916 | 0.979624 | COMPARATOR |

## Table B — Initial / candidate budget

| c | P0_size | offspring | union_raw_size | union_unique_size | P1_size | new_selected_unique |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1000 | 11000 | 12000 | 11998 | 1000 | 993 |
| 2 | 2000 | 22000 | 24000 | 23997 | 2000 | 1804 |
| 5 | 5000 | 55000 | 60000 | 59988 | 5000 | 4302 |
| 10 | 10000 | 110000 | 120000 | 119984 | 10000 | 8694 |

## Table C — GP0 capability on QD_DEV

| c | train_n | Spearman | R2 | Top10_capture | Top10_enrichment | prediction_target_SD_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1000 | 0.516702 | 0.344068 | 0.46 | 2.0541 | 0.493758 |
| 2 | 2000 | 0.520618 | 0.353228 | 0.48 | 2.09837 | 0.629333 |
| 5 | 5000 | 0.521022 | 0.348756 | 0.48 | 2.09837 | 0.651908 |
| 10 | 10000 | 0.523195 | 0.361473 | 0.48 | 2.09523 | 0.584875 |

## Table D — M0 adaptation

| c | pre_mean | post_mean | gain | Uniform_post | P1_repair | hard_n | optimizer_steps | gain_ratio_to_c10 | saturated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.0760004 | 0.0695268 | 0.0064736 | 0.0461091 | 0.0438791 | 1000 | 2000 | 0.450825 | False |
| 2 | 0.0760004 | 0.0660041 | 0.00999633 | 0.0454952 | 0.0549473 | 2000 | 4000 | 0.69615 | False |
| 5 | 0.0760004 | 0.0626709 | 0.0133295 | 0.0455997 | 0.0586673 | 5000 | 10000 | 0.928271 | False |
| 10 | 0.0760004 | 0.0616409 | 0.0143595 | 0.0472211 | 0.057403 | 10000 | 20000 | 1 | True |

## Interpretation boundaries

K=1000 is supported as a robust middle-resolution design under the preregistered K500/K1000/K2000 sensitivity audit; this is not a mathematical-optimality claim. Under the fixed 256-epoch M0 adaptation protocol, the chosen c is the smallest branch retaining at least 95% of the c=10 QD development benefit without significantly worse Uniform retention. Candidate budget was not tuned independently: lambda=11*K*c follows from one application of every FOGA-11 operator per parent.
