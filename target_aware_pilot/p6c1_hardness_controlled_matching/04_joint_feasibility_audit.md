# Joint Feasibility Audit

The 20% OT tolerance and decile-TV ≥0.05 distinctness rule were frozen before any solve. Ratios use total squared Top5 cost; all selections contain 10,000 instances, so mean-cost ratios are identical.

| replicate | union_n | OT_E1_total | OT_E1_mean | OT_C0_total | OT_C0_mean | OT_C1_min_total | OT_C1_min_mean | C1_over_E1 | C1_over_C0 | decile_total_variation_C0_C1 | constraints_exact | labels_complete | tolerance_pass | hardness_distinct_pass | replicate_gate_pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Original | 19929 | 60475.918 | 6.0475918 | 60429.067 | 6.0429067 | 81024.234 | 8.1024234 | 1.3397769 | 1.3408156 | 0.2747 | True | True | False | True | False |
| S1 | 19937 | 60751.627 | 6.0751627 | 60690.499 | 6.0690499 | 83632.203 | 8.3632203 | 1.3766249 | 1.3780115 | 0.2974 | True | True | False | True | False |
| S2 | 19927 | 60608.814 | 6.0608814 | 60578.423 | 6.0578423 | 81985.586 | 8.1985586 | 1.3527007 | 1.3533793 | 0.2808 | True | True | False | True | False |

## Composition diagnostics

| replicate | arm | ot_mean_cost | mean_m4_ce | median_m4_ce | p90_tail_fraction | p95_tail_fraction | e0_source_fraction | e1_source_fraction | hardness_decile_counts | top5_swd | top5_energy | effective_lineages | largest_lineage_share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Original | C0 | 6.0429067 | 0.061685407 | 0.055837766 | 0.0362 | 0.0149 | 0.0091 | 0.998 | 2613 1780 1382 1051 909 684 608 490 300 183 | 0.73088064 | 0.80413494 | 225.60933 | 0.0416 |
| Original | C1 | 8.1024234 | 0.088040112 | 0.080749049 | 0.1605 | 0.0819 | 0.2811 | 0.724 | 1006 1028 994 1058 1012 981 947 1000 994 980 | 0.84161205 | 1.0237249 | 323.8829 | 0.0386 |
| S1 | C0 | 6.0690499 | 0.061570402 | 0.055748988 | 0.0391 | 0.0142 | 0.009 | 0.9973 | 2609 1764 1395 1059 928 716 574 444 327 184 | 0.7323203 | 0.82408094 | 216.16777 | 0.0426 |
| S1 | C1 | 8.3632203 | 0.089402677 | 0.082986234 | 0.1665 | 0.0844 | 0.3035 | 0.701 | 977 952 946 978 1042 1023 1033 991 1055 1003 | 0.86085293 | 1.0472745 | 326.20234 | 0.0385 |
| S2 | C0 | 6.0578423 | 0.061050167 | 0.055391177 | 0.0344 | 0.0136 | 0.009 | 0.9983 | 2614 1769 1417 1069 915 751 559 433 312 161 | 0.73071333 | 0.80561017 | 221.73601 | 0.041 |
| S2 | C1 | 8.1985586 | 0.088356875 | 0.081819554 | 0.1622 | 0.0873 | 0.2882 | 0.7175 | 1017 1020 1060 964 946 996 1020 1009 951 1017 | 0.84666148 | 1.0138812 | 314.5407 | 0.0378 |

All exact quota and hardness-distinctness checks pass, but all three replicates fail the pre-registered C1 OT ≤ 1.20×E1 tolerance. Constraints are not relaxed and training is prohibited.

Formal decision: `P6C1_JOINT_FEASIBILITY_NOT_SUPPORTED`

Gate alias used in Step 7: `P6C1_JOINT_MATCHING_FEASIBILITY_NOT_SUPPORTED`
