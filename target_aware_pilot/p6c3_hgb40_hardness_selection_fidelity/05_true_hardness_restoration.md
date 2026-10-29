# True Hardness Restoration

Frozen C0 and Oracle-C1 values are reproduced below. Surrogate-C1 is absent because its frozen predicted-decile constraints fail the necessary supply condition. Consequently mean-hardness recovery, P90/P95 recovery, true-decile TV/EMD, membership overlap, source fraction, lineage diversity, operator composition, and Full-F40 clustering for Surrogate-C1 are **not estimable**. Treating them as zero would incorrectly conflate infeasibility with an observed selected population.

| replicate | selection | status | n | mean_true_m4_ce | median_true_m4_ce | p90_tail_fraction | p95_tail_fraction | true_decile_tvd_to_E0_target | true_decile_ordinal_emd | ot_mean_cost | effective_lineages | largest_lineage_share | f40_nn_median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Original | C0 | AVAILABLE_FROZEN | 10000 | 0.06168541 | 0.05583777 | 0.0362 | 0.0149 | 0.2747 | 1.8719 | 6.042907 | 225.6093 | 0.0416 | 2.208125 |
| Original | Oracle-C1 | AVAILABLE_FROZEN | 10000 | 0.08804011 | 0.08074905 | 0.1605 | 0.0819 | 0 | 0 | 8.102423 | 323.8829 | 0.0386 | 2.419953 |
| Original | Surrogate-C1 | INFEASIBLE_NOT_SELECTED | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| S1 | C0 | AVAILABLE_FROZEN | 10000 | 0.0615704 | 0.05574899 | 0.0391 | 0.0142 | 0.2974 | 1.9816 | 6.06905 | 216.1678 | 0.0426 | 2.177379 |
| S1 | Oracle-C1 | AVAILABLE_FROZEN | 10000 | 0.08940268 | 0.08298623 | 0.1665 | 0.0844 | 0 | 0 | 8.36322 | 326.2023 | 0.0385 | 2.411921 |
| S1 | Surrogate-C1 | INFEASIBLE_NOT_SELECTED | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| S2 | C0 | AVAILABLE_FROZEN | 10000 | 0.06105017 | 0.05539118 | 0.0344 | 0.0136 | 0.2808 | 1.9185 | 6.057842 | 221.736 | 0.041 | 2.211007 |
| S2 | Oracle-C1 | AVAILABLE_FROZEN | 10000 | 0.08835688 | 0.08181955 | 0.1622 | 0.0873 | 0 | 0 | 8.198559 | 314.5407 | 0.0378 | 2.419775 |
| S2 | Surrogate-C1 | INFEASIBLE_NOT_SELECTED | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
