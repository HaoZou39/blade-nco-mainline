# Robustness and Leakage

Exact coordinate overlap with historical HGB calibration is small, but root overlap remains dominant. Selection behavior on the full frozen replay pool is valid as an offline mechanism audit, while novel-lineage generalization is not established. Strict-root-safe selected subsets are small and their correlations are unstable; therefore HGB target-region utility is labelled LIMITED even when C consistently beats B. Three evolution replicates—not individual candidates—are the primary replication units. Lineage-cluster bootstrap sensitivity is saved separately.

| replicate | selection | exact_train_overlap_n | train_root_overlap_n | strict_root_safe_n | strict_root_safe_true_mean_ce | strict_root_safe_pred_true_spearman |
|---|---|---|---|---|---|---|
| Original | A | 19 | 9923 | 77 | 0.07742965 | 0.3087965 |
| Original | B | 21 | 9866 | 134 | 0.06580594 | 0.2955024 |
| Original | C | 24 | 9877 | 123 | 0.06458328 | 0.281036 |
| Original | D | 17 | 9861 | 139 | 0.08161236 | 0.2647661 |
| Original | E | 23 | 9811 | 189 | 0.05968578 | 0.2276809 |
| S1 | A | 20 | 9867 | 133 | 0.08416262 | 0.3096989 |
| S1 | B | 19 | 9798 | 202 | 0.07041641 | 0.2881828 |
| S1 | C | 21 | 9812 | 188 | 0.07250136 | 0.2777949 |
| S1 | D | 18 | 9809 | 191 | 0.08757456 | 0.3608914 |
| S1 | E | 19 | 9773 | 227 | 0.06484503 | 0.1971945 |
| S2 | A | 17 | 9905 | 95 | 0.07929867 | 0.262346 |
| S2 | B | 18 | 9825 | 175 | 0.06149869 | 0.310609 |
| S2 | C | 17 | 9838 | 162 | 0.06069237 | 0.3471529 |
| S2 | D | 14 | 9836 | 164 | 0.07974332 | 0.2551005 |
| S2 | E | 17 | 9753 | 247 | 0.05560826 | 0.1705852 |

| replicate | comparison | estimate | lineage_bootstrap_ci_low | lineage_bootstrap_ci_high | bootstrap_n | dependency_unit |
|---|---|---|---|---|---|---|
| Original | C_minus_B_mean_CE | 0.003422454 | 0.00242211 | 0.004451377 | 1000 | lineage_root |
| Original | C_minus_A_mean_CE | -0.008550766 | -0.0101339 | -0.007106679 | 1000 | lineage_root |
| Original | C_minus_A_target_relevance | 0.00218552 | 0.001665486 | 0.002812367 | 1000 | lineage_root |
| S1 | C_minus_B_mean_CE | 0.003712903 | 0.002799544 | 0.004693554 | 1000 | lineage_root |
| S1 | C_minus_A_mean_CE | -0.008701369 | -0.01025128 | -0.007075181 | 1000 | lineage_root |
| S1 | C_minus_A_target_relevance | 0.002087411 | 0.001603533 | 0.002637272 | 1000 | lineage_root |
| S2 | C_minus_B_mean_CE | 0.003297672 | 0.002364913 | 0.004237826 | 1000 | lineage_root |
| S2 | C_minus_A_mean_CE | -0.008657008 | -0.01030768 | -0.007193655 | 1000 | lineage_root |
| S2 | C_minus_A_target_relevance | 0.002170597 | 0.001627346 | 0.002785026 | 1000 | lineage_root |
