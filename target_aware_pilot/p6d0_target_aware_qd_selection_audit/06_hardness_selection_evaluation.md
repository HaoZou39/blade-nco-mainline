# Hardness Selection Evaluation

All memberships were frozen before reading true M4 CE for evaluation; only D used it during selection and is not deployable. C versus B isolates HGB ranking under identical quotas. D versus C estimates surrogate regret. P90/P95 use frozen P6-C1 global thresholds.

| replicate | selection | mean_true_m4_ce | median_true_m4_ce | frozen_p90_tail_fraction | frozen_p95_tail_fraction | mean_predicted_m4_ce | top10_enrichment_over_pool | overlap_with_oracle_D | effective_lineages_shannon | largest_lineage_share |
|---|---|---|---|---|---|---|---|---|---|---|
| Original | A | 0.08865202 | 0.08122677 | 0.1605 | 0.0818 | 0.07760106 | 1.616 | 0.5659 | 478.5243 | 0.0181 |
| Original | B | 0.07664841 | 0.06927891 | 0.1062 | 0.0517 | 0.06549916 | 1.07 | 0.6354 | 461.3892 | 0.0253 |
| Original | C | 0.08008513 | 0.07199705 | 0.1256 | 0.0652 | 0.07155319 | 1.262 | 0.6685 | 432.3064 | 0.0269 |
| Original | D | 0.09853585 | 0.09560217 | 0.198 | 0.0977 | 0.06677372 | 1.993 | 1 | 445.9471 | 0.0246 |
| Original | E | 0.0705258 | 0.06238325 | 0.0825 | 0.0431 | 0.06244625 | 0.828 | 0.6315 | 344.5292 | 0.0353 |
| S1 | A | 0.08965873 | 0.08261725 | 0.1641 | 0.0852 | 0.07749843 | 1.606 | 0.5637 | 477.0706 | 0.0144 |
| S1 | B | 0.07722042 | 0.06987015 | 0.1091 | 0.0552 | 0.0653899 | 1.074 | 0.6338 | 464.5053 | 0.0242 |
| S1 | C | 0.08091386 | 0.07299889 | 0.1285 | 0.0672 | 0.07147264 | 1.259 | 0.6683 | 436.6398 | 0.0258 |
| S1 | D | 0.09946528 | 0.09650028 | 0.2038 | 0.1005 | 0.06670326 | 1.994 | 1 | 442.796 | 0.0242 |
| S1 | E | 0.07125976 | 0.06294759 | 0.0888 | 0.0448 | 0.06237641 | 0.869 | 0.6305 | 344.6952 | 0.036 |
| S2 | A | 0.08867672 | 0.08176129 | 0.1611 | 0.0864 | 0.07713983 | 1.629 | 0.5663 | 478.8685 | 0.0146 |
| S2 | B | 0.07666363 | 0.06851073 | 0.1084 | 0.0571 | 0.06514983 | 1.093 | 0.633 | 460.5387 | 0.0249 |
| S2 | C | 0.07995952 | 0.07169989 | 0.1254 | 0.0678 | 0.07115424 | 1.267 | 0.6684 | 439.6741 | 0.0259 |
| S2 | D | 0.09877346 | 0.0954522 | 0.1972 | 0.1014 | 0.06641653 | 1.993 | 1 | 439.6319 | 0.0264 |
| S2 | E | 0.07066555 | 0.06192672 | 0.0854 | 0.0457 | 0.06202922 | 0.859 | 0.6316 | 344.2845 | 0.0368 |

## Within-niche ranking

Niches require at least 20 candidates and nondegenerate true/predicted scores. This prevents E0/E1 pooled group shifts from substituting for within-niche evidence.

| replicate | region | analyzable_niches | candidate_n | median_hgb_true_spearman | weighted_mean_hgb_true_spearman | positive_rho_fraction | median_C_minus_B_true_ce | median_D_minus_C_true_ce | median_C_p90_enrichment |
|---|---|---|---|---|---|---|---|---|---|
| Original | all | 228 | 15805 | 0.152755 | 0.1476707 | 0.7850877 | 0.003343299 | 0.02447443 | 1.194292 |
| Original | target_enriched | 74 | 6004 | 0.06855094 | 0.07996737 | 0.7297297 | 0.0001014927 | 0.01774641 | 1.087912 |
| S1 | all | 223 | 15810 | 0.1484802 | 0.1423664 | 0.8206278 | 0.003066376 | 0.02494961 | 1.121364 |
| S1 | target_enriched | 71 | 5908 | 0.07201037 | 0.05458777 | 0.7183099 | 0 | 0.01801121 | 1 |
| S2 | all | 223 | 15714 | 0.1265569 | 0.127752 | 0.7713004 | 0.003560659 | 0.02663931 | 1.167949 |
| S2 | target_enriched | 74 | 5937 | 0.04536755 | 0.05588562 | 0.6486486 | 0 | 0.01912982 | 1.043478 |
