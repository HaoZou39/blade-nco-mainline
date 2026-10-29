# Mechanism Hypothesis Assessment

- `H_A_TOO_EASY: SUPPORTED`
- `H_B_REDUNDANCY: SUPPORTED_PROXY_ONLY`
- `H_C_POOR_TRANSFER: SUPPORTED`

H_A is supported by consistently lower E1 M4 loss and tail mass. H_B has structural-proxy support, but learning-signal redundancy is not directly established. H_C is supported as distribution-specific poor transfer: E1 fits its own population, worsens CE on E0, and regresses on Original70 in all replicates; it is not broad negative transfer because several synthetic benchmarks improve.

Evidence levels: exact M4/M5 CE and frozen greedy gaps are direct observations; F40 NN, lineage, and operator concentration are structural proxies; none identifies a causal mechanism. Own-population CE is in-sample. Cross-population CE is not fully independent because frozen populations can overlap.

## Required Questions

1. **Is E1 easier for M4?** Yes. Across evolution replicates, mean CE is 0.06144 versus 0.08911 for E0 (-31.0%).
2. **Did E1 lose the high-hardness tail?** Yes. Pooled-P90 mass is 3.663% versus 16.337%; pooled-P95 mass is 1.430% versus 8.570%. All three directions agree.
3. **How does Top5 distance relate to hardness?** The pooled nearest-reference Spearman rho is 0.303, but it is arm-confounded. Within E1 it is -0.025/-0.042/-0.015 across Original/S1/S2 (essentially absent); within E0 it is about +0.10. The population shift, not a stable within-E1 monotone distance rule, explains the easier E1 set.
4. **Do both tuned models fit their own population?** Yes: mean own CE gain is 0.01512 for E0-M5 and 0.01580 for E1-M5; every replicate is positive.
5. **Does E1-M5 improve only E1?** In this two-population test, yes: own E1 gain is +0.01580, while E1-M5 changes E0 by -0.01540 (negative means worse).
6. **Cross-population CE?** E0-M5 transfers weakly to E1 (+0.00154); E1-M5 regresses on E0 (-0.01540). Only 63–73/10,000 instances overlap within each pair, though ancestry and generation are shared.
7. **Negative transfer versus M4?** Distribution-specific: E1-M5 regresses on Original70 by +0.376 pp on average and on both TSPLIB48 (+0.423) and TSPLIB22 (+0.274), while improving Uniform100, GM100, TSPLIB100, and on average GM500.
8. **Largest benchmark effects?** The largest improvement is GM100 (-0.918 pp); the clearest adverse effect is Original70 (+0.376 pp), including its 48/22 split.
9. **Is Original70 regression broad or tail-driven?** Both features appear: 49.0% of instances regress on average, so it is not only a handful, but the worst 10% account for 64.6% of total positive gap change. pr152/pr107 are recurrent extremes.
10. **Did E1 lose Full-F40 coverage?** Not as global marginal-variance collapse: the median E1/E0 non-Top5 SD ratio is 0.994, with only two non-Top5 features below 0.8 per replicate. However, calibrated 40-D NN medians are lower in all three E1 populations, so local multivariate clustering increased.
11. **Lineage/operator concentration?** Yes. Effective lineages fall from 482.7 to 220.9; maximum lineage share rises from 1.47% to 4.17%; grid mutation share rises from 9.01% to 21.37%.
12. **Direct evidence of training-signal redundancy?** No. Structural/lineage proxies support H_B, but no gradient-direction or per-instance gradient logs exist: `TRAINING_SIGNAL_REDUNDANCY_NOT_DIRECTLY_ESTABLISHED`.
13. **Do benchmark CE and greedy gap agree?** On the only three datasets with frozen expert tours (Uniform100, GM100, TSPLIB100), E1 CE and greedy gap both improve in all nine replicate×dataset comparisons. Original70/GM500 CE is unavailable, so consistency cannot be tested where the key real-data regression occurs.
14. **Best-supported failure mechanism?** The joint H_A+H_C account: exact Top5 selection chose a much easier population, E1-M5 learned it, but that adaptation transferred negatively to E0 and Original70. H_B is a secondary proxy-only contributor.
15. **Missing key evidence?** Verified expert tours for Original70/GM500, gradient-similarity diagnostics, and a causal intervention that changes hardness exposure while holding the Top5 target quota approximately fixed.
16. **Single next hypothesis?** Holding the 48-reference Top5 quota fixed while restoring the E0 M4-hardness-decile profile will improve E0/Original70 transfer without erasing the synthetic gains.

## Overall Top5/hardness correlations

| group | variable | spearman_rho | p_value_instance_descriptive | n |
|---|---|---|---|---|
| all | ot_assigned_squared_cost | 0.21459168 | 0 | 60000 |
| all | nearest_reference_distance | 0.30256756 | 0 | 60000 |
| all | phi__spn_sum_of_lowest_edge_values | -0.37460608 | 0 | 60000 |
| all | phi__spn_distance_min | -0.33374029 | 0 | 60000 |
| all | phi__spn_distance_varcoeff | 0.33089801 | 0 | 60000 |
| all | phi__spn_nng_3_strong_components_varcoeff | -0.03133325 | 1.6314692e-14 | 60000 |
| all | phi__spn_mst_dists_sd | 0.3320086 | 0 | 60000 |

## Existing-tour benchmark CE versus greedy gap

| arm | dataset | mean_ce_gain | mean_gap_delta |
|---|---|---|---|
| E0 | GaussianMixture100 | 0.020243847 | -0.70585056 |
| E0 | TSPLIB100 | 0.0035615446 | -0.081889522 |
| E0 | Uniform100 | 0.0011061642 | -0.0028846499 |
| E1 | GaussianMixture100 | 0.011512826 | -0.91773086 |
| E1 | TSPLIB100 | 0.002576862 | -0.060662318 |
| E1 | Uniform100 | 0.0024684664 | -0.023832015 |
