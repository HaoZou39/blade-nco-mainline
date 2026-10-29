# P6-D2 Final Research Conclusion

`H1_STRUCTURAL_REPLACEMENT: SUPPORTED`
`H2_HGB_STRUCTURE_BIAS: SUPPORTED`
`H3_REDUNDANCY: SUPPORTED_PROXY_ONLY`
`H4_TRANSFER_ASSOCIATION: SUPPORTED_EXPLORATORY`
`H5_ACTIONABLE_DESIGN_HYPOTHESIS: IDENTIFIED`

Exact niche balance passes for all replicates. There are 13 stable Full-F40 effects (Top5=2, remaining F35=11) at |median SMD|≥0.10 with 3/3 direction, and 1 stable tour effect(s) at the same threshold. The three-replicate TSPLIB48-proximity versus Original70-gap Spearman is 1.0000; with n=3 this is exploratory, not causal.

## Required 20-question summary

1. **B/C shared membership.** Original: Shared=6,348, B-only=C-only=3,652, replacement=36.52%, Jaccard=0.4650; S1: Shared=6,324, B-only=C-only=3,676, replacement=36.76%, Jaccard=0.4624; S2: Shared=6,320, B-only=C-only=3,680, replacement=36.80%, Jaccard=0.4620

2. **B-only/C-only counts.** They are exactly equal within every replicate and every niche: Original 3,652 each, S1 3,676 each, S2 3,680 each.

3. **Replacement niches.** Original: 473 replacement niches; largest by count 571(65), 287(60), 851(55), 654(53), 826(52); S1: 434 replacement niches; largest by count 165(58), 287(57), 654(57), 571(52), 851(52); S2: 462 replacement niches; largest by count 654(66), 287(58), 851(57), 165(53), 424(51). All replacements obey frozen per-niche quotas.

4. **Is C-only truly harder?** Yes, descriptively and under both instance and lineage-root bootstrap: Original: 0.074663→0.084073, Δ=+0.009411; instance 95% CI [0.007482, 0.011352], root-aware [0.006270, 0.012670]; S1: 0.075478→0.085526, Δ=+0.010047; instance 95% CI [0.008078, 0.011989], root-aware [0.006727, 0.013621]; S2: 0.074584→0.083540, Δ=+0.008956; instance 95% CI [0.006959, 0.010864], root-aware [0.005734, 0.012208]. C-only also has higher global-P90 fractions in all three replicates.

5. **Where does the hardness gain occur?** It is distributed across hundreds of niches, not dominated by a few: Original: 291/473 niches have positive mean gain; top-10 explain 20.97%; leading IDs 771, 165, 409, 209, 851; S1: 271/434 niches have positive mean gain; top-10 explain 22.42%; leading IDs 851, 654, 287, 165, 374; S2: 271/462 niches have positive mean gain; top-10 explain 24.82%; leading IDs 654, 827, 220, 970, 165.

6. **Largest Full-F40 differences.** The strongest stable shifts are higher `mst_dists_span`, `mst_dists_skew`, and `mst_depth_varcoeff`, lower `angles_cos_var`, `fraction_shorter_mean_distance`, and `angles_var`, plus higher `mst_dists_sd`. This points to altered MST/local-distance and angular geometry beyond the QD quotas.

7. **Cross-replicate consistency.** 13 effects have identical direction in Original/S1/S2 at |median SMD|≥0.10; most (11) are outside Top5. The ranked table is below.

| feature | feature_group | is_top5 | median_smd | direction_consistency |
|---|---|---|---|---|
| spn_mst_dists_span | minimum_spanning_tree | False | 0.3046819 | 1 |
| spn_mst_dists_skew | minimum_spanning_tree | False | 0.2385098 | 1 |
| spn_angles_cos_var | angular_spatial_distribution | False | -0.2118092 | 1 |
| spn_fraction_shorter_mean_distance | pairwise_distance | False | -0.2115565 | 1 |
| spn_mst_depth_varcoeff | minimum_spanning_tree | False | 0.2104322 | 1 |
| spn_angles_var | angular_spatial_distribution | False | -0.2026251 | 1 |
| spn_mst_dists_sd | minimum_spanning_tree | True | 0.1554902 | 1 |
| spn_centroid_sd | spatial_density_hull_outlier | False | -0.1431468 | 1 |
| spn_nng_3_strong_components_skew | nearest_neighbor_graph | False | 0.131349 | 1 |
| spn_nearest_neighbour_span | nearest_neighbor_geometry | False | 0.1272401 | 1 |

8. **Operator/lineage preference.** C systematically has fewer effective lineages than B, while operator composition shifts are real but no single operator dominates all seeds: Original: axis_projection_mutation -2.82%, grid_mutation +2.74%, compression_mutation +2.08%; S1: linear_projection_mutation -3.70%, implosion_mutation +2.83%, rotation_mutation -2.64%; S2: compression_mutation +3.48%, linear_projection_mutation -2.96%, axis_projection_mutation -2.88%. Thus the evidence supports composition bias, not one universal causal operator.

9. **Local structural redundancy.** Proxy evidence supports modestly higher concentration in C: lower full-population F40-NN distance, higher near-duplicate proxy, fewer effective lineages and slightly larger largest-lineage share in all three replicates. Original: full C/B median F40-NN 3.5896/3.6116, near-duplicate proxy 0.78%/0.61%, effective lineages 432.3/461.4; C-only/B-only effective lineages 374.1/424.9; S1: full C/B median F40-NN 3.5850/3.6221, near-duplicate proxy 0.56%/0.36%, effective lineages 436.6/464.5; C-only/B-only effective lineages 360.0/409.3; S2: full C/B median F40-NN 3.5859/3.6120, near-duplicate proxy 0.68%/0.26%, effective lineages 439.7/460.5; C-only/B-only effective lineages 374.7/413.7. Global feature spread is not reduced, so this is local/lineage concentration—not proven gradient redundancy.

10. **Optimal-tour edge patterns.** Yes, modestly. C-only has higher endpoint-neighbor rank, lower MST-edge overlap and kNN10-edge fraction, and higher tour-edge CV. Only `endpoint_neighbor_rank_mean` exceeds |median SMD|=0.10 consistently; the other shifts are smaller.

| feature | median | consistent_direction |
|---|---|---|
| endpoint_neighbor_rank_mean | 0.1590209 | True |
| tour_mst_edge_overlap_fraction | -0.09870819 | True |
| endpoint_knn10_fraction | -0.09341857 | True |
| tour_edge_cv | 0.0914825 | True |
| tour_total_over_pairwise_median | -0.06864409 | True |
| tour_edge_p95 | 0.06573684 | True |
| tour_edge_entropy_12bin | -0.04476352 | True |

11. **Tour-pattern coverage.** No broad systematic coverage collapse is established. Most tour effects are small, long-edge fraction is inconsistent, and only one of 15 tour features crosses the stable 0.10 threshold. B may retain somewhat more local/MST-like decisions, but this is a distribution shift, not proof of higher useful tour diversity.

12. **Exact B/C Top5 OT difference.** C is slightly farther from the TSPLIB48 Top5 target than B in all three replicates despite both being much better than A: Original: B=14.477987, C=14.794541, C−B=+0.316554; S1: B=14.599998, C=14.892344, C−B=+0.292346; S2: B=14.577231, C=14.813467, C−B=+0.236236. Therefore C's CE gain over B did not come from superior Top5 matching.

13. **TSPLIB48 projection clipping.** 70.8% of references hit at least one QD4 boundary and only 29 CVT niches are directly occupied. Axis diagnostics: spn_sum_of_lowest_edge_values: low=0.0%, high=47.9%, ρ(nodes)=+0.531; spn_distance_varcoeff: low=4.2%, high=2.1%, ρ(nodes)=+0.050; spn_mst_depth_mean: low=4.2%, high=37.5%, ρ(nodes)=+0.827; spn_mst_dists_median: low=29.2%, high=6.2%, ρ(nodes)=-0.762. This saturation makes QD4 niche identity a coarse target proxy and limits fine-grained cross-size interpretation.

14. **Why might B beat C for Original?** C is harder but more locally/lineage concentrated, its Top5 OT is worse by 0.317, and its mean TSPLIB48 F40-NN distance is farther by +0.0528. These observations are consistent with B's broader random within-niche sample transferring better, but do not identify the cause.

15. **Why might C beat B for S2?** In S2, C is closer to TSPLIB48 in the complete cross-size F40 representation by -0.0899 and beats B on Original70 by -0.0821 pp. This seed-specific alignment is a plausible association; C is still worse on TSPLIB22 by +0.0207 pp, so it is not a general C advantage.

16. **Where are benchmark differences concentrated?** Positive C regressions are mostly tail-heavy: the worst 10% contribute 46.1%–89.8% of positive regression mass, and at least 60% in 20/21 reported benchmark/replicate cells. Mean C−B by dataset is below. The audit can identify dataset/tail concentration, but not a reliable finer instance taxonomy from three seeds.

| replicate | dataset | mean_gap_C_minus_B | worst_decile_share_of_C_regression |
|---|---|---|---|
| Original | GaussianMixture100 | -0.1430885 | 0.75854 |
| Original | GaussianMixture500 | 0.006212465 | 0.6842185 |
| Original | OriginalTSPLIB70 | 0.23303 | 0.6231538 |
| Original | TSPLIB100 | 0.0156927 | 0.8227008 |
| Original | TSPLIB22 | -0.1094355 | 0.4607613 |
| Original | TSPLIB48 | 0.3899934 | 0.7739351 |
| Original | Uniform100 | -0.002308227 | 0.7315792 |
| S1 | GaussianMixture100 | 1.210672 | 0.7710524 |
| S1 | GaussianMixture500 | 0.7867073 | 0.6945097 |
| S1 | OriginalTSPLIB70 | 0.03070143 | 0.6849094 |
| S1 | TSPLIB100 | 0.1041157 | 0.8254122 |
| S1 | TSPLIB22 | -0.03515788 | 0.7266548 |
| S1 | TSPLIB48 | 0.06088695 | 0.7147879 |
| S1 | Uniform100 | 0.03222703 | 0.6804781 |
| S2 | GaussianMixture100 | 0.1112393 | 0.8078549 |
| S2 | GaussianMixture500 | 0.2046107 | 0.6615787 |
| S2 | OriginalTSPLIB70 | -0.08205985 | 0.754777 |
| S2 | TSPLIB100 | 0.02397175 | 0.8379384 |
| S2 | TSPLIB22 | 0.02065941 | 0.6080693 |
| S2 | TSPLIB48 | -0.1291395 | 0.8976544 |
| S2 | Uniform100 | -0.002221237 | 0.7346588 |

17. **CE transfer.** Each model fits its own replacement subset better: B has lower CE on B-only and C lower CE on C-only. These subsets overlap the corresponding training set, so this is in-sample specialization rather than independent transfer evidence. It supports a composition shift but does not explain benchmark causally.

| replicate | group | model | n | mean_ce | mean_gain_M4_minus_model |
|---|---|---|---|---|---|
| Original | B_only | B | 3652 | 0.0606992 | 0.01396348 |
| Original | B_only | C | 3652 | 0.0715832 | 0.003079469 |
| Original | C_only | B | 3652 | 0.08229018 | 0.001783003 |
| Original | C_only | C | 3652 | 0.06953819 | 0.01453499 |
| S1 | B_only | B | 3676 | 0.06249529 | 0.0129829 |
| S1 | B_only | C | 3676 | 0.07359975 | 0.001878449 |
| S1 | C_only | B | 3676 | 0.08293185 | 0.002593798 |
| S1 | C_only | C | 3676 | 0.07221012 | 0.01331553 |
| S2 | B_only | B | 3680 | 0.06053919 | 0.01404468 |
| S2 | B_only | C | 3680 | 0.0719507 | 0.00263317 |
| S2 | C_only | B | 3680 | 0.08013379 | 0.003406308 |
| S2 | C_only | C | 3680 | 0.07067364 | 0.01286646 |

18. **Single actionable direction.** Yes: replace deterministic within-niche top-HGB with a frozen rank-stratified rule that retains a fixed high-HGB fraction and samples the remainder uniformly, keeping λ, quotas, pool, labels and training budget unchanged.

19. **Evidence status.** All mechanism findings are observational associations from frozen selections plus post-hoc deterministic features. No individual instance value, negative-transfer cause, learning-gradient redundancy, or unique F40 causal mechanism is established.

20. **Proceed to controlled validation?** Yes, only as one preregistered minimal comparison of B-random vs C-top-HGB vs rank-stratified selection using existing labels. Do not tune it on Original70 and do not infer that it must improve gap.

## If only one part of the fifth-cycle selector may change

Replace deterministic within-niche top-HGB selection with one frozen rank-stratified sampling rule inside the unchanged lambda=0.5 quotas: preserve a fixed high-HGB fraction and sample the remainder uniformly within each niche. Compare it only against B-random and C-top-HGB using the same labelled pool, M4 initialization, seed and budget. This changes one selection factor and tests whether the observed HGB-linked composition/concentration, rather than target quotas, causes unstable transfer.

The stable evidence motivating this is the 13-direction-consistent Full-F40 shifts, C's three-seed local/lineage concentration proxies, and unstable B/C benchmark transfer despite C's consistently higher true CE. The modification is intended to reduce within-niche selection concentration while preserving hardness pressure. This audit cannot prove that concentration caused the gap changes, nor that rank stratification will improve training value; only the proposed controlled training can test that.

The proposed comparison is not run here. P6-D3, new training, mutation, evolution, HGB fitting and Concorde are not started.

`BLADE_P6D2_SELECTED_POPULATION_STRUCTURE_AUDIT_COMPLETE`
