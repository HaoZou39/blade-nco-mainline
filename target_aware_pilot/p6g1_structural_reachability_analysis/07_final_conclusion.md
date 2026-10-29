# P6-G1 Final Conclusion

## Decision labels

`P6_G1_INITIAL_POPULATION_BIAS = SUPPORTED`
`P6_G1_MUTATION_STRUCTURAL_EXPANSION = LIMITED`
`P6_G1_OPERATOR_LIMITATION = PARTIAL`
`P6_G1_PARENT_SELECTION_LIMITATION = PARTIAL`
`P6_G1_EVOREAL_STRUCTURAL_DIFFERENCE = CONFIRMED`
`P6_G1_CROSS_SIZE_CONFOUNDING = PARTIAL`
`P6_G1_ANALYSIS_STATUS = COMPLETE`

## Q1 — EvoReal reaches what different structure?

Both frozen representations show that EvoReal is not merely a denser copy of P03/FOGA. Its marginal F40 ranges are broader or shifted most strongly in the following dimensions:

| feature | mean_wasserstein_over_evoreal_iqr | max_abs_smd | mean_iqr_ratio_vs_evoreal |
|---|---|---|---|
| spn_nearest_neighbour_sd | 1.1727 | 2.1395 | 0.61589 |
| spn_angles_max | 1.086 | 0.88516 | 2.4148 |
| spn_hull_dists_median | 0.94219 | 1.4124 | 1.4436 |
| spn_hull_dists_max | 0.72203 | 1.5863 | 0.60818 |
| spn_angles_cos_median | 0.7005 | 1.8453 | 0.52161 |
| spn_hull_edges_median | 0.6862 | 1.043 | 0.91506 |
| spn_nearest_neighbour_span | 0.67652 | 0.97385 | 0.67878 |
| spn_centroid_span | 0.65937 | 1.2136 | 0.73743 |

For pr107/pr144, the nearest-instance distance divided by the frozen radius is:

| reference_id | EvoReal50k | Original_E0_G2_RAW | Original_E1_G2_RAW | Original_G1_RAW | P03 | S1_E0_G2_RAW | S1_E1_G2_RAW | S1_G1_RAW | S2_E0_G2_RAW | S2_E1_G2_RAW | S2_G1_RAW |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pr107 | 0.020369 | 5.6571 | 5.044 | 5.3949 | 5.7169 | 6.3594 | 5.5268 | 5.5792 | 6.5753 | 5.1859 | 5.2397 |
| pr144 | 0.061609 | 1.8042 | 1.0594 | 1.1185 | 2.0658 | 1.8068 | 1.5309 | 1.6298 | 1.728 | 1.0983 | 1.6851 |

The geometry gallery makes the missing modes interpretable. `pr107` is a highly regular, repeated two-sided column/grid layout; the P03/FOGA nearest cases remain irregular dispersed clouds, while EvoReal produces a repeated line/grid construction. `pr144` is a repeated block/cluster motif; the closest FOGA case contains only partial local clustering, while EvoReal reproduces multiple separated blocks. These patterns agree with the largest F40 shifts in nearest-neighbor dispersion, angular regularity, hull-distance/edge summaries and centroid span.

In frozen FFT_APPROX–NN space EvoReal covers 40/48, while all historical FOGA RAW reaches at most 25/48. The difference persists for the 75–150 subset (17/17 versus 15/17), although cross-size effects remain material for the other 31 references. Generator-specific rows in `02_structural_distribution.csv` show that the broader support is a mixture of different EvoReal generator regimes, not one homogeneous cloud.

The exact FFT–NN-nearest candidate is not necessarily the F40-nearest candidate: even EvoReal's pr107/pr144 FFT matches retain sizable F40 RMS differences. Thus the **population-level EvoReal structural difference is confirmed by both representations**, while individual nearest-case correspondence remains representation-dependent.

## Q2 — Did FOGA move toward the missing regions?

Yes, individual offspring from multiple operators have positive distance improvements toward pr107/pr144, and the movement distribution has a real long tail. But the best historical RAW remains at 5.04 radii for pr107 and 1.06 radii for pr144, versus 0.02 and 0.06 for EvoReal. Positive one-step moves were generally too small, started from biased parents, or moved along incomplete dimensions. Grid mutation produced the closest observed pr107 child; implosion produced the closest pr144 child, showing that no single operator uniformly dominates both targets. For all 15 references, `04_unreachable_reference_analysis.csv` gives the continuous distance/radius and the F40 discrepancy of the exact FFT–NN-nearest candidate. This supports **limited expansion**, not the stronger claim that FOGA can only make local moves.

## Q3 — What should be changed first?

The strongest current evidence points first to the parent/support supplied to mutation, jointly with testing operator scale/direction. P03 already covers only 23/48, and two generations expand the maximum by only two neighborhoods. Parent selection is a secondary, branch-level limitation: it changes G2 density and can lose rare reached regions, but E0∪E1 did not explain the globally unreachable 15. The structure representation should remain a sensitivity check—not be refit to erase the gap—because both F40 and FFT–NN detect an EvoReal difference, while nearest-instance details can still be representation-dependent.

## Q4 — Does this determine evolution depth?

No. Static two-generation histories show small coverage increments and branch-dependent density, but cannot identify an optimal depth. They justify one focused future test: under a fixed evaluation and mutation budget, does seeding parents from the missing EvoReal-like support increase *new-neighborhood arrival rate* more than increasing operator magnitude/depth from P03? That requires a prospective controlled experiment; it was not run here.

## Limits

- Operator comparisons are observational because parent distributions and repairs differ.
- FFT is explicitly FFT_APPROX; it is not claimed as an exact official EvoReal FFT reproduction.
- No F40 neighborhood threshold was invented; F40 evidence is continuous/marginal/PCA only.
- TSPLIB48 is variable-size and target-informed. The 75–150 subset reduces but cannot eliminate cross-size and small-reference-sample uncertainty.
- Structural proximity is not interpreted as BQ training value.

## 中文研究总结（≤500字）

已确认：P03仅覆盖23/48；三种历史种子下，G1/G2全部RAW最多25/48，而EvoReal50k为40/48。控制到75–150节点后仍为15/17对17/17。正式F40与冻结FFT_APPROX–NN均显示EvoReal具有P03/FOGA缺少或明显不足的结构区间。真实父子配对表明FOGA并非完全局部：多种算子存在长尾结构移动，也产生朝pr107/pr144靠近的offspring，但历史最佳距离仍未进入冻结邻域。具有证据但未确立因果：初始父代支持偏窄与算子方向/幅度共同限制可达性；父代选择会改变G2分布，却不是15个全局不可达区域的主因。跨规模与表征依赖仍是局限。下一步最值得单独验证的问题是：固定预算下，补充缺失结构支持的父代，是否比单纯增加变异幅度或演化深度更能提高新邻域到达率。

`BLADE_P6G1_STRUCTURAL_REACHABILITY_ANALYSIS_COMPLETE`
