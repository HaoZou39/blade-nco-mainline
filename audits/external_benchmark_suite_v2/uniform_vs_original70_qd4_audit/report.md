# Uniform100 vs Original70 — QD4 Structural Comparison

Read-only audit using frozen official benchmarks, formal QD4 extractor, Normalization A, and frozen CVT-1000.

## Data
- **Uniform100**: N=10000 from `assets/evaluation/uniform100`
- **Original70**: N=70 from `assets/evaluation/original_tsplib70`, aspect-ratio-preserving global minmax preprocessing

## Feature mapping
| display | code |
|---|---|
| local | `spn_sum_of_lowest_edge_values` |
| global | `spn_distance_varcoeff` |
| topology | `spn_mst_depth_mean` |
| scale | `spn_mst_dists_median` |

## Quantitative highlights

### Feature / CVT
| dataset    |     N |   occupied_niches |   occupancy_rate |   effective_niches |   normalized_entropy |   max_instances_per_niche |   raw_entropy_nats |
|:-----------|------:|------------------:|-----------------:|-------------------:|---------------------:|--------------------------:|-------------------:|
| Uniform100 | 10000 |                51 |            0.051 |            30.622  |             0.495344 |                       561 |            3.42172 |
| Original70 |    70 |                33 |            0.033 |            20.3036 |             0.708674 |                        14 |            3.0108  |

### Pairwise QD4 spread (normalized 4D)
| dataset    | mode           |   pairwise_mean |   pairwise_median |   pairwise_p90 |   pairwise_std |
|:-----------|:---------------|----------------:|------------------:|---------------:|---------------:|
| Uniform100 | subsample_2000 |        0.263269 |          0.220678 |       0.505734 |       0.166894 |
| Original70 | full           |        0.759177 |          0.739907 |       1.35513  |       0.401028 |

### Distribution distances
- Energy Distance = **1.0536**
- MMD (RBF, bandwidth=0.2569) = **0.8166**
- 1D Wasserstein (local/global/topology/scale): local=0.144, global=0.365, topology=0.464, scale=0.707

### Niche overlap
- Uniform only: 44
- Overlap: 7
- Original70 only: 26
- Coverage(Original70 niches by Uniform) = 0.212
- Coverage(Uniform niches by Original70) = 0.137

### PCA
- PC1 = 80.2%, PC2 = 14.7% (sum 94.9%)

## Answers

1. **Original70 与 Uniform100 是否明显不同？**  
   **是。** Energy=1.054；Original70 仅 21% niches 被 Uniform 覆盖；多数 feature 的 Wasserstein 距离非零。Verdict: `CLEAR_STRUCTURAL_DIFFERENCE`.

2. **Original70 是否更 heterogeneous？**  
   **相对 Uniform，按实例尺度更 heterogeneous。** Original70 pairwise P90=1.355 vs Uniform 0.506；normalized entropy 0.709 vs 0.495。绝对 occupied niches 因 N=70 较小（33 vs 51），但 per-instance diversity / entropy 更高。`ORIGINAL70_HETEROGENEITY = HIGH`。

3. **occupied / effective niches 差距？**  
   - occupied: Uniform **51** vs Original70 **33** (Δ=18)  
   - effective: Uniform **30.6** vs Original70 **20.3**

4. **PCA 关系？** 明显分离（约 11% Original70 点落在 Uniform PCA 轴对齐包围盒内）。

5. **PPT 主文推荐图：**  
   - **Fig 2**（PCA map）— 一眼看位置与重叠  
   - **Fig 1**（4-panel features）— 定量分布差异  
   - **Fig 4**（instance montage）— 直观形态对比  
   - 补充：**Fig 3** 或 **Fig 5** 讲 CVT occupancy / overlap

## Verdict
- `UNIFORM_VS_ORIGINAL70_VERDICT = CLEAR_STRUCTURAL_DIFFERENCE`
- `ORIGINAL70_HETEROGENEITY = HIGH`

## 可直接复制到汇报的结论

> 在正式冻结的 QD4 + CVT-1000 空间中，Original70 与 Uniform100 **结构分布明显不同**（Energy Distance ≈ 1.05；Original70 仅约 21% 的 niches 被 Uniform 覆盖）。Uniform100 实例多但结构集中（occupied=51，effective≈31）；Original70 规模虽小（N=70），但 **内部异质性更高**（pairwise QD4 P90≈1.36，normalized entropy=0.71），形态覆盖更广。PPT 建议主图用 PCA 结构图 + 4 维 feature 对比 + 实例 montage。

## Integrity
- [x] frozen official files only
- [x] no benchmark regeneration
- [x] formal QD4 extractor + normalization + CVT-1000
- [x] Original70 aspect-ratio-preserving preprocess
- [x] no BQ/GP training, no Concorde, no mainline changes
