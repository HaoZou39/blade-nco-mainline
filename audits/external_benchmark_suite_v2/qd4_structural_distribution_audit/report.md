# QD4 Structural Distribution Audit

Independent, read-only analysis of frozen N≈100 PRIMARY benchmarks in the formal BLADE QD4 + CVT-1000 space.

> These labels denote EvoReal generator families rather than strictly pure structural classes.

## Feature definitions (canonical code ↔ display)

| display | canonical code field |
|---|---|
| local | `spn_sum_of_lowest_edge_values` |
| global | `spn_distance_varcoeff` |
| topology | `spn_mst_depth_mean` |
| scale | `spn_mst_dists_median` |

Pipeline: formal `extract_qd4_exact` (N=100) / `extract_qd4` (Original70 variable N) → arcsinh robust transform → Normalization A P1/P99 clip to [0,1] → nearest of frozen CVT-1000 centroids.

## Datasets (frozen official files)

- **Uniform100**: N=10000; source=`assets/evaluation/uniform100/instances/coordinates.npz`; preprocess=`identity (frozen unit-square coordinates)`; extractor=`extract_qd4_exact`
- **GM100**: N=10000; source=`assets/evaluation/gaussian_mixture100/instances/coordinates.npz`; preprocess=`identity (frozen unit-square coordinates)`; extractor=`extract_qd4_exact`
- **TSPLIB100**: N=10000; source=`assets/evaluation/tsplib100/instances/coordinates.npz::model_input_coordinates`; preprocess=`frozen model_input_coordinates (unit square)`; extractor=`extract_qd4_exact`
- **Original70**: N=70; source=`assets/evaluation/original_tsplib70 (EUC_2D gzip + source_manifest_bq83.csv)`; preprocess=`lehd_aspect_ratio_preserving_normalize (global minmax over all xy)`; extractor=`extract_qd4 (variable N)`
- **EvoReal-S1 Gen.**: N=1000; source=`datasets/external_v2/EvoReal_S1_100/coordinates.npy`; preprocess=`identity (frozen [0,1] generator output)`; extractor=`extract_qd4_exact`
- **EvoReal-S2 Gen.**: N=1000; source=`datasets/external_v2/EvoReal_S2_100/coordinates.npy`; preprocess=`identity (frozen [0,1] generator output)`; extractor=`extract_qd4_exact`
- **EvoReal-S3 Gen.**: N=1000; source=`datasets/external_v2/EvoReal_S3_100/coordinates.npy`; preprocess=`identity (frozen [0,1] generator output)`; extractor=`extract_qd4_exact`

GM500 excluded from PRIMARY to avoid confounding size with structure.

## Analysis 1 — Feature marginal summaries

See `tables/02_feature_distribution_summary.csv` and `figures/fig1_qd4_marginal_distributions.(pdf|png)`.

## Analysis 2 — PCA visualization (illustrative only for quantitative claims)

- PC1 explained variance: **84.21%**
- PC2 explained variance: **11.78%**
- PC1+PC2: **95.99%**

All CVT / distance metrics remain in original normalized 4D QD space.

## Analysis 3 — CVT-1000 niche occupancy

| dataset         |     N |   occupied_niches |   occupancy_rate |   mean_instances_per_occupied_niche |   max_instances_in_one_niche |   normalized_niche_entropy |   effective_niches |   raw_entropy_nats |
|:----------------|------:|------------------:|-----------------:|------------------------------------:|-----------------------------:|---------------------------:|-------------------:|-------------------:|
| Uniform100      | 10000 |                51 |            0.051 |                           196.078   |                          561 |                   0.495344 |            30.622  |            3.42172 |
| GM100           | 10000 |               375 |            0.375 |                            26.6667  |                          697 |                   0.671122 |           103.126  |            4.63595 |
| TSPLIB100       | 10000 |               497 |            0.497 |                            20.1207  |                          193 |                   0.78195  |           221.742  |            5.40152 |
| Original70      |    70 |                33 |            0.033 |                             2.12121 |                           14 |                   0.708674 |            20.3036 |            3.0108  |
| EvoReal-S1 Gen. |  1000 |               176 |            0.176 |                             5.68182 |                          140 |                   0.629619 |            77.4207 |            4.34925 |
| EvoReal-S2 Gen. |  1000 |               189 |            0.189 |                             5.29101 |                           40 |                   0.67058  |           102.74   |            4.6322  |
| EvoReal-S3 Gen. |  1000 |               362 |            0.362 |                             2.76243 |                           27 |                   0.791113 |           236.232  |            5.46481 |

## Analysis 4 — Niche overlap (Jaccard + directional Coverage)

Absolute Jaccard is expected to be modest when |instances| << 1000 niches; directional Coverage is primary.

- TSPLIB100 vs Uniform100: Jaccard=0.100, Coverage(TSPLIB100 by Uniform100)=0.101, Coverage(Uniform100 by TSPLIB100)=0.980, Energy=0.8185
- TSPLIB100 vs GM100: Jaccard=0.563, Coverage(TSPLIB100 by GM100)=0.632, Coverage(GM100 by TSPLIB100)=0.837, Energy=0.6883
- TSPLIB100 vs EvoReal-S1 Gen.: Jaccard=0.289, Coverage(TSPLIB100 by EvoReal-S1 Gen.)=0.304, Coverage(EvoReal-S1 Gen. by TSPLIB100)=0.858, Energy=0.6921
- TSPLIB100 vs EvoReal-S2 Gen.: Jaccard=0.335, Coverage(TSPLIB100 by EvoReal-S2 Gen.)=0.346, Coverage(EvoReal-S2 Gen. by TSPLIB100)=0.910, Energy=0.4365
- TSPLIB100 vs EvoReal-S3 Gen.: Jaccard=0.510, Coverage(TSPLIB100 by EvoReal-S3 Gen.)=0.584, Coverage(EvoReal-S3 Gen. by TSPLIB100)=0.801, Energy=0.2406
- Original70 vs Uniform100: Jaccard=0.091, Coverage(Original70 by Uniform100)=0.212, Coverage(Uniform100 by Original70)=0.137, Energy=1.0536
- Original70 vs GM100: Jaccard=0.057, Coverage(Original70 by GM100)=0.667, Coverage(GM100 by Original70)=0.059, Energy=0.9206
- Original70 vs EvoReal-S1 Gen.: Jaccard=0.089, Coverage(Original70 by EvoReal-S1 Gen.)=0.515, Coverage(EvoReal-S1 Gen. by Original70)=0.097, Energy=0.7910
- Original70 vs EvoReal-S2 Gen.: Jaccard=0.110, Coverage(Original70 by EvoReal-S2 Gen.)=0.667, Coverage(EvoReal-S2 Gen. by Original70)=0.116, Energy=0.6474
- Original70 vs EvoReal-S3 Gen.: Jaccard=0.056, Coverage(Original70 by EvoReal-S3 Gen.)=0.636, Coverage(EvoReal-S3 Gen. by Original70)=0.058, Energy=0.6908

### EvoReal generator-family overlap

- EvoReal-S1 Gen. vs EvoReal-S2 Gen.: Jaccard=0.241, Coverage(EvoReal-S1 Gen. by EvoReal-S2 Gen.)=0.403, Coverage(EvoReal-S2 Gen. by EvoReal-S1 Gen.)=0.376
- EvoReal-S1 Gen. vs EvoReal-S3 Gen.: Jaccard=0.234, Coverage(EvoReal-S1 Gen. by EvoReal-S3 Gen.)=0.580, Coverage(EvoReal-S3 Gen. by EvoReal-S1 Gen.)=0.282
- EvoReal-S2 Gen. vs EvoReal-S3 Gen.: Jaccard=0.354, Coverage(EvoReal-S2 Gen. by EvoReal-S3 Gen.)=0.762, Coverage(EvoReal-S3 Gen. by EvoReal-S2 Gen.)=0.398

## Analysis 5 — Structural distances (normalized QD4 4D)

MMD RBF bandwidth (pooled median heuristic): **0.862430**

Energy Distance and MMD for Uniform100 / GM100 / TSPLIB100 used a fixed random subsample of 2000 instances per dataset (seed=0) for computational tractability; niche occupancy, unique coverage, feature summaries, Wasserstein, and all CVT assignments use the full frozen sets. All distances remain in original normalized QD4 4D space (not PCA).

Matrices: `tables/05_pairwise_energy_distance.csv`, `tables/06_pairwise_mmd.csv`, `tables/07_feature_wasserstein.csv` (also mirrored at audit root as `01_`–`08_` CSVs).

## Analysis 6 — TSPLIB / Original70 internal heterogeneity

### TSPLIB100
- occupied niches: 497
- effective niches: 221.74
- pairwise QD4 distance mean/median/P90: 0.7371 / 0.6819 / 1.2624
- feature CV/IQR: {"local": {"CV": 0.4905880882135763, "IQR": 0.4190650464343376, "range": 1.0}, "global": {"CV": 0.6978248874173375, "IQR": 0.5708797269336537, "range": 1.0}, "topology": {"CV": 0.6430441865541104, "IQR": 0.3933475776846744, "range": 1.0}, "scale": {"CV": 0.6371727743428609, "IQR": 0.48610909487518356, "range": 1.0}}
- exploratory k-means silhouettes (not claimed as true classes): [{"k": 2, "silhouette": 0.4796618738750345, "inertia": 1541.766572741594}, {"k": 3, "silhouette": 0.32881123217432395, "inertia": 1233.375689714153}, {"k": 4, "silhouette": 0.33382854507864695, "inertia": 954.0276626047169}, {"k": 5, "silhouette": 0.3328583909320685, "inertia": 796.3916018439463}]

### Original70
- occupied niches: 33
- effective niches: 20.30
- pairwise QD4 distance mean/median/P90: 0.7592 / 0.7399 / 1.3551
- feature CV/IQR: {"local": {"CV": 0.32145089013195854, "IQR": 0.2826146031591231, "range": 0.8176716377440423}, "global": {"CV": 0.706138175465896, "IQR": 0.44156206893087735, "range": 1.0}, "topology": {"CV": 0.4440099290906836, "IQR": 0.42218574539309717, "range": 1.0}, "scale": {"CV": 1.3823432339761974, "IQR": 0.4002379691742104, "range": 1.0}}
- exploratory k-means silhouettes (structural regimes only): [{"k": 2, "silhouette": 0.4801586910379377, "inertia": 14.08621752791571}, {"k": 3, "silhouette": 0.4641502210265027, "inertia": 9.201897125388149}, {"k": 4, "silhouette": 0.4632421760175465, "inertia": 7.240286282970079}, {"k": 5, "silhouette": 0.388599646492571, "inertia": 5.766801207889719}]

## Analysis 7 — Benchmark complementarity

| step            |   added_niches |   cumulative_occupied_niches |   cumulative_occupancy_rate |   delta_from_previous |
|:----------------|---------------:|-----------------------------:|----------------------------:|----------------------:|
| Uniform100      |             51 |                           51 |                       0.051 |                    51 |
| GM100           |            353 |                          404 |                       0.404 |                   353 |
| EvoReal-S1 Gen. |              9 |                          413 |                       0.413 |                     9 |
| EvoReal-S2 Gen. |             38 |                          451 |                       0.451 |                    38 |
| EvoReal-S3 Gen. |             83 |                          534 |                       0.534 |                    83 |
| TSPLIB100       |             85 |                          619 |                       0.619 |                    85 |
| Original70      |              1 |                          620 |                       0.62  |                     1 |

| dataset         |   occupied_niches |   unique_contribution |   unique_fraction_of_own |   marginal_coverage_given_others |   union_all |
|:----------------|------------------:|----------------------:|-------------------------:|---------------------------------:|------------:|
| Uniform100      |                51 |                     0 |               0          |                                0 |         620 |
| GM100           |               375 |                    24 |               0.064      |                               24 |         620 |
| TSPLIB100       |               497 |                    83 |               0.167002   |                               83 |         620 |
| Original70      |                33 |                     1 |               0.030303   |                                1 |         620 |
| EvoReal-S1 Gen. |               176 |                     1 |               0.00568182 |                                1 |         620 |
| EvoReal-S2 Gen. |               189 |                     5 |               0.026455   |                                5 |         620 |
| EvoReal-S3 Gen. |               362 |                    47 |               0.129834   |                               47 |         620 |

## Answers to required questions

**Q1. TSPLIB100 single tight distribution?** No — TSPLIB100 is not a single tight structural distribution.

**Q2. Original70 internal heterogeneity?** Original70 occupies 33 niches (0.033 rate) with effective_niches=20.3 and pairwise QD4 distance P90=1.355.

**Q3. Do TSPLIB / Original70 span Uniform-like, cluster-like, and other regimes?**

- TSPLIB100: occupied=497, effective=221.7; Coverage(TSPLIB100 niches by Uniform)=0.101, by GM100=0.632; unique niches=83
- Original70: occupied=33, effective=20.3; Coverage(Original70 niches by Uniform)=0.212, by GM100=0.667; unique niches=1

Interpretation uses niche overlap with Uniform100 (uniform-like) and GM100 (cluster-like) plus residual unique niches as other structural regimes — not discrete S1/S2/S3 class labels.

**Q4. EvoReal S1/S2/S3 Gen. separation vs overlap?** Partial separation with substantial overlap. S1↔S2 and S1↔S3 Jaccard ≈0.23–0.24; S2↔S3 Jaccard=0.35 with Coverage(S2 by S3)=0.76. Energy distances: S1–S2=1.01 (most separated), S2–S3=0.35 (closest). Labels are generator families, not pure structural classes.

- EvoReal-S1 Gen. vs EvoReal-S2 Gen.: Jaccard=0.241, Coverage(EvoReal-S1 Gen. by EvoReal-S2 Gen.)=0.403, Coverage(EvoReal-S2 Gen. by EvoReal-S1 Gen.)=0.376
- EvoReal-S1 Gen. vs EvoReal-S3 Gen.: Jaccard=0.234, Coverage(EvoReal-S1 Gen. by EvoReal-S3 Gen.)=0.580, Coverage(EvoReal-S3 Gen. by EvoReal-S1 Gen.)=0.282
- EvoReal-S2 Gen. vs EvoReal-S3 Gen.: Jaccard=0.354, Coverage(EvoReal-S2 Gen. by EvoReal-S3 Gen.)=0.762, Coverage(EvoReal-S3 Gen. by EvoReal-S2 Gen.)=0.398

**Q5. Complementary structural coverage across 7 families?** Yes, moderately-to-strongly complementary: cumulative niche coverage grows from 51 (Uniform alone) → 404 (+GM) → 534 (+EvoReal gens) → 620 (+TSPLIB/+Original70). Multiple families contribute unique niches; suite union occupies 620/1000 CVT niches.

**Q6. Most unique QD4 niches?** Top: TSPLIB100, EvoReal-S3 Gen., GM100 (83, 47, 24).

**Q7. Near-redundant families?** Candidates with near-zero unique niches relative to the suite: ['Uniform100', 'Original70', 'EvoReal-S1 Gen.'].

## Verdict

- **TSPLIB_STRUCTURAL_HETEROGENEITY** = `TSPLIB_STRUCTURAL_HETEROGENEITY_CONFIRMED`
- **BENCHMARK_COMPLEMENTARITY** = `STRONG`

Evidence flags (occupancy, feature-range width, internal pairwise spread, multivariate distance): [True, True, True, True]

## Integrity checklist

- [x] 所有 benchmark 都是 frozen official files
- [x] 没有重新生成 benchmark
- [x] 没有重新 fit CVT
- [x] 使用正式 CVT-1000 centroids
- [x] 使用正式 QD4 feature extractor
- [x] 使用正式 feature normalization
- [x] Original70 使用冻结的 aspect-ratio-preserving preprocessing
- [x] 没有训练 BQ
- [x] 没有训练 GP
- [x] 没有调用 Concorde
- [x] 没有改变任何 solver evaluation result
- [x] PCA 只用于 visualization
- [x] 所有 distance metrics 均在原始 normalized QD4 4D space 中计算

Audit complete. No BLADE mainline pipeline changes were made.
