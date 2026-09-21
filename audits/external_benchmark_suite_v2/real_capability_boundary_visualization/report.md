# Real capability-boundary visualization

## Scope and provenance

This is a read-only analysis of 30,070 real, frozen benchmark instances from Uniform100, GM100, TSPLIB100, and Original70. It trained no model, regenerated no benchmark, called no oracle, and changed no solver result. Uniform100/TSPLIB100 gaps come from the formal M0 and C00→M1 passive external evaluations. GM100/Original70 gaps come from the external-suite-v2 reconciled passive evaluations (correct GM physical-order binding and frozen Original70 aspect-preserving preprocessing).

Frozen inputs:

- `audits/external_benchmark_suite_v2/blade_trajectory/work/M0/GaussianMixture100.parquet` — SHA-256 `196ad540c648834f8c6ece6d25fdad122a3863cf2a6d22f46a9e9f5049b873b1`
- `audits/external_benchmark_suite_v2/blade_trajectory/work/M0/OriginalTSPLIB70.parquet` — SHA-256 `1ac55b7842e5f2dd0e4c65cb6c2affad196ccff281fd04486759789459ecc472`
- `audits/external_benchmark_suite_v2/blade_trajectory/work/M1/GaussianMixture100.parquet` — SHA-256 `48e92095e960644feabdcbc79c605d9c3455c8cf3287aa504f0e0ecec7456e30`
- `audits/external_benchmark_suite_v2/blade_trajectory/work/M1/OriginalTSPLIB70.parquet` — SHA-256 `2d4faf5207a8be5a15149901f2a4340de1d2daf622c3a3ee0603b2813289bdaf`
- `audits/external_benchmark_suite_v2/qd4_structural_distribution_audit/_cache/all_features.parquet` — SHA-256 `03bfc189d8f7dada2c92721dc7036b12fd065fe26f4d6ad7155e9dbc7764bdb3`
- `runs/blade_final_20cycle_seed2026_v1/01_baseline_M0/TSPLIB100.parquet` — SHA-256 `1f98d9df94ff0b9a2600133223f7a07c3c5cb9f4ed9b0f7217b7d1f8f07f7d70`
- `runs/blade_final_20cycle_seed2026_v1/01_baseline_M0/Uniform100.parquet` — SHA-256 `e28882bdcb93f9727e7f1660735a32aef2b4c20616a98677636cab25e59660f4`
- `runs/blade_final_20cycle_seed2026_v1/cycles/C00/11_external_eval/TSPLIB100.parquet` — SHA-256 `274b794bc9f3b675f99ad88b273f18cba8f2ab7830fd50011676bda3019004ca`
- `runs/blade_final_20cycle_seed2026_v1/cycles/C00/11_external_eval/Uniform100.parquet` — SHA-256 `56c9e219d33bd2d4031757a7df86b654304ab79ef97a40d3467b7365d96a5c22`

The structural representation is the formal four-feature normalized representation: `spn_sum_of_lowest_edge_values`, `spn_distance_varcoeff`, `spn_mst_depth_mean`, and `spn_mst_dists_median`. PCA was fit **once** on every eligible instance and reused unchanged for M0, M1, and Fig C. PC1/PC2 explain 85.57%/10.86% (96.43% jointly).

The plotted subset contains 160 points (Uniform100=40, GM100=40, TSPLIB100=40, Original70=40), selected deterministically by per-dataset farthest-point coverage in the original normalized four-feature space. PCA fitting and capability-field fitting use all 30,070 eligible instances, not only the displayed subset.

## Fixed visualization protocol

- Performance metric: greedy solver optimality gap only.
- Fixed threshold: **τ = 1.675967%**, the pooled M0 P70, reused for both solvers.
- Shared competence scaling: pooled M0/M1 P5=0.000000% and P95=10.026652%; gaps are clipped once and mapped to `1 - normalized_gap`.
- Capability fields: distance-weighted kNN regression in the frozen PC1/PC2 map, k=75; visualization only. The initial uniform-weight smoothers produced 10.61% (M0) and 14.54% (M1) observation/field disagreement, exceeding the 10% tolerance, so the prespecified alternative distance weighting was used. Final point-to-field disagreement at observed coordinates is 0.023% (M0) and 0.040% (M1).
- Support restriction: convex hull intersected with a local-density gate based on the training-set P95 20-neighbor radius (0.035479); no field is shown outside support.
- Fig C: HDBSCAN in the original normalized four-feature space, never in PCA. `min_cluster_size=91` (1% of the M0 hard set, rounded up), `min_samples=19` (0.2%, rounded up). No cluster count was requested or manually imposed.

## Capability transition

| quantity | value |
|---|---:|
| eligible instances | 30,070 |
| mastered fraction M0 | 70.00% |
| mastered fraction M1 | 74.77% |
| hard → mastered | 2,535 |
| mastered → hard | 1,100 |
| unchanged mastered | 19,949 |
| unchanged hard | 6,486 |
| supported-grid expansion | 4.22% |
| supported-grid contraction | 0.07% |
| supported-grid capability Jaccard | 0.9364 |

## Real hard regions

The M0 hard set contains 9,021 instances. HDBSCAN retained 2 regions and marked 150 points as noise/minor modes. Non-noise silhouette=0.7201; mean adjusted Rand stability under ±20% `min_cluster_size` perturbations=0.5384.

- **Hard region A**: n=8,634; mean/median M0 gap=7.894%/5.951%; composition (Uniform100=0, GM100=7291, TSPLIB100=1337, Original70=6); representative `GM100::gaussian_mixture100_cdist30_0473`.
- **Hard region B**: n=237; mean/median M0 gap=2.173%/1.961%; composition (Uniform100=187, GM100=1, TSPLIB100=49, Original70=0); representative `Uniform100::fresh_uniform100_02407`.

Region letters are neutral identifiers, not inferred geometry names. Each representative is the real cluster member nearest its four-dimensional feature centroid; thumbnails show its real node coordinates without a tour.

The unequal region sizes and moderate parameter stability are retained as real diagnostics: the two-region result is evidence of multiple modes, not a claim that the hard set has a universal two-class ontology.

## Answers to the required questions

**Q1. Does the competence landscape visibly change from M0 to M1 in the fixed real structural space?**  
Yes. The mastered share changes by +4.77 percentage points and 12.09% of instances cross the fixed threshold in either direction. The map and instances are fixed; only observed gaps and fields change.

**Q2. Is the M1 mastered fraction higher under the fixed τ?**  
Yes: 70.00% → 74.77%.

**Q3. How many instances move hard → mastered?**  
2,535.

**Q4. How many instances move mastered → hard?**  
1,100.

**Q5. Does the boundary expand globally or move only in parts of structural space?**  
The supported-grid boundary is predominantly expansive. Expansion covers 4.22% and contraction 0.07% of the common supported grid. The evidence supports “solver adaptation reshapes the capability boundary,” not uniform outward expansion.

**Q6. Do M0 high-gap instances form multiple structurally distinct regions?**  
Yes: the data-driven analysis retained 2 regions, with silhouette 0.7201. PCA overlap, where present, is expected because clustering was correctly performed in the original normalized space.

**Q7. What are their benchmark compositions and representative geometries?**  
The exact compositions and representatives are listed above and in `05_hard_region_summary.csv`. `figC_representative_instances` displays the underlying coordinates; no semantic geometry label is asserted.

**Q8. Is there enough real evidence for the two Page 2 claims?**  
- **“The target moves with the solver.”** Supported as a boundary-reshaping claim: 2,535 improvements and 1,100 regressions occur at fixed coordinates and fixed τ.
- **“The boundary is a set of diverse failure regions, not a single point.”** Supported: the M0 hard set contains 9,021 real instances and 2 retained structural regions, plus noise/minor modes.

The adjacent-cycle shift is material under the frozen rule, so no secondary M0-vs-M16 visualization was triggered.

## Consistency checks

| check                                        | value                                                                                   | pass   |
|:---------------------------------------------|:----------------------------------------------------------------------------------------|:-------|
| PCA coordinate equality M0/M1                | max_abs=0.0                                                                             | True   |
| instance IDs identical M0/M1                 | True                                                                                    | True   |
| structural features identical M0/M1          | single frozen matrix reused                                                             | True   |
| only performance labels change               | shared map; separate M0/M1 gaps                                                         | True   |
| same tau                                     | 1.6759665429999124                                                                      | True   |
| same color scale                             | P5=0; P95=10.026652024423681                                                            | True   |
| same axis limits                             | x=(-1.1282992041443103, 0.9143546031535303); y=(-0.5397302287017989, 0.793174959341967) | True   |
| same point size / dimensions                 | size=29.0; figsize=(7.2, 6.2)                                                           | True   |
| Fig C uses M0 hard instances only            | True                                                                                    | True   |
| no manually invented hard-region coordinates | observed frozen PC projections only                                                     | True   |
| smoothing disagreement <=10%                 | M0=0.023279%; M1=0.039907%                                                              | True   |
| all retained clusters size >=5               | 237                                                                                     | True   |

All required checks passed. Figures use identical M0/M1 dimensions, point sizes, axes, coordinates, threshold, and color scale. Fig C labels are derived only from observed M0 hard instances; no hard-region coordinates were invented.
