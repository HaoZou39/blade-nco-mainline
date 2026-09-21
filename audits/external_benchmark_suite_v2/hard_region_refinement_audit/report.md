# Hard-region refinement audit

This is a read-only audit over the frozen M0 hard set. No benchmark, solver result, model, threshold, structural feature, PCA coordinate, or previous visualization artifact was modified.

## Frozen protocol

- `audits/external_benchmark_suite_v2/real_capability_boundary_visualization/03_real_hard_region_membership.csv` — SHA-256 `8c311ac2c6b7358a12985745aa97cf675cc668a7cfa4d5e478a9897ab76fddc0`

- Hard definition: `M0_gap > τ`, with τ=1.675966543%.
- Frozen hard instances: 9,021.
- Clustering space: the original normalized 4D structural features (`local`, `global`, `topology`, `scale`).
- PCA is used only for visualization and never for cluster assignment.

## Stage 1 — global HDBSCAN

The full 40-setting sweep did **not** directly support three or more major regions at an acceptable noise rate. `non_noise_cluster_count >= 3` occurred only by assigning a large fraction of the hard set to noise; the eligible count with noise <=20% is 0. Low-noise solutions retained the original dominant A-like family plus the small B-like region.

Answer 2: **No**, Stage 1 does not directly discover >=3 defensible major hard regions.

## Stage 2 — refinement of the original largest cluster

The original Region A contains 8,634 instances.

- HDBSCAN leaf refinement did not yield a defensible split: every multi-subcluster setting used very high noise (the minimum in this scan is 79.4%), and the count with >=2 subclusters and noise <=20% is 0.
- GMM provides evidence for internal distributional structure. For k=2, both components are large ([5450,3184]), no noise is introduced, silhouette=0.220, size balance=0.584, and mean/min seed stability ARI=1.000/1.000. Relative to k=1, ΔBIC=-48,166.2 and ΔAIC=-48,272.2 (negative favors the split).
- BIC continues to improve through k=5, but silhouette falls from 0.220 at k=2 to 0.068. The extra components are therefore treated as finer density structure, not promoted to additional named regions.
- KMeans is only a sanity check; it is not used for final assignments.

Answer 1: **Yes, with a hierarchical caveat.** Region A has a stable two-component GMM structure, but it is not a clean density-separated HDBSCAN split.

Answer 3: **Yes.** Largest-cluster refinement supports two broad, stable submodes without discarding instances as noise.

Answer 4: **A1 / A2 / B**, preserving the fact that A1 and A2 are submodes of the original dominant family.

## Final regions

| region | n | mean gap | median gap | benchmark composition | normalized-feature centroid | representative |
|---|---:|---:|---:|---|---|---|
| Hard region A1 | 5,450 | 8.407% | 6.014% | Uniform100=0, GM100=4,902, TSPLIB100=547, Original70=1 | local=0.057, global=0.923, topology=0.431, scale=0.000 | `GM100::gaussian_mixture100_cdist70_1994` |
| Hard region A2 | 3,184 | 7.016% | 5.813% | Uniform100=0, GM100=2,389, TSPLIB100=790, Original70=5 | local=0.284, global=0.819, topology=0.390, scale=0.128 | `GM100::gaussian_mixture100_cdist30_0268` |
| Hard region B | 237 | 2.173% | 1.961% | Uniform100=187, GM100=1, TSPLIB100=49, Original70=0 | local=0.951, global=0.045, topology=0.285, scale=0.922 | `Uniform100::fresh_uniform100_02407` |

The structural interpretation is deliberately relative rather than semantic: A1 has lower `local` and near-zero `scale` values with higher `global`; A2 shifts toward higher `local`/`scale` and lower `global`; B occupies a sharply different high-`local`, high-`scale`, low-`global` part of the frozen space. The gap and benchmark-composition plots show the associated empirical differences. No cluster is named from one representative image.

Answer 5: Counts, gap statistics, composition, centroids, and representatives are listed above and exported in `05_final_cluster_summary.csv` and `06_representative_instances.csv`.

Answer 6: **Yes.** The data support the claim “High-gap instances are structurally heterogeneous.”

Answer 7: **Yes, if stated hierarchically.** The evidence supports “At least three structurally distinct hard regions are present,” with A1 and A2 explicitly described as stable submodes of one larger A family. It does not support presenting all three as equally independent HDBSCAN density islands.

Answer 8: `figD_refined_hard_regions_subsampled.png` is the best Page 2 panel. It retains full-data counts and density contours while preventing A1/A2 from visually overwhelming B. The publication pair `figD_refined_hard_regions.png/.pdf` uses the same recommended rendering.

## Verdict

- `HARD_REGION_REFINEMENT = PARTIAL_TWO_MAJOR_PLUS_SUBMODES`
- `DATA_SUPPORTS_AT_LEAST_THREE_MAJOR_HARD_REGIONS = YES`
- `LARGEST_CLUSTER_HAS_STABLE_SUBSTRUCTURE = YES`
- `RECOMMENDED_DISPLAY_LABELS = ["A1", "A2", "B"]`
