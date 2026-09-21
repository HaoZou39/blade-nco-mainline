# Cross-benchmark representative selection audit

Read-only representative selection over frozen A1/A2/B assignments. Cluster membership, counts, 4D structural features, PCA coordinates, τ, canonical-centroid definition, and real TSP coordinates remain unchanged.

## Frozen sources

- `audits/external_benchmark_suite_v2/hard_region_refinement_audit/04_final_cluster_assignments.csv` — SHA-256 `5999607c7f787d69902af956455c23b8c4d08d0cd39221a0c487e8f90c0ea026`
- `audits/external_benchmark_suite_v2/hard_region_refinement_audit/05_final_cluster_summary.csv` — SHA-256 `a86c25a427635fd17023db42d96d753601464c75db34e891b0a7061a240a1852`
- `audits/external_benchmark_suite_v2/hard_region_refinement_audit/06_representative_instances.csv` — SHA-256 `3ee9fd8918f285001ba16a8d409cec34f8192cff997a2b01060a28dec234e251`

## Answers

1. **Canonical benchmarks.** A1 is GM100 (`gaussian_mixture100_cdist70_1994`); A2 is GM100 (`gaussian_mixture100_cdist30_0268`); B is Uniform100 (`fresh_uniform100_02407`). These global 4D-centroid-nearest instances are retained unchanged.

2. **Benchmark composition.**

- Hard region A1: n=5,450; Uniform100=0, GM100=4,902, TSPLIB100=547, Original70=1
- Hard region A2: n=3,184; Uniform100=0, GM100=2,389, TSPLIB100=790, Original70=5
- Hard region B: n=237; Uniform100=187, GM100=1, TSPLIB100=49, Original70=0

3. **Are A1/A2 copies of the GM100 label?** No. A1 contains 547 TSPLIB100 instances and A2 contains 790; both contain multiple benchmark families. B also contains 49 TSPLIB100 instances in addition to its Uniform100 majority.

4. **Central TSPLIB100 candidate?** Yes. A2 has 790 TSPLIB100 members (24.81%); its nearest conditioned candidate `tsplib100_06697` lies at only 0.79% of the whole A2 distance distribution. It satisfies count, fraction, and <=25% ideal-distance criteria. A1 and B also have eligible TSPLIB100 alternatives documented in `benchmark_conditioned_representatives.csv`.

5. **Recommended PPT combination.** A1→GM100 canonical, A2→TSPLIB100 conditioned, B→Uniform100 canonical. This yields three benchmark families without changing membership or selecting a remote outlier.

6. **PPT representativeness audit.**

- A1: `gaussian_mixture100_cdist70_1994` (GM100); distance=0.013537; within-region percentile=0.00%; benchmark support=4,902 (89.94%); M0 gap=18.672%.
- A2: `tsplib100_06697` (TSPLIB100); distance=0.079470; within-region percentile=0.79%; benchmark support=790 (24.81%); M0 gap=8.311%.
- B: `fresh_uniform100_02407` (Uniform100); distance=0.029112; within-region percentile=0.00%; benchmark support=187 (78.90%); M0 gap=1.869%.

7. **Structural-mode claim.** Supported: **Hard regions are defined by structural features rather than benchmark identity.** Every final major region contains multiple benchmark families, and the conditioned examples remain close to their region centers.

`RECOMMENDED_PPT_REPRESENTATIVES =`

```json
[
  {
    "region": "A1",
    "dataset": "GM100",
    "instance_id": "gaussian_mixture100_cdist70_1994"
  },
  {
    "region": "A2",
    "dataset": "TSPLIB100",
    "instance_id": "tsplib100_06697"
  },
  {
    "region": "B",
    "dataset": "Uniform100",
    "instance_id": "fresh_uniform100_02407"
  }
]
```

The primary PPT export is `figures/ppt_cross_benchmark_representatives.png`; the canonical montage is retained for audit, and the two-row backup montage demonstrates within-region cross-benchmark membership.
