# M0 versus M5 capability-boundary extension

This read-only extension uses exactly the same 30,070 instances, frozen PC coordinates, 160-instance visualization subset, τ=1.675967% gap, axis limits, point size, plot dimensions, and primary M0/M1 P5/P95 competence scale as Fig A/B. Only the real M5 gaps, competence field, labels, and B₅ contour are new.

## Frozen M5 sources

- `audits/external_benchmark_suite_v2/blade_trajectory/work/M5/GaussianMixture100.parquet` — SHA-256 `e2228a3e3647a6c9491f74ec7c360d4a76fb9879040187e5dd93e96b24809c0e`
- `audits/external_benchmark_suite_v2/blade_trajectory/work/M5/OriginalTSPLIB70.parquet` — SHA-256 `0859515605fb134a69ee4d4771ba7296b38e308cb23c8754d852e49d1e76a455`
- `runs/blade_final_20cycle_seed2026_v1/cycles/C04/11_external_eval/TSPLIB100.parquet` — SHA-256 `99c46d88a880af26ce4ddcd42cc61c160a7d4e67848b72e649f2853c1c76839a`
- `runs/blade_final_20cycle_seed2026_v1/cycles/C04/11_external_eval/Uniform100.parquet` — SHA-256 `578c424fc5591ce29dd3563084aa40429ef2b59c9e62589b89fb650819ec6ccc`

## Results

| quantity | value |
|---|---:|
| mastered fraction M0 | 70.00% |
| mastered fraction M5 | 82.52% |
| hard → mastered | 4,735 |
| mastered → hard | 971 |
| mean gap M0 | 2.561% |
| mean gap M5 | 1.093% |
| supported-grid expansion | 19.57% |
| supported-grid contraction | 0.02% |
| supported-grid capability Jaccard | 0.7637 |

The M0→M5 boundary is **predominantly expansive**. The claim remains descriptive of the observed fixed structural map; no outward movement was imposed.

## Consistency checks

| check                               | value                                                                                   | pass   |
|:------------------------------------|:----------------------------------------------------------------------------------------|:-------|
| PCA coordinate equality M0/M5       | max_abs=0.0                                                                             | True   |
| instance IDs identical M0/M5        | True                                                                                    | True   |
| same frozen visualization subset    | 160                                                                                     | True   |
| same structural features            | read from primary frozen coordinate table                                               | True   |
| same tau as primary figures         | 1.6759665429999124                                                                      | True   |
| same color scale as primary figures | P5=0; P95=10.026652024423681                                                            | True   |
| same axis limits                    | x=(-1.1282992041443103, 0.9143546031535303); y=(-0.5397302287017989, 0.793174959341967) | True   |
| same point size / dimensions        | size=29.0; figsize=(7.2, 6.2)                                                           | True   |
| M5 smoothing disagreement <=10%     | 0.086465%                                                                               | True   |
| M5 results complete and finite      | 30070                                                                                   | True   |

All checks passed. `figD_M5_capability_landscape` overlays B₀ in gray and B₅ in red while preserving the primary figure protocol exactly.
