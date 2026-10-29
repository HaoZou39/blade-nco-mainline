# Archive and Training Distribution Comparison

All diagnostics were computed after R2 membership was frozen; no resampling followed.

| Distribution | Strict | SWD | 3×3 JS | Cells/effective | Max cell | Coverage | Effective lineages | Max lineage | FFT median | NN-CV median | Aux JS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E0_full_eligible39848 | 7.68% | 0.613376 | 0.486755 | 9/3.813 | 52.06% | 45/48 | 525.2 | 1.41% | 25.7318 | 1.0157 | 0.511988 |
| R0_target_archive30k | 10.05% | 0.536298 | 0.445236 | 9/4.460 | 40.91% | 45/48 | 646.3 | 0.82% | 25.0002 | 1.0155 | 0.519709 |
| R2_random_archive30k | 7.70% | 0.613708 | 0.485831 | 9/3.816 | 52.06% | 45/48 | 521.1 | 1.44% | 25.7025 | 1.0145 | 0.512585 |
| R0_random10k | 10.10% | 0.536529 | 0.447059 | 9/4.465 | 40.36% | 45/48 | 613.3 | 0.81% | 24.8897 | 1.0134 | 0.518754 |
| R2_random10k | 7.89% | 0.617114 | 0.488460 | 9/3.809 | 52.04% | 45/48 | 502.8 | 1.39% | 25.7232 | 1.0159 | 0.511180 |

- R0/R2 archive overlap: 22,552/30,000 (75.17%).
- R0/R2 final 10k overlap: 2,635/10,000 (26.35%).
- Random 30k → random 10k is marginally equivalent to a uniform 10k draw from all 39,848; the explicit two-stage implementation matches the R0 archive→acquisition workflow.
- Differences between R0/R2 training sets include finite-sample random variation; this single seed cannot remove or quantify seed-to-seed training uncertainty.
- HGB/BQ hardness was not used and no new Oracle hardness inference was run.
