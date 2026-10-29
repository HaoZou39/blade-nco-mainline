# Random10k versus Structure10k Distribution

| Metric | R0 Random10k | R1 Structure10k |
|---|---:|---:|
| Strict target fraction | 10.10% | 30.16% |
| Continuous 2D SWD | 0.536529 | 0.366734 |
| 3×3 JS | 0.447059 | 0.210295 |
| Occupied/effective cells | 9/4.465 | 8/6.307 |
| Largest cell share | 40.36% | 22.96% |
| Target-neighborhood coverage | 45/48 | 45/48 |
| Effective lineages | 613.3 | 563.7 |
| Maximum lineage share | 0.81% | 1.20% |
| FFT median | 24.8897 | 23.5447 |
| NN-CV median | 1.0134 | 1.0016 |
| FFT/NN-CV auxiliary JS | 0.518754 | 0.493910 |

- Frozen membership overlap: 3,321/10,000 (33.21%). R0 missing neighborhoods: ['d198', 'eil51', 'ts225']; R1 missing neighborhoods: ['d198', 'eil51', 'ts225'].
- Provenance JS (R0 vs R1): operator `0.001628`, generation `0.000097`, evolution origin `0.000097`. Full per-category counts are in `support_provenance_comparison.csv`.
- The structure diagnostics were computed only after Random10k membership was frozen. No resampling or distribution correction occurred.
- HGB hardness is not used. Canonical M4 teacher-forced hardness will be attached during labeling only as passive metadata.
