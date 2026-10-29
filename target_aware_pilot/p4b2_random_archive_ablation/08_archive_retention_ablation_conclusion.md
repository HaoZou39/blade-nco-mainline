# P4-B2 Archive Retention Ablation Conclusion

## `TARGET_AWARE_RETENTION_VALUE_PRELIMINARILY_SUPPORTED` — **PRELIMINARILY_SUPPORTED**

Positive R2−R0 favors target-aware R0 retention; negative favors structure-free R2. CIs use 10,000 paired bootstrap replicates.

| Dataset | M4 | Hardness | R0 TargetArchive→Random10k | R2 RandomArchive→Random10k | E0 Structure | E1 Guided | R2−R0 | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Uniform100 | 0.4504 | 0.4725 | 0.4417 | 0.4560 | 0.4380 | 0.4382 | +0.0143 | [0.0036, 0.0250] |
| GaussianMixture100 | 3.9780 | 3.9008 | 2.9654 | 3.8250 | 3.2508 | 3.4476 | +0.8596 | [0.7217, 0.9974] |
| TSPLIB100 | 0.6627 | 0.6533 | 0.5666 | 0.6292 | 0.6046 | 0.5955 | +0.0625 | [0.0381, 0.0876] |
| EvoReal_S1_100 | 3.1703 | 3.7690 | 2.5497 | 3.3365 | 2.7065 | 2.3874 | +0.7867 | [0.3732, 1.2249] |
| EvoReal_S2_100 | 0.8211 | 0.8646 | 0.7703 | 0.8270 | 0.7623 | 0.7401 | +0.0567 | [-0.0016, 0.1180] |
| EvoReal_S3_100 | 0.5591 | 0.5883 | 0.5391 | 0.5688 | 0.5289 | 0.5420 | +0.0297 | [-0.0116, 0.0719] |
| GaussianMixture500 | 5.3226 | 5.2194 | 4.5811 | 4.9233 | 4.9999 | 4.7064 | +0.3421 | [0.0483, 0.6628] |
| OriginalTSPLIB70 | 2.2170 | 2.4015 | 2.2357 | 2.1300 | 2.1337 | 1.9180 | -0.1058 | [-0.4566, 0.2157] |
| TSPLIB48_development | 1.2736 | 1.2876 | 1.4145 | 1.4148 | 1.4026 | 1.1282 | +0.0003 | [-0.4321, 0.4256] |
| TSPLIB22_previously_heldout | 4.2753 | 4.8317 | 4.0274 | 3.6902 | 3.7290 | 3.6411 | -0.3372 | [-0.8688, 0.1203] |
| GAP8_macro | 2.1477 | 2.2337 | 1.8312 | 2.0870 | 1.9281 | 1.8469 | +0.2557 | N/A |

- R0 wins 7/8 benchmark means and 5/8 paired CIs; R2 wins 1/8 means and 0/8 CIs.
- GAP8: R0 1.8312%, R2 2.0870%, R2−R0 +0.2557 pp.
- Original70 R2−R0 -0.1058 pp; Development48 +0.0003 pp; previously held-out22 -0.3372 pp with CI [-0.8688, 0.1203].
- Costs: data recovery/sampling/audit 18.18s; labeling 309.8s (4,252 reused, 5,748 new calls); training 802.4s; evaluation 665.7s.
- R2 is structure-free only at Cycle-5 survivor/archive/final selection. Its starting P03 still reflects earlier hardness-driven cycles.
- This is one frozen sampling/training seed. Mean direction, paired uncertainty and cross-dataset consistency are reported separately; no seed robustness is claimed.
- TSPLIB22 is repeatedly observed and is not pristine confirmation evidence.
- Recommendation: retain target-aware archive retention provisionally in the current method, but describe its value as preliminary and prioritize a pre-registered cross-seed replication before making it a primary paper claim. The TSPLIB subsets do not show a significant retention benefit, so this trial does not justify treating structural matching as universally beneficial.

No resampling, second seed, new evolution, Fresh10k, or follow-up ablation was started.

Decision label: `PRELIMINARILY_SUPPORTED`

`TARGET_AWARE_P4B2_RANDOM_ARCHIVE_ABLATION_COMPLETE`
