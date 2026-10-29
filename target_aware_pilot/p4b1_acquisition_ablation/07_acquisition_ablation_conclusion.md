# P4-B1 Acquisition Ablation Conclusion

## `TARGET_AWARE_ACQUISITION_VALUE_INCONCLUSIVE` — **INCONCLUSIVE**

Negative R1−R0 favors structure-aware acquisition. CIs use 10,000 paired bootstrap replicates.

| Dataset | M4 | Hardness | R0 Random10k | R1 Structure10k | E1 Guided | R1−R0 | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| Uniform100 | 0.4504 | 0.4725 | 0.4417 | 0.4380 | 0.4382 | -0.0037 | [-0.0142, 0.0065] |
| GaussianMixture100 | 3.9780 | 3.9008 | 2.9654 | 3.2508 | 3.4476 | +0.2854 | [0.1513, 0.4262] |
| TSPLIB100 | 0.6627 | 0.6533 | 0.5666 | 0.6046 | 0.5955 | +0.0379 | [0.0136, 0.0626] |
| EvoReal_S1_100 | 3.1703 | 3.7690 | 2.5497 | 2.7065 | 2.3874 | +0.1568 | [-0.1697, 0.4777] |
| EvoReal_S2_100 | 0.8211 | 0.8646 | 0.7703 | 0.7623 | 0.7401 | -0.0081 | [-0.0673, 0.0500] |
| EvoReal_S3_100 | 0.5591 | 0.5883 | 0.5391 | 0.5289 | 0.5420 | -0.0102 | [-0.0498, 0.0307] |
| GaussianMixture500 | 5.3226 | 5.2194 | 4.5811 | 4.9999 | 4.7064 | +0.4187 | [0.0894, 0.8087] |
| OriginalTSPLIB70 | 2.2170 | 2.4015 | 2.2357 | 2.1337 | 1.9180 | -0.1020 | [-0.5304, 0.3295] |
| TSPLIB48_development | 1.2736 | 1.2876 | 1.4145 | 1.4026 | 1.1282 | -0.0120 | [-0.5336, 0.5206] |
| TSPLIB22_previously_heldout | 4.2753 | 4.8317 | 4.0274 | 3.7290 | 3.6411 | -0.2984 | [-1.0322, 0.3947] |
| GAP8_macro | 2.1477 | 2.2337 | 1.8312 | 1.9281 | 1.8469 | +0.0969 | N/A |

- R1 wins 4/8 benchmark means; significant R1 wins 0/8; significant R0 wins 3/8. The formal rule is therefore INCONCLUSIVE, but the directional evidence does not support a stable R1 advantage.
- GAP8 R1−R0 +0.0969 pp. R0 has the lowest GAP8 of the five frozen models (1.8312). Development48 -0.0120 pp. Previously held-out22 -0.2984 pp, CI [-1.0322, 0.3947].
- R0 beats Hardness M5 in 8/8 benchmark means and 7/8 paired intervals; random final acquisition from this already structured archive retains a strong Cycle-5 benefit.
- Training final loss: R0 0.04937416, R1 0.04359868; identical 10,240-update protocol. R1 fits its training labels better, but that lower imitation loss does not translate into broader frozen-benchmark superiority.
- Cost: acquisition/audit/diagnostics 4.21s; labeling 339.7s with 3,321 verified reused labels and 6,679 new calls; R0 training 800.1s.
- Scope: this tests final acquisition only conditional on a target-aware retained E0 archive. It does not establish that every upstream structure-selection component is independently necessary.
- Research implication: do not promote target-aware final acquisition as the primary independently validated contribution. Structure-guided evolution may remain as an exploratory/domain-specific component, but P4-A also found its independent effect mixed; the defensible current result is the end-to-end candidate/archive plus Cycle-5 data-distribution effect, pending cross-seed and fresh external confirmation.
- TSPLIB22 is a repeatedly observed comparability set, not pristine confirmation evidence, and was not used to change R0/R1 data or training.

No resampling, second seed, Fresh10k, or other ablation was started.

Decision label: `INCONCLUSIVE`

`TARGET_AWARE_P4B1_ACQUISITION_ABLATION_COMPLETE`
