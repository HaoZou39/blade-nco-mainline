# P4-A Evolution Ablation Conclusion

## `STRUCTURE_ACQUISITION_DOMINANT_OR_INCONCLUSIVE` (CASE B)

Negative E1−E0 values favor structure-guided evolution. Confidence intervals use 10,000 paired bootstrap replicates.

| Dataset | M4 | Hardness M5 | E0 random+acq | E1 guided+acq | E1−E0 | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Uniform100 | 0.4504 | 0.4725 | 0.4380 | 0.4382 | +0.0002 | [-0.0100, 0.0103] |
| GaussianMixture100 | 3.9780 | 3.9008 | 3.2508 | 3.4476 | +0.1968 | [0.0524, 0.3397] |
| TSPLIB100 | 0.6627 | 0.6533 | 0.6046 | 0.5955 | -0.0090 | [-0.0419, 0.0239] |
| EvoReal_S1_100 | 3.1703 | 3.7690 | 2.7065 | 2.3874 | -0.3192 | [-0.6051, -0.0493] |
| EvoReal_S2_100 | 0.8211 | 0.8646 | 0.7623 | 0.7401 | -0.0222 | [-0.0825, 0.0400] |
| EvoReal_S3_100 | 0.5591 | 0.5883 | 0.5289 | 0.5420 | +0.0131 | [-0.0286, 0.0553] |
| GaussianMixture500 | 5.3226 | 5.2194 | 4.9999 | 4.7064 | -0.2934 | [-0.6748, 0.0401] |
| OriginalTSPLIB70 | 2.2170 | 2.4015 | 2.1337 | 1.9180 | -0.2158 | [-0.6027, 0.1241] |
| TSPLIB48_development | 1.2736 | 1.2876 | 1.4026 | 1.1282 | -0.2744 | [-0.7522, 0.0910] |
| TSPLIB22_repeated_heldout | 4.2753 | 4.8317 | 3.7290 | 3.6411 | -0.0879 | [-0.8192, 0.6040] |
| GAP8_macro | 2.1477 | 2.2337 | 1.9281 | 1.8469 | -0.0812 | N/A |

- E1 favors 5/8 benchmark means; significant E1 wins: 1/8; significant E0 wins: 1/8.
- GAP8 E1−E0: -0.0812 pp. Development48: -0.2744 pp. Repeated-observation TSPLIB22: -0.0879 pp, CI [-0.8192, 0.6040].
- E0 random-evolution + structure-acquisition beats Hardness M5 in 8/8 benchmark means and 6/8 paired intervals. Its GAP8 improvement is 0.3056 pp versus E1's 0.3868 pp.
- Arithmetic GAP8 decomposition: E0 final acquisition route accounts for 79.0% of the observed E1-vs-Hardness mean reduction, while guided evolution adds the remaining 0.0812 pp. This ratio is descriptive, not a cross-seed causal effect estimate.
- TSPLIB22 is explicitly not treated as a pristine final confirmation set because its P3 result was already observed. No P4-A selection or training decision used it.
- Evolution mutation wall: E0 99.5s versus E1 99.8s under equal 40,000-attempt budgets. New E0 labeling: 485.4s/10,000 calls; training: 802.6s.
- Attribution: the dominant supported signal is the shared structure-aware final acquisition. Guided survivor/parent selection improves the frozen two-feature match and the GAP8 mean, but its independent solver benefit is mixed rather than stable in this single-seed design.
- Follow-up priority: cross-seed E0/E1 repeats are needed to resolve the evolution-selection effect; an EvoReal Fresh10k control would answer the separate question of whether synthetic structure matching adds value beyond real-data fine-tuning. Neither is run automatically here.

No P4-B/P4-C, third generation, extra training seed, or mainline modification was started.

`TARGET_AWARE_P4A_EVOLUTION_ABLATION_COMPLETE`
