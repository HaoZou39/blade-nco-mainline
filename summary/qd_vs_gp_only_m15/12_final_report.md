# QD+GP versus GP-only

The causal branch is P0 at C00; M0 and M1 are shared, and C01 is the first divergent candidate generation.

Both arms run to the matched endpoint M15. Primary N=100 OOD mean gap: {'FULL': 1.2955639976857107, 'GP_ONLY': 1.214776174798655}. Harder selected training data is not treated as automatic evidence of better training value.

| arm | dataset | mean_gap_percent |
| --- | --- | --- |
| FULL | Uniform100 | 0.581791078211136 |
| FULL | GaussianMixture100 | 2.115485207389851 |
| FULL | TSPLIB100 | 0.7004333745233688 |
| FULL | EvoReal_S1_100 | 2.115219448058946 |
| FULL | EvoReal_S2_100 | 0.9230621058384814 |
| FULL | EvoReal_S3_100 | 0.6236198526179058 |
| FULL | GaussianMixture500 | 3.787941684527611 |
| FULL | OriginalTSPLIB70 | 2.4311869628141562 |
| GP_ONLY | Uniform100 | 0.5940806014202307 |
| GP_ONLY | GaussianMixture100 | 1.906158917109227 |
| GP_ONLY | TSPLIB100 | 0.7058231521003556 |
| GP_ONLY | EvoReal_S1_100 | 1.7849249857435274 |
| GP_ONLY | EvoReal_S2_100 | 1.0271305098318897 |
| GP_ONLY | EvoReal_S3_100 | 0.6498433092082753 |
| GP_ONLY | GaussianMixture500 | 3.7694321153072807 |
| GP_ONLY | OriginalTSPLIB70 | 2.2145727885314486 |
