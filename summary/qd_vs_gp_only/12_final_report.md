# QD+GP versus GP-only

The causal branch is P0 at C00; M0 and M1 are shared, and C01 is the first divergent candidate generation.

Primary N=100 OOD mean gap at M6: {'FULL': 1.5218925314412723, 'GP_ONLY': 1.423193932861473}. Harder selected training data is not treated as automatic evidence of better training value.

| arm | dataset | mean_gap_percent |
| --- | --- | --- |
| FULL | Uniform100 | 0.4555515215589771 |
| FULL | GaussianMixture100 | 3.3217501887421976 |
| FULL | TSPLIB100 | 0.6538686404795834 |
| FULL | EvoReal_S1_100 | 2.3394620496577407 |
| FULL | EvoReal_S2_100 | 0.7722327116482555 |
| FULL | EvoReal_S3_100 | 0.522149066678585 |
| FULL | GaussianMixture500 | 4.73289546125012 |
| FULL | OriginalTSPLIB70 | 2.8908700461288306 |
| GP_ONLY | Uniform100 | 0.4495311218201247 |
| GP_ONLY | GaussianMixture100 | 2.7513746441577953 |
| GP_ONLY | TSPLIB100 | 0.6136490175786514 |
| GP_ONLY | EvoReal_S1_100 | 2.317724297644244 |
| GP_ONLY | EvoReal_S2_100 | 0.8708084163997016 |
| GP_ONLY | EvoReal_S3_100 | 0.5624132885269728 |
| GP_ONLY | GaussianMixture500 | 4.789531654202158 |
| GP_ONLY | OriginalTSPLIB70 | 2.0111515194190956 |
