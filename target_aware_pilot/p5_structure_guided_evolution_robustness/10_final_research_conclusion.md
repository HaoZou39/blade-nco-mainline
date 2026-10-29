# P5 Final Research Conclusion

## `STRUCTURE_GUIDED_EVOLUTION_GAIN_INCONCLUSIVE` — **INCONCLUSIVE**

Delta = gap(E0) - gap(E1); positive favors structure-guided E1.

| Replicate | Original70 E0 | Original70 E1 | Delta | TSPLIB22 Delta | GAP5 Delta |
|---|---:|---:|---:|---:|---:|
| Original | 2.1337 | 1.9180 | +0.2158 | +0.0879 | +0.0643 |
| S1 | 2.4511 | 2.1038 | +0.3473 | +0.1159 | +0.2087 |
| S2 | 2.1658 | 2.4142 | -0.2483 | -0.3536 | -0.3210 |

## Pre-registered decision

- Original70 Delta across replicates: mean +0.1049 pp, median +0.2158, sample SD 0.3129, range [-0.2483, +0.3473].
- Preregistered gates: both new Original70 positive = False; persistent TSPLIB22 reverse >0.25 pp = False; both new GAP5 Deltas at least -0.25 pp = False.
- S1 supports E1, while S2 reverses on Original70, TSPLIB22, and GAP5. Therefore the training-value claim is not robust to evolution seed.

## Required research judgments

1. **Causal chain:** PASS. Both arms start from byte-identical ordered P03; G1 survivors become G2 parents; the independent offspring archive never supplies parents; final acquisition reads Archive30k rather than Population10k.
2. **G2 survivor effect:** `G2_SURVIVOR_SELECTION_NOT_ON_TRAINING_DATA_PATH`. G2 survivors are computed and audited, but there is no G3 and they do not modify the already accumulated offspring archive.
3. **G1 random-stream coupling:** E0/E1 use exact common mutation random numbers: identical parent order, operator schedule, and child parameter seeds with no arm/global-RNG term. The only G1 treatment difference is survivor selection.
4. **Structural replication:** E1 improves final Structure10k SWD, 3x3 JS, and strict-target fraction in all three replicates. The SWD E0-E1 gains are Original +0.1268, S1 +0.1311, S2 +0.1274; JS gains are Original +0.1406, S1 +0.1420, S2 +0.1385. G1/G2 survivor-population structural gains are likewise directionally consistent.
5. **Original70 training value:** not persistent. Delta is positive for Original and S1, but negative for S2. E1 gaps versus M4 / Hardness M5 are Original -0.2991 / -0.4835 pp, S1 -0.1132 / -0.2977 pp, S2 +0.1972 / +0.0127 pp (reference M4 2.2170%, Hardness M5 2.4015%).
6. **TSPLIB22 trend:** not consistent. Deltas are Original +0.0879 pp, S1 +0.1159 pp, S2 -0.3536 pp; every 95% paired-bootstrap CI includes zero, and S2 reverses.
7. **GAP5 without EvoReal:** does not robustly support E1. Deltas are Original +0.0643 pp, S1 +0.2087 pp, S2 -0.3210 pp. S2 is significantly negative by its paired-bootstrap CI.
8. **Core-method status:** structure-guided survivor selection is a reproducible distribution-shaping mechanism, but it is not yet supported as a robust solver-performance contribution.
9. **Next step:** re-evaluate the evolution/data-to-performance mechanism before further E1 component ablation. Do not treat the original 1.918% result as confirmed causal evidence.

## Generalization and cost diagnostics

- GaussianMixture behavior is seed-sensitive: GM100 Deltas are Original -0.1968, S1 +0.1184, S2 -0.6673 pp; GM500 Deltas are Original +0.2934, S1 +0.5445, S2 -0.6777 pp. S2 shows clear E1 degradation on both.
- Added cost for S1/S2: evolution 1219.6s; Concorde 1369.0s with 33,094 physical calls and 6,906 cache hits; BQ training 3228.5s; evaluation 2666.0s.
- Original70 is target-informed through TSPLIB48; TSPLIB22 is repeatedly observed and is not pristine. Replicates are analyzed separately; test instances were not pooled as independent evolution replicates.
- No Diversity-Only, Hardness Top10k, EvoReal Fresh10k, G3, extra seed, or other follow-up was started.

Decision label: `INCONCLUSIVE`

`BLADE_P5_STRUCTURE_GUIDED_EVOLUTION_ROBUSTNESS_COMPLETE`
