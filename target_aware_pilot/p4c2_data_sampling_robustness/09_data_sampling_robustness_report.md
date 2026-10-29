# Data-Sampling Robustness Report

Delta = GAP(R2) - GAP(R0); positive favors target-aware R0.

| Replicate | R0 GAP8 | R2 GAP8 | Delta | R0 benchmark wins |
|---|---:|---:|---:|---:|
| Original | 1.8312 | 2.0870 | +0.2557 | 7/8 |
| S1 | 2.0678 | 1.8499 | -0.2179 | 4/8 |
| S2 | 2.0943 | 1.9023 | -0.1920 | 0/8 |

- Across three sampling replicates: mean GAP8 Delta -0.0514 pp, median -0.1920, sample SD 0.2663, range [-0.2179, +0.2557].
- R0 beats Original Hardness M5 GAP8 (2.2337%) by replicate: {'Original': True, 'S1': True, 'S2': True}.
- Preregistered guards: both new GAP8 positive = False; at least 5/8 R0 wins in both = False; no severe Uniform/GM regression = False.
- R0 Archive30k is identical across replicates. Thus observed R0 variation is final-acquisition sampling; R2 variation includes both uniform retention and acquisition.
- TSPLIB22 remains exploratory and is not pristine held-out evidence.
- Costs: sampling 47.4s; labeling 458.7s (10,812 physical calls); training 3256.2s; evaluation 2681.2s.
- Training seed is fixed at 2026. P4-C1 and P4-C2 cannot be pooled as a fully crossed two-factor experiment, and cross-FOGA-pool robustness remains untested.
