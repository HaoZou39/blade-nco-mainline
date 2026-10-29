# Generalization and Forgetting Analysis

Negative deltas favor Structure M5. All confidence intervals are paired 10,000-replicate bootstrap intervals.

| Dataset | M4 | Hardness M5 | Structure M5 | Structure−Hardness | 95% CI |
|---|---:|---:|---:|---:|---:|
| Uniform100 | 0.4504 | 0.4725 | 0.4382 | -0.0344 | [-0.0452, -0.0238] |
| GaussianMixture100 | 3.9780 | 3.9008 | 3.4476 | -0.4532 | [-0.6037, -0.3031] |
| TSPLIB100 | 0.6627 | 0.6533 | 0.5955 | -0.0578 | [-0.0962, -0.0193] |
| EvoReal_S1_100 | 3.1703 | 3.7690 | 2.3874 | -1.3816 | [-2.1631, -0.7451] |
| EvoReal_S2_100 | 0.8211 | 0.8646 | 0.7401 | -0.1245 | [-0.1942, -0.0523] |
| EvoReal_S3_100 | 0.5591 | 0.5883 | 0.5420 | -0.0463 | [-0.0906, -0.0012] |
| GaussianMixture500 | 5.3226 | 5.2194 | 4.7064 | -0.5130 | [-0.8675, -0.1774] |
| OriginalTSPLIB70 | 2.2170 | 2.4015 | 1.9180 | -0.4835 | [-1.0163, -0.0219] |
| TSPLIB48_development | 1.2736 | 1.2876 | 1.1282 | -0.1595 | [-0.5225, 0.2077] |
| TSPLIB22_heldout | 4.2753 | 4.8317 | 3.6411 | -1.1905 | [-2.6098, -0.0041] |
| GAP8_macro | 2.1477 | 2.2337 | 1.8469 | -0.3868 | N/A |

- Held-out22 Structure−Hardness: -1.1905 pp, 95% CI [-2.6098, -0.0041].
- Development48 Structure−Hardness: -0.1595 pp; Original70 combined: -0.4835 pp. Original70 is not described as independent.
- GAP8 macro Structure−Hardness: -0.3868 pp.
- Material/significant Uniform/Gaussian regression guard: False.
- The three models use identical dataset SHA256 and evaluator-config SHA256 values on all eight benchmarks; instance alignment additionally matches ID, coordinate hash, node count, and optimum.
- Held-out22 clears zero only narrowly; its CI upper bound is close to zero, so this is positive but boundary-strength statistical evidence.

## Incremental cost relative to original C04

- Original C04 evolution: 110,000 attempts, 218.8s mutation wall. P2.2 deployable E1 path: 40,000 attempts across two generations, 99.8s mutation wall: **70,000 fewer attempts and -119.0s**. The full P2.2 causal audit additionally ran the matched E0 control (80,000 total attempts across both arms).
- Original C04 labeling: 8,670 new calls, 429.3s. P3: 10,000 new calls, 475.5s combined labeling/M4-hardness stage: **+1,330 calls and +46.2s**.
- Original C04 training: 828.2s. P3 Structure training: 799.1s under the same 10,240-update budget: **-29.1s**.
- Simple deployable-stage wall sum (mutation + labeling/hardness + training): original C04 1476.2s versus Structure path 1374.5s, a **-101.8s** difference. This excludes evaluation and treats recorded stage walls additively.
