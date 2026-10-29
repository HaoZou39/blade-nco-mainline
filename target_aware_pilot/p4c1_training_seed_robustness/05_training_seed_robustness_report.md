# Training-Seed Robustness Report

Delta = GAP(R2) − GAP(R0); positive favors R0.

| Seed | R0 GAP8 | R2 GAP8 | Delta | R0 benchmark wins |
|---:|---:|---:|---:|---:|
| 2026 | 1.8312 | 2.0870 | +0.2557 | 7/8 |
| 260101 | 2.0642 | 2.6304 | +0.5662 | 7/8 |
| 260102 | 2.0060 | 2.1009 | +0.0948 | 8/8 |

- Across the three paired seeds, GAP8 Delta mean +0.3056 pp, median +0.2557, sample SD 0.2396, range [+0.0948, +0.5662].
- R0 beats Original Hardness M5 GAP8 (2.2337%) by seed: {2026: True, 260101: True, 260102: True}.
- R0 benchmark-win counts by seed: {2026: 7, 260101: 7, 260102: 8}.
- Cross-seed mean Delta favors R0 on 8/8 frozen benchmarks. The three largest mean effects are GaussianMixture100 (+0.8570 pp), EvoReal_S1_100 (+0.6997 pp), GaussianMixture500 (+0.5375 pp); the advantage is broad in direction but concentrated in magnitude.
- No systematic Uniform/Gaussian-mixture degradation is observed: Uniform100 +0.0198 pp (3/3 R0-winning seeds), GaussianMixture100 +0.8570 pp (3/3), GaussianMixture500 +0.5375 pp (3/3). TSPLIB100 is also positive at +0.0860 pp (3/3).
- Real-TSPLIB evidence is weaker: OriginalTSPLIB70 mean Delta +0.1784 pp with 2/3 R0-winning seeds and 0/3 per-seed CIs excluding zero; TSPLIB22 mean Delta -0.0137 pp with 2/3 R0-winning seeds and 0/3 CIs excluding zero.
- Final-loss descriptive summary:

| Arm | Mean | SD | Min | Max |
|---|---:|---:|---:|---:|
| R0 | 0.06052963 | 0.01206764 | 0.04937416 | 0.07333887 |
| R2 | 0.06167900 | 0.01149072 | 0.05182892 | 0.07430265 |

- Final training loss is descriptive only; its small arm difference is not used as evidence of generalization.
- Per-seed CIs quantify test-instance uncertainty only. Seed-level variability is summarized descriptively from three paired seeds; instances are not pooled across seeds as independent replicates.
- TSPLIB22 is repeatedly observed and remains exploratory.
- P4-C2 data/archive-sampling-seed robustness is the appropriate next priority because P4-C1 changes only training stochasticity. It is recommended for a separate preregistered run and is not started here.
