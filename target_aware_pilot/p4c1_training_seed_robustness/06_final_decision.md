# P4-C1 Final Decision

## `TARGET_AWARE_RETENTION_TRAINING_SEED_ROBUSTNESS_SUPPORTED` — **SUPPORTED**

- Preregistered rule result: all-seed positive GAP8 Delta = **True**; R0 ≥5/8 benchmark wins in every seed = **True**; positive cross-seed mean Delta on at least 5/8 benchmarks = **True** (8/8).
- GAP8 Delta across seeds: mean +0.3056 pp, sample SD 0.2396, range [+0.0948, +0.5662].
- Directional breadth: positive cross-seed mean Delta on 8/8 benchmarks; largest effects are GaussianMixture100 (+0.8570 pp), EvoReal_S1_100 (+0.6997 pp), GaussianMixture500 (+0.5375 pp).
- No systematic Uniform/Gaussian-mixture degradation was detected. Real-TSPLIB conclusions remain limited because OriginalTSPLIB70 changes direction across seeds and neither its nor TSPLIB22's per-seed paired CIs exclude zero.
- This conclusion addresses training-seed robustness only. It does not establish robustness to archive/data-sampling seeds.
- P4-C2 is recommended as the next robustness check, but is not started automatically.

Decision label: `SUPPORTED`

`TARGET_AWARE_P4C1_TRAINING_SEED_ROBUSTNESS_COMPLETE`
