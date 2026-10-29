# Preregistered Sampling Protocol

| Replicate | Retention seed | Acquisition seed | Training seed |
|---|---:|---:|---:|
| Original | 20261026 (R2 only; R0 deterministic) | 20261025 | 2026 |
| S1 | 260201 (R2 only; ignored by deterministic R0) | 260301 | 2026 |
| S2 | 260202 (R2 only; ignored by deterministic R0) | 260302 | 2026 |

- Delta = GAP(R2) − GAP(R0); positive favors R0.
- SUPPORTED_PRELIMINARY requires both new GAP8 Deltas positive, at least 5/8 R0 benchmark wins in each new replicate, and no Delta below −0.25 pp on Uniform100/GaussianMixture100/GaussianMixture500.
- NOT_SUPPORTED requires both new GAP8 Deltas negative and at least 5/8 R2 benchmark wins in each. All other outcomes are INCONCLUSIVE.
- Membership is frozen before diagnostics, Concorde labeling, training, or evaluation. No result-dependent seed, resampling, reweighting, or rerun is permitted.

Protocol SHA256: `c18d6cccbe72e913d96db607dd644406e1ceb9a0914a6222e90cf317cd92024e`.
