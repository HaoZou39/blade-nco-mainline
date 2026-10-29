# Final Research Decision

## `STRUCTURE_C5_GAIN_CONFIRMED`

The frozen Structure M5 meets the stringent held-out, macro, and retention criteria.

- Structure M5 vs Hardness M5 on held-out22: -1.1905 percentage points, paired 95% CI [-2.6098, -0.0041].
- Development48 delta: -0.1595 pp; Original70 combined delta: -0.4835 pp.
- GAP8 macro delta: -0.3868 pp.
- Uniform/Gaussian material-regression guard: False.
- All eight dataset hashes and evaluator-config hashes match across M4, Hardness M5, and Structure M5.
- The held-out22 paired interval excludes zero only narrowly (upper bound -0.0041 pp); the gain is confirmed under the predeclared rule but should be described as boundary-significant.
- Structure distribution matching translated to solver performance only to the extent supported by the complete per-dataset table; no benchmark was cherry-picked and Original70 is not treated as independent.

The checkpoint was fixed at E256 before evaluation. No sixth cycle, data regeneration, learning-rate adjustment, extra seed, or additional ablation was started.

`TARGET_AWARE_STRUCTURE_C5_FINETUNE_COMPLETE`
