# Training Dynamics Audit

All six frozen logs contain 256 epochs and exactly 10,240 optimizer updates. Protocol fields are identical: fresh Adam, LR 3.75e-5, no scheduler, logical batch 256, microbatch 64, seed 2026. No retraining was performed.

| replicate | arm | 0.0 | 0.1 | 0.25 | 0.5 | 0.75 | 1.0 | epoch1_to_final_reduction |
|---|---|---|---|---|---|---|---|---|
| Original | E0 | 0.075168862 | 0.076337965 | 0.063127132 | 0.072891596 | 0.077730021 | 0.048896219 | 0.026272643 |
| Original | E1 | 0.044425833 | 0.044664422 | 0.037129067 | 0.041334631 | 0.042049087 | 0.025391639 | 0.019034194 |
| S1 | E0 | 0.08233957 | 0.069162112 | 0.070178366 | 0.063813537 | 0.071821688 | 0.045178601 | 0.037160969 |
| S1 | E1 | 0.051019205 | 0.042138496 | 0.036901648 | 0.039163306 | 0.046420653 | 0.02438384 | 0.026635364 |
| S2 | E0 | 0.084541434 | 0.068286128 | 0.065455211 | 0.065667304 | 0.077782157 | 0.051056876 | 0.033484557 |
| S2 | E1 | 0.050016139 | 0.039481961 | 0.037484003 | 0.035312117 | 0.05163476 | 0.0242726 | 0.025743539 |

The first logged epoch is the available initial training-loss observation; no update-zero loss was recorded. Gradient norms and parameter-update norms were not logged, so no gradient-diversity claim is made. The 5-SD jump count is a descriptive oscillation screen, not a failure test.
