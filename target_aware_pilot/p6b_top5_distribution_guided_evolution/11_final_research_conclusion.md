# P6-B Final Research Conclusion

## H1: `SUPPORTED`
## H2: `NOT_SUPPORTED`

Delta = gap(E0) − gap(E1); positive favors E1.

| Replicate | Original70 E0 | Original70 E1 | Delta | 95% CI | TSPLIB22 Delta | GAP5 Delta |
|---|---:|---:|---:|---|---:|---:|
| Original | 2.160443 | 2.761475 | -0.601031 | [-1.190776, -0.093111] | -0.601873 | +0.011882 |
| S1 | 2.370019 | 2.409727 | -0.039708 | [-0.462476, 0.407951] | +0.191144 | -0.191217 |
| S2 | 2.253034 | 2.608024 | -0.354990 | [-0.967426, 0.200787] | -0.431514 | -0.117330 |

H1 is based on all three final P2 exact-OT directions (3/3 favor E1); OT is the optimized objective, while SWD and energy are supporting structural diagnostics.
H2 is the core decision: E1 wins Original70 in 0/3 independent evolution replicates. Instance-level bootstrap intervals are conditional within replicate and do not create additional evolution replicates; training-seed variability is unmeasured.
Original70 is target-informed and TSPLIB22 has been repeatedly observed. No checkpoint or protocol was chosen from evaluation results.

`BLADE_P6B_TOP5_DISTRIBUTION_GUIDED_EVOLUTION_COMPLETE`

`H1: SUPPORTED`
`H2: NOT_SUPPORTED`
