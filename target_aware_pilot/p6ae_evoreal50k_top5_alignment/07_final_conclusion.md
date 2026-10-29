# P6-AE Final Conclusion

## `EVOREAL_TOP5_ALIGNMENT_SUPPORTED` — `SUPPORTED`

- Complete trained EvoReal50k bundle found and verified: yes; S1/S2/S3 = 19,792/12,500/17,708.
- Frozen P6-A representation reused without refitting: yes.
- EvoReal versus Uniform: OT `8.689244` vs `13.518932`; SWD `0.796259` vs `1.163749`.
- EvoReal versus P03: OT `8.689244` vs `26.610826`.
- EvoReal versus P5 E1 Archive Structure10k mean across Original/S1/S2: OT `8.689244` vs `21.193148`; individual seeds remain in the table.
- The three generators are separately quantified at equal K; pairwise SWD/assignment-mass JS show whether their structural coverage differs.
- Historical-performance direction is GAP5 一致、Original70 不一致（mixed）; no causal claim is made.
- P6-B implication: this result tests whether frozen Top5 alignment is compatible with a known useful source, but it neither validates Top5 sufficiency nor authorizes P6-B automatically.

## Non-removable limitations

- EvoReal generator construction is itself development-TSPLIB-informed, so comparison to TSPLIB48 is not independent.
- Forty-eight empirical targets are sparse in five dimensions; exact equal-mass OT is well-defined but target sampling uncertainty remains.
- P5 results span historical Original plus two new evolution seeds and are reported separately; they do not estimate all evolution randomness.
- EvoReal uses 50k/5 cycles whereas each primary draw and each BLADE set has 10k; equal-size distribution comparison does not equalize training history.
- Top5 came from an M4-conditioned surrogate and may omit training-relevant structure; P6-A already identified `distance_min` dominance/calibration-tail risk.
- Old2 JS/SWD is a historical compatibility view and does not replace frozen five-dimensional OT.

`BLADE_P6AE_EVOREAL50K_TOP5_ALIGNMENT_AUDIT_COMPLETE`
