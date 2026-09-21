# Real capability-boundary visualization v2

Read-only visualization refactor. No model was trained, no benchmark or structural feature was changed, and no solver result was recomputed.

## Frozen inputs and common visual protocol

- `audits/external_benchmark_suite_v2/real_capability_boundary_visualization/01_instance_capability_coordinates.csv` — SHA-256 `1ed3d57a9a9253a11a6bd0f8c77647611f8a9ff19fd6fe5bf1dca98662408634`
- `audits/external_benchmark_suite_v2/real_capability_boundary_visualization/03_real_hard_region_membership.csv` — SHA-256 `8c311ac2c6b7358a12985745aa97cf675cc668a7cfa4d5e478a9897ab76fddc0`
- `audits/external_benchmark_suite_v2/real_capability_boundary_visualization/05_hard_region_summary.csv` — SHA-256 `defc310f1f971361b21a4223e968e96f2931daf9e68f9fe9c1053679dc459e9f`
- `audits/external_benchmark_suite_v2/real_capability_boundary_visualization/06_consistency_checks.csv` — SHA-256 `737c43035c497dccd14c5db588e97085ffe669bfea4150d5e84be0ef977587ac`
- `audits/external_benchmark_suite_v2/real_capability_boundary_visualization/09_m5_instance_capability.csv` — SHA-256 `e3095b56bc43001a9239e580fecc3301b94c1dda85e64acf7b7ad8ccdda389f3`

- Eligible instances: 30,070; identical for M0, M1, and M5.
- Foreground subset: 280 points, with 70 structurally covering points from each benchmark family; exactly the same points appear in all three capability maps.
- Structural map: the frozen pooled PCA coordinates from v1; PCA was not refit.
- Fixed threshold: τ=1.675967% (M0 pooled P70), unchanged.
- Shared field normalization: pooled M0/M1/M5 P5=0.000000%, τ at the white midpoint, and pooled P95=8.362600%.
- All three maps use identical coordinates, axis limits, point sizes, plot dimensions, support mask, and color normalization. Only solver gaps, observed point labels, smoothed field, and capability boundary change.

## Transition summary

| quantity | M0 → M1 | M0 → M5 |
|---|---:|---:|
| mastered fraction | 70.00% → 74.77% | 70.00% → 82.52% |
| hard → mastered | 2,535 | 4,735 |
| mastered → hard | 1,100 | 971 |
| mean gap | 2.561% → 1.905% | 2.561% → 1.093% |
| supported-grid expansion | 4.22% | 19.57% |
| supported-grid contraction | 0.07% | 0.02% |

M0→M1 supplies the scientifically closest adjacent-cycle evidence: one update already reshapes the capability boundary, but the movement is relatively local. M0→M5 supplies the clearer presentation result: the longer-horizon boundary is predominantly expansive and is visually easier to distinguish from B₀.

**Recommended Page 2 update figure: M5 vs M0.** It preserves the same scientific construction while communicating the boundary movement more clearly in a PPT-sized panel. M1 vs M0 should remain available as adjacent-cycle evidence in the report or supplement.

## Hard-region interpretation

High-gap instances are not structurally homogeneous. At least two structurally distinct failure regions emerge under M0; the result should not be overstated as many stable modes.

- Hard region A: n=8,634; composition GM100=7,291, TSPLIB100=1,337, Uniform100=0, Original70=6; representative `GM100::gaussian_mixture100_cdist30_0473`.
- Hard region B: n=237; composition GM100=1, TSPLIB100=49, Uniform100=187, Original70=0; representative `Uniform100::fresh_uniform100_02407`.
- Noise / minor modes: 150.

The v2 hard-region figure keeps the frozen clustering labels but replaces the dominant covariance ellipses with light, data-derived 70% density contours.

## Answers to the requested questions

1. **Same frozen structural space?** Yes. M0/M1/M5 reuse one feature table and one frozen PCA coordinate table.
2. **Same subset, coordinates, axes, and color scale?** Yes; all exact-equality and shared-style checks pass below.
3. **M0→M1?** Mastered 70.00%→74.77%; hard→mastered 2,535; mastered→hard 1,100; expansion/contraction 4.22%/0.07%.
4. **M0→M5?** Mastered 70.00%→82.52%; hard→mastered 4,735; mastered→hard 971; expansion/contraction 19.57%/0.02%.
5. **Best Page 2 update panel?** M5 vs M0 for visual clarity; M1 vs M0 for the narrow adjacent-cycle scientific claim.
6. **Structural heterogeneity under M0?** Yes: two retained regions plus minor/noise modes, discovered in the original normalized structural space.
7. **Composition and representatives?** Listed above and retained without relabeling or synthetic geometry.
8. **Final Page 2 composition?** See the explicit recommendation below.

## Final recommendation

Recommended Page 2 composition:

- left: M0 capability map
- middle: M5 vs M0 capability update
- right: diverse hard regions, with the representative instances used as a compact inset or immediate follow-up panel

This combination most clearly supports both claims: “the capability boundary moves with the solver” and “high-gap instances are structurally diverse.”

## Consistency checks

| check                                  | value                                                              | pass   |
|:---------------------------------------|:-------------------------------------------------------------------|:-------|
| M0/M1/M5 point coordinates identical   | max_abs=0.0                                                        | True   |
| M0/M1/M5 instance IDs identical        | 280                                                                | True   |
| structural features identical          | single frozen table reused                                         | True   |
| same fixed tau                         | 1.6759665429999124                                                 | True   |
| same diverging color normalization     | P5=0; tau=1.6759665429999124; P95=8.3625999754092408               | True   |
| same axis range                        | x=(-1.1282992041, 0.9143546032); y=(-0.5397302287, 0.7931749593)   | True   |
| same 280-point subset                  | {"GM100": 70, "Original70": 70, "TSPLIB100": 70, "Uniform100": 70} | True   |
| same point sizes and figure dimensions | handled=24.0; hard=31.0; figsize=(7.2, 6.2)                        | True   |
| M0 field identical in both comparisons | True                                                               | True   |
| support mask identical                 | True                                                               | True   |
| M1/M5 final field disagreement <=10%   | M1=0.039907%; M5=0.086465%                                         | True   |
| hard-region membership unchanged       | 9021                                                               | True   |
