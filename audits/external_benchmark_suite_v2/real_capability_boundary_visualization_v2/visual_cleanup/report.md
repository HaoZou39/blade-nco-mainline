# M0 / M1 / M5 capability-field visual cleanup

Read-only visual refactor. No instance, feature, PCA coordinate, exact gap, point label, threshold, mastered fraction, transition count, or quantitative support statistic was changed.

## Frozen sources

- `audits/external_benchmark_suite_v2/real_capability_boundary_visualization/01_instance_capability_coordinates.csv` — SHA-256 `1ed3d57a9a9253a11a6bd0f8c77647611f8a9ff19fd6fe5bf1dca98662408634`
- `audits/external_benchmark_suite_v2/real_capability_boundary_visualization/03_real_hard_region_membership.csv` — SHA-256 `8c311ac2c6b7358a12985745aa97cf675cc668a7cfa4d5e478a9897ab76fddc0`
- `audits/external_benchmark_suite_v2/real_capability_boundary_visualization/05_hard_region_summary.csv` — SHA-256 `defc310f1f971361b21a4223e968e96f2931daf9e68f9fe9c1053679dc459e9f`
- `audits/external_benchmark_suite_v2/real_capability_boundary_visualization/06_consistency_checks.csv` — SHA-256 `737c43035c497dccd14c5db588e97085ffe669bfea4150d5e84be0ef977587ac`
- `audits/external_benchmark_suite_v2/real_capability_boundary_visualization/09_m5_instance_capability.csv` — SHA-256 `e3095b56bc43001a9239e580fecc3301b94c1dda85e64acf7b7ad8ccdda389f3`
- `audits/external_benchmark_suite_v2/real_capability_boundary_visualization_v2/01_instance_capability_coordinates.csv` — SHA-256 `927323d5128e84298b4881d7209f392ffc8e405529742308db4a04f4662993e4`
- `audits/external_benchmark_suite_v2/real_capability_boundary_visualization_v2/04_transition_summary_M0_M1_M5.csv` — SHA-256 `df0a772791b0189a06a4a3d3bef9dfcde51b94da484045facfb1ed2d922c6048`

## Visual-field decisions

1. **Smoothing method and parameter.** The selected method is Gaussian post-smoothing with `sigma=2.0` grid cells, applied to the existing distance-weighted kNN (`k=75`) predicted grid. It was compared with broader kNN kernels (`k=100,150,225,300`) and Gaussian sigma 1.0–3.0. At sigma 2.0, supported-grid capability Jaccard versus the original field is M0=0.983, M1=0.986, M5=0.968, while raw τ contours fall to 9/5/8.
2. **Scope of smoothing.** Smoothing affects only the PPT/display grid. It is never written back to instance gaps and is not used to recompute any reported quantity.
3. **Point integrity.** All displayed point labels are recomputed only as exact frozen `gap <= τ`; source gaps and labels are byte-for-byte untouched.
4. **Unsupported area.** Supported field cells use the blue–white–pale-red gap map. Unsupported cells use neutral `#F4F4F4`; therefore white means near τ, not missing support.
5. **Colorbar.** The former 8.362600% upper bound is exactly the pooled M0/M1/M5 P95. The audit also compares pooled P90 (5.224540%) and fixed 5%. `COLORBAR_VMAX_SELECTED = 5.0%`. `COLORBAR_VMAX_REASON = round PPT scale close to pooled P90, improving τ-to-moderately-hard contrast; values above it are explicitly labeled ≥5%.` The same scale is used for all solvers.
6. **Foreground subset.** `N_foreground = 200`: 50 points per benchmark, chosen by deterministic farthest-point coverage within M0 hard/handled strata. Exact solver labels remain independent. Counts are M0 hard=63, M1 hard=42, M5 hard=31.
7. **Boundary cleanup.** Raw→retained component counts are M0: 9→2, M1: 5→1, M5: 8→3. Each component's length and enclosed area are audited. The automatic rule retains length >= `max(3% of plot diagonal, within-model P60 component length)`; it never forces one component.
8. **Shared geometry.** Yes. M0/M1/M5 use the exact same 200 instance IDs, PC coordinates, axis limits (`x=(-1.1282992041, 0.9143546032)`, `y=(-0.5397302287, 0.7931749593)`), support mask, colormap, normalization, marker sizes, and styles.
9. **Page 2 choice.** Yes: M0 plus M5-vs-M0 remains the clearest main evidence. M1-vs-M0 remains the one-cycle backup: **Solver adaptation reshapes the capability boundary after one cycle.**
10. **Best direct PPT export.** `fig_M0_vs_M5_ppt_composite.png` is recommended because it has shared axes, one colorbar, one legend, the same subset, and directly exposes B₀→B₅.

## Quantitative results retained unchanged

- M0: mastered fraction 70.00%; mean gap 2.561%.
- M0→M1: 70.00%→74.77%; hard→mastered 2,535; mastered→hard 1,100; expansion/contraction 4.22%/0.07%. Recommended wording: **Solver adaptation reshapes the capability boundary after one cycle.**
- M0→M5: mastered 70.00%→82.52%; mean gap 2.561%→1.093%; hard→mastered 4,735; mastered→hard 971; expansion/contraction 19.57%/0.02%. Recommended wording: **The capability boundary is predominantly expansive over the longer horizon.**

The display does not imply uniform expansion or equal improvement across all regions.

`RECOMMENDED_PAGE2_CAPABILITY_FIGURE = fig_M0_vs_M5_ppt_composite.png`
