# P6-G1 Data Integrity

This audit is read-only with respect to P6-B/P6-E0/P6-G0. It performed no mutation, generation, BQ/HGB inference, Concorde labeling, or training. TSPLIB22 was never loaded.

## Artifact inventory

| replicate | arm | stage | n | unique_n | path | manifest_path |
|---|---|---|---|---|---|---|
| ALL | SHARED | P03 | 10000 | 10000 | runs/BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1/A0_FULL_BASE/cycles/C03/06_selection/P_next.npz | NA |
| Original | SHARED | G1_RAW | 109571 | 109571 | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/Original/shared_G1/mutation/offspring_coordinates.npz | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/Original/shared_G1/mutation/mutation_manifest.parquet |
| Original | E0 | G2_RAW | 109603 | 109603 | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/Original/E0/G2/mutation/offspring_coordinates.npz | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/Original/E0/G2/mutation/mutation_manifest.parquet |
| Original | E1 | G2_RAW | 109778 | 109778 | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/Original/E1/G2/mutation/offspring_coordinates.npz | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/Original/E1/G2/mutation/mutation_manifest.parquet |
| Original | E0 | G1_SELECTED | 10000 | 10000 | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/Original/E0/G1/active_population.npz | NA |
| Original | E0 | G2_SELECTED | 10000 | 10000 | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/Original/E0/G2/active_population.npz | NA |
| Original | E1 | G1_SELECTED | 10000 | 10000 | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/Original/E1/G1/active_population.npz | NA |
| Original | E1 | G2_SELECTED | 10000 | 10000 | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/Original/E1/G2/active_population.npz | NA |
| S1 | SHARED | G1_RAW | 109541 | 109541 | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S1/shared_G1/mutation/offspring_coordinates.npz | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S1/shared_G1/mutation/mutation_manifest.parquet |
| S1 | E0 | G2_RAW | 109574 | 109574 | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S1/E0/G2/mutation/offspring_coordinates.npz | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S1/E0/G2/mutation/mutation_manifest.parquet |
| S1 | E1 | G2_RAW | 109751 | 109751 | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S1/E1/G2/mutation/offspring_coordinates.npz | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S1/E1/G2/mutation/mutation_manifest.parquet |
| S1 | E0 | G1_SELECTED | 10000 | 10000 | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S1/E0/G1/active_population.npz | NA |
| S1 | E0 | G2_SELECTED | 10000 | 10000 | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S1/E0/G2/active_population.npz | NA |
| S1 | E1 | G1_SELECTED | 10000 | 10000 | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S1/E1/G1/active_population.npz | NA |
| S1 | E1 | G2_SELECTED | 10000 | 10000 | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S1/E1/G2/active_population.npz | NA |
| S2 | SHARED | G1_RAW | 109570 | 109570 | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S2/shared_G1/mutation/offspring_coordinates.npz | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S2/shared_G1/mutation/mutation_manifest.parquet |
| S2 | E0 | G2_RAW | 109619 | 109619 | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S2/E0/G2/mutation/offspring_coordinates.npz | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S2/E0/G2/mutation/mutation_manifest.parquet |
| S2 | E1 | G2_RAW | 109762 | 109762 | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S2/E1/G2/mutation/offspring_coordinates.npz | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S2/E1/G2/mutation/mutation_manifest.parquet |
| S2 | E0 | G1_SELECTED | 10000 | 10000 | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S2/E0/G1/active_population.npz | NA |
| S2 | E0 | G2_SELECTED | 10000 | 10000 | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S2/E0/G2/active_population.npz | NA |
| S2 | E1 | G1_SELECTED | 10000 | 10000 | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S2/E1/G1/active_population.npz | NA |
| S2 | E1 | G2_SELECTED | 10000 | 10000 | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S2/E1/G2/active_population.npz | NA |

G1 RAW is shared once per seed; G2 RAW is branch-specific. Every stored stage is coordinate-hash unique. Mutation manifests retain exact `parent_hash`, `child_hash`, `operator_id`, sampled parameters, validation and repair metadata, so parent–child pairing is exact rather than nearest-neighbor inferred.

## Feature integrity

- Formal F40 order is the frozen 40-column manifest; preprocessing is `asinh((raw-center_median)/scale)` using the formal scaler.
- F40 extraction: **PASS**, 959,711 new coordinates, 33.89 minutes, 32 CPU workers.
- The canonical pipeline's frozen policy is `native_HGB_missing_routing_no_imputation`. Sparse native NaNs are retained and audited in `tables/f40_native_missingness.csv`; marginal statistics use available finite values, parent–child RMS uses pairwise-available dimensions, and PCA is complete-case. No imputation was introduced.
- FFT contrast uses frozen `FFT_APPROX` + NN-ratio, TSPLIB48 median/IQR normalization and radius 0.177057522; no threshold was refit.
- EvoReal generator labels are available and retained. All synthetic sets compared here are TSP100.
