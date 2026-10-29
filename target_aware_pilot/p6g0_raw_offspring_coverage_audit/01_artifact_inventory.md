# P6-G0 Artifact Inventory

`P6_G0_RAW_OFFSPRING_AVAILABLE = YES`

All required historical stages exist. G1 RAW is shared by E0/E1 within each seed. G2 RAW is branch-specific because it is generated from the selected G1 parents. `offspring_coordinates.npz` is used for RAW; final active populations and the E0∪E1 ~20k downstream union are never substituted for RAW. Counts come from the actual NPZ and 110k-attempt manifests, not a nominal assumption.

| replicate | arm | stage | artifact_type | n | unique_n | nominal_attempts | accepted_attempts | path |
|---|---|---|---|---|---|---|---|---|
| ALL | SHARED | P03 | population | 10000 | 10000 | NA | NA | runs/BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1/A0_FULL_BASE/cycles/C03/06_selection/P_next.npz |
| Original | SHARED | G1_RAW | raw_offspring | 109571 | 109571 | 110000 | 109571 | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/Original/shared_G1/mutation/offspring_coordinates.npz |
| Original | SHARED | G1_CANDIDATE_UNION | parent_child_union | 119548 | 119548 | NA | NA | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/Original/shared_G1/candidate_union.npz |
| Original | E0 | G2_RAW | raw_offspring | 109603 | 109603 | 110000 | 109603 | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/Original/E0/G2/mutation/offspring_coordinates.npz |
| Original | E0 | G2_CANDIDATE_UNION | parent_child_union | 119594 | 119594 | NA | NA | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/Original/E0/G2/candidate_union.npz |
| Original | E1 | G2_RAW | raw_offspring | 109778 | 109778 | 110000 | 109778 | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/Original/E1/G2/mutation/offspring_coordinates.npz |
| Original | E1 | G2_CANDIDATE_UNION | parent_child_union | 119766 | 119766 | NA | NA | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/Original/E1/G2/candidate_union.npz |
| Original | E0 | G1_SELECTED | active_population | 10000 | 10000 | NA | NA | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/Original/E0/G1/active_population.npz |
| Original | E0 | G2_SELECTED | active_population | 10000 | 10000 | NA | NA | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/Original/E0/G2/active_population.npz |
| Original | E1 | G1_SELECTED | active_population | 10000 | 10000 | NA | NA | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/Original/E1/G1/active_population.npz |
| Original | E1 | G2_SELECTED | active_population | 10000 | 10000 | NA | NA | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/Original/E1/G2/active_population.npz |
| S1 | SHARED | G1_RAW | raw_offspring | 109541 | 109541 | 110000 | 109541 | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S1/shared_G1/mutation/offspring_coordinates.npz |
| S1 | SHARED | G1_CANDIDATE_UNION | parent_child_union | 119523 | 119523 | NA | NA | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S1/shared_G1/candidate_union.npz |
| S1 | E0 | G2_RAW | raw_offspring | 109574 | 109574 | 110000 | 109574 | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S1/E0/G2/mutation/offspring_coordinates.npz |
| S1 | E0 | G2_CANDIDATE_UNION | parent_child_union | 119558 | 119558 | NA | NA | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S1/E0/G2/candidate_union.npz |
| S1 | E1 | G2_RAW | raw_offspring | 109751 | 109751 | 110000 | 109751 | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S1/E1/G2/mutation/offspring_coordinates.npz |
| S1 | E1 | G2_CANDIDATE_UNION | parent_child_union | 119742 | 119742 | NA | NA | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S1/E1/G2/candidate_union.npz |
| S1 | E0 | G1_SELECTED | active_population | 10000 | 10000 | NA | NA | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S1/E0/G1/active_population.npz |
| S1 | E0 | G2_SELECTED | active_population | 10000 | 10000 | NA | NA | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S1/E0/G2/active_population.npz |
| S1 | E1 | G1_SELECTED | active_population | 10000 | 10000 | NA | NA | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S1/E1/G1/active_population.npz |
| S1 | E1 | G2_SELECTED | active_population | 10000 | 10000 | NA | NA | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S1/E1/G2/active_population.npz |
| S2 | SHARED | G1_RAW | raw_offspring | 109570 | 109570 | 110000 | 109570 | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S2/shared_G1/mutation/offspring_coordinates.npz |
| S2 | SHARED | G1_CANDIDATE_UNION | parent_child_union | 119553 | 119553 | NA | NA | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S2/shared_G1/candidate_union.npz |
| S2 | E0 | G2_RAW | raw_offspring | 109619 | 109619 | 110000 | 109619 | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S2/E0/G2/mutation/offspring_coordinates.npz |
| S2 | E0 | G2_CANDIDATE_UNION | parent_child_union | 119606 | 119606 | NA | NA | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S2/E0/G2/candidate_union.npz |
| S2 | E1 | G2_RAW | raw_offspring | 109762 | 109762 | 110000 | 109762 | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S2/E1/G2/mutation/offspring_coordinates.npz |
| S2 | E1 | G2_CANDIDATE_UNION | parent_child_union | 119754 | 119754 | NA | NA | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S2/E1/G2/candidate_union.npz |
| S2 | E0 | G1_SELECTED | active_population | 10000 | 10000 | NA | NA | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S2/E0/G1/active_population.npz |
| S2 | E0 | G2_SELECTED | active_population | 10000 | 10000 | NA | NA | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S2/E0/G2/active_population.npz |
| S2 | E1 | G1_SELECTED | active_population | 10000 | 10000 | NA | NA | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S2/E1/G1/active_population.npz |
| S2 | E1 | G2_SELECTED | active_population | 10000 | 10000 | NA | NA | target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S2/E1/G2/active_population.npz |

All coordinate sets are finite and hash-unique at the stored stage. Input hashes are retained in `tables/artifact_inventory.csv` and `manifests/input_hashes.csv`.
