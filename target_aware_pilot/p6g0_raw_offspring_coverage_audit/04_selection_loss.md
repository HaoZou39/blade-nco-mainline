# Selection Loss Audit

Selection acts on Parent+Child Union, so this audit reports both RAW-only coverage and the actual candidate-union coverage. A reference is called a selection loss only when RAW reached it but the corresponding 10k survivor did not. It is not inferred from a raw-versus-selected count difference alone.

| replicate | arm | transition | source_coverage | candidate_union_coverage | destination_coverage | lost_raw_reference_n | lost_raw_reference_ids |
|---|---|---|---|---|---|---|---|
| Original | E0 | G1_RAW_TO_SELECTED | 25 | 25 | 22 | 3 | rat99;rat783;u159 |
| Original | E0 | G2_RAW_TO_SELECTED | 24 | 24 | 23 | 1 | rat783 |
| Original | E1 | G1_RAW_TO_SELECTED | 25 | 25 | 25 | 0 |  |
| Original | E1 | G2_RAW_TO_SELECTED | 25 | 25 | 24 | 1 | pr226 |
| S1 | E0 | G1_RAW_TO_SELECTED | 24 | 24 | 21 | 3 | rat575;rat783;u159 |
| S1 | E0 | G2_RAW_TO_SELECTED | 22 | 22 | 22 | 0 |  |
| S1 | E1 | G1_RAW_TO_SELECTED | 24 | 24 | 24 | 0 |  |
| S1 | E1 | G2_RAW_TO_SELECTED | 24 | 24 | 23 | 1 | pr226 |
| S2 | E0 | G1_RAW_TO_SELECTED | 25 | 25 | 22 | 3 | rat99;rat575;rat783 |
| S2 | E0 | G2_RAW_TO_SELECTED | 24 | 24 | 23 | 1 | rat575 |
| S2 | E1 | G1_RAW_TO_SELECTED | 25 | 25 | 25 | 0 |  |
| S2 | E1 | G2_RAW_TO_SELECTED | 25 | 25 | 24 | 1 | pr226 |

Per-reference raw counts, rarity (`raw_n < 10`), parent diversity and mutation-operator evidence are in `tables/selection_loss_references.csv`. G1 and G2 transition categories are in `tables/stage_transitions.csv`.
