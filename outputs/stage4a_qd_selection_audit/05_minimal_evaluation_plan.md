# Minimal evaluation plan (not executed)

## Recommended first pass

Use C00, C03, and C09. They are suitable because they represent the initial partial-coverage phase, the first cycle with all 1,000 QD niches occupied, and the late evolutionary phase. Their candidate coordinates, entering checkpoints, frozen predictions, `Q/T/G` membership, current selected labels, and reusable-tour records are present and hash-verified.

| cycle | candidate_n | Q_n | A_T_minus_G_n | B_current_checkpoint_label_n | new_current_checkpoint_labels_needed | missing_B_with_reusable_tour_n | missing_B_without_reusable_tour_n | T_occupied_niches | G_occupied_niches | predicted_score_opportunity_cost_G_minus_T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C00 | 119781 | 435 | 41 | 0 | 41 | 9 | 32 | 435 | 394 | 0.574034 |
| C03 | 119487 | 1000 | 800 | 35 | 765 | 313 | 452 | 1000 | 200 | 14.321930 |
| C09 | 119664 | 1000 | 876 | 14 | 862 | 318 | 544 | 1000 | 124 | 83.503246 |

The union requires exactly **1,668** new current-checkpoint labels. The work decomposes into **640** BQ-only evaluations using verified reusable tours and **1,028** Concorde-plus-BQ evaluations. No additional candidate generation, HGB scoring, selection, or retraining is needed.

## Exact future procedure, only if separately authorized

1. Re-verify the file and checkpoint hashes in `01_protocol_and_frozen_inputs.md`.
2. For each missing `B=G\T` member, load coordinates by its frozen `coord_index`; reuse a verified tour when listed, otherwise obtain one Concorde tour.
3. Evaluate raw `L_BQ(M_k,x)` under the exact entering checkpoint and store `f_k(x)=log1p(L_BQ)` with instance, checkpoint, tour, and coordinate hashes.
4. Require all missing rows to be complete and finite. Do not use other-checkpoint values.
5. Only then calculate `Delta_true = sum_(G\T) f_k - sum_(T\G) f_k`. Positive favors Global Top10k on current hardness; negative favors Elite-QD; zero is equality.
6. Report each cycle as an exact finite-pool comparison. Across-cycle summaries are descriptive and dependent, not ten randomized replicates.

## If a smaller study is required

C00 alone is the smallest exact cycle: 41 labels, of which 32 currently lack a reusable tour. A paired or score/niche-stratified sample from C03 or C09 can reduce cost, but it does **not** identify the full-set sum without a declared sampling design and assumptions. Therefore sampling is not recommended as a substitute for the exact three-cycle comparison.

This plan measures current-checkpoint selection hardness only. It cannot establish future reachability or downstream BQ training value.
