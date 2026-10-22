# Stage 4A final recommendation

## A — READY_FOR_TARGETED_LABELING

The frozen counterfactual is reliable: all ten cycles exactly replay the candidate, score/hash, within-niche elite, Global Top10k, and Elite-QD ordering; `|A_k|=|B_k|<=1000` holds; checkpoint identities and coordinate bundles are present; and the label/tour gaps are enumerated per instance in `03_label_availability.csv`.

| cycle | Q_n | A_T_minus_G_n | T_occupied_niches | G_occupied_niches | T_N_eff_exp_Shannon | G_N_eff_exp_Shannon | predicted_score_opportunity_cost_G_minus_T | new_current_checkpoint_labels_needed | missing_B_without_reusable_tour_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C00 | 435 | 41 | 435 | 394 | 160.470282 | 157.229472 | 0.574034 | 41 | 32 |
| C01 | 831 | 463 | 831 | 368 | 59.883619 | 45.385515 | 4.414620 | 421 | 154 |
| C02 | 991 | 711 | 991 | 280 | 88.709438 | 57.236376 | 9.927820 | 662 | 351 |
| C03 | 1000 | 800 | 1000 | 200 | 110.412715 | 68.743599 | 14.321930 | 765 | 452 |
| C04 | 1000 | 838 | 1000 | 162 | 124.028352 | 76.703194 | 22.946129 | 803 | 506 |
| C05 | 1000 | 848 | 1000 | 152 | 112.568187 | 68.910150 | 27.599267 | 820 | 477 |
| C06 | 1000 | 853 | 1000 | 147 | 110.008882 | 67.025944 | 41.746351 | 821 | 452 |
| C07 | 1000 | 873 | 1000 | 127 | 90.607187 | 52.908503 | 75.619435 | 849 | 473 |
| C08 | 1000 | 873 | 1000 | 127 | 68.446460 | 39.166295 | 70.111363 | 860 | 511 |
| C09 | 1000 | 876 | 1000 | 124 | 73.956207 | 42.127130 | 83.503246 | 862 | 544 |

The exact current-hardness comparison is **not yet identified and was not computed**. Across C00–C09, 6,904 displaced Global members still need current-checkpoint labels. In the audited tour scope, 2,952 of those can reuse a verified optimal tour and 3,952 need Concorde first. For the recommended C00+C03+C09 subset, the counts are 1,668 labels, 640 tour reuses, and 1,028 new tours.

Elite-QD's deterministic structural benefit is observable: every cycle has at least as many occupied niches as Global Top10k, and the cyclewise file reports the exact niche, entropy-effective-count, largest-share, and normalized-QD4 dispersion differences. Its HGB opportunity cost is also exact and nonnegative. Neither quantity is true LBQ hardness, future reachability, or downstream training value.

The additional work is small relative to the frozen ten-cycle run and has a precise stopping rule, so artifact repair is not required before targeted labeling. Stage 4B is not started or authorized by this decision.
