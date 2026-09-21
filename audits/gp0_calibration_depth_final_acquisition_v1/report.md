# BLADE-NCO GP0 Calibration Depth Audit Under Final Acquisition Role

## Outcome

**C1 - ONE_PROBE_PER_REACHED_NICHE_SUFFICIENT**. The smallest qualifying existing branch is **1**. This is a read-only operational-sufficiency finding; the mainline protocol was not edited.

## Calibration branches

| model   |   depth |   train_n |   GP_fill_gain |   gain_ci_low |   gain_ci_high |   gain_retention |   delta_vs_D5 |   delta_ci_low |   delta_ci_high |   noninferiority_margin | operationally_saturated_relative_to_D5   | selected_as_minimum   | final_classification   |
|:--------|--------:|----------:|---------------:|--------------:|---------------:|-----------------:|--------------:|---------------:|----------------:|------------------------:|:-----------------------------------------|:----------------------|:-----------------------|
| GP4_D1  |       1 |       435 |       0.070842 |      0.053367 |       0.089772 |         0.998715 |     -0.000091 |      -0.000863 |        0.000644 |               -0.003547 | True                                     | True                  | C1                     |
| GP4_D2  |       2 |       822 |       0.071309 |      0.053798 |       0.090266 |         1.005298 |      0.000376 |      -0.000425 |        0.001468 |               -0.003547 | True                                     | False                 | C1                     |
| GP4_D5  |       5 |      1819 |       0.070933 |      0.053464 |       0.089885 |         1.000000 |      0.000000 |       0.000000 |        0.000000 |               -0.003547 | True                                     | False                 | C1                     |

## Primary and sensitivity GP-fill acquisition

| model   |   budget |   fill_n |   GP_fill_H0 |   random_fill_H0 |   GP_fill_gain |   gain_ci_low |   gain_ci_high |
|:--------|---------:|---------:|-------------:|-----------------:|---------------:|--------------:|---------------:|
| GP4_D1  |      602 |      301 |     0.132035 |         0.061194 |       0.070842 |      0.053367 |       0.089772 |
| GP4_D2  |      602 |      301 |     0.132502 |         0.061194 |       0.071309 |      0.053798 |       0.090266 |
| GP4_D5  |      602 |      301 |     0.132126 |         0.061194 |       0.070933 |      0.053464 |       0.089885 |
| GP4_D1  |     1204 |      903 |     0.080029 |         0.061216 |       0.018813 |      0.011701 |       0.026502 |
| GP4_D2  |     1204 |      903 |     0.081585 |         0.061216 |       0.020369 |      0.013379 |       0.027930 |
| GP4_D5  |     1204 |      903 |     0.081603 |         0.061216 |       0.020387 |      0.013456 |       0.027907 |
| GP4_D1  |     2408 |     2107 |     0.063469 |         0.061199 |       0.002270 |      0.000913 |       0.003826 |
| GP4_D2  |     2408 |     2107 |     0.063362 |         0.061199 |       0.002163 |      0.000830 |       0.003702 |
| GP4_D5  |     2408 |     2107 |     0.063493 |         0.061199 |       0.002294 |      0.000925 |       0.003875 |

## Whole hybrid populations

| model   |   budget |   centroid_n |   GP_fill_n |   hybrid_H0 |
|:--------|---------:|-------------:|------------:|------------:|
| GP4_D1  |      602 |          301 |         301 |    0.097572 |
| GP4_D2  |      602 |          301 |         301 |    0.097805 |
| GP4_D5  |      602 |          301 |         301 |    0.097618 |
| GP4_D1  |     1204 |          301 |         903 |    0.075799 |
| GP4_D2  |     1204 |          301 |         903 |    0.076966 |
| GP4_D5  |     1204 |          301 |         903 |    0.076979 |
| GP4_D1  |     2408 |          301 |        2107 |    0.063424 |
| GP4_D2  |     2408 |          301 |        2107 |    0.063331 |
| GP4_D5  |     2408 |          301 |        2107 |    0.063445 |

## Oracle fill headroom

| model   |   budget |   random_fill_H0 |   model_fill_H0 |   oracle_fill_H0 |   fill_headroom_capture |
|:--------|---------:|-----------------:|----------------:|-----------------:|------------------------:|
| GP4_D1  |      602 |         0.061194 |        0.132035 |         0.169674 |                0.653035 |
| GP4_D2  |      602 |         0.061194 |        0.132502 |         0.169674 |                0.657340 |
| GP4_D5  |      602 |         0.061194 |        0.132126 |         0.169674 |                0.653876 |
| GP4_D1  |     1204 |         0.061216 |        0.080029 |         0.105997 |                0.420101 |
| GP4_D2  |     1204 |         0.061216 |        0.081585 |         0.105997 |                0.454858 |
| GP4_D5  |     1204 |         0.061216 |        0.081603 |         0.105997 |                0.455252 |
| GP4_D1  |     2408 |         0.061199 |        0.063469 |         0.066869 |                0.400307 |
| GP4_D2  |     2408 |         0.061199 |        0.063362 |         0.066869 |                0.381566 |
| GP4_D5  |     2408 |         0.061199 |        0.063493 |         0.066869 |                0.404565 |

## Direct paired comparisons

| comparison   |   budget | metric       |     delta |    ci_low |   ci_high |
|:-------------|---------:|:-------------|----------:|----------:|----------:|
| D1_minus_D5  |      602 | GP_FILL_H    | -0.000091 | -0.000863 |  0.000644 |
| D1_minus_D5  |      602 | GP_FILL_GAIN | -0.000091 | -0.000863 |  0.000644 |
| D1_minus_D5  |      602 | HYBRID_H     | -0.000046 | -0.000285 |  0.000145 |
| D2_minus_D5  |      602 | GP_FILL_H    |  0.000376 | -0.000425 |  0.001468 |
| D2_minus_D5  |      602 | GP_FILL_GAIN |  0.000376 | -0.000425 |  0.001468 |
| D2_minus_D5  |      602 | HYBRID_H     |  0.000188 | -0.000094 |  0.000627 |
| D1_minus_D2  |      602 | GP_FILL_H    | -0.000467 | -0.001767 |  0.000663 |
| D1_minus_D2  |      602 | GP_FILL_GAIN | -0.000467 | -0.001767 |  0.000663 |
| D1_minus_D2  |      602 | HYBRID_H     | -0.000233 | -0.000727 |  0.000130 |
| D1_minus_D5  |     1204 | GP_FILL_H    | -0.001574 | -0.003443 |  0.000344 |
| D1_minus_D5  |     1204 | GP_FILL_GAIN | -0.001574 | -0.003443 |  0.000344 |
| D1_minus_D5  |     1204 | HYBRID_H     | -0.001181 | -0.002419 |  0.000120 |
| D2_minus_D5  |     1204 | GP_FILL_H    | -0.000018 | -0.001437 |  0.001469 |
| D2_minus_D5  |     1204 | GP_FILL_GAIN | -0.000018 | -0.001437 |  0.001469 |
| D2_minus_D5  |     1204 | HYBRID_H     | -0.000013 | -0.000999 |  0.001014 |
| D1_minus_D2  |     1204 | GP_FILL_H    | -0.001556 | -0.003470 |  0.000372 |
| D1_minus_D2  |     1204 | GP_FILL_GAIN | -0.001556 | -0.003470 |  0.000372 |
| D1_minus_D2  |     1204 | HYBRID_H     | -0.001167 | -0.002446 |  0.000129 |
| D1_minus_D5  |     2408 | GP_FILL_H    | -0.000024 | -0.000260 |  0.000208 |
| D1_minus_D5  |     2408 | GP_FILL_GAIN | -0.000024 | -0.000260 |  0.000208 |
| D1_minus_D5  |     2408 | HYBRID_H     | -0.000021 | -0.000228 |  0.000182 |
| D2_minus_D5  |     2408 | GP_FILL_H    | -0.000130 | -0.000697 |  0.000426 |
| D2_minus_D5  |     2408 | GP_FILL_GAIN | -0.000130 | -0.000697 |  0.000426 |
| D2_minus_D5  |     2408 | HYBRID_H     | -0.000114 | -0.000607 |  0.000374 |
| D1_minus_D2  |     2408 | GP_FILL_H    |  0.000106 | -0.000441 |  0.000678 |
| D1_minus_D2  |     2408 | GP_FILL_GAIN |  0.000106 | -0.000441 |  0.000678 |
| D1_minus_D2  |     2408 | HYBRID_H     |  0.000093 | -0.000387 |  0.000591 |

## Full-bank selection stability

| comparison   |   GP_fill_intersection |   GP_fill_jaccard |   GP_fill_overlap_fraction |   hybrid_jaccard |
|:-------------|-----------------------:|------------------:|---------------------------:|-----------------:|
| D1_vs_D5     |                   8651 |          0.825556 |                   0.904443 |         0.832509 |
| D2_vs_D5     |                   8654 |          0.826079 |                   0.904757 |         0.833013 |
| D1_vs_D2     |                   8710 |          0.835893 |                   0.910612 |         0.842469 |

| comparison   |   spearman |   top1pct_overlap_fraction |   top5pct_overlap_fraction |   top10pct_overlap_fraction |
|:-------------|-----------:|---------------------------:|---------------------------:|----------------------------:|
| D1_vs_D2     |   0.374108 |                   0.854662 |                   0.899781 |                    0.912666 |
| D1_vs_D5     |   0.939411 |                   0.869287 |                   0.927762 |                    0.927846 |
| D2_vs_D5     |   0.333177 |                   0.958867 |                   0.923189 |                    0.922725 |

Cross-scoring is predicted-score diagnostics only, never exact-hardness evidence:

| selection_model   |   selection_n | diagnostic_only   |   mean_score_GP4_D1 |   mean_score_GP4_D2 |   mean_score_GP4_D5 |
|:------------------|--------------:|:------------------|--------------------:|--------------------:|--------------------:|
| GP4_D1            |          9565 | True              |            0.068556 |            0.076139 |            0.073179 |
| GP4_D2            |          9565 | True              |            0.068082 |            0.076220 |            0.072685 |
| GP4_D5            |          9565 | True              |            0.068354 |            0.076126 |            0.073235 |

| model   |   selected_n |   centroid_floor_n |   GP_fill_n |   occupied_niches |   root_diversity |   operator_count |   operator_entropy |   niche_entropy |   normalized_niche_entropy |   niche_gini |
|:--------|-------------:|-------------------:|------------:|------------------:|-----------------:|-----------------:|-------------------:|----------------:|---------------------------:|-------------:|
| GP4_D1  |        10000 |                435 |        9565 |               435 |             7052 |               11 |           1.336097 |        5.060289 |                   0.832922 |     0.720398 |
| GP4_D2  |        10000 |                435 |        9565 |               435 |             7007 |               11 |           1.479191 |        5.118394 |                   0.842486 |     0.701164 |
| GP4_D5  |        10000 |                435 |        9565 |               435 |             7100 |               11 |           1.364705 |        5.077270 |                   0.835717 |     0.713255 |

## Sensitivity

Directional status: **SENSITIVITY_DIRECTION_SENSITIVE**. B=301 is excluded from calibration selection because it has zero GP fill.

## Required limitation

The sealed dev panel cannot exactly reproduce the full candidate-bank operating ratio. After removing the centroid floor, the full-bank GP-fill fraction is 9565/109350 = 0.087471; the primary dev B=602 fill fraction is 301/2365 = 0.127273. B=602 is the closest feasible exact-label high-selection-pressure proxy, but this remains development validation rather than exact-H0 evaluation of all 109,785 candidates.

No new instances, labels, Concorde calls, BQ hardness inference, training, fine-tuning, external evaluation, FINAL_D0, protocol modification, archive update, Cycle-0 launch, or final runner occurred. `runs/` remains empty.

Strict verification: PASS (57/57). Tests: 68/68 passed.
