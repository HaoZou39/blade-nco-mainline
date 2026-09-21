# BLADE-NCO Cycle-0 Initialization Validation Audit

## Outcome

Phase A is **A2 — CANONICAL_REPAIR_BUT_STRUCTURAL_EFFECT_REQUIRES_REVIEW**. The adapter is historically frozen, BLADE-specific canonical domain handling, and all 110,000 raw/final hashes replay exactly. Repaired children nevertheless have a distinct migration profile and uniquely reach 12 of the 384 new niches, so the effect should be documented rather than hidden.

Phase B is **B4 — GP0_ACQUISITION_SIGNAL_INSUFFICIENT**. The smallest qualifying calibration depth is **NONE**. This is a saturation result relative to D5, not a claim that D5 is mathematically optimal. The final GP0 protocol remains unchanged pending human review.

## Phase A — repair semantics

The exact source/function trace and mathematical adapter definition are in `01_repair_semantics/repair_provenance.md`. Raw QD4 is retained only as a finite geometric diagnostic; because raw repaired candidates violate the admissible domain, it is never used downstream.

| Operator                   |   Raw n |   Repair n |   Repair % |   Native new niches |   Repaired new niches |   Median QD displacement |
|:---------------------------|--------:|-----------:|-----------:|--------------------:|----------------------:|-------------------------:|
| uniform_mutation           |   10000 |          0 |     0      |                   4 |                     0 |                0.0969241 |
| normal_mutation            |   10000 |       2913 |     0.2913 |                   5 |                     2 |                0.0598098 |
| rotation_mutation          |   10000 |       8693 |     0.8693 |                   0 |                     6 |                0.14876   |
| implosion_mutation         |   10000 |          0 |     0      |                 165 |                     0 |                0.115762  |
| explosion_mutation         |   10000 |       4855 |     0.4855 |                  14 |                    18 |                0.103059  |
| cluster_mutation           |   10000 |       1669 |     0.1669 |                 214 |                   124 |                0.49866   |
| expansion_mutation         |   10000 |       8401 |     0.8401 |                  59 |                   133 |                0.313223  |
| compression_mutation       |   10000 |       4342 |     0.4342 |                  78 |                    78 |                0.175316  |
| axis_projection_mutation   |   10000 |          0 |     0      |                 272 |                     0 |                0.359359  |
| linear_projection_mutation |   10000 |       7028 |     0.7028 |                 166 |                   213 |                0.337166  |
| grid_mutation              |   10000 |        192 |     0.0192 |                   1 |                     0 |                0.122455  |

Native valid: 71907 (65.498%); repaired valid unique: 37878 (34.502%). Native/repaired new-niche reach is 372/278; shared=266, native-only=106, repaired-only=12.

## Phase B — calibration design

| Depth   |   Nominal probes/niche |   Total train n |   Occupied niches represented |   Shortage niches |
|:--------|-----------------------:|----------------:|------------------------------:|------------------:|
| D1      |                      1 |             435 |                           435 |                 0 |
| D2      |                      2 |             822 |                           435 |                48 |
| D5      |                      5 |            1819 |                           435 |               121 |

The shortfalls are real within-niche shortages and were not replaced across niches. The sealed development panel contains 2666 geometry-only selected instances and is disjoint from CAL_D5.

## Global prediction

| Model   |   Train n |   Spearman |       R2 |   Top10 capture |   Top10 enrichment |   Pred/target SD |
|:--------|----------:|-----------:|---------:|----------------:|-------------------:|-----------------:|
| GP4_D1  |       435 |   0.326736 | 0.382813 |        0.516854 |            2.41549 |         0.530737 |
| GP4_D2  |       822 |   0.335577 | 0.422536 |        0.520599 |            2.42299 |         0.671588 |
| GP4_D5  |      1819 |   0.333158 | 0.423741 |        0.520599 |            2.42077 |         0.609246 |

All HGB4/HGB20 control results are in `08_prediction_evaluation/global_metrics.csv` and `tail_metrics.csv`.

## Within-niche ranking

| Model   |   Median rho |   Mean rho |   Median tau |   Pairwise accuracy |   Positive-niche fraction |
|:--------|-------------:|-----------:|-------------:|--------------------:|--------------------------:|
| GP4_D1  |    0.0666667 |  0.0732461 |    0.0666667 |            0.525545 |                  0.60687  |
| GP4_D2  |    0.0909091 |  0.0854793 |    0.0666667 |            0.52761  |                  0.576336 |
| GP4_D5  |    0.0787879 |  0.0662701 |    0.0666667 |            0.521117 |                  0.60687  |

Niches are equally weighted in the primary summaries; candidate-weighted values are saved as secondary diagnostics.

## Acquisition

| Model   |   Top1 true H |   Top2 true H |   Top5 true H |   Random enrichment Top1 |
|:--------|--------------:|--------------:|--------------:|-------------------------:|
| GP4_D1  |     0.069165  |     0.0647541 |     0.0640471 |                  1.48622 |
| GP4_D2  |     0.0651557 |     0.0662741 |     0.0643778 |                  1.44393 |
| GP4_D5  |     0.0680386 |     0.0648941 |     0.0637752 |                  1.57388 |

## Saturation decision

|   Depth |   Top1 ratio to D5 |   Top2 ratio |   Top5 ratio | Bootstrap gate   | Stable   | Final status   |
|--------:|-------------------:|-------------:|-------------:|:-----------------|:---------|:---------------|
|       1 |           1.01656  |     0.997842 |      1.00426 | False            | True     | False          |
|       2 |           0.957627 |     1.02126  |      1.00945 | False            | True     | False          |
|       5 |           1        |     1        |      1       | True             | True     | False          |

The decision emphasizes Top1/Top2/Top5 QD-balanced acquisition. Regression, within-niche ranking, and HGB controls are secondary diagnostics for separating calibration quantity, symbolic capacity, and QD4 information limits.

## Boundaries

No final D0 was constructed or labeled; M0 was not fine-tuned; no external benchmark was opened; no final runner was started; and `runs/` remains empty. Stop for human review.
