# BLADE-NCO Cycle-0 Acquisition Policy Replay v1

## Outcome

**A1 - CENTROID_COVERAGE_PLUS_GLOBAL_GP_SUPPORTED**. This is an analysis-only replay; it does not freeze an acquisition policy or alter the mainline protocol.

## Full-bank replay

All five policies selected exactly 10,000 unique hashes from the exact 109,785-candidate Cycle-0 offspring bank. GP scores are predictions from the median of the five sealed GP4_D5 formulas and are diagnostic only--not exact-hardness evidence.

| policy                |   selected_n |   occupied_niches_represented |   coverage_fraction |   normalized_niche_entropy |   niche_count_gini |   mean_gp_score |   P90_gp_score |   unique_uniform_roots |   operator_count |
|:----------------------|-------------:|------------------------------:|--------------------:|---------------------------:|-------------------:|----------------:|---------------:|-----------------------:|-----------------:|
| GLOBAL_GP             |        10000 |                           364 |            0.836782 |                   0.829164 |           0.725904 |        0.072646 |       0.141016 |                   7110 |                6 |
| ROUND_ROBIN_GP        |        10000 |                           435 |            1.000000 |                   0.939996 |           0.420571 |        0.059453 |       0.068554 |                   6536 |               11 |
| CENTROID1_GLOBAL_GP   |        10000 |                           435 |            1.000000 |                   0.835717 |           0.713255 |        0.072583 |       0.141016 |                   7100 |               11 |
| CENTROID1_RANDOM_FILL |        10000 |                           435 |            1.000000 |                   0.759490 |           0.811048 |        0.043234 |       0.051138 |                   6509 |               11 |
| GLOBAL_RANDOM         |        10000 |                           292 |            0.671264 |                   0.734468 |           0.846479 |        0.043111 |       0.050864 |                   6475 |               11 |

The predicted score cost of the CENTROID1 coverage floor is reported in the full-bank policy table and is not used alone for the decision.

## Sealed exact-hardness dev replay

|   budget | policy                  |   selected_n |   occupied_niches_represented |   coverage_fraction |   mean_H0 |   P90_H0 |   gain_over_global_random |
|---------:|:------------------------|-------------:|------------------------------:|--------------------:|----------:|---------:|--------------------------:|
|      301 | GLOBAL_GP               |          301 |                            42 |            0.139535 |  0.140513 | 0.287508 |                  0.076622 |
|      301 | ROUND_ROBIN_GP          |          301 |                           301 |            1.000000 |  0.070578 | 0.121788 |                  0.006687 |
|      301 | CENTROID1_GLOBAL_GP     |          301 |                           301 |            1.000000 |  0.063109 | 0.109307 |                 -0.000782 |
|      301 | CENTROID1_RANDOM_FILL   |          301 |                           301 |            1.000000 |  0.063109 | 0.109307 |                 -0.000782 |
|      301 | GLOBAL_RANDOM           |          301 |                           182 |            0.604651 |  0.063891 | 0.126103 |                  0.000000 |
|      301 | GLOBAL_ORACLE           |          301 |                           143 |            0.475083 |  0.177751 | 0.296643 |                  0.113860 |
|      301 | CENTROID1_GLOBAL_ORACLE |          301 |                           301 |            1.000000 |  0.063109 | 0.109307 |                 -0.000782 |
|      602 | GLOBAL_GP               |          602 |                           110 |            0.365449 |  0.101287 | 0.202187 |                  0.040439 |
|      602 | ROUND_ROBIN_GP          |          602 |                           301 |            1.000000 |  0.066867 | 0.121662 |                  0.006019 |
|      602 | CENTROID1_GLOBAL_GP     |          602 |                           301 |            1.000000 |  0.097618 | 0.201628 |                  0.036770 |
|      602 | CENTROID1_RANDOM_FILL   |          602 |                           301 |            1.000000 |  0.063199 | 0.110657 |                  0.002351 |
|      602 | GLOBAL_RANDOM           |          602 |                           263 |            0.873754 |  0.060848 | 0.109791 |                  0.000000 |
|      602 | GLOBAL_ORACLE           |          602 |                           237 |            0.787375 |  0.133784 | 0.215922 |                  0.072936 |
|      602 | CENTROID1_GLOBAL_ORACLE |          602 |                           301 |            1.000000 |  0.116391 | 0.215922 |                  0.055544 |
|     1204 | GLOBAL_GP               |         1204 |                           197 |            0.654485 |  0.077648 | 0.154738 |                  0.017926 |
|     1204 | ROUND_ROBIN_GP          |         1204 |                           301 |            1.000000 |  0.064420 | 0.119710 |                  0.004698 |
|     1204 | CENTROID1_GLOBAL_GP     |         1204 |                           301 |            1.000000 |  0.076979 | 0.153608 |                  0.017257 |
|     1204 | CENTROID1_RANDOM_FILL   |         1204 |                           301 |            1.000000 |  0.058822 | 0.104627 |                 -0.000900 |
|     1204 | GLOBAL_RANDOM           |         1204 |                           295 |            0.980066 |  0.059722 | 0.107827 |                  0.000000 |
|     1204 | GLOBAL_ORACLE           |         1204 |                           295 |            0.980066 |  0.098155 | 0.159754 |                  0.038432 |
|     1204 | CENTROID1_GLOBAL_ORACLE |         1204 |                           301 |            1.000000 |  0.095275 | 0.159754 |                  0.035553 |
|     2408 | GLOBAL_GP               |         2408 |                           287 |            0.953488 |  0.063528 | 0.119228 |                  0.001566 |
|     2408 | ROUND_ROBIN_GP          |         2408 |                           301 |            1.000000 |  0.062180 | 0.115029 |                  0.000218 |
|     2408 | CENTROID1_GLOBAL_GP     |         2408 |                           301 |            1.000000 |  0.063445 | 0.119003 |                  0.001482 |
|     2408 | CENTROID1_RANDOM_FILL   |         2408 |                           301 |            1.000000 |  0.061436 | 0.113184 |                 -0.000526 |
|     2408 | GLOBAL_RANDOM           |         2408 |                           300 |            0.996678 |  0.061962 | 0.115216 |                  0.000000 |
|     2408 | GLOBAL_ORACLE           |         2408 |                           301 |            1.000000 |  0.066451 | 0.119592 |                  0.004488 |
|     2408 | CENTROID1_GLOBAL_ORACLE |         2408 |                           301 |            1.000000 |  0.066399 | 0.119592 |                  0.004436 |

## Oracle decomposition

|      budget |   global_oracle_mean_H0 |   centroid1_global_oracle_mean_H0 |   coverage_constraint_cost |   centroid1_global_gp_mean_H0 |   surrogate_loss_hybrid |   total_hybrid_regret |   identity_error |
|------------:|------------------------:|----------------------------------:|---------------------------:|------------------------------:|------------------------:|----------------------:|-----------------:|
|  301.000000 |                0.177751 |                          0.063109 |                   0.114642 |                      0.063109 |                0.000000 |              0.114642 |         0.000000 |
|  602.000000 |                0.133784 |                          0.116391 |                   0.017393 |                      0.097618 |                0.018774 |              0.036167 |         0.000000 |
| 1204.000000 |                0.098155 |                          0.095275 |                   0.002879 |                      0.076979 |                0.018296 |              0.021175 |         0.000000 |
| 2408.000000 |                0.066451 |                          0.066399 |                   0.000052 |                      0.063445 |                0.002954 |              0.003006 |         0.000000 |

## Hybrid comparisons

Hybrid minus GLOBAL_GP:

|      budget |   delta_H0 |   delta_H0_ci_low |   delta_H0_ci_high |   delta_coverage |   extra_niches_represented |
|------------:|-----------:|------------------:|-------------------:|-----------------:|---------------------------:|
|  301.000000 |  -0.077405 |         -0.094743 |          -0.060237 |         0.860465 |                 259.000000 |
|  602.000000 |  -0.003669 |         -0.009159 |           0.001004 |         0.634551 |                 191.000000 |
| 1204.000000 |  -0.000669 |         -0.002105 |           0.000619 |         0.345515 |                 104.000000 |
| 2408.000000 |  -0.000083 |         -0.000288 |           0.000118 |         0.046512 |                  14.000000 |

Hybrid minus ROUND_ROBIN_GP:

|      budget |   delta_H0 |   delta_H0_ci_low |   delta_H0_ci_high |   delta_coverage |
|------------:|-----------:|------------------:|-------------------:|-----------------:|
|  301.000000 |  -0.007469 |         -0.015155 |          -0.000212 |         0.000000 |
|  602.000000 |   0.030750 |          0.021040 |           0.040399 |         0.000000 |
| 1204.000000 |   0.012560 |          0.008406 |           0.016968 |         0.000000 |
| 2408.000000 |   0.001264 |          0.000414 |           0.002203 |         0.000000 |

GP value after the same centroid floor:

|      budget |   delta_H0 |   delta_H0_ci_low |   delta_H0_ci_high |
|------------:|-----------:|------------------:|-------------------:|
|  301.000000 |   0.000000 |          0.000000 |           0.000000 |
|  602.000000 |   0.034418 |          0.024247 |           0.044048 |
| 1204.000000 |   0.018157 |          0.013050 |           0.023492 |
| 2408.000000 |   0.002009 |          0.001027 |           0.003092 |

## Pareto result

CENTROID1_GLOBAL_GP is non-dominated at budgets: [602]. The operational frontier excludes Oracle references, which are retained separately in `dominated_including_oracle`. No hardness/diversity scalarization or lambda was introduced.

|   budget | policy                  |   coverage_fraction |   mean_H0 | dominated   | dominated_by                                                                                                                                                                                                                                                                                                  |
|---------:|:------------------------|--------------------:|----------:|:------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|      301 | GLOBAL_GP               |            0.139535 |  0.140513 | False       |                                                                                                                                                                                                                                                                                                               |
|      301 | ROUND_ROBIN_GP          |            1.000000 |  0.070578 | True        | B602:CENTROID1_GLOBAL_GP;B1204:CENTROID1_GLOBAL_GP                                                                                                                                                                                                                                                            |
|      301 | CENTROID1_GLOBAL_GP     |            1.000000 |  0.063109 | True        | B301:ROUND_ROBIN_GP;B602:ROUND_ROBIN_GP;B602:CENTROID1_GLOBAL_GP;B602:CENTROID1_RANDOM_FILL;B1204:ROUND_ROBIN_GP;B1204:CENTROID1_GLOBAL_GP;B2408:CENTROID1_GLOBAL_GP                                                                                                                                          |
|      301 | CENTROID1_RANDOM_FILL   |            1.000000 |  0.063109 | True        | B301:ROUND_ROBIN_GP;B602:ROUND_ROBIN_GP;B602:CENTROID1_GLOBAL_GP;B602:CENTROID1_RANDOM_FILL;B1204:ROUND_ROBIN_GP;B1204:CENTROID1_GLOBAL_GP;B2408:CENTROID1_GLOBAL_GP                                                                                                                                          |
|      301 | GLOBAL_RANDOM           |            0.604651 |  0.063891 | True        | B301:ROUND_ROBIN_GP;B602:ROUND_ROBIN_GP;B602:CENTROID1_GLOBAL_GP;B1204:GLOBAL_GP;B1204:ROUND_ROBIN_GP;B1204:CENTROID1_GLOBAL_GP                                                                                                                                                                               |
|      301 | GLOBAL_ORACLE           |            0.475083 |  0.177751 | False       |                                                                                                                                                                                                                                                                                                               |
|      301 | CENTROID1_GLOBAL_ORACLE |            1.000000 |  0.063109 | True        | B301:ROUND_ROBIN_GP;B602:ROUND_ROBIN_GP;B602:CENTROID1_GLOBAL_GP;B602:CENTROID1_RANDOM_FILL;B602:CENTROID1_GLOBAL_ORACLE;B1204:ROUND_ROBIN_GP;B1204:CENTROID1_GLOBAL_GP;B1204:CENTROID1_GLOBAL_ORACLE;B2408:CENTROID1_GLOBAL_GP;B2408:GLOBAL_ORACLE;B2408:CENTROID1_GLOBAL_ORACLE                             |
|      602 | GLOBAL_GP               |            0.365449 |  0.101287 | False       |                                                                                                                                                                                                                                                                                                               |
|      602 | ROUND_ROBIN_GP          |            1.000000 |  0.066867 | True        | B301:ROUND_ROBIN_GP;B602:CENTROID1_GLOBAL_GP;B1204:CENTROID1_GLOBAL_GP                                                                                                                                                                                                                                        |
|      602 | CENTROID1_GLOBAL_GP     |            1.000000 |  0.097618 | False       |                                                                                                                                                                                                                                                                                                               |
|      602 | CENTROID1_RANDOM_FILL   |            1.000000 |  0.063199 | True        | B301:ROUND_ROBIN_GP;B602:ROUND_ROBIN_GP;B602:CENTROID1_GLOBAL_GP;B1204:ROUND_ROBIN_GP;B1204:CENTROID1_GLOBAL_GP;B2408:CENTROID1_GLOBAL_GP                                                                                                                                                                     |
|      602 | GLOBAL_RANDOM           |            0.873754 |  0.060848 | True        | B301:ROUND_ROBIN_GP;B301:CENTROID1_GLOBAL_GP;B301:CENTROID1_RANDOM_FILL;B602:ROUND_ROBIN_GP;B602:CENTROID1_GLOBAL_GP;B602:CENTROID1_RANDOM_FILL;B1204:ROUND_ROBIN_GP;B1204:CENTROID1_GLOBAL_GP;B2408:GLOBAL_GP;B2408:ROUND_ROBIN_GP;B2408:CENTROID1_GLOBAL_GP;B2408:CENTROID1_RANDOM_FILL;B2408:GLOBAL_RANDOM |
|      602 | GLOBAL_ORACLE           |            0.787375 |  0.133784 | False       |                                                                                                                                                                                                                                                                                                               |
|      602 | CENTROID1_GLOBAL_ORACLE |            1.000000 |  0.116391 | False       |                                                                                                                                                                                                                                                                                                               |
|     1204 | GLOBAL_GP               |            0.654485 |  0.077648 | True        | B602:CENTROID1_GLOBAL_GP                                                                                                                                                                                                                                                                                      |
|     1204 | ROUND_ROBIN_GP          |            1.000000 |  0.064420 | True        | B301:ROUND_ROBIN_GP;B602:ROUND_ROBIN_GP;B602:CENTROID1_GLOBAL_GP;B1204:CENTROID1_GLOBAL_GP                                                                                                                                                                                                                    |
|     1204 | CENTROID1_GLOBAL_GP     |            1.000000 |  0.076979 | True        | B602:CENTROID1_GLOBAL_GP                                                                                                                                                                                                                                                                                      |
|     1204 | CENTROID1_RANDOM_FILL   |            1.000000 |  0.058822 | True        | B301:ROUND_ROBIN_GP;B301:CENTROID1_GLOBAL_GP;B301:CENTROID1_RANDOM_FILL;B602:ROUND_ROBIN_GP;B602:CENTROID1_GLOBAL_GP;B602:CENTROID1_RANDOM_FILL;B1204:ROUND_ROBIN_GP;B1204:CENTROID1_GLOBAL_GP;B2408:ROUND_ROBIN_GP;B2408:CENTROID1_GLOBAL_GP;B2408:CENTROID1_RANDOM_FILL                                     |
|     1204 | GLOBAL_RANDOM           |            0.980066 |  0.059722 | True        | B301:ROUND_ROBIN_GP;B301:CENTROID1_GLOBAL_GP;B301:CENTROID1_RANDOM_FILL;B602:ROUND_ROBIN_GP;B602:CENTROID1_GLOBAL_GP;B602:CENTROID1_RANDOM_FILL;B1204:ROUND_ROBIN_GP;B1204:CENTROID1_GLOBAL_GP;B2408:ROUND_ROBIN_GP;B2408:CENTROID1_GLOBAL_GP;B2408:CENTROID1_RANDOM_FILL;B2408:GLOBAL_RANDOM                 |
|     1204 | GLOBAL_ORACLE           |            0.980066 |  0.098155 | True        | B602:CENTROID1_GLOBAL_ORACLE                                                                                                                                                                                                                                                                                  |
|     1204 | CENTROID1_GLOBAL_ORACLE |            1.000000 |  0.095275 | True        | B602:CENTROID1_GLOBAL_GP;B602:CENTROID1_GLOBAL_ORACLE                                                                                                                                                                                                                                                         |
|     2408 | GLOBAL_GP               |            0.953488 |  0.063528 | True        | B301:ROUND_ROBIN_GP;B602:ROUND_ROBIN_GP;B602:CENTROID1_GLOBAL_GP;B1204:ROUND_ROBIN_GP;B1204:CENTROID1_GLOBAL_GP                                                                                                                                                                                               |
|     2408 | ROUND_ROBIN_GP          |            1.000000 |  0.062180 | True        | B301:ROUND_ROBIN_GP;B301:CENTROID1_GLOBAL_GP;B301:CENTROID1_RANDOM_FILL;B602:ROUND_ROBIN_GP;B602:CENTROID1_GLOBAL_GP;B602:CENTROID1_RANDOM_FILL;B1204:ROUND_ROBIN_GP;B1204:CENTROID1_GLOBAL_GP;B2408:CENTROID1_GLOBAL_GP                                                                                      |
|     2408 | CENTROID1_GLOBAL_GP     |            1.000000 |  0.063445 | True        | B301:ROUND_ROBIN_GP;B602:ROUND_ROBIN_GP;B602:CENTROID1_GLOBAL_GP;B1204:ROUND_ROBIN_GP;B1204:CENTROID1_GLOBAL_GP                                                                                                                                                                                               |
|     2408 | CENTROID1_RANDOM_FILL   |            1.000000 |  0.061436 | True        | B301:ROUND_ROBIN_GP;B301:CENTROID1_GLOBAL_GP;B301:CENTROID1_RANDOM_FILL;B602:ROUND_ROBIN_GP;B602:CENTROID1_GLOBAL_GP;B602:CENTROID1_RANDOM_FILL;B1204:ROUND_ROBIN_GP;B1204:CENTROID1_GLOBAL_GP;B2408:ROUND_ROBIN_GP;B2408:CENTROID1_GLOBAL_GP                                                                 |
|     2408 | GLOBAL_RANDOM           |            0.996678 |  0.061962 | True        | B301:ROUND_ROBIN_GP;B301:CENTROID1_GLOBAL_GP;B301:CENTROID1_RANDOM_FILL;B602:ROUND_ROBIN_GP;B602:CENTROID1_GLOBAL_GP;B602:CENTROID1_RANDOM_FILL;B1204:ROUND_ROBIN_GP;B1204:CENTROID1_GLOBAL_GP;B2408:ROUND_ROBIN_GP;B2408:CENTROID1_GLOBAL_GP                                                                 |
|     2408 | GLOBAL_ORACLE           |            1.000000 |  0.066451 | True        | B301:ROUND_ROBIN_GP;B602:ROUND_ROBIN_GP;B602:CENTROID1_GLOBAL_GP;B602:CENTROID1_GLOBAL_ORACLE;B1204:CENTROID1_GLOBAL_GP;B1204:CENTROID1_GLOBAL_ORACLE                                                                                                                                                         |
|     2408 | CENTROID1_GLOBAL_ORACLE |            1.000000 |  0.066399 | True        | B301:ROUND_ROBIN_GP;B602:ROUND_ROBIN_GP;B602:CENTROID1_GLOBAL_GP;B602:CENTROID1_GLOBAL_ORACLE;B1204:CENTROID1_GLOBAL_GP;B1204:CENTROID1_GLOBAL_ORACLE;B2408:GLOBAL_ORACLE                                                                                                                                     |

## Required limitation

The sealed dev panel contains only 2,666 exact-labeled candidates across 301 occupied niches and therefore cannot reproduce the full Cycle-0 operating ratio of 10,000 / 435 = 22.988506 selected instances per occupied niche. The full-bank replay validates exact structural behavior at B=10,000, while the dev replay validates true-hardness tradeoffs only at feasible matched budgets [301, 602, 1204, 2408]. Final 10k true-hardness performance is **not** directly established.

No new instances, Concorde calls, Oracle labels, model training, BQ fine-tuning, external evaluation, FINAL_D0, archive update, protocol change, Cycle-0 launch, or final runner occurred. `runs/` remains empty.

Strict verification: PASS (54/54). Tests: 56/56 passed.
