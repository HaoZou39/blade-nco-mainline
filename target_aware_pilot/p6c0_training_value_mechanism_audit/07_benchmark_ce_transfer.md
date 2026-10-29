# Benchmark Teacher-Forced CE Transfer

Existing verified expert tours were available for Uniform100, GaussianMixture100, and TSPLIB100, so read-only CE was computed for M4 and all six M5 checkpoints. No Concorde call was made. GaussianMixture500 and Original70 have no verified expert-tour bundle in the frozen benchmark assets and are not CE-evaluated.

| replicate | arm | dataset | n | mean_ce_M4 | mean_ce_M5 | mean_ce_gain_M4_minus_M5 | ce_improved_fraction | mean_gap_delta_M5_minus_M4 | spearman_ce_gain_vs_gap_delta |
|---|---|---|---|---|---|---|---|---|---|
| Original | E0 | Uniform100 | 10000 | 0.050334685 | 0.049130436 | 0.0012042494 | 0.516 | -0.0090675171 | -0.10546397 |
| Original | E0 | GaussianMixture100 | 10000 | 0.17352669 | 0.16016054 | 0.013366152 | 0.6053 | -0.49544613 | -0.084291296 |
| Original | E0 | TSPLIB100 | 10000 | 0.062281095 | 0.058491016 | 0.0037900782 | 0.5718 | -0.085415102 | -0.095688121 |
| Original | E1 | Uniform100 | 10000 | 0.050334685 | 0.047561448 | 0.0027732374 | 0.58 | -0.022278647 | -0.10459047 |
| Original | E1 | GaussianMixture100 | 10000 | 0.17352669 | 0.15742424 | 0.016102456 | 0.5992 | -1.2178334 | -0.11875341 |
| Original | E1 | TSPLIB100 | 10000 | 0.062281095 | 0.059603525 | 0.00267757 | 0.558 | -0.068176821 | -0.14221118 |
| S1 | E0 | Uniform100 | 10000 | 0.050334685 | 0.049249976 | 0.0010847092 | 0.5179 | -0.0024357609 | -0.1165051 |
| S1 | E0 | GaussianMixture100 | 10000 | 0.17352669 | 0.14867125 | 0.024855438 | 0.679 | -0.95440521 | -0.11619116 |
| S1 | E0 | TSPLIB100 | 10000 | 0.062281095 | 0.058503434 | 0.003777661 | 0.56 | -0.08920574 | -0.12152882 |
| S1 | E1 | Uniform100 | 10000 | 0.050334685 | 0.047806073 | 0.0025286127 | 0.58 | -0.03336097 | -0.13716754 |
| S1 | E1 | GaussianMixture100 | 10000 | 0.17352669 | 0.16617417 | 0.0073525183 | 0.5594 | -0.79265384 | -0.10387084 |
| S1 | E1 | TSPLIB100 | 10000 | 0.062281095 | 0.059570811 | 0.0027102838 | 0.563 | -0.084507181 | -0.11207124 |
| S2 | E0 | Uniform100 | 10000 | 0.050334685 | 0.049305151 | 0.0010295339 | 0.5089 | 0.0028493285 | -0.10052198 |
| S2 | E0 | GaussianMixture100 | 10000 | 0.17352669 | 0.15101674 | 0.022509951 | 0.6689 | -0.66770034 | -0.099549154 |
| S2 | E0 | TSPLIB100 | 10000 | 0.062281095 | 0.0591642 | 0.0031168946 | 0.5392 | -0.071047723 | -0.095561136 |
| S2 | E1 | Uniform100 | 10000 | 0.050334685 | 0.048231136 | 0.002103549 | 0.5536 | -0.015856427 | -0.12820837 |
| S2 | E1 | GaussianMixture100 | 10000 | 0.17352669 | 0.16244319 | 0.011083504 | 0.5783 | -0.74270538 | -0.11958828 |
| S2 | E1 | TSPLIB100 | 10000 | 0.062281095 | 0.059938362 | 0.0023427323 | 0.5515 | -0.029302952 | -0.11638821 |

Positive CE gain means lower M5 teacher-forced loss; positive gap delta means worse M5 greedy gap. Agreement is therefore expected to be negative in sign at instance level, but teacher-forced imitation and greedy decoding are distinct objectives.
