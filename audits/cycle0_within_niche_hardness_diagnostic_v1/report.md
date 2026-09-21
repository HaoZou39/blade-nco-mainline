# BLADE-NCO Cycle-0 Within-Niche Hardness Diagnostic v1

## Outcome

The descriptive diagnosis is **D4 — STATIC_FEATURE_WITHIN_NICHE_SIGNAL_WEAK**. This is an analysis-only conclusion, not a protocol change. D5 is the primary condition; D1 and D2 are sensitivity branches.

All nine stored prediction vectors aligned exactly to the same 2,666 sealed dev instances, raw H0 labels, and K1000 niche assignments. No generation, solver call, label computation, inference, training, fine-tuning, external evaluation, D0 construction, or final runner was performed.

## True-hardness decomposition

The exact candidate-weighted ddof=0 identity gives a between-niche variance fraction of 0.459385 and a within-niche fraction of 0.540615. This is a descriptive component decomposition, not a causal/random-effects ICC.

| Component | Variance | Fraction |
|---|---:|---:|
| Between-niche mean component | 0.00138099 | 0.459385 |
| Within-niche residual | 0.00162518 | 0.540615 |
| Total raw H0 | 0.00300617 | 1.000000 |

## Oracle headroom

|        r |   eligible_niches |   mean_random_H0 |   mean_oracle_H0 |   mean_absolute_headroom |   ratio_of_means_relative_headroom |
|---------:|------------------:|-----------------:|-----------------:|-------------------------:|-----------------------------------:|
| 1.000000 |        262.000000 |         0.062903 |         0.129533 |                 0.066630 |                           1.059246 |
| 2.000000 |        262.000000 |         0.060506 |         0.112578 |                 0.052073 |                           0.860630 |
| 5.000000 |        262.000000 |         0.059829 |         0.085176 |                 0.025347 |                           0.423652 |

The eligible population is the 262 niches with at least five sealed dev candidates. Oracle selection is exact within that population; RANDOM is sampled once per niche/r from a single seed-2026 stream and shared by every model.

## D5 signal location

Between-niche mean correlations:

| model    |   niches |   spearman |   pearson |
|:---------|---------:|-----------:|----------:|
| GP4_D5   |      301 |   0.593112 |  0.878022 |
| HGB4_D5  |      301 |   0.554995 |  0.882061 |
| HGB20_D5 |      301 |   0.617001 |  0.893281 |

Pooled within-niche-centered correlations:

| model    |    n |   spearman |   pearson |
|:---------|-----:|-----------:|----------:|
| GP4_D5   | 2666 |   0.096255 |  0.293422 |
| HGB4_D5  | 2666 |   0.067516 |  0.181470 |
| HGB20_D5 | 2666 |   0.152403 |  0.248752 |

Macro within-niche ranking:

| model    |   eligible_niches |   median_spearman |   mean_pairwise_accuracy |   candidate_weighted_spearman |
|:---------|------------------:|------------------:|-------------------------:|------------------------------:|
| GP4_D5   |               262 |          0.078788 |                 0.521117 |                      0.066046 |
| HGB20_D5 |               262 |          0.139394 |                 0.542848 |                      0.106893 |
| HGB4_D5  |               262 |          0.036364 |                 0.515285 |                      0.043828 |

## D5 acquisition

| model    |   r |   mean_selected_H0 |   mean_gain_over_random |   mean_oracle_regret |   aggregate_headroom_capture |
|:---------|----:|-------------------:|------------------------:|---------------------:|-----------------------------:|
| GP4_D5   |   1 |           0.068039 |                0.005135 |             0.061495 |                     0.077073 |
| GP4_D5   |   2 |           0.064894 |                0.004389 |             0.047684 |                     0.084277 |
| GP4_D5   |   5 |           0.063775 |                0.003946 |             0.021400 |                     0.155691 |
| HGB20_D5 |   1 |           0.071793 |                0.008890 |             0.057740 |                     0.133420 |
| HGB20_D5 |   2 |           0.071071 |                0.010565 |             0.041507 |                     0.202898 |
| HGB20_D5 |   5 |           0.064780 |                0.004951 |             0.020395 |                     0.195347 |
| HGB4_D5  |   1 |           0.066128 |                0.003224 |             0.063406 |                     0.048391 |
| HGB4_D5  |   2 |           0.065672 |                0.005166 |             0.046907 |                     0.099208 |
| HGB4_D5  |   5 |           0.063193 |                0.003364 |             0.021982 |                     0.132732 |

Capture ratios are ratios of equal-niche mean gains and are not clipped. Per-niche values outside [0,1] are preserved.

## Paired model comparisons

HGB20 minus HGB4 (feature richness):

| metric                 |   left_point |   right_point |    delta |    ci_low |   ci_high |
|:-----------------------|-------------:|--------------:|---------:|----------:|----------:|
| centered_spearman      |     0.152403 |      0.067516 | 0.084887 |  0.035345 |  0.135829 |
| median_within_spearman |     0.139394 |      0.036364 | 0.103030 |  0.006061 |  0.169697 |
| mean_pairwise_accuracy |     0.542848 |      0.515285 | 0.027563 |  0.008439 |  0.046474 |
| top1_H0                |     0.071793 |      0.066128 | 0.005665 | -0.002033 |  0.012936 |
| top1_headroom_capture  |     0.133420 |      0.048391 | 0.085029 | -0.030637 |  0.194739 |
| top2_H0                |     0.071071 |      0.065672 | 0.005399 |  0.001269 |  0.009423 |
| top2_headroom_capture  |     0.202898 |      0.099208 | 0.103690 |  0.023800 |  0.183311 |
| top5_H0                |     0.064780 |      0.063193 | 0.001587 | -0.000097 |  0.003266 |
| top5_headroom_capture  |     0.195347 |      0.132732 | 0.062615 | -0.003668 |  0.130436 |

HGB4 minus GP4 (model-family capacity on QD4):

| metric                 |   left_point |   right_point |     delta |    ci_low |   ci_high |
|:-----------------------|-------------:|--------------:|----------:|----------:|----------:|
| centered_spearman      |     0.067516 |      0.096255 | -0.028740 | -0.081102 |  0.023680 |
| median_within_spearman |     0.036364 |      0.078788 | -0.042424 | -0.133333 |  0.042424 |
| mean_pairwise_accuracy |     0.515285 |      0.521117 | -0.005831 | -0.025022 |  0.013372 |
| top1_H0                |     0.066128 |      0.068039 | -0.001911 | -0.008446 |  0.005320 |
| top1_headroom_capture  |     0.048391 |      0.077073 | -0.028682 | -0.127914 |  0.078851 |
| top2_H0                |     0.065672 |      0.064894 |  0.000778 | -0.003175 |  0.005014 |
| top2_headroom_capture  |     0.099208 |      0.084277 |  0.014931 | -0.061528 |  0.095834 |
| top5_H0                |     0.063193 |      0.063775 | -0.000582 | -0.002740 |  0.001602 |
| top5_headroom_capture  |     0.132732 |      0.155691 | -0.022960 | -0.108627 |  0.062907 |

HGB20 minus GP4 is retained as a secondary comparison (27 metric/depth rows), not used to confound the two primary attribution tests.

## Interpretation

Meaningful Oracle headroom exists at Top1/2/5. The strongest D5 capture values are reported directly (without a classification cutoff) and remain far below Oracle capture 1. HGB20 improves several ranking and acquisition summaries over HGB4, so richer static features contain some additional signal; nevertheless, that signal does not close most of the operational gap. HGB4 does not show a coherent advantage over GP4, so the evidence does not support symbolic GP capacity as the primary bottleneck.

Calibration-depth directional robustness: **YES**. Full D1/D2/D5 point effects and paired niche-bootstrap intervals are in `07_feature_bottleneck/` and `08_symbolic_capacity/`.

## Geometry, sample size, and agreement

The audit records within-niche centroid/QD4 spread, its rank associations with true H0 spread and D5 rankability, empirical sample-size strata without splitting equal-n niches, and D5 Top1/2/5 cross-model overlap. These are descriptive diagnostics; no post-hoc feature or niche-policy change is made.

## Reproducibility and boundaries

- Primary branch: D5; D1/D2 sensitivity only.
- Bootstrap: 10,000 paired niche resamples, seed 2026, percentile 95% intervals.
- Value metric: raw H0 / raw M0 L_BQ.
- Strict verification: PASS (44/44 checks).
- Tests recorded at generation: 44/44 passed.
- Every source and generated output (except the self-referential hash inventory itself) is SHA-256 inventoried.
