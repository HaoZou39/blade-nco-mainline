# Failure-Mode Analysis

The decisive failure is output compression/negative bias, not an optimization artifact: frozen high-decile thresholds sit far above most predictions, so quota supply fails before the selection LP. HGB preserves the E0>E1 mean direction, but within-E1 rho is only about 0.10–0.11 and no E1-only instance crosses either frozen P90/P95 threshold. Thus pooled ranking overstates candidate-level tail utility. True high-hardness examples are systematically assigned to lower predicted deciles. F40 extrapolation, operator and lineage summaries are diagnostic only and were not used to alter the model.

## Source means

| replicate | source | true_mean | pred_mean |
|---|---|---|---|
| Original | E0 | 0.08867647 | 0.07627437 |
| Original | E1 | 0.06169981 | 0.05206362 |
| Original | both | 0.06449371 | 0.05200667 |
| S1 | E0 | 0.09008369 | 0.07621317 |
| S1 | E1 | 0.0615528 | 0.0520402 |
| S1 | both | 0.06681107 | 0.05269032 |
| S2 | E0 | 0.08905584 | 0.07581695 |
| S2 | E1 | 0.06100445 | 0.05189343 |
| S2 | both | 0.06429274 | 0.05112087 |

## Bias and F40 range

| replicate | source | n | bias | mae | ood_any_fraction |
|---|---|---|---|---|---|
| Original | E0 | 9929 | -0.0124021 | 0.03240163 | 0.01279081 |
| Original | E1 | 9929 | -0.009636193 | 0.02507562 | 0.04028603 |
| Original | both | 71 | -0.01248704 | 0.02479948 | 0.05633803 |
| S1 | E0 | 9937 | -0.01387052 | 0.03276007 | 0.01418939 |
| S1 | E1 | 9937 | -0.009512597 | 0.02498409 | 0.03431619 |
| S1 | both | 63 | -0.01412075 | 0.0283687 | 0.01587302 |
| S2 | E0 | 9927 | -0.01323889 | 0.0329811 | 0.01531178 |
| S2 | E1 | 9927 | -0.009111016 | 0.02470049 | 0.03968974 |
| S2 | both | 73 | -0.01317187 | 0.0269683 | 0.05479452 |

## Lineage sensitivity

| replicate | lineages | lineage_mean_spearman | median_lineage_bias | p90_true_vs_pred_lineage_spearman |
|---|---|---|---|---|
| Original | 1007 | 0.5284391 | -0.01016986 | 0.1655348 |
| S1 | 1017 | 0.5493574 | -0.01038161 | 0.1749767 |
| S2 | 1016 | 0.494552 | -0.008955657 | 0.1742425 |
