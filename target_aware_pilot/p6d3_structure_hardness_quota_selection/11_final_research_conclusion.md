# P6-D3 Final Research Conclusion

`H1_STRUCTURAL_CONTROL: MIXED`
`H2_HARDNESS_PRESERVATION: SUPPORTED`
`H3_TRAINING_VALUE_GAIN: MIXED`
`H4_BEATS_RANDOM_BASELINE: MIXED`
`H5_ABSOLUTE_IMPROVEMENT: PARTIAL`
`H6_RATIO_TRADEOFF: MIXED`
`H7_ONLINE_EVOLUTION_READINESS: NOT_ESTABLISHED`

Across the three existing evolution replicates, the lowest mean Original70 gap among preregistered D arms is D75 (2.363136%). This is a development result over only three ratios and does not establish an optimal ratio.

Original70 win counts versus C: {'D25': 1, 'D50': 1, 'D75': 0}; versus B: {'D25': 1, 'D50': 1, 'D75': 1}; versus M4: {'D25': 1, 'D50': 1, 'D75': 0}. GAP5 non-worse counts versus C: {'D25': 2, 'D50': 1, 'D75': 1}; versus B: {'D25': 0, 'D50': 0, 'D75': 1}.

| replicate | B | C | D25 | D50 | D75 | M4 |
|---|---|---|---|---|---|---|
| Original | 1.9509795 | 2.1840095 | 2.3025306 | 2.4333497 | 2.295654 | 2.2170026 |
| S1 | 2.3246432 | 2.3553446 | 2.2009432 | 2.6873125 | 2.5280368 | 2.2170026 |
| S2 | 2.3310434 | 2.2489836 | 2.6198923 | 2.1527617 | 2.2657186 | 2.2170026 |

Structural monotonicity and hardness trade-off diagnostics: [{'replicate': 'Original', 'd_struct_nonincreasing': True, 'ot_nonincreasing': True, 'true_ce_nonincreasing': True, 'original70_monotone': False}, {'replicate': 'S1', 'd_struct_nonincreasing': True, 'ot_nonincreasing': True, 'true_ce_nonincreasing': True, 'original70_monotone': False}, {'replicate': 'S2', 'd_struct_nonincreasing': True, 'ot_nonincreasing': False, 'true_ce_nonincreasing': True, 'original70_monotone': False}].

The direct answer is determined by H3/H4: explicit role quotas improve stability only if one fixed ratio wins across all three evolution replicates without systematic GAP5 regression. Own-data CE is not treated as generalization. QD4 cross-size clipping and HGB root-overlap limits remain; PROMISING never means novel-offspring generalization is established.

No ratio is retuned, and no P6-D4, evolution, Concorde, surrogate training or additional seed is started.

`BLADE_P6D3_STRUCTURE_HARDNESS_QUOTA_SELECTION_COMPLETE`
