# P6-G0 Final Conclusion

## Decisions

- `P6_G0_RAW_OFFSPRING_AVAILABLE = YES`
- `P6_G0_G1_EXPANDS_COVERAGE = YES`
- `P6_G0_G1_SELECTION_LOSS = YES`
- `P6_G0_G2_EXPANDS_COVERAGE = YES`
- `P6_G0_G2_SELECTION_LOSS = YES`
- `P6_G0_GENERATION_REACHABILITY_LIMITED = YES`
- `P6_G0_PRIMARY_BOTTLENECK = GENERATION`
- `P6_G0_AUDIT_STATUS = COMPLETE`

## Frozen full-stage coverage

| replicate | arm | stage | coverage_n | population_n |
|---|---|---|---|---|
| Original | SHARED | P03 | 23 | 10000 |
| Original | SHARED | G1_RAW | 25 | 109571 |
| Original | E0 | G1_SELECTED | 22 | 10000 |
| Original | E0 | G2_RAW | 24 | 109603 |
| Original | E0 | G2_SELECTED | 23 | 10000 |
| Original | E1 | G1_SELECTED | 25 | 10000 |
| Original | E1 | G2_RAW | 25 | 109778 |
| Original | E1 | G2_SELECTED | 24 | 10000 |
| S1 | SHARED | P03 | 23 | 10000 |
| S1 | SHARED | G1_RAW | 24 | 109541 |
| S1 | E0 | G1_SELECTED | 21 | 10000 |
| S1 | E0 | G2_RAW | 22 | 109574 |
| S1 | E0 | G2_SELECTED | 22 | 10000 |
| S1 | E1 | G1_SELECTED | 24 | 10000 |
| S1 | E1 | G2_RAW | 24 | 109751 |
| S1 | E1 | G2_SELECTED | 23 | 10000 |
| S2 | SHARED | P03 | 23 | 10000 |
| S2 | SHARED | G1_RAW | 25 | 109570 |
| S2 | E0 | G1_SELECTED | 22 | 10000 |
| S2 | E0 | G2_RAW | 24 | 109619 |
| S2 | E0 | G2_SELECTED | 23 | 10000 |
| S2 | E1 | G1_SELECTED | 25 | 10000 |
| S2 | E1 | G2_RAW | 25 | 109762 |
| S2 | E1 | G2_SELECTED | 24 | 10000 |
| EvoReal | EXTERNAL | EvoReal50k | 40 | 50000 |

## Matched-size 75–150 reference coverage (17 total)

| replicate | arm | stage | coverage_n | population_n |
|---|---|---|---|---|
| Original | SHARED | P03 | 14 | 10000 |
| Original | SHARED | G1_RAW | 15 | 109571 |
| Original | E0 | G1_SELECTED | 14 | 10000 |
| Original | E0 | G2_RAW | 14 | 109603 |
| Original | E0 | G2_SELECTED | 14 | 10000 |
| Original | E1 | G1_SELECTED | 15 | 10000 |
| Original | E1 | G2_RAW | 15 | 109778 |
| Original | E1 | G2_SELECTED | 15 | 10000 |
| S1 | SHARED | P03 | 14 | 10000 |
| S1 | SHARED | G1_RAW | 14 | 109541 |
| S1 | E0 | G1_SELECTED | 14 | 10000 |
| S1 | E0 | G2_RAW | 14 | 109574 |
| S1 | E0 | G2_SELECTED | 14 | 10000 |
| S1 | E1 | G1_SELECTED | 14 | 10000 |
| S1 | E1 | G2_RAW | 14 | 109751 |
| S1 | E1 | G2_SELECTED | 14 | 10000 |
| S2 | SHARED | P03 | 14 | 10000 |
| S2 | SHARED | G1_RAW | 15 | 109570 |
| S2 | E0 | G1_SELECTED | 14 | 10000 |
| S2 | E0 | G2_RAW | 14 | 109619 |
| S2 | E0 | G2_SELECTED | 14 | 10000 |
| S2 | E1 | G1_SELECTED | 15 | 10000 |
| S2 | E1 | G2_RAW | 15 | 109762 |
| S2 | E1 | G2_SELECTED | 15 | 10000 |
| EvoReal | EXTERNAL | EvoReal50k | 17 | 50000 |

## Direct answer

**YES at the corresponding seed/branch level:** FOGA generated EvoReal-coverable neighborhoods that the corresponding final 10k lost: Original-E0 `rat99, rat783`; Original-E1 `pr226`; S1-E0 `rat575, rat783`; S1-E1 `pr226`; S2-E0 `rat99, rat575`; S2-E1 `pr226`. Exact raw and EvoReal candidate counts are in the evidence table.

**NO reference was generated and then lost by all six final populations simultaneously.** Other branches retained every globally generated reference at least once. Globally verified historical generation-limited references are `d198;d657;pcb1173;pr107;pr144;pr264;u574;u724;a280;eil51;pcb442;pr299;rat195;ts225;tsp225`.

The reconstructed E0∪E1 final coverage exactly reproduces P6-E0.5: Original 25/48, S1 24/48, S2 25/48.

At the combined-pool level, every seed's final E0∪E1 reference set is exactly equal to the union of all references reached by its G1/G2 RAW offspring. Thus branch-level selection losses are real, but the P6-E0.5 combined-pool deficit versus EvoReal is primarily a historical generation-reachability limitation. In the 75–150 subset, EvoReal reaches 17/17 while historical RAW reaches at most 15/17; the two globally unreached matched-size references are `pr107` and `pr144`.

This conclusion is limited to the historical P03 parents, two generations, recorded mutation budgets, and frozen FFT_APPROX/NN-CV geometry. It does not claim that FOGA-11 can never reach a currently unreached structure.

`BLADE_P6G0_RAW_OFFSPRING_COVERAGE_AUDIT_COMPLETE`
