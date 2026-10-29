# P6-1 Final Research Conclusion

## `DIRECT_POPULATION_STRUCTURE_GAIN_INCONCLUSIVE` — **INCONCLUSIVE**

Delta = gap(E0-Pop) - gap(E1-Pop); positive favors structure-guided E1.

| Replicate | Original70 E0-Pop | Original70 E1-Pop | Delta | 95% CI | TSPLIB22 Delta | GAP5 Delta |
|---|---:|---:|---:|---|---:|---:|
| Original | 2.1208 | 2.4126 | -0.2917 | [-1.0271, +0.2595] | +0.1237 | -0.0479 |
| S1 | 2.9430 | 2.2060 | +0.7370 | [+0.0980, +1.5808] | +0.2044 | +0.2166 |
| S2 | 2.1245 | 2.5414 | -0.4169 | [-0.8583, -0.0251] | -0.7496 | -0.1025 |

## Training-membership composition diagnostic

- E1 G2 has lower SWD and 3x3 JS than E0 G2 in all three replicates: **True**. Structural guidance therefore still shapes the final population even though solver gains are inconsistent.
- Direct G2 populations retain 10.61%–12.91% P03-origin parents; all Archive-based Structure10k sets contain 0 parents because the old acquisition selected eligible offspring only.
- E1 largest lineage share is 0.51%–0.60%, versus E0 1.36%–1.69%; no new E1 lineage concentration is observed.
- The old terminal acquisition materially changes composition and structure metrics, but those changes do not translate into a consistent Archive-vs-Population performance direction.

## Required answers

1. All six frozen G2 Population10k datasets completed integrity, labeling, M4 initialization, 256-epoch training, and frozen evaluation.
2. New unique Concorde labels: **23,468**; all six reach 10,000/10,000 valid tours.
3. Original70 gaps are reported in the table above and the full CSV.
4. Stable E1 advantage across all three evolution seeds: **False**; E1 win count 1/3.
5. TSPLIB22 E1 win count 2/3; GAP5 E1 win count 1/3. They do not uniformly support the same direction.
6. Direct Population overall descriptive Original70 advantage over Archive-based training: **False** (2/6 cells; mean -0.1936 pp). This is independent of E1-vs-E0 validity.
7. P5 S2 reverse degradation persists: population Delta -0.4169 pp versus archive -0.2483 pp. S2 structural matching remains better for E1 = True.
8. Evidence that Archive30k→Structure10k weakened training value is not supported overall; arm-specific path effects and interactions are in `06`/`08`.
9. Regardless of this result, P6-0 found no empirical/theoretical justification for the old 70/30 ratio. Keep it only as a frozen baseline; define the next structural objective explicitly rather than tuning 70/30 post hoc.

Original70 is target-informed and TSPLIB22 has been repeatedly inspected. These results are exploratory, not independent confirmation. No extra training seed, mutation, acquisition, feature selection, or P6-2 was run.

`DIRECT_POPULATION_STRUCTURE_GAIN_INCONCLUSIVE`

`INCONCLUSIVE`

`BLADE_P6_1_FINAL_POPULATION_DIRECT_TRAINING_COMPLETE`
