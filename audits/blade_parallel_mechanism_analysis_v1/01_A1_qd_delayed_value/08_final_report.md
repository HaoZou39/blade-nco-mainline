# A1 QD delayed descendant-value

## Immediate selection
QD-rescued N by cycle: [306, 745, 985, 886, 1000]
SHARED_GP N by cycle: [9694, 9255, 9015, 9114, 9000]
GP-only displaced N by cycle: [306, 745, 985, 886, 1000]

Mean GP percentile QD_RESCUED vs SHARED_GP: [72.538, 53.478, 11.849, 31.018, 10.527] vs [95.954, 96.135, 96.229, 96.185, 96.236]

Lower current GP score of QD-rescued parents is expected and is not treated as failure.

## Delayed value
QD_RESCUED future T-entry: k=1 0.1222; k=2 0.0747; k=3 0.0371
QD_RESCUED new-niche descendant rate k=1: 0.1475
Cycles with k=1 T-entry ≥ 0.03: 2

Matched SHARED_GP k=1 T-entry: rescued 0.1222 vs control 0.2047
Match coverage: 1.0000

GP_ONLY_DISPLACED parents generate no children because they were not selected. Their empty lineage is a definitional contrast, not a mutation failure.

## Verdict
**QD_DELAYED_LINEAGE_VALUE_OBSERVED**

This lineage audit does not claim causal downstream solver improvement.
