# Mutation Fairness Audit

All 12 arm-generation cells contain exactly 110,000 attempts (conceptual total 1,320,000). Each of the 11 frozen operators is attempted once for every parent.
G1 is physically generated once per replicate and reused read-only by E0/E1; union membership SHA is identical within every pair. G2 uses the arm-specific P1 and the same identity-keyed RNG rule.
No invalid-attempt compensation, archive, lineage cap, novelty exclusion, hardness, or diversity objective is used.

Physical mutation work is 990,000 attempts because each paired G1 is shared. Operator details are in `02_mutation_operator_statistics.csv`.
