# P0/P1 Go / No-Go Decision

## Decision: CONDITIONAL GO

P0/P1 produced sufficient evidence for a research decision, but the conditions for immediately starting Target-Aware 10k acquisition are not met.

## Evidence against NO-GO

- The C04 HGB40 is confirmed to be M4-conditioned and has moderate independent held-label ranking skill (Spearman 0.413, hard-tail enrichment 1.353x).
- The selected features are stable enough for an exploratory audit, interpretable, dimensionless, uniformly scale-invariant, finite on all TSPLIB48 references, and not meaningfully driven by node count.
- TSPLIB48 and BLADE Hard10k differ materially: Hard10k is shifted toward higher pair-distance variation and lower fraction-shorter values, with 72.5% in one coarse cell and coarse JS divergence 0.568 from the target.
- The FOGA union reaches 45/48 target neighborhoods, and all 11 mutation operators contribute candidates to the target region.

## Why this is not an immediate GO

- The strict smoothed target region contains 8,969 unique FOGA candidates, below the required later acquisition size of 10,000.
- Three reference neighborhoods (`d198`, `eil51`, and `ts225`) are not reached at the frozen target-derived radius.
- The FOGA distribution itself remains skewed (largest coarse-cell share 55.0%; JS divergence 0.499).
- The validation set is acquisition-shifted, so HGB40's moderate performance should not be overstated as IID generalization.

## Required next decision work before P2

1. Audit which parent regions and mutations can reach the three uncovered or weakly covered target neighborhoods.
2. Determine whether additional frozen-protocol FOGA proposals can raise strict-region unique capacity above 10,000 without collapsing density into the dominant coarse cell.
3. Re-run the same coverage definitions on the expanded candidate-only pool.
4. Only if coverage and uniqueness become adequate should a separate P2 protocol define a target-aware acquisition rule and generate data.

No target-aware 10k set was generated and no BQ fine-tuning was started.

`TARGET_AWARE_P0_P1_AUDIT_COMPLETE`
