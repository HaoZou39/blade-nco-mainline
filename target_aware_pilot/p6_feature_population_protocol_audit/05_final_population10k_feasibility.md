# Final Population10k Dataset Feasibility

| replicate | arm | rows | unique hashes | known verified labels | missing labels | lineage fields complete |
|---|---|---|---|---|---|---|
| Original | E0 | 10000 | 10000 | 8532 | 1468 | True |
| Original | E1 | 10000 | 10000 | 4440 | 5560 | True |
| S1 | E0 | 10000 | 10000 | 3370 | 6630 | True |
| S1 | E1 | 10000 | 10000 | 8361 | 1639 | True |
| S2 | E0 | 10000 | 10000 | 3368 | 6632 | True |
| S2 | E1 | 10000 | 10000 | 8357 | 1643 | True |

All six bundles have shape 10,000×100×2, finite coordinates in [0,1], exactly 100 distinct nodes per instance, 10,000 unique coordinate hashes, identical manifest/NPZ order, reproduced coordinate hashes, and complete required lineage fields. Their NPZ structure (`coordinates`, ordered `instance_hash`) is directly convertible to the existing BQ bundle format by attaching ordered `expert_tours`.

Existing verified Concorde ledger rows may be reused by instance hash. Missing counts above are the only labeling prerequisite; P6 made no Concorde calls. The sum of per-bundle missing counts can exceed the unique missing union when the same instance occurs in more than one bundle; both values are recorded in `support_label_cache_summary.json`.

M4 training overlap was deliberately neither read nor computed, as required. It is not a validity condition.

**FINAL_POPULATION10K_DIRECT_TRAINING_FEASIBLE**

‘Direct’ means no archive-based acquisition or survivor replacement is needed. It does not mean unlabeled coordinates can bypass expert labeling: each missing tour must be generated and integrity-checked before BQ training.
