# Concorde Labeling Audit

- Pipeline: original A0 `_label_population` → canonical `label_with_cache`/`solve_batch`, followed by the original M4 teacher-forced hardness attachment. Hardness metadata does not affect training membership or order.
- Selected/verified: 10,000/10,000; failures: 0; retries: 0; cache hits: 0; physical Concorde calls: 10000.
- Wall-clock: 475.527s for the original combined label + M4-hardness stage.
- Tours: shape `(10000,100)`, 0-based Hamiltonian permutations. Closure is implicit for objective evaluation and appended as node 101 only when constructing BQ expert-ordered training input.
- Objective: explicit symmetric matrix with each Euclidean edge rounded after scaling by 1,000,000. All 10,000 costs exactly reproduce the Concorde ledger.
- Ledger SHA256 `f859ea708e157fc58495a4557e1ef48f72c4faee90cd566b6ac76f21f47fc21f`; labeled bundle SHA256 `259dab4e317a66e6c67506617dcbb52da52f6e3bb17f417eaac3f3d4474712f6`; expert-tour array SHA256 `6f38146ba101f69c0562e25ac67845476545f9b1035278e37f0695e7aa34088a`.

Status: **PASS**. Training is now permitted.
