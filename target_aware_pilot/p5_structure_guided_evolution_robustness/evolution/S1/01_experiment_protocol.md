# P2.2 Experiment Protocol

- Arms: E0 random survivor control and E1 structure-guided survivors, both starting from identical formal P03.
- Active population: fixed at 10,000; Parent+Child Union selection; two continuous generations.
- Mutation: frozen `FOGA11_BASE_V1` and all 11 operators via the repository API. Each generation has 20,000 attempts: every parent receives exactly two distinct operators and each operator receives 1,818 or 1,819 attempts.
- Seed: 260501; FOGA cycle seed namespaces: {1: 2605011, 2: 2605012}. E0/E1 G1 therefore have identical mutations before survivor selection; G2 naturally differs.
- E1 selection: 70% strict target-coverage quota plus 30% non-strict frontier exploration. Frozen 3x3 target weights, 12x12 within-region microbin round-robin, and deterministic lineage caps are used. No single TSPLIB instance receives a special objective.
- E0 selection: canonical-hash ordering followed by seeded uniform sampling without replacement.
- Archive: eligible unique offspring only, C00-C03 exclusions applied, maximum 30,000. Both arms use the identical structure/diversity retention method.
- Continuous distance: mean 1D Wasserstein distance across 256 fixed random directions in the P1 median/IQR-standardized two-feature space (seed 261492).
- Final acquisition: identical structure-only cell quotas, continuous microbin diversity, lineage cap, strict-first then explicit transition backfill. HGB/BQ hardness is absent from mutation, survivor, archive, and acquisition logic.
- Decision guards frozen before results: E1 must beat E0 at G2 SWD, improve G0→G1→G2, retain at least 50% of G0 effective lineages with max lineage share <=5%, avoid geometry failures, and provide a complete eligible 10k. Auxiliary JS degradation >0.10 versus E0 is treated as a feature-gaming warning and prevents unconditional GO.
- Held-out TSPLIB22 coordinates/features are never loaded. Their names are read only for the P1 disjointness assertion.
- No Oracle, Concorde, HGB fitting/prediction, BQ inference, or BQ training is invoked.
