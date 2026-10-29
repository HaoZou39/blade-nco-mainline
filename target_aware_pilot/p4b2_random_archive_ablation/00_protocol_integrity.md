# P4-B2 Protocol Integrity

- Recovered the complete frozen E0 pre-retention pool directly from persisted G1/G2 mutation bundles and diagnostics: 39,849 eligible rows before cross-generation dedup, 1 duplicates removed, exactly 39,848 unique eligible TSP100. No mutation was rerun.
- Both 20,000-attempt manifests, accepted-child order, coordinate hashes, runtime membership hashes, lineage, generation and operator provenance reproduce. Full-pool bundle SHA256 `327265660c63607ec4c406d4955a60fa072101246ca456d9342443ae12ce6e40`.
- E0 active-population survivor selection is `random_survivor` in both generations. The initial P03 population nevertheless came from the earlier hardness-driven pipeline; R2 is a structure-free Cycle-5 selection arm, not an M0-from-scratch random baseline.
- Frozen R0 archive uses `common_bounded_archive_retention`: target-aware quota/frontier selection plus lineage diversity, 39,848 → 30,000. Frozen R0 final acquisition is uniform random sampling with seed 20261025.
- R2 archive retention uses only canonical instance hashes plus NumPy uniform sampling without replacement, seed 20261026; each offspring has retention probability 0.752860871. R2 final acquisition uses the same R0 algorithm and seed 20261025; neither stage reads structure, TSPLIB, hardness, loss, lineage, operator, generation, density or provenance.
- R2 contains zero C00–C03 M4-history instances and zero direct TSPLIB70 coordinate matches. R0/R2 overlap is not excluded or optimized.
- All five frozen checkpoints, including the formal M4 initialization, reproduce their SHA256 values.
- Predeclared decision rule before labels/training: PRELIMINARILY_SUPPORTED requires R0 to win at least 6/8 benchmark means, at least 3/8 paired CIs entirely favor R0, and no significant R2 win. NOT_SUPPORTED_IN_THIS_TRIAL is the exact mirror; otherwise INCONCLUSIVE.
- TSPLIB22 remains a repeatedly observed comparability set and is not used for sampling, tuning or checkpoint selection.

Status: **PASS**.
