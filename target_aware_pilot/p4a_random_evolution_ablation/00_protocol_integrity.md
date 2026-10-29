# P4-A Protocol Integrity

- Frozen P03 source bundle SHA256 `8ad76b90c08997bba33da1be6bf3dbfa8c28ee05bbb4a5fe4d1a9ecdb699dd62`; both arms start from this exact 10,000-member population.
- Two generations per arm, 20,000 attempts/generation, all 11 frozen `FOGA11_BASE_V1` operators, identical validity/deduplication, Parent+Child Union, 10,000 active population, and 30,000 eligible-offspring archive cap.
- E0/E1 G1 offspring coordinates and ordered IDs are bitwise identical before survivor selection. G2 differs only downstream of the designed G1 selection intervention.
- E0 uses seeded random parent/survivor selection. E1 uses target-structure-guided parent/survivor selection. Both bounded archives use the same retention logic and both final datasets use the identical `acquisition()` implementation.
- Exact deterministic replay of E0 acquisition reproduces all ordered IDs and metadata; coordinates reproduce directly from the frozen E0 bounded archive. The P2.0/P2.1 229,148 pool is not read.
- No hardness/HGB prediction participates in mutation, survivor selection, archive retention, or acquisition. TSPLIB22 coordinates/features are not loaded; the set is used only after the model is frozen.
- Frozen E1 data/checkpoint hashes: `5287932231529e617383388bb64d607985d07944799c3d661c5a737338bdece1` / `ace34d712ca8d20e730edeed0666362e6cc03008f9b5c0411d53c444856e3233`. They will not be regenerated or retrained.
- Predeclared conclusion rule: CASE A requires E1−E0 mean gap < 0 on at least 6/8 frozen benchmarks, paired CI upper < 0 on at least 3/8, and no paired-significant E1 regression. CASE C is the exact mirror for E0. Other evidence is CASE B.

Status: **PASS**.
