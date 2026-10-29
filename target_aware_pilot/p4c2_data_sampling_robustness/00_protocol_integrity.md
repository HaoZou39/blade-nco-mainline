# P4-C2 Protocol Integrity

- Frozen E0 pool: 39,848 unique eligible TSP100, membership SHA256 `2565bd7088d6bdc7b6659add93c5dbabfe848725a99f175328fecda3586d6d0d`; G1/G2 persisted mutation artifacts and provenance reproduce. No FOGA mutation was rerun.
- R0 target-aware retention is **fully deterministic**: the selector has no RNG/seed argument; quota allocation, microbin ordering, lineage caps, backfill, and final instance-hash tie-breaking are deterministic. A fresh replay exactly reproduces all 30,000 ordered members and selection metadata.
- Consequently, S1/S2 cannot vary R0 Archive30k without changing the algorithm. They vary only R0 final Random10k. R2 legitimately varies both uniform retention and final acquisition.
- Original seeds: R2 retention `20261026`; common final acquisition `20261025`. R0 has no retention seed.
- New seeds were frozen before generation: S1 `260201/260301`, S2 `260202/260302`; all BQ training uses seed `2026`.
- Formal M4 SHA256 `c749311bd2a461084d0cbcfd383a4c0a1fc1268762b72e5f7220952f108f8190` and the P4-B2 trainer/evaluator protocol are frozen.
- Labeling will use the canonical Concorde/cache path without BQ hardness inference.

Status: **PASS**. Scope is acquisition/data-sampling robustness within one frozen E0 pool, not R0 retention-seed robustness or cross-evolution-pool robustness.
