# Preregistered Training-Seed Protocol

- Original paired seed: `2026` (existing R0 and R2).
- New paired seeds: `260101`, `260102`; both arms run for each seed, exactly four new trainings.
- Delta = GAP(R2) − GAP(R0); positive favors target-aware archive retention.
- SUPPORTED: all three paired seeds have positive GAP8 Delta, R0 wins at least 5/8 benchmark means in every seed, and cross-seed mean Delta is positive on at least 5/8 benchmarks.
- NOT_SUPPORTED: exact directional mirror. All other outcomes are INCONCLUSIVE.
- Per-seed paired bootstrap quantifies test-instance uncertainty. The three seed-level Deltas are summarized descriptively and are never expanded into a pseudo-independent instance sample.
- No result-dependent extra seed, rerun, checkpoint selection, dataset weighting or protocol change is permitted.

Base protocol SHA256: `80852d9e42501c7d84b7c3fd19c1c3dd23e9581d7fb6815256050c68bc41e13d`.
