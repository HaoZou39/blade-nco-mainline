# P2.0 Parent Selection Protocol

## Frozen inputs

- Source pool: C04 Parent+Child union, 119,530 unique instances.
- Target references: the P1-frozen TSPLIB48 development set. Held-out TSPLIB22 identities were checked only for disjointness and no held-out coordinates or features were loaded.
- Structure coordinates: `spn_distance_varcoeff` and `spn_fraction_shorter_mean_distance` with the P1-frozen target median/IQR standardization.
- Strict radius: 0.095955812384, unchanged from P1.
- Parents per arm: 10,000 unique; operators: 11; attempts per arm: 110,000; mutation depth: 1; mutation cycle seed namespace: 5.
- No BQ loss, predicted hardness, Oracle, Concorde, surrogate fitting, or BQ training is used.

## E0 random control

Candidates are sorted by instance hash and sampled uniformly without replacement using NumPy seed 20261024. The resulting membership hash is `c24da1d8e9b74f4ef334ef69ed8d97f7528190bb2fa57a9e01be3a486fb5de4a`.

## E1 structure-guided selection

- 7,000 Target Coverage parents and 3,000 Frontier Exploration parents.
- Target Coverage cell weights are the frozen TSPLIB48 3x3 counts `[11, 5, 0, 1, 10, 5, 4, 1, 11]`. Capacity-aware deterministic redistribution is applied because the strict-region availability is `[550, 213, 0, 1113, 1187, 86, 3803, 988, 1029]`. Final parent quotas are `[550, 213, 0, 656, 1187, 86, 2623, 656, 1029]`.
- Within each cell, selection round-robins across nearest TSPLIB references, placing currently scarce references first and then using target proximity. A global lineage cap starts at 12 and is relaxed only if required; maximum used was 96.
- Frontier quota is fixed at 1,000 for each of `d198`, `eil51`, and `ts225`. It draws only non-region C04 candidates, from the 5,000 closest per target, round-robin across eight directions in standardized feature space.
- No real TSPLIB instance is ever used as a mutation parent.

E0/E1 share 780 parents; shared parent/operator pairs therefore receive identical frozen child seeds. E1 membership hash is `e46ac944212540a1d8117351b4e52ed6efeb4641bd0449ac9e601441d243e97c`.

All original A0 artifacts were read-only. New artifacts are isolated below this directory.
