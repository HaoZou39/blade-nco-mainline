# P5 Protocol and Causal-Path Audit

Status: **PASS**. P2.2 source and frozen artifacts are internally consistent.

- P03 is exactly the formal C03 `P_next` and C04 parent population: 10,000 aligned TSP100, bundle SHA256 `8ad76b90c08997bba33da1be6bf3dbfa8c28ee05bbb4a5fe4d1a9ecdb699dd62`, membership `558ea278823a39275704d71e8e999efe58a56217fb23c7555cabe035c986d190`.
- Each arm executes two generations × 20,000 FOGA-11 attempts. Mutation reads the current `population_frames`; G1 survivors are written back and are the G2 parents.
- The archive is updated only with each generation's eligible offspring. It never supplies parents. Final Structure10k acquisition reads Archive30k, not Population10k.
- G1 survivor selection affects G2 parents, hence G2 offspring and the cumulative training archive. G2 survivor selection happens only after G2 offspring already exist and there is no G3. Therefore: `G2_SURVIVOR_SELECTION_NOT_ON_TRAINING_DATA_PATH`.
- G2 survivors are computed and audited but do **not** causally affect the present M5 training data.
- G1 E0/E1 mutation streams are matched without an adapter in the original semantics: same ordered P03, deterministic operator schedule, and child seeds based only on generation namespace + parent hash + operator. No arm name or mutable global RNG enters mutation.
- E0 survivor selection alone is seeded random; E1 selection, common Archive30k retention, and Structure10k acquisition are deterministic.
- New seeds use the preregistered namespaces S1 `{1: 2605011, 2: 2605012}`, S2 `{1: 2605021, 2: 2605022}`. This changes random realization but not operator distributions, budgets, mutation kernels, selection rules, features, quotas, or tie-breaking. Historical Original namespaces remain `{1: 5, 2: 6}`.
- Original E0/E1 datasets, labels, M4 initialization, 10,240-update training, checkpoints, and evaluator hashes reproduce, so Original is included as a comparable historical replicate.
- TSPLIB22 is never used in evolution, acquisition, training, or model selection.

No mutation, Concorde, or new BQ training occurred during this audit.
