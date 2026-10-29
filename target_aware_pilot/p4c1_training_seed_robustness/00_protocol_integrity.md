# P4-C1 Protocol Integrity

- R0 and R2 each reuse the exact frozen 10,000 coordinates and exact existing Concorde label bundle; all bundle, membership, label and original-checkpoint SHA256 values reproduce.
- Both original runs are a strict paired training-seed comparison at seed `2026`: same formal M4 `c749311bd2a461084d0cbcfd383a4c0a1fc1268762b72e5f7220952f108f8190`, fresh Adam, LR 3.75e-5, 256 epochs, 10,240 updates, batch/microbatch 256/64, dropout 0, single worker and fixed E256 checkpoint rule.
- Training randomness is confined to one `np.random.default_rng(2026)` in the formal trainer, controlling epoch permutations and suffix-length augmentation. The P4-C1 adapter maps only that single audited call to each preregistered seed, asserts exactly one call with original argument 2026, and restores NumPy immediately afterward. Formal A0 source is not modified.
- Existing R0/R2 E000 weights equal formal M4; every new run must repeat the same tensorwise gate.
- Frozen greedy evaluator dataset/config hashes match across original R0/R2 for all eight benchmarks.
- No FOGA, archive retention, acquisition, Concorde, HGB, structure feature or TSPLIB reference computation is rerun.

Status: **PASS**.
