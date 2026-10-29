# Frozen Training Protocol

| authority | trainer_sha256 | m4_sha256 | training_seed | optimizer | scheduler | learning_rate | epochs | optimizer_updates | logical_batch_size | microbatch_size | loss | checkpoint_rule | evaluation | formal_source | training_order | device |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P6-B actual frozen C04->M5 implementation | e9fc32126cf841502901e36c48a648c28e03c986cdfbe942dfe022e9925f94b5 | c749311bd2a461084d0cbcfd383a4c0a1fc1268762b72e5f7220952f108f8190 | 2026 | fresh Adam | NA | 3.75e-05 | 256 | 10240 | 256 | 64 | teacher-forced imitation | fixed E256 | P6-B frozen greedy evaluator | P6-B/P6-C2 core.train | ["Original-A", "Original-B", "Original-C", "S1-A", "S1-B", "S1-C", "S2-A", "S2-B", "S2-C"] | cuda:1 |

Every run checks E000 tensor equality against formal M4 and fixed-E256 completion. No early stopping or benchmark-driven selection is permitted.
