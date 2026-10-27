# Actual compute audit

## Concorde

- Requested evaluable instances: 356.
- Exact-hash cache hits: 44 (30 parents, 14 children).
- New solver calls: 312; solved and verified: 312; solver failures: 0; solver retries in the successful run: 0.
- Successful-run wall time: 33.7092 s with eight workers.
- Sum of per-instance solver elapsed time: 260.6305 s; mean 0.8354 s; median 0.6981 s.
- Python process CPU time: 8.6563 s.
- Binary SHA-256: `38a647d3f04196231c725e90d23c2c883c0a23899f9bab8a257ae6f777d1dec0`.

An initial sandboxed launch could not access WSL (`Wsl/Service/E_ACCESSDENIED`). It recorded 312 infrastructure failures after one wrapper retry per instance in `work/concorde_failures.csv`; no Concorde solution was produced. The identical frozen list was then run with WSL permission and succeeded without solver retry. This incident changed neither membership nor protocol.

## M5 teacher-forced LBQ

- Reused M5 labels: 20 (six parents, 14 children).
- New forward calls: 336 (24 parents, 312 children); complete/nonfinite failures: 0; retries: 0.
- Device: NVIDIA GeForce RTX 5060 Ti, CUDA 12.8, PyTorch 2.11.0+cu128.
- Checkpoint load: 0.4577 s wall, 0.2813 s process CPU.
- Evaluation: 9.4547 s wall, 9.4537 s CUDA-event time, 8.9844 s process CPU; batch size 128.
- Checkpoint SHA-256: `ef07226bb8c3e8cd3a1404eece474df44d378316fd66509b762eb5f8bc26558c`.
- Protocol: 99-step mean expert-action NLL, ground-truth-prefix teacher forcing, target index 1, no greedy rollout, zero parameter updates.

Successful required compute totaled about 43.16 s excluding checkpoint load (about 43.62 s including it), below the DT1-A0 proxy. Scaling only these measured rates to the remaining frozen panel gives roughly 329 s for 3,044 tours and 92 s for 3,258 forwards before orchestration overhead—consistent with a several-minute full evaluation, not a cost block.

Exact records are `work/concorde_timing.json`, `work/new_concorde_ledger.parquet`, `work/lbq_timing.json`, and `work/new_M5_lbq_labels.parquet`. No frozen input was modified.

