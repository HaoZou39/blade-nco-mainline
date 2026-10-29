# Training Protocol and Log

- Directly reused formal P3/P4-A `train`; E000 exactly equals formal M4: **True**.
- Fresh Adam, LR 3.75e-5, no scheduler/AMP/clipping, batch/microbatch 256/64, 256 epochs, 40 updates/epoch, 10,240 total updates, seed 2026, all 10,000 instances once/epoch.
- R0 final loss 0.04937416, wall 800.072s, peak GPU bytes 2,137,318,912. R1 final loss 0.04359868, wall 802.570s.
- R0 checkpoint SHA256 `72e3d4dcb4a0bc1c5562e9ab161deaffddb03e0829a01798cec078668540e8a1`; membership SHA256 `54bdb644a348dff070379cab6e1a3be316786f011b79d475139e05ed4fa5d38a`; label bundle SHA256 `45161315a9ec570c0ac49bedf270dff2b3d40dc8752cd4e201f0720ba36c9580`.

E256 was frozen before evaluation; no test result selected the checkpoint or protocol.
