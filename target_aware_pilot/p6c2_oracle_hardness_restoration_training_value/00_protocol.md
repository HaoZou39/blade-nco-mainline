# P6-C2 Frozen Protocol

P6-C2 directly consumes the six byte-frozen P6-C1 optimal memberships. The P6-C1 1.20× OT gate remains a historical NO-GO result but is report-only here. All six runs use the official M4, fresh Adam, LR 3.75e-5, seed 2026, 256 epochs, 10,240 updates, fixed E256 selection, and the P6-B greedy evaluator. No selection is recomputed.

H2 and H3 operational rules were frozen before training and are recorded in `configs/frozen_p6c2_config.json`. No benchmark result may alter them.
