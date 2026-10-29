# Historical experiment integrity

**Result: PASS.** Stored per-instance files reproduce P5-E1 Original = **1.9180%**, S1 = **2.1038%**, S2 = **2.4142%**. M4 initialization, trainer, optimizer, LR, scheduler, batch, 256 epochs/10,240 updates, training seed, E256 checkpoint rule, and greedy evaluator match P6.

The main correction is semantic: P5 did **not** train on an achieved 70/30 strict/frontier set. The survivor selector requested 70% target and 30% frontier, but shortages were reassigned. Final Structure10k acquisition contained strict-cell quota counts 4,120 / 4,030 / 4,003 and transition/backfill counts 5,880 / 5,970 / 5,997 for Original/S1/S2.

P5 and P6 are paired by evolution seed and initial P03 lineage, but not by candidate coordinates or candidate budget. Therefore the observed gap difference is valid; a selection-only causal reading is not.

The P5 support export stores two byte-identical Original70 row copies per seed (420 raw rows). The audit explicitly de-duplicates on replicate + instance identity + coordinate hash + gap and verifies 210 unique paired evaluations before any statistic.
