# Cycle-0 GP bootstrap audit protocol

Question: does one deterministic uniform Random500 from frozen O0 retain at least 90% of historical Centroid435 ENR10?

- Frozen: M0, N=100, O0=109,785, QD4/P1-P99/K=1000, canonical GP4 2000x60, seeds 2026-2030, teacher-forced H0, five-formula median.
- Arms only: OLD_CENTROID435 and RANDOM500.
- Random500: coordinate-hash sorted O0, first 500 indices of `default_rng(2026).permutation`; one draw, no reroll.
- Evaluation: exact E1/E2 P0, removing only overlap with either arm.
- Gate: `ENR10_RANDOM500 / ENR10_OLD_CENTROID435 >= 0.90`, valid formulas, integrity pass.

No offspring generation, external benchmark, BQ update, GP retuning, or formal-run write.
