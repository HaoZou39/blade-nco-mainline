# BLADE-NCO V2 amended protocol

Shared C00 computes M0, O0, Random500 B0, GP0, global-GP T0, A1, and M1 exactly once. The branch is only P0_FULL (QD centroid coverage plus GP fill) versus P0_GPONLY (global GP Top10k, identical to T0). First divergent offspring generation is C01. Every adaptation is 10000 hard plus the fixed 10000 Uniform anchor at LR 3.75e-5 for exactly 256 epochs; E256 is the only scientific checkpoint. External evaluation is passive.

At the user's direction, the FULL training horizon was shortened from M20 to M15, and the GP-only arm was subsequently extended to the matched endpoint M15. All other scientific settings are unchanged.
