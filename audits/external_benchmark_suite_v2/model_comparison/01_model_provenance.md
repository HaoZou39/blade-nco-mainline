# Model provenance

## BLADE / BQ
- Family: BQ greedy TSP solver from frozen formal run `blade_final_20cycle_seed2026_v1`.
- Comparison checkpoints: every contiguous valid scientific state M0...M16. Intermediate cycles are diagnostic; latest is M16, not a selected best checkpoint.
- These are locally trained by the frozen BLADE protocol.

## Vanilla LEHD
- Repository / code: local checkout `D:\NCO\LEHD` corresponding to CIAM-Group/NCO_code LEHD-TSP.
- Author-released checkpoint: `D:\NCO\LEHD\TSP\result\20230509_153705_train\checkpoint-150.pt`
- Epoch: 150
- SHA-256: `ead8dc462d9b1f6a09fa77025ac8a0c594660f3f1f02fb0d104eeffe48fffa8c`
- Origin: author-released, not locally trained in this task.
- Primary inference: official greedy (start at node 0, argmax, RRC budget = 0).

## EvoReal-adapted LEHD
- Official repository bundle: `C:\Users\Hao Zou\Downloads\EvoReal-main`
- Checkpoint: `EvoReal_Main\EXP_checkpoints\LEHD_TSP_EvoReal_checkpoints\p2-checkpoint-3.pt`
- This is the author-supplied phase-2 (final) LEHD-TSP fine-tune, epoch 3.
- SHA-256: `c788ebd22a126e1eaf3f5767a127f74c078607c86bbdd0623c35911173102b0c`
- Origin: author-released / officially supplied in the EvoReal EXP_checkpoints folder. Not trained in this task.
- Model code: official LEHD TSPModel (identical architecture; 73 matching state-dict tensors).
- Primary inference: the same official greedy protocol as Vanilla LEHD.

## OriginalTSPLIB70 input preprocessing (LEHD-native)
Official `TSPEnv_inTSPlib.load_problems` uses a single global min/max over all x and y values of that instance:
`(xy - min(xy)) / (max(xy) - min(xy))`.
This is not per-axis min-max and not BLADE's translate-by-axis-minima / divide-by-max-span transform.
Final tour cost is still scored with native TSPLIB EUC_2D on the original unnormalized coordinates.

## Synthetic / frozen N=100 and GM500
LEHD receives the frozen coordinates as stored (already unit-square model-input for TSPLIB100). No extra geometry-changing normalization.

## Status
Primary greedy 8-set comparison executed. RRC-50 was not run (secondary only).
