# Training Protocol Comparison

| Field | Original C04 Hard10k→M5 | P3 Structure10k→Structure M5 |
|---|---|---|
| Initialization | M4 `c749311bd2a461084d0cbcfd383a4c0a1fc1268762b72e5f7220952f108f8190` | identical M4 |
| Architecture | `BQModel_TSP_2x192x512_L9_H12_ReLU_Softmax_ReZero` | identical |
| Training set | frozen C04 Hard10k | frozen P2.2 E1 Structure10k |
| Unique/occurrences per epoch | 10,000 / 10,000 | identical |
| Epochs | 256 | identical |
| Logical batch / microbatch | 256 / 64 | identical |
| Updates | 40/epoch, 10,240 total | identical |
| Optimizer | fresh Adam, LR 3.75e-5, weight decay 0 | identical |
| Scheduler / AMP / clipping | none / false / none | identical |
| RNG | NumPy seed 2026; one permutation per epoch | identical |
| Loss/input | `torch.nn.CrossEntropyLoss`; `expert_ordered_closed_tour_then_random_suffix` | identical |
| Checkpoint selection | E256 fixed before evaluation | identical |
| Decoder | `greedy`, beam 1, KNN=250 | identical |

The only planned core variable is fifth-cycle training membership. The original A0 `train` and `evaluate` functions are invoked directly after redirecting output paths; no trainer or evaluator is reimplemented. TSPLIB22 is evaluated only after E256 is frozen and is never used for checkpoint or hyperparameter selection.

Predeclared interpretation: `STRUCTURE_C5_GAIN_CONFIRMED` requires a paired-bootstrap improvement over Hardness M5 on held-out22 with CI upper below zero, non-worse GAP8 macro, and no material (>0.1 percentage point or significant) regression on Uniform/Gaussian controls. Mixed evidence maps to `STRUCTURE_C5_MIXED`; concordant non-improvement/regression maps to `STRUCTURE_C5_NO_GAIN`; protocol or execution failure maps to `STRUCTURE_C5_INCONCLUSIVE`.
