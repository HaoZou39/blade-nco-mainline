# P6-C0 Protocol

This is a read-only mechanism audit over the frozen P6-B artifacts. No mutation, evolution, survivor selection, Concorde call, training, HGB refit, checkpoint write, or new benchmark generation is permitted.

- M4 SHA256: `c749311bd2a461084d0cbcfd383a4c0a1fc1268762b72e5f7220952f108f8190`
- P6-B completion SHA256: `bc1be7ecb677b2c270b49f7e63a4a649a28124fafa40d03b5d4674d2881bbeb8`
- Exact hardness: `-(1/T) * sum_t log p_theta(a_t^* | s_t^*)` over all 99 TSP100 decisions, batch 128, eval mode.
- Hard-tail thresholds: pooled M4 P90/P95 across all 60,000 memberships.
- Structural near-duplicate thresholds: pooled P1/P5/P10 NN distances, all reported (no post-hoc threshold choice).
- Instance bootstraps are conditional descriptive intervals and are not evolution-replicate uncertainty.
