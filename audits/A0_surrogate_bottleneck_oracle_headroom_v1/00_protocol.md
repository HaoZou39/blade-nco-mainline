# A0 protocol (as executed)

Mode: READ-ONLY / OFFLINE / NO NEW TRAINING / NO NEW CONCORDE.

Frozen snapshot: the A1-A4 file `audits/blade_parallel_mechanism_analysis_v1/00_INPUT_SNAPSHOT.json`.
ANALYSIS_CYCLES = C00-C04. SOLVERS = M0-M5. C05+ is excluded even if E4.6 later completes.

## Non-interference
No BQ training, GP refit, mutation, Concorde, T/P/O/C edits, QD-map edits, archive edits,
external-benchmark reruns, A1-A4 overwrites, or writes into active run cycle directories.
No GPU BQ inference. Missing current-H is marked CURRENT_H_NOT_AVAILABLE.

## Independent probe hierarchy
A. Permanent sealed P0 with exact H_s under the *union* current solver.
B. Other leakage-excluded GP working-set probes with matching-solver exact H.
C. Other sealed target-blind exact current-H probes.

T_s is never the primary independent evaluation set. B_s trained GP_s and is excluded.

Union C00 / M0: sealed P0 minus union B0, exact H from S1 `06_p0_current_hardness.parquet` (M0).
Union C01-C04: sealed P0 exists but exact H under union M1-M4 does not; offspring-only S1 H_s is not a substitute.

## GP application
Frozen union `gp4_ensemble.joblib`, five formulas, median ensemble, QD4 features from the sealed P0 CSV
(the same four features used by formal GP4). No refit, no recalibration.

## Metrics
k_q = max(1, floor(q * n)). H_RANDOM = panel mean exact H_s.
GP_TOP = top k by GP prediction; ORACLE_TOP = top k by exact H_s; tie-break canonical instance_hash.
ENR = H_sel / H_RANDOM. ETA = (H_GP - H_RANDOM) / (H_ORACLE - H_RANDOM). Not clipped.
Primary q = 0.10. Sensitivity q = 0.05 and 0.20. Operational q = 10000/|C_s|.
All Oracle quantities are PANEL-ORACLE, not full-candidate-pool Oracle.

Bootstrap: 10,000 paired instance resamples, seed 2026, cycle-specific namespace; rankings recomputed inside each replicate.

## QD descendants
A1 QD_RESCUED / SHARED_GP / GP_ONLY_DISPLACED identities are reused (hash-verified).
Descendants are expanded from actual offspring parent_hash links. Missing H is not computed.
TIER1 = target-blind independent exact H. TIER2 = selection-conditioned (T_r / B_r). Never pooled.

## Downstream
A0 measures hardness-acquisition headroom, not external-gap headroom.
