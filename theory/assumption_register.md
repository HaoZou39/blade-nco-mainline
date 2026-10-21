# Assumption register — Stage 1.5

The three status axes are intentionally separate:

- **Mathematical validity:** `PROVED`, `CONDITIONAL`, `DISPROVED`, or `OPEN`.
- **S6-AE applicability:** `VERIFIED`, `PARTIAL`, `OPEN`, or `NOT APPLICABLE`.
- **Scientific novelty:** recorded separately in `proof_status.md`; no assumption is itself a novelty claim.

| ID | Assumption or modeling choice | Mathematical status | S6-AE applicability | What verifies, limits, or falsifies it |
|---|---|---|---|---|
| A1 | TSP candidates live in a measurable solver-visible instance space `(X, 𝓧)`; for fixed N=100, a valid canonical Borel subset of `[0,1]^(100×2)` is sufficient. | CONDITIONAL modeling premise | PARTIAL | Coordinate arrays, validity/repair, and content hashes are verified. A complete proof that every implementation map/kernel is measurable is not present. |
| A2 | The 1,000 QD niches form a finite measurable partition through the frozen four-descriptor map, normalization, nearest-centroid rule, and declared tie handling. | CONDITIONAL | PARTIAL | QD4, frozen normalization, K=1000, and recorded assignments are verified. Boundary/tie measurability is standard for a declared deterministic rule but not separately formalized in code documentation. |
| A3 | A finite search budget makes only the realized sample finite; it does not make the continuous TSP instance space or mutation support finite. | PROVED clarification | VERIFIED | This replaces the incorrect Stage 1 finite-state assumption. The finite toy counterexamples remain valid as counterexamples only. |
| A4 | BQ cycle `k` is distinct from within-cycle mutation attempt `(i,o)`. `q_k` and the 10,000-parent population are fixed during the mutation/acquisition part of cycle `k`. | CONDITIONAL model alignment | VERIFIED | S6-AE calls `core.run_cycle`; it freezes `T/P`, labels `T`, adapts BQ, then prepares the next cycle’s offspring. |
| A5 | Each of 10,000 parents receives each of the 11 FOGA operators exactly once; parent and operator choice are not sampled. | Deterministic implementation statement | VERIFIED | `generate_offspring` loops over stored parents and `OPERATOR_NAMES`. All C00–C09 manifests contain 110,000 rows, 10,000 parents, 11 operators, and exactly 11 rows per parent. |
| A6 | A declared probability space carries only named randomness: initial-root ensemble if varied, content-derived mutation/repair draws, calibration sampling, HGB seeds, BQ permutations/suffixes, comparator randomness, and any declared numerical nondeterminism. | CONDITIONAL | PARTIAL | Seeds and schedules are recorded. With all sources/seeds/runtime conditioned to their frozen values, S6-AE is a realized trajectory and many probabilities are 0 or 1. A scientific repeated-run ensemble is OPEN. |
| A7 | The true hard region is defined by a declared threshold on current true BQ hardness `h_k^BQ`, not by HGB40 prediction. | CONDITIONAL | OPEN | Current LBQ labels and solver checkpoints exist, but the theorem’s scientific threshold κ and treatment of unmeasured candidates remain unspecified. |
| A8 | The target used in a hazard event is predictable before the random draw(s) resolved by that event. | CONDITIONAL | PARTIAL | `q_k` is fixed before S6-AE mutation/acquisition, but a theorem must state whether the opportunity is one mutation, a generation, or acquisition and what randomness remains after conditioning. |
| A9 | Generation discovery, acquisition discovery, HGB thresholding, and post-adaptation hardness are distinct events/quantities. | PROVED definitional separation | VERIFIED | S6-AE scores all candidate-union members with HGB40 but obtains exact current-hardness labels only for frozen `T_k`; BQ adaptation then produces `q_(k+1)`. |
| A10 | A history-uniform lower bound exists for BLADE generation- or acquisition-level true-hardness hazard. | OPEN | OPEN | Requires a declared ensemble and worst-history bounds. Existing traces do not establish these bounds. |
| A11 | For a generic randomly selected parent, `p_M ≥ r γ α` follows from ordered conditional bounds. | CONDITIONAL | NOT APPLICABLE to S6-AE | The algebra is valid for such a policy, but S6-AE has no random parent/operator selection. Its relevant form is a direct batch bound or `p_M ≥ r α_batch`. |
| A12 | Fixed-generator or HAC hit hazards can be upper-bounded in comparable compute units. | OPEN | OPEN | Exact comparator policies, target event, and matched cost traces are required. |
| A13 | Candidate opportunities have equal deterministic cost in Proposition 1. | CONDITIONAL | OPEN | S6-AE mutation, F40/HGB, QD, exact labeling, archive, and BQ costs are heterogeneous. A generation-level cost can be declared, or the lemma’s variable deterministic costs retained; random costs need another argument. |
| A14 | Perfect current-hardness prediction preserves future mutation reachability under elite selection. | DISPROVED | NOT APPLICABLE as a guarantee | Equal-current-hardness candidates can have different mutation futures; a deterministic tie-break may discard the only bridge. No claim is made that this occurred in S6-AE. |
| A15 | HGB40 ranking/calibration is sufficient to transfer generation-level true-hardness reachability into acquisition-level discovery. | OPEN | OPEN | Requires a retention theorem or verified error/ranking condition on the evolving candidate distribution. Current-hardness accuracy alone is insufficient. |
| A16 | BQ co-adaptation is represented by `q_k`, and selection/labeling under `q_k` precede the update to `q_(k+1)`. | CONDITIONAL model premise | VERIFIED | The S6-AE run-cycle order and checkpoint manifests establish this timing. |
| A17 | Niche aggregation is strongly lumpable. | OPEN; generally false without equal block-transition probabilities | OPEN | Must show equality of transitions into every aggregate block for all full states in each source block. The finite counterexample shows this cannot be assumed from niche identity alone. |
| A18 | A weaker distribution-dependent Markov aggregation holds for a specified initial law. | CONDITIONAL possibility | OPEN | Must prove that the conditional within-block mixture produces history-independent aggregate transitions for that law (possibly time-inhomogeneous). One-time averaging is insufficient. |
| A19 | The target remains fixed for `W` declared comparison opportunities. | CONDITIONAL | PARTIAL | It is fixed within a cycle while `q_k` is frozen; it need not be fixed across BQ adaptations. |
| A20 | Per-opportunity hit events are conditionally independent. | PROVED unnecessary for the survival proof | NOT APPLICABLE | History-conditional hazard bounds suffice. |
| A21 | Population memory universally improves discovery over fixed generators or HAC. | DISPROVED | NOT APPLICABLE | Preserved stale-memory and path/gradient counterexamples reverse the claimed ordering. |

## Verified S6-AE implementation facts

- The current grounding is `scripts/run_s6ae_hgb40_eliteqd1000_pt_10k_hard_only_x1_m0_m10.py` through the S6-AC/S5-B/S4-B implementation chain, not the historical GP4 protocol.
- S6-AE strict verification reports 40/40 PASS.
- C00 uses the hash-locked corrected `unique(U0 ∪ O0)` candidate union; later cycles use `unique(previous P ∪ current O)`.
- HGB40 is a five-seed median HistGradientBoostingRegressor on the frozen ordered F40 list, targeting `log1p(LBQ)` from a current-solver archive calibration set.
- Selection is deterministic: one maximum-current-HGB candidate per occupied niche, then global descending HGB fill to 10,000, with ascending instance hash as tie-break.
- `P_(k+1) = T_k` exactly. C00, C05, and C09 artifacts (and strict checks across all cycles) confirm equal membership hashes.
- Exact current-hardness labeling occurs after `T_k` is frozen. BQ training then produces the next solver checkpoint.
- BQ training randomness (per-epoch hard-set permutation and suffix sampling) is separate from acquisition and parent/operator scheduling.

## Still OPEN

- The scientific hard-region threshold and whether a generation hit must be exactly evaluated to count operationally.
- A probability law representing scientific repetition rather than one frozen seeded execution.
- History-uniform generation/acquisition reachability bounds and a bridge from HGB/QD selection to true-hardness retention.
- A compute-matched fixed-generator or HAC kernel and its hazard bound.
- Strong or weak lumpability of any proposed niche-only population state.
