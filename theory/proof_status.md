# Proof status — Stage 1

| Result | Status | Proof/obstruction |
|---|---|---|
| History-based non-stationary survival and cost bound | **PROVED** | Induction on conditional survival; deterministic proposal costs and pre-proposal predictable targets. The lemma is standard and not claimed as novel. |
| Finite-window memory-vs-comparator comparison (Proposition 1) | **CONDITIONAL** | Follows if memory has a lower conditional hit hazard and comparator an upper hazard on every no-hit history, with equal costs and a stable target window. |
| Derivation `p_M >= r gamma alpha` | **CONDITIONAL** | Requires sequential conditional lower bounds for useful-parent availability, its selection, and true-target mutation/evaluation. No independence assumption. |
| Inherited parent memory cannot hurt first hitting | **DISPROVED** | Exact two-state stale-memory example: expected cost 100 versus 2 for the fixed generator. |
| Diversity always reduces first-hitting time | **DISPROVED** | Exact three-state example: niche-preserving selection has expected cost 2 versus focused policy cost 1. |
| Structural mutation universally beats HAC-style adaptation | **DISPROVED** | Exact path example with an informative deterministic gradient and rare structural progress. |
| Surrogate elite retention is harmless for reachability | **DISPROVED** | Exact one-slot example where a false high surrogate score removes the only bridge to the target. |
| Niche label/count alone is Markov | **DISPROVED** | Two full states in the same niche have different next-niche transitions; lumpability fails. |
| BLADE-NCO beats a fixed generator or HAC under normalized compute | **OPEN** | Exact transition laws, target definition, cost traces, and empirical survival/hazard data are missing. |
| Novelty of a BLADE-specific theorem | **OPEN** | Existing dynamic evolutionary optimization theory covers related population/diversity questions; no comprehensive novelty search or exact BLADE formulation is available here. |

No theorem statement was silently changed after proof work. The broad
advantage hypotheses are retained as false or open claims rather than
weakened post hoc; Proposition 1 is stated from the outset as a conditional
hazard certificate.