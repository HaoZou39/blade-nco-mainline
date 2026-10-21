# Proof status — Stage 1.5

Mathematical validity, implementation applicability, and scientific novelty are separate judgments. `PROVED` in the second column never implies `VERIFIED` or `NOVEL` in the next columns.

| Result or claim | Mathematical validity | S6-AE applicability | Scientific novelty | Proof, obstruction, or evidence |
|---|---|---|---|---|
| Measurable TSP instance-space formulation with finite measurable QD partition | **CONDITIONAL** modeling framework | **PARTIAL** | **NOT CLAIMED** | Fixed-N coordinate space can be modeled as a Borel subset; QD niches are preimages of a finite centroid assignment. Full measurability of every implementation map is not separately proved. |
| History-based non-stationary survival and deterministic-cost bound | **PROVED** | **PARTIAL** | **NOT CLAIMED** | Induction on conditional survival. It applies after choosing a precise event, filtration, probability space, and cost unit. It is standard. |
| Finite-window memory-vs-comparator comparison (Proposition 1) | **CONDITIONAL** | **PARTIAL** | **NOT CLAIMED** | Follows from uniform lower/upper conditional hazards, equal opportunity costs, and a stable target. S6-AE scheduling is mapped, but the hazard and comparator bounds are unproved. |
| Generic derivation `p_M ≥ r γ α` for a policy that randomly selects a parent | **CONDITIONAL** | **NOT APPLICABLE** | **NOT CLAIMED** | Ordered conditional bounds suffice without independence. S6-AE does not randomly choose a parent/operator. |
| S6-AE deterministic-batch derivation `p_M ≥ r α_batch` | **CONDITIONAL** | **OPEN** | **NOT CLAIMED** | The schedule is verified, but a history-uniform true-hardness batch-reachability lower bound under a declared RNG ensemble is absent. |
| Strong lumpability of a niche-only state | **OPEN in general; counterexample blocks an unconditional claim** | **OPEN** | **NOT CLAIMED** | Equal aggregate-block transition probabilities for every pair of full states in a block have not been shown. |
| Distribution-dependent Markov aggregation for a specified initial law | **CONDITIONAL possibility** | **OPEN** | **NOT CLAIMED** | Requires history-independent conditional within-block mixtures for that law; it is weaker than strong lumpability and was not established. |
| Inherited parent memory cannot hurt first hitting | **DISPROVED** | **NOT APPLICABLE as a universal claim** | **NOT CLAIMED** | Exact two-state stale-memory example: expected cost 100 versus 2 for a fixed generator. |
| Diversity always reduces first-hitting time | **DISPROVED** | **NOT APPLICABLE as a universal claim** | **NOT CLAIMED** | Exact three-state example: niche-preserving selection costs 2 versus focused cost 1. |
| Structural mutation universally beats HAC-style adaptation | **DISPROVED** | **NOT APPLICABLE as a universal claim** | **NOT CLAIMED** | Exact path example with deterministic gradient progress and rare structural progress. |
| Perfect current-hardness prediction guarantees preservation of future mutation reachability | **DISPROVED** | **NOT APPLICABLE as a guarantee** | **NOT CLAIMED** | Equal-hardness candidates may have different mutation futures; a deterministic tie-break can discard the only bridge even when prediction is perfect. |
| Generation-level true-hardness discovery implies acquisition-level discovery | **OPEN without a retention condition** | **OPEN** | **NOT CLAIMED** | HGB/QD selection can omit generated true-hard candidates; exact hardness is measured only after `T_k` is frozen. |
| BLADE-NCO beats a fixed generator or HAC under normalized compute | **OPEN** | **OPEN** | **OPEN** | Requires exact target, ensemble, transition/hazard bounds, and matched compute accounting. |
| Novelty of a BLADE-specific mechanistic reachability theorem | **NOT A PROOF-STATUS QUESTION** | **NOT APPLICABLE** | **OPEN** | No comprehensive literature-priority search or completed BLADE-specific theorem exists here. |

## Preserved Stage 1 conclusion

The strongest surviving result remains a conditional hazard certificate. The corrected model does not establish a BLADE advantage, and no theorem statement was weakened after proof work to manufacture one. The universal memory/diversity/structural-mutation claims remain disproved; the BLADE-vs-fixed-generator/HAC claim remains open.
