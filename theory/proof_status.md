# Proof status — Stages 1.5–3B

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
| Rejection-aware extended mutation kernel with no resampling | **PROVED modeling construction** | **VERIFIED** | **NOT CLAIMED** | `∂` carries invalid/rejected mass; each scheduled call occurs once. Accepted duplicates are removed by a deterministic first-occurrence map. |
| Per-niche floor `Q_k` equals the full selected population `T_k` | **DISPROVED except when no fill is needed** | **FALSE in S6-AE** | **NOT CLAIMED** | `Q_k` has one member per occupied niche; `T_k` adds global HGB fill to reach 10,000. |
| Fixed-partition occupied-niche persistence, `S(P_k) ⊆ S(P_(k+1))` | **PROVED** | **VERIFIED** | **NOT CLAIMED / STANDARD ARCHIVE CONSEQUENCE** | Parent-inclusive union plus a complete per-niche floor and `P=T` give `S(P_(k+1))=S(C_k)`. |
| Preservation of individual parent hashes from niche persistence | **DISPROVED as an implication** | **NOT GUARANTEED** | **NOT CLAIMED** | A parent can lose both the within-niche elite test and global-fill rank while another candidate preserves its niche. |
| Monotone niche occupancy preserves future hard-region reachability | **DISPROVED** | **NOT APPLICABLE as a guarantee** | **NOT CLAIMED** | Exact capacity-aware counterexample progressively replaces all bridge representatives while preserving every occupied niche. |
| Exact elite/global-fill membership criterion | **PROVED** | **VERIFIED** | **NOT CLAIMED** | Direct characterization of the deterministic selector. |
| Multistep bridge-chain finite-budget bound (Proposition 3) | **PROVED under explicit transition and retention premises** | **PARTIAL/OPEN** | **NOT CLAIMED; STANDARD PROBABILITY ARGUMENT** | Corrected ordered conditional products use pre-generation-conditioned retention or a pathwise deterministic rank certificate, with causally chosen terminal set. Schedule and selector match S6-AE; positive kernel and bridge-retention certificates are absent. |
| Repeated-window expected-cost bound `E[C_tau] ≤ C/p` | **PROVED under regeneration** | **OPEN** | **NOT CLAIMED** | Geometric block survival is valid only if every no-hit block restores the initial bridge condition and the same conditional bounds. |
| Current-HGB elite quality implies future-reachability value | **DISPROVED without alignment assumptions** | **OPEN for any restricted S6-AE claim** | **NOT CLAIMED** | Perfect current-hardness ranking can prefer structurally dead decoys over lower-current-hardness bridges. |
| Structural comparison with reset evolution/stationary generation/HAC | **CONDITIONAL framework only** | **OPEN** | **OPEN** | Requires method-specific kernels, restart laws, target events, and charged compute; step counts are not interchangeable. |
| BLADE-NCO beats a fixed generator or HAC under normalized compute | **OPEN** | **OPEN** | **OPEN** | Requires exact target, ensemble, transition/hazard bounds, and matched compute accounting. |
| Novelty of a BLADE-specific mechanistic reachability theorem | **NOT A PROOF-STATUS QUESTION** | **NOT APPLICABLE** | **OPEN** | No comprehensive literature-priority search or completed BLADE-specific theorem exists here. |

| Fully scored pool admits a nontrivial fractional retention probability conditioned on all selector inputs | **DISPROVED** | **VERIFIED deterministic selection** | **NOT CLAIMED** | Given complete candidates, scores, niches, and tie breaks, retention is an indicator; a fractional bound must condition on pre-generation information. |
| Causally adapted bridge target under moving BQ solver | **PROVED conditional theorem formulation** | **PARTIAL / OPEN** | **NOT CLAIMED** | Select each next bridge set pre-batch; final bridge is included in the current entering-checkpoint true-hard region. No future-checkpoint or offspring look-ahead is allowed. |
| Hash first-occurrence dedup guarantees exact niche/bridge set identities unconditionally | **CONDITIONAL** | **PARTIAL** | **NOT CLAIMED** | Niche equality requires equal-hash niche compatibility; bridge chaining requires preservation of bridge membership or injectivity on the realized pool. |
| Parent-inclusive true-hardness presence implies new hard-instance discovery | **DISPROVED** | **VERIFIED modeling distinction** | **NOT CLAIMED** | Old hard parents can satisfy presence without new accepted offspring. A cycle-0 valid child identical to its current parent is not a new phenotype; new-generation events must exclude P_k as well as earlier generated pools. Acquisition novelty, reactivation, and new niches remain distinct. |
| Per-niche current-cycle approximate argmax, `f_k(e_(k,j)) >= sup_(C_(k,j)) f_k - 2 epsilon_(k,j)` | **PROVED under uniform error** | **PARTIAL / OPEN** | **STANDARD APPROXIMATE-ARGMAX LEMMA** | The selector semantics are verified; a complete-pool per-niche uniform HGB40 error certificate is not. |
| One-step moving-solver guarantee with loss `2 epsilon+2 delta` | **PROVED under local drift** | **PARTIAL / OPEN** | **STANDARD TRIANGLE-INEQUALITY EXTENSION** | Parent inclusion and retained-elite availability are verified. A suitable drift bound on the compared candidates is not. |
| Historical-unavailability decomposition, `R_(k,j) <= 2 epsilon_(k,j)+A_(k,j)` | **PROVED under current-pool uniform error** | **OPEN** | **STANDARD DECOMPOSITION** | The historical benchmark and penalty are latent because complete historical pools are not re-evaluated under the current solver. |
| Historical tracking-regret recurrence, `R_(k,j) <= R_(k-1,j)+2 epsilon_(k,j)+2 delta_(k-1,j)` | **PROVED under complete-pool error, elite availability, and full-history drift** | **OPEN** | **STANDARD DYNAMIC-REGRET-STYLE RECURRENCE** | The proof is valid; its strongest premises are not justified by current S6-AE artifacts. |
| Bounded per-cycle error and drift imply a bounded long-run tracking-regret steady state | **DISPROVED** | **NOT APPLICABLE as a guarantee** | **NOT CLAIMED** | A zero-drift, one-niche, capacity-one construction realizes linear `2k epsilon` regret. A contraction, refresh, archive, or bounded-unavailability condition is additionally necessary. |
| Low current-hardness regret implies future reachability or downstream BQ training benefit | **DISPROVED without alignment assumptions** | **NOT APPLICABLE as a guarantee** | **NOT CLAIMED** | Separate exact examples give zero tracking regret with bridge loss and with unchanged or worse downstream evaluation. |
| Exact Elite-QD/global exchange, `T_k \ G_k = Q_k \ G_k` and symmetric difference at most `2K` | **PROVED** | **VERIFIED in C00-C09** | **STANDARD CONSTRAINED TOP-K EXCHANGE** | The global-fill portion lies inside `G_k`; only floor elites outside `G_k` are forced, displacing the same number of global members. |
| Nonnegative predicted QD opportunity cost `Delta_QD` | **PROVED** | **VERIFIED / EXACTLY COMPUTABLE** | **STANDARD** | Pure Global Top10k maximizes the HGB score sum. The result concerns surrogate scores, not true LBQ. |
| Selected/current labels point-identify full-pool niche maxima, elite regret, or hard-tail recall | **DISPROVED** | **NOT IDENTIFIABLE FROM INSPECTED ARTIFACTS** | **STANDARD SELECTIVE-LABEL IMPOSSIBILITY** | Two finite truth completions agree on every observable and differ arbitrarily on unlabeled candidates. |
| Existing labels point-identify the true current-checkpoint `T_k`-versus-`G_k` comparison | **DISPROVED for the frozen label pattern** | **NOT IDENTIFIABLE FROM INSPECTED ARTIFACTS** | **NOT CLAIMED** | Every cycle has unlabeled members of `G_k \ T_k`; their values can reverse the full-set comparison. |
| Known bounded hardness range gives sharp `G_k`-minus-`T_k` sum interval | **PROVED conditional** | **OPEN** | **STANDARD PARTIAL IDENTIFICATION** | Endpoint assignment to each unlabeled displaced member attains the bounds. No finite scientific upper endpoint is verified. |
| At-most-`K` paired-label certificate for exact true `T_k` versus `G_k` | **PROVED** | **STRUCTURAL PART VERIFIED; LABELS INCOMPLETE** | **ELEMENTARY BUT PRACTICALLY USEFUL** | Label only missing members of `G_k \ T_k` under the same checkpoint; `T_k \ G_k` is already labeled and common members cancel. |
| Selected-population hardness distribution and threshold count | **PROVED identifiable on `T_k`** | **VERIFIED** | **DESCRIPTIVE, NOT NOVEL** | Every acquired member has a current-checkpoint label. The claim does not extend to `C_k`. |
| Acquisition hardness identifies downstream solver-training value | **DISPROVED without a causal assumption** | **NOT APPLICABLE as a guarantee** | **NOT CLAIMED** | Training response is a distinct outcome requiring an intervention or validated causal model. |

## Preserved Stage 1 conclusion

Stage 2 adds one unconditional implementation-level structural invariant:
fixed-partition niche occupancy is monotone under the verified union/floor/`P=T`
mechanism. It does not establish structural-representative or true-hardness
persistence. The strongest mechanistic result is the conditional multistep
bridge-chain bound. Its proof is valid, but its BLADE-specific transition and
retention premises remain open, and no BLADE-vs-fixed-generator/HAC advantage
or theorem novelty is claimed.

Stage 3A adds only conditional tracking statements. The one-step bounds are
mathematically sound and the historical recurrence is sound under its stated
premises, but complete-candidate HGB error and full-history solver-drift bounds
are not verified for S6-AE. The recurrence can accumulate linearly, and low
current-hardness regret does not by itself certify reachability or training
value. The Stage 3A decision is **C -- TRACKING_ASSUMPTIONS_NOT_JUSTIFIED**.

Stage 3B adds one exact implementation-level selection identity and a standard
selective-label impossibility result. The deterministic `Q/T/G` tradeoff and
predicted-score cost are fully observable. Key full-pool true-hardness claims
are not point identified because most unselected candidates, including some
members of `G_k \ T_k` in every cycle, lack current-checkpoint labels. An exact
paired selector comparison would need at most 1,000 additional displaced-set
labels, but Stage 3B does not obtain them. The Stage 3B decision is
**B -- IDENTIFIABILITY_LIMIT_CONFIRMED**.
