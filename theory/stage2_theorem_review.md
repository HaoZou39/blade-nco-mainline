# Stage 2 theorem review

This review records the focused structural-persistence and mechanistic-reachability investigation. It extends Stage 1.5 without replacing its proofs, counterexamples, or conservative conclusion. No new experiment, Oracle/true-LBQ evaluation, HGB40 retraining, BQ adaptation, evolution, or algorithm modification was performed.

## 1. Structural persistence proposition

For a finite candidate set `A`, define its occupied niches by

`S(A) = {g(x) : x in A}`,

where `g` is the fixed QD partition. For candidate union `C`, let `Q(C)` contain the maximum-current-HGB candidate in every occupied niche, with ascending instance hash breaking ties. Let `T_m(C)` contain `Q(C)` plus the first `m-|Q(C)|` remaining candidates in global descending-HGB/ascending-hash order.

Under the following assumptions:

1. `P_k ⊆ C_k`;
2. candidate hashes are unique after the declared union rule;
3. the niche map is fixed across cycles;
4. `|S(C_k)| ≤ m ≤ |C_k|`;
5. selection contains every member of `Q(C_k)`; and
6. `P_(k+1) = T_k`,

the deterministic invariant is

`S(P_(k+1)) = S(C_k) = S(P_k) ∪ S(O_k) ⊇ S(P_k)`.

Proof: every occupied candidate niche contributes its elite to `Q(C_k) ⊆ T_k`, giving `S(C_k) ⊆ S(T_k)`. Since `T_k ⊆ C_k`, the reverse inclusion also holds. Parent inclusion then gives monotonicity.

### Status

- **Mathematical validity:** PROVED.
- **S6-AE applicability:** VERIFIED for the inspected path. C00 uses the corrected parent–child union; later cycles use `unique(previous P ∪ current O)`; QD4/normalization/centroids are frozen; at most 1,000 niches must fit in a 10,000-member population; all occupied niches contribute one elite; and `P=T`.
- **Scientific novelty:** NOT CLAIMED. This is an elementary consequence of fixed-cell elitist archive selection. MAP-Elites and subsequent QD theory already use one quality representative per fixed feature cell.

The proposition does **not** imply that an individual parent hash survives, that candidates in one niche have equivalent mutation laws, or that future hard-region reachability is monotone.

The frozen `_selection_info.json` records provide a direct realized-path check: `Q_n = occupied_niches` in every C00–C09 cycle, with counts `435, 831, 991, 1000, 1000, 1000, 1000, 1000, 1000, 1000`; `T_n=P_n=10000`; and `P_equals_T_membership=true`. These observations agree with the proof assumptions but are not the proof itself.

## 2. Rejection-aware mutation model

Each scheduled parent–operator attempt has an extended kernel

`K~_(k,o)(x, · | s)` on `X ∪ {∂}`.

An outcome in `X` is an accepted, valid repaired phenotype. `∂` denotes a rejected or invalid proposal. Every one of the 11 operators is invoked exactly once for every parent. An outcome `∂` produces no child and is not redrawn. Accepted duplicate phenotypes remain accepted outcomes but are removed by the deterministic first-occurrence map; they are also not redrawn.

Conditioning the kernel on acceptance would renormalize away rejection and would model an algorithm with implicit resampling, which is not S6-AE.

- **Mathematical validity:** PROVED modeling construction.
- **S6-AE applicability:** VERIFIED from `generate_offspring` and the mutation manifests.
- **Scientific novelty:** NOT CLAIMED.

## 3. Exact limitation: progressive bridge loss

The counterexample in `counterexamples.tex` uses population capacity `m` and niches `0,1,...,L`. Initially, niches `1,...,L` each contain a future-useful bridge `a_i`; niche `0` contains `m-L` high-current-hardness fillers. At cycle `i-1`, a scheduled operator produces a decoy `d_i` in the same niche as `a_i`. The current true hardness and perfect HGB predictions order fillers above decoys and decoys above bridges.

The niche elite therefore replaces `a_i` by `d_i`, while the fillers consume every global-fill slot. Exactly one bridge is lost per cycle, but all niches remain occupied. After `L` cycles, no bridge remains. A later target is reachable surely from any `a_i` and unreachable from every decoy/filler, so future reachability has fallen to zero despite monotone niche occupancy.

- **Mathematical validity:** PROVED counterexample.
- **S6-AE applicability:** NOT evidence of occurrence; it proves only that the verified selector does not rule the failure out.
- **Scientific novelty:** NOT CLAIMED.

The obstruction is not surrogate error. HGB can be perfect for current hardness while current-hardness quality and future-reachability value disagree.

## 4. Strongest mechanistic reachability theorem candidate

Let `B_0,...,B_L` be measurable, time-indexed bridge sets. `B_l` is intended to be represented in `P_(k+l)`, and `B_L` lies in the true BQ hard region under the entering solver checkpoint of the final cycle.

For each bridge step `l`, assume:

- **Accepted operator transition:** conditional on every admissible history in which a member of `B_l` is retained, the fixed 11-operator batch puts a member of `B_(l+1)` in the candidate union with probability at least `a_l`.
- **Selection retention:** conditional on a generated `B_(l+1)` member, the elite-plus-global-fill selector retains at least one such member with probability at least `rho_l`.
- **Initial bridge:** `P_k ∩ B_0` is nonempty.
- **Terminal correctness:** `B_L` is a subset of the declared true BQ hard region for the final pre-adaptation checkpoint.
- **Cost:** the charged costs of the `L` cycles are declared and sum to at most budget `B`.

Then

`Pr(generation-level hit within B) ≥ product_l a_l × product_(l=0)^(L-2) rho_l`,

and

`Pr(acquisition-level hit within B) ≥ product_l (a_l rho_l)`.

The proof uses ordered conditional probabilities and the tower property; it assumes no independence. A designated operator `o_l` with a uniform lower bound

`K~_(k+l,o_l)(x, B_(l+1) | S_(k+l)) ≥ a_l`

for all admissible retained bridge parents is sufficient for the generation factor because every operator is scheduled. Rejection mass remains included.

### Selection certificates

The retention factor is identifiable from the actual selector. A bridge member survives exactly when it is either:

1. the tie-broken current-HGB elite of its niche; or
2. within the first `m-|Q(C)|` positions of the global order after removing the niche elites.

Thus `rho_l=1` can be certified if every admissible candidate union containing the next bridge has a bridge member satisfying one of those tests. Stronger sufficient conditions include within-niche transition equivalence or a proved monotone relation between current score order and future-reachability value. Neither condition has been verified.

### Status

- **Mathematical validity:** PROVED under the explicit transition, retention, terminal-set, and cost assumptions.
- **S6-AE applicability:** PARTIAL/OPEN. The deterministic all-operator schedule, rejection semantics, capacity, selector, and `P=T` hold. Positive history-uniform `a_l`, non-vacuous bridge sets, and `rho_l` certificates do not yet exist.
- **Scientific novelty:** the product bound itself is STANDARD and NOT CLAIMED as novel. A potentially publishable contribution would require deriving useful BLADE-specific constants under solver co-adaptation or proving a strict compute-normalized separation from a specified comparator.

If every failed `L`-cycle block regenerates the initial bridge condition and the same bounds, block success is at least `p`, yielding `Pr(C_tau ≤ rC) ≥ 1-(1-p)^r` and `E[C_tau] ≤ C/p`. Regeneration is OPEN for S6-AE, so the one-block finite-budget bound is currently the strongest applicable candidate.

## 5. Unresolved implementation assumptions

- A scientific probability ensemble over roots, stochastic operator parameters, calibration/model seeds, and solver-training randomness.
- Measurable support or history-uniform accepted-transition lower bounds for the 11 full mutation/repair kernels.
- Scientifically meaningful bridge sets tied to changing BQ checkpoints.
- A score/global-rank certificate showing that current-HGB selection retains those bridges.
- A true-hardness threshold and operational definition of a generation versus acquisition hit.
- Regeneration after failed bridge chains.
- Comparable charged-compute models for reset evolution, stationary generation, and HAC.

## 6. Comparator limitations

### Reset evolution

Resetting can destroy intermediate bridge states, so it may lack a multistep path available to inherited populations. It can nevertheless win by drawing directly from the true-hard region or by avoiding stale bridges. No ordering follows without its root law, reset frequency, evaluation policy, and cost.

### Stationary generator

A stationary generator's structural coverage and hard-region mass are separate quantities. Its direct hard-region probability may exceed BLADE's chain probability even if it covers fewer niches. Comparison requires its sampling kernel and per-sample scoring/verification cost.

### HAC with random restarts

HAC requires a transition kernel on the full gradient state, initialization/restart law, stopping rule, access to solver gradients, and forward/backward compute accounting. A gradient step, one FOGA attempt, and one 110,000-attempt BLADE generation are not equal-cost opportunities. The preserved toy path example disproves universal structural-mutation superiority but establishes no full HAC ranking.

For all three comparators, structural coverage advantage must be separated from true-hardness generation and acquisition advantage.

## 7. Novelty audit

The limited primary-source audit considered:

- Mouret and Clune, *Illuminating Search Spaces by Mapping Elites*, arXiv:1504.04909.
- Pugh, Soros, and Stanley, *Quality Diversity: A New Frontier for Evolutionary Computation*, doi:10.3389/frobt.2016.00040.
- Dang, Jansen, and Lehre, *Populations Can Be Essential in Tracking Dynamic Optima*, doi:10.1007/s00453-016-0187-y.
- Oliveto and Zarges, *Analysis of Diversity Mechanisms for Optimisation in Dynamic Environments with Low Frequencies of Change*, doi:10.1016/j.tcs.2014.10.028.
- Bossek and Sudholt, *Runtime Analysis of Quality Diversity Algorithms*, doi:10.1007/s00453-024-01254-z.
- Qian, Xue, and Wang, *Quality-Diversity Algorithms Can Provably Be Helpful for Optimization*, arXiv:2401.10539.

Existing theory already covers fixed feature-cell elites, QD coverage/runtime on explicit benchmark structures, stepping-stone benefits on selected combinatorial problems, and population/diversity effects in dynamic optimization. Therefore:

- niche occupancy persistence is standard support, not a publishable theorem by itself;
- the conditional bridge-product proof is also standard support;
- the progressive bridge-loss example is a useful exact methodological warning, but novelty is not claimed;
- a potentially publishable result would need BLADE-specific transition/retention bounds under a moving solver-relative hardness landscape, or a rigorous cost-matched separation from reset/stationary/HAC mechanisms.

No such BLADE-specific result is proved here. The novelty status remains **OPEN** pending a comprehensive literature review and non-vacuous implementation instantiation.

## 8. Stopping assessment

The revised model is sufficiently precise to state and prove a conditional mechanistic reachability theorem without conflating niche occupancy, representative identity, current hardness, surrogate score, and future reachability. It is not yet sufficiently instantiated to prove a nontrivial BLADE-specific reachability advantage. Stage 2 stops at that boundary.
