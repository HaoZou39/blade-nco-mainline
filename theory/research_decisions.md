# Research decisions — Stage 1

1. **Target the mathematical question narrowly.** Analyze finite-budget first discovery of a time-indexed true-hardness region, not generic optimization quality or unrelated BLADE components.
2. **Use phenotype-level structural states.** Encodings with the same solver-visible instance map to one phenotype; lineage and archive details remain in the full memory state because they can affect transitions.
3. **Keep true hardness separate from HGB40.** The hit event uses a declared true/verified hardness metric; HGB40 is a selection surrogate whose error can change parent retention.
4. **Represent BQ co-adaptation explicitly.** Hardness is `h(x; theta_t, q_t)`, so solver updates can move the hard region even if the instance generator is unchanged.
5. **Normalize by compute, not generations.** The comparison charges the work needed to produce and evaluate each proposal. Equal deterministic costs are an explicit restriction of the first proof.
6. **Select a conditional hazard certificate as the strongest result.** It is rigorous and potentially testable but not novel by itself. Any BLADE-specific contribution must derive the hazard advantage from actual structural reachability, survival, selection, surrogate, and cost data.
7. **Reject universal memory/diversity claims.** Exact finite-state counterexamples disprove that stale memory or diversity preservation must help, and show HAC can win when it has a useful gradient signal.
8. **Do not infer implementation behavior from labels.** The exact meanings of FOGA-11, Elite-QD1000 (P=T), HGB40, BQ, and HAC require implementation documentation or logs.
9. **Record novelty conservatively.** Prior theory already establishes population/diversity effects in particular dynamic optimization settings. The distinction here may be solver-relative hard-instance discovery under compute cost, but novelty is OPEN.
10. **Stop at Stage 1.** No BLADE implementation changes, Oracle evaluations, solver training, evolutionary runs, or new experimental designs were undertaken. Further theory work waits for human approval.

## Statement-change log

- No candidate statement was altered to rescue a proof.
- The generic claims that inherited memory or diversity always improves hitting time were kept as universal claims and marked **DISPROVED** by explicit examples.
- The surviving finite-window claim was formulated conditionally before its proof: it requires a verified conditional hit-hazard advantage and equal per-opportunity compute cost.
- The BLADE-vs-HAC/fixed-generator claim remains **OPEN**; it was not replaced with the conditional proposition and presented as if equivalent.

## Stage 1.5 addendum

11. **Replace the finite reachable-state premise with a measurable instance space.** A finite run produces a finite realized set, while continuous-coordinate FOGA kernels can retain uncountable support. The finite object is the QD partition index set.
12. **Separate the two clocks.** `k` indexes BQ solver/adaptation cycles; `(i,o)` indexes the deterministic parent–operator attempts inside a cycle. The proof’s `j` indexes whichever comparison opportunity is explicitly declared.
13. **Ground the current method in S6-AE, not historical GP4.** The inspected S6-AE/S6-AC/S5-B/S4-B path and its 40/40 strict verification define the implementation mapping used here.
14. **Do not model deterministic scheduling as random selection.** All 10,000 parents receive all 11 operators. Randomness belongs to declared roots, operator parameters/repair, calibration, estimator/training seeds, comparator policies, or an explicitly modeled execution ensemble.
15. **Separate four quantities.** True current BQ hardness, HGB40 prediction, generation-level hard-region discovery, and acquisition-level discovery are not interchangeable.
16. **Separate three status axes.** Mathematical proof validity, applicability to verified S6-AE, and scientific novelty are recorded independently.
17. **Correct the surrogate bridge obstruction.** Even perfect current-hardness prediction cannot distinguish equal-hardness candidates with different mutation futures; deterministic tie-breaking can remove the only bridge.
18. **State lumpability at the right strength.** Strong lumpability is a sufficient all-initial-laws condition. A weaker distribution-dependent Markov aggregation is possible only after proving the relevant conditional mixture property for a specified law.

## Stage 1.5 statement-change log

- Proposition 1 and its proof engine are preserved. Only the opportunity index and S6-AE applicability mapping were corrected.
- The generic `r γ α` factorization remains mathematically available for genuinely random parent-selection policies, but it is explicitly marked inapplicable to S6-AE. The current-method analogue is a direct batch hazard or `r α_batch` bound.
- The stale-memory, gradient, and diversity counterexamples are unchanged in substance.
- The former “surrogate error” counterexample is corrected: surrogate error is unnecessary; an equal-hardness tie under perfect prediction already blocks a universal future-reachability guarantee.
- No new theorem, empirical claim, or novelty claim is introduced.

## Stage 2 addendum

19. **Separate the elite floor from the complete population.** `Q_k` contains one tie-broken HGB maximum per occupied candidate niche. `T_k=P_(k+1)` additionally contains the global HGB fill to 10,000.
20. **Prove only the invariant supported by that mechanism.** With a fixed partition, parent-inclusive union, sufficient capacity, the complete niche floor, and `P=T`, occupied niches are monotone. Parent hashes and within-niche transition structure are not.
21. **Represent rejection as probability mass, not resampling.** Each scheduled operator has an extended kernel on `X ∪ {∂}`. Invalid/rejected proposals produce `∂`; accepted duplicates are removed by deterministic first occurrence.
22. **Treat bridge viability as a separate quantity.** Current true BQ hardness, HGB prediction, and future-reachability value can order candidates differently. A per-niche HGB elite preserves a bridge only under an explicit score/rank-to-viability condition.
23. **Use a path certificate rather than an assumed aggregate hazard.** The multistep result factors discovery into operator-specific accepted-transition bounds and explicit selection-retention bounds along time-indexed bridge sets.
24. **Allow solver drift explicitly.** Bridge sets, kernels, HGB scores, and the terminal true-hard region are cycle-indexed. The terminal set is defined under the entering BQ checkpoint of the final acquisition cycle.
25. **Keep finite-population effects in the selector.** The exact survival test is niche-elite membership or rank within the remaining global-fill capacity. Niche occupancy alone is insufficient.
26. **Compare mechanisms in their own cost units.** Reset evolution, stationary generation, and HAC with random restarts require separate kernels, restart laws, and charged compute. Mutation attempts, generations, and gradient steps are not equated.
27. **Classify the proved mathematics conservatively.** Niche persistence is an elementary archive consequence; the bridge-chain product is a standard conditional-probability argument. A potentially publishable contribution would require non-vacuous BLADE-specific bounds or a verified separation, neither of which is supplied here.
28. **Stop after the focused Stage 2 investigation.** No experiment, true-hardness evaluation, HGB/BQ training, evolutionary run, or algorithm modification is authorized or performed.

## Stage 2 statement-change log

- Proposition 1 and every Stage 1/1.5 counterexample remain intact.
- A new deterministic niche-persistence proposition is proved under assumptions that are verified for the inspected S6-AE path.
- An exact counterexample blocks the stronger inference from niche persistence to future reachability, even with perfect current-hardness prediction and the complete global-fill mechanism.
- A multistep bridge-chain proposition is proved conditionally. It is not relabeled as a BLADE advantage because its positive transition and retention constants remain unverified.
- The novelty audit now includes MAP-Elites/QD runtime theory, but remains limited rather than claiming exhaustive literature priority.

## Stage 2.1 independent mathematical review addendum

29. **Repair the retention filtration.** Conditional on the complete scored candidate union, Elite-QD is deterministic and bridge retention is 0/1. A nontrivial fractional retention certificate must condition on the pre-generation information and the bridge-generation event; alternatively use a pathwise selector certificate with rho=1.
30. **Prohibit look-ahead in bridge selection.** Select each next bridge set before the batch meant to reach it. The final target can depend on the currently available BQ checkpoint, not future checkpoints or already observed offspring. Bridge/kernel lower bounds must hold over the admissible resulting histories.
31. **State hash compatibility by target property.** Niche persistence only needs deduplication to preserve niche labels, while the bridge theorem needs the stronger preservation of bridge-set membership. True phenotype uniqueness on the realized pool is sufficient for both.
32. **Separate presence from new discovery.** The bridge-chain theorem remains a presence guarantee. New hard offspring, new acquisition, hardness reactivation, and novel occupied QD niches have distinct event definitions.
33. **Preserve Stage 2's conservative status.** The revised theorem is a correct conditional certificate. No numerical a_l/rho_l, BLADE-over-HAC bound, or publishable novelty has been established, and no algorithm, training, or Oracle calls have been changed.

## Stage 3A addendum

34. **Track hardness on the deployed selection scale.** Define `f_k(x)=log(1+LBQ(M_k,x))` and keep raw LBQ, HGB40 prediction, true-hard-region membership, and downstream training value distinct.
35. **Prove the local statement first.** Uniform prediction error on a candidate niche gives a `2 epsilon` current-cycle approximate-argmax guarantee for its tie-broken HGB elite; this is standard and does not establish future value.
36. **Index solver drift by its comparison domain.** The one-step result needs drift only on the current candidates being compared. A historical regret recurrence needs drift on the complete latent historical niche ledger, a much stronger premise.
37. **Use a latent historical oracle only as a proof benchmark.** The union of all prior candidate pools is not an S6-AE archive and is not re-evaluated under each current checkpoint.
38. **Exploit only verified one-step availability.** Parent-inclusive union and `Q subseteq T=P` keep the previous niche elite eligible at the next cycle; they do not preserve every historical representative.
39. **Do not convert an additive recurrence into a steady-state claim.** The bound can grow as the sum of per-cycle prediction and drift terms. An exact zero-drift construction attains linear growth from bounded prediction error.
40. **Separate tracking quality from causal utility.** Low or zero current-hardness regret does not imply bridge retention, future hard-region reachability, or improved downstream BQ evaluation without additional alignment assumptions.
41. **Classify novelty conservatively.** Approximate argmax and dynamic-regret-style drift accounting are standard supporting mathematics. No BLADE-specific publishable tracking theorem is claimed without non-vacuous verified premises.
42. **Stop at the justified conclusion.** The Stage 3A decision is **C -- TRACKING_ASSUMPTIONS_NOT_JUSTIFIED**. No experiment, exact-hardness evaluation, HGB retraining, BQ adaptation, evolution, or algorithm modification was performed.

## Stage 3A statement-change log

- Stage 1 through Stage 2.1 proofs, counterexamples, and research decisions remain unchanged.
- The new one-step propositions are additions on the verified S6-AE selector semantics, not replacements for the mechanistic bridge theorem.
- Historical regret is benchmarked against a latent union and is never described as an implemented archive statistic.
- The strongest recurrence is retained with its full assumptions and is not weakened after the fact to manufacture applicability.
- Five exact counterexamples preserve the distinctions among solver drift, surrogate error, niche coverage, reachability, and downstream training value.
