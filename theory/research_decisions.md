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
