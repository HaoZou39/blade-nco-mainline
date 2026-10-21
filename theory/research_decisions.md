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
- The surviving finite-window claim was formulated conditionally before its proof: it requires a verified conditional hit-hazard advantage and equal per-proposal compute cost.
- The BLADE-vs-HAC/fixed-generator claim remains **OPEN**; it was not replaced with the conditional proposition and presented as if equivalent.