# Final decision

## A — DT1A_READY_FOR_PILOT

The original 300×11 pairing is complete; the M5 identity is exact; the outcome-blind sample and 30-parent pilot are frozen; label/tour gaps are enumerated by exact hash; and the measured-order cost is acceptable. This authorizes submission of the 30-parent pilot for human approval, not execution.

Evidence:

- 10,000 unique entering parents and exactly 110,000 frozen C05 attempts.
- Sample: 300 parents, all 16 coarse QD4 regions, all 11 lineage operators, both entering selection components.
- Sample attempts: 3,281 valid unique children, 19 explicit repair failures, zero duplicates.
- M5: `M5_E256.tsp.best`, SHA-256 `ef07226bb8c3e8cd3a1404eece474df44d378316fd66509b762eb5f8bc26558c`.
- Full gap: 3,258 new M5 forward labels, including 3,044 instances needing Concorde; conservative same-environment combined proxy ≈7.1 minutes.
- Pilot gap: 336 new M5 forward labels, including 312 Concorde calls; proxy ≈44 seconds.
- The current 237 labeled-child subset is retention-selected and is explicitly barred from standalone operator inference.

Non-blocking caveat: current mutation source files postdate the run and do not equal the run-recorded implementation hash. No replay is allowed or needed; the fixed manifests, coordinates, seeds, hashes, and stage checksums are sufficient for this matched-record experiment. Raw pre-repair coordinates are unavailable, so raw displacement is not an empirical endpoint.

Minimum next step: a human reviews this package and, if approved, runs only the 30 rows marked `pilot30=True`, reusing the recorded coordinates/tours and checkpoint. Apply the gate in `07_preregistered_statistical_analysis.md`; do not use pilot outcomes to select operators or revise the primary analysis. Then STOP pending a second human decision on the remaining 270 parents.

No pilot, LBQ, Concorde, mutation, training, SVM, dual-checkpoint experiment, theorem extension, or PR merge was run by DT1-A0.

Core source: C05 mutation manifest SHA-256 `e21fc9f4e989f14193407b6194b885ab4e5e8ab523ba300bec3550caf8bfd158`; child coordinate bundle SHA-256 `4382db31bac27b070fb731138b1653b3b8a36bd765c0a7d8e92f31f775b65ac6`; machine-readable audit counts and source checksums are in `integrity_summary.json`.
