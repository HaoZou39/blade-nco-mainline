# Final Recommendation

1. **M0/M4 importance consistency:** partially consistent, not identical. The exact Top5/Top10 overlap and all 40 raw ranks are in the comparison outputs. Because C00/C04 validation sets and calibration sizes differ, drop changes are descriptive and not a causal M0→M4 adaptation estimate.
2. **Why not Top1/Top2:** P0 preserved them but chose ranks 3 and 6 for a cross-node-size two-dimensional matching space: both were finite, dimensionless, interpretable, and comparatively nonredundant. HGB40 did not itself select the frozen pair.
3. **Exclude scale-related features?** No. Scale/node dependence justifies caution or explicit normalization for cross-size matching, not deletion from importance/modeling. Raw values and ranks must remain.
4. **Exact 70/30:** per generation it requests 7,000 strict-region survivors and 3,000 non-strict frontier survivors from the parent+child union; strict shortages are reassigned to frontier, then 3x3 quotas, microbin spreading, lineage caps, and backfill apply. Realized proportions need not be 70/30.
5. **Support for 70/30:** no literature, theory, or ratio-ablation justification was found. `70_30_RATIO_EMPIRICAL_JUSTIFICATION_NOT_FOUND`.
6. **Actual optimization:** no single objective; it is a lexicographic capacity/quota/microbin/distance/lineage/tie-break heuristic. It uses no hardness.
7. **Six G2 Population10k:** yes, all are valid direct training memberships after attaching expert tours. Reusable/missing label counts: Original-E0 1,468, Original-E1 5,560, S1-E0 6,630, S1-E1 1,639, S2-E0 6,632, S2-E1 1,643. Unique missing union: 23,468.
8. **Keep 70/30?** Keep it only as the frozen B arm for causal comparison. Redefine the next method around an explicit distribution-plus-lineage objective; do not present 70/30 as optimal.
9. **Can the next stage directly compare G2 E0/E1?** Yes. The six frozen G2 populations provide three paired evolution replicates and need no new mutation/acquisition. First label only missing tours, then use the identical M4→M5 protocol. Preregister endpoints and keep TSPLIB22 out of tuning/model selection.

`BLADE_P6_0_FEATURE_AND_POPULATION_PROTOCOL_AUDIT_COMPLETE`
