# Final Research Conclusion

1. **Model identity:** yes—formal C04 HGB40 predicts `log1p` of formal-M4 teacher-forced CE; this audit applies the frozen inverse transform.
2. **Leakage:** exact training-coordinate overlap is 27–34 per replicate (68 global-unique), while training-root overlap is 98.25%–98.57%; strict root-safe subsets contain 285/349/345 examples.
3. **Strict evaluation:** possible but small and restricted to this evolutionary ecosystem. It shows consistent modest ranking, not deployment-grade calibration.
4. **Source behavior:** full-pool rho is about 0.38–0.40; E0-only about 0.30–0.33; E1-only about 0.10–0.11. HGB correctly orders E0 mean above E1 mean, but underpredicts both.
5. **Tail behavior:** raw-threshold recall is only about 3.3%–3.7% at P90 and 0.1%–0.9% at P95 on full unions; E1-only and strict-root-safe recall is zero because no prediction crosses the thresholds. Continuous AP/AUROC are better than threshold recall but do not repair calibration.
6. **Selection:** no replicate has enough predicted-bin candidates for the frozen quotas, so exact 10k/48-reference/predicted-decile Surrogate-C1 is infeasible and no selection is created.
7. **Restoration/structure:** true-hardness recovery, tail recovery, Oracle overlap, and Surrogate OT are undefined—not zero—because no lawful Surrogate-C1 exists.
8. **Concentration:** existing C0/Oracle lineage and Full-F40 diagnostics are reported; no new Surrogate concentration can occur or be assessed.
9. **Mechanism:** P6-C2 established preliminary value for restoring *true* hardness. P6-C3 shows that the frozen HGB40 cannot reproduce that intervention under the frozen raw-decile constraints. These are distinct conclusions.
10. **Minimal next experiment:** build a time-separated, root-disjoint M4-labeled validation set and audit a leakage-safe, tail-aware/calibrated surrogate offline. Because retraining is outside P6-C3, none is launched.

`P6C3_PREDICTION_GENERALIZATION_SUPPORTED` (modest ranking only; strict subsets are small)

`P6C3_HARD_TAIL_IDENTIFICATION_NOT_SUPPORTED`

`P6C3_SURROGATE_SELECTION_FIDELITY_NOT_SUPPORTED`

`P6C3_DEPLOYMENT_READINESS: NOT_SUPPORTED`

`BLADE_P6C3_HGB40_HARDNESS_SELECTION_FIDELITY_AUDIT_COMPLETE`
