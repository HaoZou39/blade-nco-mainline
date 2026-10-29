# P6-AQ Frozen Protocol

- Offline sensitivity audit only; P6-A/P6-AE are read-only. No evolution, Concorde, BQ/HGB training, benchmark evaluation, feature reselection, reweighting, or P6-B.
- Fifteen preregistered perturbations: a={0.05,0.25,0.50} × per-file lexical q and seeds {2026,260101,260102,260103,260104}; plus nominal baseline.
- Each epsilon is added in the original TSPLIB file coordinate unit; the frozen aspect normalization is applied afterward.
- Candidate Top5, full Uniform calibration, TSPLIB48 membership/weights, unweighted squared distance, exact solver, 10k capacity, and 208/209 quotas remain frozen.
- Split-half calibration is the pre-existing calibration-index parity split and cannot replace the official transform.
- Perturbation is a counterfactual stress test, not reconstructed TSPLIB measurement error.
