# P6-F0 frozen protocol

This audit is a read-only transformation of historical assets. It ran no FOGA mutation, candidate generation, Concorde, HGB/GP fitting, BQ training, greedy inference, or cross-population CE inference.

- Scope: P5-E1 Original/S1/S2 versus P6 B/C/D25/D50/D75 and frozen M4.
- Primary endpoint: stored OriginalTSPLIB70 per-instance greedy gap.
- Split: frozen TSPLIB48/22 manifests; the 22 identities are used only to report evaluation.
- Predeclared structural variables: the two actual P5 features `spn_distance_varcoeff` and `spn_fraction_shorter_mean_distance`; P6 Top5/F40 are not post-hoc screened against gap.
- Uncertainty: no pooling of 210 repeated benchmark evaluations as independent samples. The P5-vs-B CI resamples evolution seed and TSPLIB identity hierarchically (10,000 draws, seed 20261028).
- Tour geometry: existing Concorde tours only; deterministic lexicographically-smallest 256 coordinate hashes per population.
- Missing evidence remains missing: no new P5 hardness or cross-population CE inference was launched.
