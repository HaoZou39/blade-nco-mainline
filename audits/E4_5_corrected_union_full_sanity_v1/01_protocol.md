# E4.5 Corrected Union Full Sanity Protocol

- Run: `blade_v2_corrected_union_sanity_seed2026_v1`
- Cycles: exactly C00-C04, producing M1-M5 from frozen M0
- Candidate pool: `unique(previous actual parents UNION new offspring)`
- C00 GP calibration: Random500(O0), while selection uses U0 UNION O0
- Training T: pure global historical GP Top10k from the union
- Parent P: frozen-centroid QD floor plus global GP fill from the union
- BQ: 10k hard + fixed 10k Uniform, LR 3.75e-5, 256 epochs, fixed E256
- Age: observational only; no cap, penalty, quota, or forced replacement
- Archive: cumulative unique formally labelled B/T only; no persistent exact elite
- External evaluation: passive M0-M5 on the frozen eight-set suite
