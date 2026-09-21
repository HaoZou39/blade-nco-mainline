# E4.6 Corrected Union Long-Run M15 Protocol

- Source: verified E4.5 C00-C04, final M5 `398eef00c2a8f44e07fc099dd4ffaa97a718d0bc9d0107e9e4771ffa93219b11`
- New cycles: C05-C14 only; final solver M15
- Candidate pool: `unique(previous actual P UNION new O)`
- GP: deterministic Random500(A_s), frozen GP4 and seeds 2026-2030
- T: pure global GP Top10k; P: frozen-centroid QD floor + GP fill
- BQ: 10k hard + frozen 10k Uniform, LR 3.75e-5, 256E, fixed E256
- Age ledger continues from C04; no cap, penalty or offspring quota
- Evaluation: passive M0-M15 frozen eight-set suite; M0-M5 reused after hash verification
- Comparator: verified historical offspring-only Full M0-M15
- Execution: one deterministic runner, atomic stages, manual `--resume`; no watchdog, polling loop or automatic restart
- Storage: D-path C05-C14 junctions backed by `C:\NCO_E4_6_corrected_union_longrun_M15_v1`; logical task paths are unchanged
