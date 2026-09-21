# E4 Parent + Offspring Survivor Replay Protocol

## Scope

- Audit: `E4_parent_offspring_survivor_replay_v1`
- Mode: offline, read-only scientific replay
- Primary arm: completed FULL / QD+GP trajectory
- Verified cycles: C00–C14
- Formal artifacts fingerprinted before replay: 229

## Counterfactual

For C00, the union is `unique(U0 UNION O0)`. For C01+, the union is
`unique(actual historical P_(s-1) UNION O_s)`. Every cycle is a one-step local
replay. Counterfactual parents are never propagated into later cycles.

## Frozen mechanisms

- Historical GP4 ensemble for each cycle is loaded from the completed run and is not refit.
- Saved exact QD4 descriptors, frozen P1/P99 normalization, and frozen 1000 CVT centroids are reused.
- Training selection is stable global Top10k by `(-GP score, canonical coordinate hash)`.
- Full parent selection is one centroid-nearest candidate per reached niche, then stable global GP fill.
- Canonical `instance_hash` is the coordinate hash and is the union deduplication identity.

## Prohibited work

No BQ training, checkpoint writing, mutation generation, GP fitting, QD refitting,
external evaluation, or Concorde invocation is present in this replay. Cached
exact-hardness diagnostics use only already-saved cycle `T/labeled.parquet`
values and are explicitly non-primary.
