# DT1-A0 research question and frozen protocol

## Decision question

Under the frozen entering checkpoint `M5`, do the eleven scheduled FOGA11_HIGH_V1 operators systematically produce children with greater true BQ hardness than their matched C05 parents?

The estimand is restricted to the observed C05 parent population and the frozen deterministic seed schedule. It is not an operator-wide expectation over every state or every random draw.

## Frozen experiment unit and sample

- Matched unit: one parent. Its eleven original C05 attempts stay together.
- Population: the 10,000 unique entering C05 parents in `C04/06_selection/P_next.npz` and `C05/04_qd/parent_qd.parquet`.
- Sample: 300 parents, seed `20261027`, selected without reading LBQ outcomes.
- Strata: four QD4 normalized coordinates dichotomized at the protocol-fixed midpoint `0.5` (16 regions), recorded lineage operator, and entering C04 selection component (`quality_qd_elite`/`hgb_fill`). Every nonempty joint stratum receives one parent; 80 remaining places are allocated proportional to residual stratum size by largest remainder. Within-stratum order is SHA-256 pseudorandom.
- Weights: `stratum_size/sample_n`; both design-weighted population estimates and unweighted design-sample estimates must be reported.
- Pilot: the 30 rows marked `pilot30=True` in `03_parent_sampling_manifest.csv`. The fixed rule gives 2 or 3 parents to every recorded lineage operator. The pilot may validate mechanics and cost only; it may not choose operators, thresholds, or analyses.

The sample covers all 16 QD4 regions, all 11 lineage operators, and both selection components. There is no frozen categorical field named “initial structural type.” `origin_operator` plus frozen QD4 location are the auditable proxies; no missing type label is invented.

## Frozen outcomes

For instance `x`:

`h5(x) = L_BQ(M5, x)` and `f5(x) = log(1 + h5(x))`.

For valid unique child `(i,o)`:

`delta_h(i,o) = h5(child(i,o)) - h5(parent_i)` and `delta_f(i,o) = f5(child(i,o)) - f5(parent_i)`.

- Positive improvement: `Y(i,o) = 1[delta_h > 0]`.
- Material improvement: `1[delta_h > 0.01]`; `0.01` is frozen here before completing missing labels.
- Hard hit: `Z(i,o) = 1[h5(child) >= tau]`.
- `tau = 0.12629708954782198`, the independently frozen P90 of the C05 10,000-instance archive calibration panel in `C05/_calibration_info.json`. It is not derived from the 300-parent offspring outcomes.

Raw LBQ and `log1p(LBQ)` must remain separate. Invalid, rejected, or duplicate attempts have no `delta_h`; they count as zero successes in per-scheduled-attempt yields and remain explicit in conditional analyses.

## Execution boundary

DT1-A0 is a read-only readiness audit. It did not run BQ, Concorde, mutation, training, a pilot, dual-checkpoint evaluation, SVM fitting, or a PR operation. If a human approves DT1-A1, it must reuse the hashes and coordinate indices in the CSV manifests; it must not regenerate offspring.

## Identity and source

- Checkpoint: `audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C04/09_bq_training/M5_E256.tsp.best`.
- Checkpoint SHA-256: `ef07226bb8c3e8cd3a1404eece474df44d378316fd66509b762eb5f8bc26558c`.
- Original attempts: `.../cycles/C05/03_mutation/mutation_manifest.parquet`, 110,000 rows, SHA-256 `e21fc9f4e989f14193407b6194b885ab4e5e8ab523ba300bec3550caf8bfd158`.
- Frozen child coordinates: `.../cycles/C05/03_mutation/offspring_coordinates.npz`, 109,554 rows, SHA-256 `4382db31bac27b070fb731138b1653b3b8a36bd765c0a7d8e92f31f775b65ac6`.
- Sample/attempt volumes: 300 parents, 3,300 attempts, 3,281 evaluable unique children, 19 rejected attempts.

