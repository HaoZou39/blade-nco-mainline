# Preregistered statistical analysis

## Analysis populations

1. Scheduled-attempt population: all 300 attempts per operator. Rejection and duplicate outcomes remain in denominators.
2. Valid-unique population: accepted, `valid_after_repair=True`, nonduplicate children only.
3. Complete matched contrasts: for each operator pair, parents for which both children have M5 labels and are valid unique. The denominator is always printed.

The primary full analysis waits until every valid child's M5 label is present. Existing selected-only child labels are reported as a biased availability subset and are not used for operator claims.

## Primary outcomes

For each operator report:

- valid acceptance and unique usable yield per 300 scheduled attempts;
- mean, median, standard deviation, quartiles, and 95% parent-bootstrap CI for raw `delta_h` and `delta_f`, conditional on valid unique output;
- positive improvement fraction `delta_h>0`, both per attempt and conditional on valid;
- material improvement fraction `delta_h>0.01`, both denominators;
- hard-hit rate using frozen `tau=0.12629708954782198`, both denominators.

For unconditional binary yield, invalid/rejected/duplicate attempts equal zero success. No numeric `delta_h` is imputed for them.

## Matched inference

- Resampling unit: parent, retaining its eleven attempts. Bootstrap: 10,000 replicates, seed `20261028`.
- Report unweighted sample estimands and design-weighted C05-population estimands using `analysis_weight` from the parent manifest.
- Omnibus operator tests precede pairwise interpretation: parent-blocked/permutation comparison for binary yields and parent-fixed-effect comparison for continuous valid-pair outcomes.
- Pairwise operator contrasts: 55 predeclared pairs. Apply Holm correction separately to the raw-`delta_h`, positive-yield, and hard-hit families at familywise alpha 0.05. Report corrected p-values, effect sizes, CIs, and pair denominators; do not reduce results to significance labels.
- Heterogeneity: repeat descriptive effects by QD4 region and entering component only when a cell has at least 20 parents. Otherwise pool and list the sparse cell. These are exploratory and do not create operator-selection claims.

## Structural alignment and retention

- Post-repair structural displacement: child-parent Euclidean distance in frozen normalized QD4; also report four signed feature changes and niche migration.
- Raw geometric displacement cannot be reconstructed from the frozen records because raw coordinates are absent; `raw_child_hash` is not a coordinate payload. Operator formula descriptions remain engineering context only.
- Relate post-repair QD4 changes to `delta_h` using parent-clustered descriptive regressions with operator indicators and parent baseline `h5`. Do not call correlations causal or convergence evidence.
- Retention fields: `entered_T05`, `T05_component`, new valid phenotype, and failure/duplicate status. Separate mutation reachability from selection retention.
- HGB diagnostic: residual `hgb_predicted_log1p_hardness - f5(child)`, MAE, RMSE, Spearman, and error by operator/QD region. HGB is never ground truth.

## Pilot gate

The 30-parent pilot does not test operator superiority. Expansion to the remaining 270 requires: zero hash/coordinate mismatches; zero unresolved checkpoint mismatches; all requested valid instances labeled; no verified Concorde failures; repeated M5 labels agreeing within `1e-4`; and observed combined wall time no more than twice the preflight proxy unless separately approved. Failure pauses for artifact repair; thresholds and outcomes remain frozen.

Identity: `M5_E256` SHA-256 `ef07226bb8c3e8cd3a1404eece474df44d378316fd66509b762eb5f8bc26558c`; 300 matched parents, 3,300 original attempts; source manifest SHA-256 `e21fc9f4e989f14193407b6194b885ab4e5e8ab523ba300bec3550caf8bfd158`.

