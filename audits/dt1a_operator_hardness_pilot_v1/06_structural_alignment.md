# Structural alignment

## Post-repair displacement

`delta_phi` is Euclidean displacement in the four frozen normalized QD coordinates. In a descriptive model with operator and parent fixed effects, the `delta_phi` coefficient was `-0.02876`, parent-cluster robust SE `0.01864`, 95% interval `[-0.06529, 0.00778]` (`n=326`, 30 parent clusters). This does not show a stable monotone structural-displacement/hardness relationship.

Operator-specific Spearman estimates were mostly weak and their bootstrap intervals crossed zero. Grid had the most negative estimate, `rho=-0.374` with exploratory 95% interval `[-0.689,-0.004]`; this is one of eleven exploratory correlations, is not multiplicity-confirmed, and must not be promoted to a mechanism claim.

The pilot covers 12 of the sample's 16 coarse QD4 regions. Operator×region cells contain only 1–7 parents, below the preregistered minimum of 20; therefore `work/qd_region_descriptive.csv` is descriptive only and no region-specific operator conclusion is made.

## Repair and rejection

- 221 accepted children were valid before repair: mean `delta_h=0.00796`, positive fraction `0.575`, hard-hit fraction `0.190`.
- 105 accepted children required repair: mean `delta_h=0.01484`, positive fraction `0.552`, hard-hit fraction `0.190`.
- Four additional attempts failed after 22–40 repair passes and have no hardness delta.

This raw grouping is confounded by operator and parent region. It shows why failures cannot be dropped, but it does not estimate a causal repair effect.

Raw pre-repair coordinates were not frozen; `raw_child_hash` cannot reconstruct them. Consequently this report distinguishes operator-formula geometry from measured post-repair QD4 displacement and makes no raw-contraction claim.

Sources: `02_30parent_sample_manifest.csv`, `03_true_lbq_evaluation_manifest.csv`, and frozen C05 `candidate_manifest.csv`. Checkpoint SHA-256 `ef07226bb8c3e8cd3a1404eece474df44d378316fd66509b762eb5f8bc26558c`; child coordinate source remains the frozen C05 bundle SHA-256 `4382db31bac27b070fb731138b1653b3b8a36bd765c0a7d8e92f31f775b65ac6`.

