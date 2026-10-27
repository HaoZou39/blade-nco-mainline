# Operator comparison

## Outcome

The pilot validates the paired analysis but does not establish a statistically reliable ranking among all eleven operators. The complete-parent raw-`delta_h` omnibus permutation test used 26 parents and returned `p=0.3932`. Scheduled positive-yield and hard-hit omnibus tests used all 30 parents and returned `p=0.3635` and `p=0.5329`. None of the 55 pairwise contrasts survived Holm correction in the raw-delta, positive-yield, or hard-hit families.

Two marginal, operator-specific parent-bootstrap intervals for mean raw `delta_h` excluded zero: implosion `0.02156 [0.00779, 0.03630]` and compression `0.01903 [0.00206, 0.03676]`. These are candidate signals, not multiplicity-adjusted between-operator evidence.

| Operator | Valid / scheduled | Mean raw Δh (95% parent bootstrap) | Median Δh | Positive / valid | Hard hit / valid | T05 / scheduled |
|---|---:|---:|---:|---:|---:|---:|
| implosion | 30/30 | 0.02156 [0.00779, 0.03630] | 0.01033 | 0.667 | 0.333 | 0.133 |
| linear projection | 27/30 | 0.02124 [-0.00364, 0.04618] | 0.01605 | 0.593 | 0.259 | 0.000 |
| compression | 30/30 | 0.01903 [0.00206, 0.03676] | 0.00523 | 0.567 | 0.233 | 0.100 |
| cluster | 30/30 | 0.01418 [-0.01069, 0.03625] | 0.02618 | 0.700 | 0.233 | 0.033 |
| rotation | 30/30 | 0.01216 [-0.00715, 0.03159] | 0.01155 | 0.533 | 0.167 | 0.000 |
| normal | 30/30 | 0.01208 [-0.00710, 0.03133] | 0.00464 | 0.567 | 0.133 | 0.067 |
| explosion | 30/30 | 0.01026 [-0.00056, 0.02206] | 0.00802 | 0.600 | 0.167 | 0.067 |
| grid | 30/30 | 0.00506 [-0.01607, 0.02484] | 0.00295 | 0.533 | 0.167 | 0.033 |
| axis projection | 30/30 | 0.00124 [-0.02015, 0.02018] | 0.01119 | 0.567 | 0.167 | 0.000 |
| expansion | 29/30 | -0.00188 [-0.02208, 0.02113] | -0.00732 | 0.379 | 0.138 | 0.000 |
| uniform | 30/30 | -0.00228 [-0.01675, 0.01049] | 0.00205 | 0.533 | 0.100 | 0.033 |

Nine operators had positive unweighted mean `delta_h`; expansion and uniform were negative. Design-weighted point estimates are included in `04_operator_paired_results.csv` because DT1-A0 oversampled sparse structural strata. Weighting changes magnitudes—for example compression is `0.03294`, linear projection `0.02996`, and expansion `-0.01528`—which reinforces the need for the full panel.

HGB was evaluated only as a diagnostic. Across 326 valid children, `predicted log1p(LBQ) - true log1p(LBQ)` had mean `-0.01407`, MAE `0.03164`, RMSE `0.04474`, and Spearman `0.4321`; it therefore cannot replace true M5 labels.

Invalid attempts remain explicit: three linear-projection and one expansion attempt failed for `excessive_repair`. Per-attempt yields in `04_operator_paired_results.csv` include these four failures as zero successes; conditional deltas do not impute them.

Identity: 30 parents, 330 scheduled attempts, 326 labeled valid unique children; M5 SHA-256 `ef07226bb8c3e8cd3a1404eece474df44d378316fd66509b762eb5f8bc26558c`. Full row-level results are in `work/paired_attempts.csv` and multiplicity-adjusted contrasts in `work/pairwise_operator_contrasts.csv`.

