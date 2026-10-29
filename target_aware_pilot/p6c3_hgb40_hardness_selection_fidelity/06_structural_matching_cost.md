# Structural Matching Cost

C0 and Oracle-C1 exact costs/SWD/energy are immutable P6-C1 measurements. No Surrogate/Oracle or Surrogate/C0 ratio exists because strict Surrogate-C1 selection is infeasible. The P6-C1 1.20× gate is not applied here.

| replicate | selection | status | ot_total_cost | ot_mean_cost | top5_swd | top5_energy | mean_true_m4_ce | p90_tail_fraction | p95_tail_fraction |
|---|---|---|---|---|---|---|---|---|---|
| Original | C0 | AVAILABLE_FROZEN | 60429.07 | 6.042907 | 0.7308806 | 0.8041349 | 0.06168541 | 0.0362 | 0.0149 |
| Original | Oracle-C1 | AVAILABLE_FROZEN | 81024.23 | 8.102423 | 0.841612 | 1.023725 | 0.08804011 | 0.1605 | 0.0819 |
| Original | Surrogate-C1 | INFEASIBLE_NOT_SELECTED | NA | NA | NA | NA | NA | NA | NA |
| S1 | C0 | AVAILABLE_FROZEN | 60690.5 | 6.06905 | 0.7323203 | 0.8240809 | 0.0615704 | 0.0391 | 0.0142 |
| S1 | Oracle-C1 | AVAILABLE_FROZEN | 83632.2 | 8.36322 | 0.8608529 | 1.047274 | 0.08940268 | 0.1665 | 0.0844 |
| S1 | Surrogate-C1 | INFEASIBLE_NOT_SELECTED | NA | NA | NA | NA | NA | NA | NA |
| S2 | C0 | AVAILABLE_FROZEN | 60578.42 | 6.057842 | 0.7307133 | 0.8056102 | 0.06105017 | 0.0344 | 0.0136 |
| S2 | Oracle-C1 | AVAILABLE_FROZEN | 81985.59 | 8.198559 | 0.8466615 | 1.013881 | 0.08835688 | 0.1622 | 0.0873 |
| S2 | Surrogate-C1 | INFEASIBLE_NOT_SELECTED | NA | NA | NA | NA | NA | NA | NA |
