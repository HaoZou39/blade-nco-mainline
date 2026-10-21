# Assumption register — Stage 1

Statuses: **PROVED**, **CONDITIONAL**, **CONJECTURE**, **DISPROVED**, **OPEN**.
An assumption marked OPEN has not been verified from implementation artifacts.

| ID | Assumption or modeling choice | Status | What would verify or falsify it |
|---|---|---|---|
| A1 | A finite-budget-reachable phenotype set can be treated as finite; encodings are mapped to solver-visible structural phenotypes. | CONDITIONAL | FOGA-11 branching is finite per proposal and canonicalization/instance representation is specified. |
| A2 | The true hard region is defined by a fixed threshold on a declared BQ performance metric at each evaluation epoch. | OPEN | Specify hardness metric, threshold, solver version/state, and whether a candidate counts on estimate or verification. |
| A3 | The current target is predictable before the candidate draw/evaluation used to test the hitting event. | CONDITIONAL | State when the task/solver changes relative to proposal, score, and evaluation. |
| A4 | Parent availability, parent selection, and mutation-to-target probability have conditional lower bounds r, gamma, alpha on every no-hit history. | OPEN | Log survival, selection, operator, parent, and true target-hit outcomes; verify worst-history or justified uniform bounds. |
| A5 | Fixed-generator or HAC conditional hit hazards can be upper-bounded in comparable compute units. | OPEN | Exact comparator policy and proposal-level candidate/hardness/cost records. |
| A6 | Candidate costs are equal deterministic units for the finite-window theorem. | CONDITIONAL | Include generation, HGB40, archive maintenance, BQ calls/verification, and accelerator/CPU usage. If variable, use a cost-aware bound. |
| A7 | HGB40 ranking is sufficiently calibrated to infer true-hardness reachability or safe elite retention. | OPEN | Calibration/error bounds on the deployment distribution, including distribution shift after BQ updates. |
| A8 | BQ co-adaptation is captured in the state and induces a meaningful time-varying hardness function h(x; theta_t,q_t). | CONDITIONAL | Record solver checkpoints/versions and identify when solver updates affect labels. |
| A9 | Niche-level state aggregation is Markov/lumpable. | OPEN; generally false absent conditions | Show equal aggregate-to-aggregate transition probabilities for all full states in each niche block, or augment state. |
| A10 | The target remains fixed for W evaluation cycles in the window comparison. | CONDITIONAL | Estimate/establish solver and task change rates; otherwise apply history-dependent hazards p_i instead of a fixed-window claim. |
| A11 | Per-proposal hit events are conditionally independent. | PROVED unnecessary for this proof | History-conditional hazard bounds suffice. |
| A12 | Population memory universally improves discovery over fixed generators or HAC. | DISPROVED | Two-state stale-memory and path/gradient counterexamples. |

## Implementation facts not inferred

- `FOGA-11` does not by itself specify mutation support or probabilities.
- `Elite-QD1000 (P=T)` is treated as a supplied label; the meanings of `P` and `T`, archive updates, and selection behavior are unknown.
- HGB40 is modeled as a surrogate because the relationship between its score and verified BQ hardness was not supplied.
- HAC is treated only as a comparator class; no gradient access, update rule, or compute cost is assumed.
- No implementation code, logs, training, oracle calls, or evolutionary runs were examined or executed.