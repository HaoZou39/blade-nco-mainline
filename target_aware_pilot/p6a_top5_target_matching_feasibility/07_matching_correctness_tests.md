# Matching Correctness Tests

| fixture | exact_cost | lp_cost | lp_equals_enumeration | greedy_cost | greedy_equals_exact | unique_pass | capacity_pass | objective_consistent | deterministic | used_k |
|---|---|---|---|---|---|---|---|---|---|---|
| separable_optimal | 2 | 2 | True | 2 | True | True | True | True | True | 2.0 |
| shared_best_greedy_suboptimal | 0.3 | 0.3 | True | 0.40000001 | False | True | True | True | True | 2.0 |
| exact_ties | 2 | 2 | True | 2 | True | True | True | True | True | 2.0 |
| near_ties | 0 | 0 | True | 0 | True | True | True | True | True | 2.0 |
| duplicate_ID_precheck | NA | NA | NA | NA | NA | True | True | True | True | NA |
| insufficient_candidates | NA | NA | NA | NA | NA | True | True | True | True | NA |

The LP equals hand enumeration on every finite fixture, including the case where greedy is deliberately suboptimal. Feasibility, uniqueness, objective consistency, deterministic reruns, ties, duplicate-ID rejection, insufficient-candidate rejection, and numerical near-ties pass.
