# Hard-case exact-reference escalation policy

The hard-case escalation policy was defined uniformly for any benchmark instance that failed to obtain an exact certificate after the standard 300-second Concorde protocol. The escalation changes only the Concorde time limit from 300 to 3600 seconds and allows one additional attempt. No dataset, solver parameter, metric, coordinate representation, or exact-reference acceptance criterion is changed.

The policy is independent of BQ model performance and is used solely to obtain exact benchmark reference solutions.

## Frozen tiers

### Tier 1 — Standard
Canonical Concorde; timeout = 300 s.

### Tier 2 — Hard-case escalation
For any benchmark instance that remains unresolved after exhausting the standard protocol:

- use the SAME canonical Concorde pipeline
- use the SAME input instance
- use the SAME coordinate representation
- use the SAME integer scaling
- use the SAME Euclidean metric
- use the SAME Concorde binary
- use the SAME command-line options
- use the SAME exact verification logic
- change ONLY timeout: 300 s -> 3600 s

Each unresolved instance receives AT MOST ONE Tier-2 run.

Eligibility is discovered from the persisted blocker/attempt ledger. The rule is generic and does not hard-code any instance ID.

## Exact-reference admission

Only `HARDCASE_EXACT_SUCCESS` may enter the exact-reference ledger.

A legal Hamiltonian candidate emitted without an optimality certificate is diagnostic only and is not an exact reference.

## Exhaustion

If the single 3600 s attempt does not produce a verified exact reference:

- `hardcase_escalation_exhausted = true`
- `exact_reference_available = false`
- the instance is retained in the benchmark
- no automatic replacement, deletion, heuristic substitution, or longer retry is performed
- scientific status is `EXTERNAL_BENCHMARK_SUITE_V2_ONE_EXACT_REFERENCE_UNRESOLVED`
- further action requires an explicit human decision

## Status language

- During Tier-2 execution: `HARD_CASE_EXACT_REFERENCE_IN_PROGRESS`
- If all references close: `EXTERNAL_BENCHMARK_SUITE_V2_READY_FOR_EVALUATION`
- If Tier-2 is exhausted but one reference remains unresolved: `EXTERNAL_BENCHMARK_SUITE_V2_ONE_EXACT_REFERENCE_UNRESOLVED`
- `EXTERNAL_BENCHMARK_SUITE_V2_BLOCKED` is reserved for genuine integrity/protocol failures (dataset hash mismatch, corrupted ledger, metric inconsistency, source provenance failure, canonical solver invocation corruption). An isolated unresolved exact-reference hard case is not an integrity failure.
