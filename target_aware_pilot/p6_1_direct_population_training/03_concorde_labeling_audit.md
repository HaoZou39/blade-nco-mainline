# Concorde Expert Label Recovery Audit

| Dataset | Pre-P6 reused | Dataset missing before P6 | Verified | Label bundle SHA256 |
|---|---:|---:|---:|---|
| Original-E0 | 8,532 | 1,468 | 10,000 | `40c1857182372d81ad5d4541357bfeae62ba644865e487d9873016d25ac7ec7e` |
| Original-E1 | 4,440 | 5,560 | 10,000 | `f58731429dcaee5a13689df287e573f9b6f6b44bb789de12970f6881da322302` |
| S1-E0 | 3,370 | 6,630 | 10,000 | `5d164489c3c72bab39e78f455356259bc7ea0c3a5d800925ff8d2e65df15eac0` |
| S1-E1 | 8,361 | 1,639 | 10,000 | `952b759dbec51f514032a11f96bede6e6722593ed7debaaf4b149ee074c47157` |
| S2-E0 | 3,368 | 6,632 | 10,000 | `3c718036604dd609fc2f7034b96aca710433534fa058f9b712bd48b5ff9d377c` |
| S2-E1 | 8,357 | 1,643 | 10,000 | `23fe9bd27cb4fe449114a8f3cf1674e3f2ab22347ffe44d8d827af3beaf4fd41` |

The six memberships contain 55,399 unique hashes. The verified pre-P6 cache covered 31,931; exactly **23,468 unique Concorde calls** were required and shared across datasets.
Global labeling wall-clock: 960.065s; failures: 0; canonical Concorde SHA256 `38a647d3f04196231c725e90d23c2c883c0a23899f9bab8a257ae6f777d1dec0`.
Sandbox diagnostic failures/retries before the unsandboxed canonical run: {"diagnosis": "sandbox syscall restriction; identical canonical binary succeeded outside sandbox", "instance_hash": "8ae2543ea6ad267a86c0ae2258da6f61fb61433e0d9458193a24e34ccce8911f", "instance_replaced": false, "sandbox_batch_failed_attempts": 1, "sandbox_exit_code": -31, "sandbox_isolated_failed_attempts": 3, "unsandboxed_isolated_successes": 1}.

All tours are 0-based Hamiltonian permutations with implicit cycle closure. Integer objectives were recomputed from the exact frozen coordinates; BQ closed-coordinate inputs were validated. No failed instance was replaced.

The prompt table values 1,468/5,560/... are confirmed as missing counts; their complementary reuse counts are 8,532/4,440/....

Status: **PASS**.
