# Proposed Frozen P6-B Evolution Protocol

```text
P0 = frozen P03 Population10k
for g in {1,2}:
    apply each of 11 FOGA operators once to every current parent
    Union = unique(Parents + eligible Offspring), retaining parent on collision
    extract the frozen raw Top5 and frozen independent-Uniform conditional-score representation
    assign capacities 32x208 + 16x209 by the frozen hash rule
    select exactly 10,000 using the frozen capacity-constrained matcher
    survivors become the next parents
D_train = G2 Population10k
```

E0 uses uniform-random Survivor10k; E1 uses target matching. Both use identical P03, 11-per-parent mutation budget, two generations, and direct G2 training budget. Parents may survive and historical M4 overlap is not penalized.

Cancelled: 70/30, strict/frontier quotas, microbins, lineage caps, Archive30k, final Structure10k acquisition, hardness fitness, and any extra diversity objective. This document specifies a future experiment only; P6-B was not run.
