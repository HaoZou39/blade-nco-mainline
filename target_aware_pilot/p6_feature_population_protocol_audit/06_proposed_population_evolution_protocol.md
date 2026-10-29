# Proposed Population Evolution Protocol

## Frozen causal skeleton

```text
P0 = formal P03 Population10k
for g in {1,2}:
    offspring = FOGA11(P[g-1], 20,000 attempts)
    union = unique(P[g-1] + eligible offspring), keeping parent on hash collision
    P[g] = SurvivorSelection(union, 10,000)
D_train = P2
attach/reuse verified Concorde tours
BQ M4 -> M5: frozen 256 epochs / 10,240 updates
evaluate frozen Original70, TSPLIB22, GAP5 protocol
```

Archive30k retention and archive-based Structure10k acquisition are removed from the training path. An archive may remain diagnostic-only for lineage/reachability and must not determine D_train.

## A — original random survivor

Canonical hash-sort the unique parent+child union, then sample exactly 10,000 without replacement using the preregistered generation seed. This is an interpretable uniform baseline with no structure objective.

## B — original 70/30 survivor

Apply the audited multistage two-feature heuristic: request 70% strict target coverage, reassign infeasible shortfall to frontier, match 3x3 reference masses capacity-wise, use 12x12 microbin round-robin ordering and relaxable lineage caps, then deterministic backfill. It has no hardness term and no demonstrated optimality for the 70/30 ratio.

## C — recommended objective-defined successor (design only)

Use a fixed, preregistered feature representation and choose the 10,000-subset S from union U by the explicit objective

`min_S  MMD_w^2(F(S), F(TSPLIB48)) + lambda * HHI(lineage(S))`, subject to `|S|=10,000`, unique valid TSP100.

Here MMD uses a fixed characteristic kernel; `w` is a frozen nonnegative importance weighting from M4 (reported without deleting scale-sensitive features); HHI is sum of squared lineage shares. Kernel bandwidth, lambda, feature list, missing handling, and any dimensional/node normalization must be declared before solver evaluation. Deterministic hash ordering resolves ties. Capacity infeasibility is naturally represented by the best achievable objective rather than an arbitrary target/frontier ratio.

For scale-sensitive features, retain raw features and add explicitly declared normalized companions or condition the reference transform on n/coordinate scale. Because all candidate populations are TSP100 in [0,1], candidate-to-candidate comparison is already controlled for n and coordinate range; the cross-size TSPLIB48 reference is the part needing a nuisance treatment. Do not tune feature subset, lambda, kernel, or normalization against Original70/TSPLIB22 solver gaps.

C is a proposal, not an implemented or validated method. The immediate clean experiment should first compare existing A versus existing B on final G2 populations, because those six frozen datasets already isolate the survivor-to-next-generation causal path and require no new mutation.
