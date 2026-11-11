# Unified-12 operator reference

Default library: `UNIFIED12_DEFAULT`. The predecessor `UNIFIED12_FROZEN_V1` is commit `0aa106ac48b39d9b2098805acce1985285ec688b`; its profile sha256 is `19e9548f417e46e0c1f17f7053f887645cc4e72e1778bca5e783529667a02f9e`. This tree does not load that profile.

Configuration: `src/mutation/unified12_frozen.yaml`.

Each call draws one of the 12 operators with equal probability, then samples that operator's parameters from the default distributions. The same operator may be drawn again on the next call. Strength is not scheduled. The first ten operators keep the audited CURRENT distributions. Stripe and motif are the structural refactor of that library: stripes only, a four-arm cross, and stars with five or six arms.

FOGA operators use the existing repair: minimum spacing `1e-5`, at most 20 passes, and a repair fraction above 0.20 is a failed proposal. Stripe and motif do not use that repair. An invalid child is rejected. Each of those two operators may draw again, up to three times, on the same call. A failed record replays as a failure and does not build another child.

Coordinates are in the unit square. A call does not change the number of nodes and does not write into the parent array.

```python
from mutation import apply_operator
result = apply_operator("implosion_mutation", parent, rng)
```

Replace the operator name to call any of the twelve. `result.sampled_parameters` replays that child when passed back as the explicit parameter dictionary.

## P1 single-step contribution

Source: `outputs/blade_unified12_operator_contribution_audit/coverage_sets/summary.json`. Equal budget, 1500 attempts, 250 roots, depths 0/3/8. Coverage is the frozen reference-set rate. S1, S2, and S3 are EvoReal-2 classes. Stable exclusive regions are R-space probes covered by that operator and by no other operator, with at least the audit's stable-root rule. Retention is the historical-chain rate at which a child kept parent-covered R probes. These are single-step and historical descriptions, not a ranking score.

| Operator | Changed nodes | Displacement | R | E2 | E2 S1 | E2 S2 | E2 S3 | Stable exclusive R | Retention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform | 6.06 | 0.514 | 0.382 | 0.593 | 0.212 | 0.678 | 0.818 | 0 | 0.395 |
| normal | 12.57 | 0.037 | 0.398 | 0.572 | 0.232 | 0.580 | 0.784 | 0 | 0.518 |
| rotation | 37.45 | 0.214 | 0.374 | 0.562 | 0.180 | 0.536 | 0.814 | 2 | 0.187 |
| implosion | 22.35 | 0.096 | 0.338 | 0.549 | 0.374 | 0.592 | 0.744 | 0 | 0.200 |
| explosion | 9.37 | 0.144 | 0.373 | 0.586 | 0.238 | 0.638 | 0.788 | 0 | 0.439 |
| cluster | 27.55 | 0.506 | 0.278 | 0.230 | 0.330 | 0.298 | 0.500 | 0 | 0.022 |
| expansion | 27.51 | 0.162 | 0.323 | 0.552 | 0.262 | 0.568 | 0.800 | 0 | 0.090 |
| compression | 27.74 | 0.059 | 0.353 | 0.585 | 0.272 | 0.604 | 0.838 | 0 | 0.302 |
| axis projection | 24.77 | 0.288 | 0.368 | 0.519 | 0.180 | 0.508 | 0.788 | 0 | 0.105 |
| grid | 25.77 | 0.249 | 0.377 | 0.586 | 0.176 | 0.776 | 0.770 | 0 | 0.189 |
| lattice | 60.71 | 0.087 | 0.391 | 0.587 | 0.222 | 0.844 | 0.694 | 19 | 0.117 |
| motif | 29.95 | 0.150 | 0.322 | 0.481 | 0.336 | 0.582 | 0.712 | 0 | 0.078 |

Uniform moves few nodes a long way. Lattice moves many nodes a short way and is the only operator with a large stable exclusive R region. Implosion and motif are the strong E2 S1 operators. Cluster's broad coverage is low and its historical retention is near zero; its S1 coverage is still real, which is why P1 did not delete it.

## P2-A strength audit

P2-A held every non-target random draw fixed and compared LOW, MID, HIGH, and CURRENT for implosion strength, lattice alpha, and motif alpha. Decision: `DECISION_B`. No alternate third produced a pre-registered primary gain of at least 0.03 that was stable across roots and free of a larger loss in another reported space. The formal yaml therefore keeps the full CURRENT intervals. LOW, MID, and HIGH are not configuration options.

## 01 uniform_mutation

1. Operator name: `uniform_mutation`.
2. Geometric purpose: replace a small set of nodes with new uniform locations.
3. Mathematical transformation: `x_i <- u_i`, with `u_i` uniform in `[0,1]^2`.
4. Affected nodes: a count equal to `round(fraction * N)`, at least 1, sampled without replacement.
5. Strength: the fraction is the only strength control. There is no displacement scale.
6. Distribution: fraction is a choice of `0.03`, `0.05`, or `0.10`.
7. Bounds: those three values.
8. Sampling: draw the fraction, draw the indices, draw the replacement coordinates, draw `repair_seed`.
9. Validation and repair: FOGA repair.
10. Example: `apply_operator("uniform_mutation", parent, rng)`.
11. P1: mean 6.06 nodes moved, mean displacement 0.514. Broad R and E2 coverage stay high. Retention 0.395.

## 02 normal_mutation

1. Operator name: `normal_mutation`.
2. Geometric purpose: add a small Gaussian kick to selected nodes.
3. Mathematical transformation: `x_i <- x_i + ε_i`, `ε_i ~ N(0, σ^2 I)`.
4. Affected nodes: `round(fraction * N)` nodes without replacement.
5. Strength: `σ`.
6. Distributions: fraction `Uniform(0.05, 0.20)`; `σ` log-uniform on `[0.005, 0.05]`.
7. Bounds: those intervals.
8. Sampling: fraction, sigma, indices, noise, `repair_seed`.
9. Validation and repair: FOGA repair.
10. Example: `apply_operator("normal_mutation", parent, rng)`.
11. P1: 12.57 nodes, displacement 0.037. Highest historical retention, 0.518. The coordinate move is small relative to uniform.

## 03 rotation_mutation

1. Operator name: `rotation_mutation`.
2. Geometric purpose: rotate a subset around an explicit center.
3. Mathematical transformation: `x_i <- R(θ)(x_i - c) + c`.
4. Affected nodes: `round(fraction * N)`. The frozen Unified-12 fraction is wider than FOGA11 base.
5. Strength: the angle.
6. Distributions: fraction `Uniform(0.25, 0.50)`; angle `Uniform(-60°, 60°)`.
7. Bounds: those intervals. FOGA11 base still uses fraction `Uniform(0.10, 0.35)` when called through `apply_foga11_operator`.
8. Sampling: fraction, center, indices, angle, `repair_seed`.
9. Validation and repair: FOGA repair.
10. Example: `apply_operator("rotation_mutation", parent, rng)`.
11. P1: 37.45 nodes, displacement 0.214. Two stable exclusive R probes. Retention 0.187.

## 04 implosion_mutation

1. Operator name: `implosion_mutation`.
2. Geometric purpose: pull nodes inside a disk toward its center.
3. Mathematical transformation: `x_i <- x_i + strength * (c - x_i)`.
4. Affected nodes: all nodes within `radius` of `c`. If the disk is empty, the center moves to a random node and the disk is evaluated again.
5. Strength: `strength` in the convex update. Radius selects the nodes and is not the blend weight.
6. Distributions: radius `Uniform(0.20, 0.40)`; strength `Uniform(0.20, 0.80)`.
7. Bounds: those intervals. FOGA11 base radius remains `Uniform(0.10, 0.30)`.
8. Sampling: radius, center, indices, strength, `repair_seed`. Strength does not change which nodes are selected.
9. Validation and repair: FOGA repair. The audited calls had repair fraction 0.
10. Example: `apply_operator("implosion_mutation", parent, rng)`.
11. P1: 22.35 nodes, displacement 0.096, E2 S1 0.374. P2-A found that the upper third of strength raises S1 further and cuts broad R and E2 by more than that gain. The full interval stays.

## 05 explosion_mutation

1. Operator name: `explosion_mutation`.
2. Geometric purpose: push nodes inside a disk onto and beyond its boundary.
3. Mathematical transformation: `x_i <- c + d_i * (radius + extra_i)`, where `d_i` is the unit direction from the center.
4. Affected nodes: nodes within `radius` of a uniform center.
5. Strength: `extra_i`, an exponential kick past the radius.
6. Distributions: radius `Uniform(0.10, 0.25)`; extra shift exponential with scale `0.03`, capped at `0.15`.
7. Bounds: those values. Unchanged from FOGA11 base.
8. Sampling: radius, center, indices, one exponential draw per selected node, `repair_seed`.
9. Validation and repair: FOGA repair.
10. Example: `apply_operator("explosion_mutation", parent, rng)`.
11. P1: 9.37 nodes, displacement 0.144, retention 0.439.

## 06 cluster_mutation

1. Operator name: `cluster_mutation`.
2. Geometric purpose: move selected nodes onto a Gaussian cloud around a new center.
3. Mathematical transformation: `x_i <- c + ε_i`, `ε_i ~ N(0, σ^2 I)`.
4. Affected nodes: an integer count, sampled without replacement.
5. Strength: `σ`, and the count.
6. Distributions: count integer-uniform on `[15, 40]`; `σ` log-uniform on `[0.005, 0.08]`.
7. Bounds: those intervals. FOGA11 base count remains `[5, 25]`.
8. Sampling: count, center, sigma, indices, offsets, `repair_seed`.
9. Validation and repair: FOGA repair.
10. Example: `apply_operator("cluster_mutation", parent, rng)`.
11. P1: 27.55 nodes, displacement 0.506. R 0.278 and E2 0.230 are the low broad-coverage pair. E2 S1 is 0.330. Retention 0.022. P1 did not delete it.

## 07 expansion_mutation

1. Operator name: `expansion_mutation`.
2. Geometric purpose: push nodes that lie in a tube away from the tube's line.
3. Mathematical transformation: `x_i <- projection_i + sign(d_i) * (tube_width + extra_i) * normal`. The node leaves the tube by `extra_i`.
4. Affected nodes: nodes whose absolute signed distance to the random line is strictly less than `tube_width`. The line is redrawn until at least two nodes fall inside, up to 64 times.
5. Strength: `extra`, exponential.
6. Distributions: line angle `Uniform(-80°, 80°)`; tube width `Uniform(0.08, 0.22)`; extra exponential scale `0.03`, cap `0.12`.
7. Bounds: those values. Unchanged from FOGA11 base.
8. Sampling: angle, line point, tube width, indices, extra shifts, `repair_seed`.
9. Validation and repair: FOGA repair.
10. Example: `apply_operator("expansion_mutation", parent, rng)`.
11. P1: 27.51 nodes, displacement 0.162, retention 0.090.

## 08 compression_mutation

1. Operator name: `compression_mutation`.
2. Geometric purpose: pull nodes inside a tube toward the tube's line, without placing them on the line.
3. Mathematical transformation: `x_i <- x_i + strength * (projection_i - x_i)`.
4. Affected nodes: nodes inside the tube.
5. Strength: `strength`.
6. Distributions: angle `Uniform(-80°, 80°)`; tube width `Uniform(0.08, 0.22)`; strength `Uniform(0.30, 0.85)`.
7. Bounds: those intervals. Unchanged from FOGA11 base.
8. Sampling: angle, line point, tube width, indices, strength, `repair_seed`.
9. Validation and repair: FOGA repair.
10. Example: `apply_operator("compression_mutation", parent, rng)`.
11. P1: 27.74 nodes, displacement 0.059. E2 S3 0.838 is the highest in the library. Retention 0.302.

## 09 axis_projection_mutation

1. Operator name: `axis_projection_mutation`.
2. Geometric purpose: move one coordinate of selected nodes to a common horizontal or vertical line, plus jitter.
3. Mathematical transformation: `x_{i,d} <- target + ε_i`, the other coordinate unchanged.
4. Affected nodes: `round(fraction * N)`.
5. Strength: the jitter scale.
6. Distributions: fraction `Uniform(0.10, 0.40)`; axis uniform over `{x, y}`; target `Uniform(0.15, 0.85)`; jitter `Uniform(0.002, 0.03)`.
7. Bounds: those intervals. Unchanged from FOGA11 base.
8. Sampling: fraction, indices, axis, target, jitter, noise, `repair_seed`.
9. Validation and repair: FOGA repair.
10. Example: `apply_operator("axis_projection_mutation", parent, rng)`.
11. P1: 24.77 nodes, displacement 0.288, E2 S1 0.180, retention 0.105.

`linear_projection_mutation` is the oblique version of this move. It remains in `foga2019_operators.yaml` and in `apply_foga11_operator`. It is not in the default 12.

## 10 grid_mutation

1. Operator name: `grid_mutation`.
2. Geometric purpose: snap nodes inside a box onto a local grid, with optional rotation and jitter.
3. Mathematical transformation: grid sites span `±0.38` of the box width and height about the box center. If rotation is on, those sites rotate about the center, then Gaussian jitter is added: `x_i <- R(θ)(g_i - c) + c + ε_i`.
4. Affected nodes: nodes inside a random axis-aligned box. The box is redrawn until at least 4 nodes are inside, up to 64 times.
5. Strength: jitter, and the box size.
6. Distributions: box width and height `Uniform(0.35, 0.65)`; rotation with probability 0.50 and angle `Uniform(-45°, 45°)` when rotation happens, otherwise angle 0; jitter `Uniform(0.001, 0.02)`.
7. Bounds: those intervals. FOGA11 base boxes remain `Uniform(0.25, 0.50)`.
8. Sampling: repeated box width, height, and center until the box is populated; then the rotation coin, the angle only when that coin is true, jitter, the grid coordinates, the noise, and `repair_seed`.
9. Validation and repair: FOGA repair.
10. Example: `apply_operator("grid_mutation", parent, rng)`.
11. P1: 25.77 nodes, displacement 0.249, E2 S2 0.776. Stable exclusive R regions: 0. Lattice, not grid, carries the exclusive R probes.

## 11 stripe_reorganization

1. Operator name: `stripe_reorganization`. The predecessor name `lattice_reorganization` is not in this registry.
2. Geometric purpose: blend a local, regional, or global set of nodes toward rotated stripes. The regular grid template now belongs only to `grid_mutation`.
3. Mathematical transformation: `x_i <- (1 - α) x_i + α m_i`, where `m_i` is the Hungarian match of the selected node to a stripe site. Sites are rotated by a small angle, shrunk into `[0.02, 0.98]` around the template center, then jittered by `N(0, s^2)` with `s ~ Uniform(0, gap/4)`.
4. Affected nodes: every node inside the chosen square. Global scope uses all N nodes and center `(0.5, 0.5)`. Local side length is `Uniform(0.25, 0.45)`. Regional side length is `Uniform(0.55, 0.85)`. Fewer than 4 nodes is an empty region and is recorded as a failed result.
5. Strength: `α`. It does not choose the nodes, the scope, or the jitter.
6. Distributions: scope `{local, regional, global}` with probabilities `0.25, 0.35, 0.40`; angle `Uniform(-10°, 10°)`; `α ~ Uniform(0.35, 1.0)`. Stripes use an axis in `{x, y}` and an integer count in `{2, 3, 4}`.
7. Bounds: global stripe coordinates are placed in `[0.02, 0.98]` on both axes before rotation. Local and regional stripes use the chosen window on both axes. A template whose minimum spacing is below `0.015` is rejected before the blend.
8. Sampling order: scope, angle, alpha, then the scope-specific side and center, then stripe axis and count, then spacing check, jitter scale, jitter noise. A failed draw consumes that attempt's random numbers and the call tries again, at most three times.
9. Validation and repair: no FOGA repair. The child must have shape `(100, 2)`, lie in `[0, 1]^2`, and have pairwise spacing at least `1e-4`. Otherwise the attempt fails. Replaying a failed record keeps the failure and returns the parent coordinates.
10. Example: `apply_operator("stripe_reorganization", parent, rng)`.
11. The predecessor lattice measurements are historical only. At commit `0aa106ac` the P1 lattice call changed 60.71 nodes, with mean displacement 0.087. Those figures are not measurements of this stripes-only operator.

## 12 motif_assembly

1. Operator name: `motif_assembly`.
2. Geometric purpose: blend one local neighborhood toward one motif.
3. Mathematical transformation: the same blend as stripe, `x_i <- (1 - α) x_i + α m_i`, with `m_i` the Hungarian match onto the motif sites.
4. Affected nodes: the `k` nearest nodes to the best of 8 candidate centers. The center maximizes the mean distance of those `k` nodes, which places the motif away from an already tight cluster. `k` is one integer. One motif is built per attempt.
5. Strength: `α`.
6. Distributions, version `v3`: motif uniform over `{arc, cross, l_shape, star}`; `k` integer-uniform on `[20, 40]`; scale `Uniform(0.08, 0.22)`; orientation `Uniform(0°, 180°)`; `α ~ Uniform(0.45, 1.0)`; centers uniform in `[0.18, 0.82]^2`. Arc span is `Uniform(70°, 220°)` and the arc start is `Uniform(0°, 360°)`, drawn only when the motif is an arc. Star arm count is an integer in `{5, 6}`. An L uses a fraction `Uniform(0.35, 0.65)` and keeps two arms. A cross is one center plus four rays separated by 90 degrees; it does not use `cross_fraction`. `grid_block` is not a motif.
7. Bounds: those values. The v1 motif used `k` in `[8, 24]`. The default library is v2.
8. Sampling order: motif type, `k`, scale, orientation, alpha, eight centers, then only the random numbers of the chosen motif, then up to eight shrink steps that do not draw random numbers, then jitter scale and jitter noise. Up to three attempts.
9. Validation and repair: same closed check as stripe. No FOGA repair. Sites are shrunk toward the motif center until they lie in `[0.02, 0.98]`. A motif that cannot be built, including a cross with fewer than five nodes, returns a failed result instead of raising.
10. Example: `apply_operator("motif_assembly", parent, rng)`.
11. P1: 29.95 nodes, displacement 0.150, E2 S1 0.336, retention 0.078. In P2-A, the high third of alpha moved E2 S1 from 0.314 to 0.356 and combined coverage from 0.491 to 0.3115. That trade was not adopted.

## What is not in the default library

- `linear_projection_mutation`, kept for FOGA11 replay.
- `outer_ring_mutation`.
- Crossover of any kind.
- GP-chosen operator probabilities.
- The P2-A LOW, MID, and HIGH parameter slices.
