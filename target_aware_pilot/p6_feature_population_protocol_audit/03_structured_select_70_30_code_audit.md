# `structured_select` 70/30 Code Audit

## Exact definition

For survivor capacity N=10,000, the code requests `round(0.70*N)=7,000` **strict target-coverage candidates** and assigns the remainder to **frontier exploration**. This is a requested count split per generation, not a loss weight, probability, mutation mixture, or training-data ratio (`run_p2_bounded_evolution.py:65-81, 709-735`).

Strict means both `nearest TSPLIB48 distance <= r` in median/IQR-standardized two-feature space and inside the target bounding box. The radius r is the median nearest-neighbor distance among the 48 targets. The 3x3 cells use per-axis target terciles (`:217-266, 269-283, 727-735`).

If fewer than 7,000 strict candidates exist, `target_n=min(7000,total_strict_capacity)` and all shortfall moves explicitly to frontier; the population stays 10,000. Consequently, ‘70/30’ is an aspiration, not a guaranteed realized split. For example, Original E1 realized 2,149/7,851 at G1 and 4,813/5,187 at G2.

## Accurate pseudocode

```text
input: unique parent+eligible-child union U, capacity N, TSPLIB48 two-feature geometry
requested_strict = round(0.70 N)
strict = {u: within radius of a target AND inside target box}
strict_n = min(requested_strict, |strict|)
allocate strict_n across 3x3 cells proportional to TSPLIB48 counts, capacity-aware
within each cell: round-robin 12x12 microbins; then distance; lineage cap; hash tie-break
frontier = unselected candidates NOT in strict region
allocate N-strict_n by nearest target cell proportional to TSPLIB48 counts, capacity-aware
within each frontier cell: same microbin/distance/lineage rule
if short: deterministic microbin/lineage backfill from all remaining candidates
return exactly N, sorted by component then instance hash
```

Capacity-aware 3x3 allocation is in `capacity_quotas` and its calls at `:727-759`. Microbin ordering is at `:661-672`: 12x12 target-range bins, candidates sorted in each bin by target distance/hash, then round-robin by within-bin rank. `lineage_take` starts at cap 40 for active populations and doubles it when necessary (`:675-706`); thus diversity is a soft feasibility device, not an optimized diversity score. Deterministic final backfill is `:761-770`. No BQ loss, HGB score, or hardness enters this function (`:717`).

## What is optimized

There is no single scalar objective. It is a lexicographic, capacity-constrained heuristic: (1) attempt a strict/frontier count; (2) approximate TSPLIB48 3x3 mass by quotas; (3) spread across 12x12 microbins; (4) prefer smaller target distance within tie structure; (5) constrain lineage concentration with relaxable caps; (6) deterministic hash tie-break/backfill. It does **not** explicitly minimize JS, Wasserstein, MMD, lineage HHI, or expected hardness.

Geometry preprocessing costs O(48N) time and O(48N) temporary distance storage in the current vectorized implementation; partition sorting is O(N log N) in the usual bound, and lineage passes can repeat while the cap doubles. Feature extraction is separate and normally dominates. Stored selection metadata makes shortages and realized components auditable.

## Evidence for 70/30

The P2.0 request explicitly called it an exploratory configuration and prohibited a parameter search; P2.2 allowed the first version to continue it. Repository code/config/reports and the supplied study records contain no cited literature derivation, theoretical optimum, or prior ratio ablation.

`70_30_RATIO_EMPIRICAL_JUSTIFICATION_NOT_FOUND`

This is a faithful status, not evidence that 70/30 is necessarily poor.

## Why ‘Structure-Guided Evolution’ is still accurate

- Mutation generation: FOGA-11 samples only from the current active population; mutation itself is not steered by the two features (`:1322-1331`).
- Survivor replacement: E0 uniformly samples the canonicalized union with a fixed seed (`:790-801, 1354-1358`); E1 applies `structured_select` (`:1359-1369`). This is the causal structure-guided link.
- Recurrence: saved G1 survivors become G2 parents (`:1367-1371`), so structural selection changes the next mutation distribution.
- Archive retention: the old common bounded archive separately applied `structured_select` above capacity (`:811-831, 1376-1383`).
- Final acquisition: a later, separate archive-based selector created old Structure10k (`:1386-1424`). It is not survivor selection and is removed from the proposed direct-P2 training protocol.
