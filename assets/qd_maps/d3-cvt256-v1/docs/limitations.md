# Limitations of d3-cvt256-v1

- The map is a frozen working version. It is not an optimal descriptor or an optimal partition.
- K=256 was not selected by a new optimality search.
- The reference contains 16,500 short Unified-12 states from 1,500 Uniform100 roots at depths 0-10. It does not cover the TSP100 space.
- Stage 2B-D found that depth 1000 stays inside the pre-registered support rule, while a long-chain reference was modestly tighter. Reference adequacy is `REFERENCE_INCONCLUSIVE`. The centers were not replaced.
- Stage 2A remains `DECISION_C`. Points that are close in D3 can still be far in geometry. A niche is a behavior region, not one geometric class.
- Stage 2B remains `DECISION_B`. The behavior space is partially supported, not certified complete.
- Support labels use the E_SHORT depth-10 nearest-distance P95 and P99. They diagnose extrapolation. They do not withhold a niche id.
- One-step descriptor autocorrelation on long Unified-12 chains is high. Many successive states are not independent structural samples.
- Five KMeans seeds on a long reference did not reproduce the same integer labels. Co-assignment was much more stable than the adjusted Rand index. Niche integers from any other fit must not be compared with these integers.
- The extractor depends on R, salesperson 1.0.0, and data.table. The packaged R file and schema are the pinned versions. A different salesperson build can change raw features; the manifest then fails its hash check instead of refitting.
