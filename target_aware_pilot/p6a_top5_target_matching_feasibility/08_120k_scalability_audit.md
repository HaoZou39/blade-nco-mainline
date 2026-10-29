# 120k-Scale Performance Audit

The real formal C04 union contains 119,530 unique TSP100 candidates and has coordinate-bundle SHA256 `0e8ce149cd6dc93c7cae285483fc67e73bf37460b5ac7e89ab5ccb4256490bb6`. It is used only for offline representation/matching; no historical benchmark performance enters this audit.

| candidate_n | feature_extraction_wall_seconds | feature_extraction_timing_status | cached_top5_transform_wall_seconds | distance_wall_seconds | matching_wall_seconds | selection_validation_seconds | matching_peak_rss_mb | candidate_edges_per_row | mean_cost | greedy_excess_over_exact | solver_reported_optimality_gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10000 | 137.40 | linear estimate from C04 exact F40 109530-new-row run | 0.010 | 0.006 | 7.501 | 0.0002 | 1088.3 | 48 | 26.0477196 | 0.0827 | 0.0 |
| 30000 | 412.21 | linear estimate from C04 exact F40 109530-new-row run | 0.022 | 0.008 | 18.652 | 0.0003 | 1996.1 | 48 | 22.0458055 | 0.1302 | 0.0 |
| 60000 | 824.42 | linear estimate from C04 exact F40 109530-new-row run | 0.033 | 0.015 | 35.386 | 0.0002 | 3331.9 | 48 | 18.9973456 | 0.1913 | 0.0 |
| 119530 | 1504.99 | C04 measured full-F40 run applies to 109530 new + 10000 cache hits | 0.051 | 0.026 | 66.240 | 0.0003 | 5701.1 | 48 | 15.5665252 | 0.2551 | 0.0 |

Feature cost boundary: the frozen C04 run **measured** 1504.99s to service 119,530 F40 requests (109,530 new, 10,000 cache hits) with the exact formal extractor. Smaller-size entries are explicitly linear throughput estimates, not invented measurements. Peak memory was not captured by that historical run and is reported NA; matching RSS is sampled in this audit.

A conservative single-generation estimate using the historical full F40 path is 1790.1s (29.8 min): mutation 218.8s + F40 service 1505.0s + current distance/matching/validation. This is an estimate, not a new evolution timing. A Top5-only production cache may be faster but was not assumed.

MST/pairwise-distance extraction dominates structural computation; matching is not the bottleneck. The full exact solve peaked at 5701.1 MiB RSS on this host, so future runners should provision at least 8 GiB for this implementation. GPU is neither required nor used.
