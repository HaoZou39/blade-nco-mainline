# Target-Aware Niche Map

TSPLIB48 aspect-preserving coordinates use the frozen salesperson/F40 transforms, QD P1/P99 clip, and CVT assignment. A Gaussian kernel is evaluated in four-dimensional normalized QD space with bandwidth equal to the median centroid nearest-neighbor distance (`0.10968374`); all 48 references have equal weight. The map has 29 directly occupied reference niches and effective kernel support 132.26. Target-enriched means `p_target >= 1/1000`, frozen without hardness.

Projection is computationally legal but cross-scale validity is limited: 70.8% of references hit at least one clipping boundary and 32.8% of QD cells clip. Scale correlations are below. Top5 metrics remain the independent structural authority.

| feature | spearman_with_n_nodes | clip_zero_fraction | clip_one_fraction | transformed_min | transformed_max |
|---|---|---|---|---|---|
| spn_sum_of_lowest_edge_values | 0.531258 | 0 | 0.4791667 | -0.5901225 | 2.67231 |
| spn_distance_varcoeff | 0.0496645 | 0.04166667 | 0.02083333 | -0.9369803 | 2.771764 |
| spn_mst_depth_mean | 0.8273975 | 0.04166667 | 0.375 | -1.358191 | 3.420494 |
| spn_mst_dists_median | -0.7615404 | 0.2916667 | 0.0625 | -1.56943 | 2.060632 |
