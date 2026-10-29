# Top Feature Properties and Scale Audit

## Definition source and representation boundary

Definitions were checked against the installed `salesperson` 1.0.0 functions actually invoked by `src/features/extract_salesperson_raw_features.R` (`normalize=FALSE`), not inferred from names. Pairwise distances are Euclidean; ordinary R `sd` is sample SD. The HGB files store `arcsinh((raw - frozen_median)/frozen_scale)`. For the table below those transformed values were exactly inverted to raw feature units.

For convex-hull distances, the package computes perpendicular distance to each hull edge's **supporting line** and takes the minimum; it does not clip the projection to the edge segment. For the 3-NNG feature it constructs a directed k=3 graph and summarizes strong-component sizes.

## M4 Top10 properties

### 1. `spn_sum_of_lowest_edge_values`

- Definition: sum of the n smallest values among all n(n-1)/2 undirected Euclidean pairwise distances.
- Coordinate scaling: degree 1: f(aX)=a f(X), a>0.
- Node-count behavior: explicitly selects n edges; edge order statistics and typical spacing change with n.
- Missing/undefined: finite for nondegenerate n>=2 input; zero/duplicate-heavy inputs can collapse it; TSPLIB48 aspect missing count = 0.
- Predictive contribution: M4 permutation Spearman drop 0.365431.
- Strongest absolute TSP100 candidate-pool Top10 correlation: `spn_distance_min` (rho 0.745). This diagnoses redundancy but is not an elimination rule.

### 2. `spn_distance_min`

- Definition: minimum of all n(n-1)/2 undirected Euclidean pairwise distances.
- Coordinate scaling: degree 1.
- Node-count behavior: extreme statistic over O(n^2) pairs, normally decreases as n grows at fixed area.
- Missing/undefined: finite for n>=2; equals zero when any coordinates duplicate; TSPLIB48 aspect missing count = 0.
- Predictive contribution: M4 permutation Spearman drop 0.056990.
- Strongest absolute TSP100 candidate-pool Top10 correlation: `spn_sum_of_lowest_edge_values` (rho 0.745). This diagnoses redundancy but is not an elimination rule.

### 3. `spn_distance_varcoeff`

- Definition: sample SD of all undirected pairwise distances divided by their mean.
- Coordinate scaling: dimensionless; invariant to positive uniform scaling.
- Node-count behavior: no algebraic n factor, but finite-sample/distributional dependence remains.
- Missing/undefined: undefined if mean distance is zero; finite for all TSPLIB48 aspect variants; TSPLIB48 aspect missing count = 0.
- Predictive contribution: M4 permutation Spearman drop 0.037355.
- Strongest absolute TSP100 candidate-pool Top10 correlation: `spn_mst_dists_sd` (rho 0.486). This diagnoses redundancy but is not an elimination rule.

### 4. `spn_nng_3_strong_components_varcoeff`

- Definition: sample SD / mean of strong-component sizes in the directed 3-nearest-neighbor graph.
- Coordinate scaling: invariant under positive uniform scaling, except distance ties/numerics.
- Node-count behavior: component count/size distribution depends on n although k=3 is fixed.
- Missing/undefined: R sample SD is NA for one component; missing on eil51 and rat195 in TSPLIB48 aspect data; TSPLIB48 aspect missing count = 2.
- Predictive contribution: M4 permutation Spearman drop 0.020274.
- Strongest absolute TSP100 candidate-pool Top10 correlation: `spn_mst_dists_sd` (rho -0.148). This diagnoses redundancy but is not an elimination rule.

### 5. `spn_mst_dists_sd`

- Definition: sample SD of the n-1 edge lengths in the Prim spanning tree.
- Coordinate scaling: degree 1.
- Node-count behavior: MST edge scale changes with point density and sample size n-1.
- Missing/undefined: undefined for fewer than two MST edges; finite for audited data; TSPLIB48 aspect missing count = 0.
- Predictive contribution: M4 permutation Spearman drop 0.020060.
- Strongest absolute TSP100 candidate-pool Top10 correlation: `spn_mst_dists_span` (rho 0.524). This diagnoses redundancy but is not an elimination rule.

### 6. `spn_fraction_shorter_mean_distance`

- Definition: fraction of undirected pairwise distances strictly below their arithmetic mean.
- Coordinate scaling: dimensionless; invariant to positive uniform scaling.
- Node-count behavior: pair count changes with n; finite-sample and geometry dependence remain.
- Missing/undefined: finite for n>=2 and all TSPLIB48 aspect data; TSPLIB48 aspect missing count = 0.
- Predictive contribution: M4 permutation Spearman drop 0.019439.
- Strongest absolute TSP100 candidate-pool Top10 correlation: `spn_hull_dists_median` (rho 0.270). This diagnoses redundancy but is not an elimination rule.

### 7. `spn_mst_dists_span`

- Definition: maximum minus minimum among the n-1 MST edge lengths.
- Coordinate scaling: degree 1.
- Node-count behavior: MST extreme range changes with density and n.
- Missing/undefined: finite for n>=2; degenerates to zero for a single edge; TSPLIB48 aspect missing count = 0.
- Predictive contribution: M4 permutation Spearman drop 0.016596.
- Strongest absolute TSP100 candidate-pool Top10 correlation: `spn_mst_dists_sd` (rho 0.524). This diagnoses redundancy but is not an elimination rule.

### 8. `spn_hull_dists_varcoeff`

- Definition: sample SD / mean of per-node distances to the nearest convex-hull supporting line; hull nodes contribute zero.
- Coordinate scaling: dimensionless; invariant to positive uniform scaling.
- Node-count behavior: hull size, zero fraction, and interior composition change with n.
- Missing/undefined: undefined when mean hull distance is zero (e.g. all nodes on hull); TSPLIB48 aspect missing count = 0.
- Predictive contribution: M4 permutation Spearman drop 0.015525.
- Strongest absolute TSP100 candidate-pool Top10 correlation: `spn_hull_dists_median` (rho -0.809). This diagnoses redundancy but is not an elimination rule.

### 9. `spn_mst_depth_mean`

- Definition: mean of salesperson spandepth values for the Prim spanning tree.
- Coordinate scaling: invariant to uniform scaling, except MST ties/numerics.
- Node-count behavior: tree depth depends directly on tree size, topology, and implementation root.
- Missing/undefined: finite for audited data; TSPLIB48 aspect missing count = 0.
- Predictive contribution: M4 permutation Spearman drop 0.014588.
- Strongest absolute TSP100 candidate-pool Top10 correlation: `spn_nng_3_strong_components_varcoeff` (rho -0.135). This diagnoses redundancy but is not an elimination rule.

### 10. `spn_hull_dists_median`

- Definition: median per-node distance to the nearest convex-hull supporting line; hull nodes contribute zero.
- Coordinate scaling: degree 1.
- Node-count behavior: hull-point fraction and interior distances change with n.
- Missing/undefined: finite; may be zero when at least half the nodes lie on the hull; TSPLIB48 aspect missing count = 0.
- Predictive contribution: M4 permutation Spearman drop 0.013420.
- Strongest absolute TSP100 candidate-pool Top10 correlation: `spn_hull_dists_varcoeff` (rho -0.809). This diagnoses redundancy but is not an elimination rule.

## Raw feature distributions

| source | feature | finite n | missing n | q25 | median | q75 |
|---|---|---|---|---|---|---|
| TSP100_C04_candidate_pool | spn_sum_of_lowest_edge_values | 119530 | 0 | 0.235527 | 0.437784 | 0.662866 |
| TSP100_C04_candidate_pool | spn_distance_min | 119530 | 0 | 0.000187592 | 0.000392932 | 0.00077775 |
| TSP100_C04_candidate_pool | spn_distance_varcoeff | 119530 | 0 | 0.536266 | 0.55918 | 0.5862 |
| TSP100_C04_candidate_pool | spn_nng_3_strong_components_varcoeff | 119530 | 0 | 0.840252 | 0.967415 | 1.13255 |
| TSP100_C04_candidate_pool | spn_mst_dists_sd | 119530 | 0 | 0.0486317 | 0.0517974 | 0.0549199 |
| TSP100_C04_candidate_pool | spn_fraction_shorter_mean_distance | 119530 | 0 | 0.48 | 0.497374 | 0.515556 |
| TSP100_C04_candidate_pool | spn_mst_dists_span | 119530 | 0 | 0.196714 | 0.22283 | 0.262999 |
| TSP100_C04_candidate_pool | spn_hull_dists_varcoeff | 119530 | 0 | 0.665997 | 0.743258 | 0.871455 |
| TSP100_C04_candidate_pool | spn_mst_depth_mean | 119530 | 0 | 6.36 | 7.15 | 8.11 |
| TSP100_C04_candidate_pool | spn_hull_dists_median | 119530 | 0 | 0.0931296 | 0.138715 | 0.184832 |
| TSPLIB48_aspect | spn_sum_of_lowest_edge_values | 48 | 0 | 3.93009 | 5.55374 | 7.92969 |
| TSPLIB48_aspect | spn_distance_min | 48 | 0 | 0.00254908 | 0.00497052 | 0.00942782 |
| TSPLIB48_aspect | spn_distance_varcoeff | 48 | 0 | 0.493538 | 0.531131 | 0.549195 |
| TSPLIB48_aspect | spn_nng_3_strong_components_varcoeff | 46 | 2 | 0.84282 | 1.22092 | 1.7631 |
| TSPLIB48_aspect | spn_mst_dists_sd | 48 | 0 | 0.00962843 | 0.0213864 | 0.0273841 |
| TSPLIB48_aspect | spn_fraction_shorter_mean_distance | 48 | 0 | 0.512563 | 0.530451 | 0.550009 |
| TSPLIB48_aspect | spn_mst_dists_span | 48 | 0 | 0.0738839 | 0.107452 | 0.217127 |
| TSPLIB48_aspect | spn_hull_dists_varcoeff | 48 | 0 | 0.691828 | 0.813968 | 0.883837 |
| TSPLIB48_aspect | spn_mst_depth_mean | 48 | 0 | 7.97479 | 9.84 | 14.0432 |
| TSPLIB48_aspect | spn_hull_dists_median | 48 | 0 | 0.0623447 | 0.0849932 | 0.114117 |

The TSP100 pool has fixed n=100 and coordinates in [0,1]; TSPLIB48 spans node counts and native conventions but uses the frozen aspect-preserving coordinate normalization before extraction. Remaining distribution differences mix structure with node-count effects. In particular, raw order/extreme statistics (`sum_of_lowest_edge_values`, `distance_min`, MST lengths) are not directly exchangeable across n without an explicit nuisance model.

## Normalization is not elimination

There is **no sufficient basis to delete** a high-importance feature merely because it is scale- or node-sensitive. The old P0 reason supports excluding an *unnormalized raw value from a two-dimensional cross-size density-matching space*, not excluding it from HGB importance or future models.

For any degree-1 distance feature `f`, a transparent coordinate-scale correction is `f_scale = f / D`, where `D = sqrt((xmax-xmin)^2 + (ymax-ymin)^2)` is the bounding-box diagonal. If density rather than global scale is the nuisance, a candidate is `f_density = f / sqrt(A/n)` with bounding-box area A, but it is unstable when A=0 and changes the scientific quantity. Node-order effects in a sum/minimum cannot generally be removed by a single dimensional multiplier; they need conditioning on n, a fixed-n target, or an empirically preregistered residualization model.

All such transforms discard some absolute scale/density signal that may genuinely predict solver-relative hardness. Therefore the original feature and raw importance must remain reported alongside any normalized companion. The frozen HGB arcsinh standardization is a robust numeric transform, **not** physical scale/node normalization.
