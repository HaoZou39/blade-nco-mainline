# Full-F40 Structural Difference

Standardization uses each replicate's frozen E0∪E1 candidate pool and is independent of benchmark gaps. Exploratory p/q values treat instances as observations and are not causal or evolution-replicate significance.

Top effects across replicates:

| feature | feature_group | is_top5 | median_smd | min_smd | max_smd | direction_consistency | mean_overlap | abs_median_smd |
|---|---|---|---|---|---|---|---|---|
| spn_mst_dists_span | minimum_spanning_tree | False | 0.3046819 | 0.3028548 | 0.3123786 | 1 | 0.8581893 | 0.3046819 |
| spn_mst_dists_skew | minimum_spanning_tree | False | 0.2385098 | 0.2331229 | 0.243007 | 1 | 0.9011602 | 0.2385098 |
| spn_angles_cos_var | angular_spatial_distribution | False | -0.2118092 | -0.2517155 | -0.1980404 | 1 | 0.8856083 | 0.2118092 |
| spn_fraction_shorter_mean_distance | pairwise_distance | False | -0.2115565 | -0.2326933 | -0.1844208 | 1 | 0.8835188 | 0.2115565 |
| spn_mst_depth_varcoeff | minimum_spanning_tree | False | 0.2104322 | 0.1770835 | 0.2604746 | 1 | 0.8806465 | 0.2104322 |
| spn_angles_var | angular_spatial_distribution | False | -0.2026251 | -0.24433 | -0.186984 | 1 | 0.8897071 | 0.2026251 |
| spn_mst_dists_sd | minimum_spanning_tree | True | 0.1554902 | 0.1367029 | 0.1597367 | 1 | 0.900072 | 0.1554902 |
| spn_centroid_sd | spatial_density_hull_outlier | False | -0.1431468 | -0.1445919 | -0.1105513 | 1 | 0.9033609 | 0.1431468 |
| spn_nng_3_strong_components_skew | nearest_neighbor_graph | False | 0.131349 | 0.1259065 | 0.1317923 | 1 | 0.9210606 | 0.131349 |
| spn_nearest_neighbour_span | nearest_neighbor_geometry | False | 0.1272401 | 0.08433863 | 0.1399392 | 1 | 0.9181759 | 0.1272401 |
| spn_mst_depth_skew | minimum_spanning_tree | False | 0.116785 | 0.08534145 | 0.1334133 | 1 | 0.9252116 | 0.116785 |
| spn_angles_varcoeff | angular_spatial_distribution | False | -0.112093 | -0.1238165 | -0.1036882 | 1 | 0.9105034 | 0.112093 |
| spn_nng_3_strong_components_varcoeff | nearest_neighbor_graph | True | 0.1012624 | 0.0759064 | 0.1080604 | 1 | 0.9118169 | 0.1012624 |
| spn_nearest_neighbour_skew | nearest_neighbor_geometry | False | 0.09878812 | 0.06329025 | 0.1031279 | 1 | 0.9281348 | 0.09878812 |
| spn_centroid_min | spatial_density_hull_outlier | False | 0.09816587 | 0.06688178 | 0.1007934 | 1 | 0.9296171 | 0.09816587 |
| spn_hull_edges_mean | spatial_density_hull_outlier | False | 0.07136742 | 0.0434932 | 0.07405361 | 1 | 0.9276835 | 0.07136742 |
| spn_distance_span | pairwise_distance | False | 0.0667329 | 0.06404849 | 0.07451709 | 1 | 0.927481 | 0.0667329 |
| spn_hull_edges_sd | spatial_density_hull_outlier | False | 0.0666147 | 0.05763202 | 0.07274977 | 1 | 0.9216079 | 0.0666147 |
| spn_nearest_neighbour_sd | nearest_neighbor_geometry | False | 0.06511319 | 0.02615955 | 0.07561367 | 1 | 0.9233998 | 0.06511319 |
| spn_hull_dists_varcoeff | spatial_density_hull_outlier | False | -0.06147644 | -0.07241807 | -0.06017343 | 1 | 0.9162306 | 0.06147644 |

Top5 and F35 are explicitly identified in `03_f40_comparison.csv`; niche-level effects are in `03_within_niche_feature_effects.csv`.
