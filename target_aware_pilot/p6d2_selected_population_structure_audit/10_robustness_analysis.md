# Statistical and Robustness Analysis

Original/S1/S2 are shown separately. Instance bootstrap describes frozen memberships; root-aware bootstrap resamples lineage roots and is the dependence sensitivity analysis. With three evolution replicates, direction consistency and effect size are primary; instance-level multiplicity-adjusted q-values remain exploratory.

| replicate | metric | estimate | instance_ci_low | instance_ci_high | root_ci_low | root_ci_high | root_count_a | root_count_b |
|---|---|---|---|---|---|---|---|---|
| Original | true_m4_ce | 0.009410506 | 0.007426233 | 0.01137441 | 0.006291742 | 0.01265072 | 701 | 728 |
| Original | spn_mst_dists_span | 0.2203515 | 0.1894826 | 0.2563794 | 0.1487511 | 0.3049307 | 701 | 728 |
| Original | spn_mst_dists_skew | 0.1613038 | 0.1314753 | 0.1901262 | 0.07384998 | 0.2425697 | 701 | 728 |
| Original | spn_angles_cos_var | -0.167311 | -0.196472 | -0.1373935 | -0.2267687 | -0.1113695 | 701 | 728 |
| Original | spn_fraction_shorter_mean_distance | -0.1637372 | -0.1985147 | -0.1268801 | -0.2530895 | -0.08793533 | 701 | 728 |
| Original | spn_mst_depth_varcoeff | 0.1076726 | 0.07919514 | 0.1369753 | 0.06481893 | 0.1533193 | 701 | 728 |
| Original | spn_angles_var | -0.1609357 | -0.1921472 | -0.1292472 | -0.2178736 | -0.1059765 | 701 | 728 |
| Original | spn_mst_dists_sd | 0.1365551 | 0.1025994 | 0.1756065 | 0.02935842 | 0.2498897 | 701 | 728 |
| Original | spn_centroid_sd | -0.08665982 | -0.1222087 | -0.0514788 | -0.1924265 | 0.0178801 | 701 | 728 |
| Original | tour::endpoint_neighbor_rank_mean | 0.07704272 | 0.05623097 | 0.09696478 | 0.02076414 | 0.1301476 | 701 | 728 |
| Original | tour::tour_mst_edge_overlap_fraction | -0.002898647 | -0.004367885 | -0.001460318 | -0.00492919 | -0.0008034421 | 701 | 728 |
| Original | tour::endpoint_knn10_fraction | -0.002239869 | -0.003267011 | -0.001234734 | -0.004791839 | 0.000530012 | 701 | 728 |
| Original | tour::tour_edge_cv | 0.02337527 | 0.01301958 | 0.03363173 | -0.005486699 | 0.05285787 | 701 | 728 |
| Original | tour::tour_total_over_pairwise_median | -0.145759 | -0.2422688 | -0.04370159 | -0.4038776 | 0.1161398 | 701 | 728 |
| S1 | true_m4_ce | 0.01004745 | 0.008150836 | 0.01194593 | 0.006818698 | 0.01368299 | 681 | 733 |
| S1 | spn_mst_dists_span | 0.2231392 | 0.1889512 | 0.25517 | 0.1402987 | 0.304106 | 681 | 733 |
| S1 | spn_mst_dists_skew | 0.1556393 | 0.1266777 | 0.1854463 | 0.07877212 | 0.2392943 | 681 | 733 |
| S1 | spn_angles_cos_var | -0.1414433 | -0.1735417 | -0.1112307 | -0.2017587 | -0.08901894 | 681 | 733 |
| S1 | spn_fraction_shorter_mean_distance | -0.1801999 | -0.2158223 | -0.1435818 | -0.2709368 | -0.09662823 | 681 | 733 |
| S1 | spn_mst_depth_varcoeff | 0.1280712 | 0.1004921 | 0.155958 | 0.08407734 | 0.1714149 | 681 | 733 |
| S1 | spn_angles_var | -0.1342405 | -0.1675915 | -0.1030159 | -0.1955041 | -0.07890468 | 681 | 733 |
| S1 | spn_mst_dists_sd | 0.1397936 | 0.1023119 | 0.1780368 | 0.02563906 | 0.253692 | 681 | 733 |
| S1 | spn_centroid_sd | -0.1136284 | -0.14914 | -0.07695512 | -0.2374801 | 0.009723662 | 681 | 733 |
| S1 | tour::endpoint_neighbor_rank_mean | 0.07437704 | 0.05365696 | 0.09369763 | 0.01664652 | 0.1270436 | 681 | 733 |
| S1 | tour::tour_mst_edge_overlap_fraction | -0.0030968 | -0.004393843 | -0.001766646 | -0.005196681 | -0.0009879084 | 681 | 733 |
| S1 | tour::endpoint_knn10_fraction | -0.00179543 | -0.002747824 | -0.000867519 | -0.004443867 | 0.0006667707 | 681 | 733 |
| S1 | tour::tour_edge_cv | 0.02209751 | 0.01226287 | 0.03303936 | -0.007719994 | 0.05330168 | 681 | 733 |
| S1 | tour::tour_total_over_pairwise_median | -0.1709487 | -0.2691571 | -0.07932696 | -0.4465343 | 0.08607207 | 681 | 733 |
| S2 | true_m4_ce | 0.008956222 | 0.007084752 | 0.01093443 | 0.005655492 | 0.01220141 | 701 | 730 |
| S2 | spn_mst_dists_span | 0.2161431 | 0.1846237 | 0.2497397 | 0.1321994 | 0.3033769 | 701 | 730 |
| S2 | spn_mst_dists_skew | 0.1611764 | 0.1310003 | 0.1895998 | 0.08105564 | 0.2440614 | 701 | 730 |
| S2 | spn_angles_cos_var | -0.1321675 | -0.1628382 | -0.09919767 | -0.1926193 | -0.0709767 | 701 | 730 |
| S2 | spn_fraction_shorter_mean_distance | -0.1424742 | -0.1752162 | -0.1049111 | -0.2255839 | -0.05700027 | 701 | 730 |
| S2 | spn_mst_depth_varcoeff | 0.1576732 | 0.1281528 | 0.1834653 | 0.1132403 | 0.2038283 | 701 | 730 |
| S2 | spn_angles_var | -0.123577 | -0.1523261 | -0.09428456 | -0.1852923 | -0.06715109 | 701 | 730 |
| S2 | spn_mst_dists_sd | 0.1196087 | 0.08330371 | 0.1543546 | 0.003733619 | 0.2358244 | 701 | 730 |
| S2 | spn_centroid_sd | -0.1134205 | -0.1510006 | -0.07626595 | -0.2200052 | 0.001612053 | 701 | 730 |
| S2 | tour::endpoint_neighbor_rank_mean | 0.07360462 | 0.05319494 | 0.09461827 | 0.01829258 | 0.1264705 | 701 | 730 |
| S2 | tour::tour_mst_edge_overlap_fraction | -0.00334047 | -0.004716046 | -0.001874108 | -0.005594559 | -0.0009871176 | 701 | 730 |
| S2 | tour::endpoint_knn10_fraction | -0.0020625 | -0.003084375 | -0.001047418 | -0.004659742 | 0.0005206519 | 701 | 730 |
| S2 | tour::tour_edge_cv | 0.02064645 | 0.01005985 | 0.03138347 | -0.01069599 | 0.04979504 | 701 | 730 |
| S2 | tour::tour_total_over_pairwise_median | -0.1310567 | -0.2171675 | -0.02877667 | -0.3938085 | 0.1508276 | 701 | 730 |
