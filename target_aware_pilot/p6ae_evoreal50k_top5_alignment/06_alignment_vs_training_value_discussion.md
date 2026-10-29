# Alignment versus Historical Training Value

Verified historical artifacts:

- EvoReal50k M5: Original70 `2.285487%`, GAP5 `1.457347%`; source SHA256 `87f57933e41348125153852d3ce882d74b5fb929f87e0b0b761ffdb061b0508d`.
- P3 Structure M5: Original70 `1.917951%`, GAP5 `2.221138%`; source SHA256 `77be33ec28af3663e2e8eafc86c76947edf1c8a403528aff57e8e4ee5bb6bcc8`.

The structural-distance direction versus historical performance is **GAP5 一致、Original70 不一致（mixed）**: EvoReal is closer in frozen Top5 and has the lower GAP5, but P3 E1 has the lower Original70 gap. This remains explanatory only: EvoReal uses 50k over five cycles from a different initialization/adaptation path, while P3 replaces only cycle 5 with one 10k set. Original70 also contains the 48 development instances and is not independent. Lower frozen-Top5 distance therefore cannot be interpreted as causing lower gap.

Largest EvoReal-versus-Uniform marginal separators:

| feature | wasserstein_EvoReal_vs_Uniform | median_shift |
|---|---|---|
| spn_distance_varcoeff | 1.856976 | 2.298098 |
| spn_sum_of_lowest_edge_values | 1.678448 | -2.149684 |
| spn_mst_dists_sd | 1.399266 | 1.793737 |
| spn_distance_min | 0.598340 | -0.448809 |
| spn_nng_3_strong_components_varcoeff | 0.486418 | -0.456647 |

If GAP5 and Top5 disagree, plausible—not proven—explanations include incomplete Top5 sufficiency, TSPLIB48 not representing the whole GAP5 mixture, other generator geometry, and training-path/budget differences. If they agree, the maximum supported claim is distributional consistency with the real-structure motivation.
