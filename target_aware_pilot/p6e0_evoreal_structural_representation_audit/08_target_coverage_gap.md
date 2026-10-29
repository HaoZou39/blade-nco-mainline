# Target Coverage Gap

All distances use TSPLIB48-only median/IQR scaling frozen before TSPLIB22 access. EvoReal full-50k and ten fixed 10k subsamples are both reported. Type labels are exploratory and class-proportion matching is not treated as sufficient evidence.

| dataset | coarse_3x3_js | reference_neighborhood_coverage_fraction | population_within_reference_radius_fraction | effective_reference_neighborhoods | largest_reference_share | S1_fraction | S2_fraction | S3_fraction |
|---|---|---|---|---|---|---|---|---|
| B_Original | 0.370160 | 0.520833 | 0.351500 | 7.571580 | 0.332100 | 0.035800 | 0.063700 | 0.900500 |
| C_Original | 0.356634 | 0.520833 | 0.337800 | 7.551811 | 0.333200 | 0.046200 | 0.064400 | 0.889400 |
| B_S1 | 0.375067 | 0.500000 | 0.356200 | 7.537109 | 0.328500 | 0.033300 | 0.059400 | 0.907300 |
| C_S1 | 0.358397 | 0.500000 | 0.346800 | 7.496804 | 0.332200 | 0.042700 | 0.060800 | 0.896500 |
| B_S2 | 0.371634 | 0.500000 | 0.351000 | 7.390059 | 0.333800 | 0.032700 | 0.061100 | 0.906200 |
| C_S2 | 0.350668 | 0.520833 | 0.344800 | 7.454688 | 0.340300 | 0.044600 | 0.064800 | 0.890600 |
| EvoReal50k_full | 0.111166 | 0.833333 | 0.473440 | 22.966884 | 0.153260 | 0.195420 | 0.280700 | 0.523880 |

EvoReal matched-10k mean JS `0.111729`, reference coverage `0.829167`.
