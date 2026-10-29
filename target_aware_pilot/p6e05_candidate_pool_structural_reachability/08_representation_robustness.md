# Representation Robustness

Top5 sensitivity uses a predeclared TSPLIB48-only median leave-one-reference nearest-neighbor radius, frozen before candidate inspection. Its percentages are not numerically interchangeable with FFT–NN. Both representations indicate incomplete pool reachability, but the specific missing matched references differ; conclusion: `REPRESENTATION_DEPENDENT` / MIXED rather than a confirmed mutation failure.

| dataset | replicate | population_n | all48_coverage_count | matched_coverage_count | population_within_radius_fraction | missing_matched_reference_ids |
|---|---|---|---|---|---|---|
| U_Original | Original | 19929 | 25 | 14 | 0.098700 | pr107;pr136;rat99 |
| B_Original | Original | 10000 | 21 | 13 | 0.104800 | pr107;pr136;rat99;kroA150 |
| C_Original | Original | 10000 | 20 | 12 | 0.103500 | pr107;pr136;rat99;kroA100;kroA150 |
| U_S1 | S1 | 19937 | 23 | 14 | 0.101620 | pr107;pr136;rat99 |
| B_S1 | S1 | 10000 | 21 | 12 | 0.102000 | pr107;pr136;pr144;rat99;kroA150 |
| C_S1 | S1 | 10000 | 21 | 13 | 0.101600 | pr107;pr136;rat99;kroA150 |
| U_S2 | S2 | 19927 | 25 | 13 | 0.099312 | pr107;pr136;pr144;rat99 |
| B_S2 | S2 | 10000 | 22 | 13 | 0.102900 | pr107;pr136;pr144;rat99 |
| C_S2 | S2 | 10000 | 21 | 13 | 0.103200 | pr107;pr136;pr144;rat99 |
