# Coordinate Normalization

Algorithm 1 shared-max-span normalization is used for every source: subtract per-axis minima, divide both axes by the single maximum axis span. Aspect ratio is preserved. Canonical-float64 original and normalized coordinate SHA256 values, N, source and normalization version are retained in feature artifacts. Degenerate input produces zero coordinates in the historical approximation and would make NN-CV invalid; no audited instance is degenerate. Translation, uniform-scale, permutation and 90-degree-rotation diagnostics are in `manifests/invariance_and_nn_tests.csv`.

| instance_id | test | fft_abs_diff | nn_abs_diff |
|---|---|---|---|
| d198 | translation | 0.000000 | 0.000000 |
| d198 | uniform_scale | 0.000000 | 0.000000 |
| d198 | rotation90 | 1.567902 | 0.000000 |
| d198 | permutation | 0.000000 | 0.000000 |
| d493 | translation | 0.000000 | 0.000000 |
| d493 | uniform_scale | 0.000000 | 0.000000 |
| d493 | rotation90 | 2.065074 | 0.000000 |
| d493 | permutation | 0.000000 | 0.000000 |
| d657 | translation | 0.000000 | 0.000000 |
| d657 | uniform_scale | 0.000000 | 0.000000 |
| d657 | rotation90 | 0.106727 | 0.000000 |
| d657 | permutation | 0.000000 | 0.000000 |
| d1291 | translation | 0.000000 | 0.000000 |
| d1291 | uniform_scale | 0.000000 | 0.000000 |
| d1291 | rotation90 | 1.130455 | 0.000000 |
| d1291 | permutation | 0.000000 | 0.000000 |
| Original_B_0 | kdtree_direct_nn | NA | 0.000000 |
| Original_B_1 | kdtree_direct_nn | NA | 0.000000 |
| Original_B_2 | kdtree_direct_nn | NA | 0.000000 |
| S1_B_0 | kdtree_direct_nn | NA | 0.000000 |
| S1_B_1 | kdtree_direct_nn | NA | 0.000000 |
| S1_B_2 | kdtree_direct_nn | NA | 0.000000 |
| S2_B_0 | kdtree_direct_nn | NA | 0.000000 |
| S2_B_1 | kdtree_direct_nn | NA | 0.000000 |
| S2_B_2 | kdtree_direct_nn | NA | 0.000000 |
