# NN-Ratio Audit

NN-Ratio is population-SD/mean of self-excluded nearest-neighbor distances (`k=2`). KDTree agrees with direct distances to maximum error `0.000e+00`. Spearman NN-CV vs N on TSPLIB48 is 0.1566. Duplicate and zero-mean checks are retained per instance. Threshold-based type labels remain exploratory because the first-stage FFT implementation is approximate.
