# P6-AE Protocol

- Scope: offline distribution audit only; no evolution, Concorde, BQ training/evaluation, HGB retraining, or P6-B.
- Frozen P6-A configuration SHA256: `77e2f8374f2547760f165cdc952436c55fc0e0eb498b3e7752db1a66e1394666`.
- Frozen representation: independent Uniform conditional empirical-CDF score, continuous interpolation, IQR tail continuation, and asinh compression.
- Distance: unweighted squared Euclidean in ordered Top5; target is the equal-mass empirical TSPLIB48 distribution.
- Primary K: 10,000. EvoReal uses all 20 predeclared uniform no-replacement subsamples; fixed 10k sets use every instance.
- Matching: exact SciPy HiGHS dual-simplex full transport; all 10k points assigned, with frozen 32x208 + 16x209 capacities.
- TSPLIB22 was not read or used. EvoReal was not filtered or selected by features.
- P5 seed handling: historical Original plus S1 and S2 are all retained as separate fixed datasets; no favorable replicate was selected.
