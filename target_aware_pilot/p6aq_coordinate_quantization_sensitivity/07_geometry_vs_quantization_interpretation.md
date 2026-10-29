# Geometry versus Quantization Interpretation

- **Meaningful geometric signal:** raw and dimensionless `distance_min` differences are real properties of the stored point sets. The full exact OT contribution is 26.03%, distinct from the 83.7% all-candidate nearest-reference diagnostic.
- **Coordinate-precision sensitivity:** perturbations are hypothetical around lexical storage q, not recovered measurement errors. Retention ranges from 0.9797–0.9894 at a=0.05 and 0.9735–0.9836 at a=0.50. This quantifies sensitivity but cannot prove that the nominal target is erroneous.
- **CDF-calibration sensitivity:** target coordinates are unchanged in this test. Even/odd half-calibration retention is 0.9656; feature-score distribution details and rank diagnostics are explicit in `cache/cdf_score_stability.csv`.
- **Invariance control:** translation/scale/node permutation are invariant to at most 8.038e-14 in phi for the four continuous metric features. The NNG SCC-CV feature changes for 10/48 references under translation and 7/48 under scale despite normalized-coordinate differences of only 3.331e-16; node permutation is invariant. This is consistent with floating tie resolution in the discrete nearest-neighbor graph and is a real reproducibility caveat, not coordinate quantization evidence.

These mechanisms need not be exclusive. Survivor hashes are a stricter stability test than selected-population distributions: many near-tied candidate substitutions can alter IDs while leaving SWD/energy and cross-target objectives nearly unchanged.
