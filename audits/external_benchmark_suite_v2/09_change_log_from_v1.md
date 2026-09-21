# Change log from v1

- Removed LEHD_TSP200, LEHD_TSP500, and LEHD_TSP1000 from the active/final suite and all v2 aggregates.
- Preserved historical LEHD raw artifacts with `EXCLUDED_FROM_FINAL_BENCHMARK_SUITE` marker.
- Added pinned official EvoReal S1/S2/S3 at N=100 and GaussianMixture500.
- Corrected GaussianMixture100 Oracle binding to stable IDs/hashes.
- Reconciled OriginalTSPLIB70 model input to aspect-preserving normalization while retaining original-geometry EUC_2D cost.
- This independent v2 pipeline did not modify the formal run or its frozen configuration.
