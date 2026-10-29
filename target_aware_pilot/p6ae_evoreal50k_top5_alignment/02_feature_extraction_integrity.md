# Feature Extraction Integrity

| dataset | n | hash_unique | nan_raw_nng | nonfinite_phi | representation_sha256 |
|---|---|---|---|---|---|
| EvoReal50k | 50000 | 50000 | 2779 | 0 | 5bfa6d127a3085e8ecae2a3eb987565c2a896fed3f13f90d8d99d126dfd32066 |
| Uniform_D0_10k | 10000 | 10000 | 0 | 0 | bbdda9d83f03946adfe929ac0182cdf00bd14d691d5f416069f0ca85a204773f |
| P03_Population10k | 10000 | 10000 | 0 | 0 | 9f0e52882d329178f2fe868c914412c1bdd904ac15d12a6bf77eca0c141d4d3e |
| P5_Original_E1_ArchiveStructure10k | 10000 | 10000 | 0 | 0 | 13a805063ed615fb447dcd68bdb8a70da321249ef6757edfb7cf51149021997c |
| P5_Original_E1_G2_Population10k | 10000 | 10000 | 0 | 0 | 0e25b98e3f157368f5894b7ae4c5ca0cc7b3309dcc654bf0592252c6aa98bd0b |
| P5_S1_E1_ArchiveStructure10k | 10000 | 10000 | 0 | 0 | 8005b01bbb1bbde82854dca25347cf20e69aa0b6983a19ea68c1838ee9a8368d |
| P5_S1_E1_G2_Population10k | 10000 | 10000 | 0 | 0 | d9b34dd2a179879bdd95551a31e019806f62263e1d2eb904a6e8d8a5b7a5aeb5 |
| P5_S2_E1_ArchiveStructure10k | 10000 | 10000 | 0 | 0 | d1e7597a0beb322c438eb1406eaf73cc3e3f86d40c19871fb9190b17330f5757 |
| P5_S2_E1_G2_Population10k | 10000 | 10000 | 0 | 0 | 4dffba7cd94ca2265a12a9c3e9982e90e71cf4f42d87b7c92ec02141898f238b |

P03 reuses the formal full-F40 cache after exact inverse scaling; a deterministic 32-instance salesperson re-extraction cross-check passed with max raw absolute error 6.11e-16. Other missing Top5/old2 features were computed by the same `salesperson` definitions in isolated P6-AE shards. Raw values, theoretical scale companions, and frozen transformed values are preserved. Undefined NNG values retain P6-A's missing-singleton semantic atom; transformed vectors contain no NaN/Inf.

Resource accounting: feature prepare wall approximately 266.83s; final exact solver wall summed across 58 comparisons was 397.60s. The profiled matching/reporting run peaked at 1095.2 MiB RSS. GPU used: no. Full profile is in `configs/runtime_resource_profile.txt`; accounting metadata is in `configs/resource_accounting.json`.
