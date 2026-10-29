# P4-B1 Protocol Integrity

- Common candidate source: frozen P2.2 E0 G2 bounded archive, 30,000 unique eligible TSP100, bundle SHA256 `fd9bd10e5bd79f20054ea3b78dad62b541c0cae6226a661ffcea2b466c8bcb0a`; coordinate IDs, validity, lineage, and manifest order all reproduce.
- No C00–C03 training instance and no direct OriginalTSPLIB70 coordinate hash occurs in the archive or Random10k.
- Important scope limitation: archive retention itself was target-aware. The 39,848 eligible union was reduced to 30,000 using common structured target/frontier quotas and lineage diversity (`common_bounded_archive_retention`). P4-B1 isolates only final acquisition conditional on this archive.
- R0: canonical-hash ordered archive followed by uniform sampling without replacement with NumPy seed 20261025; every archive member has probability 1/3. No feature, hardness, HGB, lineage, density, quota, or provenance value is read by sampling.
- R1: frozen P4-A E0 Structure10k/checkpoint `d7d2ffc9ebafa5adb2bca9ccd90f6e8bd0d0487bdf1b4f68cf98dbf425cd4c58`; it is not regenerated or retrained.
- Both models start from formal M4 `c749311bd2a461084d0cbcfd383a4c0a1fc1268762b72e5f7220952f108f8190` and use the identical P3/P4-A 256-epoch, 10,240-update training/evaluation protocol.
- Predeclared conclusion rule before R0 labels/training: SUPPORTED requires R1−R0 mean gap < 0 on at least 6/8 benchmarks, paired CI upper < 0 on at least 3/8, and no significant R1 regression. NOT_SUPPORTED is the exact mirror; otherwise INCONCLUSIVE.
- TSPLIB22 is now a repeatedly observed comparison set, never a selection/tuning set or pristine final confirmation set.

Status: **PASS**.
