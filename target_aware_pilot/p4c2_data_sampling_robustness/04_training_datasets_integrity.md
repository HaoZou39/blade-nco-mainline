# Training Dataset Integrity

- Preregistered protocol SHA256 `c18d6cccbe72e913d96db607dd644406e1ceb9a0914a6222e90cf317cd92024e` was present before S1/S2 generation.
- Frozen source pool: 39,848 unique TSP100, membership `2565bd7088d6bdc7b6659add93c5dbabfe848725a99f175328fecda3586d6d0d`; no mutation was run.
- Four new datasets are each 10,000 unique, finite, [0,1]-bounded TSP100 with 100 distinct nodes and reproduced coordinate hashes.
- Every new dataset has zero overlap with C00–C03 M4 history and zero direct OriginalTSPLIB70 coordinate overlap. C04 overlap is reported, not silently filtered, because the frozen requirement only excludes M4 history.
- R0 archive is identical in Original/S1/S2 because retention is deterministic; R0 final acquisition and both R2 stages change exactly through preregistered uniform seeds.

| Replicate | Arm | Archive overlap R0/R2 | Training overlap R0/R2 | Strict target | 3x3 JS | SWD | Eff. lineages | Max lineage |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Original | R0 | 22,552 | 2,635 | 10.10% | 0.447059 | 0.536529 | 613.3 | 0.81% |
| Original | R2 | 22,552 | 2,635 | 7.89% | 0.488460 | 0.617114 | 502.8 | 1.39% |
| S1 | R0 | 22,630 | 2,598 | 10.09% | 0.452103 | 0.539692 | 622.5 | 0.91% |
| S1 | R2 | 22,630 | 2,598 | 8.03% | 0.485906 | 0.608404 | 495.4 | 1.42% |
| S2 | R0 | 22,595 | 2,572 | 10.16% | 0.450290 | 0.536217 | 615.3 | 0.87% |
| S2 | R2 | 22,595 | 2,572 | 7.56% | 0.485803 | 0.615189 | 498.4 | 1.37% |

Diagnostics were computed only after membership was frozen. They did not control resampling.

Status: **PASS**.
