# Runtime Audit

- Sum of per-job inference/extraction wall clocks (parallel jobs not de-overlapped): 2950.71s.
- Maximum recorded GPU allocation: 0.334 GiB.
- GPU operations were eval-mode forward inference only. CPU work comprised integrity, F40 extraction, exact frozen Top5 assignments, and statistics.
- No Concorde, training, mutation, evolution, or benchmark generation ran.
