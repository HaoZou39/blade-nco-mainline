# Runtime and Resource Audit

- Exact full-bank selections completed: 18; fixed-membership exact evaluations: 34.
- Total recorded matching/evaluation wall: 1563.5s.
- Maximum sampled RSS: 7240.9 MiB.
- All full selections use 119,530 candidates and 48 references; no approximation or sample-size reduction was used.
- Python 3.10.12, NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3; GPU unused.
- Per-operation status, objective, wall time, and memory are in `logs/runtime_events.csv`; every saved selection has an integrity JSON and complete hash list.
