# Runtime and Resource Audit

- Initial cold feature-cache construction observed wall time: approximately 67.51 s (the pass then stopped on a held-out path-resolution error, fixed without recomputing features).
- Final cached deterministic validation pass: 10.64 s.
- Peak RSS in final pass: 475.1 MiB.
- GPU/Concorde/HGB/BQ/evolution: none.
- Unique fixed populations are cached by coordinate hash membership; FFT/NN is not recomputed for B/C/D subsets.
