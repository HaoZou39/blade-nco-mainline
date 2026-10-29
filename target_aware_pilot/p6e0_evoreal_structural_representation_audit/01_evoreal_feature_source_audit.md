# EvoReal Feature Source Audit

- Paper: [https://arxiv.org/abs/2511.10233](https://arxiv.org/abs/2511.10233).
- Official repository: [https://github.com/HenryZhu1029/EvoReal](https://github.com/HenryZhu1029/EvoReal), current and pinned commit `919616337105e894e06463734ec1dd71f6451309`.
- Full 1,494-file official tree was searched at that commit for FFT, histogram, nearest-neighbor, NN-ratio and coefficient-of-variation implementations; none was published. Only the three author validation coordinate files were found.
- Coordinate normalization and NN-CV definitions are explicit in the paper. FFT grid resolution, frequency-region mask, magnitude-vs-power and normalization are not.
- Reused implementation: `scripts/evoreal_structure_alignment_audit_corrected_v2.py`: 32x32 raw-count histogram, mean `|FFT|` over shifted annulus `1 < r <= 4`, DC excluded, population SD for NN-CV.
- Status: `FFT_APPROX`; NN-Ratio definition is faithful but the combined reproduction is `PARTIAL`.
- The implementation was historically calibrated on the same 48 author references. Threshold agreement is therefore descriptive reproduction, not independent validation.
- Frozen project and official repository counts are `{'S1': 19, 'S2': 12, 'S3': 17}`. The protocol text's `{'S1': 27, 'S2': 15, 'S3': 6}` is not present in the official files and is not substituted.
