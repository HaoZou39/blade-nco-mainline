# P6-G3 Intervention Protocol

The protocol was frozen before inspecting intervention outcomes.

- Arm A: unchanged formal P03, 10,000 parents.
- Arm B: a fixed random 9,000-member P03 subset plus 1,000 independently generated TSP100 parents (exactly 200 from each of five generic generator families).
- Removal seed `20261031`; supplement seed `20261030`. No hardness or reference similarity enters removal or supplement acceptance.
- Supplement families: parallel lines, jittered global grids, separated cluster blocks, repeated local motifs, and anisotropic multiscale layouts. Parameters, orientation, scale, and translation are sampled from the frozen ranges in `configs/frozen_intervention_config.json`.
- No EvoReal instance or generator code is used. No TSPLIB coordinate is copied, cropped, resampled, or used to accept/reject a parent. TSPLIB48 enters only after generation as the frozen evaluation reference; TSPLIB22 is never loaded.
- Both arms have 10,000 parents and 110,000 conceptual FOGA-11 attempts per seed. For the 9,000 common parents, hash-derived RNG makes offspring exactly paired. Historical Arm-A artifacts are accepted only after parent/order/profile/namespace checks and deterministic replay.
- RAW analysis includes only valid, accepted, non-identity offspring; coordinate bundles are unique within generation. No survivor selection and no G2 are run.
- Frozen FFT–NN radius, P6-G2 F40 robust standardization, and P6-G2 independent geometry diagnostics are reused without retuning.

Protocol SHA256: `6297a8b80ef95fdcca0a67f252d5f107a4f7938672432b7510d780ccc66011d4`.
