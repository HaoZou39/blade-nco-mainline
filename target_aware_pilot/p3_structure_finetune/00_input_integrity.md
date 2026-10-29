# P3 Input Integrity

- Structure10k: 10,000 unique TSP100; bundle SHA256 `5287932231529e617383388bb64d607985d07944799c3d661c5a737338bdece1`; membership SHA256 `10382659919fbc7f0d4baf8ab681024b7424ff92ae62d3e0fd76623ab07c8a99`.
- Coordinates are finite, in `[0,1]`, contain 100 unique nodes per instance, and reproduce every manifest ID.
- C00-C03 known M4-training overlap: **0**. Original C04 Hard10k overlap: **0**.
- Formal M4: `runs/BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1/A0_FULL_BASE/checkpoints/M4_E256.tsp.best`, SHA256 `c749311bd2a461084d0cbcfd383a4c0a1fc1268762b72e5f7220952f108f8190`.
- Original Hardness M5: `runs/BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1/A0_FULL_BASE/checkpoints/M5_E256.tsp.best`, SHA256 `41d189dfeb986a070bd7685f203172d1dad4376d8d0134151d7488fb079a4545`.
- Original C04 stage records M4 SHA `c749311bd2a461084d0cbcfd383a4c0a1fc1268762b72e5f7220952f108f8190` and its E000 state tensors exactly equal formal M4.
- M4 checkpoint contains model weights/epoch/metadata and no optimizer state. Original C04 creates a fresh Adam and has no scheduler; P3 will do the same through the original `train` function.
- Canonical Concorde SHA256 `38a647d3f04196231c725e90d23c2c883c0a23899f9bab8a257ae6f777d1dec0`. Shared-cache overlap is 0; no replacement or alternate instance source is permitted.
- Held-out TSPLIB22 coordinates/features are not loaded by input auditing, labeling, or training.

Status: **PASS**.
