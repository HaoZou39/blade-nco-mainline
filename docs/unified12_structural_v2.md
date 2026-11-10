# Unified-12 Structural V2 — proposed, opt-in

V1 is **not** changed or deprecated automatically. V2 removes overlap between the frozen standalone `grid_mutation`, the lattice Grid option, and the motif `grid_block` option. The V1 `cross` incorrectly used only two positive perpendicular arms, making it an L-shape.

## Changes

- Keep the ten existing V1 base operators unchanged, including `grid_mutation`.
- Replace V1 `lattice_reorganization` with `stripe_reorganization`: stripes only, original local/regional/global scope, matching, interpolation, and validation retained.
- Replace V1 `motif_assembly` with `motif_assembly_v3`: allowed templates `arc`, `cross`, `l_shape`, `star`; Cross is now four rotated rays from one central node.
- Keep exactly twelve operators with uniform 1/12 selection. V2 profile is `unified12_structural_v2.yaml`.

## Usage

```python
import numpy as np
from mutation.registry_v2 import OPERATOR_NAMES, sample_operator_name, apply_operator
rng = np.random.default_rng(7)
parent = rng.random((100, 2))
result = apply_operator("stripe_reorganization", parent, rng)
```

## Test

`python -m pytest -q tests/test_unified12_structural_v2.py tests/test_unified12_library.py`

## Scientific boundaries

- Frozen `UNIFIED12_FROZEN_V1`, the original operator modules, default imports/registry, hashes and saved experiments remain unchanged; V2 requires explicit opt-in.
- This is a proposed version, not a verified training improvement. It changes the generator distribution; D3-CVT-256 Working Map v1 uses a V1-derived reference cloud, so V2 support coverage must be audited before any new long-cycle experiment.
- Do not restart or rewrite the paused 12-cycle training using V2 or interpret previous results as V2 results.
