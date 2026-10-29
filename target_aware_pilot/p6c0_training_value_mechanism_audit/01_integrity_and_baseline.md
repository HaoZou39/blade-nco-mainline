# Integrity and Baseline Reproduction

All six Population10k bundles pass coordinate range, coordinate-hash, within-population uniqueness, manifest-order, label-order, and legal Hamiltonian-tour checks. All six M5 hashes match their training integrity records; each record asserts direct E000 tensor equality to the same formal M4. The formal M4 checkpoint hash also passes.

Across-population memberships: 60,000; globally unique coordinate hashes: 59,678.

P6-B Original70 means reproduce the frozen conclusion within 5e-6 percentage points:

| replicate | mean_gap_E0 | mean_gap_E1 | Delta_E0_minus_E1 |
|---|---|---|---|
| Original | 2.1604434 | 2.7614748 | -0.60103137 |
| S1 | 2.3700186 | 2.4097267 | -0.039708008 |
| S2 | 2.2530337 | 2.6080242 | -0.35499044 |
| ALL_EVOLUTION_REPLICATES | 2.2611653 | 2.5930752 | -0.33190994 |

Hardness is the canonical teacher-forced CE, not greedy gap and not HGB prediction.
