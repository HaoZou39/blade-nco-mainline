# Frozen artifact integrity

## Result

The parent-child pairing required for DT1-A1 is recoverable and internally complete.

- Entering C05 population: 10,000 unique parent hashes and 1,000 occupied niches.
- Mutation schedule: 110,000 rows, 10,000 parent hashes, 11 operator identities, exactly one row per parent/operator pair.
- Accepted unique coordinates: 109,554 rows. Their hash order exactly matches accepted, valid, nonduplicate manifest rows.
- C05 sample: 3,300 scheduled attempts; 3,281 accepted unique children; 19 failures (`linear_projection=14`, `expansion=4`, `explosion=1`), all marked `excessive_repair`; zero recorded duplicates.
- Every evaluable sampled child joins to a C05 QD row and HGB prediction. All 300 mutation `parent_index` values join exactly to the frozen C04 `P_next.npz` order.
- T05 retention is exact-hash joined: 237 sampled children entered T05.

## M5 identity

`C04/M_next_checkpoint_manifest.json` identifies the entering C05 model as `M5`, path `C04/09_bq_training/M5_E256.tsp.best`, SHA-256 `ef07226bb8c3e8cd3a1404eece474df44d378316fd66509b762eb5f8bc26558c`. This equals the manifest's `sha256` and C05 calibration stage input model hash.

The canonical target implementation is `src/bq/training_hardness.py`: mean expert-next-action negative log probability across the expert-ordered closed-tour suffix, teacher-forced ground-truth prefixes, no greedy rollout, no parameter update.

## Label integrity and selection bias

M5 labels are joined only from C05 `B_current_hardness.parquet` and `selected_exact_hardness.parquet`. Their 412 overlapping hashes agree to maximum absolute difference `1.2331e-6`, within the repository's `1e-4` repeated-forward cross-check tolerance.

Only 86/300 sampled parents and 237/3,281 valid children already have M5 labels. The 237 labeled children are exactly retained T05 members and are therefore selection-biased. They must be reported separately and must not be used alone to estimate operator improvement. Historical M0-M4/M6-M9 labels are classification B only and never substitute for M5.

## Mutation-code provenance caveat

The run-level `07_mutation_manifest.json` records the executed mutation entry-point SHA-256 as `fd8c08b750a67779d2fe6f68d3fb67652f3722d4a0d4e3160bd7e29728e3f7a6`. The current checkout of `src/mutation/__init__.py` hashes to `78ee6dce5e66637021dd8bb61896324df6bfd06f2a7c764a0d85f2e5f51bcc62`, and the current HIGH worker/profile were committed after the frozen run. Consequently the current code must not be used to regenerate C05. This does not break the fixed-record experiment because the original attempt table, child hashes, coordinates, seeds, status, and stage checksums are intact; it does limit exact source-level replay of the historical kernel.

## Required checksums

| Artifact | Rows/bytes | SHA-256 |
|---|---:|---|
| C04 `P_next.npz` | 10,000 coordinate rows / 14,154,825 B | `2064462edf623af67bd40e9cca7e923af72e34ec5b3f556965d0424a2ada46c4` |
| C05 `parent_qd.parquet` | 10,000 / 1,147,044 B | `47818038abc2df48dadde917cc1ad4f85035e6c9c983633fac2add5bc94bd7c4` |
| C05 `mutation_manifest.parquet` | 110,000 / 18,763,295 B | `e21fc9f4e989f14193407b6194b885ab4e5e8ab523ba300bec3550caf8bfd158` |
| C05 `offspring_coordinates.npz` | 109,554 / 52,992,648 B | `4382db31bac27b070fb731138b1653b3b8a36bd765c0a7d8e92f31f775b65ac6` |
| C05 `candidate_predictions.parquet` | 119,547 / 15,078,028 B | `3edf0e81c04323296c1956649889d9d06d6cabd42c33e5e8a0c6c6fbcb288ab8` |
| C05 `B_current_hardness.parquet` | 10,000 / 895,848 B | `5ccba82a51d2d3b81dc08a178a5de77b87b2de60e0e923ce7865cda5ab4cc10c` |
| C05 `selected_exact_hardness.parquet` | 10,000 / 847,484 B | `6283e01abe564dd2fc02153d267d0409dc82e801129adee375851c5e03fec2f1` |
| M5 checkpoint | 12,470,329 B | `ef07226bb8c3e8cd3a1404eece474df44d378316fd66509b762eb5f8bc26558c` |

Machine-readable source paths, sizes, and hashes are in `integrity_summary.json`. No frozen artifact was modified.
