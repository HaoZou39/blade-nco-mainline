# TV03 GitHub sync report

Status: `TV03_CODE_SYNC_PARTIAL`

This sync only published existing code and the frozen D3-CVT-256 map package. It did not start TV-03, training, Concorde, or TSPLIB evaluation, and it did not change the scientific method.

## 1. Repository

- GitHub: https://github.com/HaoZou39/blade-nco-mainline
- Remote: `origin` fetch and push both point at that URL. The remote was not modified.
- Branch: `sync/tv03-acquisition-code-20261113`
- Base: `main` at `bf3304b0` (`Refactor default Unified-12 structural mutations: stripes and distinct motifs`)
- Code commit: `2be4915f59b04ddb86e761963aa20dc6b7a579c8`
- Message: `Sync BLADE-NCO TV03 acquisition and evolution code`
- This inventory is the next commit on the same branch. Use `git rev-parse HEAD` after pull for the branch tip.
- `main` was not updated. No force push. No history rewrite.

## 2. Files in this commit

| Path | Class | Role |
| --- | --- | --- |
| `src/final_run/science.py` | CODE | Existing evaluator correction: fixed-n100 optima follow manifest row order; OriginalTSPLIB70 model input divides by the longer axis. |
| `scripts/run_blade_d3_cvt256_q4_12cycle_8gen.py` | CODE | D3 12-cycle runner: Unified-12 offspring, Q4 parents, global top-10k true-CE training. |
| `scripts/d3_cvt256_q4_policy.py` | CODE | Q4 niche-elite parent selection plus global CE fill. |
| `scripts/test_d3_cvt256_q4_selection.py` | CODE | Selection unit test. |
| `scripts/qd_map_tsp100.py` | CODE | Entry point for the frozen D3 map. |
| `configs/blade_d3_cvt256_q4_12cycle_8gen.yaml` | CONFIG | Frozen D3 run numbers. |
| `assets/qd_maps/d3-cvt256-v1/` | CONFIG / SMALL_ASSET | Frozen map: manifest, scaler, schema, R extractor, mapper, checksums, docs, regression fixtures, `centroids.npy`. |

`centroids.npy` is 6272 bytes. `tests/golden_inputs.npy` is 4928 bytes. `tests/golden_single.npy` is 1728 bytes. They are covered by the blanket `*.npy` ignore rule, and were added with an explicit path because the mapper refuses to load without the centroid file. No other ignored `npy` was added.

## 3. Unified-12 code

Already on `main` before this branch, so a pull of this branch includes it:

- Operators and sampling: `src/mutation/`, profile `src/mutation/unified12_frozen.yaml`
- Profile SHA256: `19e9548f417e46e0c1f17f7053f887645cc4e72e1778bca5e783529667a02f9e`
- Twelve operators, uniform 1/12, one mutation per offspring: uniform, normal, rotation, implosion, explosion, cluster, expansion, compression, axis projection, grid, lattice reorganization, motif assembly
- Parent selection for the D3 run is the new `scripts/d3_cvt256_q4_policy.py`
- Candidate generation is the new runner above

There is no unique approved Unified-12 candidate pool whose true CE was computed by formal M3. Syncing the generator does not sync a pool.

## 4. D3-CVT-256

The frozen working map v1 is now on this branch: centroids, scaler, schema, R extractor, mapper, and checksums. Feature order is `spn_mst_dists_varcoeff`, `spn_distance_median`, `spn_angles_mean`. Scaling is `asinh((x-median)/IQR)`. K = 256, KMeans `n_init=1`, `random_state=20261108`.

The D3 experiment archive is not in Git. It remains at `outputs/blade_d3_cvt256_q4_12cycle_8gen/` (about 15 GB). Cycle 0 is about 1.1 GB of parquet, npy, and json. Its true-CE column was computed with canonical M0, not formal M3.

## 5. True CE and M3

Teacher-forced true CE is `teacher_forced_training_hardness` in `src/bq/training_hardness.py`. It was already on `main`.

Formal M3 SHA256 requested for TV-03:

`3288582ffbc8acdbcd0589984ae6d34d03f527093c257849049bb43206d02ab6`

That hash is not present in this working tree. The only local `M03.pt` is the D3 run checkpoint after cycle 2:

- Path: `outputs/blade_d3_cvt256_q4_12cycle_8gen/checkpoints/M03.pt`
- SHA256: `d95b797b539dbe7c6302a6d6ce246444b8664a4677c7b866aaa3c59488911ba4`

That file is not formal M3. Its CE must not be labeled as M3 CE. The shipped solver weight is M0, `assets/solver/M0/tsp.best`, SHA256 `6826d7a71b30d882e4673a26ced162b7391157795931efe230f4d019292bba15`. The Unified-12 pool below records that same M0 hash as its checkpoint.

## 6. Acquisition

Already on `main`:

- `src/qd/acquisition.py`
- Random and global top-CE selection used by the D3 runner (`acquire_top10k`: sort true CE descending, hash ascending, take 10000)
- Q4 / CVT parent selection, added in this commit
- Structural features: `assets/qd_maps/d3-cvt256-v1/mapping/` and the earlier feature code under `src/`
- TSPLIB integer distance: `src/bq/tsp_metric.py` (`integer_tour_cost`) plus the OriginalTSPLIB70 edge sum in `src/final_run/science.py`
- Target-aware acquisition code and tables: `target_aware_pilot/`, already committed on `main` (`ae7b8d64`)

No new acquisition algorithm was written. If TV-03 expects a selector that is not one of the above, that selector is still absent.

## 7. Candidate pools that were not uploaded

| Asset | Path | Size | Format | SHA256 | CE checkpoint |
| --- | --- | --- | --- | --- | --- |
| Unified-12 pool used by the stage-2C pilot | `outputs/blade_unified12_qd_redesign/stage2c_qd_hardness_elite_pilot/candidate_cache/pool.npz` | 10540251 bytes | npz | `079e64892303b106843c67149d88ac0ad84fc3b363289248fbf5092c9b8c3c57` | M0 `6826d7a7...`, not formal M3 |
| D3 cycle-0 offspring archive | `outputs/blade_d3_cvt256_q4_12cycle_8gen/cycles/c00/` | about 1.1 GB | parquet + npy + json | not hashed as one blob | M0 true CE |
| D3 full run tree | `outputs/blade_d3_cvt256_q4_12cycle_8gen/` | about 15 GB | mixed | not hashed | later cycles are not M0 and are not formal M3 |

`*.npy`, `*.npz`, `*.parquet`, and `*.pt` stay gitignored. These pools were not committed and Git LFS was not added.

## 8. Left unstaged on purpose

- `third_party/concorde/concorde` (3.7 MB executable, local modification)
- `outputs/`, `analysis_outputs/`, `experiments/`
- `scripts/run_from_scratch_5cycle.py` (a separate smaller from-scratch runner, not the D3 protocol)
- `third_party/bq_nco_official/`

## 9. What the other machine still needs

Git pull of this branch restores the evolution code, the D3 map, the Q4 rule, the D3 yaml, and the evaluator correction. It does not restore a formal-M3 candidate pool, because that pool is not on this machine.

Still required before TV-03 can pass the stated preflight:

1. A checkpoint whose file SHA256 is exactly `3288582ffbc8acdbcd0589984ae6d34d03f527093c257849049bb43206d02ab6`. It is not in this repo.
2. One approved Unified-12 candidate pool whose true CE was computed by that checkpoint. The local pools are M0-CE.
3. If TV-03 must reuse the D3 offspring bank, copy `outputs/blade_d3_cvt256_q4_12cycle_8gen/` by a channel other than Git. Do not treat its CE columns as M3 CE.

Suggested transfer, pending confirmation: copy the formal M3 weight and the approved pool with `rsync` or a release artifact, then verify the file SHA256 before any training. Do not put those files into normal Git history.

## Answers

A. The code that was only on this machine and that TV-03 needs to locate the D3 map, Q4 selection, and the corrected TSPLIB evaluator is on GitHub branch `sync/tv03-acquisition-code-20261113`. Unified-12 operators, true-CE code, and the existing target-aware pilot were already on `main`.

B. `git pull` of that branch restores the code and the frozen configs, including the small centroid file. It does not restore candidate pools, archives, or the formal M3 weight.

C. Non-code assets still to transfer: the formal M3 checkpoint named above, and a candidate pool whose true CE was produced by that checkpoint. The local Unified-12 `pool.npz` and the D3 cycle archives are the wrong CE source for that requirement.
