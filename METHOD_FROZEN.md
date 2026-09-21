# BLADE-NCO final frozen method

Status: **FROZEN**. Run ID: `blade_final_20cycle_seed2026_v1`.

The final method executes exactly twenty solver updates, C00 through C19, from the byte-frozen Uniform-trained BQ checkpoint M0 to M20. It has no scientific early stop and cannot create M21.

The fixed solver-independent map uses QD4 in the exact order `spn_sum_of_lowest_edge_values`, `spn_distance_varcoeff`, `spn_mst_depth_mean`, `spn_mst_dists_median`; deep-reference P1/P99 normalization clipped to `[0,1]`; and the byte-frozen K1000 CVT centroids. The deep cloud is map geometry only. Normalization, features, centroids, and K are never refit.

C00 begins from the exact canonical 10,000 Uniform `saved_depth == 0` roots. Each of the eleven FOGA operators is applied exactly once to each parent with deterministic hash-derived RNG, giving 110,000 nominal children. The output hashes must match the audited canonical replay. GP0 receives one difficulty-blind centroid-nearest probe from each reached offspring niche. GP0 is GP4 on QD4 with `log1p(L_BQ)`, population 2000, 60 generations, Huber fitness, SPERF selection, seeds 2026–2030, and median ensemble output.

Every active population has exactly 10,000 unique instances. Acquisition first retains one centroid-nearest candidate from every occupied pool niche, then fills the remaining budget by one global descending GP score order with coordinate-hash tie breaks. C00's pool is offspring only. C01–C19 use current offspring plus the current exact sparse elites. Round-robin, per-niche quotas beyond Centroid1, and diversity scalarization are forbidden.

The cumulative formal Oracle archive contains only independently selected formal instances. Audit-only labels confer no acquisition eligibility or preference. If an independently selected hash has a verified development tour, the exact tour may be reused and is marked `FORMAL_SELECTION_OCCURRED=YES`. At every later cycle the entire formal archive is rescored under the current solver. The exact archive retains the current exact-hardest observed instance per niche, at most one per niche; its maximum depth one rule is immutable. GP4, HGB4, and HGB20 train with equal weight on the full refreshed formal archive; only GP4 drives acquisition.

Every M_s→M_{s+1} update uses the selected hard10k and fixed Uniform anchor10k at 1:1 for exactly 256 epochs, learning rate 7.5e-5, fresh Adam, no scheduler, no weight decay, no clipping, no AMP, and no early stopping. E256 is the sole scientific checkpoint.

M0 and every M1–M20 checkpoint receive passive greedy evaluation on Uniform100, GaussianMixture100, TSPLIB100, OriginalTSPLIB70, and LEHD TSP200/500/1000. These results never affect execution.

Every major stage has an atomic hash-bearing `STAGE_DONE.json`. Resume verifies hashes before skipping; mismatches stop the run. Crash checkpoints are recovery artifacts only. Each completed cycle produces its strict verification, ten mandatory report tables, eighteen figures, live trajectory tables, and additive timing records. BQ internal component timers are explicitly non-additive diagnostics. `ACCOUNTED_TIME_FRACTION` is checked against 0.98 without becoming a scientific stop rule.

The canonical entrypoint is `scripts/run_final_20cycle.py`. A first launch omits `--resume`; a later human-authorized recovery includes it. Protocol, source snapshot, environment, configuration hash, asset-manifest hash, exclusive lock, PID, command, and launch timestamp are recorded before scientific execution.
