# Original QD Mechanism Audit

The formal five-cycle A0 implementation uses transformed QD4 (`spn_sum_of_lowest_edge_values`, `spn_distance_varcoeff`, `spn_mst_depth_mean`, `spn_mst_dists_median`), frozen deep-reference P1/P99 clipping, Euclidean nearest assignment to 1,000 frozen CVT centroids, and HGB40 conditioned on the checkpoint entering each cycle. C00–C03 therefore use M0/M1/M2/M3-conditioned HGB models; C04 is the M4-conditioned model reused here. There is no GP/HGB switch in this formal run.

For A0 selection, the candidate pool is the deduplicated parent-first Parent+Child union. One maximum-HGB item per reached niche is retained (stable hash tie-break), then remaining slots are filled by global HGB rank to exactly 10,000. There is no 10-slot-per-niche rule, microbin, lineage cap, round robin, or persistent capacity-one QD archive in this runner. The cumulative labeled archive is used to sample HGB calibration records, not as the active QD survivor archive. `T_selected == P_next`: the exact same 10k is Concorde-labeled, trains BQ, and becomes the next generation's 10k parents. Every parent receives one application of each FOGA-11 operator.

| cycle | candidate_n | reached_niches | qd_elites | global_hgb_fill | selected_n | unique_n | ordered_membership_exact | set_membership_exact | stored_membership_sha256 | replayed_membership_sha256 | P_equals_T | lineage_constraint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C00 | 119781 | 435 | 435 | 9565 | 10000 | 10000 | True | True | 2762e3843315bc11bce8b4bc91e9da965bb55f720ba52468a134db9809f0798f | 2762e3843315bc11bce8b4bc91e9da965bb55f720ba52468a134db9809f0798f | True | False |
| C01 | 119681 | 821 | 821 | 9179 | 10000 | 10000 | True | True | c745d14f72a714fcf7b0b4fd77420041aa12731f551a3a7fdbf25e050cb2c7da | c745d14f72a714fcf7b0b4fd77420041aa12731f551a3a7fdbf25e050cb2c7da | True | False |
| C02 | 119496 | 990 | 990 | 9010 | 10000 | 10000 | True | True | 9e7a9ecc301ceb2bcd8f0438f1f4fda0e4d2b18aaec97a0034c639fdd7e01583 | 9e7a9ecc301ceb2bcd8f0438f1f4fda0e4d2b18aaec97a0034c639fdd7e01583 | True | False |
| C03 | 119549 | 1000 | 1000 | 9000 | 10000 | 10000 | True | True | 558ea278823a39275704d71e8e999efe58a56217fb23c7555cabe035c986d190 | 558ea278823a39275704d71e8e999efe58a56217fb23c7555cabe035c986d190 | True | False |
