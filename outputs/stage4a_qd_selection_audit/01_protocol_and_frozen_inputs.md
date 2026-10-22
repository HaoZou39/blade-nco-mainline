# Stage 4A protocol and frozen inputs

## Scope and stopping rule

This is a read-only artifact and cost audit of the frozen S6-AE C00–C09 run. It did not invoke LBQ/BQ inference, Concorde, HGB fitting, BQ training, FOGA mutation, evolution, or algorithm code. The exact true-hardness counterfactual was not calculated because required current-checkpoint labels are missing.

The entering checkpoint for cycle `Ckk` is `Mkk`. The true score is

```text
f_k(x) = log(1 + L_BQ(M_k, x)).
```

The surrogate score is the frozen median HGB40 prediction on that same log1p target scale. These two quantities are never substituted for one another.

## Frozen selector replay

For each cycle, the candidate table and prediction table are aligned in frozen candidate order. The strict total order is HGB score descending and instance hash ascending. `G` is the first 10,000 candidates. `Q` is the first candidate under score/hash order within each occupied frozen-QD4 niche, emitted in niche-id order. `T` is `Q` followed by the first `10,000-|Q|` candidates in global order outside `Q`.

All ten cycles pass exact ordered replay of `Q`, `G`, and `T`, coordinate/hash alignment, stored membership hashes, and the identity `T\G = Q\G`, with equal directed differences no larger than 1,000. No prediction was recomputed.

The QD map is the fixed 1,000-centroid partition in `assets/qd/cvt/centroids.csv`; QD4 values use the frozen P1/P99 clipped normalization in `assets/qd/normalization/normalization_A_P1P99.json`. `N_eff` in the cyclewise output is `exp(Shannon entropy)`, matching the implementation. `qd4_rms_radius` is the square root of the sum of the four population variances on the frozen normalized descriptors.

## Label provenance rule

A current label is accepted only when the same instance hash is present in either the cycle's selected exact-hardness table or its archive-calibration current-hardness table, both produced with the entering checkpoint recorded by the calibration stage. Selected `current_hardness` is raw LBQ and is transformed with `log1p`; calibration `y_log1p` is already on the required scale. Labels found under another `M_j` are listed separately and never reused as `f_k`.

An optimal tour is checkpoint-independent. Tour availability is accepted only from a solved, verified ledger/archive row with a non-null tour and tour hash. Duplicate sources for every exchange member are checked for consistent optimal cost; tour hashes are reported separately because rotation, reversal, or distinct equal-cost optima can produce different valid encodings. The source catalog below includes the implementation-declared base/historical caches, frozen S6-AE ledgers/archives, and matched S6-AC/S6-AD ledgers present locally.

## Input manifest

Every file actually read by this audit is identified below by repository-relative path and SHA256. Checkpoint rows use their content hash, not merely a model name.

| scope | cycle | role | path | sha256 | bytes |
| --- | --- | --- | --- | --- | --- |
| global |  | 00_run_config.json | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/00_run_config.json | b204a8ab85641abfb39083b101477806f5e4be26ce63c917361340449b064ed0 | 4080 |
| global |  | 01_protocol.md | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/01_protocol.md | bb033f5fd5adf287e2f04f34a27d771104d2e8bfeab0eb657849569c91d80b2d | 403 |
| global |  | M0_M10_checkpoint_hashes.csv | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/M0_M10_checkpoint_hashes.csv | 7e5ab5b70967bbf0e65b83e10f2be53b71f36c418d62b2b375a10cd4435ef29b | 2207 |
| global |  | centroids.csv | assets/qd/cvt/centroids.csv | 0b1cccaded7c81f8e4466b05cb31b81c82e27d8a113d0cc7569a4b9bf6cef9fd | 82344 |
| global |  | normalization_A_P1P99.json | assets/qd/normalization/normalization_A_P1P99.json | 874027c8849c4a370eaa949b3a3ebdf15546770b23bd0bf73dc5661b3f323707 | 898 |
| global |  | 09_feature_subset_manifests.json | audits/S3_B2_RF_HGB_feature_label_scaling_v1/09_feature_subset_manifests.json | adecfaab6efc7629d511b9eed7526a0b7ca1c92f90daf5beb0915b2ba3567b6f | 5059 |
| global |  | 10_feature_subset_hashes.json | audits/S3_B2_RF_HGB_feature_label_scaling_v1/10_feature_subset_hashes.json | 55ba4f15ae5a8b94fb69a5972fc997744c7d738697ee5d4e597203de5bccf20a | 316 |
| global |  | run_s6ae_hgb40_eliteqd1000_pt_10k_hard_only_x1_m0_m10.py | scripts/run_s6ae_hgb40_eliteqd1000_pt_10k_hard_only_x1_m0_m10.py | 977979ccfd29ea51f1aec01d2f64cc8c7cdef8f296f4202fabdb3d58c251390f | 41751 |
| global |  | run_s6ac_hgb40_eliteqd1000_pt_10k_hard_only.py | scripts/run_s6ac_hgb40_eliteqd1000_pt_10k_hard_only.py | 19d04c3176733202051aea9f51328c753cd07bd9442d5a9d978ab36ee84c4304 | 54422 |
| global |  | run_s5b_qd_aware_training.py | scripts/run_s5b_qd_aware_training.py | a6a8ea46c8a475df4f3434bb7284a799b9b450fb5d7117f6e141dcb8850c3d3b | 28681 |
| global |  | run_s5a_quality_aware_qd_parent.py | scripts/run_s5a_quality_aware_qd_parent.py | 8fbdbe223c83592d3e6c5647308facabcd4b8986e3b32df845de36d52e70df8e | 25686 |
| global |  | run_s5a_quality_aware_qd_parent_v2.py | scripts/run_s5a_quality_aware_qd_parent_v2.py | 47d7fd9f4bc5182d907a578e9af8cdd6e0f4e546afed32cb9adeb93995489356 | 6034 |
| global |  | science.py | src/final_run/science.py | a3a03fa330417805dee5f1ffe68aa15ad4c9320e5915e556c471fbd3413d235e | 23583 |
| cycle | C00 | candidate_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C00/candidate_manifest.csv | 0f131437885c5cb568e38995b528d4f2071e29322c2e264f16dd3f0fd737a1f7 | 44028636 |
| cycle | C00 | candidate_predictions | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C00/candidate_predictions.parquet | a153db4ef34fc804a31a8382e72d2ee65e880b18419b6d366c088d4826c0b2d3 | 15104884 |
| cycle | C00 | qd_assignment | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C00/QD_assignment.csv | e561f86820136abac756ce7f28dc909d8891cec8712ae3d399ecc0156199f620 | 26147207 |
| cycle | C00 | q_elites | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C00/QD_quality_elites.csv | 6d8a2da06568af3d95103c1d14d9f8295c8b172c141c17d4cbdc6c6949157c7a | 67826 |
| cycle | C00 | actual_T | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C00/T_hard_top10k_membership.csv | 1a87216c401bed314a18c501bd627472aa3cfc53ec15d8a3642c7a95db9882d7 | 1057104 |
| cycle | C00 | counterfactual_G | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C00/G_pure_hgb_top10k_membership.csv | 4d3cd62efd43d3d7bd2abc2e915abc9bf93e3af50fc38cd40c2f650b5403dc3a | 1256083 |
| cycle | C00 | selected_current_hardness | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C00/selected_exact_hardness.parquet | 0ad9de2b28120b75ce0388bf75a823bf76798139618c615d37163aa7b7e118eb | 848631 |
| cycle | C00 | calibration_current_hardness | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C00/B_current_hardness.parquet | fc2675e19137633871b1e2e68b23d76a3c2657189891eebfab6e6820db246d98 | 47238 |
| cycle | C00 | candidate_coordinates | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C00/05_scoring/union/C_union.npz | 2e74619fc22be7191dd14eb44ce8ecde27436daa2be0dc525af07b17a75e915e | 68068525 |
| cycle | C00 | candidate_union_parquet | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C00/05_scoring/union/C_union.parquet | 6e00a7ef3ee7ee440fb98b757c845e260806b1c4c7ca1a95a48e81b49be30060 | 18061765 |
| cycle | C00 | archive_with_tours | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C00/01_inputs/archive_in.parquet | 66d61eafa005c0fdcef1310c48c50ae4bd334714ff8598532b0308da656ef0e6 | 266899 |
| cycle | C00 | selection_info | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C00/_selection_info.json | 1ac9b9aba8bf79cb64052c797c04259e5228842d92b51bb073ce3941a9a5398b | 1527 |
| cycle | C00 | selection_stage_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C00/_stages/05_selection/STAGE_DONE.json | 06df15fdfb9932904b2e42396086a1dc943a5fab189694642617bfadd6732ff7 | 6042 |
| cycle | C00 | calibration_stage_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C00/_stages/01_calibration/STAGE_DONE.json | e7e7d73d1cb739d002592f4bdb5eb98b9e21fe902ba7ed921ab355091aba4b16 | 2116 |
| cycle | C00 | hgb_model_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C00/HGB_model_manifest.json | be49cd6315faee90a05aa83dffce4ba38172ddaeb4ba081aaa26824dad2298ad | 4088 |
| checkpoint | C00 | entering_M0 | assets/solver/M0/tsp.best | 6826d7a71b30d882e4673a26ced162b7391157795931efe230f4d019292bba15 | 12467743 |
| cycle | C01 | candidate_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C01/candidate_manifest.csv | ff94eaa77528354583a8e7e8bcb5078afa3dc1a8af4f4663feaaf2cdc432d5be | 41856838 |
| cycle | C01 | candidate_predictions | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C01/candidate_predictions.parquet | ed1b05dc9c2bd7ca6ca9df5faafc48346a73219f563cec7c66dbcebda6eab95c | 15111518 |
| cycle | C01 | qd_assignment | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C01/QD_assignment.csv | 21d93c8666bb63e74ad3d017f7978554ede6847685150b9540cf668c45074bb9 | 26802271 |
| cycle | C01 | q_elites | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C01/QD_quality_elites.csv | 6170f26330f70169cf6bf6f1174a13a02dc5eb90e0ce1a44a0c8bd53116f8a98 | 130382 |
| cycle | C01 | actual_T | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C01/T_hard_top10k_membership.csv | 3210d1598a97265c8171679680e4406cdb7595b61ffb1dbccbaa64cda7ab3d38 | 1058912 |
| cycle | C01 | counterfactual_G | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C01/G_pure_hgb_top10k_membership.csv | 598059f993907d8350e289c3ab8510b207eef48e3c3583e89a79884eb57a736a | 1257055 |
| cycle | C01 | selected_current_hardness | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C01/selected_exact_hardness.parquet | f45aeec6e29fdcfbaf667fe583a88f5dda086a7e5392bbef6d466be637929bb1 | 839310 |
| cycle | C01 | calibration_current_hardness | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C01/B_current_hardness.parquet | e496f9f49cea7ba6255596ffb5e788e0b16682f7432922753d16498b1c938f9d | 893636 |
| cycle | C01 | candidate_coordinates | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C01/05_scoring/union/C_union.npz | 11090b0bbc4fe17df162107f6d74325226109946d23e6b4c02d832fb4147d1b0 | 68103416 |
| cycle | C01 | candidate_union_parquet | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C01/05_scoring/union/C_union.parquet | e3806bb8d1324d308da51305a4468846edf9874a2dfac829a063b49a833ff751 | 17086872 |
| cycle | C01 | archive_with_tours | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C01/01_inputs/archive_in.parquet | ae823e23c4d8a2e139792773f6496a70c27c4d6117b5d2ffbb6d9d8916db2df3 | 5113256 |
| cycle | C01 | selection_info | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C01/_selection_info.json | 79cb9af83a5a8fe9c26981cf84cfb5981c499ac18083817914049f1dc6aba6ec | 1532 |
| cycle | C01 | selection_stage_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C01/_stages/05_selection/STAGE_DONE.json | c2872949ad6ac004824b19cfdf5e1b5dc684154085ae82088346811568395104 | 6044 |
| cycle | C01 | calibration_stage_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C01/_stages/01_calibration/STAGE_DONE.json | 0fd2bf44ca19745cdefdd9965ce10afcd1bcf984e39f14449aa8866aaf46f86c | 2119 |
| cycle | C01 | hgb_model_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C01/HGB_model_manifest.json | cd221c9893065706dc05201cbcc1b3d1c1ad4c3d8a5434e5feab6926fa75ac82 | 4090 |
| checkpoint | C01 | entering_M1 | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C00/09_bq_training/M1_E256.tsp.best | e928d05017a3559cdc39a741d605d24357564a2787d070b0b9c9a469b9f177ed | 12470329 |
| cycle | C02 | candidate_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C02/candidate_manifest.csv | 9a5c006de520311615d27683f8efe4cc1e5d14eb4a523fbd5dc04cfca7940e35 | 41791772 |
| cycle | C02 | candidate_predictions | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C02/candidate_predictions.parquet | 488e249fd22b47fa62288368630d11920766c4cf4dc1e751d148561553e8e7e0 | 15092194 |
| cycle | C02 | qd_assignment | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C02/QD_assignment.csv | e8a7f88111b9b47abbd836d0c34b180da980d95ca16aa31ff501079ab93d98ad | 26763486 |
| cycle | C02 | q_elites | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C02/QD_quality_elites.csv | 29029c5f34fa51f44d56b200f6d2be09b4336e1e3b6d886c2d065e51505300e0 | 155560 |
| cycle | C02 | actual_T | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C02/T_hard_top10k_membership.csv | c8de896c78dee5bfcff49f6dbb169be46f7e4b03c970f1a059132853646dab46 | 1059098 |
| cycle | C02 | counterfactual_G | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C02/G_pure_hgb_top10k_membership.csv | b81b0a9f4c2be1f45050938308c3b9bcac4654e0256b0ab6d7e7b58d07f49bf9 | 1256900 |
| cycle | C02 | selected_current_hardness | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C02/selected_exact_hardness.parquet | 6078f9150fe13408e7d35f409f5d815b44cf8810fdc223166b489f594c95431f | 840987 |
| cycle | C02 | calibration_current_hardness | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C02/B_current_hardness.parquet | da18fa4de18283fa868455d1a7fa3f36183ef861b0b47232820049997b607a80 | 905957 |
| cycle | C02 | candidate_coordinates | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C02/05_scoring/union/C_union.npz | 36257ca89ecb84649b5907d88f57369267166e0756542d5f7b28f521165de06c | 67781623 |
| cycle | C02 | candidate_union_parquet | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C02/05_scoring/union/C_union.parquet | 9d9345697e9e16fa2f975c4d26d75ad24994b445dcb31a253c11454716704791 | 16466678 |
| cycle | C02 | archive_with_tours | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C02/01_inputs/archive_in.parquet | 5ea8c943ae266e4734c525d302614c06d718e8f485f548b29ec6465f26333dbf | 9108998 |
| cycle | C02 | selection_info | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C02/_selection_info.json | fc1345fee669f42d8490a9941b50a7582c41bcd71ce42bc57939975227fa842e | 1549 |
| cycle | C02 | selection_stage_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C02/_stages/05_selection/STAGE_DONE.json | 34a5a115092929cc0ff141a52feaaf7e38b56306b97b2a55ab1806645d2f0407 | 6044 |
| cycle | C02 | calibration_stage_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C02/_stages/01_calibration/STAGE_DONE.json | eae4577102e776730157059451a27d1f26868b3f0be048be8421fc7c2271aab1 | 2119 |
| cycle | C02 | hgb_model_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C02/HGB_model_manifest.json | b642fd0ab8c0f5c25f4495bddc172fce7b9afa131e22d66054fdde0c6b83bf74 | 4086 |
| checkpoint | C02 | entering_M2 | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C01/09_bq_training/M2_E256.tsp.best | 40a5222ba112c1d7f59b21a666f1e5e76db1f3a4c2352dbb66eb64acf94b8aa8 | 12470329 |
| cycle | C03 | candidate_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C03/candidate_manifest.csv | 393cf9afdd615e55a1722f025c7479fda5e86936949c82e4d224f55b1b4398c9 | 41306697 |
| cycle | C03 | candidate_predictions | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C03/candidate_predictions.parquet | cab1eb395702cbe9750f237c803158a284d49c321b8c899d06391cc683f418d9 | 15081172 |
| cycle | C03 | qd_assignment | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C03/QD_assignment.csv | cc04d63d6ab5ea809ec238db838346a782f2eeff78796e66f6e6b21a80f03739 | 26281101 |
| cycle | C03 | q_elites | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C03/QD_quality_elites.csv | 1329966e33e1669f444f6ff7b4be2753f169adb7a60bf38d6b39d895f4137cda | 157009 |
| cycle | C03 | actual_T | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C03/T_hard_top10k_membership.csv | bc523f4123cbef894f7b17bdf4742e9fa53c605bb24bc523bada3b1bd87dc017 | 1058849 |
| cycle | C03 | counterfactual_G | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C03/G_pure_hgb_top10k_membership.csv | 8668bebd6b2b65ed8e5e16b0d4000bc5d8dec0d516b5fe8096a7c7db65e12330 | 1256662 |
| cycle | C03 | selected_current_hardness | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C03/selected_exact_hardness.parquet | f107b6837afeea77f666132641bea063022e54672eb1565838e5371d4e437f05 | 844759 |
| cycle | C03 | calibration_current_hardness | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C03/B_current_hardness.parquet | a7ef82d417e3ad8d6a01814781439413b86c8c7e0c788c7a4f939abb3b358fdb | 904273 |
| cycle | C03 | candidate_coordinates | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C03/05_scoring/union/C_union.npz | 7ec1dd12a537edc22769c21af4c969548a79d9db7bc95ec436abcab4e9c33d37 | 67853955 |
| cycle | C03 | candidate_union_parquet | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C03/05_scoring/union/C_union.parquet | 21dbb62d546da382b44c6f065afa51788234250cd2ddb772d41ac8c6a3bf6747 | 16114304 |
| cycle | C03 | archive_with_tours | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C03/01_inputs/archive_in.parquet | 2e5da9784bdec5c60202cf16028c7eaa7299067ead96bc50050a57b5dc2d5ad6 | 12958128 |
| cycle | C03 | selection_info | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C03/_selection_info.json | d11e80c6e9d77baba70af878ab9c30454efcc6c99373b1be948466c2f0df214e | 1536 |
| cycle | C03 | selection_stage_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C03/_stages/05_selection/STAGE_DONE.json | a0479d8cf3a71e65b9a3ad4400b96f936145a8f1fe5f5c5b982f5577fbccf762 | 6044 |
| cycle | C03 | calibration_stage_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C03/_stages/01_calibration/STAGE_DONE.json | 618c92decc2414ccfbecaaa0d5f796c59aff5c89eabc816d7628b8566b64e92d | 2119 |
| cycle | C03 | hgb_model_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C03/HGB_model_manifest.json | 1f4b68550e763ee16988be5cc87d7d696235e7a41e14f1661f977c9ec25071d3 | 4087 |
| checkpoint | C03 | entering_M3 | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C02/09_bq_training/M3_E256.tsp.best | f36881d48f923345f1e6c7c7aa7381c82c6c52dbc57a69d86a8d71027f38aeed | 12470329 |
| cycle | C04 | candidate_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C04/candidate_manifest.csv | b24f4b8ec064a9902ffa8e8dce21d8898ffb47f92b84330b7149c1c6a79a4497 | 40779862 |
| cycle | C04 | candidate_predictions | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C04/candidate_predictions.parquet | 92c447b7f3802521f0fc63605c06f57b99f7e4f9d566864d7e520389eca2f9c7 | 15059571 |
| cycle | C04 | qd_assignment | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C04/QD_assignment.csv | 68576088cf1aa8ea9298ec7a26842874ac38e3034bd8477956406c27dd1bd0fc | 25748321 |
| cycle | C04 | q_elites | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C04/QD_quality_elites.csv | 6a674c574ae1dd1f60165a5130f8d5eb340861a1ed69c3003c7069a50e6bcddb | 157070 |
| cycle | C04 | actual_T | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C04/T_hard_top10k_membership.csv | 36a27701e48060eafcce15542ba1f4091faaef865ce0455e6ef1914f0b37c0c0 | 1058380 |
| cycle | C04 | counterfactual_G | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C04/G_pure_hgb_top10k_membership.csv | 5fcc6eb0cfb9e0a2c273c3bdb68d136804cf94b5a3f884204ff2746e6b4bffba | 1256234 |
| cycle | C04 | selected_current_hardness | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C04/selected_exact_hardness.parquet | cbb6a6c1caacab334d80056ebd701a8e693405b02ac9c086fd7b01347b278ec4 | 851331 |
| cycle | C04 | calibration_current_hardness | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C04/B_current_hardness.parquet | fc742db2927702377e0f4f5c310f7177c614b3a0ddbfededbe3560fa8a9ee975 | 911352 |
| cycle | C04 | candidate_coordinates | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C04/05_scoring/union/C_union.npz | f21b34c41ef864fde9df4b931a9cc9db7a6c7f5d238d391f5948da501ffcfb5a | 67805422 |
| cycle | C04 | candidate_union_parquet | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C04/05_scoring/union/C_union.parquet | 37f8f653e7e6d67ddb2c46a4053e6f76697c0248915be7050703c6a496d68112 | 15742078 |
| cycle | C04 | archive_with_tours | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C04/01_inputs/archive_in.parquet | 69897988342a92c8e2c8b919c43281e576f144272796dd4f4a6d8052a3d39181 | 16870420 |
| cycle | C04 | selection_info | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C04/_selection_info.json | a7aed5990119ca940c4dad9a22cb8c531eebcdbd56ad4b0d5db6f12ebd9710c3 | 1537 |
| cycle | C04 | selection_stage_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C04/_stages/05_selection/STAGE_DONE.json | 9ecb21333c12195d7cf23aa7e14e272586c52c6165223060e5683bffa783418d | 6044 |
| cycle | C04 | calibration_stage_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C04/_stages/01_calibration/STAGE_DONE.json | c02d31231c77b0b474a1223ba8e11015e502512b6ee834b31cb23a4b02c47b1b | 2119 |
| cycle | C04 | hgb_model_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C04/HGB_model_manifest.json | d0e3c8de89559481fe352e70e3dbdaf02b9aa6c6aadb4999ab80dd6134ed36a7 | 4084 |
| checkpoint | C04 | entering_M4 | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C03/09_bq_training/M4_E256.tsp.best | a83d24e29180f57ac088e55eda2be4005b9e6fc40e4d6bcb4d3f244be943ec79 | 12470329 |
| cycle | C05 | candidate_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C05/candidate_manifest.csv | d4e044377bc25bd288376c60f9dd57eb61e3c846bca290f87d5ad8b1b924fbc1 | 40244467 |
| cycle | C05 | candidate_predictions | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C05/candidate_predictions.parquet | 3edf0e81c04323296c1956649889d9d06d6cabd42c33e5e8a0c6c6fbcb288ab8 | 15078028 |
| cycle | C05 | qd_assignment | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C05/QD_assignment.csv | 7b74722526593c1ce344169064528898409a3b608793d048f075d312551eb207 | 25210729 |
| cycle | C05 | q_elites | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C05/QD_quality_elites.csv | bac0180859e5b7cfade6de8f0cf370297c0c9b6623f006f42ae21d9347be44a4 | 157114 |
| cycle | C05 | actual_T | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C05/T_hard_top10k_membership.csv | eef567ccfb243e81f61445cc413f23f9fbdfbf037a4b5efe731ad3c74312567c | 1058115 |
| cycle | C05 | counterfactual_G | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C05/G_pure_hgb_top10k_membership.csv | d78e2c9084db4ac873d9cca42554b3dda4a6d9066d6ada979a15410e4693b7e4 | 1255990 |
| cycle | C05 | selected_current_hardness | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C05/selected_exact_hardness.parquet | 6283e01abe564dd2fc02153d267d0409dc82e801129adee375851c5e03fec2f1 | 847484 |
| cycle | C05 | calibration_current_hardness | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C05/B_current_hardness.parquet | 5ccba82a51d2d3b81dc08a178a5de77b87b2de60e0e923ce7865cda5ab4cc10c | 895848 |
| cycle | C05 | candidate_coordinates | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C05/05_scoring/union/C_union.npz | 8607fa1ae37db4760ed41d78253860df595f51caaa1a83de68d0131ef09af5d6 | 67146269 |
| cycle | C05 | candidate_union_parquet | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C05/05_scoring/union/C_union.parquet | 28024955d798cf810ee8bcbce7b85cf6004db6e6b393dc189a59c02add40ecaf | 15331856 |
| cycle | C05 | archive_with_tours | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C05/01_inputs/archive_in.parquet | 7d7582ccb7ab3e08bd85c7a6a49d754ba2c8144162eaeac619a18a576e07dc7a | 20815421 |
| cycle | C05 | selection_info | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C05/_selection_info.json | 81a1c1cf70a85757ffdb7fdf8ea274a923cd076fdbe14b5bba6dd4a87b8ea12d | 1538 |
| cycle | C05 | selection_stage_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C05/_stages/05_selection/STAGE_DONE.json | 7f6069d9ee692735ced9e79680029a77ed26a42150d441c1760b7e823f4bcd55 | 6043 |
| cycle | C05 | calibration_stage_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C05/_stages/01_calibration/STAGE_DONE.json | ac0914b4806cc51f692da0b35187d1c55e0cf1974426d19c855dae9eeb18d17d | 2118 |
| cycle | C05 | hgb_model_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C05/HGB_model_manifest.json | 7902706fa1dc5e24c8b4b4d9b2949fb3d71b606b5dbdf2dd0498000ca442fed8 | 4086 |
| checkpoint | C05 | entering_M5 | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C04/09_bq_training/M5_E256.tsp.best | ef07226bb8c3e8cd3a1404eece474df44d378316fd66509b762eb5f8bc26558c | 12470329 |
| cycle | C06 | candidate_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C06/candidate_manifest.csv | 1e0af5594d35d7ea24310cf008bb2db82ebfe575515d595ca75e5b9bc5eb9826 | 40066459 |
| cycle | C06 | candidate_predictions | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C06/candidate_predictions.parquet | 6aaa458c349e34976376782c5ca68031f1672811e73779590f03cbc2f38ac1c7 | 15109839 |
| cycle | C06 | qd_assignment | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C06/QD_assignment.csv | 67b42c3251a4e8d4c8602696a3a16d8551fc065e9781b73f172d1a89f398a543 | 25016248 |
| cycle | C06 | q_elites | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C06/QD_quality_elites.csv | 84872328cc5b2417aaa938e866c91847d015c301981b825cb7d24dbd17b4ee48 | 157251 |
| cycle | C06 | actual_T | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C06/T_hard_top10k_membership.csv | 8aff2761be3f12dd89a1d17221f91a86b2c9fe87fe0d13b9881595feaeeef1d3 | 1057143 |
| cycle | C06 | counterfactual_G | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C06/G_pure_hgb_top10k_membership.csv | 007bd9a55072d52e2c0c7c039664cb49c01c948397338e054c81aa575979eae2 | 1254868 |
| cycle | C06 | selected_current_hardness | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C06/selected_exact_hardness.parquet | 0a479591de1f636a0568a1935a8dd55cab7ca5cf3debf5f3e525a34be01e3af8 | 851954 |
| cycle | C06 | calibration_current_hardness | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C06/B_current_hardness.parquet | 09921a46c5c127697abd8a69dc9c4768a1380b80f9cb347475c1e267b8c207a4 | 908686 |
| cycle | C06 | candidate_coordinates | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C06/05_scoring/union/C_union.npz | d82212c6c6c834bb536e99b11c7ba17e20814ab74a185bb34491984f40becdff | 66841387 |
| cycle | C06 | candidate_union_parquet | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C06/05_scoring/union/C_union.parquet | 9c15d33a99dbe738063ee5933f4b24c30908c415c9816e4e274abbf80b70ce08 | 15242500 |
| cycle | C06 | archive_with_tours | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C06/01_inputs/archive_in.parquet | 15144ac91cc00d141029e8ad2f68889cbe2ecaa0e1d8734611145545e4c5d482 | 24784209 |
| cycle | C06 | selection_info | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C06/_selection_info.json | 73dcf1f849e71b2aa37765501fc0d0d8098d829833fcaa77acbb13666cbc886b | 1538 |
| cycle | C06 | selection_stage_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C06/_stages/05_selection/STAGE_DONE.json | 5f429dff5a5930af9318b21ecf64d95f63e41dabe69639645708292033990770 | 6044 |
| cycle | C06 | calibration_stage_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C06/_stages/01_calibration/STAGE_DONE.json | 4dce4c0dad94e4c81fc29b4d3ec28056fcd8e85d52fb7b3e2d859c01de042a62 | 2119 |
| cycle | C06 | hgb_model_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C06/HGB_model_manifest.json | 712234c1b83d297c953f4dcb84f1b6ba7aaf344e7ceeac6d33b45f007337fcac | 4085 |
| checkpoint | C06 | entering_M6 | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C05/09_bq_training/M6_E256.tsp.best | 05986eda65fdea971160aeb945f61a2ff648766e8cf878a46a39da90e4fe6eff | 12470329 |
| cycle | C07 | candidate_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C07/candidate_manifest.csv | 1bbf55dd2e8a53af393eeb9591a4c3e0aed102f959998fa55d7db107dab21223 | 39842260 |
| cycle | C07 | candidate_predictions | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C07/candidate_predictions.parquet | 0b8ed8f638fdd2f6fdcea624c7572e8d6575ed9702d0692c7be0d339d0b9f637 | 15145746 |
| cycle | C07 | qd_assignment | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C07/QD_assignment.csv | 5a6e873f85609e5aaecb37d0d7e205b05f3a331532d53720eebb01cc0d1dbf9d | 24792895 |
| cycle | C07 | q_elites | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C07/QD_quality_elites.csv | d707a0afa94d08c6a05b3578f389e9c2c537853283afeea5a5088060dca68ff1 | 157582 |
| cycle | C07 | actual_T | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C07/T_hard_top10k_membership.csv | af8bb111cd982188d462d48f269355dc43f0954e0cb84820e7cee4bcba136863 | 1056933 |
| cycle | C07 | counterfactual_G | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C07/G_pure_hgb_top10k_membership.csv | c592868e17cad140cee9e93686be45bf99e869d96f198e111c151c2afa9bf8a7 | 1254730 |
| cycle | C07 | selected_current_hardness | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C07/selected_exact_hardness.parquet | 126c19ed9862926e66774328c91318d1db1d8e0f7456632af03f074dd450cc3e | 854052 |
| cycle | C07 | calibration_current_hardness | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C07/B_current_hardness.parquet | fbe16c3af97d567cf77b46c15739e09ecfaddb60d0b202e8f55e8242dbb2df35 | 898323 |
| cycle | C07 | candidate_coordinates | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C07/05_scoring/union/C_union.npz | 11e346ca03bd7e1021c34e79687cd8815e2c127f066fafc577a885df2d4971dd | 67140632 |
| cycle | C07 | candidate_union_parquet | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C07/05_scoring/union/C_union.parquet | 96c5041feaa5d338e374c7c757fec949305e3f7b69f291d6499774ed21401c2f | 15141282 |
| cycle | C07 | archive_with_tours | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C07/01_inputs/archive_in.parquet | f627a14eaf39b0703cc1935a998f186d189b23a156050202635c09980318830b | 28701153 |
| cycle | C07 | selection_info | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C07/_selection_info.json | 14f6f07da778f4d1b06f81ed8cfb4ed3d75c11fe58094a9621e0adb843a2b66e | 1539 |
| cycle | C07 | selection_stage_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C07/_stages/05_selection/STAGE_DONE.json | 1bb85678a5fd5b4cbfcb686d26c6b07df1c970839508d8060a6d972024e39e88 | 6044 |
| cycle | C07 | calibration_stage_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C07/_stages/01_calibration/STAGE_DONE.json | 74251e41a2b990a496c4423fe124346ece379bde1fc4c147efef9d4194662c2d | 2119 |
| cycle | C07 | hgb_model_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C07/HGB_model_manifest.json | 077115447d0a0601d76aac72d63ea1bce33b47bf32019eb80325e971b0b401eb | 4084 |
| checkpoint | C07 | entering_M7 | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C06/09_bq_training/M7_E256.tsp.best | 5e1b14e0765643fb57ef778dea8b245b98acd0c2e28271905b3a3300a0241eae | 12470329 |
| cycle | C08 | candidate_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C08/candidate_manifest.csv | 953470327a18b8bc3adaecda4703a8910755b4225bedad1662f5f6d532a1204c | 39752627 |
| cycle | C08 | candidate_predictions | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C08/candidate_predictions.parquet | eaab03829d5925a9c834b414d7ee8303f082327ffb8322b57ae7ee622f74fffe | 15119617 |
| cycle | C08 | qd_assignment | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C08/QD_assignment.csv | 093f47f7934d5db130eb7affa4cf9f7a74cecb4af65193ba2ad4521b9c84af0a | 24698019 |
| cycle | C08 | q_elites | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C08/QD_quality_elites.csv | 02e400e72e64646f1b9b22b78842c04a1c35e949aeabe4ae16900060f957bf00 | 157730 |
| cycle | C08 | actual_T | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C08/T_hard_top10k_membership.csv | 60a2f25f137f4b61d4c876cdb93092e4fee79537b357cb57e008dd0238fc88d9 | 1056520 |
| cycle | C08 | counterfactual_G | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C08/G_pure_hgb_top10k_membership.csv | f8f49967c9ab4a316f863e93a27e82ce27d9b965831f5bd973bfd2f9a7553cee | 1254225 |
| cycle | C08 | selected_current_hardness | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C08/selected_exact_hardness.parquet | ee55546980e24a38692dc898100a4dddb8df96f5001f7d62e09c1a977f9dc914 | 851071 |
| cycle | C08 | calibration_current_hardness | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C08/B_current_hardness.parquet | 0364d63ff77e56c8aad0ccff0194f7b11a984c53c5536d310b535792ab125ebb | 907622 |
| cycle | C08 | candidate_coordinates | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C08/05_scoring/union/C_union.npz | fba98f38695c59a09a8052fbabd1e9da005a6e27fb7555fa7c817cc0a1d69f31 | 67070178 |
| cycle | C08 | candidate_union_parquet | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C08/05_scoring/union/C_union.parquet | f3e1584538dd38a4351e22d2363af14863e7c52a62f9c84eca69ab03c0ec1766 | 15026010 |
| cycle | C08 | archive_with_tours | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C08/01_inputs/archive_in.parquet | b57cb1b24ea5c88c492b1195383ffa7b1324807b50a766a7e0e868f70721f01d | 32616932 |
| cycle | C08 | selection_info | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C08/_selection_info.json | 98249a465e8c4ce43d7d785b0c234f0146e374be33846bbcb37bfe34055784df | 1536 |
| cycle | C08 | selection_stage_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C08/_stages/05_selection/STAGE_DONE.json | 26657eff69d85d9c6b9830a923b45167b3307445d9dca1eb61ffead7ab27fdfc | 6044 |
| cycle | C08 | calibration_stage_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C08/_stages/01_calibration/STAGE_DONE.json | 03f42a36dde97f13913c534c4192c76960138c635a3f905e7912a19e517ce52d | 2119 |
| cycle | C08 | hgb_model_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C08/HGB_model_manifest.json | aa91697800b0fa6ecd6889fac726a8614d8bf71da523c3c7e0ada32388e00334 | 4085 |
| checkpoint | C08 | entering_M8 | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C07/09_bq_training/M8_E256.tsp.best | 0a7cdfdfc608f59f75964b0baee87cd037d3454bfdb455088a5b56b0daaa6a2e | 12470329 |
| cycle | C09 | candidate_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C09/candidate_manifest.csv | 7d1d5c813516291bce3382eba0e72887bcd9e61df09af4d16864f88e0551006b | 39206525 |
| cycle | C09 | candidate_predictions | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C09/candidate_predictions.parquet | 2ca630d2d146c8ed9bf2e2b8941863bb03cadcaa23e5378567696e6726d77a0f | 15096492 |
| cycle | C09 | qd_assignment | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C09/QD_assignment.csv | dc8d532ca4cbb3010525bb8b5b3dd3148dbd2bddd7c89be629ac181f46e6ec83 | 24156997 |
| cycle | C09 | q_elites | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C09/QD_quality_elites.csv | 68a6bee273fae740475990e6d59e8912a2a518a007f9666441ca6e3fa5f5283b | 157689 |
| cycle | C09 | actual_T | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C09/T_hard_top10k_membership.csv | 81ae3a732b7308d8cec6165d51f4996b50e20ba7a713a4b6d52f713dba6d48f4 | 1056727 |
| cycle | C09 | counterfactual_G | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C09/G_pure_hgb_top10k_membership.csv | dadf9f364a2f4166e9cead8a24dbd0cbdeec3b3c71e7fc9bcc9e50c4cde522e4 | 1254475 |
| cycle | C09 | selected_current_hardness | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C09/selected_exact_hardness.parquet | 8fca22a3e9cc9306b94ab65cdad443dd1a35a397d6fa5d720018db37df4a5aad | 853093 |
| cycle | C09 | calibration_current_hardness | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C09/B_current_hardness.parquet | ed03088a6205ce9459957f4ca03ee2399e954bd5f3380562064a4fa745d49235 | 919658 |
| cycle | C09 | candidate_coordinates | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C09/05_scoring/union/C_union.npz | c7b18f41ca696b2c08bed7e8b3c6af5c2b552adc978e849291a51056fe0720c8 | 67106659 |
| cycle | C09 | candidate_union_parquet | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C09/05_scoring/union/C_union.parquet | 7674a38526a7972747b8e073f1abeb273cf5df0a3099d88888e1eb58f05546b3 | 14759866 |
| cycle | C09 | archive_with_tours | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C09/01_inputs/archive_in.parquet | 4b681e8bea852b297fa931ac5e96a43677790a900905117afcf8c2ff2acd32d1 | 36493133 |
| cycle | C09 | selection_info | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C09/_selection_info.json | 29c5d6b5253581103b3231d62b8ec3dfc6bd47f2902f259f5fb473a73f771f0c | 1539 |
| cycle | C09 | selection_stage_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C09/_stages/05_selection/STAGE_DONE.json | fa26845dd4b71134a6643f0f3660ecbf105e10b150ffd3c2b81f1c7af20e52f5 | 6043 |
| cycle | C09 | calibration_stage_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C09/_stages/01_calibration/STAGE_DONE.json | 093a1f51cd4f64bf951943bd355ea51f2de7f9aa33fd03578ca71af3d37ce626 | 2119 |
| cycle | C09 | hgb_model_manifest | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C09/HGB_model_manifest.json | 998b63e347e47951acaf20bebd991e6b26aa60832bceedc520051bb7a310b84a | 4079 |
| checkpoint | C09 | entering_M9 | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C08/09_bq_training/M9_E256.tsp.best | b1b9a921f07d6ff1c833f6f5232933d2e0d58e90dd398898eaea0feaff58d49c | 12470329 |
| tour_cache |  | S6AE_cycle_ledger | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C00/07_oracle/T/ledger.parquet | 67ceee0e8c07e3383d837937f8f43524b00bf382cd7deedeae5aba5f63b99b97 | 3279976 |
| tour_cache |  | S6AE_cycle_ledger | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C01/07_oracle/T/ledger.parquet | 85cf4aa306b02e3bdbab7710bbc23c86457332f03baf919b30176b32d3831af1 | 3250535 |
| tour_cache |  | S6AE_cycle_ledger | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C02/07_oracle/T/ledger.parquet | a9cb84e51d5a2bd46896e6603e06d2cbe4bcb824498af8d9b5c34b59255d806d | 3241689 |
| tour_cache |  | S6AE_cycle_ledger | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C03/07_oracle/T/ledger.parquet | d16dbce9a77890661222713009fbccbeebed3145d968449069af8239c81e6598 | 3230359 |
| tour_cache |  | S6AE_cycle_ledger | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C04/07_oracle/T/ledger.parquet | bd77f734f484ab60472756db514d50221ce3fa065629beb999bce362bb6185ff | 3209299 |
| tour_cache |  | S6AE_cycle_ledger | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C05/07_oracle/T/ledger.parquet | d723067397fca31ec6f7fc3834fb32ccbc0ac46021a63c29c72dbae2cb50db1c | 3179644 |
| tour_cache |  | S6AE_cycle_ledger | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C06/07_oracle/T/ledger.parquet | 0d84ffb4e230791bb5f1b9ee583578c65266abe7e07ec0caf4bb5c33bbc84da5 | 3204529 |
| tour_cache |  | S6AE_cycle_ledger | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C07/07_oracle/T/ledger.parquet | bf3bceb62d7ab9735d0cd784277e2eede3fdc3ea6a0ee4ca0b787e5735062b05 | 3171573 |
| tour_cache |  | S6AE_cycle_ledger | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C08/07_oracle/T/ledger.parquet | 297372ab10da96c713f5a429e986d20748f13c2b74bebab49c7160b6e9dc4c0e | 3157410 |
| tour_cache |  | S6AE_cycle_ledger | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/cycles/C09/07_oracle/T/ledger.parquet | dd57a3e87fb9f5e0bfadac61c0a28a5ada06dd32bbc5a54a622814a7aea824f2 | 3154562 |
| tour_cache |  | S6AE_initial_archive | audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/_state/A0/A0.parquet | 66d61eafa005c0fdcef1310c48c50ae4bd334714ff8598532b0308da656ef0e6 | 266899 |
| tour_cache |  | implementation_base_cache | audits/cycle0_initialization_validation_v1/03_oracle_labels/expert_tours.parquet | 29940cdb8a08363838f8d1ad894c5585b64ae515d6af2956b0d41028618f7ff3 | 1494221 |
| tour_cache |  | implementation_historical_cache | runs/S5_B_HGB40_quality_QD_parent_and_training_10cycle_M0_v1/cycles/C00/07_oracle/T/ledger.parquet | 67ceee0e8c07e3383d837937f8f43524b00bf382cd7deedeae5aba5f63b99b97 | 3279976 |
| tour_cache |  | implementation_historical_cache | runs/S5_B_HGB40_quality_QD_parent_and_training_10cycle_M0_v1/cycles/C01/07_oracle/T/ledger.parquet | 090b021382f2922d2bf44063ca5c17d515422b0721999d30d113bf46304ed83d | 3265482 |
| tour_cache |  | implementation_historical_cache | runs/S5_B_HGB40_quality_QD_parent_and_training_10cycle_M0_v1/cycles/C02/07_oracle/T/ledger.parquet | 84c5f6d29879e445304d010190bccf1a54ae15b2d85b36b67da3ca8cac6643a2 | 3234249 |
| tour_cache |  | implementation_historical_cache | runs/S5_B_HGB40_quality_QD_parent_and_training_10cycle_M0_v1/cycles/C03/07_oracle/T/ledger.parquet | bfe59284461f9b971d8d941cb68b6b30d979295953aa9c398f9dd75e8204c7f0 | 3226382 |
| tour_cache |  | implementation_historical_cache | runs/S5_B_HGB40_quality_QD_parent_and_training_10cycle_M0_v1/cycles/C04/07_oracle/T/ledger.parquet | 989e4013718d56ffce57d99649bd1ef137a4768133839bd5b53eb2c350d3a7e5 | 3217442 |
| tour_cache |  | implementation_historical_cache | runs/S5_B_HGB40_quality_QD_parent_and_training_10cycle_M0_v1/cycles/C05/07_oracle/T/ledger.parquet | 5b907c06790078f7dd11d82951b6910c662b45dbae083316f82c2078a81c6cb7 | 3183738 |
| tour_cache |  | implementation_historical_cache | runs/S5_B_HGB40_quality_QD_parent_and_training_10cycle_M0_v1/cycles/C06/07_oracle/T/ledger.parquet | b694f666032868f455aaf35b89f75785773ea88c2c2aaf77b06a95d94a7223c9 | 3168286 |
| tour_cache |  | implementation_historical_cache | runs/S5_B_HGB40_quality_QD_parent_and_training_10cycle_M0_v1/cycles/C07/07_oracle/T/ledger.parquet | 8c70bcf1eb4465db27b62e1515ce51b531ab704343a08749a643fd12f33aa82e | 3179141 |
| tour_cache |  | implementation_historical_cache | runs/S5_B_HGB40_quality_QD_parent_and_training_10cycle_M0_v1/cycles/C08/07_oracle/T/ledger.parquet | de342afdef37c4ba20116feb32b56b09e933ce1a9cf9ce3535eccb43939683aa | 3182645 |
| tour_cache |  | implementation_historical_cache | runs/S5_B_HGB40_quality_QD_parent_and_training_10cycle_M0_v1/cycles/C09/07_oracle/T/ledger.parquet | 0c6f7df623632f33e6516c39b5f047a0f0bc0cad483d7eac270a7bc841e4bf58 | 3168427 |
| tour_cache |  | implementation_base_cache | runs/blade_final_20cycle_seed2026_v1/cycles/C00/07_oracle/selected/ledger.parquet | 6ee11d483082d8c386e6c4cbb758000c472525f96428ccf66425781fc0dcb5d8 | 3283901 |
| tour_cache |  | implementation_base_cache | runs/blade_final_20cycle_seed2026_v1/cycles/C01/07_oracle/selected/ledger.parquet | d57b2ef7984a98a265faaa451f2911397a176ccdf60e9f6eed52aa4930e9f628 | 3191246 |
| tour_cache |  | implementation_base_cache | runs/blade_final_20cycle_seed2026_v1/cycles/C02/07_oracle/selected/ledger.parquet | 45348f16e18da64397635ef0543021fcd215734b1f52fbdabb9f8920b330172c | 3266969 |
| tour_cache |  | implementation_base_cache | runs/blade_final_20cycle_seed2026_v1/cycles/C03/07_oracle/selected/ledger.parquet | 8d991512fb4bd88d79442a34af48387403b0e0016b36710288c0dd070b25ae04 | 3075463 |
| tour_cache |  | implementation_base_cache | runs/blade_final_20cycle_seed2026_v1/cycles/C04/07_oracle/selected/ledger.parquet | 626576ed9598d4ff29594f96a3c2800826f938094c3e40a5c392c8480976551b | 3257766 |
| tour_cache |  | implementation_base_cache | runs/blade_final_20cycle_seed2026_v1/cycles/C05/07_oracle/selected/ledger.parquet | e7960f8140523a0576161f9a1a723f32e4afb5717a414c35a5ba0170e56e2243 | 2997657 |
| tour_cache |  | implementation_base_cache | runs/blade_final_20cycle_seed2026_v1/cycles/C06/07_oracle/selected/ledger.parquet | dd21b4d30b7de2bcf14b7c6d2ea68198f92784247c2a9aa7aa238e4e775fbe81 | 2937774 |
| tour_cache |  | implementation_base_cache | runs/blade_final_20cycle_seed2026_v1/cycles/C07/07_oracle/selected/ledger.parquet | 271bdaa010126c13b816f0fac64c0d91b1fa07f7c9ff3b0a6826e45ea5bcc3e8 | 3163936 |
| tour_cache |  | implementation_base_cache | runs/blade_final_20cycle_seed2026_v1/cycles/C08/07_oracle/selected/ledger.parquet | fd5e9a3463b94068be289fd021fe67e3bab90c64f6e68b1222d25e81dbfcc154 | 2930032 |
| tour_cache |  | implementation_base_cache | runs/blade_final_20cycle_seed2026_v1/cycles/C09/07_oracle/selected/ledger.parquet | a0402baa2ccec8f57177f713d05bfc4faeab7f5d4b5af2240cc8a6846256e51a | 3113078 |
| tour_cache |  | implementation_base_cache | runs/blade_final_20cycle_seed2026_v1/cycles/C10/07_oracle/selected/ledger.parquet | a9bf372c34c2753e934841451ad4a67a2088e0572bd917b63caedca40253c25b | 3122662 |
| tour_cache |  | implementation_base_cache | runs/blade_final_20cycle_seed2026_v1/cycles/C11/07_oracle/selected/ledger.parquet | 888579f44f5f7f8ea2f281d3ebe5254f2ddd67cf9cef25e491be91e945d4a3dc | 2918543 |
| tour_cache |  | implementation_base_cache | runs/blade_final_20cycle_seed2026_v1/cycles/C12/07_oracle/selected/ledger.parquet | dcd7fab7cdc7fbe6dae08b5b08ab21570d44e454b96a40b09b0c0d4f5781abc0 | 3069695 |
| tour_cache |  | implementation_base_cache | runs/blade_final_20cycle_seed2026_v1/cycles/C13/07_oracle/selected/ledger.parquet | 6580df7fc4aa97c8ab9533e6b3bc148159391b21118102502588b44dfcef2dd4 | 2690474 |
| tour_cache |  | implementation_base_cache | runs/blade_final_20cycle_seed2026_v1/cycles/C14/07_oracle/selected/ledger.parquet | 52a3726bf1c13ba5b9384c6750407aa2c9a6fa79a980336db71ca2c32a0d7531 | 2967069 |
| tour_cache |  | implementation_base_cache | runs/blade_final_20cycle_seed2026_v1/cycles/C15/07_oracle/selected/ledger.parquet | 17ab430a30a61b4b39899a6237f581bc6f910cf45b2e2e252318dbb64e9db61c | 3018178 |
| tour_cache |  | implementation_base_cache | runs/blade_final_20cycle_seed2026_v1/cycles/C16/07_oracle/selected/ledger.parquet | 5afd9f468143f0ecbac7a81b750cfbb0a1a1825a76a7bd588cc1a91d58ce8595 | 2670310 |
| tour_cache |  | matched_S6_control_ledger | audits/S6_AC_HGB40_EliteQD1000_PT_10kHardOnly_M0_M5_v1/cycles/C00/07_oracle/T/ledger.parquet | 67ceee0e8c07e3383d837937f8f43524b00bf382cd7deedeae5aba5f63b99b97 | 3279976 |
| tour_cache |  | matched_S6_control_ledger | audits/S6_AC_HGB40_EliteQD1000_PT_10kHardOnly_M0_M5_v1/cycles/C01/07_oracle/T/ledger.parquet | 3b9ae8f3d48f0eaead99222b28fdbf0156179d984de09b8891a1b3f692cbc532 | 3264765 |
| tour_cache |  | matched_S6_control_ledger | audits/S6_AC_HGB40_EliteQD1000_PT_10kHardOnly_M0_M5_v1/cycles/C02/07_oracle/T/ledger.parquet | 174da1d09e11383a8141be4d9d509eb6fffc974289494ff8b686d17f018d8d87 | 3249117 |
| tour_cache |  | matched_S6_control_ledger | audits/S6_AC_HGB40_EliteQD1000_PT_10kHardOnly_M0_M5_v1/cycles/C03/07_oracle/T/ledger.parquet | e1a01506c0bbfd31a2b7d525491ff8fc27020bdc0de391ac998e60def9523c4b | 3233920 |
| tour_cache |  | matched_S6_control_ledger | audits/S6_AC_HGB40_EliteQD1000_PT_10kHardOnly_M0_M5_v1/cycles/C04/07_oracle/T/ledger.parquet | 92a20830c50bde7fccf5777a2d200f2b96dd024cebfdb17fe30bcec9ec6c169a | 3217920 |
| tour_cache |  | matched_S6_control_ledger | audits/S6_AD_HGB40_EliteQD1000_PT_10kHardOnly_M5_M10_v1/cycles/C05/07_oracle/T/ledger.parquet | 86f33f2933d6486379713289d955d17389a035d87fa4f5cca00af1d56d9eddd9 | 3224266 |
| tour_cache |  | matched_S6_control_ledger | audits/S6_AD_HGB40_EliteQD1000_PT_10kHardOnly_M5_M10_v1/cycles/C06/07_oracle/T/ledger.parquet | 4e0ae33bd489fe0df53ac9e0581a245e62f344aabb6178a41e942c79836923f4 | 3181622 |
| tour_cache |  | matched_S6_control_ledger | audits/S6_AD_HGB40_EliteQD1000_PT_10kHardOnly_M5_M10_v1/cycles/C07/07_oracle/T/ledger.parquet | 7044754ffcf2a2f7afec5c488413b2beaf83e1b04487011d4484c273a36afa35 | 3169961 |
| tour_cache |  | matched_S6_control_ledger | audits/S6_AD_HGB40_EliteQD1000_PT_10kHardOnly_M5_M10_v1/cycles/C08/07_oracle/T/ledger.parquet | 09aebd4168f506a265c54ae7975ade122fd42a935c1ab87faa384ac6513883ea | 3186343 |
| tour_cache |  | matched_S6_control_ledger | audits/S6_AD_HGB40_EliteQD1000_PT_10kHardOnly_M5_M10_v1/cycles/C09/07_oracle/T/ledger.parquet | 3658b7a94cf43f2634a9c35ba240517db1bc9f1fe65150fdd7dc34228870ed60 | 3140369 |
