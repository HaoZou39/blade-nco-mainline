# Dataset Integrity

All six datasets are the exact P6-C1 frozen memberships; no selection was re-solved. Coordinates, ordered hashes, selected-hash files, labels, legal TSP100 tours, 48 reference quotas, C1 decile quotas, objectives, and F40 coverage pass. The P6-C1 1.20× gate remains failed and is not overwritten.

| replicate | arm | n | unique_n | membership_sha256 | manifest_sha256 | bundle_sha256 | all_tours_legal | reference_quotas_exact | hardness_decile_requirement |
|---|---|---|---|---|---|---|---|---|---|
| Original | C0 | 10000 | 10000 | 1ace440c24b61ecf5795d29a9af7fbb5251fae0f3a97a76da84e62f4fa84c2cd | 4907cab06cd89449bf91bdf75853106a3c58b595f401a9db8d00a5a02c6b1907 | cd232d08c5934e4f4ba0f9cd713b015172f173e94d79e2cba98444a831ca473f | True | True | NOT_APPLICABLE |
| Original | C1 | 10000 | 10000 | 6e8725e29ff61baf89ea32de2ed97f2ca468c17eec2ffb5dd0c4be29d198b012 | 582462a2e312fc6b43b5d8ace40eec95958bd62d1642037f3c6127ac72ee12ac | 1b9f1bf7a2d13c5063bb2d1ceae7f950c92fd9631e20d80b024d2f1ea0e3ab83 | True | True | PASS |
| S1 | C0 | 10000 | 10000 | f8429e49bf0db1e13a43cdfc46c1c494fda16456d0ae4d373594eede37129b97 | 9d7f142a2430f5325fac1aa268076a86712634d389cedb44ed82364f6e830bb0 | abd6495ec6e4b48772f00d380f7a2bbd908a77bca15a7fb35eee2d5fbbe7581a | True | True | NOT_APPLICABLE |
| S1 | C1 | 10000 | 10000 | c9d18de72c5ebeeb00d9802ca8ed7474f5d037af43bf8d4f2719089ac30e0c86 | 21e90ac5cdbd0e6451b55551c1aafb5e33dc156a6a20265345c97596722544d7 | 6c0bba1a3e45cfc70c6c8fc713031e958965d338b8d540c6b8e0b43833b9381e | True | True | PASS |
| S2 | C0 | 10000 | 10000 | ad3acdb50ae497807c6dcda646dc1096df3f0caed04983f475899f261d596826 | f7d6f2b6bb646b1f8e7c4070bc13ad220cc940f752b5e2426fc29d1387753649 | b067f73d6d453fdab9d5a9176cfd926aefe9fa1ebc491d531f083dca9fca7f26 | True | True | NOT_APPLICABLE |
| S2 | C1 | 10000 | 10000 | a027f23ff8286a763d10333dcd0da2f4342e7f3aada75a99be33f1837cb37bc0 | 6018c68d9a734602dd6865d06965ab7505049e1e32f3ad4117164c87fdd36884 | 9a263c4143f4414c95ba1ec33e680f716108500ac890d47f588af7730f7f73f6 | True | True | PASS |
