# P6-G4 Preflight

状态：**PASS**。六组输入均为P6-G3冻结的G1 RAW，实际各有10,000父代×11算子=110,000 attempts；bundle只含有效、非identity、unique offspring。候选池不含父代，也不使用P6-B E0/E1或P6-H1。正式M4、C04 M4-conditioned HGB40、QD4归一化、CVT-1000及trainer均已哈希冻结。正式选择无lineage cap。

## RAW输入

| replicate | arm | raw_n | raw_unique_n | mutation_attempts | bundle_sha256 | runtime_sha256 | mutation_manifest_sha256 | parent_manifest_sha256 |
|---|---|---|---|---|---|---|---|---|
| Original | A | 109548 | 109548 | 110000 | 89112331d607b3dd2d89ceb87f11be59e1400d9cf30ec4b19789f4bea93ed27b | af3ae490fc53d59ad2e256c1e3b69e8b306f9fdfcb25ff623d723063eaab947f | 1498dc0e65b639dd790aa902ed8d113eec2a28449b813f4a93d527494a0680b8 | 123fbd09808eb63f495ce13523ba0027bb923f727bed18f1ae322380be6ba17a |
| Original | B | 109573 | 109573 | 110000 | 734f708850768638d2ac561677fe49f5556bc10d75992fa6a4e9dd054a62970d | 79ca077b195bbb48f266955d5b5bf3e03d9d1980633f3defcb8cd888811e0ffd | 1562871b09514c213b848cc7746c5be655b6dd7281e1cec617402ff2c45a3209 | f429e1bb3aadcd03324b8a21a19ce7628601466be53f0139f3a10fa5cb180ea8 |
| S1 | A | 109523 | 109523 | 110000 | 625dfae84ed6c925fdefd9d6a47612d6feb7eecb7ffd2c3e5c9067928ab00529 | d5d5705253d1fba19c19f8658ec8754997078a15d45ab2fd5b9307895993bfea | fa3b2cba178a0bf7a3083326c1dc9c6886c8b2f280374b624f2a5bb0ae1805f1 | 123fbd09808eb63f495ce13523ba0027bb923f727bed18f1ae322380be6ba17a |
| S1 | B | 109546 | 109546 | 110000 | efae60ad0b118b691ea87f60c71bc513f279ffd3890c5ec61d04adfff979e95d | 9f9b29b7f6402ab8cc7df2caa4b4f1132c1e283ec5f3c0af2011acef309464e4 | 486a49c2852b01f76ed095934d7d5c068b22b034fd93d3a3f2b750a6eb2ee4bc | f429e1bb3aadcd03324b8a21a19ce7628601466be53f0139f3a10fa5cb180ea8 |
| S2 | A | 109553 | 109553 | 110000 | 3eba97a8019044447898fe3a6c4057d36daff9021d6752e9711049072c9023be | daf2571b769815406e7d34bb840e8b45764c726fb542a0ca6b51ed5588bcebe5 | c13824677d25f1b5c8b4fa1cf22b0a199c7db78efe798f1d022232fbdad72fab | 123fbd09808eb63f495ce13523ba0027bb923f727bed18f1ae322380be6ba17a |
| S2 | B | 109582 | 109582 | 110000 | a2299a9d1d8fed9363ba1968d04d43b4d67d530fde6a66bdd467542f9bc57df7 | a68c9d2c335b1ee1c43731d5f31c0c1b1ff2f666269033788c1218927cbe4239 | c9d80c9c5d28c9466284b61005649f639614c2a41fad696f9ced0e5ce7971956 | f429e1bb3aadcd03324b8a21a19ce7628601466be53f0139f3a10fa5cb180ea8 |

## 冻结资产

| asset | path | sha256 |
|---|---|---|
| M4 | /home/haozou/blade-nco-mainline/runs/BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1/A0_FULL_BASE/checkpoints/M4_E256.tsp.best | c749311bd2a461084d0cbcfd383a4c0a1fc1268762b72e5f7220952f108f8190 |
| HGB manifest | /home/haozou/blade-nco-mainline/runs/BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1/A0_FULL_BASE/cycles/C04/HGB_model_manifest.json | 59b5265d9d61cd80044b58af29f3c27c278ff8578e7accf455ceee3268c17ba1 |
| QD normalization | /home/haozou/blade-nco-mainline/assets/qd/normalization/normalization_A_P1P99.json | b0de1323bb8315abd03f7e73668edafc279135c66661b15dff92a72352fa953f |
| CVT1000 | /home/haozou/blade-nco-mainline/assets/qd/cvt/centroids.csv | 739a00064870cda2384eafd7a98d5fa4feafbbf2a7a0e327d4fc4429c669bcc3 |
| formal trainer | /home/haozou/blade-nco-mainline/scripts/run_blade_core_ablation_5c_noanchor_foga11base_v1.py | e9fc32126cf841502901e36c48a648c28e03c986cdfbe942dfe022e9925f94b5 |

Protocol SHA256: `483047bb37296fdb69e5c58d2482b87bb46936e7a1de0176827eac17b5a79932`。
