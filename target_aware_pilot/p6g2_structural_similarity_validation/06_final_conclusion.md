# P6-G2 Final Conclusion

## 判定标签

`P6_G2_FFT_COVERAGE_REPRODUCED = YES`
`P6_G2_FFT_F40_CONSISTENCY = MIXED`
`P6_G2_FFT_GEOMETRY_CONSISTENCY = MIXED`
`P6_G2_EVOREAL_STRUCTURAL_ADVANTAGE = PARTIAL`
`P6_G2_PR107_GEOMETRIC_MATCH = PARTIAL`
`P6_G2_PR144_GEOMETRIC_MATCH = PARTIAL`
`P6_G2_ANALYSIS_STATUS = COMPLETE`

## 历史覆盖复现

完整池严格复现 P03 **23/48**、FOGA RAW union **25/48**、EvoReal50k **40/48**。75–150节点控制组分别为 **14/17**、**15/17**、**17/17**。这是原冻结FFT–NN指标复现，不是新覆盖定义。

## FFT 与 F40/几何一致性

| pool | top1_same_fraction | top5_jaccard_mean | fft_top1_f40_rank_fraction_median | balanced10k_fft_f40_spearman_median |
|---|---|---|---|---|
| EvoReal50k | 0 | 0 | 0.3328 | 0.52414 |
| FOGA_RAW | 0 | 0 | 0.3068 | 0.40337 |
| P03 | 0.020833 | 0.042245 | 0.17255 | 0.31473 |

EvoReal FFT覆盖的40条中，F40与独立几何相对P03/完整FOGA池均占优 `24` 条，仅一项占优 `13` 条，两项均不占优 `3` 条。逐reference证据：

| reference_id | n_nodes | matched_75_150 | EvoReal50k_fft_covered | EvoReal_f40_better_than_P03_FOGA_full | EvoReal_geometry_better_than_P03_FOGA_full | evidence_category |
|---|---|---|---|---|---|---|
| d198 | 198 | False | True | True | True | FFT_F40_GEOMETRY_SUPPORTED |
| d493 | 493 | False | False | True | True | NONFFT_STRUCTURAL_ADVANTAGE |
| d657 | 657 | False | True | True | True | FFT_F40_GEOMETRY_SUPPORTED |
| d1291 | 1291 | False | False | False | False | NO_EVOREAL_ADVANTAGE |
| fl417 | 417 | False | False | False | True | NONFFT_STRUCTURAL_ADVANTAGE |
| p654 | 654 | False | False | False | True | NONFFT_STRUCTURAL_ADVANTAGE |
| pcb1173 | 1173 | False | True | False | True | FFT_PARTIALLY_SUPPORTED |
| pr107 | 107 | True | True | True | True | FFT_F40_GEOMETRY_SUPPORTED |
| pr136 | 136 | True | True | True | False | FFT_PARTIALLY_SUPPORTED |
| pr144 | 144 | True | True | True | True | FFT_F40_GEOMETRY_SUPPORTED |
| pr152 | 152 | False | True | True | False | FFT_PARTIALLY_SUPPORTED |
| pr226 | 226 | False | True | True | True | FFT_F40_GEOMETRY_SUPPORTED |
| pr264 | 264 | False | True | True | True | FFT_F40_GEOMETRY_SUPPORTED |
| pr439 | 439 | False | False | False | True | NONFFT_STRUCTURAL_ADVANTAGE |
| pr1002 | 1002 | False | False | False | False | NO_EVOREAL_ADVANTAGE |
| u574 | 574 | False | True | True | True | FFT_F40_GEOMETRY_SUPPORTED |
| u724 | 724 | False | True | True | True | FFT_F40_GEOMETRY_SUPPORTED |
| u1060 | 1060 | False | False | True | True | NONFFT_STRUCTURAL_ADVANTAGE |
| vm1084 | 1084 | False | False | True | False | NONFFT_STRUCTURAL_ADVANTAGE |
| a280 | 280 | False | True | True | False | FFT_PARTIALLY_SUPPORTED |
| ch150 | 150 | True | True | True | True | FFT_F40_GEOMETRY_SUPPORTED |
| eil51 | 51 | False | True | True | True | FFT_F40_GEOMETRY_SUPPORTED |
| eil101 | 101 | True | True | True | True | FFT_F40_GEOMETRY_SUPPORTED |
| pcb442 | 442 | False | True | False | False | FFT_ONLY_OR_INCONSISTENT |
| pr299 | 299 | False | True | True | False | FFT_PARTIALLY_SUPPORTED |
| rat99 | 99 | True | True | True | True | FFT_F40_GEOMETRY_SUPPORTED |
| rat195 | 195 | False | True | True | True | FFT_F40_GEOMETRY_SUPPORTED |
| rat575 | 575 | False | True | True | True | FFT_F40_GEOMETRY_SUPPORTED |
| rat783 | 783 | False | True | True | True | FFT_F40_GEOMETRY_SUPPORTED |
| ts225 | 225 | False | True | True | True | FFT_F40_GEOMETRY_SUPPORTED |
| tsp225 | 225 | False | True | True | True | FFT_F40_GEOMETRY_SUPPORTED |
| bier127 | 127 | True | True | False | False | FFT_ONLY_OR_INCONSISTENT |
| ch130 | 130 | True | True | False | True | FFT_PARTIALLY_SUPPORTED |
| gil262 | 262 | False | True | False | True | FFT_PARTIALLY_SUPPORTED |
| kroA100 | 100 | True | True | True | True | FFT_F40_GEOMETRY_SUPPORTED |
| kroB100 | 100 | True | True | True | True | FFT_F40_GEOMETRY_SUPPORTED |
| kroC100 | 100 | True | True | True | False | FFT_PARTIALLY_SUPPORTED |
| kroD100 | 100 | True | True | True | True | FFT_F40_GEOMETRY_SUPPORTED |
| kroE100 | 100 | True | True | True | False | FFT_PARTIALLY_SUPPORTED |
| kroA150 | 150 | True | True | True | False | FFT_PARTIALLY_SUPPORTED |
| kroB150 | 150 | True | True | True | True | FFT_F40_GEOMETRY_SUPPORTED |
| kroA200 | 200 | False | True | True | True | FFT_F40_GEOMETRY_SUPPORTED |
| kroB200 | 200 | False | True | True | True | FFT_F40_GEOMETRY_SUPPORTED |
| lin105 | 105 | True | True | True | False | FFT_PARTIALLY_SUPPORTED |
| lin318 | 318 | False | True | False | True | FFT_PARTIALLY_SUPPORTED |
| rd100 | 100 | True | True | False | False | FFT_ONLY_OR_INCONSISTENT |
| rd400 | 400 | False | True | False | True | FFT_PARTIALLY_SUPPORTED |
| u159 | 159 | False | True | True | True | FFT_F40_GEOMETRY_SUPPORTED |

FFT–NN非常近并不保证完整结构一致：Top5集合和Top1的一致率、跨平衡10k的距离相关性均直接记录于上表。独立几何也只验证局部/全局统计与多尺度占据，不被解释成唯一“真实结构”度量。

## 固定10k公平比较

| dataset | fft_coverage_48 | fft_coverage_48_p025 | fft_coverage_48_p975 | fft_coverage_17 | best_f40_mean | best_f40_mean_p025 | best_f40_mean_p975 |
|---|---|---|---|---|---|---|---|
| EvoReal50k | 39.7 | 39 | 40 | 17 | 0.68322 | 0.67708 | 0.69364 |
| Original_E0_G2_RAW | 22.367 | 21 | 23 | 14 | 0.8363 | 0.82094 | 0.84987 |
| Original_E1_G2_RAW | 23.933 | 22 | 25 | 14.233 | 0.77459 | 0.76419 | 0.78416 |
| Original_G1_RAW | 22.467 | 21.725 | 24 | 14.033 | 0.84088 | 0.82594 | 0.85496 |
| P03 | 23 | 23 | 23 | 14 | 0.86686 | 0.86686 | 0.86686 |
| S1_E0_G2_RAW | 22 | 22 | 22 | 14 | 0.83916 | 0.82689 | 0.8527 |
| S1_E1_G2_RAW | 23.733 | 23 | 24 | 14 | 0.77669 | 0.76824 | 0.78504 |
| S1_G1_RAW | 22.467 | 21.725 | 23.275 | 14 | 0.84476 | 0.83509 | 0.86138 |
| S2_E0_G2_RAW | 22.333 | 22 | 23.275 | 14 | 0.8354 | 0.82254 | 0.84727 |
| S2_E1_G2_RAW | 23.733 | 22.725 | 25 | 14.2 | 0.77316 | 0.76352 | 0.784 |
| S2_G1_RAW | 22.467 | 22 | 23.275 | 14.1 | 0.84323 | 0.83046 | 0.85597 |

完整池结果回答历史预算内“曾发现过什么”；30次10k结果回答相同样本量下的典型表现。二者不能互换。

## 三个研究问题

1. **EvoReal 40/48有多少来自可解释结构？** 40条FFT覆盖中有 `24` 条同时得到F40与独立几何的相对支持，另有 `13` 条只获部分支持，`3` 条两项均不支持；因此不能把40/48整体视为40条几何同构，但也不是纯粹由两个统计量制造。75–150节点的17条中，双重支持 `9`、单项支持 `6`、两项均不支持 `2`。
2. **哪些差异有意义，哪些可能是度量表象？** 15条历史不可达reference中，双重支持11条：`d198, d657, pr107, pr144, pr264, u574, u724, eil51, rat195, ts225, tsp225`；部分支持3条：`pcb1173, a280, pr299`；`pcb442` 的FFT优势未得到F40或几何支持，最可能属于当前表示造成的表观差异。pr107的EvoReal FFT实例复现重复线段和近邻规律，却没有复现TSPLIB双侧长列的全局布局；pr144复现分离块与局部规则点阵，但没有完整复现原实例的重复微结构，所以两者均为PARTIAL。跨规模reference仍受节点密度和有限采样分辨率影响。
3. **下一步优先级？** 15条不可达reference中14条至少得到F40或几何支持、11条得到二者共同支持，因此证据更支持**优先验证补充初始结构支持**；但FFT/F40 Top5几乎不重合，后续结构表示至少应增加多维一致性检查，避免把单一FFT–NN覆盖当成充分条件。当前实验不提出新fitness、权重或演化深度。

## 证据边界

这些结果验证统计距离和可解释几何相似性，不验证任何BQ训练收益。所有相对比较均保留连续距离与排名，没有把几何诊断重新阈值化成另一套“覆盖率”。

`BLADE_P6G2_STRUCTURAL_SIMILARITY_VALIDATION_COMPLETE`
