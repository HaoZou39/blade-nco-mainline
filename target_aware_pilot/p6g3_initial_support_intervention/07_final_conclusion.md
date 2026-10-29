# P6-G3 Final Conclusion

## 判定标签

`P6_G3_PARENT_SUPPORT_EXPANDED = YES`
`P6_G3_G1_NEW_REACHABILITY = SUPPORTED`
`P6_G3_STRUCTURE_VALIDITY = SUPPORTED`
`P6_G3_INITIAL_SEED_EFFECT = STRONG`
`P6_G3_MUTATION_EXPANSION_BEYOND_SEEDS = YES`
`P6_G3_ORIGINAL_SUPPORT_RETENTION = PASS`
`P6_G3_CAUSAL_CONCLUSION = SUPPORTED_FOR_THIS_INTERVENTION`
`P6_G3_ANALYSIS_STATUS = COMPLETE`

## A/B 主结果

| seed | A_parent_48 | B_parent_48 | A_G1_48 | B_G1_48 | delta_G1_48 | A_G1_17 | B_G1_17 | delta_G1_17 | B_only_reference_n | B_only_references | A_only_reference_n | A_only_references | B_true_new_beyond_B_initial_n | B_true_new_beyond_B_initial_ids | B_focus11_reached |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Original | 23 | 36 | 25 | 36 | 11 | 15 | 16 | 1 | 11 | d657;pr144;pr264;u574;u724;a280;eil51;pcb442;pr299;rat195;tsp225 | 0 |  | 1 | d657 | 8 |
| S1 | 23 | 36 | 24 | 36 | 12 | 14 | 17 | 3 | 12 | pr107;pr144;pr264;u574;u724;a280;eil51;pcb442;pr299;rat99;rat195;tsp225 | 0 |  | 1 | pr107 | 8 |
| S2 | 23 | 36 | 25 | 37 | 12 | 15 | 17 | 2 | 12 | d198;pr107;pr144;pr264;u574;u724;a280;eil51;pcb442;pr299;rat195;tsp225 | 0 |  | 2 | d198;pr107 | 9 |

固定10k抽样结果：

| group | seed | coverage_48 | coverage_48_p025 | coverage_48_p975 | coverage_17 | coverage_focus11 |
|---|---|---|---|---|---|---|
| A | Original | 22.633 | 21.725 | 24 | 14.1 | 0 |
| A | S1 | 22.533 | 21 | 24 | 14 | 0 |
| A | S2 | 22.633 | 22 | 24 | 14.1 | 0 |
| B | Original | 33.767 | 32.725 | 35 | 16 | 6.9 |
| B | S1 | 33.533 | 32 | 35.275 | 16.067 | 6.7333 |
| B | S2 | 33.467 | 32 | 34.275 | 16.133 | 6.8333 |

## 五个问题

1. **补充父代是否增加G1可达区域？** Arm B初始覆盖 `36/48`，Arm A为 `23/48`。三个seed的G1 A/B及差值见上表；结论为 **SUPPORTED**，仅适用于冻结的10%干预。
2. **直接注入还是FOGA继续扩展？** B相对自身初始父代、由非恒等G1 offspring新进入的reference在三个seed合计 `4` 个seed-reference事件。直接注入与变异维持/扩展已在 `04_new_neighborhood_arrivals.csv` 分开记录，不能把B初始增益算作FOGA新增。
3. **F40与几何是否支持？** 对4个真正超出B初始覆盖的seed-reference事件，F40最近距离改善 `3/4`，独立几何最近诊断改善 `4/4`，判定 **SUPPORTED**。作为更宽的辅助统计，B在历史缺失区域的全部到达事件中，F40或几何至少一项优于同seed A的比例为 `0.971`。连续值已同时写入 `04_new_neighborhood_arrivals.csv` 与 `05_f40_geometry_validation.csv`，没有创建新覆盖阈值。
4. **是否损失原P03结构？** 三seed共出现 `0` 个A覆盖而B未覆盖的reference事件，判定 **PASS**。这同时报告了随机移除1000个P03父代的代价。
5. **是否值得作为完整第五轮候选初始化？** `值得作为候选进入预注册的完整演化验证，但尚不能视为最优初始化。` 本实验不证明10%最优、演化深度最优或BQ训练增益。

## 证据边界

三seed是小样本探索性重复。FFT覆盖、F40距离和独立几何是结构诊断，不是Original70 Gap或训练价值证据。A的历史缓存复用经过确定性回放，但物理计算成本不等同于全量重跑；共同父代的数学结果完全配对。

`BLADE_P6G3_INITIAL_SUPPORT_INTERVENTION_COMPLETE`
