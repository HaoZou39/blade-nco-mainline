# P6-H2 Final Mechanism Conclusion

## 判定

`P6_H2_SURROGATE_RELIABILITY = SUPPORTED`

`P6_H2_TRUE_HARDNESS_PRESERVED = YES`

`P6_H2_TARGET_CE_TRANSFER = NOT_SUPPORTED`

`P6_H2_CE_GREEDY_MISMATCH = SUPPORTED`

`P6_H2_LINEAGE_REDUNDANCY = SUPPORTED`

`P6_H2_PRIMARY_EXPLANATION = MIXED`

`BLADE_P6H2_TRAINING_VALUE_MECHANISM_AUDIT_COMPLETE`

## 测量

六组各 2000 条固定样本。P03 对照样本真实 M4 CE 均值 0.084936，P75 0.105875。训练 CE 下降的模型数 6/6。Original70 上训练 CE 下降且平均 Gap 上升的模型数 5/6。TSPLIB48 兼容 expert-tour 数 0（N=100 的 TSPLIB48 名称有 6 个，但评测资产只提供最优代价，没有可用于 teacher forcing 的专家路径）。

预测与真实 CE 的样本汇总见 `02_predicted_vs_true_hardness.csv`。训练集 CE 下降不被解释为 Original70 泛化改善。

## 为什么 Original70 没有稳定改善

主解释是 `MIXED`：训练 CE 和贪心 Gap 脱节，同时两代种群出现 lineage 冗余。预测失准不是主因。

六组 2000 条样本上，HGB40 预测与真实 M4 CE 的 Spearman 为 0.610–0.723。预测比真实平均低 0.012–0.019，MAE 约 0.035–0.040。这是轻度低估，不是 2G 系统性高估；2G 的真实均值 0.113–0.120 仍高于 1G 的 0.109–0.110。预测最高的一成，真实 CE 约 0.167–0.200，高于样本均值 0.109–0.120，难例排序还在。

被选 10k 的真实 M4 CE 均值 0.109–0.120，高于 P03 样本 0.085，且六组 P90 都高于 P03 的 P75。联合选择保住的是真实难度，不是代理偏差。archive 没有 expert tour，没有用 Concorde 去补。

六个 M5 在同一批训练样本上的 CE 都下降，降幅 0.023–0.031，降到大约 0.084–0.089，接近 P03 的水平。这只说明新数据被拟合了。TSPLIB48 没有可对齐的 expert tour，目标 CE 无法计算，因此不能把训练集 CE 下降说成目标迁移。

Original70 的平均 `Gap(M5)−Gap(M4)` 只有 Original 1G 为负（−0.028，95% 区间 [−0.388, +0.391]）。其余五个为正，最大是 S2 2G 的 +0.343。六个区间都跨过 0。其中 5/6 同时出现训练 CE 下降和 Original70 平均 Gap 上升。TSPLIB48 的平均差在 1G/2G 都为正（约 +0.283 / +0.470），退化贡献大于改善贡献。TSPLIB22 的平均差为负，但只作事后描述，三个 seed 的改善方向也不一致：1G 只有 15/70 个实例三个 seed 同时变好，2G 是 14/70。

2G 的 effective lineage 从约 225–234 降到 155–166，最大 lineage 占比从约 0.046 升到 0.073–0.078，群体内部 F40 最近邻中位数从约 0.266 降到 0.246–0.248。与 P03 的精确实例重叠为 0；落在 P03 自身最近邻半径内的比例从约 0.52 降到 0.35。冗余是 lineage 和局部结构变紧，不是 niche 被掏空，也不是更贴着 P03。

Original70 到训练集的 F40 最近距离均值约 0.88–0.89，全部高于训练集内部最近邻半径，所以按该半径计的邻域计数全是 0，不能用来解释逐实例 Gap。DeltaGap 与这个 F40 距离的 Spearman 只有 −0.17 到 +0.15，符号也不稳定。实例级结构邻近度没有解释谁变好、谁变差。
