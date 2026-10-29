# P6-H2 Artifact And Protocol Audit

本文件在任何新的 teacher-forced CE 计算之前写定。阈值来自本脚本常量，不随结果调整。

## 冻结输入

- 正式 M4：`c749311bd2a461084d0cbcfd383a4c0a1fc1268762b72e5f7220952f108f8190`
- 六个 P6-H1 M5 checkpoint、最终 10k 标签 bundle、population parquet、H1 评测 instance metrics
- P03 正式标签 bundle（与 `P_next` 同序，含已有 expert tour）
- M4-conditioned HGB40 的预测值直接读取 H1 population 的 `hardness` 列，不重新训练、不重新预测
- Original70 greedy gap 读取已有 M4 与六个 M5 的 instance metrics
- F40 距离复用 P6-G2 `f40_distance_config.json` 的 robust RMS
- 不演化、不训练 BQ/HGB、不调用 Concorde

## CE 定义

真实 CE 是 `teacher_forced_training_hardness` 的 `loss_raw`：专家前缀逐步交叉熵的实例均值。预测 Hardness 是 H1 已保存的 `expm1(median log1p HGB40)`，与 `loss_raw` 同一尺度。六个最终 10k 没有逐实例 CE 缓存，因此每组用固定盐 `P6H2|sample|20261029|{group}` 无放回抽取 2000 条。P03 使用独立样本。TSPLIB48 只有在 N=100 且已有 expert tour 能对上坐标哈希时才计算；少于 8 条则目标 CE 迁移视为没有可执行证据。

## 判定

`P6_H2_SURROGATE_RELIABILITY`

- SUPPORTED：六组 Spearman(预测, 真实 M4 CE) 都 ≥ 0.60，且 |mean(预测−真实)| ≤ 0.20×mean(真实)，且 MAE ≤ 0.35×mean(真实)，并且不是三个 seed 的 2G 偏差都比 1G 高过 0.02
- PARTIAL：至少四组 Spearman ≥ 0.40，但上面的校准或 2G 高估条件失败
- WEAK：其余情况

`P6_H2_TRUE_HARDNESS_PRESERVED`

比较各组 2000 条真实 M4 CE 与 P03 独立 2000 条。YES：六组均值都高于 P03 且六组 P90 ≥ P03 P75。NO：六组均值都不高于 P03。否则 INCONCLUSIVE。archive 没有 tour，不用 Concorde 补，因此不对 offspring archive 做真实 CE 比较。

`P6_H2_TARGET_CE_TRANSFER`

训练集 CE 下降只作为描述，不解释成泛化。SUPPORTED：兼容 TSPLIB 子集不少于 8 条，且至少 4/6 个模型的子集平均 CE 低于 M4。PARTIAL：2–3 个模型下降。否则 NOT_SUPPORTED。

`P6_H2_CE_GREEDY_MISMATCH`

SUPPORTED：至少 4/6 个模型同时满足训练样本平均 M5 CE < M4 CE，且 Original70 平均 Gap(M5) > Gap(M4)。PARTIAL：2–3 个。否则 NOT_SUPPORTED。

`P6_H2_LINEAGE_REDUNDANCY`

SUPPORTED：三个 seed 都满足 2G effective lineage < 1G、2G 群体内 F40 最近邻中位数 < 1G、且 2G 最大 lineage 占比 > 1G。PARTIAL：三个 seed 的 lineage 都下降，且最近邻变近或占比上升二者居其一。否则 NOT_SUPPORTED。

`P6_H2_PRIMARY_EXPLANATION`

1. surrogate 为 WEAK，且 mismatch 与 redundancy 不同时为 SUPPORTED → SURROGATE_BIAS
2. 二者同时为 SUPPORTED → MIXED
3. 仅 mismatch 为 SUPPORTED → CE_GREEDY_MISMATCH
4. 仅 redundancy 为 SUPPORTED → DIVERSITY_CONCENTRATION
5. 至少 4/6 训练 CE 下降、目标迁移不是 SUPPORTED、Original70 没有稳定改善 → TRANSFER_FAILURE
6. 其余 → NOT_IDENTIFIABLE

TSPLIB22 只进入事后分组，不参与上述阈值。
