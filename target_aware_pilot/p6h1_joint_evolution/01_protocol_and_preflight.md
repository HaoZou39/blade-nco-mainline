# P6-H1 协议与预检

本文件在任何新种群的适应度被用于改规则之前写入。后续只允许修复明确的软件错误。

## 冻结身份

- 正式 M4 SHA256：`c749311bd2a461084d0cbcfd383a4c0a1fc1268762b72e5f7220952f108f8190`。
- 正式 P03 bundle SHA256：`8ad76b90c08997bba33da1be6bf3dbfa8c28ee05bbb4a5fe4d1a9ecdb699dd62`；父代 membership `558ea278823a39275704d71e8e999efe58a56217fb23c7555cabe035c986d190`。
- HGB40：A0 C04 五个种子 2026–2030，目标 `log1p_LBQ`，部署为 median 后 `expm1` 一次。
- F40 距离配置 SHA256：`5db26bc67c8cd9ecc6845eee795d23a2a057919bf9b9333342c5674f1b261fc6`。在 P03 上与 P6-G2 缓存的最近距离最大绝对差 `5.959e-08`。
- QD4/CVT-1000 与 P03 的 niche 完全一致；`extract_qd4_exact` 与正式 F40 的 QD 四列最大差 `7.039e-14`。
- 结构距离使用 P6-G2 的全 40 维稳健标准化和 RMS，参考为 aspect 坐标下的 TSPLIB48。不用 HGB importance，不用 TSPLIB22，不用 FFT–NN 作优化目标。

## 已解决的规则冲突

历史 A0 在每个 niche 保留最高 HGB，再用全局 HGB 补足，且 `lineage_constraint=false`。P6-H1 不引入人工权重，因此把排序键从单一 HGB 换成双目标 Pareto rank，再用 NSGA-II 拥挤距离。多样性骨架仍是：每个已占据的冻结 QD niche 先留一名代表，再补到 10k。

谱系没有名额上限。`lineage_root` 从父代继承，只在 Pareto rank 与拥挤距离都相同后，按 `lineage_root`、`instance_hash` 的字典序打破平局。中间父代和最终训练集使用同一个函数。

候选池只含修复后、代内去重、且不与直接父代哈希碰撞的 offspring。不把 P03 或历史 E0/E1 放进候选池。

## 随机流

`cycle_child_seed` 没有 attempt 序号。前 110k 使用 P6-B 第一代正式 namespace，因此与历史 shared G1 原始 offspring 同分布且应逐成员一致；这只用于配对，不把历史 E0/E1 的 10k 幸存者当作候选池。后 110k 与第二代使用本次预先指定的 namespace：

- block2：`{"Original": 25, "S1": 2605015, "S2": 2605025}`
- G2：`{"Original": 26, "S1": 2605016, "S2": 2605026}`

两种深度的算子配置都是 FOGA11_BASE_V1。1G 是同一 P03 上连续两个 110k 块。2G 的 G2 父代必须是 G1 offspring 的联合选择结果。

## 训练与判定

每个最终 10k 从同一 M4 微调：fresh Adam，学习率 3.75e-5，无 scheduler，逻辑 batch 256，microbatch 64，256 epoch，10240 updates，训练种子 2026，固定 E256。专家路径只对最终 10k 求解，不对全体 offspring 调用 Concorde 或真实 BQ CE。

标签规则：

- `P6_H1_STRUCTURE_IMPROVEMENT`：六个最终种群相对 P03 的 F40 最近距离中位数全部下降，且至少四个的 FFT 参考覆盖数不低于 P03，为 SUPPORTED；至少三个距离下降但未达到上述条件为 PARTIAL；否则 NOT_SUPPORTED。
- `P6_H1_HARDNESS_PRESERVATION`：选中集预测 hardness 均值高于其 offspring archive 均值，且选中集 P90 不低于 archive P75。6/6 为 SUPPORTED，3–5 为 PARTIAL，否则 NOT_SUPPORTED。
- `P6_H1_1G_VS_M4` / `P6_H1_2G_VS_M4` / `P6_H1_2G_VS_1G`：Original70 均值在三个 seed 上全部更好为 BETTER，全部更差为 WORSE，否则 MIXED。更好指 gap 更低。
- `P6_H1_STABLE_ORIGINAL70_GAIN`：六个模型均值都低于 M4，且每个 seed×深度的实例配对（M4 gap 减模型 gap）95% bootstrap 下界大于 0，为 YES；六个均值都不低于 M4 为 NO；否则 INCONCLUSIVE。
- `P6_H1_JOINT_EVOLUTION_VALID`：预算、起点、G2 父代、选择规则和训练完整性全部通过，且最终集不等于历史 E0 或 E1，为 YES。

选择实现 SHA256：`2f6858055cff0ec3b8df228639ea532f3ff0cd5a35cd598ad0c7e09e03d2ab67`。

预检：**PASS**。
