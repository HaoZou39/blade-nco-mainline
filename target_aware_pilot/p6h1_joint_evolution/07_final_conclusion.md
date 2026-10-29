# P6-H1 最终结论

本实验比较的是固定 220k mutation attempts 下，一代直接搜索与两代选择后再搜索的完整效果。它不能单独分离演化深度、每代预算和中间选择的因果贡献。

## 标签

`P6_H1_JOINT_EVOLUTION_VALID = YES`
`P6_H1_STRUCTURE_IMPROVEMENT = SUPPORTED`
`P6_H1_HARDNESS_PRESERVATION = SUPPORTED`
`P6_H1_1G_VS_M4 = MIXED`
`P6_H1_2G_VS_M4 = WORSE`
`P6_H1_2G_VS_1G = MIXED`
`P6_H1_STABLE_ORIGINAL70_GAIN = INCONCLUSIVE`
`P6_H1_STATUS = COMPLETE`

## 结构

P03 的 F40 最近距离中位数是 0.867486。1G 三个最终种群的中位数平均为 0.698728，2G 为 0.656836。距离下降的最终种群有 6/6。P03 的 FFT–NN 参考覆盖是 23/48。

| replicate | depth | median_distance | median_hardness | occupied_niches | effective_lineages | fft_reference_coverage_n | pareto_rank1_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original | 1G | 0.700611 | 0.092646 | 1000 | 224.548898 | 24 | 0.002400 |
| Original | 2G | 0.660136 | 0.093108 | 1000 | 160.646750 | 22 | 0.002400 |
| S1 | 1G | 0.697642 | 0.092257 | 1000 | 227.318043 | 23 | 0.003300 |
| S1 | 2G | 0.654342 | 0.092731 | 1000 | 155.109356 | 22 | 0.002700 |
| S2 | 1G | 0.697930 | 0.092311 | 1000 | 233.834798 | 24 | 0.002800 |
| S2 | 2G | 0.656030 | 0.092609 | 1000 | 166.303715 | 23 | 0.002900 |

FFT–NN 与独立几何只作诊断。几何复合值见 `manifests/fft_geometry.csv`。

## 硬度

相对各自 offspring archive，满足“均值高于 archive 且 P90 不低于 archive P75”的最终种群有 6/6。

## Original70

M4 为 2.217003%。1G 三 seed 平均 2.306735%，2G 三 seed 平均 2.463349%。

| replicate | depth | mean_gap | mean_delta_M4_minus_model |
| --- | --- | --- | --- |
| Original | 1G | 2.189099 | 0.027904 |
| Original | 2G | 2.505821 | -0.288818 |
| S1 | 1G | 2.481112 | -0.264109 |
| S1 | 2G | 2.324452 | -0.107450 |
| S2 | 1G | 2.249994 | -0.032992 |
| S2 | 2G | 2.559774 | -0.342772 |

训练墙钟与中间选择耗时见 `02_evolution_manifest.csv` 和 `04_training_runs.csv`。

## 四个问题

联合选择是否改善结构探索：标签 SUPPORTED。6/6 个最终种群的 F40 最近距离中位数低于 P03（0.8675 → 1G 均值 0.6987，2G 均值 0.6568）。改善指相对起点 P03 的可测量接近，不是相对一个硬度单目标对照臂的因果增益。
相同 mutation 预算下一代还是两代更好：Original70 标签 MIXED。1G 三 seed 平均 gap 2.3067%，2G 2.4633%。该比较绑定了中间选择，不能解释成单纯“代数”效应。
Original70 训练收益：1G 相对 M4 为 MIXED，2G 相对 M4 为 WORSE，稳定增益标签 INCONCLUSIVE。M4 2.2170%，1G 平均 2.3067%，2G 平均 2.4633%。
若结构距离下降但 Original70 收益不稳定，证据指向结构接近并没有变成 teacher-forced 模仿所利用的训练信号：F40 最近邻改善停留在诊断空间，BQ 的贪心解码误差仍由硬度尾部、谱系覆盖或跨规模特征错配主导。硬度保持标签为 SUPPORTED。

`BLADE_P6H1_HARDNESS_STRUCTURE_JOINT_EVOLUTION_COMPLETE`
