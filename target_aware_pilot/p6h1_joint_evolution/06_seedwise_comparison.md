# P6-H1 分 seed 比较

M4 Original70 均值 gap 为 2.217003%。三个 seed 都必须同时报告。

## Original70

| replicate | depth | mean_gap | mean_gap_M4 | mean_delta_M4_minus_model |
| --- | --- | --- | --- | --- |
| Original | 1G | 2.189099 | 2.217003 | 0.027904 |
| Original | 2G | 2.505821 | 2.217003 | -0.288818 |
| S1 | 1G | 2.481112 | 2.217003 | -0.264109 |
| S1 | 2G | 2.324452 | 2.217003 | -0.107450 |
| S2 | 1G | 2.249994 | 2.217003 | -0.032992 |
| S2 | 2G | 2.559774 | 2.217003 | -0.342772 |

## 实例配对 bootstrap

正的 mean_delta 表示比较中的后者 gap 更低。`1G_vs_M4` 与 `2G_vs_M4` 的 delta 是 M4 减模型；`2G_vs_1G` 的 delta 是 1G 减 2G。区间条件于冻结的 70 个实例。

| replicate | comparison | dataset | n | mean_delta | ci95_low | ci95_high | model_better_instances |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original | 1G_vs_M4 | OriginalTSPLIB70 | 70 | 0.027904 | -0.380924 | 0.386862 | 32 |
| Original | 2G_vs_M4 | OriginalTSPLIB70 | 70 | -0.288818 | -0.786473 | 0.158748 | 31 |
| Original | 2G_vs_1G | OriginalTSPLIB70 | 70 | -0.316722 | -0.847820 | 0.197835 | 25 |
| S1 | 1G_vs_M4 | OriginalTSPLIB70 | 70 | -0.264109 | -0.732858 | 0.185664 | 30 |
| S1 | 2G_vs_M4 | OriginalTSPLIB70 | 70 | -0.107450 | -0.818039 | 0.392341 | 30 |
| S1 | 2G_vs_1G | OriginalTSPLIB70 | 70 | 0.156660 | -0.532498 | 0.739761 | 36 |
| S2 | 1G_vs_M4 | OriginalTSPLIB70 | 70 | -0.032992 | -0.477327 | 0.378836 | 30 |
| S2 | 2G_vs_M4 | OriginalTSPLIB70 | 70 | -0.342772 | -0.772597 | 0.062922 | 25 |
| S2 | 2G_vs_1G | OriginalTSPLIB70 | 70 | -0.309780 | -0.735654 | 0.101153 | 20 |

## 其他冻结评估集

| replicate | depth | dataset | mean_gap | mean_gap_M4 | mean_delta_M4_minus_model |
| --- | --- | --- | --- | --- | --- |
| Original | 1G | Uniform100 | 0.465153 | 0.450387 | -0.014766 |
| Original | 1G | GaussianMixture100 | 3.804544 | 3.978012 | 0.173468 |
| Original | 1G | TSPLIB48 | 1.456665 | 1.273610 | -0.183055 |
| Original | 1G | TSPLIB22 | 3.787136 | 4.275313 | 0.488177 |
| Original | 2G | Uniform100 | 0.454871 | 0.450387 | -0.004483 |
| Original | 2G | GaussianMixture100 | 4.131957 | 3.978012 | -0.153946 |
| Original | 2G | TSPLIB48 | 1.791318 | 1.273610 | -0.517708 |
| Original | 2G | TSPLIB22 | 4.064736 | 4.275313 | 0.210578 |
| S1 | 1G | Uniform100 | 0.459062 | 0.450387 | -0.008675 |
| S1 | 1G | GaussianMixture100 | 3.869843 | 3.978012 | 0.108169 |
| S1 | 1G | TSPLIB48 | 1.666180 | 1.273610 | -0.392569 |
| S1 | 1G | TSPLIB22 | 4.259146 | 4.275313 | 0.016167 |
| S1 | 2G | Uniform100 | 0.455415 | 0.450387 | -0.005028 |
| S1 | 2G | GaussianMixture100 | 4.350150 | 3.978012 | -0.372139 |
| S1 | 2G | TSPLIB48 | 1.706497 | 1.273610 | -0.432887 |
| S1 | 2G | TSPLIB22 | 3.672719 | 4.275313 | 0.602595 |
| S2 | 1G | Uniform100 | 0.451515 | 0.450387 | -0.001128 |
| S2 | 1G | GaussianMixture100 | 3.836110 | 3.978012 | 0.141901 |
| S2 | 1G | TSPLIB48 | 1.545742 | 1.273610 | -0.272132 |
| S2 | 1G | TSPLIB22 | 3.786543 | 4.275313 | 0.488770 |
| S2 | 2G | Uniform100 | 0.462194 | 0.450387 | -0.011807 |
| S2 | 2G | GaussianMixture100 | 4.205063 | 3.978012 | -0.227051 |
| S2 | 2G | TSPLIB48 | 1.732340 | 1.273610 | -0.458730 |
| S2 | 2G | TSPLIB22 | 4.365085 | 4.275313 | -0.089772 |
