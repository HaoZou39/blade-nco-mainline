# P6-H3 Protocol And Preflight

写于演化覆盖统计之前。阈值不随结果修改。

## 设计

四组连续 10 轮，不是 10 次独立重启。每轮每个父代调用全部 11 个正式算子各一次。

| 组 | 父代 | 选择 |
|---|---|---|
| A | Uniform10k | 均匀随机 10k |
| B | Uniform10k | P6-H1 Hardness–Structure Pareto/QD |
| C | P03 10k | 均匀随机 10k |
| D | P03 10k | P6-H1 Hardness–Structure Pareto/QD |

算子 ID：`['uniform_mutation', 'normal_mutation', 'rotation_mutation', 'implosion_mutation', 'explosion_mutation', 'cluster_mutation', 'expansion_mutation', 'compression_mutation', 'axis_projection_mutation', 'linear_projection_mutation', 'grid_mutation']`。命名空间：`{"Original": {"1": 81, "2": 82, "3": 83, "4": 84, "5": 85, "6": 86, "7": 87, "8": 88, "9": 89, "10": 90}, "S1": {"1": 2605101, "2": 2605102, "3": 2605103, "4": 2605104, "5": 2605105, "6": 2605106, "7": 2605107, "8": 2605108, "9": 2605109, "10": 2605110}, "S2": {"1": 2605201, "2": 2605202, "3": 2605203, "4": 2605204, "5": 2605205, "6": 2605206, "7": 2605207, "8": 2605208, "9": 2605209, "10": 2605210}}`。选择随机流盐为 `P6H3|select|{arm}|{seed}|{gen}`，与 `cycle_child_seed` 分离。同一命名空间、同一父代哈希产生相同 offspring；A/B 在父代还相同的轮次共享变异，C/D 同样。父代一旦因选择分叉，后续变异自然不同。

Uniform10k 使用正式管线冻结的 U0：`/home/haozou/blade-nco-mainline/runs/blade_v2_corrected_union_sanity_seed2026_v1/C00/01_inputs/parents.npz`，成员哈希 `edf1719ecc114261ba26b21f7bb8ff113b8247958679abb3d177a858cd7c2c5b`，顺序与 `assets/qd/roots/root_hashes.txt` 一致。原始 `fixed_uniform_anchor_10k_coordinates.npz` 不在磁盘上，因此不用另一个均匀样本顶替；本文件记录的是已经由正式实验冻结、并且坐标哈希可复核的那一份。P03 是 M4 对应的 `P_next.npz`。

FFT–NN 半径、中心和 IQR 尺度读取冻结的 P6-G0 规则，不重新估计。F40 距离读取 P6-G2 robust RMS。几何比较调用 P6-G2 `geometry_compare`。TSPLIB22 不进入选择。

## 判定

到达增益 = G10 累计 RAW 覆盖的 11 个双重支持 reference 数 − G0。种子中位数取三个 seed 的中间值。

- `P6_H3_UNIFORM_10G_REACHABILITY`：A/B 中增益中位数较大者 ≥3 为 EXPANDED，>0 为 LIMITED，否则 NONE。P03 的 C/D 同理。
- `P6_H3_INITIAL_SUPPORT_EFFECT`：随机比较（A 对 C）和联合比较（B 对 D）的 G10 累计 11 覆盖中位数都是 Uniform 至少多 2，则为 SUPPORTED；只有一组如此为 PARTIAL；否则 NOT_SUPPORTED。
- `P6_H3_SELECTION_BOTTLENECK`：某一父代族内随机比联合的累计 RAW 11 覆盖中位数至少多 2，或至少两个 arm 的累计 RAW 比累计 SELECTED 至少多 2，为 SUPPORTED；对应差距为 1 则为 PARTIAL；否则 NOT_SUPPORTED。
- `P6_H3_MUTATION_REACHABILITY_LIMIT`：四个 arm 的增益中位数都 ≤1，且 12 条轨迹的 pr107 与 pr144 在 G10 仍在冻结半径之外，为 SUPPORTED_WITHIN_BUDGET。任一 arm 增益中位数 ≥3 为 NOT_SUPPORTED。其余 INCONCLUSIVE。这不是数学不可能。
- `P6_H3_DEPTH_EXPANSION`：某个 arm 至少 2/3 seed 在 G10 的累计 11 覆盖高于 G2，或 pr107/pr144 的最近距离小于 G2 的 95%，则该 arm 算深度有增益。≥2 个 arm 如此为 SUPPORTED，1 个为 PARTIAL，否则 NOT_SUPPORTED。
- `P6_H3_GEOMETRIC_VALIDITY`：双重支持 reference 的 FFT 首次命中里，至少一半同时优于历史 FOGA RAW 与 P03 的几何 composite，且这样的命中不少于 3，为 SUPPORTED；至少 1 个几何更优为 PARTIAL；否则 NOT_SUPPORTED。
- `P6_H3_PRIMARY_BOTTLENECK`：初始支持、选择、预算内变异限制三者恰好一个成立时分别标 INITIAL_SUPPORT / SELECTION / MUTATION_TRANSITIONS；两个及以上为 MIXED；都没有为 NOT_IDENTIFIABLE。

## 预检

```json
{
  "dual11": [
    "d198",
    "d657",
    "pr107",
    "pr144",
    "pr264",
    "u574",
    "u724",
    "eil51",
    "rat195",
    "ts225",
    "tsp225"
  ],
  "fft_radius": 0.1770575219387968,
  "finished_utc": "2026-10-29T00:41:46Z",
  "hist15": [
    "d198",
    "d657",
    "pcb1173",
    "pr107",
    "pr144",
    "pr264",
    "u574",
    "u724",
    "a280",
    "eil51",
    "pcb442",
    "pr299",
    "rat195",
    "ts225",
    "tsp225"
  ],
  "namespaces": {
    "Original": {
      "1": 81,
      "2": 82,
      "3": 83,
      "4": 84,
      "5": 85,
      "6": 86,
      "7": 87,
      "8": 88,
      "9": 89,
      "10": 90
    },
    "S1": {
      "1": 2605101,
      "2": 2605102,
      "3": 2605103,
      "4": 2605104,
      "5": 2605105,
      "6": 2605106,
      "7": 2605107,
      "8": 2605108,
      "9": 2605109,
      "10": 2605110
    },
    "S2": {
      "1": 2605201,
      "2": 2605202,
      "3": 2605203,
      "4": 2605204,
      "5": 2605205,
      "6": 2605206,
      "7": 2605207,
      "8": 2605208,
      "9": 2605209,
      "10": 2605210
    }
  },
  "operators": [
    "uniform_mutation",
    "normal_mutation",
    "rotation_mutation",
    "implosion_mutation",
    "explosion_mutation",
    "cluster_mutation",
    "expansion_mutation",
    "compression_mutation",
    "axis_projection_mutation",
    "linear_projection_mutation",
    "grid_mutation"
  ],
  "p03_bundle_sha256": "8ad76b90c08997bba33da1be6bf3dbfa8c28ee05bbb4a5fe4d1a9ecdb699dd62",
  "selection_source_sha256": "2f6858055cff0ec3b8df228639ea532f3ff0cd5a35cd598ad0c7e09e03d2ab67",
  "size17": [
    "pr107",
    "pr136",
    "pr144",
    "ch150",
    "eil101",
    "rat99",
    "bier127",
    "ch130",
    "kroA100",
    "kroB100",
    "kroC100",
    "kroD100",
    "kroE100",
    "kroA150",
    "kroB150",
    "lin105",
    "rd100"
  ],
  "status": "PASS",
  "uniform_file_sha256": "2b9e98ade546785be888f804b5a531641f340954e5a294db01af982beb1136ad",
  "uniform_membership": "edf1719ecc114261ba26b21f7bb8ff113b8247958679abb3d177a858cd7c2c5b"
}
```
