# D3-CVT-256 v1 封存记录

状态：`QD_WORKING_MAP_V1_FROZEN`

版本：`d3-cvt256-v1`。封存日期 2026-11-09。软件提交 `0aa106ac48b39d9b2098805acce1985285ec688b`。

这次只复制和校验已经通过的资产。没有重拟合中心，没有重估缩放器，没有改 K，没有改描述子，也没有改 Stage 2A、Stage 2B 或 Stage 2B-D 的历史结论。

## 身份

描述子 D3，顺序为 `spn_mst_dists_varcoeff`、`spn_distance_median`、`spn_angles_mean`。

分区是 `KMeans(n_clusters=256, n_init=1, random_state=20261108)`。参照是 Stage 2A R_small：1500 个 Uniform100 根，深度 0–10，16500 个状态。

中心数组 SHA256，按 float64 连续字节计算：

`798a341de070fcf21242af853ff18b307641005635540b7bd7945ef44dbaa99a`

中心文件 SHA256，包含 NumPy 文件头：

`be817c1160de638a925bd3670da0165572a5c6495eb6b731aca0fff3fef020d6`

缩放器 SHA256 `281d887d952cc557b2b472d8e11f4214569720d4a102347591235a7dd1009d6d`。

特征脚本 SHA256 `37f676211b6ebca55321fe6d7035b3832625d9f49878f079561c6cd3e2c760e8`。

schema SHA256 `a1eb4c2b28aeb7adb447c1afb90cf3d4933edef86c050893258c07c15727b62a`。

映射实现 SHA256 `a2acf6b2baa43385c7daa3cad808c1e79fd0c6555505dc5b730c301565797809`。

Manifest SHA256 `36854f7bc06f4c10facf7415f7cf11e91e9268e695487a23460bf0160ea63348`。

niche 编号是缩放 D3 空间里的欧氏最近中心下标。距离并列时取较小下标。支持状态只作诊断，越界实例仍然返回最近 niche。

## 回归

容差沿用 Stage 2B-D 的 `1e-5`。三条黄金样本的原始 D3、缩放 D3、niche、距离和边际的实际最大绝对差是 0。

| 样本 | niche |
| --- | --- |
| Stage 2B 短链探针 | 46 |
| E_DEEP 深度 1000 | 227 |
| Stage 2C 历史实例 | 202 |

单实例命令和三实例批量命令都成功。形状 `(99, 2)` 和非有限坐标都被拒绝。

运行时路径都在 `assets/qd_maps/d3-cvt256-v1/` 内。特征脚本是源文件的字节副本，不再依赖 Stage 2B-D 目录。

## 科学状态

Stage 2A 保持 `DECISION_C`。Stage 2B 保持 `DECISION_B`。Stage 2B-D 保持 `INCONCLUSIVE`。映射保持 `MAPPING_PASS`，深层支持保持 `DEEP_SUPPORT_OK`，参照充分性保持 `REFERENCE_INCONCLUSIVE`。

本目录被直接修改时不再是 v1。下一版使用 `d3-cvt256-v2`。
