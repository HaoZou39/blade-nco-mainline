# D3-CVT-256 Working Map v1

版本 `d3-cvt256-v1`。状态 `FROZEN_WORKING_MAP`。这是可复现的行为映射，不是最优描述子，也不是最优分区。

运行时只读本目录里的中心、缩放器、R 特征脚本和 schema。不读取 Stage 2B-D 的实验目录，不重新拟合。

映射入口：

```bash
/home/haozou/miniconda3/envs/bq-nco/bin/python \
  /home/haozou/blade-nco-mainline/scripts/qd_map_tsp100.py \
  --input /home/haozou/blade-nco-mainline/assets/qd_maps/d3-cvt256-v1/tests/golden_single.npy \
  --map /home/haozou/blade-nco-mainline/assets/qd_maps/d3-cvt256-v1/map_manifest.json \
  --output /home/haozou/blade-nco-mainline/assets/qd_maps/d3-cvt256-v1/tests/cli_single.json
```

批量入口把 `(N, 100, 2)` 的数组一次送入同一命令，`--input` 换成 `tests/golden_inputs.npy`，`--output` 换成 `tests/cli_batch.json`。

回归测试：

```bash
/home/haozou/miniconda3/envs/bq-nco/bin/python \
  /home/haozou/blade-nco-mainline/assets/qd_maps/d3-cvt256-v1/tests/test_mapping_regression.py
```

后续改动新建 `d3-cvt256-v2`，不覆盖本目录。建议标签名是 `qd-map-d3-cvt256-v1`。需要时手动执行，不要把它当成已经打过的标签：

```bash
git tag -a qd-map-d3-cvt256-v1 -m "Freeze D3-CVT-256 working map v1"
```
