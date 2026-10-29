#!/usr/bin/env python3
"""P6-H2 training-value audit of the six frozen P6-H1 M5 models.

No evolution, no BQ/HGB training, and no Concorde calls. Teacher-forced CE
uses existing optimal tours only. Decision thresholds below are fixed in this
source file before any new CE number is computed.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("/home/haozou/blade-nco-mainline")
OUT = Path(__file__).resolve().parent
H1 = ROOT / "target_aware_pilot/p6h1_joint_evolution"
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "scripts"),
    str(ROOT / "target_aware_pilot/p6h1_joint_evolution"),
    str(ROOT / "target_aware_pilot/p6b_top5_distribution_guided_evolution"),
    str(ROOT / "target_aware_pilot/p2_bounded_evolution"),
    str(ROOT / "target_aware_pilot/p5_structure_guided_evolution_robustness"),
    str(ROOT / "target_aware_pilot/p6a_top5_target_matching_feasibility"),
    str(ROOT / "target_aware_pilot/p6_1_direct_population_training"),
    str(ROOT / "target_aware_pilot/p4b1_acquisition_ablation"),
    str(ROOT / "target_aware_pilot/p4c2_data_sampling_robustness"),
]

import run_p6b as p6b  # noqa: E402
import run_p6h1 as h1  # noqa: E402
from final_run.science import hardness, membership_hash  # noqa: E402
from mutation.result import coordinate_hash  # noqa: E402

REPLICATES = ("Original", "S1", "S2")
DEPTHS = ("1G", "2G")
SAMPLE_N = 2000
SAMPLE_SALT = "P6H2|sample|20261029"
BOOTSTRAP_N = 10_000
MIN_TARGET_N = 8
M4_EVAL = ROOT / "runs/BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1/A0_FULL_BASE/cycles/C03/evaluation_work/M4/OriginalTSPLIB70/instance_metrics.parquet"
P03_LABELS = ROOT / "runs/BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1/A0_FULL_BASE/cycles/C03/07_oracle/T/labeled_bundle.npz"
HELD22 = json.loads((ROOT / "target_aware_pilot/p1_heldout22_names.json").read_text())["names"]
TSPLIB48 = pd.read_csv(ROOT / "target_aware_pilot/03_tsplib48_reference_manifest.csv").instance_id.astype(str).tolist()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def sample_seed(key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{SAMPLE_SALT}|{key}".encode()).digest()[:8], "little")


def choose_indices(n: int, key: str) -> np.ndarray:
    rng = np.random.default_rng(sample_seed(key))
    return np.sort(rng.choice(n, SAMPLE_N, replace=False).astype(int))


def quantile_dict(values: np.ndarray, prefix: str) -> dict[str, float]:
    x = np.asarray(values, float)
    out = {f"{prefix}_mean": float(np.mean(x)), f"{prefix}_std": float(np.std(x))}
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        out[f"{prefix}_p{int(q * 100)}"] = float(np.quantile(x, q))
    return out


def bootstrap_mean(values: np.ndarray, key: str) -> tuple[float, float]:
    rng = np.random.default_rng(sample_seed(f"bootstrap|{key}"))
    x = np.asarray(values, float)
    draw = np.empty(BOOTSTRAP_N)
    for index in range(BOOTSTRAP_N):
        draw[index] = x[rng.integers(0, len(x), len(x))].mean()
    return float(np.quantile(draw, 0.025)), float(np.quantile(draw, 0.975))


def robust_scale(features: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    center = np.asarray(config["center_values"], np.float64)
    scale = np.asarray(config["scale_values"], np.float64)
    return (np.asarray(features, np.float64) - center) / scale


def pairwise_nn(scaled: np.ndarray, eff: np.ndarray, block: int = 1000) -> np.ndarray:
    from scipy.spatial.distance import cdist
    mask = np.asarray(eff, bool)
    matrix = np.asarray(scaled[:, mask], np.float64)
    if not np.isfinite(matrix).all():
        raise RuntimeError("F40 robust coordinates are not finite")
    scale = float(np.sqrt(mask.sum()))
    nearest = np.empty(len(matrix), np.float64)
    for start in range(0, len(matrix), block):
        dist = cdist(matrix[start:start + block], matrix) / scale
        for row in range(len(dist)):
            dist[row, start + row] = np.inf
        nearest[start:start + len(dist)] = dist.min(1)
    return nearest


def cross_nn(query: np.ndarray, gallery: np.ndarray, eff: np.ndarray, block: int = 1000) -> np.ndarray:
    from scipy.spatial.distance import cdist
    mask = np.asarray(eff, bool)
    q = np.asarray(query[:, mask], np.float64)
    g = np.asarray(gallery[:, mask], np.float64)
    nearest = np.empty(len(q), np.float64)
    if np.isfinite(q).all() and np.isfinite(g).all():
        scale = float(np.sqrt(mask.sum()))
        for start in range(0, len(q), block):
            nearest[start:start + block] = cdist(q[start:start + block], g).min(1) / scale
        return nearest
    for index, row in enumerate(q):
        delta = g - row
        valid = np.isfinite(delta)
        count = valid.sum(1)
        sq = np.nansum(delta * delta, axis=1)
        dist = np.sqrt(np.divide(sq, count, out=np.full(len(g), np.inf), where=count > 0))
        dist[count < 30] = np.inf
        nearest[index] = dist.min()
    return nearest


def load_features(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["instance_hash"] = frame.instance_hash.astype(str)
    return frame.drop_duplicates("instance_hash", keep="first").set_index("instance_hash")


def protocol_text() -> str:
    return """# P6-H2 Artifact And Protocol Audit

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
"""


def load_group(replicate: str, depth: str) -> dict[str, Any]:
    labels = H1 / "labels" / f"{replicate}_{depth}" / "labeled_bundle.npz"
    population = H1 / "populations" / f"{replicate}_{depth}.parquet"
    checkpoint = H1 / "checkpoints" / f"{replicate}_{depth}_M5_E256.tsp.best"
    xy, hashes, tours = p6b.load_bundle(labels, True)
    if tours is None or len(hashes) != 10_000:
        raise RuntimeError(f"missing tours: {replicate} {depth}")
    if [coordinate_hash(row) for row in xy] != hashes:
        raise RuntimeError(f"label hash mismatch: {replicate} {depth}")
    frame = pd.read_parquet(population)
    if frame.instance_hash.astype(str).tolist() != hashes:
        frame = frame.set_index(frame.instance_hash.astype(str), drop=False).loc[hashes].reset_index(drop=True)
    return {"xy": xy, "hashes": hashes, "tours": tours, "frame": frame, "checkpoint": checkpoint}


def infer_split(model: Any, xy: np.ndarray, tours: np.ndarray, device: Any) -> np.ndarray:
    values = []
    for start in range(0, len(xy), 512):
        values.append(hardness(model, xy[start:start + 512], tours[start:start + 512], device, batch_size=64))
    return np.concatenate(values)


def main() -> None:
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    (OUT / "01_artifact_protocol_audit.md").write_text(protocol_text())
    import torch
    from bq.checkpoint import load_canonical_checkpoint

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    groups = {f"{rep}_{depth}": load_group(rep, depth) for rep in REPLICATES for depth in DEPTHS}
    p03_xy, p03_hashes, p03_tours = p6b.load_bundle(P03_LABELS, True)
    if p03_tours is None or membership_hash(p03_hashes) != membership_hash(
        np.load(ROOT / "runs/BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1/A0_FULL_BASE/cycles/C03/06_selection/P_next.npz")["instance_hash"].astype(str).tolist()
    ):
        raise RuntimeError("P03 tour bundle is not the frozen parent set")
    indices = {key: choose_indices(10_000, key) for key in [*groups, "P03"]}
    write_json(OUT / "logs" / "sample_indices.json", {key: value.tolist() for key, value in indices.items()})

    cache_path = OUT / "logs" / "ce_cache.parquet"
    cached: dict[tuple[str, str], np.ndarray] = {}
    if cache_path.is_file():
        stored = pd.read_parquet(cache_path)
        for row in stored.itertuples(index=False):
            cached[(row.model_key, row.group_key)] = np.asarray(json.loads(row.ce_values), float)

    def cached_ce(model_key: str, group_key: str, checkpoint: Path, xy: np.ndarray, tours: np.ndarray, idx: np.ndarray) -> np.ndarray:
        if (model_key, group_key) in cached and len(cached[(model_key, group_key)]) == SAMPLE_N:
            return cached[(model_key, group_key)]
        model, meta = load_canonical_checkpoint(checkpoint, device, expected_sha256=None)
        if model_key == "M4" and meta["sha256"] != h1.M4_SHA:
            raise RuntimeError("M4 hash drift")
        values = infer_split(model, xy[idx], tours[idx], device)
        cached[(model_key, group_key)] = values
        pd.DataFrame([
            {"model_key": model, "group_key": group, "ce_values": json.dumps(array.tolist())}
            for (model, group), array in cached.items()
        ]).to_parquet(cache_path, index=False)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[P6-H2] CE {model_key} {group_key} mean={values.mean():.6f}", flush=True)
        return values

    rows = []
    transfer_rows = []
    ce_store = []
    m4_model_path = p6b.M4
    p03_idx = indices["P03"]
    p03_true = cached_ce("M4", "P03", m4_model_path, p03_xy, p03_tours, p03_idx)
    p03_mean = float(p03_true.mean())
    p03_p75 = float(np.quantile(p03_true, 0.75))
    for key, payload in groups.items():
        idx = indices[key]
        predicted = payload["frame"].hardness.to_numpy(float)[idx]
        true_m4 = cached_ce("M4", key, m4_model_path, payload["xy"], payload["tours"], idx)
        true_m5 = cached_ce(key, key, payload["checkpoint"], payload["xy"], payload["tours"], idx)
        full_pred = payload["frame"].hardness.to_numpy(float)
        population_p90 = float(np.quantile(full_pred, 0.9))
        top_sample = predicted >= np.quantile(predicted, 0.9)
        top_population = predicted >= population_p90
        spearman = float(spearmanr(predicted, true_m4).statistic)
        bias = float(np.mean(predicted - true_m4))
        mae = float(np.mean(np.abs(predicted - true_m4)))
        row = {
            "record_type": "sample_summary", "group": key, "n": SAMPLE_N, "spearman": spearman,
            "mae": mae, "bias_pred_minus_true": bias, "true_mean": float(true_m4.mean()),
            "sample_top10_true_mean": float(true_m4[top_sample].mean()),
            "sample_top10_pred_mean": float(predicted[top_sample].mean()),
            "population_top10_in_sample_n": int(top_population.sum()),
            "population_top10_in_sample_true_mean": float(true_m4[top_population].mean()) if top_population.any() else np.nan,
            "population_predicted_p90": population_p90,
            "m5_ce_mean": float(true_m5.mean()), "ce_delta_m5_minus_m4": float(np.mean(true_m5 - true_m4)),
            "train_ce_decreased": bool(true_m5.mean() < true_m4.mean()),
        }
        row.update(quantile_dict(predicted, "predicted"))
        row.update(quantile_dict(true_m4, "true_m4"))
        rows.append(row)
        transfer_rows.append({
            "group": key, "train_m4_ce": float(true_m4.mean()), "train_m5_ce": float(true_m5.mean()),
            "train_delta_m5_minus_m4": float(np.mean(true_m5 - true_m4)),
            "train_ce_decreased": bool(true_m5.mean() < true_m4.mean()),
            "p03_sample_m4_ce": p03_mean, "selected_minus_p03": float(true_m4.mean() - p03_mean),
        })
        ce_store.append({"model_key": "M4", "group_key": key, "ce_values": json.dumps(true_m4.tolist())})
        ce_store.append({"model_key": key, "group_key": key, "ce_values": json.dumps(true_m5.tolist())})
    ce_store.append({"model_key": "M4", "group_key": "P03", "ce_values": json.dumps(p03_true.tolist())})
    pd.DataFrame(ce_store).to_parquet(cache_path, index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "02_predicted_vs_true_hardness.csv", index=False)

    # Original70 gaps from frozen evaluation tables.
    m4_gap = pd.read_parquet(M4_EVAL)
    m4_gap["instance_id"] = m4_gap.instance_id.astype(str)
    gap_rows = []
    for key in groups:
        rep, depth = key.split("_")
        path = H1 / "logs" / key / "evaluation_work" / f"P6H1_{key}_M5" / "OriginalTSPLIB70" / "instance_metrics.parquet"
        cur = pd.read_parquet(path)
        merged = m4_gap[["instance_id", "n_nodes", "gap_percent"]].merge(
            cur[["instance_id", "gap_percent"]], on="instance_id", suffixes=("_m4", "_m5"), validate="one_to_one")
        merged["delta_gap"] = merged.gap_percent_m5 - merged.gap_percent_m4
        merged["direction"] = np.where(merged.delta_gap < 0, "improved", np.where(merged.delta_gap > 0, "worsened", "unchanged"))
        merged["subset"] = np.where(merged.instance_id.isin(TSPLIB48), "TSPLIB48", np.where(merged.instance_id.isin(HELD22), "TSPLIB22", "other"))
        low, high = bootstrap_mean(merged.delta_gap.to_numpy(float), f"gap|{key}")
        transfer_rows_key = next(item for item in transfer_rows if item["group"] == key)
        transfer_rows_key["original70_delta_gap"] = float(merged.delta_gap.mean())
        transfer_rows_key["original70_gap_worsened"] = bool(merged.delta_gap.mean() > 0)
        transfer_rows_key["mismatch"] = bool(transfer_rows_key["train_ce_decreased"] and merged.delta_gap.mean() > 0)
        for record in merged.itertuples(index=False):
            gap_rows.append({
                "record_type": "instance", "group": key, "replicate": rep, "depth": depth,
                "instance_id": record.instance_id, "n_nodes": int(record.n_nodes), "subset": record.subset,
                "gap_m4": float(record.gap_percent_m4), "gap_m5": float(record.gap_percent_m5),
                "delta_gap": float(record.delta_gap), "direction": record.direction,
                "mean_delta_ci95_low": low, "mean_delta_ci95_high": high,
            })
        for subset, part in merged.groupby("subset"):
            gap_rows.append({
                "record_type": "subset_mean", "group": key, "replicate": rep, "depth": depth,
                "instance_id": subset, "n_nodes": int(len(part)), "subset": subset,
                "gap_m4": float(part.gap_percent_m4.mean()), "gap_m5": float(part.gap_percent_m5.mean()),
                "delta_gap": float(part.delta_gap.mean()), "direction": "",
                "mean_delta_ci95_low": low, "mean_delta_ci95_high": high,
                "n_improved": int((part.delta_gap < 0).sum()), "n_worsened": int((part.delta_gap > 0).sum()),
                "n_unchanged": int((part.delta_gap == 0).sum()),
                "improved_contribution": float(part.loc[part.delta_gap < 0, "delta_gap"].sum()),
                "worsened_contribution": float(part.loc[part.delta_gap > 0, "delta_gap"].sum()),
            })
    gap_frame = pd.DataFrame(gap_rows)
    agreement = []
    inst = gap_frame[gap_frame.record_type.eq("instance")]
    for depth in DEPTHS:
        part = inst[inst.depth.eq(depth)]
        wide = part.pivot(index="instance_id", columns="replicate", values="delta_gap")
        signs = np.sign(wide.to_numpy(float))
        agreement.append({
            "record_type": "seed_agreement", "group": f"all_{depth}", "replicate": "all", "depth": depth,
            "instance_id": "", "n_nodes": int(len(wide)), "subset": "Original70",
            "gap_m4": np.nan, "gap_m5": np.nan, "delta_gap": float(np.mean(wide.to_numpy(float))),
            "direction": "", "mean_delta_ci95_low": np.nan, "mean_delta_ci95_high": np.nan,
            "n_all_improved": int(np.sum(np.all(signs < 0, axis=1))),
            "n_all_worsened": int(np.sum(np.all(signs > 0, axis=1))),
            "n_mixed": int(np.sum(~(np.all(signs < 0, axis=1) | np.all(signs > 0, axis=1) | np.any(signs == 0, axis=1)))),
        })
    gap_frame = pd.concat([gap_frame, pd.DataFrame(agreement)], ignore_index=True)
    gap_frame.to_csv(OUT / "04_original70_instance_gap.csv", index=False)

    # Structure association and redundancy. TSPLIB22 is descriptive only.
    config = h1.load_distance_config()
    _ref_frame, ref_values = h1.reference_f40()
    ref_z, eff = h1.standardized_references(config, ref_values)
    feature_bank = []
    for rep in REPLICATES:
        for tag in (f"{rep}_1G_archive", f"{rep}_2G_archive"):
            feature_bank.append(load_features(H1 / "features" / f"{tag}.parquet"))
    bank = pd.concat(feature_bank)
    bank = bank[~bank.index.duplicated(keep="first")]
    p03_missing = [value for value in p03_hashes if value not in bank.index]
    if p03_missing:
        known = h1.formal_f40()
        bank = pd.concat([bank, known])
        bank = bank[~bank.index.duplicated(keep="first")]
    redundancy = []
    selected_scaled = {}
    for key, payload in groups.items():
        hashes = payload["hashes"]
        missing = [value for value in hashes if value not in bank.index]
        if missing:
            raise RuntimeError(f"F40 missing for {len(missing)} selected instances in {key}")
        scaled = robust_scale(bank.loc[hashes, list(h1.F40)].to_numpy(float), config)
        selected_scaled[key] = scaled
        nn = pairwise_nn(scaled, eff)
        roots = payload["frame"].lineage_root.astype(str)
        effective, max_share = h1.shannon(roots)
        niches = payload["frame"].niche.astype(int)
        counts = niches.value_counts(normalize=True).to_numpy(float)
        redundancy.append({
            "group": key, "unique_n": int(pd.Series(hashes).nunique()),
            "effective_lineage": effective, "max_lineage_share": max_share,
            "median_internal_f40_nn": float(np.median(nn)), "p10_internal_f40_nn": float(np.quantile(nn, 0.1)),
            "niche_occupancy": int(niches.nunique()), "niche_hhi": float(np.sum(counts ** 2)),
            "max_niche_share": float(counts.max()), "mean_predicted_hardness": float(payload["frame"].hardness.mean()),
            "p90_predicted_hardness": float(np.quantile(payload["frame"].hardness, 0.9)),
            "hash_overlap_p03": float(np.mean(pd.Series(hashes).isin(set(p03_hashes)))),
        })
    p03_scaled = robust_scale(bank.loc[p03_hashes, list(h1.F40)].to_numpy(float), config)
    p03_nn = pairwise_nn(p03_scaled, eff)
    p03_radius = float(np.median(p03_nn))
    for item in redundancy:
        scaled = selected_scaled[item["group"]]
        dist = cross_nn(scaled, p03_scaled, eff)
        item["median_distance_to_p03"] = float(np.median(dist))
        item["fraction_inside_p03_median_nn"] = float(np.mean(dist <= p03_radius))
        item["p03_median_internal_nn"] = p03_radius
    red = pd.DataFrame(redundancy)

    # Original70 coordinates through the same passive loader used for the gaps.
    import external_v2_passive_blade_eval as evaluator
    payload = evaluator.load_fixed_references("OriginalTSPLIB70")
    records = payload["records"]
    ids = [item[0] for item in records]
    normalized = [np.asarray(evaluator.aspect_preserving_normalize(item[1]), np.float64) for item in records]
    f40_path = OUT / "logs" / "original70_f40.parquet"
    hashes70 = [coordinate_hash(row) for row in normalized]
    if f40_path.is_file():
        feat = pd.read_parquet(f40_path)
    else:
        from collections import defaultdict
        buckets: dict[int, list[tuple[str, np.ndarray]]] = defaultdict(list)
        for value, row in zip(hashes70, normalized, strict=True):
            buckets[len(row)].append((value, row))
        rscript = __import__("shutil").which("Rscript") or "/home/haozou/micromamba/envs/blade-r/bin/Rscript"
        parts = []
        for size, items in buckets.items():
            shard = OUT / "logs" / f"f40_original70_n{size}.parquet"
            h1.run_feature_job({
                "coordinates": np.stack([row for _value, row in items]),
                "hashes": [value for value, _row in items],
                "names": list(h1.F40),
                "out": str(shard),
                "rscript": rscript,
                "r_script": str(h1.R_EXTRACTOR),
                "params": h1.scaler_params(),
            })
            parts.append(pd.read_parquet(shard))
        feat = pd.concat(parts, ignore_index=True)
        feat.to_parquet(f40_path, index=False)
    if feat.instance_hash.astype(str).tolist() != hashes70:
        feat = feat.set_index(feat.instance_hash.astype(str), drop=False).loc[hashes70].reset_index(drop=True)
    query = robust_scale(feat[list(h1.F40)].to_numpy(float), config)
    assoc_rows = []
    inst = gap_frame[gap_frame.record_type.eq("instance")].copy()
    for key in groups:
        dist = cross_nn(query, selected_scaled[key], eff)
        radius = float(red.loc[red.group.eq(key), "median_internal_f40_nn"].iloc[0])
        part = inst[inst.group.eq(key)].set_index("instance_id").loc[ids]
        neighbors = np.empty(len(ids), int)
        # Count is the number of training instances inside that population's own median NN radius.
        mask = np.asarray(eff, bool)
        q = query[:, mask]
        g = selected_scaled[key][:, mask]
        for index in range(len(q)):
            delta = g - q[index]
            valid = np.isfinite(delta)
            count = valid.sum(1)
            sq = np.nansum(delta * delta, axis=1)
            value = np.sqrt(np.divide(sq, count, out=np.full(len(g), np.inf), where=count > 0))
            value[count < 30] = np.inf
            neighbors[index] = int(np.sum(value <= radius))
        finite = np.isfinite(dist)
        spear_dist = float(spearmanr(part.delta_gap.to_numpy(float)[finite], dist[finite]).statistic) if int(finite.sum()) >= 8 else float("nan")
        spear_nn = float(spearmanr(part.delta_gap, neighbors).statistic)
        for index, instance_id in enumerate(ids):
            assoc_rows.append({
                "record_type": "instance", "group": key, "instance_id": instance_id,
                "subset": "TSPLIB48" if instance_id in TSPLIB48 else "TSPLIB22",
                "delta_gap": float(part.delta_gap.iloc[index]),
                "f40_distance_to_train": float(dist[index]),
                "train_neighbors_within_median_nn": int(neighbors[index]),
                "true_m4_ce": np.nan,
                "spearman_delta_vs_f40_distance": spear_dist,
                "spearman_delta_vs_neighbor_count": spear_nn,
            })
    pd.DataFrame(assoc_rows).to_csv(OUT / "05_structure_gap_association.csv", index=False)

    # Compatible TSPLIB48 target CE: N=100 and an already stored tour. None of the
    # Original70 loader records include expert tours, and this audit does not call Concorde.
    n100 = [item[0] for item in records if len(item[1]) == 100 and item[0] in TSPLIB48]
    compatible_n = 0
    for item in transfer_rows:
        item["tsplib48_compatible_n"] = compatible_n
        item["tsplib48_n100_present"] = len(n100)
        item["tsplib48_ce_measured"] = False
        item["target_note"] = "Original70 provides optimal costs but not expert tours; teacher-forced CE was not computed and Concorde was not called"
    pd.DataFrame(transfer_rows).to_csv(OUT / "03_train_target_ce_transfer.csv", index=False)

    # Labels from the frozen rules.
    def passes_rank(row: pd.Series, floor: float) -> bool:
        return float(row.spearman) >= floor

    def passes_calibration(row: pd.Series) -> bool:
        return abs(float(row.bias_pred_minus_true)) <= 0.20 * float(row.true_mean) and float(row.mae) <= 0.35 * float(row.true_mean)

    biases = {row.group: float(row.bias_pred_minus_true) for row in summary.itertuples(index=False)}
    extra = all(biases[f"{rep}_2G"] - biases[f"{rep}_1G"] > 0.02 for rep in REPLICATES)
    rank_ok = all(passes_rank(summary[summary.group.eq(key)].iloc[0], 0.60) for key in groups)
    cal_ok = all(passes_calibration(summary[summary.group.eq(key)].iloc[0]) for key in groups)
    if rank_ok and cal_ok and not extra:
        surrogate = "SUPPORTED"
    elif sum(passes_rank(summary[summary.group.eq(key)].iloc[0], 0.40) for key in groups) >= 4:
        surrogate = "PARTIAL"
    else:
        surrogate = "WEAK"
    harder = summary.true_mean.to_numpy(float)
    p90s = summary.true_m4_p90.to_numpy(float)
    if np.all(harder > p03_mean) and np.all(p90s >= p03_p75):
        preserved = "YES"
    elif np.all(harder <= p03_mean):
        preserved = "NO"
    else:
        preserved = "INCONCLUSIVE"
    transfer = pd.DataFrame(transfer_rows)
    train_down = int(transfer.train_ce_decreased.sum())
    target = "NOT_SUPPORTED"
    mismatch_n = int(transfer.mismatch.sum())
    if mismatch_n >= 4:
        mismatch = "SUPPORTED"
    elif mismatch_n >= 2:
        mismatch = "PARTIAL"
    else:
        mismatch = "NOT_SUPPORTED"
    lineage_drop = []
    nn_drop = []
    share_up = []
    for rep in REPLICATES:
        a = red[red.group.eq(f"{rep}_1G")].iloc[0]
        b = red[red.group.eq(f"{rep}_2G")].iloc[0]
        lineage_drop.append(b.effective_lineage < a.effective_lineage)
        nn_drop.append(b.median_internal_f40_nn < a.median_internal_f40_nn)
        share_up.append(b.max_lineage_share > a.max_lineage_share)
    if all(lineage_drop) and all(nn_drop) and all(share_up):
        redundancy_label = "SUPPORTED"
    elif all(lineage_drop) and (all(nn_drop) or all(share_up)):
        redundancy_label = "PARTIAL"
    else:
        redundancy_label = "NOT_SUPPORTED"
    original_means = inst[inst.record_type.eq("instance")].groupby("group").delta_gap.mean()
    original_not_stable = not bool((original_means < 0).all())
    if surrogate == "WEAK" and not (mismatch == "SUPPORTED" and redundancy_label == "SUPPORTED"):
        primary = "SURROGATE_BIAS"
    elif mismatch == "SUPPORTED" and redundancy_label == "SUPPORTED":
        primary = "MIXED"
    elif mismatch == "SUPPORTED":
        primary = "CE_GREEDY_MISMATCH"
    elif redundancy_label == "SUPPORTED":
        primary = "DIVERSITY_CONCENTRATION"
    elif train_down >= 4 and target != "SUPPORTED" and original_not_stable:
        primary = "TRANSFER_FAILURE"
    else:
        primary = "NOT_IDENTIFIABLE"
    labels = {
        "P6_H2_SURROGATE_RELIABILITY": surrogate,
        "P6_H2_TRUE_HARDNESS_PRESERVED": preserved,
        "P6_H2_TARGET_CE_TRANSFER": target,
        "P6_H2_CE_GREEDY_MISMATCH": mismatch,
        "P6_H2_LINEAGE_REDUNDANCY": redundancy_label,
        "P6_H2_PRIMARY_EXPLANATION": primary,
        "P6_H2_STATUS": "COMPLETE",
        "tag": "BLADE_P6H2_TRAINING_VALUE_MECHANISM_AUDIT_COMPLETE",
    }
    red_lines = ["# P6-H2 Population Redundancy", "", f"P03 internal median F40 NN = {p03_radius:.6f}. Hash overlap with P03 is the exact instance intersection. Structural overlap is the fraction of the final 10k whose nearest P03 neighbor lies inside that P03 radius.", "", "| group | unique | effective lineage | max share | median internal NN | niche HHI | max niche share | hash overlap P03 | inside P03 radius | median distance to P03 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in red.itertuples(index=False):
        red_lines.append(f"| {row.group} | {row.unique_n} | {row.effective_lineage:.2f} | {row.max_lineage_share:.4f} | {row.median_internal_f40_nn:.4f} | {row.niche_hhi:.4f} | {row.max_niche_share:.4f} | {row.hash_overlap_p03:.4f} | {row.fraction_inside_p03_median_nn:.4f} | {row.median_distance_to_p03:.4f} |")
    red_lines.append("")
    red_lines.append(f"Lineage drop on all seeds: {all(lineage_drop)}. Internal NN closer on all seeds: {all(nn_drop)}. Max share higher on all seeds: {all(share_up)}. Label `{redundancy_label}`.")
    (OUT / "06_population_redundancy.md").write_text("\n".join(red_lines) + "\n")
    conclusion = f"""# P6-H2 Final Mechanism Conclusion

## 判定

`P6_H2_SURROGATE_RELIABILITY = {surrogate}`

`P6_H2_TRUE_HARDNESS_PRESERVED = {preserved}`

`P6_H2_TARGET_CE_TRANSFER = {target}`

`P6_H2_CE_GREEDY_MISMATCH = {mismatch}`

`P6_H2_LINEAGE_REDUNDANCY = {redundancy_label}`

`P6_H2_PRIMARY_EXPLANATION = {primary}`

`BLADE_P6H2_TRAINING_VALUE_MECHANISM_AUDIT_COMPLETE`

## 测量

六组各 2000 条固定样本。P03 对照样本真实 M4 CE 均值 {p03_mean:.6f}，P75 {p03_p75:.6f}。训练 CE 下降的模型数 {train_down}/6。Original70 上训练 CE 下降且平均 Gap 上升的模型数 {mismatch_n}/6。TSPLIB48 兼容 expert-tour 数 {compatible_n}（N=100 的 TSPLIB48 名称有 {len(n100)} 个，但评测资产只提供最优代价，没有可用于 teacher forcing 的专家路径）。

预测与真实 CE 的样本汇总见 `02_predicted_vs_true_hardness.csv`。训练集 CE 下降不被解释为 Original70 泛化改善。

## 为什么 Original70 没有稳定改善

主解释是 `{primary}`。六个模型的 Original70 paired 平均差仍应与 H1 的区间一起看：本审计的逐实例差和 95% bootstrap 区间在 `04_original70_instance_gap.csv`。结构距离、训练邻域计数与 DeltaGap 的 Spearman 在 `05_structure_gap_association.csv`，TSPLIB22 只作为子集描述。
"""
    (OUT / "07_final_mechanism_conclusion.md").write_text(conclusion)
    write_json(OUT / "completion.json", {**labels, "finished_utc": now(), "p03_true_mean": p03_mean, "train_ce_decreased_n": train_down, "mismatch_n": mismatch_n, "sample_n": SAMPLE_N, "device": str(device)})
    write_json(OUT / "logs" / "redundancy.json", red.to_dict(orient="records"))
    print(json.dumps(labels, indent=2), flush=True)


if __name__ == "__main__":
    main()
