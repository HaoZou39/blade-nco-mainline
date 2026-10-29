#!/usr/bin/env python3
"""P6-H3: ten continuous FOGA-11 generations, 2x2 parent/selection design.

Mutation code, repair, operator parameters and the P6-H1 Pareto/QD rule are
reused unchanged. No BQ, HGB, GP or Concorde training/calls. Decision
thresholds in this file are fixed before coverage is computed.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path("/home/haozou/blade-nco-mainline")
OUT = Path(__file__).resolve().parent
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
    str(ROOT / "target_aware_pilot/p6g2_structural_similarity_validation"),
]

import run_p6b as p6b  # noqa: E402
import run_p6h1 as h1  # noqa: E402
from final_run.science import membership_hash  # noqa: E402
from mutation import OPERATOR_NAMES  # noqa: E402
from mutation.result import coordinate_hash  # noqa: E402

SEEDS = ("Original", "S1", "S2")
ARMS = {
    "A": {"parents": "uniform", "selection": "random"},
    "B": {"parents": "uniform", "selection": "joint"},
    "C": {"parents": "p03", "selection": "random"},
    "D": {"parents": "p03", "selection": "joint"},
}
GENERATIONS = 10
POPULATION_N = 10_000
FEATURE_WORKERS = 12
MAX_PARALLEL = 3
EXPECTED_OPERATORS = (
    "uniform_mutation", "normal_mutation", "rotation_mutation", "implosion_mutation",
    "explosion_mutation", "cluster_mutation", "expansion_mutation", "compression_mutation",
    "axis_projection_mutation", "linear_projection_mutation", "grid_mutation",
)
# Fresh namespaces, distinct from P5/P6-B (5, 6, 2605011/12, 2605021/22)
# and from the in-memory P6-H1 keys (11, 12, 25, 26, 2605015/16, 2605025/26).
NAMESPACES = {
    "Original": {g: 80 + g for g in range(1, GENERATIONS + 1)},
    "S1": {g: 2_605_100 + g for g in range(1, GENERATIONS + 1)},
    "S2": {g: 2_605_200 + g for g in range(1, GENERATIONS + 1)},
}
HIST15 = (
    "d198", "d657", "pcb1173", "pr107", "pr144", "pr264", "u574", "u724",
    "a280", "eil51", "pcb442", "pr299", "rat195", "ts225", "tsp225",
)
DUAL11 = ("d198", "d657", "pr107", "pr144", "pr264", "u574", "u724", "eil51", "rat195", "ts225", "tsp225")
U0 = ROOT / "runs/blade_v2_corrected_union_sanity_seed2026_v1/C00/01_inputs/parents.npz"
U0_MEMBERSHIP = "edf1719ecc114261ba26b21f7bb8ff113b8247958679abb3d177a858cd7c2c5b"
ROOT_HASHES = ROOT / "assets/qd/roots/root_hashes.txt"
SELECTION_SHA = "2f6858055cff0ec3b8df228639ea532f3ff0cd5a35cd598ad0c7e09e03d2ab67"
FFT_RULES = ROOT / "target_aware_pilot/p6e0_evoreal_structural_representation_audit/configs/reference_rules_frozen.json"
REF_FFT = ROOT / "target_aware_pilot/p6e0_evoreal_structural_representation_audit/features/TSPLIB48.parquet"
G2_VALIDATION = ROOT / "target_aware_pilot/p6g2_structural_similarity_validation/04_reference_level_validation.csv"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def stream_seed(material: str) -> int:
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "little")


def locked(path: Path, action):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            return action()
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_refs() -> pd.DataFrame:
    manifest = pd.read_csv(ROOT / "target_aware_pilot/03_tsplib48_reference_manifest.csv")
    fft = pd.read_parquet(REF_FFT)
    frame = manifest[["instance_id", "n_nodes", "source_file"]].merge(
        fft[["instance_id", "fft_energy", "nn_ratio"]], on="instance_id", validate="one_to_one")
    if len(frame) != 48:
        raise RuntimeError("TSPLIB48 FFT table is not 48 rows")
    return frame


def reference_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    size17 = frame.loc[(frame.n_nodes >= 75) & (frame.n_nodes <= 150), "instance_id"].astype(str).tolist()
    return {"all48": frame.instance_id.astype(str).tolist(), "size17": size17, "hist15": list(HIST15), "dual11": list(DUAL11)}


def fft_features(coordinates: np.ndarray) -> np.ndarray:
    from evoreal_structure_alignment_audit_corrected_v2 import fft_energy, nn_ratio_stats
    out = np.empty((len(coordinates), 2), np.float32)
    for index, xy in enumerate(coordinates):
        out[index, 0] = np.float32(fft_energy(xy))
        out[index, 1] = np.float32(nn_ratio_stats(xy)["nn_ratio"])
        if index and index % 20000 == 0:
            print(f"[P6-H3] fft {index}/{len(coordinates)}", flush=True)
    return out


def distance_matrix(features: np.ndarray, refs: pd.DataFrame, rules: dict[str, Any]) -> np.ndarray:
    from scipy.spatial.distance import cdist
    center = np.asarray(rules["center"], np.float64)
    scale = np.asarray(rules["scale_iqr"], np.float64)
    cand = (features.astype(np.float64) - center) / scale
    ref = (refs[["fft_energy", "nn_ratio"]].to_numpy(np.float64) - center) / scale
    out = np.empty((len(cand), len(ref)), np.float32)
    for start in range(0, len(cand), 8192):
        out[start:start + 8192] = cdist(cand[start:start + 8192], ref).astype(np.float32)
    return out


def coverage_counts(dist: np.ndarray, mask: np.ndarray, sets: dict[str, list[str]], ids: list[str], radius: float) -> dict[str, int]:
    hit = (dist[mask].min(0) <= radius) if mask.any() else np.zeros(dist.shape[1], bool)
    return {name: int(sum(bool(hit[ids.index(ref)]) for ref in members)) for name, members in sets.items()}


def protocol_text(extra: dict[str, Any]) -> str:
    return f"""# P6-H3 Protocol And Preflight

写于演化覆盖统计之前。阈值不随结果修改。

## 设计

四组连续 10 轮，不是 10 次独立重启。每轮每个父代调用全部 11 个正式算子各一次。

| 组 | 父代 | 选择 |
|---|---|---|
| A | Uniform10k | 均匀随机 10k |
| B | Uniform10k | P6-H1 Hardness–Structure Pareto/QD |
| C | P03 10k | 均匀随机 10k |
| D | P03 10k | P6-H1 Hardness–Structure Pareto/QD |

算子 ID：`{list(EXPECTED_OPERATORS)}`。命名空间：`{json.dumps(NAMESPACES)}`。选择随机流盐为 `P6H3|select|{{arm}}|{{seed}}|{{gen}}`，与 `cycle_child_seed` 分离。同一命名空间、同一父代哈希产生相同 offspring；A/B 在父代还相同的轮次共享变异，C/D 同样。父代一旦因选择分叉，后续变异自然不同。

Uniform10k 使用正式管线冻结的 U0：`{U0}`，成员哈希 `{U0_MEMBERSHIP}`，顺序与 `assets/qd/roots/root_hashes.txt` 一致。原始 `fixed_uniform_anchor_10k_coordinates.npz` 不在磁盘上，因此不用另一个均匀样本顶替；本文件记录的是已经由正式实验冻结、并且坐标哈希可复核的那一份。P03 是 M4 对应的 `P_next.npz`。

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
{json.dumps(extra, indent=2, sort_keys=True)}
```
"""


def preflight() -> dict[str, Any]:
    operators = tuple(OPERATOR_NAMES)
    if operators != EXPECTED_OPERATORS:
        raise RuntimeError(f"operator list drift: {operators}")
    h1._self_test_selection()
    rule = h1.source_hash([h1.pareto_ranks, h1.crowding_distance, h1.select_joint, h1.min_f40_distance])
    if rule != SELECTION_SHA:
        raise RuntimeError(f"selection rule drift: {rule}")
    if sha256_file(p6b.M4) != h1.M4_SHA or sha256_file(p6b.P03_BUNDLE) != h1.P03_BUNDLE_SHA:
        raise RuntimeError("M4 or P03 bundle drift")
    with np.load(U0) as payload:
        xy = np.asarray(payload["coordinates"])
        hashes = payload["instance_hash"].astype(str).tolist()
    roots = [line.strip() for line in ROOT_HASHES.read_text().splitlines() if line.strip()]
    if xy.shape != (10_000, 100, 2) or hashes != roots or membership_hash(hashes) != U0_MEMBERSHIP:
        raise RuntimeError("Uniform10k is not the frozen U0 root manifest")
    if any(coordinate_hash(row) != value for row, value in zip(xy[::500], hashes[::500])):
        raise RuntimeError("Uniform coordinate hash sample failed")
    full_match = all(coordinate_hash(row) == value for row, value in zip(xy, hashes))
    if not full_match:
        raise RuntimeError("Uniform coordinate bytes do not reproduce root hashes")
    refs = load_refs()
    sets = reference_sets(refs)
    if len(sets["size17"]) != 17 or any(ref not in sets["all48"] for ref in HIST15):
        raise RuntimeError("reference partition drift")
    rules = json.loads(FFT_RULES.read_text())
    if abs(float(rules["radius"]) - 0.1770575219387968) > 1e-15:
        raise RuntimeError("FFT radius drift")
    g2 = pd.read_csv(G2_VALIDATION)
    g2["reference_id"] = g2.reference_id.astype(str)
    dual = set(g2.loc[g2.reference_id.isin(HIST15) & g2.evidence_category.eq("FFT_F40_GEOMETRY_SUPPORTED"), "reference_id"])
    partial = set(g2.loc[g2.reference_id.isin(("pcb1173", "a280", "pr299")) & g2.evidence_category.eq("FFT_PARTIALLY_SUPPORTED"), "reference_id"])
    if dual != set(DUAL11) or partial != {"pcb1173", "a280", "pr299"} or g2.loc[g2.reference_id.eq("pcb442"), "evidence_category"].iloc[0] != "FFT_ONLY_OR_INCONSISTENT":
        raise RuntimeError(f"historical evidence categories drifted: dual={sorted(dual)}")
    info = {
        "status": "PASS", "operators": list(operators), "namespaces": NAMESPACES,
        "uniform_membership": U0_MEMBERSHIP, "uniform_file_sha256": sha256_file(U0),
        "p03_bundle_sha256": h1.P03_BUNDLE_SHA, "selection_source_sha256": rule,
        "fft_radius": float(rules["radius"]), "size17": sets["size17"], "dual11": list(DUAL11),
        "hist15": list(HIST15), "finished_utc": now(),
    }
    (OUT / "configs").mkdir(exist_ok=True)
    write_json(OUT / "configs" / "frozen_p6h3_config.json", info)
    (OUT / "01_protocol_and_preflight.md").write_text(protocol_text(info))
    print("[P6-H3] preflight PASS", flush=True)
    return info


def load_uniform() -> tuple[np.ndarray, pd.DataFrame]:
    with np.load(U0) as payload:
        xy = np.asarray(payload["coordinates"], np.float64)
        hashes = payload["instance_hash"].astype(str).tolist()
    frame = pd.DataFrame({"instance_hash": hashes, "lineage_root": hashes, "parent_hash": "", "operator_id": "uniform_anchor"})
    return xy, frame


def load_p03() -> tuple[np.ndarray, pd.DataFrame]:
    xy, hashes, _ = p6b.load_bundle(p6b.P03_BUNDLE)
    meta = pd.read_parquet(p6b.P03_META)
    if meta.instance_hash.astype(str).tolist() != hashes:
        meta = meta.set_index(meta.instance_hash.astype(str), drop=False).loc[hashes].reset_index(drop=True)
    frame = pd.DataFrame({
        "instance_hash": hashes,
        "lineage_root": meta.lineage_root.astype(str).tolist(),
        "parent_hash": meta.parent_hash.astype(str).tolist() if "parent_hash" in meta else [""] * len(hashes),
        "operator_id": meta.operator_id.astype(str).tolist() if "operator_id" in meta else ["p03"] * len(hashes),
    })
    return xy, frame


def run_dir(arm: str, seed: str) -> Path:
    return OUT / "runs" / arm / seed


def shared_dir(seed: str, generation: int, parent_membership: str) -> Path:
    return OUT / "shared" / seed / f"g{generation:02d}" / parent_membership


def extract_f40(coordinates: np.ndarray, hashes: list[str], destination: Path) -> pd.DataFrame:
    return locked(destination.with_name(destination.name + ".lock"), lambda: _extract_f40(coordinates, hashes, destination))


def _extract_f40(coordinates: np.ndarray, hashes: list[str], destination: Path) -> pd.DataFrame:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        cached = pd.read_parquet(destination)
        if cached.instance_hash.astype(str).tolist() == hashes:
            return cached
    cache = OUT / "f40_shards"
    cache.mkdir(parents=True, exist_ok=True)
    db_path = cache / "index.sqlite"
    import sqlite3
    connection = sqlite3.connect(db_path, timeout=180)
    connection.execute("create table if not exists f40(hash text primary key, shard text)")
    known = {}
    for start in range(0, len(hashes), 4000):
        block = hashes[start:start + 4000]
        query = ",".join("?" for _ in block)
        known.update(dict(connection.execute(f"select hash, shard from f40 where hash in ({query})", block).fetchall()))
    missing = [index for index, value in enumerate(hashes) if value not in known]
    if missing:
        rscript = shutil.which("Rscript") or "/home/haozou/micromamba/envs/blade-r/bin/Rscript"
        params = h1.scaler_params()
        jobs = []
        for start in range(0, len(missing), h1.FEATURE_SHARD_N):
            indices = missing[start:start + h1.FEATURE_SHARD_N]
            block = [hashes[index] for index in indices]
            token = hashlib.sha256("\n".join(block).encode()).hexdigest()[:24]
            jobs.append({
                "coordinates": coordinates[np.asarray(indices, int)],
                "hashes": block,
                "names": list(h1.F40),
                "out": str(cache / f"f40_{token}.parquet"),
                "rscript": rscript,
                "r_script": str(h1.R_EXTRACTOR),
                "params": params,
            })
        print(f"[P6-H3] F40 missing={len(missing)} shards={len(jobs)}", flush=True)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=min(FEATURE_WORKERS, len(jobs))) as pool:
            futures = [pool.submit(h1.run_feature_job, job) for job in jobs]
            for done, future in enumerate(as_completed(futures), 1):
                future.result()
                if done == 1 or done % 20 == 0 or done == len(futures):
                    print(f"[P6-H3] F40 shards {done}/{len(futures)}", flush=True)
        with connection:
            for job in jobs:
                for value in job["hashes"]:
                    connection.execute("insert or ignore into f40(hash, shard) values (?, ?)", (value, job["out"]))
                    known[value] = job["out"]
    connection.close()
    shards = sorted({known[value] for value in hashes})
    local = pd.concat([pd.read_parquet(path) for path in shards], ignore_index=True).drop_duplicates("instance_hash", keep="first")
    local = local.set_index(local.instance_hash.astype(str)).loc[hashes].reset_index(drop=True)
    local.to_parquet(destination, index=False)
    return local


def score_pool(features: pd.DataFrame, pool: pd.DataFrame, models: list[Any], config: dict[str, Any], ref_z: np.ndarray, eff: np.ndarray) -> pd.DataFrame:
    scored = pool.reset_index(drop=True).copy()
    started = time.perf_counter()
    scored["hardness"] = h1.predict_hardness(models, features)
    distance, which = h1.min_f40_distance(features[list(h1.F40)].to_numpy(float), config, ref_z, eff)
    scored["distance"] = distance
    scored["nearest_reference_index"] = which
    scored["niche"] = h1.niche_from_features(features)
    scored.attrs["hgb_wall_s"] = time.perf_counter() - started
    return scored


def ref_f40_mins(features: pd.DataFrame, config: dict[str, Any], ref_z: np.ndarray, eff: np.ndarray, ids: list[str], wanted: tuple[str, ...]) -> dict[str, float]:
    out = {}
    matrix = features[list(h1.F40)].to_numpy(float)
    for name in wanted:
        index = ids.index(name)
        distance, _which = h1.min_f40_distance(matrix, config, ref_z[index:index + 1], eff)
        out[name] = float(np.min(distance))
    return out


def evolve_one(arm: str, seed: str) -> None:
    for generation, namespace in NAMESPACES[seed].items():
        p6b.NAMESPACES[seed][generation] = namespace
    spec = ARMS[arm]
    directory = run_dir(arm, seed)
    directory.mkdir(parents=True, exist_ok=True)
    if spec["parents"] == "uniform":
        parent_xy, parents = load_uniform()
    else:
        parent_xy, parents = load_p03()
    refs = load_refs()
    sets = reference_sets(refs)
    rules = json.loads(FFT_RULES.read_text())
    radius = float(rules["radius"])
    ids = refs.instance_id.astype(str).tolist()
    models = h1.hgb_models() if spec["selection"] == "joint" else []
    config = h1.load_distance_config()
    _manifest, ref_values = h1.reference_f40()
    ref_z, eff = h1.standardized_references(config, ref_values)
    g0 = directory / "g00.json"
    if not g0.is_file():
        print(f"[P6-H3] {arm}/{seed} generation 0", flush=True)
        features0 = fft_features(parent_xy)
        dist0 = distance_matrix(features0, refs, rules)
        mask = np.ones(len(parent_xy), bool)
        f40 = extract_f40(parent_xy, parents.instance_hash.astype(str).tolist(), directory / "g00_f40.parquet")
        record = base_record(arm, seed, 0, parents, None, dist0, mask, mask, mask, refs, sets, radius, ids)
        record["f40_min"] = ref_f40_mins(f40, config, ref_z, eff, ids, tuple(HIST15))
        record["effective_lineage"], record["max_lineage_share"] = h1.shannon(parents.lineage_root)
        np.save(directory / "g00_fft.npy", features0)
        nearest0 = {name: parent_xy[int(np.argmin(dist0[:, ids.index(name)]))] for name in HIST15}
        np.savez_compressed(directory / "g00_nearest.npz", **nearest0)
        write_json(g0, record)
    for generation in range(1, GENERATIONS + 1):
        record_path = directory / f"g{generation:02d}.json"
        selected_path = directory / f"g{generation:02d}_selected.npz"
        if record_path.is_file() and selected_path.is_file():
            parent_xy, hashes, _ = p6b.load_bundle(selected_path)
            parents = pd.read_parquet(directory / f"g{generation:02d}_selected.parquet")
            if parents.instance_hash.astype(str).tolist() != hashes:
                raise RuntimeError(f"resume alignment failure {arm}/{seed}/g{generation}")
            print(f"[P6-H3] {arm}/{seed} g{generation} cached", flush=True)
            continue
        parent_hashes = parents.instance_hash.astype(str).tolist()
        parent_membership = membership_hash(parent_hashes)
        mutation = shared_dir(seed, generation, parent_membership)
        print(f"[P6-H3] {arm}/{seed} g{generation} mutate ns={NAMESPACES[seed][generation]}", flush=True)
        started = time.perf_counter()
        lock_path = OUT / "locks" / f"{seed}_g{generation:02d}_{parent_membership}.lock"
        raw_xy, manifest, runtime = locked(lock_path, lambda: mutate_clean(
            seed, generation, parent_xy, parents, mutation))
        child = manifest.loc[manifest.proposal_status.eq("accepted") & manifest.valid_after_repair & ~manifest.within_offspring_duplicate].reset_index(drop=True)
        if child.child_hash.astype(str).tolist() != [coordinate_hash(row) for row in raw_xy]:
            raise RuntimeError("offspring alignment failure")
        fft_path = mutation / "fft_f32.npy"
        if fft_path.is_file() and np.load(fft_path, mmap_mode="r").shape[0] == len(raw_xy):
            features = np.load(fft_path)
        else:
            features = fft_features(raw_xy)
            np.save(fft_path, features)
        dist = distance_matrix(features, refs, rules)
        keep = ~child.child_hash.astype(str).isin(set(parent_hashes)).to_numpy()
        pool_idx = np.flatnonzero(keep)
        if len(pool_idx) < POPULATION_N:
            raise RuntimeError(f"eligible offspring {len(pool_idx)} < 10000")
        roots = dict(zip(parent_hashes, parents.lineage_root.astype(str), strict=True))
        pool = child.iloc[pool_idx].reset_index(drop=True)
        pool["instance_hash"] = pool.child_hash.astype(str)
        pool["lineage_root"] = pool.parent_hash.astype(str).map(roots)
        if pool.lineage_root.isna().any():
            raise RuntimeError("lineage inheritance failure")
        f40_started = time.perf_counter()
        if spec["selection"] == "joint":
            f40 = extract_f40(raw_xy[pool_idx], pool.instance_hash.tolist(), mutation / "f40_pool.parquet")
            scored = score_pool(f40, pool[["instance_hash", "parent_hash", "operator_id", "lineage_root"]], models, config, ref_z, eff)
            selected = h1.select_joint(scored)
            raw_positions = pool_idx[selected.source_position.to_numpy(int)]
            f40_min = ref_f40_mins(f40, config, ref_z, eff, ids, tuple(HIST15))
        else:
            rng = np.random.default_rng(stream_seed(f"P6H3|select|{arm}|{seed}|{generation}"))
            chosen = rng.choice(len(pool), POPULATION_N, replace=False)
            selected = pool.iloc[chosen].copy().reset_index(drop=True)
            selected["source_position"] = chosen
            selected["component"] = "uniform_random"
            selected["selection_rank"] = np.arange(1, POPULATION_N + 1)
            raw_positions = pool_idx[chosen]
            f40 = extract_f40(raw_xy[pool_idx], pool.instance_hash.tolist(), mutation / "f40_pool.parquet")
            f40_min = ref_f40_mins(f40, config, ref_z, eff, ids, tuple(HIST15))
        f40_wall = time.perf_counter() - f40_started
        sample_rng = np.random.default_rng(stream_seed(f"P6H3|rawsample|{arm}|{seed}|{generation}"))
        sample = np.zeros(len(raw_xy), bool)
        sample[sample_rng.choice(len(raw_xy), POPULATION_N, replace=False)] = True
        selected_mask = np.zeros(len(raw_xy), bool)
        selected_mask[raw_positions] = True
        raw_mask = np.ones(len(raw_xy), bool)
        record = base_record(arm, seed, generation, parents, manifest, dist, raw_mask, selected_mask, sample, refs, sets, radius, ids)
        record.update({
            "namespace": NAMESPACES[seed][generation], "parent_membership": parent_membership,
            "offspring_membership": runtime["offspring_membership_sha256"], "attempts": int(runtime["nominal_attempts"]),
            "valid": int(runtime["valid_attempts"]), "invalid": int(runtime["invalid_attempts"]),
            "unique": int(runtime["valid_unique_within_generation"]), "failure_rate": float(runtime["invalid_attempts"]) / float(runtime["nominal_attempts"]),
            "mutation_wall_s": float(runtime["wall_seconds"]), "wrapper_wall_s": time.perf_counter() - started,
            "f40_wall_s": f40_wall, "mutation_dir": str(mutation), "f40_min": f40_min,
        })
        record["effective_lineage"], record["max_lineage_share"] = h1.shannon(selected.lineage_root)
        if "niche" in selected:
            counts = selected.niche.astype(int).value_counts(normalize=True).to_numpy(float)
            record["niche_occupancy"] = int(selected.niche.nunique())
            record["niche_hhi"] = float(np.sum(counts ** 2))
        parent_fft_path = directory / f"g{generation - 1:02d}_fft.npy" if generation > 1 else directory / "g00_fft.npy"
        if generation == 1:
            parent_fft_path = directory / "g00_fft.npy"
        else:
            parent_fft_path = directory / f"g{generation - 1:02d}_selected_fft.npy"
        if parent_fft_path.is_file():
            parent_feat = np.load(parent_fft_path)
            parent_dist = distance_matrix(parent_feat, refs, rules)
            record["operator_shift"] = operator_shift(child, dist, parent_dist, ids)
        selected_xy = raw_xy[raw_positions]
        selected_hashes = selected.instance_hash.astype(str).tolist()
        if [coordinate_hash(row) for row in selected_xy] != selected_hashes:
            raise RuntimeError("selected hash mismatch")
        p6b.save_bundle(selected_path, selected_xy, selected_hashes)
        selected.to_parquet(directory / f"g{generation:02d}_selected.parquet", index=False)
        np.save(directory / f"g{generation:02d}_selected_fft.npy", features[raw_positions])
        marks = child[["child_hash", "parent_hash", "operator_id", "attempt_index"]].copy()
        marks["selected"] = selected_mask
        marks.to_parquet(directory / f"g{generation:02d}_selection_marks.parquet", index=False)
        # Keep the nearest RAW coordinate for the historical references.
        nearest = {}
        for name in HIST15:
            column = ids.index(name)
            index = int(np.argmin(dist[:, column]))
            nearest[name] = raw_xy[index]
        np.savez_compressed(directory / f"g{generation:02d}_nearest.npz", **nearest)
        write_json(record_path, record)
        parent_xy, parents = selected_xy, selected
        del raw_xy, dist, features, manifest
        print(f"[P6-H3] {arm}/{seed} g{generation} done unique={record['unique']} raw11={record['coverage']['raw_dual11']}", flush=True)


def mutate_clean(seed: str, generation: int, parent_xy: np.ndarray, parents: pd.DataFrame, directory: Path) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    bundle, manifest, runtime = p6b.mutation_paths(directory)
    if (bundle.is_file() or manifest.is_file()) and not runtime.is_file():
        shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
    return p6b.mutate(seed, f"shared_g{generation}", generation, parent_xy, parents[["instance_hash"]], directory)


def base_record(arm: str, seed: str, generation: int, parents: pd.DataFrame, manifest: pd.DataFrame | None,
                dist: np.ndarray, raw_mask: np.ndarray, selected_mask: np.ndarray, sample_mask: np.ndarray,
                refs: pd.DataFrame, sets: dict[str, list[str]], radius: float, ids: list[str]) -> dict[str, Any]:
    references = {}
    for name in ids:
        column = ids.index(name)
        raw_col = dist[raw_mask, column]
        sel_col = dist[selected_mask, column]
        sample_col = dist[sample_mask, column]
        nearest = int(np.argmin(dist[:, column]))
        references[name] = {
            "raw_min": float(raw_col.min()), "raw_covered": bool(raw_col.min() <= radius), "raw_count": int(np.sum(raw_col <= radius)),
            "selected_min": float(sel_col.min()) if selected_mask.any() else None,
            "selected_covered": bool(selected_mask.any() and sel_col.min() <= radius),
            "sample_covered": bool(sample_mask.any() and sample_col.min() <= radius),
            "nearest_index": nearest, "nearest_distance": float(dist[nearest, column]),
        }
    if manifest is not None:
        child = manifest.loc[manifest.proposal_status.eq("accepted") & manifest.valid_after_repair & ~manifest.within_offspring_duplicate].reset_index(drop=True)
        for name in HIST15:
            references[name]["nearest_operator"] = str(child.operator_id.iloc[references[name]["nearest_index"]])
            references[name]["nearest_hash"] = str(child.child_hash.iloc[references[name]["nearest_index"]])
            references[name]["nearest_parent"] = str(child.parent_hash.iloc[references[name]["nearest_index"]])
    return {
        "status": "PASS", "arm": arm, "seed": seed, "generation": generation, "finished_utc": now(),
        "references": references,
        "coverage": {
            "raw_all48": coverage_counts(dist, raw_mask, {"all48": sets["all48"]}, ids, radius)["all48"],
            "raw_size17": coverage_counts(dist, raw_mask, {"size17": sets["size17"]}, ids, radius)["size17"],
            "raw_hist15": coverage_counts(dist, raw_mask, {"hist15": sets["hist15"]}, ids, radius)["hist15"],
            "raw_dual11": coverage_counts(dist, raw_mask, {"dual11": sets["dual11"]}, ids, radius)["dual11"],
            "selected_all48": coverage_counts(dist, selected_mask, {"all48": sets["all48"]}, ids, radius)["all48"],
            "selected_dual11": coverage_counts(dist, selected_mask, {"dual11": sets["dual11"]}, ids, radius)["dual11"],
            "selected_hist15": coverage_counts(dist, selected_mask, {"hist15": sets["hist15"]}, ids, radius)["hist15"],
            "sample_all48": coverage_counts(dist, sample_mask, {"all48": sets["all48"]}, ids, radius)["all48"],
            "sample_dual11": coverage_counts(dist, sample_mask, {"dual11": sets["dual11"]}, ids, radius)["dual11"],
        },
    }


def operator_shift(child: pd.DataFrame, dist: np.ndarray, parent_dist: np.ndarray, ids: list[str]) -> dict[str, Any]:
    out = {}
    parent_index = child.parent_index.to_numpy(int) if "parent_index" in child else None
    if parent_index is None:
        return out
    for name in ("pr107", "pr144"):
        column = ids.index(name)
        child_d = dist[:, column]
        parent_d = parent_dist[parent_index, column]
        improved = parent_d - child_d
        for operator, index in child.groupby("operator_id").groups.items():
            idx = np.asarray(list(index), int)
            bucket = out.setdefault(str(operator), {})
            bucket[f"{name}_n"] = int(len(idx))
            bucket[f"{name}_median_improvement"] = float(np.median(improved[idx]))
            bucket[f"{name}_p90_improvement"] = float(np.quantile(improved[idx], 0.9))
            bucket[f"{name}_improved_fraction"] = float(np.mean(improved[idx] > 0))
            bucket[f"{name}_best_child"] = float(np.min(child_d[idx]))
    return out


def load_records() -> list[dict[str, Any]]:
    records = []
    for arm in ARMS:
        for seed in SEEDS:
            for generation in range(0, GENERATIONS + 1):
                path = run_dir(arm, seed) / f"g{generation:02d}.json"
                if not path.is_file():
                    raise RuntimeError(f"missing generation record {path}")
                records.append(json.loads(path.read_text()))
    return records


def cumulative(flags: list[bool]) -> list[int]:
    seen = 0
    out = []
    running = False
    for flag in flags:
        running = running or flag
        seen += int(running and (len(out) == 0 or out[-1] != seen + 1 or True))
        # count of distinct refs is handled outside; this helper is unused for sets
        out.append(int(running))
    return out


def analyze() -> dict[str, Any]:
    records = load_records()
    frame = pd.json_normalize(records, sep="_")
    manifest_rows = []
    coverage_rows = []
    hit_rows = []
    operator_rows = []
    for record in records:
        manifest_rows.append({
            "arm": record["arm"], "seed": record["seed"], "generation": record["generation"],
            "namespace": record.get("namespace"), "attempts": record.get("attempts", 0),
            "valid": record.get("valid", len(record.get("references", {})) and None),
            "invalid": record.get("invalid"), "unique": record.get("unique"),
            "failure_rate": record.get("failure_rate"), "mutation_wall_s": record.get("mutation_wall_s"),
            "f40_wall_s": record.get("f40_wall_s"), "effective_lineage": record.get("effective_lineage"),
            "max_lineage_share": record.get("max_lineage_share"), "parent_membership": record.get("parent_membership"),
            "offspring_membership": record.get("offspring_membership"),
        })
        coverage = dict(record["coverage"])
        coverage.update({"arm": record["arm"], "seed": record["seed"], "generation": record["generation"],
                         "pr107_raw_min": record["references"]["pr107"]["raw_min"],
                         "pr144_raw_min": record["references"]["pr144"]["raw_min"],
                         "effective_lineage": record.get("effective_lineage"),
                         "max_lineage_share": record.get("max_lineage_share")})
        for name in HIST15:
            coverage[f"f40_{name}"] = (record.get("f40_min") or {}).get(name)
        coverage_rows.append(coverage)
        for operator, stats in (record.get("operator_shift") or {}).items():
            operator_rows.append({"arm": record["arm"], "seed": record["seed"], "generation": record["generation"], "operator_id": operator, **stats})
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        by_key.setdefault((record["arm"], record["seed"]), []).append(record)
    for key, series in by_key.items():
        series.sort(key=lambda item: item["generation"])
        seen_raw: dict[str, int] = {}
        seen_selected: dict[str, int] = {}
        for record in series:
            for name in HIST15:
                ref = record["references"][name]
                if ref["raw_covered"] and name not in seen_raw:
                    seen_raw[name] = int(record["generation"])
                if ref["selected_covered"] and name not in seen_selected:
                    seen_selected[name] = int(record["generation"])
        for name in HIST15:
            first = seen_raw.get(name)
            row = {"arm": key[0], "seed": key[1], "reference_id": name, "dual11": name in DUAL11,
                   "first_hit_generation": first, "cumulative_raw_reached": first is not None,
                   "cumulative_selected_reached": name in seen_selected,
                   "selected_retained_at_first_hit": False, "next_generation_reached": False,
                   "selection_loss": False}
            if first is not None:
                hit = next(item for item in series if item["generation"] == first)
                row["selected_retained_at_first_hit"] = bool(hit["references"][name]["selected_covered"])
                row["nearest_operator"] = hit["references"][name].get("nearest_operator")
                row["nearest_hash"] = hit["references"][name].get("nearest_hash")
                row["nearest_distance"] = hit["references"][name]["nearest_distance"]
                nxt = next((item for item in series if item["generation"] == first + 1), None)
                row["next_generation_reached"] = bool(nxt and nxt["references"][name]["raw_covered"])
                row["selection_loss"] = bool(hit["references"][name]["raw_covered"] and not hit["references"][name]["selected_covered"])
            hit_rows.append(row)
    cov = pd.DataFrame(coverage_rows)
    cum_rows = []
    for (arm, seed), series in by_key.items():
        raw_seen: set[str] = set()
        sel_seen: set[str] = set()
        for record in series:
            for name in HIST15:
                if record["references"][name]["raw_covered"]:
                    raw_seen.add(name)
                if record["references"][name]["selected_covered"]:
                    sel_seen.add(name)
            dual_raw = len(raw_seen & set(DUAL11))
            dual_sel = len(sel_seen & set(DUAL11))
            cum_rows.append({"arm": arm, "seed": seed, "generation": record["generation"],
                             "cumulative_raw_hist15": len(raw_seen), "cumulative_selected_hist15": len(sel_seen),
                             "cumulative_raw_dual11": dual_raw, "cumulative_selected_dual11": dual_sel})
    cum = pd.DataFrame(cum_rows)
    cov = cov.merge(cum, on=["arm", "seed", "generation"], validate="one_to_one")
    pd.DataFrame(manifest_rows).to_csv(OUT / "02_generation_manifest.csv", index=False)
    cov.to_csv(OUT / "03_generation_coverage.csv", index=False)
    hits = pd.DataFrame(hit_rows)
    hits.to_csv(OUT / "04_first_hit_and_selection_loss.csv", index=False)
    pd.DataFrame(operator_rows).to_csv(OUT / "05_operator_reachability.csv", index=False)
    geometry = geometry_table(by_key)
    geometry.to_csv(OUT / "06_f40_geometry_validation.csv", index=False)
    labels = decide(cov, hits, geometry)
    write_figures(cov)
    write_reports(cov, hits, geometry, labels)
    write_json(OUT / "completion.json", {**labels, "finished_utc": now(), "tag": "BLADE_P6H3_TEN_GENERATION_REACHABILITY_AUDIT_COMPLETE"})
    print(json.dumps(labels, indent=2), flush=True)
    return labels


def geometry_table(by_key: dict[tuple[str, str], list[dict[str, Any]]]) -> pd.DataFrame:
    import run_p6g2_analysis as g2
    refs = load_refs()
    ref_xy = {row.instance_id: g2.parse_tsplib(ROOT / row.source_file) for row in refs.itertuples(index=False)}
    historical = pd.read_csv(G2_VALIDATION).set_index("reference_id")
    rows = []
    figure_dir = OUT / "figures" / "geometry"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for (arm, seed), series in by_key.items():
        for name in ("pr107", "pr144", *DUAL11):
            generations = {10}
            for record in series:
                if record["references"][name]["raw_covered"]:
                    generations.add(int(record["generation"]))
                    break
            candidates = []
            for generation in sorted(generations):
                path = run_dir(arm, seed) / f"g{generation:02d}_nearest.npz"
                if not path.is_file():
                    continue
                with np.load(path) as payload:
                    if name not in payload.files:
                        continue
                    xy = np.asarray(payload[name]).copy()
                record = next(item for item in series if item["generation"] == generation)
                ref = record["references"][name]
                if not ref["raw_covered"] and name not in ("pr107", "pr144"):
                    continue
                compared = g2.geometry_compare(ref_xy[name], xy)
                hist = historical.loc[name]
                row = {"arm": arm, "seed": seed, "generation": generation, "reference_id": name,
                       "fft_distance": ref["nearest_distance"], "fft_covered": ref["raw_covered"],
                       "f40_min": (record.get("f40_min") or {}).get(name),
                       "operator_id": ref.get("nearest_operator"), "candidate_hash": ref.get("nearest_hash"),
                       "better_than_foga": float(compared["geometry_composite"]) <= float(hist.FOGA_RAW_best_geometry),
                       "better_than_p03": float(compared["geometry_composite"]) <= float(hist.P03_best_geometry),
                       **compared}
                rows.append(row)
                candidates.append((generation, xy, row))
            if candidates and (name in ("pr107", "pr144") or any(item[2]["fft_covered"] for item in candidates)):
                plot_geometry(figure_dir / f"{arm}_{seed}_{name}.png", ref_xy[name], candidates[-1][1], f"{arm}/{seed} {name}")
    return pd.DataFrame(rows)


def plot_geometry(path: Path, reference: np.ndarray, candidate: np.ndarray, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import run_p6g2_analysis as g2
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for axis, xy, label in ((axes[0], reference, "TSPLIB"), (axes[1], candidate, "nearest offspring")):
        z = g2.shape_normalize(xy)
        axis.scatter(z[:, 0], z[:, 1], s=8, c="#1f4e79")
        axis.set_aspect("equal")
        axis.set_title(label)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def median_of(values: list[float]) -> float:
    return float(np.median(np.asarray(values, float)))


def decide(cov: pd.DataFrame, hits: pd.DataFrame, geometry: pd.DataFrame) -> dict[str, str]:
    def gain(arm: str) -> float:
        values = []
        for seed in SEEDS:
            part = cov[(cov.arm.eq(arm)) & (cov.seed.eq(seed))].sort_values("generation")
            values.append(float(part.iloc[-1].cumulative_raw_dual11 - part.iloc[0].cumulative_raw_dual11))
        return median_of(values)

    def g10(arm: str, column: str) -> float:
        values = [float(cov[(cov.arm.eq(arm)) & (cov.seed.eq(seed)) & (cov.generation.eq(10))][column].iloc[0]) for seed in SEEDS]
        return median_of(values)

    uniform_gain = max(gain("A"), gain("B"))
    p03_gain = max(gain("C"), gain("D"))
    def reach(value: float) -> str:
        if value >= 3:
            return "EXPANDED"
        if value > 0:
            return "LIMITED"
        return "NONE"
    initial_random = g10("A", "cumulative_raw_dual11") - g10("C", "cumulative_raw_dual11")
    initial_joint = g10("B", "cumulative_raw_dual11") - g10("D", "cumulative_raw_dual11")
    if initial_random >= 2 and initial_joint >= 2:
        initial = "SUPPORTED"
    elif initial_random >= 2 or initial_joint >= 2:
        initial = "PARTIAL"
    else:
        initial = "NOT_SUPPORTED"
    selection_gaps = [g10("A", "cumulative_raw_dual11") - g10("B", "cumulative_raw_dual11"),
                      g10("C", "cumulative_raw_dual11") - g10("D", "cumulative_raw_dual11")]
    loss_arms = 0
    loss_partial = 0
    for arm in ARMS:
        gap = g10(arm, "cumulative_raw_dual11") - g10(arm, "cumulative_selected_dual11")
        loss_arms += int(gap >= 2)
        loss_partial += int(gap >= 1)
    if any(gap >= 2 for gap in selection_gaps) or loss_arms >= 2:
        selection = "SUPPORTED"
    elif any(gap >= 1 for gap in selection_gaps) or loss_partial >= 2:
        selection = "PARTIAL"
    else:
        selection = "NOT_SUPPORTED"
    outside = True
    for arm in ARMS:
        for seed in SEEDS:
            row = cov[(cov.arm.eq(arm)) & (cov.seed.eq(seed)) & (cov.generation.eq(10))].iloc[0]
            outside &= float(row.pr107_raw_min) > 0.1770575219387968 and float(row.pr144_raw_min) > 0.1770575219387968
    gains = [gain(arm) for arm in ARMS]
    if all(value <= 1 for value in gains) and outside:
        mutation = "SUPPORTED_WITHIN_BUDGET"
    elif any(value >= 3 for value in gains):
        mutation = "NOT_SUPPORTED"
    else:
        mutation = "INCONCLUSIVE"
    depth_arms = 0
    for arm in ARMS:
        improved = 0
        for seed in SEEDS:
            part = cov[(cov.arm.eq(arm)) & (cov.seed.eq(seed))].set_index("generation")
            cover_up = float(part.loc[10, "cumulative_raw_dual11"]) > float(part.loc[2, "cumulative_raw_dual11"])
            closer = min(float(part.loc[10, "pr107_raw_min"]), float(part.loc[10, "pr144_raw_min"])) < 0.95 * min(float(part.loc[2, "pr107_raw_min"]), float(part.loc[2, "pr144_raw_min"]))
            improved += int(cover_up or closer)
        depth_arms += int(improved >= 2)
    depth = "SUPPORTED" if depth_arms >= 2 else "PARTIAL" if depth_arms == 1 else "NOT_SUPPORTED"
    covered = geometry[geometry.fft_covered.astype(bool) & geometry.reference_id.isin(DUAL11)] if len(geometry) else geometry
    if len(covered):
        good = covered[covered.better_than_foga.astype(bool) & covered.better_than_p03.astype(bool)]
        if len(good) >= 3 and len(good) >= 0.5 * len(covered):
            geometric = "SUPPORTED"
        elif len(good) >= 1:
            geometric = "PARTIAL"
        else:
            geometric = "NOT_SUPPORTED"
    else:
        geometric = "NOT_SUPPORTED"
    flags = [("INITIAL_SUPPORT", initial == "SUPPORTED"), ("SELECTION", selection == "SUPPORTED"),
             ("MUTATION_TRANSITIONS", mutation == "SUPPORTED_WITHIN_BUDGET")]
    active = [name for name, value in flags if value]
    primary = active[0] if len(active) == 1 else "MIXED" if len(active) > 1 else "NOT_IDENTIFIABLE"
    return {
        "P6_H3_UNIFORM_10G_REACHABILITY": reach(uniform_gain),
        "P6_H3_P03_10G_REACHABILITY": reach(p03_gain),
        "P6_H3_INITIAL_SUPPORT_EFFECT": initial,
        "P6_H3_SELECTION_BOTTLENECK": selection,
        "P6_H3_MUTATION_REACHABILITY_LIMIT": mutation,
        "P6_H3_DEPTH_EXPANSION": depth,
        "P6_H3_GEOMETRIC_VALIDITY": geometric,
        "P6_H3_PRIMARY_BOTTLENECK": primary,
        "P6_H3_STATUS": "COMPLETE",
    }


def write_figures(cov: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = OUT / "figures"
    fig_dir.mkdir(exist_ok=True)
    colors = {"A": "#1f4e79", "B": "#c45911", "C": "#548235", "D": "#833c0c"}
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for arm, part in cov.groupby("arm"):
        mean = part.groupby("generation").cumulative_raw_dual11.mean()
        axis.plot(mean.index, mean.values, label=arm, color=colors[arm], lw=2)
        for seed, sub in part.groupby("seed"):
            axis.plot(sub.generation, sub.cumulative_raw_dual11, color=colors[arm], alpha=0.25, lw=1)
    axis.set_xlabel("generation")
    axis.set_ylabel("cumulative RAW coverage of 11 dual-support refs")
    axis.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "01_cumulative_dual11.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, column, title in ((axes[0], "pr107_raw_min", "pr107"), (axes[1], "pr144_raw_min", "pr144")):
        for arm, part in cov.groupby("arm"):
            mean = part.groupby("generation")[column].mean()
            axis.plot(mean.index, mean.values, label=arm, color=colors[arm], lw=2)
        axis.axhline(0.1770575219387968, color="black", ls="--", lw=1)
        axis.set_title(title)
        axis.set_xlabel("generation")
        axis.set_ylabel("min FFT–NN distance")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "02_pr107_pr144_distance.png", dpi=140)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for arm, part in cov.groupby("arm"):
        raw = part.groupby("generation").cumulative_raw_dual11.mean()
        sel = part.groupby("generation").cumulative_selected_dual11.mean()
        axis.plot(raw.index, raw.values, color=colors[arm], lw=2, label=f"{arm} RAW")
        axis.plot(sel.index, sel.values, color=colors[arm], lw=1.5, ls="--", label=f"{arm} selected")
    axis.set_xlabel("generation")
    axis.set_ylabel("cumulative dual-support coverage")
    axis.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "03_raw_vs_selected.png", dpi=140)
    plt.close(fig)
    ops = pd.read_csv(OUT / "05_operator_reachability.csv") if (OUT / "05_operator_reachability.csv").is_file() else pd.DataFrame()
    fig, axis = plt.subplots(figsize=(9, 4.5))
    if len(ops) and "pr107_best_child" in ops:
        summary = ops.groupby("operator_id").pr107_best_child.min().sort_values()
        axis.barh(summary.index, summary.values, color="#1f4e79")
        axis.axvline(0.1770575219387968, color="black", ls="--", lw=1)
        axis.set_xlabel("best observed pr107 FFT–NN distance")
    fig.tight_layout()
    fig.savefig(fig_dir / "04_operator_pr107.png", dpi=140)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for arm, part in cov.groupby("arm"):
        mean = part.groupby("generation").effective_lineage.mean()
        axis.plot(mean.index, mean.values, label=arm, color=colors[arm], lw=2)
    axis.set_xlabel("generation")
    axis.set_ylabel("effective lineage of selected 10k")
    axis.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "05_effective_lineage.png", dpi=140)
    plt.close(fig)


def write_reports(cov: pd.DataFrame, hits: pd.DataFrame, geometry: pd.DataFrame, labels: dict[str, str]) -> None:
    lines = ["# P6-H3 Parent Selection Comparison", "", "数字是三个 seed 的中位数。累计覆盖是从 G0 到该轮、RAW 或被选父代曾经进入冻结 FFT 半径的双重支持 reference 数。", "", "| arm | G0 dual11 | G2 dual11 | G10 dual11 | G10 selected | pr107 G0 | pr107 G10 | pr144 G0 | pr144 G10 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        def med(generation: int, column: str) -> float:
            return float(np.median(cov[(cov.arm.eq(arm)) & (cov.generation.eq(generation))][column]))
        lines.append(f"| {arm} | {med(0, 'cumulative_raw_dual11'):.1f} | {med(2, 'cumulative_raw_dual11'):.1f} | {med(10, 'cumulative_raw_dual11'):.1f} | {med(10, 'cumulative_selected_dual11'):.1f} | {med(0, 'pr107_raw_min'):.4f} | {med(10, 'pr107_raw_min'):.4f} | {med(0, 'pr144_raw_min'):.4f} | {med(10, 'pr144_raw_min'):.4f} |")
    lost = hits[hits.selection_loss.fillna(False).astype(bool)] if "selection_loss" in hits else hits.iloc[0:0]
    lines.append("")
    lines.append(f"首次 RAW 命中后当轮没有留在 10k 父代里的 (arm, seed, reference) 记录有 {len(lost)} 条。明细在 `04_first_hit_and_selection_loss.csv`。")
    (OUT / "07_parent_selection_comparison.md").write_text("\n".join(lines) + "\n")
    body = f"""# P6-H3 Final Conclusion

## 判定

`P6_H3_UNIFORM_10G_REACHABILITY = {labels['P6_H3_UNIFORM_10G_REACHABILITY']}`

`P6_H3_P03_10G_REACHABILITY = {labels['P6_H3_P03_10G_REACHABILITY']}`

`P6_H3_INITIAL_SUPPORT_EFFECT = {labels['P6_H3_INITIAL_SUPPORT_EFFECT']}`

`P6_H3_SELECTION_BOTTLENECK = {labels['P6_H3_SELECTION_BOTTLENECK']}`

`P6_H3_MUTATION_REACHABILITY_LIMIT = {labels['P6_H3_MUTATION_REACHABILITY_LIMIT']}`

`P6_H3_DEPTH_EXPANSION = {labels['P6_H3_DEPTH_EXPANSION']}`

`P6_H3_GEOMETRIC_VALIDITY = {labels['P6_H3_GEOMETRIC_VALIDITY']}`

`P6_H3_PRIMARY_BOTTLENECK = {labels['P6_H3_PRIMARY_BOTTLENECK']}`

`P6_H3_STATUS = {labels['P6_H3_STATUS']}`

`BLADE_P6H3_TEN_GENERATION_REACHABILITY_AUDIT_COMPLETE`

## 怎么读

主瓶颈标签是 `{labels['P6_H3_PRIMARY_BOTTLENECK']}`。Uniform 十轮可达性是 `{labels['P6_H3_UNIFORM_10G_REACHABILITY']}`，P03 是 `{labels['P6_H3_P03_10G_REACHABILITY']}`。初始父代效应 `{labels['P6_H3_INITIAL_SUPPORT_EFFECT']}`，选择瓶颈 `{labels['P6_H3_SELECTION_BOTTLENECK']}`，预算内变异限制 `{labels['P6_H3_MUTATION_REACHABILITY_LIMIT']}`。深度相对历史两轮的继续变化是 `{labels['P6_H3_DEPTH_EXPANSION']}`。FFT 命中的几何确认是 `{labels['P6_H3_GEOMETRIC_VALIDITY']}`。

没有进入冻结半径只说明这 10 轮、每轮 11 算子、10k 父代宽度下没有观察到到达，不能说 FOGA-11 在数学上生成不了这些结构。FFT 命中如果没有 F40 和几何同时支持，也不算结构重现。几何表有 {len(geometry)} 行。
"""
    (OUT / "08_final_conclusion.md").write_text(body)


def supervise() -> None:
    pending = [(arm, seed) for arm in ARMS for seed in SEEDS]
    running: list[tuple[subprocess.Popen, str, str, Any]] = []
    failures = []
    while pending or running:
        while pending and len(running) < MAX_PARALLEL:
            arm, seed = pending.pop(0)
            log_path = OUT / "logs" / f"{arm}_{seed}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a")
            proc = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--phase", "evolve", "--arm", arm, "--seed", seed],
                                    cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT)
            running.append((proc, arm, seed, handle))
            print(f"[P6-H3] start {arm}/{seed} pid={proc.pid}", flush=True)
        time.sleep(15)
        alive = []
        for proc, arm, seed, handle in running:
            code = proc.poll()
            if code is None:
                alive.append((proc, arm, seed, handle))
            else:
                handle.close()
                print(f"[P6-H3] exit {arm}/{seed} code={code}", flush=True)
                if code != 0:
                    failures.append(f"{arm}/{seed} exit {code}")
        running = alive
    if failures:
        raise RuntimeError("evolution failures: " + "; ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "evolve", "analyze", "all"), default="all")
    parser.add_argument("--arm", choices=(*ARMS, "all"), default="all")
    parser.add_argument("--seed", choices=(*SEEDS, "all"), default="all")
    args = parser.parse_args()
    if args.phase in ("preflight", "all"):
        preflight()
    if args.phase == "evolve":
        if args.arm == "all" or args.seed == "all":
            raise RuntimeError("evolve requires one arm and one seed")
        evolve_one(args.arm, args.seed)
    elif args.phase == "all":
        supervise()
        analyze()
    elif args.phase == "analyze":
        analyze()


if __name__ == "__main__":
    main()
