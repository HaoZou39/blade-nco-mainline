#!/usr/bin/env python3
"""P6-H1 hardness–structure joint evolution from formal M4 and P03.

The selection rule, distance formula, namespaces and budgets below are fixed
before any new population is scored. Historical E0/E1 survivor sets are never
mutation parents or candidate pools.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "scripts"),
    str(ROOT / "target_aware_pilot/p6b_top5_distribution_guided_evolution"),
    str(ROOT / "target_aware_pilot/p2_bounded_evolution"),
    str(ROOT / "target_aware_pilot/p5_structure_guided_evolution_robustness"),
    str(ROOT / "target_aware_pilot/p6a_top5_target_matching_feasibility"),
    str(ROOT / "target_aware_pilot/p6_1_direct_population_training"),
    str(ROOT / "target_aware_pilot/p4b1_acquisition_ablation"),
    str(ROOT / "target_aware_pilot/p4c2_data_sampling_robustness"),
]

import run_p6b as p6b  # noqa: E402
from features.qd4 import QD4_FEATURE_NAMES, extract_qd4_exact  # noqa: E402
from final_run.science import label_with_cache, membership_hash  # noqa: E402
from mutation.result import coordinate_hash  # noqa: E402
from oracle.concorde import CANONICAL_CONCORDE_SHA256, ORACLE_LEDGER_COLUMNS  # noqa: E402
from qd.coordinates import assign_nearest, normalize_qd4  # noqa: E402

REPLICATES = ("Original", "S1", "S2")
DEPTHS = ("1G", "2G")
POPULATION_N = 10_000
BLOCK_ATTEMPTS = 110_000
TRAINING_SEED = 2026
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20261029
FEATURE_SHARD_N = 250
FEATURE_WORKERS = 48
MIN_F40_DIMS = 30
# Frozen before any H1 fitness was observed. cycle_child_seed has no attempt
# index, so a second 110k block and the second generation need distinct
# namespaces. Block 1 keeps the official P6-B generation-1 namespaces.
BLOCK2_NS = {"Original": 25, "S1": 2605015, "S2": 2605025}
G2_NS = {"Original": 26, "S1": 2605016, "S2": 2605026}
BLOCK2_KEY = 11
G2_KEY = 12
for _rep in REPLICATES:
    p6b.NAMESPACES[_rep][BLOCK2_KEY] = BLOCK2_NS[_rep]
    p6b.NAMESPACES[_rep][G2_KEY] = G2_NS[_rep]

P6B = ROOT / "target_aware_pilot/p6b_top5_distribution_guided_evolution"
P6G2 = ROOT / "target_aware_pilot/p6g2_structural_similarity_validation"
P6G1 = ROOT / "target_aware_pilot/p6g1_structural_reachability_analysis"
F40_MANIFEST = ROOT / "audits/S3_B2_RF_HGB_feature_label_scaling_v1/09_feature_subset_manifests.json"
FORMAL_F40 = P6G1 / "cache/formal_f40_all.parquet"
DISTANCE_CONFIG = P6G2 / "manifests/f40_distance_config.json"
REF_MANIFEST = ROOT / "target_aware_pilot/03_tsplib48_reference_manifest.csv"
REF_F40 = ROOT / "target_aware_pilot/p1_tsplib48_normalization_sensitivity.csv"
HELD22 = ROOT / "target_aware_pilot/p1_heldout22_names.json"
FFT_RULES = ROOT / "target_aware_pilot/p6e0_evoreal_structural_representation_audit/configs/reference_rules_frozen.json"
HGB_DIR = ROOT / "runs/BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1/A0_FULL_BASE/cycles/C04/HGB_seed_models"
M4_EVAL = ROOT / "runs/BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1/A0_FULL_BASE/cycles/C03/evaluation_work/M4"
SCALER = ROOT / "assets/qd/reference_cloud/formal_raw_scaler.json"
WORKER = ROOT / "scripts/f40_one_shard.py"
R_EXTRACTOR = ROOT / "src/features/extract_salesperson_raw_features.R"
F40 = tuple(json.loads(F40_MANIFEST.read_text())["F40"])
QD4 = tuple(QD4_FEATURE_NAMES)
HGB_SEEDS = (2026, 2027, 2028, 2029, 2030)
M4_SHA = "c749311bd2a461084d0cbcfd383a4c0a1fc1268762b72e5f7220952f108f8190"
P03_BUNDLE_SHA = "8ad76b90c08997bba33da1be6bf3dbfa8c28ee05bbb4a5fe4d1a9ecdb699dd62"
HISTORICAL_G1_OFFSPRING = {
    "Original": "3bc43bd97caa6778d6abc5e700e6594bfd3412ac6f7ae0662d9a02be69f25c20",
    "S1": "9f05708303da3fabedbf8211dcf4cf5a7cc5ced5561b410177a5abfa8a59d28d",
    "S2": "72ea147ad0567f91aa2845633abe7eda40c4a7f03227329c4f70d08ee09b249b",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_convert) + "\n")
    os.replace(temporary, path)


def _convert(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hash(functions: list[Any]) -> str:
    payload = "\n".join(inspect.getsource(fn) for fn in functions)
    return hashlib.sha256(payload.encode()).hexdigest()


class Status:
    def update(self, **values: Any) -> None:
        current = {}
        path = OUT / "status.json"
        if path.is_file():
            current = json.loads(path.read_text())
        current.update(values)
        current["timestamp_utc"] = now()
        write_json(path, current)


def ensure_dirs() -> None:
    for name in ("configs", "evolution", "populations", "features", "labels", "checkpoints", "logs", "datasets", "manifests"):
        (OUT / name).mkdir(parents=True, exist_ok=True)


class FenwickMax:
    def __init__(self, n: int) -> None:
        self.n = int(n)
        self.tree = [0] * (self.n + 1)

    def update(self, index: int, value: int) -> None:
        i = int(index)
        while i <= self.n:
            if value > self.tree[i]:
                self.tree[i] = value
            else:
                return
            i += i & -i

    def query(self, index: int) -> int:
        i = int(index)
        best = 0
        while i > 0:
            if self.tree[i] > best:
                best = self.tree[i]
            i -= i & -i
        return best


def pareto_ranks(hardness: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """2-objective NSGA-II rank. Higher hardness and lower distance dominate.

    Equal objective pairs do not dominate each other. Non-finite distance is
    worse than every finite distance. Rank equals one plus the maximum rank
    among dominating points.
    """
    hardness = np.asarray(hardness, np.float64)
    distance = np.asarray(distance, np.float64)
    n = len(hardness)
    if not np.isfinite(hardness).all():
        raise RuntimeError("hardness contains non-finite values")
    finite = np.isfinite(distance)
    distance = np.where(finite, distance, np.inf)
    finite_values = np.unique(distance[finite]) if finite.any() else np.empty(0, np.float64)
    lookup = {float(value): i + 1 for i, value in enumerate(finite_values)}
    inf_code = len(finite_values) + 1
    codes = np.fromiter((lookup[float(value)] if np.isfinite(value) else inf_code for value in distance), np.int32, n)
    tree = FenwickMax(inf_code)
    order = np.lexsort((codes, -hardness))
    ranks = np.empty(n, np.int32)
    cursor = 0
    while cursor < n:
        stop = cursor + 1
        h0 = hardness[order[cursor]]
        c0 = int(codes[order[cursor]])
        while stop < n and hardness[order[stop]] == h0 and int(codes[order[stop]]) == c0:
            stop += 1
        rank = tree.query(c0) + 1
        ranks[order[cursor:stop]] = rank
        tree.update(c0, int(rank))
        cursor = stop
    return ranks


def crowding_distance(hardness: np.ndarray, distance: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    hardness = np.asarray(hardness, np.float64)
    distance = np.where(np.isfinite(distance), distance, np.inf)
    crowd = np.zeros(len(hardness), np.float64)
    for rank in np.unique(ranks):
        index = np.flatnonzero(ranks == rank)
        if len(index) <= 2:
            crowd[index] = np.inf
            continue
        for values, sign in ((hardness, 1.0), (distance, -1.0)):
            objective = sign * values[index]
            order = np.argsort(objective, kind="mergesort")
            crowd[index[order[0]]] = np.inf
            crowd[index[order[-1]]] = np.inf
            span = float(objective[order[-1]] - objective[order[0]])
            if not np.isfinite(span) or span <= 1e-15:
                continue
            for position in range(1, len(order) - 1):
                slot = index[order[position]]
                if np.isfinite(crowd[slot]):
                    crowd[slot] += (objective[order[position + 1]] - objective[order[position - 1]]) / span
    return crowd


def select_joint(frame: pd.DataFrame) -> pd.DataFrame:
    """One Pareto representative per occupied QD niche, then global fill.

    Tie order is Pareto rank, NSGA-II crowding, lineage_root, instance_hash.
    The frozen A0 lineage rule has no cap; lineage is inheritance plus this
    static tie-break. The same function selects intermediate parents and the
    final training set. Parents are not in ``frame``.
    """
    if frame.instance_hash.astype(str).duplicated().any():
        raise RuntimeError("candidate pool contains duplicate hashes")
    work = frame.reset_index(drop=True).copy()
    work["pareto_rank"] = pareto_ranks(work.hardness.to_numpy(float), work.distance.to_numpy(float))
    work["crowding"] = crowding_distance(work.hardness.to_numpy(float), work.distance.to_numpy(float), work.pareto_rank.to_numpy(int))
    work["_crowd_key"] = np.where(np.isfinite(work.crowding.to_numpy(float)), -work.crowding.to_numpy(float), -1e300)
    elites: list[int] = []
    for _, group in work.groupby("niche", sort=True):
        ordered = group.sort_values(["pareto_rank", "_crowd_key", "lineage_root", "instance_hash"], ascending=True, kind="mergesort")
        elites.append(int(ordered.index[0]))
    if len(elites) > POPULATION_N:
        elite_frame = work.loc[elites].sort_values(["pareto_rank", "_crowd_key", "lineage_root", "instance_hash"], ascending=True, kind="mergesort")
        elites = [int(i) for i in elite_frame.index[:POPULATION_N]]
        fill: list[int] = []
    else:
        rest = work.drop(index=elites)
        rest = rest.sort_values(["pareto_rank", "_crowd_key", "lineage_root", "instance_hash"], ascending=True, kind="mergesort")
        need = POPULATION_N - len(elites)
        if len(rest) < need:
            raise RuntimeError(f"candidate pool cannot fill {POPULATION_N}: niches={len(elites)} rest={len(rest)}")
        fill = [int(i) for i in rest.index[:need]]
    chosen = elites + fill
    selected = work.iloc[chosen].copy().reset_index(drop=True)
    selected["source_position"] = chosen
    selected["component"] = ["qd_pareto_elite"] * len(elites) + ["pareto_crowding_lineage_fill"] * len(fill)
    selected["selection_rank"] = np.arange(1, len(selected) + 1)
    if len(selected) != POPULATION_N or selected.instance_hash.astype(str).nunique() != POPULATION_N:
        raise RuntimeError("joint selection size/uniqueness failure")
    return selected.drop(columns="_crowd_key")


def _self_test_selection() -> None:
    hardness = np.array([5.0, 4.0, 3.0, 3.0, 1.0, 2.0])
    distance = np.array([1.0, 2.0, 0.0, 0.0, 3.0, 0.5])
    ranks = pareto_ranks(hardness, distance)
    if ranks.tolist() != [1, 2, 1, 1, 3, 2]:
        raise RuntimeError(f"Pareto self-test failed: {ranks.tolist()}")
    frame = pd.DataFrame({
        "instance_hash": [f"h{i}" for i in range(6)],
        "hardness": [5, 4, 3, 1, 2, 0],
        "distance": [1, 2, 0.2, 0.1, 0.3, 5],
        "niche": [0, 0, 1, 1, 2, 2],
        "lineage_root": ["b", "a", "c", "a", "b", "a"],
    })
    # Temporarily shrink the budget through the public constant only inside this test.
    global POPULATION_N
    saved = POPULATION_N
    POPULATION_N = 4
    try:
        selected = select_joint(frame)
    finally:
        POPULATION_N = saved
    if selected.component.tolist().count("qd_pareto_elite") != 3:
        raise RuntimeError("niche-elite self-test failed")
    if selected.instance_hash.nunique() != 4:
        raise RuntimeError("selection uniqueness self-test failed")


def load_distance_config() -> dict[str, Any]:
    config = json.loads(DISTANCE_CONFIG.read_text())
    if config["features"] != list(F40) or config["effective_n"] != 40 or config["HGB_weighting"] or config["TSPLIB_used_to_fit"]:
        raise RuntimeError("frozen F40 distance config drift")
    return config


def reference_f40() -> tuple[pd.DataFrame, np.ndarray]:
    manifest = pd.read_csv(REF_MANIFEST)
    if len(manifest) != 48 or manifest.instance_id.nunique() != 48:
        raise RuntimeError("TSPLIB48 manifest drift")
    table = pd.read_csv(REF_F40)
    aspect = table[table.coordinate_variant.eq("aspect")].set_index("instance_id").loc[manifest.instance_id.astype(str)]
    values = aspect[list(F40)].to_numpy(float)
    if values.shape != (48, 40):
        raise RuntimeError("TSPLIB48 F40 shape drift")
    return manifest, values


def min_f40_distance(features: np.ndarray, config: dict[str, Any], ref_z: np.ndarray, eff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(config["center_values"], np.float64)
    scale = np.asarray(config["scale_values"], np.float64)
    scaled = (np.asarray(features, np.float64) - center) / scale
    out = np.empty(len(scaled), np.float64)
    which = np.empty(len(scaled), np.int32)
    for start in range(0, len(scaled), 20000):
        block = scaled[start:start + 20000]
        best = np.full(len(block), np.inf)
        arg = np.full(len(block), -1, np.int32)
        for ref_index, ref in enumerate(ref_z):
            mask = eff & np.isfinite(ref)
            delta = block[:, mask] - ref[mask]
            valid = np.isfinite(delta)
            count = valid.sum(1)
            value = np.sqrt(np.divide(np.nansum(delta * delta, axis=1), count, out=np.full(len(delta), np.inf), where=count > 0))
            value[count < min(MIN_F40_DIMS, int(mask.sum()))] = np.inf
            take = value < best
            best[take] = value[take]
            arg[take] = ref_index
        out[start:start + len(block)] = best
        which[start:start + len(block)] = arg
    return out, which


def standardized_references(config: dict[str, Any], ref_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(config["center_values"], np.float64)
    scale = np.asarray(config["scale_values"], np.float64)
    eff = np.asarray(config["effective_mask"], bool)
    return (ref_values - center) / scale, eff


_F40_CACHE: pd.DataFrame | None = None


def formal_f40() -> pd.DataFrame:
    global _F40_CACHE
    if _F40_CACHE is None:
        frame = pd.read_parquet(FORMAL_F40)
        if list(frame.columns) != ["instance_hash", *F40]:
            raise RuntimeError("formal F40 column drift")
        _F40_CACHE = frame.set_index("instance_hash", drop=False)
    return _F40_CACHE


def scaler_params() -> dict[str, tuple[float, float]]:
    payload = json.loads(SCALER.read_text())
    values = {row["feature_name"]: (float(row["center_median"]), float(row["scale"])) for row in payload["parameters"]}
    missing = [name for name in F40 if name not in values]
    if missing:
        raise RuntimeError(f"F40 scaler missing {missing[:4]}")
    return values


def run_feature_job(job: dict[str, Any]) -> None:
    process = subprocess.Popen([sys.executable, str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        _stdout, stderr = process.communicate(pickle_dumps(job), timeout=900)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, 9)
        process.wait()
        raise RuntimeError(f"F40 shard timeout {job['out']}")
    if process.returncode:
        raise RuntimeError(f"F40 shard failed: {stderr.decode(errors='replace')[-2000:]}")


def pickle_dumps(job: dict[str, Any]) -> bytes:
    import pickle
    return pickle.dumps(job)


def align_f40(tag: str, coordinates: np.ndarray, hashes: list[str]) -> pd.DataFrame:
    path = OUT / "features" / f"{tag}.parquet"
    if path.is_file():
        cached = pd.read_parquet(path)
        if cached.instance_hash.astype(str).tolist() == hashes:
            return cached
    known = formal_f40()
    local_dir = OUT / "features" / "cache"
    local_dir.mkdir(parents=True, exist_ok=True)
    have: dict[str, int] = {}
    missing = [i for i, value in enumerate(hashes) if value not in known.index and value not in have]
    # Local shards from previous attempts.
    if missing:
        present = set()
        for shard in sorted(local_dir.glob("f40_*.parquet")):
            present.update(pd.read_parquet(shard, columns=["instance_hash"]).instance_hash.astype(str))
        missing = [i for i in missing if hashes[i] not in present]
    started = time.perf_counter()
    if missing:
        rscript = shutil.which("Rscript") or "/home/haozou/micromamba/envs/blade-r/bin/Rscript"
        params = scaler_params()
        jobs = []
        for start in range(0, len(missing), FEATURE_SHARD_N):
            indices = missing[start:start + FEATURE_SHARD_N]
            block_hashes = [hashes[i] for i in indices]
            token = hashlib.sha256("\n".join(block_hashes).encode()).hexdigest()[:20]
            jobs.append({
                "coordinates": coordinates[np.asarray(indices, int)],
                "hashes": block_hashes,
                "names": list(F40),
                "out": str(local_dir / f"f40_{token}.parquet"),
                "rscript": rscript,
                "r_script": str(R_EXTRACTOR),
                "params": params,
            })
        print(f"[P6-H1][{tag}] F40 missing={len(missing)} shards={len(jobs)}", flush=True)
        with ThreadPoolExecutor(max_workers=min(FEATURE_WORKERS, len(jobs))) as executor:
            futures = [executor.submit(run_feature_job, job) for job in jobs]
            for done, future in enumerate(as_completed(futures), 1):
                future.result()
                if done == 1 or done % 20 == 0 or done == len(jobs):
                    print(f"[P6-H1][{tag}] shards {done}/{len(jobs)}", flush=True)
    rows = []
    local_frames = [pd.read_parquet(shard) for shard in sorted(local_dir.glob("f40_*.parquet"))] if missing or any(value not in known.index for value in hashes) else []
    local = pd.concat(local_frames, ignore_index=True).drop_duplicates("instance_hash", keep="first").set_index("instance_hash") if local_frames else None
    ordered = []
    for value in hashes:
        if value in known.index:
            ordered.append(known.loc[value, list(F40)].to_numpy(float))
        elif local is not None and value in local.index:
            ordered.append(local.loc[value, list(F40)].to_numpy(float))
        else:
            raise RuntimeError(f"missing F40 for {value}")
    matrix = np.vstack(ordered)
    frame = pd.DataFrame(matrix, columns=list(F40))
    frame.insert(0, "instance_hash", hashes)
    frame.to_parquet(path, index=False)
    write_json(path.with_suffix(".json"), {"tag": tag, "n": len(hashes), "missing_extracted": len(missing), "wall_s": time.perf_counter() - started, "peak_rss_mb": rss_mb()})
    return frame


def hgb_models() -> list[Any]:
    import joblib
    payloads = [joblib.load(HGB_DIR / f"HGB40_seed_{seed}.joblib") for seed in HGB_SEEDS]
    for payload in payloads:
        if list(payload["features"]) != list(F40) or payload["target"] != "log1p_LBQ":
            raise RuntimeError("HGB40 payload drift")
    return [payload["model"] for payload in payloads]


def predict_hardness(models: list[Any], features: pd.DataFrame) -> np.ndarray:
    matrix = features[list(F40)].to_numpy(float)
    stacked = np.vstack([model.predict(matrix) for model in models])
    return np.expm1(np.median(stacked, axis=0))


def niche_from_features(features: pd.DataFrame) -> np.ndarray:
    raw = features[list(QD4)].to_numpy(float)
    if not np.isfinite(raw).all():
        raise RuntimeError("QD4 features are not finite")
    return assign_nearest(normalize_qd4(raw))


def score_population(tag: str, coordinates: np.ndarray, meta: pd.DataFrame, models: list[Any], config: dict[str, Any], ref_z: np.ndarray, eff: np.ndarray) -> pd.DataFrame:
    path = OUT / "evolution" / f"{tag}_scores.parquet"
    hashes = meta.instance_hash.astype(str).tolist()
    if path.is_file():
        cached = pd.read_parquet(path)
        if cached.instance_hash.astype(str).tolist() == hashes:
            return cached
    started = time.perf_counter()
    features = align_f40(tag, coordinates, hashes)
    hardness = predict_hardness(models, features)
    distance, nearest = min_f40_distance(features[list(F40)].to_numpy(float), config, ref_z, eff)
    niches = niche_from_features(features)
    scored = meta.copy().reset_index(drop=True)
    scored["hardness"] = hardness
    scored["distance"] = distance
    scored["nearest_reference_index"] = nearest
    scored["niche"] = niches
    if not np.isfinite(scored.hardness).all():
        raise RuntimeError(f"non-finite hardness: {tag}")
    scored.to_parquet(path, index=False)
    write_json(path.with_suffix(".json"), {"tag": tag, "n": len(scored), "wall_s": time.perf_counter() - started, "median_hardness": float(np.median(hardness)), "median_distance": float(np.median(distance)), "niches": int(pd.Series(niches).nunique())})
    return scored


def mutation_dir(replicate: str, block: str) -> Path:
    return OUT / "evolution" / replicate / block / "mutation"


def run_block(replicate: str, block: str, generation_key: int, parent_xy: np.ndarray, parents: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    directory = mutation_dir(replicate, block)
    started = time.perf_counter()
    coordinates, manifest, runtime = p6b.mutate(replicate, block, generation_key, parent_xy, parents, directory)
    runtime = dict(runtime)
    runtime["scientific_block"] = block
    runtime["wrapper_wall_s"] = time.perf_counter() - started
    if block == "block1" and runtime["offspring_membership_sha256"] != HISTORICAL_G1_OFFSPRING[replicate]:
        raise RuntimeError(f"block1 random stream does not reproduce frozen P6-B G1: {replicate}")
    if runtime["nominal_attempts"] != BLOCK_ATTEMPTS:
        raise RuntimeError("mutation attempt budget drift")
    return coordinates, manifest, runtime


def eligible_offspring(coordinates: np.ndarray, manifest: pd.DataFrame, parents: pd.DataFrame, scientific_generation: int, attempt_block: int) -> tuple[np.ndarray, pd.DataFrame]:
    accepted = manifest.proposal_status.eq("accepted") & manifest.valid_after_repair & ~manifest.within_offspring_duplicate
    child = manifest.loc[accepted].reset_index(drop=True)
    if len(child) != len(coordinates):
        raise RuntimeError("offspring coordinate alignment failure")
    parent_hashes = set(parents.instance_hash.astype(str))
    keep = ~child.child_hash.astype(str).isin(parent_hashes).to_numpy()
    child = child.loc[keep].reset_index(drop=True)
    xy = coordinates[keep]
    roots = dict(zip(parents.instance_hash.astype(str), parents.lineage_root.astype(str), strict=True))
    child["instance_hash"] = child.child_hash.astype(str)
    child["lineage_root"] = child.parent_hash.astype(str).map(roots)
    if child.lineage_root.isna().any():
        raise RuntimeError("lineage inheritance failure")
    child["scientific_generation"] = scientific_generation
    child["attempt_block"] = attempt_block
    child["attempt_index"] = child.attempt_index.astype(int) + (0 if attempt_block == 1 else BLOCK_ATTEMPTS if attempt_block == 2 else 0)
    if attempt_block == 3:
        child["attempt_index"] = child.attempt_index.astype(int)  # G2 manifest index stays local; generation separates it
    columns = ["instance_hash", "parent_hash", "operator_id", "lineage_root", "scientific_generation", "attempt_block", "attempt_index"]
    return xy, child[columns].reset_index(drop=True)


def dedup_archive(parts: list[tuple[np.ndarray, pd.DataFrame]]) -> tuple[np.ndarray, pd.DataFrame]:
    seen: set[str] = set()
    rows: list[pd.Series] = []
    coords: list[np.ndarray] = []
    for xy, frame in parts:
        hashes = frame.instance_hash.astype(str).tolist()
        for index, value in enumerate(hashes):
            if value in seen:
                continue
            seen.add(value)
            rows.append(frame.iloc[index])
            coords.append(xy[index])
    if not rows:
        raise RuntimeError("empty archive")
    return np.stack(coords), pd.DataFrame(rows).reset_index(drop=True)


def save_selected(path_stem: Path, coordinates: np.ndarray, selected: pd.DataFrame, source_xy: np.ndarray) -> None:
    xy = source_xy[selected.source_position.to_numpy(int)]
    hashes = selected.instance_hash.astype(str).tolist()
    if [coordinate_hash(row) for row in xy] != hashes:
        raise RuntimeError(f"selected coordinate hash mismatch: {path_stem.name}")
    p6b.save_bundle(path_stem.with_suffix(".npz"), xy, hashes)
    selected.to_parquet(path_stem.with_suffix(".parquet"), index=False)


def population_stem(replicate: str, name: str) -> Path:
    return OUT / "populations" / f"{replicate}_{name}"


def evolve_replicate(replicate: str, models: list[Any], config: dict[str, Any], ref_z: np.ndarray, eff: np.ndarray) -> None:
    marker = OUT / "evolution" / replicate / "completion.json"
    if marker.is_file() and json.loads(marker.read_text()).get("status") == "PASS":
        return
    Status().update(phase="evolve", replicate=replicate)
    parent_xy, parents = p6b.base_population()
    block1_xy, block1_manifest, block1_runtime = run_block(replicate, "block1", 1, parent_xy, parents)
    block2_xy, block2_manifest, block2_runtime = run_block(replicate, "block2", BLOCK2_KEY, parent_xy, parents)
    e1_xy, e1_meta = eligible_offspring(block1_xy, block1_manifest, parents, 1, 1)
    e2_xy, e2_meta = eligible_offspring(block2_xy, block2_manifest, parents, 1, 2)
    one_xy, one_meta = dedup_archive([(e1_xy, e1_meta), (e2_xy, e2_meta)])
    scored_g1 = score_population(f"{replicate}_block1", e1_xy, e1_meta, models, config, ref_z, eff)
    select_started = time.perf_counter()
    parents_selected = select_joint(scored_g1)
    parent_select_s = time.perf_counter() - select_started
    save_selected(population_stem(replicate, "2G_parents"), parent_xy_source := e1_xy, parents_selected, e1_xy)
    del parent_xy_source
    g1_parent_xy, g1_parent_hashes, _ = p6b.load_bundle(population_stem(replicate, "2G_parents").with_suffix(".npz"))
    g1_parents = pd.read_parquet(population_stem(replicate, "2G_parents").with_suffix(".parquet"))
    if g1_parent_hashes != g1_parents.instance_hash.astype(str).tolist():
        raise RuntimeError("G1 parent reload drift")
    g2_started = time.perf_counter()
    g2_xy_raw, g2_manifest, g2_runtime = run_block(replicate, "g2", G2_KEY, g1_parent_xy, g1_parents)
    g2_mutation_s = time.perf_counter() - g2_started
    g2_xy, g2_meta = eligible_offspring(g2_xy_raw, g2_manifest, g1_parents, 2, 3)
    two_xy, two_meta = dedup_archive([(e1_xy, e1_meta), (g2_xy, g2_meta)])
    scored_one = score_population(f"{replicate}_1G_archive", one_xy, one_meta, models, config, ref_z, eff)
    scored_g2 = score_population(f"{replicate}_g2", g2_xy, g2_meta, models, config, ref_z, eff)
    scored_two = score_population(f"{replicate}_2G_archive", two_xy, two_meta, models, config, ref_z, eff)
    t1 = time.perf_counter()
    final_1g = select_joint(scored_one)
    final_1g_s = time.perf_counter() - t1
    t2 = time.perf_counter()
    final_2g = select_joint(scored_two)
    final_2g_s = time.perf_counter() - t2
    save_selected(population_stem(replicate, "1G"), one_xy, final_1g, one_xy)
    save_selected(population_stem(replicate, "2G"), two_xy, final_2g, two_xy)
    historical = {}
    for arm in ("E0", "E1"):
        _xy, hashes, _t = p6b.load_bundle(P6B / "population_manifests" / f"{replicate}_{arm}_P2.npz")
        historical[arm] = set(hashes)
    summary = {
        "status": "PASS",
        "replicate": replicate,
        "block1_attempts": block1_runtime["nominal_attempts"],
        "block1_valid_unique": block1_runtime["valid_unique_within_generation"],
        "block1_offspring_sha256": block1_runtime["offspring_membership_sha256"],
        "block1_matches_historical_g1_raw": True,
        "block1_wall_s": block1_runtime["wall_seconds"],
        "block2_attempts": block2_runtime["nominal_attempts"],
        "block2_valid_unique": block2_runtime["valid_unique_within_generation"],
        "block2_wall_s": block2_runtime["wall_seconds"],
        "block2_namespace": BLOCK2_NS[replicate],
        "g1_eligible_n": len(e1_meta),
        "g1_parent_selection_wall_s": parent_select_s,
        "g1_parent_membership_sha256": membership_hash(g1_parent_hashes),
        "g2_attempts": g2_runtime["nominal_attempts"],
        "g2_valid_unique": g2_runtime["valid_unique_within_generation"],
        "g2_mutation_wall_s": g2_runtime["wall_seconds"],
        "g2_wrapper_wall_s": g2_mutation_s,
        "g2_namespace": G2_NS[replicate],
        "g2_parent_membership_sha256": g2_runtime["parent_membership_sha256"],
        "g2_eligible_n": len(g2_meta),
        "archive_1g_n": len(one_meta),
        "archive_2g_n": len(two_meta),
        "final_1g_sha256": membership_hash(final_1g.instance_hash.astype(str).tolist()),
        "final_2g_sha256": membership_hash(final_2g.instance_hash.astype(str).tolist()),
        "final_1g_selection_wall_s": final_1g_s,
        "final_2g_selection_wall_s": final_2g_s,
        "final_1g_overlap_historical_E0": len(set(final_1g.instance_hash.astype(str)) & historical["E0"]),
        "final_1g_overlap_historical_E1": len(set(final_1g.instance_hash.astype(str)) & historical["E1"]),
        "final_2g_overlap_historical_E0": len(set(final_2g.instance_hash.astype(str)) & historical["E0"]),
        "final_2g_overlap_historical_E1": len(set(final_2g.instance_hash.astype(str)) & historical["E1"]),
        "parents_equal_historical_E0": set(g1_parent_hashes) == historical["E0"],
        "parents_equal_historical_E1": set(g1_parent_hashes) == historical["E1"],
        "peak_rss_mb": rss_mb(),
    }
    if summary["parents_equal_historical_E0"] or summary["parents_equal_historical_E1"]:
        raise RuntimeError("G2 parents reproduced a historical survivor set; refusing to continue")
    if set(final_1g.instance_hash.astype(str)) in (historical["E0"], historical["E1"]) or set(final_2g.instance_hash.astype(str)) in (historical["E0"], historical["E1"]):
        raise RuntimeError("final training set equals a historical E0/E1 population")
    write_json(marker, summary)
    print(f"[P6-H1][{replicate}] evolution PASS archive1={summary['archive_1g_n']} archive2={summary['archive_2g_n']}", flush=True)


def shannon(values: pd.Series) -> tuple[float, float]:
    probability = values.astype(str).value_counts(normalize=True).to_numpy(float)
    return float(np.exp(-(probability * np.log(probability)).sum())), float(probability.max())


def audit() -> pd.DataFrame:
    rows = []
    manifest_rows = []
    config = load_distance_config()
    ref_manifest, ref_values = reference_f40()
    ref_z, eff = standardized_references(config, ref_values)
    parent_xy, parents = p6b.base_population()
    # P03 baseline uses frozen formal features, not a new selector.
    p03_features = formal_f40().loc[parents.instance_hash.astype(str).tolist(), list(F40)]
    p03_distance, _ = min_f40_distance(p03_features.to_numpy(float), config, ref_z, eff)
    p03_niches = niche_from_features(p03_features.reset_index(drop=True))
    eff_l, max_l = shannon(parents.lineage_root)
    rows.append({
        "replicate": "ALL", "depth": "P03", "role": "initial_parent", "n": len(parents),
        "mean_hardness": np.nan, "median_hardness": np.nan, "p90_hardness": np.nan,
        "mean_distance": float(np.mean(p03_distance)), "median_distance": float(np.median(p03_distance)),
        "p10_distance": float(np.quantile(p03_distance, .1)), "occupied_niches": int(pd.Series(p03_niches).nunique()),
        "effective_lineages": eff_l, "max_lineage_share": max_l, "pareto_rank1_fraction": np.nan,
        "elite_fraction": np.nan, "generation2_fraction": 0.0,
    })
    for replicate in REPLICATES:
        summary = json.loads((OUT / "evolution" / replicate / "completion.json").read_text())
        if summary.get("status") != "PASS":
            raise RuntimeError(f"evolution incomplete {replicate}")
        if summary["g2_parent_membership_sha256"] != membership_hash(pd.read_parquet(population_stem(replicate, "2G_parents").with_suffix(".parquet")).instance_hash.astype(str).tolist()):
            raise RuntimeError("G2 parent identity drift")
        for depth, role in (("1G", "final"), ("2G", "final"), ("2G", "intermediate_parent")):
            name = f"{depth}_parents" if role == "intermediate_parent" else depth
            frame = pd.read_parquet(population_stem(replicate, name).with_suffix(".parquet"))
            xy, hashes, _ = p6b.load_bundle(population_stem(replicate, name).with_suffix(".npz"))
            if hashes != frame.instance_hash.astype(str).tolist():
                raise RuntimeError(f"population alignment {replicate} {name}")
            archive_name = f"{replicate}_1G_archive" if depth == "1G" else f"{replicate}_2G_archive"
            if role == "intermediate_parent":
                archive = pd.read_parquet(OUT / "evolution" / f"{replicate}_block1_scores.parquet")
            else:
                archive = pd.read_parquet(OUT / "evolution" / f"{archive_name}_scores.parquet")
            if not set(hashes).issubset(set(archive.instance_hash.astype(str))):
                raise RuntimeError(f"selected instance outside offspring archive {replicate} {name}")
            eff_n, max_share = shannon(frame.lineage_root)
            rows.append({
                "replicate": replicate, "depth": depth, "role": role, "n": len(frame),
                "mean_hardness": float(frame.hardness.mean()), "median_hardness": float(frame.hardness.median()),
                "p90_hardness": float(frame.hardness.quantile(.9)),
                "archive_mean_hardness": float(archive.hardness.mean()), "archive_p75_hardness": float(archive.hardness.quantile(.75)),
                "archive_p90_hardness": float(archive.hardness.quantile(.9)),
                "mean_distance": float(frame.distance.mean()), "median_distance": float(frame.distance.median()),
                "p10_distance": float(frame.distance.quantile(.1)),
                "archive_median_distance": float(archive.distance.median()),
                "occupied_niches": int(frame.niche.nunique()), "archive_niches": int(archive.niche.nunique()),
                "effective_lineages": eff_n, "max_lineage_share": max_share,
                "pareto_rank1_fraction": float((frame.pareto_rank == 1).mean()),
                "elite_fraction": float(frame.component.eq("qd_pareto_elite").mean()),
                "generation2_fraction": float(frame.scientific_generation.eq(2).mean()) if "scientific_generation" in frame else np.nan,
                "operators": int(frame.operator_id.nunique()),
            })
        manifest_rows.append(summary)
    populations = pd.DataFrame(rows)
    populations.to_csv(OUT / "03_population_structure_hardness.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(OUT / "02_evolution_manifest.csv", index=False)
    (OUT / "logs" / "p03_distance_baseline.json").write_text(json.dumps({"median_distance": float(np.median(p03_distance)), "n": len(p03_distance)}, indent=2) + "\n")
    return populations


def verified_prior_labels() -> pd.DataFrame:
    return p6b.verified_label_cache()


def label_selected() -> None:
    prior = verified_prior_labels()
    known = set(prior.instance_hash.astype(str))
    coordinate_map: dict[str, np.ndarray] = {}
    order: list[str] = []
    for replicate in REPLICATES:
        for depth in DEPTHS:
            xy, hashes, _ = p6b.load_bundle(population_stem(replicate, depth).with_suffix(".npz"))
            for value, row in zip(hashes, xy, strict=True):
                if value not in coordinate_map:
                    coordinate_map[value] = row
                    order.append(value)
                elif not np.array_equal(coordinate_map[value], row):
                    raise RuntimeError(f"coordinate collision {value}")
    union_xy = np.stack([coordinate_map[value] for value in order])
    global_dir = OUT / "labels" / "global_unique_union"
    integrity = global_dir / "labeling_integrity.json"
    missing_n = sum(value not in known for value in order)
    started = time.perf_counter()
    if integrity.is_file():
        info = json.loads(integrity.read_text())
        ledger = pd.read_parquet(global_dir / "ledger.parquet")
        if info.get("union_membership_sha256") != membership_hash(order):
            raise RuntimeError("label cache drift")
    else:
        Status().update(phase="label", unique_n=len(order), missing_n=missing_n)
        ledger = label_with_cache(union_xy, order, global_dir, prior)
        ledger.to_parquet(global_dir / "ledger.parquet", index=False)
        info = {"status": "PASS", "unique_n": len(order), "reused": len(order) - missing_n, "new_concorde": missing_n, "wall_s": time.perf_counter() - started, "union_membership_sha256": membership_hash(order), "concorde_sha256": CANONICAL_CONCORDE_SHA256}
        write_json(integrity, info)
    lookup = ledger.set_index("instance_hash", drop=False)
    for replicate in REPLICATES:
        for depth in DEPTHS:
            directory = OUT / "labels" / f"{replicate}_{depth}"
            directory.mkdir(parents=True, exist_ok=True)
            xy, hashes, _ = p6b.load_bundle(population_stem(replicate, depth).with_suffix(".npz"))
            selected = lookup.loc[hashes].reset_index(drop=True)
            tours = p6b.validate_ledger(xy, hashes, selected[list(ORACLE_LEDGER_COLUMNS)] if set(ORACLE_LEDGER_COLUMNS).issubset(selected.columns) else selected)
            selected.to_parquet(directory / "ledger.parquet", index=False)
            p6b.save_bundle(directory / "labeled_bundle.npz", xy, hashes, tours)
            p6b.save_bundle(OUT / "datasets" / f"{replicate}_{depth}_train10k.npz", xy, hashes, tours)
            write_json(directory / "labeling_integrity.json", {"status": "PASS", "replicate": replicate, "depth": depth, "verified": 10000, "membership_sha256": membership_hash(hashes)})


def train_one(replicate: str, depth: str, device_name: str) -> dict[str, Any]:
    import torch
    from bq.checkpoint import load_canonical_checkpoint
    import run_blade_core_ablation_5c_noanchor_foga11base_v1 as core
    integrity_path = OUT / "logs" / f"{replicate}_{depth}" / "training_integrity.json"
    checkpoint = OUT / "checkpoints" / f"{replicate}_{depth}_M5_E256.tsp.best"
    if integrity_path.is_file():
        existing = json.loads(integrity_path.read_text())
        if existing.get("status") == "PASS" and checkpoint.is_file() and sha256_file(checkpoint) == existing.get("checkpoint_sha256"):
            return existing
    label_info = json.loads((OUT / "labels" / f"{replicate}_{depth}" / "labeling_integrity.json").read_text())
    if label_info.get("verified") != 10000:
        raise RuntimeError("label gate failed")
    coordinates, hashes, tours = p6b.load_bundle(OUT / "labels" / f"{replicate}_{depth}" / "labeled_bundle.npz", True)
    assert tours is not None
    frozen_xy, frozen_hashes, _ = p6b.load_bundle(population_stem(replicate, depth).with_suffix(".npz"))
    if hashes != frozen_hashes or not np.array_equal(coordinates, frozen_xy):
        raise RuntimeError("training membership drift")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    directory = OUT / "logs" / f"{replicate}_{depth}" / "training_cycle"
    core.RUN_ID = f"TARGET_AWARE_P6H1_{replicate}_{depth}"
    core.cycle_dir = lambda selected_arm, cycle: directory
    core.model_path = lambda selected_arm, index: checkpoint if index == 5 else p6b.M4
    model, model_meta = load_canonical_checkpoint(p6b.M4, device, expected_sha256=M4_SHA)
    if model_meta["sha256"] != M4_SHA:
        raise RuntimeError("M4 load drift")
    Status().update(phase="train", replicate=replicate, depth=depth, device=device_name)
    started = time.perf_counter()
    trained, meta, timing = core.train(f"P6H1_{replicate}_{depth}", 4, model, model_meta["sha256"], coordinates, hashes, tours, device, Status(), True)
    stage = directory / "09_bq_training"
    e000 = torch.load(stage / "M5_E000.tsp.best", map_location="cpu", weights_only=False)
    m4 = torch.load(p6b.M4, map_location="cpu", weights_only=False)
    equal = set(e000["net"]) == set(m4["net"]) and all(torch.equal(e000["net"][key], m4["net"][key]) for key in m4["net"])
    curve = pd.read_csv(stage / "training_loss_curve.csv")
    if (not equal or len(curve) != 256 or int(curve.optimizer_updates.sum()) != 10240
            or not curve.hard_instances_seen.eq(10000).all() or meta["sha256"] != sha256_file(checkpoint)):
        raise RuntimeError("training integrity failure")
    result = {
        "status": "PASS", "replicate": replicate, "depth": depth, "training_seed": TRAINING_SEED,
        "m4_sha256": M4_SHA, "e000_weights_equal_m4": True, "epochs": 256,
        "optimizer_steps": int(timing["optimizer_steps"]), "final_loss": float(timing["final_loss"]),
        "training_loop_wall_s": float(timing["training_loop_wall_s"]), "invocation_wall_s": time.perf_counter() - started,
        "gpu_peak_allocated_bytes": int(timing["gpu_peak_allocated_bytes"]), "checkpoint_sha256": meta["sha256"],
        "checkpoint_path": str(checkpoint), "membership_sha256": membership_hash(hashes),
        "dataset_bundle_sha256": sha256_file(population_stem(replicate, depth).with_suffix(".npz")),
    }
    write_json(integrity_path, result)
    del trained, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return result


def evaluate_one(replicate: str, depth: str, device_name: str) -> dict[str, Any]:
    import torch
    from bq.checkpoint import load_canonical_checkpoint
    import run_blade_core_ablation_5c_noanchor_foga11base_v1 as core
    integrity_path = OUT / "logs" / f"{replicate}_{depth}" / "evaluation_integrity.json"
    output = OUT / "logs" / f"{replicate}_{depth}" / "external_benchmark_results.csv"
    training = json.loads((OUT / "logs" / f"{replicate}_{depth}" / "training_integrity.json").read_text())
    checkpoint = OUT / "checkpoints" / f"{replicate}_{depth}_M5_E256.tsp.best"
    if training["checkpoint_sha256"] != sha256_file(checkpoint):
        raise RuntimeError("checkpoint drift")
    if integrity_path.is_file() and output.is_file():
        existing = json.loads(integrity_path.read_text())
        if existing.get("status") == "PASS" and existing.get("checkpoint_sha256") == training["checkpoint_sha256"]:
            return existing
    device = torch.device(device_name)
    model, meta = load_canonical_checkpoint(checkpoint, device, expected_sha256=training["checkpoint_sha256"])
    Status().update(phase="evaluate", replicate=replicate, depth=depth)
    frame, wall = core.evaluate(model, checkpoint, f"P6H1_{replicate}_{depth}_M5", output, Status(), f"P6H1_{replicate}_{depth}")
    historical = pd.read_csv(P6B / "logs" / f"{replicate}_E0" / "external_benchmark_results.csv").set_index("dataset")
    current = frame.set_index("dataset")
    for dataset in p6b.DATASETS:
        if current.at[dataset, "dataset_sha256"] != historical.at[dataset, "dataset_sha256"] or current.at[dataset, "evaluator_config_sha256"] != historical.at[dataset, "evaluator_config_sha256"]:
            raise RuntimeError(f"evaluation protocol drift {dataset}")
    result = {"status": "PASS", "replicate": replicate, "depth": depth, "checkpoint_sha256": meta["sha256"], "evaluation_wall_s": wall, "results_sha256": sha256_file(output)}
    write_json(integrity_path, result)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def detail(replicate: str, depth: str, dataset: str) -> pd.DataFrame:
    return pd.read_parquet(OUT / "logs" / f"{replicate}_{depth}" / "evaluation_work" / f"P6H1_{replicate}_{depth}_M5" / dataset / "instance_metrics.parquet")


def bootstrap_mean(values: np.ndarray, key: str) -> tuple[float, float]:
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"P6H1|{BOOTSTRAP_SEED}|{key}".encode()).digest()[:8], "little"))
    draw = np.empty(BOOTSTRAP_N)
    x = np.asarray(values, float)
    for index in range(BOOTSTRAP_N):
        draw[index] = x[rng.integers(0, len(x), len(x))].mean()
    return float(np.quantile(draw, .025)), float(np.quantile(draw, .975))


def fft_matrix(coordinates: np.ndarray) -> np.ndarray:
    sys.path.insert(0, str(ROOT / "scripts"))
    from evoreal_structure_alignment_audit_corrected_v2 import fft_energy, nn_ratio_stats
    out = np.empty((len(coordinates), 2), np.float64)
    for index, xy in enumerate(coordinates):
        out[index, 0] = fft_energy(xy)
        out[index, 1] = nn_ratio_stats(xy)["nn_ratio"]
    return out


def fft_coverage(coordinates: np.ndarray, refs: pd.DataFrame) -> dict[str, float]:
    rules = json.loads(FFT_RULES.read_text())
    center = np.asarray(rules["center"], float)
    scale = np.asarray(rules["scale_iqr"], float)
    radius = float(rules["radius"])
    cand = (fft_matrix(coordinates) - center) / scale
    ref = (refs[["fft_energy", "nn_ratio"]].to_numpy(float) - center) / scale
    from scipy.spatial.distance import cdist
    dist = cdist(cand, ref)
    return {
        "fft_reference_coverage_n": int(np.sum(dist.min(axis=0) <= radius)),
        "fft_candidate_inside_fraction": float(np.mean(dist.min(axis=1) <= radius)),
        "fft_median_min_distance": float(np.median(dist.min(axis=1))),
    }


def geometry_summary(replicate: str, depth: str) -> dict[str, float]:
    sys.path.insert(0, str(P6G2))
    import run_p6g2_analysis as g2
    frame = pd.read_parquet(population_stem(replicate, depth).with_suffix(".parquet"))
    xy, hashes, _ = p6b.load_bundle(population_stem(replicate, depth).with_suffix(".npz"))
    manifest, _values = reference_f40()
    coords = {row.instance_id: g2.parse_tsplib(ROOT / row.source_file) for row in manifest.itertuples(index=False)}
    composites = []
    consistent = 0
    rules = json.loads(FFT_RULES.read_text())
    # FFT of the single nearest-F40 candidate per reference.
    fft = fft_matrix(xy)
    center = np.asarray(rules["center"], float)
    scale = np.asarray(rules["scale_iqr"], float)
    radius = float(rules["radius"])
    ref_fft = pd.read_parquet(ROOT / "target_aware_pilot/p6e0_evoreal_structural_representation_audit/features/TSPLIB48.parquet")
    ref_fft = manifest[["instance_id"]].merge(ref_fft[["instance_id", "fft_energy", "nn_ratio"]], on="instance_id", validate="one_to_one")
    ref_z = (ref_fft[["fft_energy", "nn_ratio"]].to_numpy(float) - center) / scale
    cand_z = (fft - center) / scale
    for ref_index, row in enumerate(manifest.itertuples(index=False)):
        group = np.flatnonzero(frame.nearest_reference_index.to_numpy(int) == ref_index)
        if len(group) == 0:
            position = int(np.argmin(frame.distance.to_numpy(float)))
        else:
            position = int(group[np.argmin(frame.distance.to_numpy(float)[group])])
        geo = g2.geometry_compare(coords[row.instance_id], xy[position])
        composites.append(float(geo["geometry_composite"]))
        fft_d = float(np.linalg.norm(cand_z[position] - ref_z[ref_index]))
        consistent += int(fft_d <= radius)
    return {"f40_nearest_geometry_mean": float(np.mean(composites)), "f40_nearest_also_fft_covered_n": consistent}


def preflight() -> dict[str, Any]:
    ensure_dirs()
    _self_test_selection()
    if sha256_file(p6b.M4) != M4_SHA:
        raise RuntimeError("M4 checkpoint is not the formal checkpoint")
    if sha256_file(p6b.P03_BUNDLE) != P03_BUNDLE_SHA:
        raise RuntimeError("P03 bundle is not the formal P6-B parent")
    parent_xy, parents = p6b.base_population()
    if len(parents) != 10000 or parents.instance_hash.nunique() != 10000:
        raise RuntimeError("P03 size drift")
    config = load_distance_config()
    ref_manifest, ref_values = reference_f40()
    ref_z, eff = standardized_references(config, ref_values)
    cached = np.load(P6G2 / "cache/P03_f40_dist.npy", mmap_mode="r")
    features = formal_f40().loc[parents.instance_hash.astype(str).tolist()]
    if features.instance_hash.astype(str).tolist() != parents.instance_hash.astype(str).tolist():
        raise RuntimeError("P03 F40 alignment drift")
    mine, _ = min_f40_distance(features[list(F40)].to_numpy(float), config, ref_z, eff)
    cached_min = np.asarray(cached, np.float64).min(axis=1)
    gap = float(np.max(np.abs(mine - cached_min)))
    if gap > 1e-4:
        raise RuntimeError(f"F40 distance does not reproduce P6-G2 on P03: max abs {gap}")
    qd_gap = float(np.max(np.abs(features[list(QD4)].to_numpy(float) - parents[list(QD4)].to_numpy(float))))
    niches = niche_from_features(features.reset_index(drop=True))
    niche_mismatch = int(np.sum(niches != parents.niche_id.to_numpy(int)))
    sample = extract_qd4_exact(parent_xy[:128])
    exact_gap = float(np.max(np.abs(sample - features[list(QD4)].to_numpy(float)[:128])))
    if qd_gap > 1e-6 or niche_mismatch or exact_gap > 1e-5:
        raise RuntimeError(f"QD identity failed qd_gap={qd_gap} niche_mismatch={niche_mismatch} exact_gap={exact_gap}")
    models = hgb_models()
    probe = predict_hardness(models, features.iloc[:32])
    if not np.isfinite(probe).all() or len(models) != 5:
        raise RuntimeError("HGB40 probe failed")
    for dataset in p6b.DATASETS:
        if not (M4_EVAL / dataset / "instance_metrics.parquet").is_file():
            raise RuntimeError(f"missing frozen M4 evaluation {dataset}")
    rule_hash = source_hash([pareto_ranks, crowding_distance, select_joint, min_f40_distance])
    protocol = {
        "experiment": "BLADE-NCO P6-H1",
        "m4_sha256": M4_SHA,
        "p03_bundle_sha256": P03_BUNDLE_SHA,
        "p03_parent_membership_sha256": membership_hash(parents.instance_hash.astype(str).tolist()),
        "hgb_dir": str(HGB_DIR),
        "hgb_sha256": {str(seed): sha256_file(HGB_DIR / f"HGB40_seed_{seed}.joblib") for seed in HGB_SEEDS},
        "f40_distance_config_sha256": sha256_file(DISTANCE_CONFIG),
        "f40_distance_p03_max_abs_gap": gap,
        "selection_source_sha256": rule_hash,
        "block1_namespace": {rep: p6b.NAMESPACES[rep][1] for rep in REPLICATES},
        "block2_namespace": BLOCK2_NS,
        "g2_namespace": G2_NS,
        "attempts": {"1G": 220000, "2G_g1": 110000, "2G_g2": 110000},
        "selection": "one best Pareto candidate per occupied QD niche, then Pareto/crowding/lineage_root/hash fill to 10000",
        "lineage_cap": False,
        "candidate_pool": "valid repaired deduplicated offspring only",
        "historical_e0_e1_used_as_pool": False,
        "training": p6b.base_training_protocol(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "created_utc": now(),
    }
    write_json(OUT / "configs" / "frozen_p6h1_config.json", protocol)
    lines = [
        "# P6-H1 协议与预检",
        "",
        "本文件在任何新种群的适应度被用于改规则之前写入。后续只允许修复明确的软件错误。",
        "",
        "## 冻结身份",
        "",
        f"- 正式 M4 SHA256：`{M4_SHA}`。",
        f"- 正式 P03 bundle SHA256：`{P03_BUNDLE_SHA}`；父代 membership `{protocol['p03_parent_membership_sha256']}`。",
        f"- HGB40：A0 C04 五个种子 2026–2030，目标 `log1p_LBQ`，部署为 median 后 `expm1` 一次。",
        f"- F40 距离配置 SHA256：`{protocol['f40_distance_config_sha256']}`。在 P03 上与 P6-G2 缓存的最近距离最大绝对差 `{gap:.3e}`。",
        f"- QD4/CVT-1000 与 P03 的 niche 完全一致；`extract_qd4_exact` 与正式 F40 的 QD 四列最大差 `{exact_gap:.3e}`。",
        "- 结构距离使用 P6-G2 的全 40 维稳健标准化和 RMS，参考为 aspect 坐标下的 TSPLIB48。不用 HGB importance，不用 TSPLIB22，不用 FFT–NN 作优化目标。",
        "",
        "## 已解决的规则冲突",
        "",
        "历史 A0 在每个 niche 保留最高 HGB，再用全局 HGB 补足，且 `lineage_constraint=false`。P6-H1 不引入人工权重，因此把排序键从单一 HGB 换成双目标 Pareto rank，再用 NSGA-II 拥挤距离。多样性骨架仍是：每个已占据的冻结 QD niche 先留一名代表，再补到 10k。",
        "",
        "谱系没有名额上限。`lineage_root` 从父代继承，只在 Pareto rank 与拥挤距离都相同后，按 `lineage_root`、`instance_hash` 的字典序打破平局。中间父代和最终训练集使用同一个函数。",
        "",
        "候选池只含修复后、代内去重、且不与直接父代哈希碰撞的 offspring。不把 P03 或历史 E0/E1 放进候选池。",
        "",
        "## 随机流",
        "",
        "`cycle_child_seed` 没有 attempt 序号。前 110k 使用 P6-B 第一代正式 namespace，因此与历史 shared G1 原始 offspring 同分布且应逐成员一致；这只用于配对，不把历史 E0/E1 的 10k 幸存者当作候选池。后 110k 与第二代使用本次预先指定的 namespace：",
        "",
        f"- block2：`{json.dumps(BLOCK2_NS)}`",
        f"- G2：`{json.dumps(G2_NS)}`",
        "",
        "两种深度的算子配置都是 FOGA11_BASE_V1。1G 是同一 P03 上连续两个 110k 块。2G 的 G2 父代必须是 G1 offspring 的联合选择结果。",
        "",
        "## 训练与判定",
        "",
        "每个最终 10k 从同一 M4 微调：fresh Adam，学习率 3.75e-5，无 scheduler，逻辑 batch 256，microbatch 64，256 epoch，10240 updates，训练种子 2026，固定 E256。专家路径只对最终 10k 求解，不对全体 offspring 调用 Concorde 或真实 BQ CE。",
        "",
        "标签规则：",
        "",
        "- `P6_H1_STRUCTURE_IMPROVEMENT`：六个最终种群相对 P03 的 F40 最近距离中位数全部下降，且至少四个的 FFT 参考覆盖数不低于 P03，为 SUPPORTED；至少三个距离下降但未达到上述条件为 PARTIAL；否则 NOT_SUPPORTED。",
        "- `P6_H1_HARDNESS_PRESERVATION`：选中集预测 hardness 均值高于其 offspring archive 均值，且选中集 P90 不低于 archive P75。6/6 为 SUPPORTED，3–5 为 PARTIAL，否则 NOT_SUPPORTED。",
        "- `P6_H1_1G_VS_M4` / `P6_H1_2G_VS_M4` / `P6_H1_2G_VS_1G`：Original70 均值在三个 seed 上全部更好为 BETTER，全部更差为 WORSE，否则 MIXED。更好指 gap 更低。",
        "- `P6_H1_STABLE_ORIGINAL70_GAIN`：六个模型均值都低于 M4，且每个 seed×深度的实例配对（M4 gap 减模型 gap）95% bootstrap 下界大于 0，为 YES；六个均值都不低于 M4 为 NO；否则 INCONCLUSIVE。",
        "- `P6_H1_JOINT_EVOLUTION_VALID`：预算、起点、G2 父代、选择规则和训练完整性全部通过，且最终集不等于历史 E0 或 E1，为 YES。",
        "",
        f"选择实现 SHA256：`{rule_hash}`。",
        "",
        "预检：**PASS**。",
    ]
    (OUT / "01_protocol_and_preflight.md").write_text("\n".join(lines) + "\n")
    Status().update(phase="preflight", status="PASS", selection_source_sha256=rule_hash)
    return protocol


def md_table(frame: pd.DataFrame) -> str:
    show = frame.copy()
    for column in show.columns:
        if pd.api.types.is_float_dtype(show[column]):
            show[column] = show[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    header = "| " + " | ".join(map(str, show.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in show.columns) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in show.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *body])


def conclude() -> dict[str, Any]:
    populations = pd.read_csv(OUT / "03_population_structure_hardness.csv")
    p03 = populations[(populations.depth == "P03")].iloc[0]
    # FFT and geometry are diagnostics. They are computed once and cached.
    fft_path = OUT / "manifests" / "fft_geometry.csv"
    if fft_path.is_file():
        diagnostics = pd.read_csv(fft_path)
    else:
        Status().update(phase="diagnostics")
        ref_manifest, _ = reference_f40()
        ref_fft = pd.read_parquet(ROOT / "target_aware_pilot/p6e0_evoreal_structural_representation_audit/features/TSPLIB48.parquet")
        refs = ref_manifest[["instance_id", "n_nodes"]].merge(ref_fft[["instance_id", "fft_energy", "nn_ratio"]], on="instance_id", validate="one_to_one")
        parent_xy, _parents = p6b.base_population()
        diag_rows = [{"replicate": "ALL", "depth": "P03", "role": "initial_parent", **fft_coverage(parent_xy, refs)}]
        for replicate in REPLICATES:
            for depth in DEPTHS:
                xy, _hashes, _ = p6b.load_bundle(population_stem(replicate, depth).with_suffix(".npz"))
                row = {"replicate": replicate, "depth": depth, "role": "final", **fft_coverage(xy, refs)}
                row.update(geometry_summary(replicate, depth))
                diag_rows.append(row)
                print(f"[P6-H1] diagnostics {replicate} {depth} fft_refs={row['fft_reference_coverage_n']}", flush=True)
            parent_xy_g1, _h, _t = p6b.load_bundle(population_stem(replicate, "2G_parents").with_suffix(".npz"))
            diag_rows.append({"replicate": replicate, "depth": "2G", "role": "intermediate_parent", **fft_coverage(parent_xy_g1, refs)})
        diagnostics = pd.DataFrame(diag_rows)
        diagnostics.to_csv(fft_path, index=False)
    p03_fft = int(diagnostics[(diagnostics.depth == "P03")].fft_reference_coverage_n.iloc[0])
    finals = populations[populations.role.eq("final")].copy()
    distance_improved = finals.median_distance < float(p03.median_distance)
    fft_final = diagnostics[diagnostics.role.eq("final")]
    merged = finals.merge(fft_final[["replicate", "depth", "fft_reference_coverage_n"]], on=["replicate", "depth"], how="left")
    coverage_ok = merged.fft_reference_coverage_n >= p03_fft
    n_distance = int(distance_improved.sum())
    n_both = int((distance_improved.to_numpy() & coverage_ok.to_numpy()).sum()) if len(merged) == len(finals) else 0
    if n_distance == 6 and int(coverage_ok.sum()) >= 4:
        structure = "SUPPORTED"
    elif n_distance >= 3:
        structure = "PARTIAL"
    else:
        structure = "NOT_SUPPORTED"
    hardness_keep = (finals.mean_hardness > finals.archive_mean_hardness) & (finals.p90_hardness >= finals.archive_p75_hardness)
    n_keep = int(hardness_keep.sum())
    hardness_label = "SUPPORTED" if n_keep == 6 else "PARTIAL" if n_keep >= 3 else "NOT_SUPPORTED"
    eval_rows = []
    per_rows = []
    stat_rows = []
    dev = set(pd.read_csv(REF_MANIFEST).instance_id.astype(str))
    held = set(json.loads(HELD22.read_text())["names"])
    m4_frames = {}
    for dataset in ("OriginalTSPLIB70", "Uniform100", "GaussianMixture100"):
        m4_frames[dataset] = pd.read_parquet(M4_EVAL / dataset / "instance_metrics.parquet")
    original_m4 = m4_frames["OriginalTSPLIB70"]
    for dataset, names in (("TSPLIB48", dev), ("TSPLIB22", held)):
        part = original_m4[original_m4.instance_id.astype(str).isin(names)].copy()
        if len(part) != len(names):
            raise RuntimeError(f"M4 split drift {dataset}")
        m4_frames[dataset] = part
    gaps = {}
    for replicate in REPLICATES:
        for depth in DEPTHS:
            for dataset in ("OriginalTSPLIB70", "Uniform100", "GaussianMixture100"):
                frame = detail(replicate, depth, dataset)
                m4 = m4_frames[dataset][["instance_id", "coordinate_hash", "gap_percent"]].rename(columns={"gap_percent": "gap_M4", "coordinate_hash": "hash_M4"})
                merged_gap = frame.merge(m4, on="instance_id", validate="one_to_one")
                if not merged_gap.coordinate_hash.astype(str).eq(merged_gap.hash_M4.astype(str)).all():
                    raise RuntimeError(f"paired instance drift {replicate} {depth} {dataset}")
                gaps[(replicate, depth, dataset)] = merged_gap
                eval_rows.append({"replicate": replicate, "depth": depth, "model": f"M5-{depth}", "dataset": dataset, "n": len(merged_gap), "mean_gap": float(merged_gap.gap_percent.mean()), "mean_gap_M4": float(merged_gap.gap_M4.mean()), "mean_delta_M4_minus_model": float((merged_gap.gap_M4 - merged_gap.gap_percent).mean())})
            base = gaps[(replicate, depth, "OriginalTSPLIB70")]
            for dataset, names in (("TSPLIB48", dev), ("TSPLIB22", held)):
                part = base[base.instance_id.astype(str).isin(names)]
                eval_rows.append({"replicate": replicate, "depth": depth, "model": f"M5-{depth}", "dataset": dataset, "n": len(part), "mean_gap": float(part.gap_percent.mean()), "mean_gap_M4": float(part.gap_M4.mean()), "mean_delta_M4_minus_model": float((part.gap_M4 - part.gap_percent).mean())})
            per = base[["instance_id", "gap_percent", "gap_M4"]].rename(columns={"gap_percent": f"gap_{depth}"})
            per.insert(0, "replicate", replicate)
            per_rows.append(per)
            delta = (base.gap_M4 - base.gap_percent).to_numpy(float)
            lo, hi = bootstrap_mean(delta, f"{replicate}|{depth}|vsM4")
            stat_rows.append({"replicate": replicate, "comparison": f"{depth}_vs_M4", "dataset": "OriginalTSPLIB70", "n": len(delta), "mean_delta": float(delta.mean()), "ci95_low": lo, "ci95_high": hi, "model_better_instances": int((delta > 0).sum())})
        left = gaps[(replicate, "1G", "OriginalTSPLIB70")].set_index("instance_id")
        right = gaps[(replicate, "2G", "OriginalTSPLIB70")].set_index("instance_id")
        if not left.index.equals(right.index):
            right = right.loc[left.index]
        delta = (left.gap_percent - right.gap_percent).to_numpy(float)
        lo, hi = bootstrap_mean(delta, f"{replicate}|2G_vs_1G")
        stat_rows.append({"replicate": replicate, "comparison": "2G_vs_1G", "dataset": "OriginalTSPLIB70", "n": len(delta), "mean_delta": float(delta.mean()), "ci95_low": lo, "ci95_high": hi, "model_better_instances": int((delta > 0).sum())})
    evaluation = pd.DataFrame(eval_rows)
    stats = pd.DataFrame(stat_rows)
    evaluation.to_csv(OUT / "logs" / "external_evaluation_long.csv", index=False)
    stats.to_csv(OUT / "logs" / "paired_bootstrap.csv", index=False)
    # Required wide-enough evaluation table.
    m4_rows = []
    for dataset in ("OriginalTSPLIB70", "TSPLIB48", "TSPLIB22", "Uniform100", "GaussianMixture100"):
        m4_rows.append({"replicate": "M4", "depth": "M4", "model": "M4", "dataset": dataset, "n": len(m4_frames[dataset] if dataset in m4_frames else original_m4), "mean_gap": float(m4_frames[dataset].gap_percent.mean()), "mean_gap_M4": float(m4_frames[dataset].gap_percent.mean()), "mean_delta_M4_minus_model": 0.0})
    pd.concat([pd.DataFrame(m4_rows), evaluation, stats.assign(record="paired")], ignore_index=True, sort=False).to_csv(OUT / "05_external_evaluation.csv", index=False)
    training_rows = []
    for replicate in REPLICATES:
        for depth in DEPTHS:
            training_rows.append(json.loads((OUT / "logs" / f"{replicate}_{depth}" / "training_integrity.json").read_text()))
    training = pd.DataFrame(training_rows)
    training.to_csv(OUT / "04_training_runs.csv", index=False)
    original = evaluation[evaluation.dataset.eq("OriginalTSPLIB70")]
    def verdict(mask_better: pd.Series) -> str:
        if int(mask_better.sum()) == 3:
            return "BETTER"
        if int((~mask_better).sum()) == 3:
            return "WORSE"
        return "MIXED"
    g1 = original[original.depth.eq("1G")].set_index("replicate")
    g2 = original[original.depth.eq("2G")].set_index("replicate")
    label_1g = verdict(g1.mean_gap < g1.mean_gap_M4)
    label_2g = verdict(g2.mean_gap < g2.mean_gap_M4)
    label_21 = verdict(g2.mean_gap < g1.mean_gap)
    stable_rows = stats[stats.comparison.str.endswith("vs_M4")]
    stable = "YES" if bool((original.mean_gap < original.mean_gap_M4).all() and (stable_rows.ci95_low > 0).all()) else "NO" if bool((original.mean_gap >= original.mean_gap_M4).all()) else "INCONCLUSIVE"
    evolutions = [json.loads((OUT / "evolution" / rep / "completion.json").read_text()) for rep in REPLICATES]
    training_ok = bool(training.status.eq("PASS").all() and training.epochs.eq(256).all() and training.optimizer_steps.eq(10240).all() and training.e000_weights_equal_m4.all())
    evolution_ok = all(item["status"] == "PASS" and item["block1_attempts"] == 110000 and item["block2_attempts"] == 110000 and item["g2_attempts"] == 110000 and not item["parents_equal_historical_E0"] and not item["parents_equal_historical_E1"] for item in evolutions)
    valid = "YES" if evolution_ok and training_ok else "NO"
    # Narrative numbers.
    def mean_gap(depth: str) -> float:
        return float(original[original.depth.eq(depth)].mean_gap.mean())
    m4_gap = float(original.mean_gap_M4.iloc[0])
    answers = {
        "P6_H1_JOINT_EVOLUTION_VALID": valid,
        "P6_H1_STRUCTURE_IMPROVEMENT": structure,
        "P6_H1_HARDNESS_PRESERVATION": hardness_label,
        "P6_H1_1G_VS_M4": label_1g,
        "P6_H1_2G_VS_M4": label_2g,
        "P6_H1_2G_VS_1G": label_21,
        "P6_H1_STABLE_ORIGINAL70_GAIN": stable,
        "P6_H1_STATUS": "COMPLETE",
    }
    seed_lines = [
        "# P6-H1 分 seed 比较",
        "",
        f"M4 Original70 均值 gap 为 {m4_gap:.6f}%。三个 seed 都必须同时报告。",
        "",
        "## Original70",
        "",
        md_table(original[["replicate", "depth", "mean_gap", "mean_gap_M4", "mean_delta_M4_minus_model"]]),
        "",
        "## 实例配对 bootstrap",
        "",
        "正的 mean_delta 表示比较中的后者 gap 更低。`1G_vs_M4` 与 `2G_vs_M4` 的 delta 是 M4 减模型；`2G_vs_1G` 的 delta 是 1G 减 2G。区间条件于冻结的 70 个实例。",
        "",
        md_table(stats),
        "",
        "## 其他冻结评估集",
        "",
        md_table(evaluation[evaluation.dataset.ne("OriginalTSPLIB70")][["replicate", "depth", "dataset", "mean_gap", "mean_gap_M4", "mean_delta_M4_minus_model"]]),
    ]
    (OUT / "06_seedwise_comparison.md").write_text("\n".join(seed_lines) + "\n")
    g1_med = float(finals[finals.depth.eq("1G")].median_distance.mean())
    g2_med = float(finals[finals.depth.eq("2G")].median_distance.mean())
    p03_med = float(p03.median_distance)
    conclusion = [
        "# P6-H1 最终结论",
        "",
        "本实验比较的是固定 220k mutation attempts 下，一代直接搜索与两代选择后再搜索的完整效果。它不能单独分离演化深度、每代预算和中间选择的因果贡献。",
        "",
        "## 标签",
        "",
        *[f"`{key} = {value}`" for key, value in answers.items()],
        "",
        "## 结构",
        "",
        f"P03 的 F40 最近距离中位数是 {p03_med:.6f}。1G 三个最终种群的中位数平均为 {g1_med:.6f}，2G 为 {g2_med:.6f}。距离下降的最终种群有 {n_distance}/6。P03 的 FFT–NN 参考覆盖是 {p03_fft}/48。",
        "",
        md_table(merged[["replicate", "depth", "median_distance", "median_hardness", "occupied_niches", "effective_lineages", "fft_reference_coverage_n", "pareto_rank1_fraction"]]),
        "",
        "FFT–NN 与独立几何只作诊断。几何复合值见 `manifests/fft_geometry.csv`。",
        "",
        "## 硬度",
        "",
        f"相对各自 offspring archive，满足“均值高于 archive 且 P90 不低于 archive P75”的最终种群有 {n_keep}/6。",
        "",
        "## Original70",
        "",
        f"M4 为 {m4_gap:.6f}%。1G 三 seed 平均 {mean_gap('1G'):.6f}%，2G 三 seed 平均 {mean_gap('2G'):.6f}%。",
        "",
        md_table(original[["replicate", "depth", "mean_gap", "mean_delta_M4_minus_model"]]),
        "",
        "训练墙钟与中间选择耗时见 `02_evolution_manifest.csv` 和 `04_training_runs.csv`。",
        "",
        "## 四个问题",
        "",
        structure_answer(structure, n_distance, p03_med, g1_med, g2_med),
        depth_answer(label_21, mean_gap("1G"), mean_gap("2G")),
        gain_answer(label_1g, label_2g, stable, m4_gap, mean_gap("1G"), mean_gap("2G")),
        mechanism_answer(structure, stable, hardness_label),
        "",
        "`BLADE_P6H1_HARDNESS_STRUCTURE_JOINT_EVOLUTION_COMPLETE`",
    ]
    (OUT / "07_final_conclusion.md").write_text("\n".join(conclusion) + "\n")
    completion = {"status": "COMPLETE", "final_tag": "BLADE_P6H1_HARDNESS_STRUCTURE_JOINT_EVOLUTION_COMPLETE", "labels": answers, "created_utc": now(), "m4_sha256": M4_SHA, "p03_bundle_sha256": P03_BUNDLE_SHA}
    write_json(OUT / "completion.json", completion)
    return completion


def structure_answer(label: str, n_distance: int, p03: float, g1: float, g2: float) -> str:
    return f"联合选择是否改善结构探索：标签 {label}。{n_distance}/6 个最终种群的 F40 最近距离中位数低于 P03（{p03:.4f} → 1G 均值 {g1:.4f}，2G 均值 {g2:.4f}）。改善指相对起点 P03 的可测量接近，不是相对一个硬度单目标对照臂的因果增益。"


def depth_answer(label: str, g1: float, g2: float) -> str:
    return f"相同 mutation 预算下一代还是两代更好：Original70 标签 {label}。1G 三 seed 平均 gap {g1:.4f}%，2G {g2:.4f}%。该比较绑定了中间选择，不能解释成单纯“代数”效应。"


def gain_answer(g1: str, g2: str, stable: str, m4: float, a: float, b: float) -> str:
    return f"Original70 训练收益：1G 相对 M4 为 {g1}，2G 相对 M4 为 {g2}，稳定增益标签 {stable}。M4 {m4:.4f}%，1G 平均 {a:.4f}%，2G 平均 {b:.4f}%。"


def mechanism_answer(structure: str, stable: str, hardness: str) -> str:
    if structure == "SUPPORTED" and stable != "YES":
        return "若结构距离下降但 Original70 收益不稳定，证据指向结构接近并没有变成 teacher-forced 模仿所利用的训练信号：F40 最近邻改善停留在诊断空间，BQ 的贪心解码误差仍由硬度尾部、谱系覆盖或跨规模特征错配主导。硬度保持标签为 " + hardness + "。"
    if structure != "SUPPORTED" and stable != "YES":
        return "结构和 Original70 都没有形成稳定改进。证据指向在这条 FOGA-11 预算内，Pareto 联合选择既没有把种群稳定推进到更可训练的 TSPLIB 结构，也没有靠硬度单独换来稳定 gap 下降。硬度保持标签为 " + hardness + "。"
    return "结构标签为 " + structure + "，稳定 Original70 标签为 " + stable + "，硬度保持标签为 " + hardness + "。两者同时成立才说明联合目标转化成了训练收益；否则诊断空间的距离不能替代 Original70。"


def write_training_manifest(rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(OUT / "04_training_runs.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all", choices=["preflight", "evolve", "audit", "label", "train", "evaluate", "conclude", "all"])
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--replicate", default="ALL")
    args = parser.parse_args()
    ensure_dirs()
    if args.phase in {"preflight", "evolve", "all"}:
        protocol = preflight()
        config = load_distance_config()
        _manifest, ref_values = reference_f40()
        ref_z, eff = standardized_references(config, ref_values)
        models = hgb_models() if args.phase in {"evolve", "all"} else []
        if args.phase in {"evolve", "all"}:
            reps = REPLICATES if args.replicate == "ALL" else (args.replicate,)
            for replicate in reps:
                evolve_replicate(replicate, models, config, ref_z, eff)
            del models
            gc.collect()
        if args.phase == "preflight":
            print(json.dumps({"status": "PASS", "selection": protocol["selection_source_sha256"]}))
            return
    if args.phase in {"audit", "all"}:
        audit()
    if args.phase in {"label", "all"}:
        label_selected()
    if args.phase in {"train", "evaluate", "all"}:
        rows = []
        reps = REPLICATES if args.replicate == "ALL" else (args.replicate,)
        for replicate in reps:
            for depth in DEPTHS:
                if args.phase in {"train", "all"}:
                    rows.append(train_one(replicate, depth, args.device))
                    write_training_manifest(rows)
                if args.phase in {"evaluate", "all"}:
                    evaluate_one(replicate, depth, args.device)
    if args.phase in {"conclude", "all"}:
        completion = conclude()
        print(json.dumps(completion["labels"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
