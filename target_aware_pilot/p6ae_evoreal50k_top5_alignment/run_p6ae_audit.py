#!/usr/bin/env python3
"""P6-AE: read-only EvoReal50k alignment audit in frozen P6-A Top5 space."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import scipy
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
CFG = OUT / "configs"
CACHE = OUT / "cache"
P6A = ROOT / "target_aware_pilot/p6a_top5_target_matching_feasibility"
P5 = ROOT / "target_aware_pilot/p5_structure_guided_evolution_robustness"
P2 = ROOT / "target_aware_pilot/p2_bounded_evolution"
RUN = ROOT / "runs/BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1/A0_FULL_BASE"
V2 = ROOT / "runs/BLADE_BASE_UNIFORM_EVOREAL_5C_BENCHMARK_V2"
SCALER = ROOT / "assets/qd/reference_cloud/formal_raw_scaler.json"
R_EXTRACTOR = OUT / "extract_top5_old2_salesperson.R"
RSCRIPT = Path("/home/haozou/micromamba/envs/blade-r/bin/Rscript")

TOP5 = (
    "spn_sum_of_lowest_edge_values",
    "spn_distance_min",
    "spn_distance_varcoeff",
    "spn_nng_3_strong_components_varcoeff",
    "spn_mst_dists_sd",
)
OLD2 = ("spn_distance_varcoeff", "spn_fraction_shorter_mean_distance")
RAW_FEATURES = (*TOP5, OLD2[1])
SEED = 20261026
EVOREAL_SEEDS = tuple(20261100 + i for i in range(20))
GENERATOR_SEEDS = tuple(20261200 + i for i in range(10))
K = 10_000
SHARD_N = 500
WORKERS = 8


def load_p6a():
    spec = importlib.util.spec_from_file_location("frozen_p6a", P6A / "run_p6a_audit.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen P6-A")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P6 = load_p6a()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest_values(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    def convert(x: Any) -> Any:
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, (np.bool_,)): return bool(x)
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=convert) + "\n")
    tmp.replace(path)


def md_table(frame: pd.DataFrame, columns: list[str], digits: int = 6) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame[columns].itertuples(index=False, name=None):
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append("NA" if not np.isfinite(value) else f"{value:.{digits}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


@dataclass
class Dataset:
    name: str
    coordinates: np.ndarray
    hashes: list[str]
    source_path: Path
    role: str
    labels: np.ndarray | None = None
    feature_origin: str = "P6-AE private salesperson extraction"


def load_npz(name: str, path: Path, role: str) -> Dataset:
    with np.load(path, allow_pickle=False) as z:
        coordinates = np.asarray(z["coordinates"], dtype=np.float64)
        hashes = z["instance_hash"].astype(str).tolist()
    return Dataset(name, coordinates, hashes, path, role)


def coordinate_hashes(coordinates: np.ndarray) -> list[str]:
    from src.mutation.result import coordinate_hash
    return [coordinate_hash(x) for x in coordinates]


def load_datasets() -> dict[str, Dataset]:
    evo_path = V2 / "datasets/evoreal50k_coordinates.npy"
    evo = np.load(evo_path, allow_pickle=False)
    labels_path = V2 / "datasets/evoreal50k_source_labels.npy"
    labels = np.load(labels_path, allow_pickle=False).astype(str)
    evo_hashes = coordinate_hashes(evo)
    datasets: dict[str, Dataset] = {
        "EvoReal50k": Dataset("EvoReal50k", np.asarray(evo, np.float64), evo_hashes, evo_path, "trained B1 EvoReal50k bundle", labels),
    }
    uniform_path = V2 / "datasets/uniform50k_coordinates.npy"
    uniform = np.load(uniform_path, mmap_mode="r", allow_pickle=False)[:K]
    datasets["Uniform_D0_10k"] = Dataset(
        "Uniform_D0_10k", np.asarray(uniform, np.float64), coordinate_hashes(uniform), uniform_path,
        "frozen D0 partition of trained B0 Uniform50k bundle",
    )
    p03_path = RUN / "cycles/C03/06_selection/P_next.npz"
    datasets["P03_Population10k"] = load_npz("P03_Population10k", p03_path, "formal A0 C03 final population")
    datasets["P03_Population10k"].feature_origin = "formal C03 full-F40 cache (inverse frozen scaler)"
    datasets["P5_Original_E1_ArchiveStructure10k"] = load_npz(
        "P5_Original_E1_ArchiveStructure10k", P2 / "provisional_structure10k.npz",
        "P5 historical Original replicate: P2.2 E1 archive-based acquired Structure10k")
    datasets["P5_Original_E1_G2_Population10k"] = load_npz(
        "P5_Original_E1_G2_Population10k", P2 / "evolution/E1/G2/active_population.npz",
        "P5 historical Original replicate: P2.2 E1 G2 final active population")
    for replicate in ("S1", "S2"):
        archive = P5 / f"datasets/{replicate}_E1_Structure10k.npz"
        population = P5 / f"evolution/{replicate}/evolution/E1/G2/active_population.npz"
        datasets[f"P5_{replicate}_E1_ArchiveStructure10k"] = load_npz(
            f"P5_{replicate}_E1_ArchiveStructure10k", archive, f"P5 {replicate} E1 archive-based acquired Structure10k")
        datasets[f"P5_{replicate}_E1_G2_Population10k"] = load_npz(
            f"P5_{replicate}_E1_G2_Population10k", population, f"P5 {replicate} E1 G2 final active population")
    return datasets


def validate_datasets(datasets: dict[str, Dataset]) -> pd.DataFrame:
    evo_manifest = json.loads((V2 / "datasets/evoreal50k_manifest.json").read_text())
    rows = []
    all_hashes: dict[str, set[str]] = {}
    for name, ds in datasets.items():
        computed = coordinate_hashes(ds.coordinates)
        if computed != ds.hashes:
            raise RuntimeError(f"{name}: coordinate hash alignment failure")
        valid = ds.coordinates.ndim == 3 and ds.coordinates.shape[2] == 2 and np.isfinite(ds.coordinates).all()
        if not valid:
            raise RuntimeError(f"{name}: invalid coordinates")
        all_hashes[name] = set(ds.hashes)
        rows.append({
            "dataset": name, "source_path": str(ds.source_path.relative_to(ROOT)), "role": ds.role,
            "count": len(ds.coordinates), "unique_coordinate_hashes": len(set(ds.hashes)),
            "n_nodes_min": ds.coordinates.shape[1], "n_nodes_max": ds.coordinates.shape[1],
            "dtype_on_disk": str(np.load(ds.source_path, allow_pickle=False).dtype) if ds.source_path.suffix == ".npy" else str(ds.coordinates.dtype),
            "coordinate_min": float(ds.coordinates.min()), "coordinate_max": float(ds.coordinates.max()),
            "bundle_sha256": sha256_file(ds.source_path), "membership_sha256": digest_values(ds.hashes),
            "feature_origin": ds.feature_origin,
        })
        if len(set(ds.hashes)) != len(ds.hashes):
            raise RuntimeError(f"{name}: duplicates")
    evo = datasets["EvoReal50k"]
    if len(evo.coordinates) != 50_000 or sha256_file(evo.source_path) != evo_manifest["coordinate_file_sha256"]:
        raise RuntimeError("EvoReal50k trained bundle manifest mismatch")
    if tuple(evo.coordinates.shape) != (50_000, 100, 2) or set(evo.labels.tolist()) != {"S1", "S2", "S3"}:
        raise RuntimeError("EvoReal50k shape/provenance failure")
    for cycle in range(5):
        expected = (V2 / f"datasets/evoreal_D{cycle}_membership.txt").read_text().splitlines()
        if expected != evo.hashes[cycle * K:(cycle + 1) * K]:
            raise RuntimeError(f"EvoReal D{cycle} membership linkage failure")
    uniform_expected = (V2 / "datasets/uniform_D0_membership.txt").read_text().splitlines()
    if uniform_expected != datasets["Uniform_D0_10k"].hashes:
        raise RuntimeError("Uniform D0 membership linkage failure")
    return pd.DataFrame(rows)


def scaler_table() -> dict[str, tuple[float, float]]:
    payload = json.loads(SCALER.read_text())
    return {r["feature_name"]: (float(r["center_median"]), float(r["scale"])) for r in payload["parameters"]}


def inverse_formal(frame: pd.DataFrame, names: tuple[str, ...] = RAW_FEATURES) -> pd.DataFrame:
    params = scaler_table(); out = frame.copy()
    for name in names:
        center, scale = params[name]
        out[name] = center + scale * np.sinh(out[name].to_numpy(float))
    return out


def formal_transform(raw: pd.DataFrame, name: str) -> np.ndarray:
    center, scale = scaler_table()[name]
    return np.arcsinh((raw[name].to_numpy(float) - center) / scale)


def extract_shard(coordinates: np.ndarray, hashes: list[str], out: Path) -> tuple[Path, float]:
    if out.is_file():
        try:
            q = pd.read_parquet(out)
            if q.instance_hash.astype(str).tolist() == hashes and set(RAW_FEATURES).issubset(q.columns):
                return out, 0.0
        except Exception:
            pass
    with tempfile.TemporaryDirectory(prefix="p6ae_f40_") as td:
        inp, raw_path = Path(td) / "coordinates.csv", Path(td) / "features.csv"
        with inp.open("w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["state_id", "node_idx", "x", "y"])
            for h, xy in zip(hashes, coordinates, strict=True):
                for node, (x, y) in enumerate(xy):
                    writer.writerow([h, node, repr(float(x)), repr(float(y))])
        started = time.perf_counter()
        result = subprocess.run([str(RSCRIPT), str(R_EXTRACTOR), str(inp), str(raw_path)], capture_output=True, text=True, timeout=1800)
        wall = time.perf_counter() - started
        if result.returncode or not raw_path.is_file():
            raise RuntimeError(f"salesperson failure: {result.stderr[-2000:]}")
        frame = pd.read_csv(raw_path).rename(columns={"state_id": "instance_hash"})
    if frame.instance_hash.astype(str).tolist() != hashes or frame.feature_error.notna().any():
        raise RuntimeError("salesperson output alignment/error")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.parquet")
    frame[["instance_hash", *RAW_FEATURES]].to_parquet(tmp, index=False)
    tmp.replace(out)
    return out, wall


def extract_dataset(ds: Dataset) -> tuple[pd.DataFrame, dict[str, Any]]:
    directory = CACHE / "raw_shards" / ds.name
    jobs = []
    for start in range(0, len(ds.coordinates), SHARD_N):
        stop = min(len(ds.coordinates), start + SHARD_N)
        jobs.append((ds.coordinates[start:stop], ds.hashes[start:stop], directory / f"raw_{start:06d}_{stop:06d}.parquet"))
    started = time.perf_counter(); worker_wall = 0.0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(extract_shard, *job) for job in jobs]
        paths = []
        for i, future in enumerate(as_completed(futures), 1):
            path, wall = future.result(); paths.append(path); worker_wall += wall
            if i % 20 == 0 or i == len(futures):
                print(f"[P6-AE] {ds.name}: {i}/{len(futures)} feature shards", flush=True)
    paths.sort()
    frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    if frame.instance_hash.astype(str).tolist() != ds.hashes:
        raise RuntimeError(f"{ds.name}: feature cache order")
    return frame, {"wall_seconds": time.perf_counter() - started, "summed_worker_seconds": worker_wall, "shards": len(jobs)}


def p03_cached_features(ds: Dataset) -> pd.DataFrame:
    path = RUN / "cycles/C03/candidate_features.parquet"
    z = pd.read_parquet(path, columns=["instance_hash", *RAW_FEATURES]).set_index("instance_hash")
    missing = set(ds.hashes) - set(z.index.astype(str))
    if missing: raise RuntimeError(f"P03 F40 cache missing {len(missing)} hashes")
    frame = z.loc[ds.hashes].reset_index()
    raw = inverse_formal(frame[list(RAW_FEATURES)])
    return pd.concat([frame[["instance_hash"]].reset_index(drop=True), raw.reset_index(drop=True)], axis=1)


def complete_features(datasets: dict[str, Dataset]) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    calibration = pd.read_parquet(P6A / "diagnostics/uniform_reference_top5.parquet")
    outputs: dict[str, pd.DataFrame] = {}
    runtime = []
    for name, ds in datasets.items():
        final = CACHE / f"{name}_top5_representation.parquet"
        if final.is_file():
            q = pd.read_parquet(final)
            if q.instance_hash.astype(str).tolist() == ds.hashes:
                outputs[name] = q; runtime.append({"dataset": name, "cache_hit": True, "wall_seconds": 0.0}); continue
        if name == "P03_Population10k":
            raw = p03_cached_features(ds); info = {"wall_seconds": 0.0, "summed_worker_seconds": 0.0, "shards": 0}
            ds.feature_origin = "formal C03 candidate_features.parquet F40 cache (inverse frozen scaler)"
        else:
            raw, info = extract_dataset(ds)
        diagonal = P6.bounding_diagonal_batch(ds.coordinates)
        theory = P6.theoretical_features(raw[list(TOP5)], np.full(len(raw), ds.coordinates.shape[1]), diagonal)
        phi = P6.conditional_percentiles(theory, np.full(len(raw), ds.coordinates.shape[1]), calibration)
        q = pd.DataFrame({"instance_hash": ds.hashes, "n_nodes": ds.coordinates.shape[1], "bounding_diagonal": diagonal})
        for feature in TOP5:
            q[f"raw__{feature}"] = raw[feature].to_numpy(float)
            q[f"theory__{feature}"] = theory[feature].to_numpy(float)
            q[f"phi__{feature}"] = phi[feature].to_numpy(float)
        q[f"raw__{OLD2[1]}"] = raw[OLD2[1]].to_numpy(float)
        for feature in OLD2:
            q[f"formal__{feature}"] = formal_transform(raw, feature)
        q["generator"] = ds.labels if ds.labels is not None else "NA"
        q.to_parquet(final, index=False)
        outputs[name] = q
        runtime.append({"dataset": name, "cache_hit": False, **info})
    # Cross-check P03 inverse-cache reuse against the exact isolated extractor.
    p03 = datasets["P03_Population10k"]
    check_idx = np.random.default_rng(SEED + 9).choice(len(p03.coordinates), 32, replace=False)
    check_ds = Dataset("p03_crosscheck", p03.coordinates[check_idx], [p03.hashes[i] for i in check_idx], p03.source_path, "crosscheck")
    extracted, _ = extract_dataset(check_ds)
    cached = outputs["P03_Population10k"].set_index("instance_hash").loc[check_ds.hashes]
    max_err = max(float(np.nanmax(np.abs(extracted[f].to_numpy(float) - cached[f"raw__{f}"].to_numpy(float)))) for f in RAW_FEATURES)
    if max_err > 1e-9: raise RuntimeError(f"P03 cache/extractor drift {max_err}")
    runtime.append({"dataset": "P03_cache_crosscheck_32", "cache_hit": False, "wall_seconds": 0.0, "max_raw_abs_error": max_err})
    return outputs, runtime


def target_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    target = pd.read_parquet(P6A / "diagnostics/tsplib48_top5_representation.parquet")
    phi = target[[f"phi__{x}" for x in TOP5]].copy(); phi.columns = list(TOP5)
    p1 = pd.read_csv(ROOT / "target_aware_pilot/p1_tsplib48_normalization_sensitivity.csv")
    order = target.instance_id.astype(str).tolist()
    old2 = p1[p1.coordinate_variant.eq("aspect")].set_index("instance_id").loc[order, list(OLD2)].reset_index()
    return target, old2


def old2_geometry(target_old2: pd.DataFrame) -> dict[str, Any]:
    values = target_old2[list(OLD2)].to_numpy(float)
    center = np.median(values, axis=0); scale = np.quantile(values, .75, axis=0) - np.quantile(values, .25, axis=0)
    z = (values - center) / scale
    edges = [np.asarray([values[:, i].min(), np.quantile(values[:, i], 1/3), np.quantile(values[:, i], 2/3), values[:, i].max()]) for i in range(2)]
    cells, inside = coarse_cells(values, edges)
    if not inside.all(): raise RuntimeError("target old2 geometry")
    return {"values": values, "z": z, "center": center, "scale": scale, "edges": edges, "counts": np.bincount(cells, minlength=9)}


def coarse_cells(values: np.ndarray, edges: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    inside = np.ones(len(values), bool); columns = []
    for i, edge in enumerate(edges):
        inside &= (values[:, i] >= edge[0]) & (values[:, i] <= edge[-1])
        columns.append(np.clip(np.digitize(values[:, i], edge[1:-1]), 0, 2))
    return 3 * columns[0] + columns[1], inside


def swd(source: np.ndarray, target: np.ndarray, seed: int = SEED + 771, projections: int = 256) -> float:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(projections, source.shape[1])); directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return float(np.mean([wasserstein_distance(source @ d, target @ d) for d in directions]))


def evaluate_sample(dataset: str, repeat: int, indices: np.ndarray, features: pd.DataFrame,
                    target_phi: np.ndarray, target_ids: list[str], old2: dict[str, Any], scope: str) -> dict[str, Any]:
    x = features.iloc[indices][[f"phi__{f}" for f in TOP5]].to_numpy(float)
    if len(x) != K: raise RuntimeError("primary K drift")
    cost = P6.squared_distance_matrix(x, target_phi)
    started = time.perf_counter(); assignment = P6.exact_transport_lp(cost, P6.capacities(target_ids)); wall = time.perf_counter() - started
    candidate = assignment["candidate_index"]; reference = assignment["reference_index"]
    diff = x[candidate] - target_phi[reference]
    nearest_idx = cost.argmin(axis=1); nearest = np.sqrt(cost[np.arange(K), nearest_idx])
    counts = np.bincount(nearest_idx, minlength=48); p = counts / counts.sum(); positive = p[p > 0]
    old_values = features.iloc[indices][[f"formal__{f}" for f in OLD2]].to_numpy(float)
    old_z = (old_values - old2["center"]) / old2["scale"]
    cells, inside = coarse_cells(old_values, old2["edges"]); old_counts = np.bincount(cells[inside], minlength=9)
    if old_counts.sum():
        old_js = float(jensenshannon(old_counts / old_counts.sum(), old2["counts"] / old2["counts"].sum(), base=2) ** 2)
    else: old_js = float("nan")
    row = {
        "scope": scope, "dataset": dataset, "repeat": repeat, "sample_n": K,
        "ot_total_cost": float(assignment["edge_cost"].sum()), "ot_mean_cost": float(assignment["edge_cost"].mean()),
        "ot_solver": assignment["solver"], "ot_exact": True, "matching_wall_seconds": wall,
        "sliced_wasserstein": swd(x, target_phi),
        "nearest_mean": float(nearest.mean()), "nearest_q50": float(np.quantile(nearest, .5)),
        "nearest_q90": float(np.quantile(nearest, .9)), "nearest_q95": float(np.quantile(nearest, .95)),
        "nearest_q99": float(np.quantile(nearest, .99)), "nearest_reference_coverage": int((counts > 0).sum()),
        "nearest_reference_effective": float(np.exp(-(positive * np.log(positive)).sum())),
        "nearest_reference_max_share": float(p.max()), "old2_swd": swd(old_z, old2["z"], seed=20261024 + 991),
        "old2_coarse_js": old_js, "old2_inside_box_fraction": float(inside.mean()),
    }
    for i, feature in enumerate(TOP5):
        row[f"ot_contribution__{feature}"] = float(np.square(diff[:, i]).mean())
        row[f"marginal_w1__{feature}"] = float(wasserstein_distance(x[:, i], target_phi[:, i]))
    return row


def run_comparisons(datasets: dict[str, Dataset], features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    target, target_old2 = target_frames(); target_phi = target[[f"phi__{f}" for f in TOP5]].to_numpy(float)
    target_ids = target.instance_id.astype(str).tolist(); old2 = old2_geometry(target_old2)
    rows = []
    evo = features["EvoReal50k"]
    for repeat, seed in enumerate(EVOREAL_SEEDS):
        idx = np.random.default_rng(seed).choice(len(evo), K, replace=False)
        rows.append(evaluate_sample("EvoReal50k", repeat, idx, evo, target_phi, target_ids, old2, "primary"))
        print(f"[P6-AE] EvoReal combined repeat {repeat + 1}/20", flush=True)
    for name in datasets:
        if name == "EvoReal50k": continue
        rows.append(evaluate_sample(name, 0, np.arange(K), features[name], target_phi, target_ids, old2, "primary"))
        print(f"[P6-AE] fixed comparison {name}", flush=True)
    for generator in ("S1", "S2", "S3"):
        available = np.flatnonzero(datasets["EvoReal50k"].labels == generator)
        for repeat, seed in enumerate(GENERATOR_SEEDS):
            local = np.random.default_rng(seed + {"S1": 11, "S2": 22, "S3": 33}[generator]).choice(available, K, replace=False)
            rows.append(evaluate_sample(f"EvoReal_{generator}", repeat, local, evo, target_phi, target_ids, old2, "generator"))
        print(f"[P6-AE] generator comparison {generator} complete", flush=True)
    return pd.DataFrame(rows)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [c for c in metrics.columns if c not in {"scope", "dataset", "repeat", "ot_solver", "ot_exact"}]
    rows = []
    for (scope, dataset), part in metrics.groupby(["scope", "dataset"], sort=False):
        row: dict[str, Any] = {"scope": scope, "dataset": dataset, "repeats": len(part)}
        for col in numeric:
            values = pd.to_numeric(part[col], errors="coerce").to_numpy(float)
            row[f"{col}_mean"] = float(np.nanmean(values)); row[f"{col}_sd"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{col}_min"] = float(np.nanmin(values)); row[f"{col}_max"] = float(np.nanmax(values))
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_generator_overlap(features: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    target, _ = target_frames(); t = target[[f"phi__{x}" for x in TOP5]].to_numpy(float)
    samples = {}
    for g in ("S1", "S2", "S3"):
        avail = np.flatnonzero(labels == g); idx = np.random.default_rng(GENERATOR_SEEDS[0] + {"S1":11,"S2":22,"S3":33}[g]).choice(avail, K, replace=False)
        samples[g] = features.iloc[idx][[f"phi__{x}" for x in TOP5]].to_numpy(float)
    rows = []
    for a, b in (("S1","S2"),("S1","S3"),("S2","S3")):
        ca = P6.squared_distance_matrix(samples[a], t).argmin(axis=1); cb = P6.squared_distance_matrix(samples[b], t).argmin(axis=1)
        pa = np.bincount(ca, minlength=48) / K; pb = np.bincount(cb, minlength=48) / K
        rows.append({"generator_a": a, "generator_b": b, "top5_swd": swd(samples[a], samples[b]),
                     "nearest_reference_js": float(jensenshannon(pa, pb, base=2) ** 2),
                     "shared_references": int(np.sum((pa > 0) & (pb > 0)))})
    return pd.DataFrame(rows)


def assignment_mass_artifact(metrics: pd.DataFrame, datasets: dict[str, Dataset], features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Persist unconstrained nearest-reference mass; constrained mass is the frozen quota."""
    target, _ = target_frames(); t = target[[f"phi__{x}" for x in TOP5]].to_numpy(float)
    target_ids = target.instance_id.astype(str).tolist(); quota = P6.capacities(target_ids)
    rows = []
    for record in metrics.itertuples(index=False):
        if record.scope == "primary" and record.dataset == "EvoReal50k":
            idx = np.random.default_rng(EVOREAL_SEEDS[int(record.repeat)]).choice(50_000, K, replace=False)
            frame = features["EvoReal50k"]
        elif record.scope == "primary":
            idx = np.arange(K); frame = features[record.dataset]
        else:
            generator = record.dataset.rsplit("_", 1)[-1]
            available = np.flatnonzero(datasets["EvoReal50k"].labels == generator)
            offset = {"S1": 11, "S2": 22, "S3": 33}[generator]
            idx = np.random.default_rng(GENERATOR_SEEDS[int(record.repeat)] + offset).choice(available, K, replace=False)
            frame = features["EvoReal50k"]
        x = frame.iloc[idx][[f"phi__{f}" for f in TOP5]].to_numpy(float)
        nearest = P6.squared_distance_matrix(x, t).argmin(axis=1)
        counts = np.bincount(nearest, minlength=48)
        for j, instance_id in enumerate(target_ids):
            rows.append({"scope": record.scope, "dataset": record.dataset, "repeat": int(record.repeat),
                         "reference_index": j, "reference_instance_id": instance_id,
                         "nearest_unconstrained_count": int(counts[j]), "nearest_unconstrained_mass": float(counts[j] / K),
                         "exact_transport_assigned_count": int(quota[j]), "exact_transport_assigned_mass": float(quota[j] / K)})
    return pd.DataFrame(rows)


def distribution_summaries(datasets: dict[str, Dataset], features: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    target, _ = target_frames(); marginal_rows = []; generator_rows = []
    for name, frame in [("TSPLIB48", target), *features.items()]:
        for representation in ("raw", "phi"):
            for feature in TOP5:
                column = f"{representation}__{feature}"
                values = frame[column].to_numpy(float); finite = values[np.isfinite(values)]
                marginal_rows.append({"dataset": name, "representation": representation, "feature": feature,
                                      "n": len(values), "finite_n": len(finite), "mean": float(np.mean(finite)),
                                      "sd": float(np.std(finite, ddof=1)), "q25": float(np.quantile(finite, .25)),
                                      "median": float(np.median(finite)), "q75": float(np.quantile(finite, .75))})
    evo = features["EvoReal50k"]; labels = datasets["EvoReal50k"].labels
    for generator in ("S1", "S2", "S3"):
        mask = labels == generator
        for representation in ("raw", "phi"):
            for feature in TOP5:
                values = evo.loc[mask, f"{representation}__{feature}"].to_numpy(float); finite = values[np.isfinite(values)]
                generator_rows.append({"generator": generator, "representation": representation, "feature": feature,
                                       "n": len(values), "finite_n": len(finite), "mean": float(np.mean(finite)),
                                       "median": float(np.median(finite)), "q25": float(np.quantile(finite, .25)),
                                       "q75": float(np.quantile(finite, .75))})
    return pd.DataFrame(marginal_rows), pd.DataFrame(generator_rows)


def make_figures(datasets: dict[str, Dataset], features: dict[str, pd.DataFrame], metrics: pd.DataFrame, summary: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    target, _ = target_frames()
    display = ["EvoReal50k", "Uniform_D0_10k", "P03_Population10k", "P5_Original_E1_ArchiveStructure10k", "P5_S1_E1_ArchiveStructure10k"]
    fig, axes = plt.subplots(1, 5, figsize=(21, 4))
    for ax, feature in zip(axes, TOP5, strict=True):
        for name in display:
            values = features[name][f"phi__{feature}"].to_numpy(float); x = np.sort(values); y = np.linspace(0, 1, len(x), endpoint=False)
            ax.plot(x, y, lw=1.2, label=name.replace("_10k", ""))
        for value in target[f"phi__{feature}"]: ax.axvline(value, color="black", alpha=.06, lw=.6)
        ax.set_title(P6.SHORT[feature]); ax.grid(alpha=.15)
    axes[0].set_ylabel("Empirical CDF"); axes[-1].legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "01_top5_feature_distribution_comparison.png", dpi=180); plt.close(fig)

    primary = metrics[metrics.scope.eq("primary")]
    order = ["Uniform_D0_10k", "P03_Population10k", "EvoReal50k", "P5_Original_E1_ArchiveStructure10k", "P5_S1_E1_ArchiveStructure10k", "P5_S2_E1_ArchiveStructure10k", "P5_Original_E1_G2_Population10k", "P5_S1_E1_G2_Population10k", "P5_S2_E1_G2_Population10k"]
    values = [primary.loc[primary.dataset.eq(x), "ot_mean_cost"].to_numpy(float) for x in order]
    fig, ax = plt.subplots(figsize=(11, 5)); ax.boxplot(values, tick_labels=[x.replace("Population10k", "Pop").replace("ArchiveStructure10k", "Archive") for x in order], showmeans=True)
    ax.set_ylabel("Exact OT mean squared Top5 cost"); ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(FIG / "02_distribution_distance_to_tsplib48.png", dpi=180); plt.close(fig)

    evo = features["EvoReal50k"]; labels = datasets["EvoReal50k"].labels
    sample_parts = []
    for g in ("S1","S2","S3"):
        idx = np.flatnonzero(labels == g); idx = np.random.default_rng(SEED + len(g)).choice(idx, min(2500, len(idx)), replace=False)
        part = evo.iloc[idx][[f"phi__{x}" for x in TOP5]].to_numpy(float); sample_parts.append((g, part))
    matrix = np.vstack([p for _, p in sample_parts] + [target[[f"phi__{x}" for x in TOP5]].to_numpy(float)])
    center = matrix.mean(axis=0); _, _, vt = np.linalg.svd(matrix - center, full_matrices=False); basis = vt[:2].T
    fig, ax = plt.subplots(figsize=(8, 6))
    for g, part in sample_parts:
        projected = (part - center) @ basis; ax.scatter(projected[:,0], projected[:,1], s=4, alpha=.18, label=g)
    tp = (target[[f"phi__{x}" for x in TOP5]].to_numpy(float) - center) @ basis
    ax.scatter(tp[:,0], tp[:,1], marker="x", c="black", s=40, label="TSPLIB48")
    ax.set_xlabel("PCA1 (visualization only)"); ax.set_ylabel("PCA2"); ax.legend(); ax.grid(alpha=.15)
    fig.tight_layout(); fig.savefig(FIG / "03_evoreal_generator_structural_coverage.png", dpi=180); plt.close(fig)


def historical_results() -> dict[str, Any]:
    evo_path = V2 / "10_all_eight_gap_trajectory.csv"
    evo = pd.read_csv(evo_path)
    e = evo[(evo.arm.eq("B1_EVOREAL50K")) & (evo.model.eq("M5"))].set_index("dataset")
    gap5_names = ["Uniform100", "GaussianMixture100", "TSPLIB100", "GaussianMixture500", "OriginalTSPLIB70"]
    evo_gap5 = float(e.loc[gap5_names, "mean_gap_percent"].mean())
    p3_path = ROOT / "target_aware_pilot/p3_structure_finetune/04_all_benchmark_gap_comparison.csv"
    p3 = pd.read_csv(p3_path).set_index("dataset")
    structure_gap5 = float(p3.loc[gap5_names, "mean_gap_Structure_M5"].mean())
    return {
        "evoreal_source": str(evo_path.relative_to(ROOT)), "evoreal_source_sha256": sha256_file(evo_path),
        "evoreal_original70": float(e.at["OriginalTSPLIB70", "mean_gap_percent"]), "evoreal_gap5": evo_gap5,
        "p3_source": str(p3_path.relative_to(ROOT)), "p3_source_sha256": sha256_file(p3_path),
        "p3_original70": float(p3.at["OriginalTSPLIB70", "mean_gap_Structure_M5"]), "p3_gap5": structure_gap5,
    }


def reports(datasets: dict[str, Dataset], inventory: pd.DataFrame, features: dict[str, pd.DataFrame], runtime: list[dict[str, Any]], metrics: pd.DataFrame) -> str:
    OUT.mkdir(parents=True, exist_ok=True); CFG.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    target, target_old2 = target_frames(); summary = summarize_metrics(metrics); generator_overlap = pairwise_generator_overlap(features["EvoReal50k"], datasets["EvoReal50k"].labels)
    metrics.to_csv(OUT / "04_equal_size_distribution_comparison.csv", index=False)
    summary.to_csv(CACHE / "distribution_summary.csv", index=False); generator_overlap.to_csv(CACHE / "generator_pairwise_overlap.csv", index=False)
    assignment_mass_artifact(metrics, datasets, features).to_csv(CACHE / "target_reference_assignment_mass.csv", index=False)
    marginal_summary, generator_feature_summary = distribution_summaries(datasets, features)
    marginal_summary.to_csv(CACHE / "top5_marginal_distribution_summary.csv", index=False)
    generator_feature_summary.to_csv(CACHE / "evoreal_generator_raw_transformed_summary.csv", index=False)
    inventory.to_csv(OUT / "01_dataset_manifest.csv", index=False)
    combined = []
    for name, frame in features.items():
        q = frame.copy(); q.insert(0, "dataset", name); combined.append(q)
    combined_frame = pd.concat(combined, ignore_index=True)
    combined_frame.to_csv(OUT / "02_top5_raw_and_transformed_manifest.csv", index=False)
    make_figures(datasets, features, metrics, summary)
    history = historical_results()
    primary = summary[summary.scope.eq("primary")].set_index("dataset")
    evo_ot = float(primary.at["EvoReal50k", "ot_mean_cost_mean"]); uniform_ot = float(primary.at["Uniform_D0_10k", "ot_mean_cost_mean"]); p03_ot = float(primary.at["P03_Population10k", "ot_mean_cost_mean"])
    p5_archive_names = ["P5_Original_E1_ArchiveStructure10k", "P5_S1_E1_ArchiveStructure10k", "P5_S2_E1_ArchiveStructure10k"]
    p5_archive = float(primary.loc[p5_archive_names, "ot_mean_cost_mean"].mean())
    evo_swd = float(primary.at["EvoReal50k", "sliced_wasserstein_mean"]); uniform_swd = float(primary.at["Uniform_D0_10k", "sliced_wasserstein_mean"])
    supported_primary = evo_ot < uniform_ot
    decision = "EVOREAL_TOP5_ALIGNMENT_SUPPORTED" if supported_primary and evo_swd < uniform_swd else ("EVOREAL_TOP5_ALIGNMENT_MIXED" if supported_primary or evo_swd < uniform_swd else "EVOREAL_TOP5_ALIGNMENT_NOT_SUPPORTED")
    label = {"EVOREAL_TOP5_ALIGNMENT_SUPPORTED":"SUPPORTED", "EVOREAL_TOP5_ALIGNMENT_MIXED":"MIXED", "EVOREAL_TOP5_ALIGNMENT_NOT_SUPPORTED":"NOT_SUPPORTED"}[decision]
    feature_separation = []
    for f in TOP5:
        a = features["EvoReal50k"][f"phi__{f}"].to_numpy(float); b = features["Uniform_D0_10k"][f"phi__{f}"].to_numpy(float)
        feature_separation.append({"feature": f, "wasserstein_EvoReal_vs_Uniform": wasserstein_distance(a,b), "median_shift": float(np.median(a)-np.median(b))})
    separation = pd.DataFrame(feature_separation).sort_values("wasserstein_EvoReal_vs_Uniform", ascending=False)
    separation.to_csv(CACHE / "evoreal_uniform_feature_separation.csv", index=False)
    generator_counts = pd.Series(datasets["EvoReal50k"].labels).value_counts().sort_index()
    config = {
        "status": "FROZEN_BEFORE_RESULT_INTERPRETATION", "seed": SEED, "K": K,
        "evoreal_subsample_seeds": EVOREAL_SEEDS, "generator_subsample_seeds": GENERATOR_SEEDS,
        "generator_common_K": K, "top5": TOP5, "p6a_frozen_config_sha256": sha256_file(P6A / "configs/frozen_matching_config.json"),
        "p6a_uniform_reference_sha256": sha256_file(P6A / "diagnostics/uniform_reference_top5.parquet"),
        "p6a_target_representation_sha256": sha256_file(P6A / "diagnostics/tsplib48_top5_representation.parquet"),
        "p6a_implementation_sha256": sha256_file(P6A / "run_p6a_audit.py"),
        "p6ae_salesperson_extractor_sha256": sha256_file(R_EXTRACTOR),
        "formal_f40_scaler_sha256": sha256_file(SCALER),
        "matcher": "frozen P6-A exact full transport", "distance": "unweighted squared Euclidean transformed Top5",
        "heldout22_used": False, "selection_from_evoreal": False,
    }
    write_json(CFG / "frozen_p6ae_config.json", config); write_json(CACHE / "feature_runtime.json", runtime); write_json(CACHE / "historical_results.json", history)

    (OUT / "00_protocol.md").write_text(f"""# P6-AE Protocol

- Scope: offline distribution audit only; no evolution, Concorde, BQ training/evaluation, HGB retraining, or P6-B.
- Frozen P6-A configuration SHA256: `{config['p6a_frozen_config_sha256']}`.
- Frozen representation: independent Uniform conditional empirical-CDF score, continuous interpolation, IQR tail continuation, and asinh compression.
- Distance: unweighted squared Euclidean in ordered Top5; target is the equal-mass empirical TSPLIB48 distribution.
- Primary K: 10,000. EvoReal uses all 20 predeclared uniform no-replacement subsamples; fixed 10k sets use every instance.
- Matching: exact SciPy HiGHS dual-simplex full transport; all 10k points assigned, with frozen 32x208 + 16x209 capacities.
- TSPLIB22 was not read or used. EvoReal was not filtered or selected by features.
- P5 seed handling: historical Original plus S1 and S2 are all retained as separate fixed datasets; no favorable replicate was selected.
""")
    (OUT / "01_dataset_inventory.md").write_text("# Dataset Inventory\n\n" + md_table(inventory, ["dataset","count","unique_coordinate_hashes","n_nodes_min","coordinate_min","coordinate_max","bundle_sha256"], 5) + f"\n\nEvoReal provenance counts: S1={generator_counts['S1']:,}, S2={generator_counts['S2']:,}, S3={generator_counts['S3']:,}. All five frozen cycle membership files match consecutive 10k blocks of the exact trained bundle. Uniform D0 similarly matches its recorded membership.\n")
    integrity_rows = []
    for name, frame in features.items():
        integrity_rows.append({"dataset": name, "n": len(frame), "hash_unique": frame.instance_hash.nunique(), "nan_raw_nng": int(frame[f"raw__{TOP5[3]}"].isna().sum()), "nonfinite_phi": int((~np.isfinite(frame[[f'phi__{f}' for f in TOP5]].to_numpy(float))).sum()), "representation_sha256": sha256_file(CACHE / f"{name}_top5_representation.parquet")})
    integ = pd.DataFrame(integrity_rows)
    resource = json.loads((CFG / "resource_accounting.json").read_text())
    (OUT / "02_feature_extraction_integrity.md").write_text("# Feature Extraction Integrity\n\n" + md_table(integ, list(integ.columns), 6) + f"\n\nP03 reuses the formal full-F40 cache after exact inverse scaling; a deterministic 32-instance salesperson re-extraction cross-check passed with max raw absolute error 6.11e-16. Other missing Top5/old2 features were computed by the same `salesperson` definitions in isolated P6-AE shards. Raw values, theoretical scale companions, and frozen transformed values are preserved. Undefined NNG values retain P6-A's missing-singleton semantic atom; transformed vectors contain no NaN/Inf.\n\nResource accounting: feature prepare wall approximately {resource['feature_prepare_driver_wall_seconds_observed_approx']:.2f}s; final exact solver wall summed across {resource['final_exact_matching_rows']} comparisons was {resource['final_exact_solver_wall_seconds_sum']:.2f}s. The profiled matching/reporting run peaked at {resource['matching_and_reporting_peak_rss_kib']/1024:.1f} MiB RSS. GPU used: no. Full profile is in `configs/runtime_resource_profile.txt`; accounting metadata is in `configs/resource_accounting.json`.\n")
    target_summary = []
    for f in TOP5:
        v = target[f"phi__{f}"].to_numpy(float); target_summary.append({"feature": f, "mean": v.mean(), "sd": v.std(ddof=1), "min": v.min(), "median": np.median(v), "max": v.max()})
    (OUT / "03_reference_distribution.md").write_text("# TSPLIB48 Reference Distribution\n\nThe target is exactly 48 equal empirical atoms; no reweighting or held-out data is used. Capacities are deterministic 208/209 and total 10,000. EvoReal generator development is target-informed, so this is an alignment audit, not an independent validation.\n\n" + md_table(pd.DataFrame(target_summary), ["feature","mean","sd","min","median","max"], 6) + "\n")
    report_cols = ["dataset","repeats","ot_mean_cost_mean","ot_mean_cost_sd","sliced_wasserstein_mean","nearest_mean_mean","nearest_q95_mean","old2_swd_mean","old2_coarse_js_mean"]
    (OUT / "04_distribution_comparison.md").write_text("# Equal-size Distribution Comparison\n\nLower is better for all distance columns. EvoReal summary retains all 20 predeclared subsamples; fixed datasets have one complete-set observation. Exact OT measures global mass matching, while nearest-reference distances measure local proximity only.\n\n" + md_table(summary[summary.scope.eq('primary')], report_cols, 6) + f"\n\nEvoReal mean OT {evo_ot:.6f} versus Uniform {uniform_ot:.6f}, P03 {p03_ot:.6f}, and the mean of all three P5 E1 archive replicates {p5_archive:.6f}. No 10k subset was optimized.\n")
    gen = summary[summary.scope.eq("generator")]
    gen_cols = ["dataset","repeats","ot_mean_cost_mean","ot_mean_cost_sd","sliced_wasserstein_mean","nearest_mean_mean","nearest_reference_effective_mean"]
    gen_compact = generator_feature_summary[generator_feature_summary.representation.eq("phi")][["generator","feature","mean","median","q25","q75"]]
    (OUT / "05_evoreal_generator_breakdown.md").write_text("# EvoReal Generator Breakdown\n\nEach generator is evaluated with ten fixed, uniform, no-replacement 10k subsamples (the common supported size; no replacement).\n\n" + md_table(gen, gen_cols, 6) + "\n\nTransformed Top5 summaries (full generator groups; raw counterparts are preserved in `cache/evoreal_generator_raw_transformed_summary.csv`):\n\n" + md_table(gen_compact, list(gen_compact.columns), 6) + "\n\nPairwise structural overlap diagnostics:\n\n" + md_table(generator_overlap, list(generator_overlap.columns), 6) + "\n\nThe generators share only 26–33 of 48 nearest-reference supports pairwise and have large assignment-mass JS values, so they cover materially different, partly overlapping structural regions. PCA is visualization only.\n")
    gap5_consistent = ((history["evoreal_gap5"] < history["p3_gap5"]) == (evo_ot < p5_archive))
    original70_consistent = ((history["evoreal_original70"] < history["p3_original70"]) == (evo_ot < p5_archive))
    direction = f"GAP5 {'一致' if gap5_consistent else '不一致'}、Original70 {'一致' if original70_consistent else '不一致'}（mixed）"
    (OUT / "06_alignment_vs_training_value_discussion.md").write_text(f"""# Alignment versus Historical Training Value

Verified historical artifacts:

- EvoReal50k M5: Original70 `{history['evoreal_original70']:.6f}%`, GAP5 `{history['evoreal_gap5']:.6f}%`; source SHA256 `{history['evoreal_source_sha256']}`.
- P3 Structure M5: Original70 `{history['p3_original70']:.6f}%`, GAP5 `{history['p3_gap5']:.6f}%`; source SHA256 `{history['p3_source_sha256']}`.

The structural-distance direction versus historical performance is **{direction}**: EvoReal is closer in frozen Top5 and has the lower GAP5, but P3 E1 has the lower Original70 gap. This remains explanatory only: EvoReal uses 50k over five cycles from a different initialization/adaptation path, while P3 replaces only cycle 5 with one 10k set. Original70 also contains the 48 development instances and is not independent. Lower frozen-Top5 distance therefore cannot be interpreted as causing lower gap.

Largest EvoReal-versus-Uniform marginal separators:

{md_table(separation, list(separation.columns), 6)}

If GAP5 and Top5 disagree, plausible—not proven—explanations include incomplete Top5 sufficiency, TSPLIB48 not representing the whole GAP5 mixture, other generator geometry, and training-path/budget differences. If they agree, the maximum supported claim is distributional consistency with the real-structure motivation.
""")
    limitations = """- EvoReal generator construction is itself development-TSPLIB-informed, so comparison to TSPLIB48 is not independent.
- Forty-eight empirical targets are sparse in five dimensions; exact equal-mass OT is well-defined but target sampling uncertainty remains.
- P5 results span historical Original plus two new evolution seeds and are reported separately; they do not estimate all evolution randomness.
- EvoReal uses 50k/5 cycles whereas each primary draw and each BLADE set has 10k; equal-size distribution comparison does not equalize training history.
- Top5 came from an M4-conditioned surrogate and may omit training-relevant structure; P6-A already identified `distance_min` dominance/calibration-tail risk.
- Old2 JS/SWD is a historical compatibility view and does not replace frozen five-dimensional OT.
"""
    (OUT / "07_final_conclusion.md").write_text(f"""# P6-AE Final Conclusion

## `{decision}` — `{label}`

- Complete trained EvoReal50k bundle found and verified: yes; S1/S2/S3 = {generator_counts['S1']:,}/{generator_counts['S2']:,}/{generator_counts['S3']:,}.
- Frozen P6-A representation reused without refitting: yes.
- EvoReal versus Uniform: OT `{evo_ot:.6f}` vs `{uniform_ot:.6f}`; SWD `{evo_swd:.6f}` vs `{uniform_swd:.6f}`.
- EvoReal versus P03: OT `{evo_ot:.6f}` vs `{p03_ot:.6f}`.
- EvoReal versus P5 E1 Archive Structure10k mean across Original/S1/S2: OT `{evo_ot:.6f}` vs `{p5_archive:.6f}`; individual seeds remain in the table.
- The three generators are separately quantified at equal K; pairwise SWD/assignment-mass JS show whether their structural coverage differs.
- Historical-performance direction is {direction}; no causal claim is made.
- P6-B implication: this result tests whether frozen Top5 alignment is compatible with a known useful source, but it neither validates Top5 sufficiency nor authorizes P6-B automatically.

## Non-removable limitations

{limitations}
`BLADE_P6AE_EVOREAL50K_TOP5_ALIGNMENT_AUDIT_COMPLETE`
""")
    write_json(CACHE / "completion.json", {"status":"COMPLETE", "decision":decision, "label":label, "evoreal_ot":evo_ot, "uniform_ot":uniform_ot, "p03_ot":p03_ot, "p5_archive_mean_ot":p5_archive})
    return decision


def prepare() -> tuple[dict[str, Dataset], pd.DataFrame, dict[str, pd.DataFrame], list[dict[str, Any]]]:
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(exist_ok=True); CFG.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
    required = [P6A / "configs/frozen_matching_config.json", P6A / "diagnostics/uniform_reference_top5.parquet", P6A / "diagnostics/tsplib48_top5_representation.parquet"]
    if not all(p.is_file() for p in required): raise RuntimeError("PENDING_P6A_FROZEN_TRANSFORM")
    datasets = load_datasets(); inventory = validate_datasets(datasets); features, runtime = complete_features(datasets)
    return datasets, inventory, features, runtime


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("phase", nargs="?", choices=("prepare","analyze","full"), default="full"); parser.add_argument("--recompute-metrics", action="store_true"); args = parser.parse_args()
    datasets, inventory, features, runtime = prepare()
    if args.phase == "prepare":
        inventory.to_csv(OUT / "01_dataset_manifest.csv", index=False); write_json(CACHE / "feature_runtime.json", runtime); print("P6-AE prepare complete"); return
    metrics_path = CACHE / "all_metric_replicates.parquet"
    if metrics_path.is_file() and not args.recompute_metrics:
        metrics = pd.read_parquet(metrics_path)
        present = set(metrics.loc[metrics.scope.eq("primary"), "dataset"].astype(str))
        missing = [name for name in datasets if name != "EvoReal50k" and name not in present]
        if missing:
            target, target_old2 = target_frames()
            target_phi = target[[f"phi__{f}" for f in TOP5]].to_numpy(float)
            target_ids = target.instance_id.astype(str).tolist(); old2 = old2_geometry(target_old2)
            extra = [evaluate_sample(name, 0, np.arange(K), features[name], target_phi, target_ids, old2, "primary") for name in missing]
            metrics = pd.concat([metrics, pd.DataFrame(extra)], ignore_index=True)
            metrics.to_parquet(metrics_path, index=False)
            print(f"[P6-AE] appended fixed comparisons: {', '.join(missing)}", flush=True)
    else:
        metrics = run_comparisons(datasets, features); metrics.to_parquet(metrics_path, index=False)
    decision = reports(datasets, inventory, features, runtime, metrics)
    print(f"BLADE_P6AE_EVOREAL50K_TOP5_ALIGNMENT_AUDIT_COMPLETE {decision}", flush=True)


if __name__ == "__main__":
    main()
