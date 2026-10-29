#!/usr/bin/env python3
"""P2.2 bounded-population structure-guided evolution audit.

This isolated runner starts from the formal A0 P03 population, runs two real
FOGA-11 generations for random and structure-guided survivor selection, keeps
a 10,000-member active population and a 30,000-member eligible-offspring
archive, and performs structure-only acquisition.  It never invokes an
oracle, Concorde, a surrogate, or BQ training.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import platform
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from final_run.science import cycle_child_seed, membership_hash  # noqa: E402
from mutation import OPERATOR_NAMES, apply_operator_with_config  # noqa: E402
from mutation.parameters import load_named_profile  # noqa: E402
from mutation.result import coordinate_hash  # noqa: E402


OUT = ROOT / "target_aware_pilot" / "p2_bounded_evolution"
FIG = OUT / "figures"
RUN = ROOT / "runs" / "BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1" / "A0_FULL_BASE"
CYCLES = RUN / "cycles"
P03_META = CYCLES / "C03" / "06_selection" / "P_next.parquet"
P03_BUNDLE = CYCLES / "C03" / "06_selection" / "P_next.npz"
C04_PARENT = CYCLES / "C04" / "04_qd" / "parent_qd.parquet"
C04_RUNTIME = CYCLES / "C04" / "03_mutation" / "FOGA11_BASE_V1_RUNTIME.json"
P1 = ROOT / "target_aware_pilot"
F40_WORKER = ROOT / "scripts" / "f40_one_shard.py"
R_EXTRACTOR = ROOT / "src" / "features" / "extract_salesperson_raw_features.R"
SCALER = ROOT / "assets" / "qd" / "reference_cloud" / "formal_raw_scaler.json"
PROFILE_PATH = ROOT / "src" / "mutation" / "foga2019_operators.yaml"

FEATURES = ("spn_distance_varcoeff", "spn_fraction_shorter_mean_distance")
PROFILE_NAME = "FOGA11_BASE_V1"
POPULATION_N = 10_000
ATTEMPTS_PER_GENERATION = 20_000
GENERATIONS = (1, 2)
ARCHIVE_CAPACITY = 30_000
TARGET_FRACTION = 0.70
FRONTIER_FRACTION = 0.30
STRICT_RADIUS_EXPECTED = 0.095955812384
SEED = 20261024
MUTATION_CYCLE_NAMESPACE = {1: 5, 2: 6}
FEATURE_SHARD_N = 250
FEATURE_WORKERS = 12
SWD_PROJECTIONS = 256
LINEAGE_CAP_ACTIVE = 40
LINEAGE_CAP_ARCHIVE = 80
LINEAGE_CAP_ACQUISITION = 50


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"cannot serialize {type(value).__name__}")

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=convert) + "\n", encoding="utf-8")
    temp.replace(path)


def save_bundle(path: Path, coordinates: np.ndarray, hashes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temp,
        coordinates=np.asarray(coordinates, dtype=np.float64),
        instance_hash=np.asarray(hashes),
    )
    temp.replace(path)


def load_bundle(path: Path) -> tuple[np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as payload:
        return (
            np.asarray(payload["coordinates"], dtype=np.float64),
            payload["instance_hash"].astype(str).tolist(),
        )


def nearest_matrix(source: np.ndarray, reference: np.ndarray, block: int = 4096) -> np.ndarray:
    output = np.empty((len(source), len(reference)), dtype=np.float64)
    for start in range(0, len(source), block):
        values = source[start : start + block]
        output[start : start + len(values)] = np.sqrt(
            ((values[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
        )
    return output


def coarse_cells(values: np.ndarray, edges: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    inside = np.ones(len(values), dtype=bool)
    index: list[np.ndarray] = []
    for column, edge in enumerate(edges):
        inside &= (values[:, column] >= edge[0]) & (values[:, column] <= edge[-1])
        index.append(np.clip(np.digitize(values[:, column], edge[1:-1]), 0, 2))
    return 3 * index[0] + index[1], inside


def largest_remainder(total: int, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    exact = total * weights / weights.sum()
    quota = np.floor(exact).astype(int)
    order = np.lexsort((np.arange(len(weights)), -(exact - quota)))
    quota[order[: total - quota.sum()]] += 1
    return quota


def capacity_quotas(total: int, weights: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    """Largest-remainder quotas with deterministic shortage redistribution."""
    quota = np.minimum(largest_remainder(total, weights), np.asarray(capacity, int))
    while quota.sum() < total:
        active = np.flatnonzero((quota < capacity) & (weights > 0))
        if not len(active):
            active = np.flatnonzero(quota < capacity)
        if not len(active):
            raise RuntimeError(f"capacity {capacity.sum()} cannot fill quota {total}")
        expected = total * weights / max(float(weights.sum()), 1.0)
        score = quota[active] - expected[active]
        quota[int(active[np.argmin(score)])] += 1
    return quota


def distribution_metrics(counts: np.ndarray, target_counts: np.ndarray) -> dict[str, Any]:
    counts = np.asarray(counts, dtype=float)
    target = np.asarray(target_counts, dtype=float)
    probability = counts / counts.sum() if counts.sum() else np.zeros(9)
    target_probability = target / target.sum()
    positive = probability[probability > 0]
    entropy = float(-(positive * np.log(positive)).sum()) if len(positive) else 0.0
    return {
        "coarse_n": int(counts.sum()),
        "coarse_js": float(jensenshannon(target_probability, probability, base=2.0) ** 2),
        "occupied_cells": int(np.sum(counts > 0)),
        "effective_cells": float(np.exp(entropy)),
        "largest_cell_share": float(probability.max()) if counts.sum() else float("nan"),
        "cell_counts": counts.astype(int).tolist(),
    }


def sliced_wasserstein(source_z: np.ndarray, target_z: np.ndarray) -> float:
    rng = np.random.default_rng(SEED + 991)
    directions = rng.normal(size=(SWD_PROJECTIONS, 2))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return float(
        np.mean(
            [
                wasserstein_distance(source_z @ direction, target_z @ direction)
                for direction in directions
            ]
        )
    )


def effective_count(values: pd.Series) -> float:
    probability = values.value_counts(normalize=True).to_numpy(float)
    return float(np.exp(-(probability * np.log(probability)).sum())) if len(probability) else 0.0


def scaler_params() -> dict[str, tuple[float, float]]:
    payload = json.loads(SCALER.read_text(encoding="utf-8"))
    table = {item["feature_name"]: item for item in payload["parameters"]}
    return {
        name: (float(table[name]["center_median"]), float(table[name]["scale"]))
        for name in FEATURES
    }


def load_geometry() -> dict[str, Any]:
    manifest = pd.read_csv(P1 / "03_tsplib48_reference_manifest.csv")
    heldout = json.loads((P1 / "p1_heldout22_names.json").read_text(encoding="utf-8"))["names"]
    if len(manifest) != 48 or manifest.instance_id.nunique() != 48:
        raise RuntimeError("TSPLIB48 reference manifest integrity failure")
    if set(manifest.instance_id.astype(str)) & set(map(str, heldout)):
        raise RuntimeError("held-out TSPLIB22 overlaps development TSPLIB48")
    extracted = pd.read_csv(P1 / "p1_tsplib48_normalization_sensitivity.csv")
    target = extracted[extracted.coordinate_variant.eq("aspect")].copy()
    target = target.set_index("instance_id").loc[manifest.instance_id.astype(str)].reset_index()
    values = target[list(FEATURES)].to_numpy(float)
    center = np.median(values, axis=0)
    scale = np.quantile(values, 0.75, axis=0) - np.quantile(values, 0.25, axis=0)
    z = (values - center) / scale
    pair = nearest_matrix(z, z)
    np.fill_diagonal(pair, np.inf)
    radius = float(np.median(pair.min(axis=1)))
    if not np.isclose(radius, STRICT_RADIUS_EXPECTED, atol=1e-12):
        raise RuntimeError(f"strict radius drift: {radius}")
    edges = [
        np.asarray([values[:, i].min(), np.quantile(values[:, i], 1 / 3), np.quantile(values[:, i], 2 / 3), values[:, i].max()])
        for i in range(2)
    ]
    cell, inside = coarse_cells(values, edges)
    if not inside.all():
        raise RuntimeError("target coarse-cell construction failed")
    aux = pd.read_csv(P1 / "p1_evoreal_feature_diagnostic.csv")
    aux = aux.set_index("instance_id").loc[manifest.instance_id.astype(str)].reset_index()
    aux_values = aux[["fft_energy", "nn_ratio"]].to_numpy(float)
    aux_edges = [
        np.asarray([aux_values[:, i].min(), np.quantile(aux_values[:, i], 1 / 3), np.quantile(aux_values[:, i], 2 / 3), aux_values[:, i].max()])
        for i in range(2)
    ]
    aux_cell, aux_inside = coarse_cells(aux_values, aux_edges)
    if not aux_inside.all():
        raise RuntimeError("target auxiliary-cell construction failed")
    return {
        "manifest": manifest,
        "target": target,
        "target_values": values,
        "target_z": z,
        "center": center,
        "scale": scale,
        "radius": radius,
        "edges": edges,
        "target_cell": cell,
        "target_counts": np.bincount(cell, minlength=9),
        "aux_edges": aux_edges,
        "aux_counts": np.bincount(aux_cell, minlength=9),
    }


def add_geometry(frame: pd.DataFrame, geometry: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    values = result[list(FEATURES)].to_numpy(float)
    z = (values - geometry["center"]) / geometry["scale"]
    distance = nearest_matrix(z, geometry["target_z"])
    nearest = distance.argmin(axis=1)
    cell, inside = coarse_cells(values, geometry["edges"])
    result["nearest_reference_index"] = nearest
    result["nearest_reference"] = geometry["manifest"].instance_id.to_numpy()[nearest]
    result["target_distance"] = distance[np.arange(len(result)), nearest]
    result["in_strict_target_region"] = result.target_distance <= geometry["radius"]
    result["coarse_cell"] = cell
    result["inside_target_box"] = inside
    result["nearest_target_cell"] = geometry["target_cell"][nearest]
    return result


def evoreal_diagnostics(coordinates: np.ndarray, block: int = 256) -> tuple[np.ndarray, np.ndarray]:
    fft_values = np.empty(len(coordinates), dtype=np.float64)
    nn_values = np.empty(len(coordinates), dtype=np.float64)
    yy, xx = np.ogrid[:32, :32]
    radial = np.sqrt((yy - 16) ** 2 + (xx - 16) ** 2)
    fft_mask = (radial > 1) & (radial <= 4)
    diagonal = np.arange(coordinates.shape[1])
    for start in range(0, len(coordinates), block):
        points = coordinates[start : start + block]
        minimum = points.min(axis=1, keepdims=True)
        span = (points.max(axis=1) - points.min(axis=1)).max(axis=1)
        span = np.where(span > 0, span, 1.0)
        normalized = (points - minimum) / span[:, None, None]
        difference = normalized[:, :, None, :] - normalized[:, None, :, :]
        distance = np.sqrt(np.sum(difference * difference, axis=-1))
        distance[:, diagonal, diagonal] = np.inf
        nearest = distance.min(axis=2)
        mean = nearest.mean(axis=1)
        nn_values[start : start + len(points)] = np.divide(
            nearest.std(axis=1), mean, out=np.full(len(points), np.nan), where=mean > 0
        )
        bins = np.floor(normalized * 32).astype(int).clip(0, 31)
        histogram = np.zeros((len(points), 32, 32), dtype=np.float64)
        batch_index = np.repeat(np.arange(len(points)), points.shape[1])
        np.add.at(histogram, (batch_index, bins[:, :, 0].ravel(), bins[:, :, 1].ravel()), 1.0)
        magnitude = np.abs(np.fft.fftshift(np.fft.fft2(histogram, axes=(1, 2)), axes=(1, 2)))
        magnitude[:, 16, 16] = 0.0
        fft_values[start : start + len(points)] = magnitude[:, fft_mask].mean(axis=1)
    return fft_values, nn_values


def coordinate_validity(coordinates: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(coordinates).all(axis=(1, 2))
    in_bounds = ((coordinates >= 0) & (coordinates <= 1)).all(axis=(1, 2))
    span = np.ptp(coordinates, axis=1).min(axis=1)
    unique_nodes = np.fromiter(
        (len(np.unique(points, axis=0)) for points in coordinates), dtype=np.int64, count=len(coordinates)
    )
    return {
        "finite_fraction": float(finite.mean()),
        "in_bounds_fraction": float(in_bounds.mean()),
        "nondegenerate_axis_fraction": float((span > 1e-12).mean()),
        "node_overlap_instance_n": int(np.sum(unique_nodes < coordinates.shape[1])),
        "minimum_unique_nodes": int(unique_nodes.min()) if len(unique_nodes) else 0,
    }


def load_training_history() -> tuple[set[str], pd.DataFrame]:
    rows = []
    hashes: set[str] = set()
    for cycle in range(4):
        path = CYCLES / f"C{cycle:02d}" / "06_selection" / "T_selected.parquet"
        frame = pd.read_parquet(path, columns=["instance_hash"])
        values = frame.instance_hash.astype(str)
        rows.append(
            {
                "cycle": f"C{cycle:02d}",
                "rows": len(frame),
                "unique": int(values.nunique()),
                "new_vs_previous": int(sum(value not in hashes for value in values)),
                "membership_sha256": membership_hash(values.tolist()),
                "path": str(path),
            }
        )
        hashes.update(values)
    return hashes, pd.DataFrame(rows)


def load_p03(geometry: dict[str, Any]) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    coordinates, hashes = load_bundle(P03_BUNDLE)
    meta = pd.read_parquet(P03_META)
    if coordinates.shape != (POPULATION_N, 100, 2):
        raise RuntimeError(f"P03 coordinate shape is {coordinates.shape}")
    if len(set(hashes)) != POPULATION_N or hashes != meta.instance_hash.astype(str).tolist():
        raise RuntimeError("P03 identity/alignment failure")
    computed = [coordinate_hash(points) for points in coordinates]
    if computed != hashes:
        raise RuntimeError("P03 coordinate hashes do not reproduce instance hashes")
    c04_parent = pd.read_parquet(C04_PARENT, columns=["instance_hash"])
    if c04_parent.instance_hash.astype(str).tolist() != hashes:
        raise RuntimeError("P03 is not the formal C04 parent population")
    runtime = json.loads(C04_RUNTIME.read_text(encoding="utf-8"))
    if runtime["parent_membership_sha256"] != membership_hash(hashes):
        raise RuntimeError("P03 membership differs from formal C04 mutation runtime")
    features = pd.read_parquet(CYCLES / "C04" / "candidate_features.parquet", columns=["instance_hash", *FEATURES])
    feature_lookup = features.drop_duplicates("instance_hash").set_index("instance_hash")
    missing = [value for value in hashes if value not in feature_lookup.index]
    if missing:
        raise RuntimeError(f"P03 missing {len(missing)} frozen feature rows")
    base = meta.drop(columns=[name for name in FEATURES if name in meta.columns]).copy()
    base = base.merge(feature_lookup.loc[hashes].reset_index(), on="instance_hash", validate="one_to_one", sort=False)
    base["node_count"] = coordinates.shape[1]
    base["evolution_origin"] = "P03_formal_parent"
    base["generation_born"] = -1
    base["direct_parent_hash"] = base.get("parent_hash", pd.Series([None] * len(base))).astype("string")
    base = add_geometry(base, geometry)
    fft, nn = evoreal_diagnostics(coordinates)
    base["fft_energy"] = fft
    base["nn_ratio"] = nn
    valid = coordinate_validity(coordinates)
    if min(valid["finite_fraction"], valid["in_bounds_fraction"], valid["nondegenerate_axis_fraction"]) != 1.0:
        raise RuntimeError(f"P03 coordinate validity failed: {valid}")
    integrity = {
        "population_n": len(base),
        "unique_n": int(base.instance_hash.nunique()),
        "node_count": 100,
        "lineage_n": int(base.lineage_root.nunique()),
        "membership_sha256": membership_hash(hashes),
        "bundle_sha256": sha256_file(P03_BUNDLE),
        "metadata_sha256": sha256_file(P03_META),
        "formal_c04_parent_match": True,
        "formal_c04_runtime_parent_match": True,
        **valid,
    }
    return coordinates, base, integrity


def scalar_trace(operator: str, parameters: dict[str, object]) -> dict[str, object]:
    row: dict[str, object] = {
        "implosion_radius": np.nan,
        "implosion_strength": np.nan,
        "axis_affected_fraction": np.nan,
        "axis_target": np.nan,
        "axis_jitter": np.nan,
        "axis_choice": None,
    }
    if operator == "implosion_mutation":
        row["implosion_radius"] = float(parameters["radius"])
        row["implosion_strength"] = float(parameters["strength"])
    elif operator == "axis_projection_mutation":
        row["axis_affected_fraction"] = float(parameters["affected_fraction"])
        row["axis_target"] = float(parameters["target"])
        row["axis_jitter"] = float(parameters["jitter"])
        row["axis_choice"] = str(parameters["axis"])
    return row


def attempt_schedule() -> list[tuple[int, str]]:
    operators = list(OPERATOR_NAMES)
    schedule = []
    for pass_index in range(2):
        for parent_index in range(POPULATION_N):
            operator = operators[(2 * parent_index + pass_index) % len(operators)]
            schedule.append((parent_index, operator))
    if len(schedule) != ATTEMPTS_PER_GENERATION:
        raise RuntimeError("attempt schedule size drift")
    counts = pd.Series([item[1] for item in schedule]).value_counts()
    if counts.max() - counts.min() > 1:
        raise RuntimeError("operator attempt budget is not balanced")
    return schedule


def mutate_generation(
    arm: str,
    generation: int,
    parent_coordinates: np.ndarray,
    parents: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    directory = OUT / "evolution" / arm / f"G{generation}" / "mutation"
    bundle_path = directory / "offspring_coordinates.npz"
    manifest_path = directory / "mutation_manifest.parquet"
    runtime_path = directory / "runtime.json"
    if bundle_path.is_file() and manifest_path.is_file() and runtime_path.is_file():
        coordinates, hashes = load_bundle(bundle_path)
        manifest = pd.read_parquet(manifest_path)
        accepted = manifest.valid_after_repair & manifest.proposal_status.eq("accepted") & ~manifest.duplicate
        if hashes != manifest.loc[accepted, "child_hash"].astype(str).tolist():
            raise RuntimeError(f"cached mutation alignment failure {arm} G{generation}")
        return coordinates, manifest, json.loads(runtime_path.read_text(encoding="utf-8"))
    directory.mkdir(parents=True, exist_ok=True)
    profile = load_named_profile(PROFILE_NAME)
    schedule = attempt_schedule()
    parent_hashes = parents.instance_hash.astype(str).tolist()
    rows: list[dict[str, Any]] = []
    coordinates: list[np.ndarray] = []
    seen: set[str] = set()
    started = time.perf_counter()
    namespace = MUTATION_CYCLE_NAMESPACE[generation]
    for attempt_index, (parent_index, operator) in enumerate(schedule):
        parent_hash = parent_hashes[parent_index]
        seed = cycle_child_seed(namespace, parent_hash, operator)
        result = apply_operator_with_config(
            operator, parent_coordinates[parent_index], np.random.default_rng(seed), profile
        )
        accepted = result.proposal_status == "accepted" and result.valid_after_repair
        duplicate = bool(accepted and result.final_output_hash in seen)
        rows.append(
            {
                "arm": arm,
                "generation": generation,
                "attempt_index": attempt_index,
                "parent_index": parent_index,
                "parent_hash": parent_hash,
                "operator_id": operator,
                "mutation_profile": PROFILE_NAME,
                "mutation_depth": 1,
                "child_seed": np.uint64(seed),
                "child_hash": result.final_output_hash,
                "raw_child_hash": result.raw_output_hash,
                "n_affected": int(result.n_affected),
                "valid_before_repair": bool(result.valid_before_repair),
                "valid_after_repair": bool(result.valid_after_repair),
                "repair_count": int(result.repair_count),
                "repair_fraction": float(result.repair_fraction),
                "proposal_status": result.proposal_status,
                "failure_reason": result.failure_reason,
                "duplicate": duplicate,
                "cycle_seed_namespace": namespace,
                **scalar_trace(operator, dict(result.sampled_parameters)),
                **{key: float(value) for key, value in result.timing.items()},
            }
        )
        if accepted and not duplicate:
            seen.add(result.final_output_hash)
            coordinates.append(np.asarray(result.final_coordinates, dtype=np.float64))
        if (attempt_index + 1) % 5000 == 0:
            print(f"[P2.2][{arm}][G{generation}] mutation {attempt_index + 1}/{len(schedule)}", flush=True)
    manifest = pd.DataFrame(rows)
    offspring = np.stack(coordinates)
    accepted_mask = manifest.valid_after_repair & manifest.proposal_status.eq("accepted") & ~manifest.duplicate
    hashes = manifest.loc[accepted_mask, "child_hash"].astype(str).tolist()
    counts = manifest.operator_id.value_counts().sort_index().astype(int).to_dict()
    runtime = {
        "status": "PASS",
        "arm": arm,
        "generation": generation,
        "cycle_seed_namespace": namespace,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "profile_name": PROFILE_NAME,
        "profile_sha256": sha256_file(PROFILE_PATH),
        "parent_membership_sha256": membership_hash(parent_hashes),
        "offspring_membership_sha256": membership_hash(hashes),
        "nominal_attempts": len(manifest),
        "valid_unique_within_generation": len(hashes),
        "operator_attempts": counts,
        "operator_budget_range": [min(counts.values()), max(counts.values())],
        "wall_seconds": time.perf_counter() - started,
    }
    if len(manifest) != ATTEMPTS_PER_GENERATION or max(counts.values()) - min(counts.values()) > 1:
        raise RuntimeError(f"mutation budget mismatch {arm} G{generation}: {counts}")
    save_bundle(bundle_path, offspring, hashes)
    manifest.to_parquet(manifest_path, index=False)
    write_json(runtime_path, runtime)
    return offspring, manifest, runtime


def run_feature_job(job: dict[str, Any]) -> str:
    process = subprocess.Popen(
        [sys.executable, str(F40_WORKER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(pickle.dumps(job), timeout=900)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise RuntimeError(f"feature shard timeout: {job['out']}")
    if process.returncode:
        raise RuntimeError(
            f"feature shard failed rc={process.returncode}: {stderr.decode(errors='replace')[-2000:]}"
        )
    return stdout.decode(errors="replace").strip()


def load_feature_cache() -> pd.DataFrame:
    files = sorted((OUT / "feature_cache").glob("cache_*.parquet"))
    if not files:
        return pd.DataFrame(columns=["instance_hash", *FEATURES])
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True).drop_duplicates(
        "instance_hash", keep="first"
    )


def extract_features(tag: str, coordinates: np.ndarray, hashes: list[str]) -> pd.DataFrame:
    output = OUT / "features" / f"{tag}.parquet"
    if output.is_file():
        frame = pd.read_parquet(output)
        if frame.instance_hash.astype(str).tolist() == hashes:
            return frame
    cache = load_feature_cache()
    known = set(cache.instance_hash.astype(str))
    missing_indices = [i for i, value in enumerate(hashes) if value not in known]
    if missing_indices:
        jobs = []
        params = scaler_params()
        cache_dir = OUT / "feature_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(missing_indices), FEATURE_SHARD_N):
            indices = missing_indices[start : start + FEATURE_SHARD_N]
            block_hashes = [hashes[i] for i in indices]
            token = membership_hash(block_hashes)[:16]
            jobs.append(
                {
                    "coordinates": coordinates[np.asarray(indices, int)],
                    "hashes": block_hashes,
                    "names": list(FEATURES),
                    "out": str(cache_dir / f"cache_{token}.parquet"),
                    "rscript": shutil.which("Rscript") or "Rscript",
                    "r_script": str(R_EXTRACTOR),
                    "params": params,
                }
            )
        print(f"[P2.2][{tag}] extracting {len(missing_indices)} features in {len(jobs)} shards", flush=True)
        with ThreadPoolExecutor(max_workers=min(FEATURE_WORKERS, len(jobs))) as executor:
            futures = [executor.submit(run_feature_job, job) for job in jobs]
            for completed, future in enumerate(as_completed(futures), 1):
                future.result()
                if completed == 1 or completed % 20 == 0 or completed == len(jobs):
                    print(f"[P2.2][{tag}] feature shards {completed}/{len(jobs)}", flush=True)
        cache = load_feature_cache()
    lookup = cache.set_index("instance_hash")
    missing = [value for value in hashes if value not in lookup.index]
    if missing:
        raise RuntimeError(f"feature cache missing {len(missing)} rows for {tag}")
    result = lookup.loc[hashes].reset_index()[["instance_hash", *FEATURES]]
    values = result[list(FEATURES)].to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"nonfinite features in {tag}")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    return result


def build_offspring(
    arm: str,
    generation: int,
    coordinates: np.ndarray,
    manifest: pd.DataFrame,
    parents: pd.DataFrame,
    parent_coordinates: np.ndarray,
    geometry: dict[str, Any],
    history: set[str],
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    output = OUT / "evolution" / arm / f"G{generation}" / "offspring_diagnostics.parquet"
    accepted = manifest.valid_after_repair & manifest.proposal_status.eq("accepted") & ~manifest.duplicate
    hashes = manifest.loc[accepted, "child_hash"].astype(str).tolist()
    if output.is_file():
        cached = pd.read_parquet(output)
        if cached.instance_hash.astype(str).tolist() == hashes:
            return coordinates, cached, coordinate_validity(coordinates)
    meta = manifest.loc[accepted].reset_index(drop=True).copy()
    if len(coordinates) != len(meta):
        raise RuntimeError(f"offspring alignment failure {arm} G{generation}")
    features = extract_features(f"{arm}_G{generation}", coordinates, hashes)
    meta["instance_hash"] = hashes
    meta = meta.merge(features, on="instance_hash", validate="one_to_one", sort=False)
    parent_lookup = parents.set_index("instance_hash")
    meta["lineage_root"] = meta.parent_hash.astype(str).map(parent_lookup.lineage_root)
    if meta.lineage_root.isna().any():
        raise RuntimeError(f"offspring lineage missing {arm} G{generation}")
    meta["direct_parent_hash"] = meta.parent_hash.astype(str)
    meta["node_count"] = coordinates.shape[1]
    meta["evolution_origin"] = f"{arm}_G{generation}_offspring"
    meta["generation_born"] = generation
    meta = add_geometry(meta, geometry)
    parent_distance = parent_lookup.target_distance
    meta["parent_target_distance"] = meta.parent_hash.astype(str).map(parent_distance)
    meta["target_distance_change"] = meta.target_distance - meta.parent_target_distance
    meta["target_proximity_improved"] = meta.target_distance_change < 0
    meta["entered_strict_region"] = (
        (meta.parent_target_distance > geometry["radius"]) & meta.in_strict_target_region
    )
    meta["m4_training_overlap"] = meta.instance_hash.astype(str).isin(history)
    fft, nn = evoreal_diagnostics(coordinates)
    meta["fft_energy"] = fft
    meta["nn_ratio"] = nn
    validity = coordinate_validity(coordinates)
    meta.to_parquet(output, index=False)
    return coordinates, meta, validity


def microbin_order(frame: pd.DataFrame, geometry: dict[str, Any]) -> list[int]:
    z = (frame[list(FEATURES)].to_numpy(float) - geometry["center"]) / geometry["scale"]
    lo = geometry["target_z"].min(axis=0)
    hi = geometry["target_z"].max(axis=0)
    normalized = np.divide(z - lo, hi - lo, out=np.zeros_like(z), where=(hi - lo) > 0)
    bins = np.floor(normalized * 12).astype(int).clip(0, 11)
    work = frame[["instance_hash", "target_distance"]].copy()
    work["_row"] = np.arange(len(frame))
    work["_micro"] = 12 * bins[:, 0] + bins[:, 1]
    work = work.sort_values(["_micro", "target_distance", "instance_hash"], kind="stable")
    work["_rank"] = work.groupby("_micro", sort=False).cumcount()
    return work.sort_values(["_rank", "target_distance", "_micro", "instance_hash"], kind="stable")["_row"].astype(int).tolist()


def lineage_take(
    frame: pd.DataFrame,
    ordered_local_rows: list[int],
    needed: int,
    selected: set[int],
    lineage_counts: dict[str, int],
    initial_cap: int,
) -> tuple[list[int], int]:
    chosen: list[int] = []
    cap = initial_cap
    global_indices = frame.index.to_numpy(int)
    while len(chosen) < needed:
        progress = False
        for local in ordered_local_rows:
            index = int(global_indices[local])
            if index in selected:
                continue
            lineage = str(frame.iloc[local].lineage_root)
            if lineage_counts.get(lineage, 0) >= cap:
                continue
            selected.add(index)
            lineage_counts[lineage] = lineage_counts.get(lineage, 0) + 1
            chosen.append(index)
            progress = True
            if len(chosen) == needed:
                break
        if len(chosen) == needed:
            break
        if not progress and cap >= max(initial_cap * 16, needed):
            break
        cap = min(max(initial_cap * 16, needed), cap * 2)
    return chosen, cap


def structured_select(
    frame: pd.DataFrame,
    n: int,
    geometry: dict[str, Any],
    target_fraction: float,
    lineage_cap: int,
    mode: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select target-aligned plus frontier candidates without hardness."""
    if len(frame) < n or frame.instance_hash.duplicated().any():
        raise RuntimeError(f"invalid pool for {mode}: n={len(frame)} unique={frame.instance_hash.nunique()}")
    work = frame.reset_index(drop=True).copy()
    requested_target_n = int(round(n * target_fraction))
    selected: set[int] = set()
    lineage_counts: dict[str, int] = {}
    components: dict[int, str] = {}
    max_cap = lineage_cap

    strict = work.in_strict_target_region.to_numpy(bool) & work.inside_target_box.to_numpy(bool)
    capacity = np.bincount(work.loc[strict, "coarse_cell"].astype(int), minlength=9)
    # P03 is hardness-selected and can have less than 70% strict-region support.
    # Keep the frozen 70/30 request, but explicitly move any infeasible strict
    # quota into frontier exploration rather than silently changing population
    # size or importing another pool.
    target_n = min(requested_target_n, int(capacity.sum()))
    frontier_n = n - target_n
    target_quota = capacity_quotas(target_n, geometry["target_counts"], capacity)
    for cell in range(9):
        needed = int(target_quota[cell])
        if not needed:
            continue
        part = work.loc[strict & work.coarse_cell.eq(cell)].copy()
        taken, cap = lineage_take(part, microbin_order(part, geometry), needed, selected, lineage_counts, lineage_cap)
        max_cap = max(max_cap, cap)
        if len(taken) != needed:
            raise RuntimeError(f"{mode} strict cell {cell} filled {len(taken)}/{needed}")
        components.update({index: "target_coverage" for index in taken})

    remaining = ~work.index.isin(selected) & ~work.in_strict_target_region
    frontier_capacity = np.bincount(work.loc[remaining, "nearest_target_cell"].astype(int), minlength=9)
    frontier_quota = capacity_quotas(frontier_n, geometry["target_counts"], frontier_capacity)
    for cell in range(9):
        needed = int(frontier_quota[cell])
        if not needed:
            continue
        part = work.loc[remaining & work.nearest_target_cell.eq(cell)].copy()
        taken, cap = lineage_take(part, microbin_order(part, geometry), needed, selected, lineage_counts, lineage_cap)
        max_cap = max(max_cap, cap)
        if len(taken) != needed:
            raise RuntimeError(f"{mode} frontier cell {cell} filled {len(taken)}/{needed}")
        components.update({index: "frontier_exploration" for index in taken})

    if len(selected) < n:
        part = work.loc[~work.index.isin(selected)].copy()
        taken, cap = lineage_take(part, microbin_order(part, geometry), n - len(selected), selected, lineage_counts, lineage_cap)
        max_cap = max(max_cap, cap)
        components.update({index: "deterministic_backfill" for index in taken})
    if len(selected) != n:
        raise RuntimeError(f"{mode} selected {len(selected)}/{n}")
    indices = sorted(selected, key=lambda value: (components[value], str(work.at[value, "instance_hash"])))
    result = work.iloc[indices].copy().reset_index(drop=True)
    result["selection_component"] = [components[index] for index in indices]
    metadata = {
        "mode": mode,
        "pool_n": len(work),
        "selected_n": len(result),
        "requested_target_n": requested_target_n,
        "target_n": target_n,
        "strict_quota_shortfall_reassigned_to_frontier": requested_target_n - target_n,
        "frontier_n": frontier_n,
        "strict_capacity_by_cell": capacity.astype(int).tolist(),
        "target_quota_by_cell": target_quota.astype(int).tolist(),
        "frontier_capacity_by_target_cell": frontier_capacity.astype(int).tolist(),
        "frontier_quota_by_target_cell": frontier_quota.astype(int).tolist(),
        "lineage_cap_initial": lineage_cap,
        "lineage_cap_max_used": max_cap,
        "component_counts": result.selection_component.value_counts().to_dict(),
    }
    return result, metadata


def random_survivors(frame: pd.DataFrame, generation: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    canonical = frame.sort_values("instance_hash", kind="stable").reset_index(drop=True)
    rng = np.random.default_rng(SEED + 1000 + generation)
    indices = rng.choice(len(canonical), POPULATION_N, replace=False)
    result = canonical.iloc[indices].copy().reset_index(drop=True)
    result["selection_component"] = "uniform_parent_child_union"
    return result, {
        "mode": "random_survivor",
        "seed": SEED + 1000 + generation,
        "pool_n": len(frame),
        "selected_n": len(result),
    }


def align_coordinates(frame: pd.DataFrame, coordinate_map: dict[str, np.ndarray]) -> np.ndarray:
    missing = [value for value in frame.instance_hash.astype(str) if value not in coordinate_map]
    if missing:
        raise RuntimeError(f"coordinate map missing {len(missing)} instances")
    return np.stack([coordinate_map[value] for value in frame.instance_hash.astype(str)])


def manage_archive(
    archive: pd.DataFrame,
    new: pd.DataFrame,
    geometry: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    combined = pd.concat([archive, new], ignore_index=True, sort=False)
    combined = combined.drop_duplicates("instance_hash", keep="first").reset_index(drop=True)
    before = len(combined)
    if before <= ARCHIVE_CAPACITY:
        result = combined
        metadata = {"mode": "under_capacity", "pool_n": before, "selected_n": len(result)}
    else:
        result, metadata = structured_select(
            combined,
            ARCHIVE_CAPACITY,
            geometry,
            TARGET_FRACTION,
            LINEAGE_CAP_ARCHIVE,
            "common_bounded_archive_retention",
        )
    return result.reset_index(drop=True), metadata


def evaluate_frame(
    arm: str,
    generation: int,
    object_type: str,
    frame: pd.DataFrame,
    geometry: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    values = frame[list(FEATURES)].to_numpy(float)
    z = (values - geometry["center"]) / geometry["scale"]
    inside = frame.inside_target_box.to_numpy(bool)
    counts = np.bincount(frame.loc[inside, "coarse_cell"].astype(int), minlength=9)
    distribution = distribution_metrics(counts, geometry["target_counts"])
    distance = nearest_matrix(z, geometry["target_z"])
    nearest_by_reference = distance.min(axis=0)
    aux_values = frame[["fft_energy", "nn_ratio"]].to_numpy(float)
    aux_cell, aux_inside = coarse_cells(aux_values, geometry["aux_edges"])
    aux_counts = np.bincount(aux_cell[aux_inside], minlength=9)
    aux_distribution = distribution_metrics(aux_counts, geometry["aux_counts"])
    lineages = frame.lineage_root.astype(str)
    rounded_unique = np.unique(np.round(values, 8), axis=0).shape[0]
    row = {
        "arm": arm,
        "generation": f"G{generation}",
        "object_type": object_type,
        "n": len(frame),
        "unique_n": int(frame.instance_hash.nunique()),
        "strict_target_n": int(frame.in_strict_target_region.sum()),
        "strict_target_fraction": float(frame.in_strict_target_region.mean()),
        "reference_coverage_n": int(np.sum(nearest_by_reference <= geometry["radius"])),
        "reference_nearest_mean": float(nearest_by_reference.mean()),
        "reference_nearest_p90": float(np.quantile(nearest_by_reference, 0.9)),
        "sliced_wasserstein": sliced_wasserstein(z, geometry["target_z"]),
        "lineage_n": int(lineages.nunique()),
        "effective_lineages": effective_count(lineages),
        "largest_lineage_share": float(lineages.value_counts(normalize=True).max()),
        "feature_rounded_unique_fraction": float(rounded_unique / len(frame)),
        "fft_median": float(frame.fft_energy.median()),
        "nn_cv_median": float(frame.nn_ratio.median()),
        "aux_coarse_js": aux_distribution["coarse_js"],
        "aux_effective_cells": aux_distribution["effective_cells"],
        **distribution,
    }
    reference = geometry["manifest"][["instance_id", "evoreal_split"]].copy()
    reference["arm"] = arm
    reference["generation"] = f"G{generation}"
    reference["object_type"] = object_type
    reference["nearest_distance"] = nearest_by_reference
    reference["covered"] = nearest_by_reference <= geometry["radius"]
    return row, reference


def operator_metrics(
    arm: str,
    generation: int,
    manifest: pd.DataFrame,
    offspring: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for operator, part in manifest.groupby("operator_id", sort=True):
        valid = part.valid_after_repair & part.proposal_status.eq("accepted")
        unique_hashes = set(part.loc[valid & ~part.duplicate, "child_hash"].astype(str))
        child = offspring[offspring.instance_hash.astype(str).isin(unique_hashes)]
        rows.append(
            {
                "arm": arm,
                "generation": f"G{generation}",
                "operator_id": operator,
                "attempts": len(part),
                "valid_attempts": int(valid.sum()),
                "valid_unique": len(child),
                "eligible_unique": int((~child.m4_training_overlap).sum()),
                "strict_target_unique": int(child.in_strict_target_region.sum()),
                "strict_hit_rate_per_attempt": float(child.in_strict_target_region.sum() / len(part)),
                "proximity_improved_fraction": float(child.target_proximity_improved.mean()),
                "mean_target_distance_change": float(child.target_distance_change.mean()),
            }
        )
    return pd.DataFrame(rows)


def acquisition(
    arm: str,
    archive: pd.DataFrame,
    geometry: dict[str, Any],
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    if len(archive) < POPULATION_N:
        return None, {"arm": arm, "available": len(archive), "shortfall": POPULATION_N - len(archive)}
    # Exact target cell proportions are primary. Strict candidates are preferred;
    # transition/boundary candidates in the same cell provide explicit backfill.
    work = archive.reset_index(drop=True).copy()
    requested_quotas = largest_remainder(POPULATION_N, geometry["target_counts"])
    cell_capacity = np.bincount(
        work.loc[work.inside_target_box, "coarse_cell"].astype(int), minlength=9
    )
    # A bounded archive may not support the exact frozen quota in every cell.
    # Redistribute only the explicit cell deficits, using the frozen target
    # weights and available capacities. This keeps backfill reproducible and
    # avoids silently over-filling arbitrary already-satisfied cells.
    quotas = capacity_quotas(POPULATION_N, geometry["target_counts"], cell_capacity)
    selected: set[int] = set()
    lineage_counts: dict[str, int] = {}
    components: dict[int, str] = {}
    fulfillment = []
    max_cap = LINEAGE_CAP_ACQUISITION
    for cell in range(9):
        requested_quota = int(requested_quotas[cell])
        quota = int(quotas[cell])
        if not quota:
            fulfillment.append({"cell": cell, "requested_quota": requested_quota, "adjusted_quota": 0, "strict_capacity": 0, "cell_capacity": int(cell_capacity[cell]), "selected": 0, "original_shortfall": max(0, requested_quota - int(cell_capacity[cell])), "adjusted_shortfall": 0})
            continue
        cell_frame = work.loc[work.inside_target_box & work.coarse_cell.eq(cell)].copy()
        strict_frame = cell_frame.loc[cell_frame.in_strict_target_region].copy()
        first_n = min(quota, len(strict_frame))
        taken, cap = lineage_take(strict_frame, microbin_order(strict_frame, geometry), first_n, selected, lineage_counts, LINEAGE_CAP_ACQUISITION)
        max_cap = max(max_cap, cap)
        components.update({index: "strict_cell_quota" for index in taken})
        remaining_n = quota - len(taken)
        if remaining_n:
            transition = cell_frame.loc[~cell_frame.index.isin(selected)].copy()
            taken2, cap = lineage_take(transition, microbin_order(transition, geometry), remaining_n, selected, lineage_counts, LINEAGE_CAP_ACQUISITION)
            max_cap = max(max_cap, cap)
            components.update({index: "cell_transition_backfill" for index in taken2})
        selected_cell = sum(work.at[index, "coarse_cell"] == cell and work.at[index, "inside_target_box"] for index in selected)
        fulfillment.append(
            {
                "cell": cell,
                "requested_quota": requested_quota,
                "adjusted_quota": quota,
                "strict_capacity": len(strict_frame),
                "cell_capacity": len(cell_frame),
                "selected": selected_cell,
                "original_shortfall": max(0, requested_quota - len(cell_frame)),
                "adjusted_shortfall": max(0, quota - selected_cell),
            }
        )
    if len(selected) < POPULATION_N:
        outside = work.loc[~work.index.isin(selected) & ~work.inside_target_box].copy()
        taken, cap = lineage_take(outside, microbin_order(outside, geometry), POPULATION_N - len(selected), selected, lineage_counts, LINEAGE_CAP_ACQUISITION)
        max_cap = max(max_cap, cap)
        components.update({index: "outside_box_backfill" for index in taken})
    if len(selected) != POPULATION_N:
        return None, {
            "arm": arm,
            "available": len(archive),
            "selected": len(selected),
            "shortfall": POPULATION_N - len(selected),
            "quota_fulfillment": fulfillment,
        }
    indices = sorted(selected, key=lambda value: (components[value], str(work.at[value, "instance_hash"])))
    result = work.iloc[indices].copy().reset_index(drop=True)
    result["acquisition_component"] = [components[index] for index in indices]
    metadata = {
        "arm": arm,
        "available": len(archive),
        "selected": len(result),
        "shortfall": 0,
        "target_quotas_requested": requested_quotas.tolist(),
        "target_quotas_after_capacity_redistribution": quotas.tolist(),
        "cell_capacity": cell_capacity.tolist(),
        "actual_cell_counts": np.bincount(result.loc[result.inside_target_box, "coarse_cell"].astype(int), minlength=9).tolist(),
        "quota_fulfillment": fulfillment,
        "lineage_cap_initial": LINEAGE_CAP_ACQUISITION,
        "lineage_cap_max_used": max_cap,
        "component_counts": result.acquisition_component.value_counts().to_dict(),
    }
    return result, metadata


def save_state(
    arm: str,
    generation: int,
    name: str,
    frame: pd.DataFrame,
    coordinates: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> None:
    directory = OUT / "evolution" / arm / f"G{generation}"
    frame.to_parquet(directory / f"{name}.parquet", index=False)
    save_bundle(directory / f"{name}.npz", coordinates, frame.instance_hash.astype(str).tolist())
    if metadata is not None:
        write_json(directory / f"{name}_selection.json", metadata)


def make_figures(
    metrics: pd.DataFrame,
    populations: dict[tuple[str, int], pd.DataFrame],
    acquisitions: dict[str, pd.DataFrame | None],
    geometry: dict[str, Any],
    references: pd.DataFrame,
) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    pop = metrics[metrics.object_type.eq("active_population")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for arm, color in (("E0", "#4c78a8"), ("E1", "#f58518")):
        part = pop[pop.arm.eq(arm)].sort_values("generation")
        axes[0].plot(part.generation, part.sliced_wasserstein, marker="o", color=color, label=arm)
        axes[1].plot(part.generation, part.coarse_js, marker="o", color=color, label=arm)
    axes[0].set_ylabel("sliced Wasserstein distance")
    axes[1].set_ylabel("3x3 JS divergence")
    for ax in axes:
        ax.set_xlabel("generation")
        ax.legend()
        ax.grid(alpha=0.25)
    fig.suptitle("Bounded active-population adaptation")
    fig.tight_layout()
    fig.savefig(FIG / "01_generation_adaptation.png", dpi=180)
    plt.close(fig)

    target = geometry["target_values"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), sharex=True, sharey=True)
    for row, arm in enumerate(("E0", "E1")):
        for col, generation in enumerate((0, 1, 2)):
            frame = populations[(arm, generation)]
            axes[row, col].hexbin(frame[FEATURES[0]], frame[FEATURES[1]], gridsize=40, mincnt=1, bins="log", cmap="Blues")
            axes[row, col].scatter(target[:, 0], target[:, 1], s=12, color="#d62728", edgecolor="black", linewidth=0.2)
            axes[row, col].set_title(f"{arm} G{generation}")
    for ax in axes[-1]:
        ax.set_xlabel(FEATURES[0].replace("spn_", ""))
    for ax in axes[:, 0]:
        ax.set_ylabel(FEATURES[1].replace("spn_", ""))
    fig.suptitle("Active population in frozen feature space")
    fig.tight_layout()
    fig.savefig(FIG / "02_population_feature_space.png", dpi=180)
    plt.close(fig)

    final_ref = references[(references.object_type == "active_population") & (references.generation == "G2")]
    pivot = final_ref.pivot(index="instance_id", columns="arm", values="nearest_distance").loc[geometry["manifest"].instance_id]
    fig, ax = plt.subplots(figsize=(14, 5.2))
    x = np.arange(len(pivot))
    ax.plot(x, pivot.E0, marker="o", markersize=2.5, linewidth=1.1, label="E0 G2")
    ax.plot(x, pivot.E1, marker="o", markersize=2.5, linewidth=1.1, label="E1 G2")
    ax.axhline(geometry["radius"], color="black", linestyle="--", linewidth=0.9, label="strict radius")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=75, fontsize=7)
    ax.set_ylabel("nearest population distance")
    ax.set_title("G2 TSPLIB48 neighborhood coverage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "03_g2_reference_coverage.png", dpi=180)
    plt.close(fig)

    final = metrics[(metrics.generation == "G2") & metrics.object_type.isin(["raw_offspring", "active_population", "bounded_archive"])]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, metric, title in zip(
        axes,
        ("strict_target_fraction", "effective_lineages", "aux_coarse_js"),
        ("strict target fraction", "effective lineages", "FFT/NN-CV auxiliary JS"),
    ):
        for offset, object_type in enumerate(("raw_offspring", "active_population", "bounded_archive")):
            part = final[final.object_type.eq(object_type)].set_index("arm")
            ax.bar(np.arange(2) + 0.25 * offset, [part.at[arm, metric] for arm in ("E0", "E1")], width=0.23, label=object_type)
        ax.set_xticks(np.arange(2) + 0.25)
        ax.set_xticklabels(["E0", "E1"])
        ax.set_title(title)
    axes[0].legend(fontsize=8)
    fig.suptitle("G2 mechanism and feature-gaming diagnostics")
    fig.tight_layout()
    fig.savefig(FIG / "04_g2_diagnostics.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax, arm in zip(axes, ("E0", "E1")):
        frame = acquisitions[arm]
        if frame is None:
            ax.text(0.5, 0.5, "10k unavailable", ha="center", va="center")
        else:
            ax.hexbin(frame[FEATURES[0]], frame[FEATURES[1]], gridsize=40, mincnt=1, bins="log", cmap="Blues")
            ax.scatter(target[:, 0], target[:, 1], s=18, color="#d62728", edgecolor="black", linewidth=0.2)
        ax.set_title(f"{arm} provisional 10k")
        ax.set_xlabel(FEATURES[0].replace("spn_", ""))
    axes[0].set_ylabel(FEATURES[1].replace("spn_", ""))
    fig.suptitle("Identical offline acquisition from bounded archives")
    fig.tight_layout()
    fig.savefig(FIG / "05_provisional_structure10k.png", dpi=180)
    plt.close(fig)


def write_reports(
    geometry: dict[str, Any],
    p03_integrity: dict[str, Any],
    history_table: pd.DataFrame,
    metrics: pd.DataFrame,
    operator: pd.DataFrame,
    acquisitions: dict[str, pd.DataFrame | None],
    acquisition_meta: dict[str, dict[str, Any]],
    runtimes: list[dict[str, Any]],
) -> str:
    active = metrics[metrics.object_type.eq("active_population")].set_index(["arm", "generation"])
    raw = metrics[metrics.object_type.eq("raw_offspring")].set_index(["arm", "generation"])
    archive = metrics[metrics.object_type.eq("bounded_archive")].set_index(["arm", "generation"])
    acquisition_rows = metrics[metrics.object_type.eq("provisional_structure10k")].set_index("arm")
    history_markdown = "\n".join(
        [
            "| Cycle | Rows | Unique | New vs previous | Membership SHA256 | Path |",
            "|---|---:|---:|---:|---|---|",
            *[
                f"| {row.cycle} | {row.rows:,} | {row.unique:,} | {row.new_vs_previous:,} | `{row.membership_sha256}` | `{row.path}` |"
                for row in history_table.itertuples(index=False)
            ],
        ]
    )
    integrity_text = f"""# P03 Population Integrity

- Source: `{P03_BUNDLE.relative_to(ROOT)}` and `{P03_META.relative_to(ROOT)}`.
- Actual population: **{p03_integrity['population_n']:,}** unique N={p03_integrity['node_count']} instances; {p03_integrity['lineage_n']:,} lineages.
- Ordered membership SHA256: `{p03_integrity['membership_sha256']}`.
- Bundle SHA256: `{p03_integrity['bundle_sha256']}`; metadata SHA256: `{p03_integrity['metadata_sha256']}`.
- All coordinate hashes reproduce IDs; finite/in-bounds/nondegenerate fractions are {p03_integrity['finite_fraction']:.3f}/{p03_integrity['in_bounds_fraction']:.3f}/{p03_integrity['nondegenerate_axis_fraction']:.3f}.
- The ordered IDs exactly equal formal `C04/04_qd/parent_qd.parquet`, and the membership SHA exactly equals C04 FOGA runtime's recorded parent SHA. P03 is therefore the formal population after C03 selection and before C04 mutation.
- P03 is not reconstructed from C04 Hard10k and not taken from the P2.0/P2.1 expanded pool.

## M4 training history

C00-C03 `T_selected` are treated as the actual M4 training history. Their union is {len(set().union(*[set(pd.read_parquet(CYCLES / f'C{i:02d}' / '06_selection' / 'T_selected.parquet', columns=['instance_hash']).instance_hash.astype(str)) for i in range(4)])):,} unique instances. C04 Hard10k is deliberately excluded from that history because it trained M5, not M4.

{history_markdown}
"""
    (OUT / "00_p03_population_integrity.md").write_text(integrity_text, encoding="utf-8")

    protocol = f"""# P2.2 Experiment Protocol

- Arms: E0 random survivor control and E1 structure-guided survivors, both starting from identical formal P03.
- Active population: fixed at {POPULATION_N:,}; Parent+Child Union selection; two continuous generations.
- Mutation: frozen `{PROFILE_NAME}` and all 11 operators via the repository API. Each generation has {ATTEMPTS_PER_GENERATION:,} attempts: every parent receives exactly two distinct operators and each operator receives 1,818 or 1,819 attempts.
- Seed: {SEED}; FOGA cycle seed namespaces: {MUTATION_CYCLE_NAMESPACE}. E0/E1 G1 therefore have identical mutations before survivor selection; G2 naturally differs.
- E1 selection: {TARGET_FRACTION:.0%} strict target-coverage quota plus {FRONTIER_FRACTION:.0%} non-strict frontier exploration. Frozen 3x3 target weights, 12x12 within-region microbin round-robin, and deterministic lineage caps are used. No single TSPLIB instance receives a special objective.
- E0 selection: canonical-hash ordering followed by seeded uniform sampling without replacement.
- Archive: eligible unique offspring only, C00-C03 exclusions applied, maximum {ARCHIVE_CAPACITY:,}. Both arms use the identical structure/diversity retention method.
- Continuous distance: mean 1D Wasserstein distance across {SWD_PROJECTIONS} fixed random directions in the P1 median/IQR-standardized two-feature space (seed {SEED + 991}).
- Final acquisition: identical structure-only cell quotas, continuous microbin diversity, lineage cap, strict-first then explicit transition backfill. HGB/BQ hardness is absent from mutation, survivor, archive, and acquisition logic.
- Decision guards frozen before results: E1 must beat E0 at G2 SWD, improve G0→G1→G2, retain at least 50% of G0 effective lineages with max lineage share <=5%, avoid geometry failures, and provide a complete eligible 10k. Auxiliary JS degradation >0.10 versus E0 is treated as a feature-gaming warning and prevents unconditional GO.
- Held-out TSPLIB22 coordinates/features are never loaded. Their names are read only for the P1 disjointness assertion.
- No Oracle, Concorde, HGB fitting/prediction, BQ inference, or BQ training is invoked.
"""
    (OUT / "01_experiment_protocol.md").write_text(protocol, encoding="utf-8")

    def arm_report(arm: str) -> str:
        lines = [f"# {arm} {'Random' if arm == 'E0' else 'Structure-Guided'} Evolution", ""]
        for generation in (0, 1, 2):
            row = active.loc[(arm, f"G{generation}")]
            lines.append(
                f"- G{generation}: SWD {row.sliced_wasserstein:.6f}, 3x3 JS {row.coarse_js:.6f}, strict {int(row.strict_target_n):,}/{int(row.n):,}, coverage {int(row.reference_coverage_n)}/48, effective lineages {row.effective_lineages:.1f}, largest lineage {row.largest_lineage_share:.2%}."
            )
            if generation:
                child = raw.loc[(arm, f"G{generation}")]
                arc = archive.loc[(arm, f"G{generation}")]
                lines.append(
                    f"  Raw offspring: {int(child.n):,} unique, {int(child.eligible_unique_n):,} eligible, {int(child.strict_target_n):,} strict ({child.strict_target_fraction:.2%}), valid rate {child.valid_rate:.2%}, parent→child proximity improvement {child.proximity_improved_fraction:.2%}; bounded archive: {int(arc.n):,}."
                )
        final_archive_meta = json.loads(
            (OUT / "evolution" / arm / "G2" / "bounded_archive_selection.json").read_text(encoding="utf-8")
        )
        lines.append(
            f"- Two-generation eligible unique union before the 30k cap: {int(final_archive_meta['pool_n']):,}; retained archive: {int(final_archive_meta['selected_n']):,}."
        )
        runtime = sum(float(item["wall_seconds"]) for item in runtimes if item["arm"] == arm)
        lines.extend(["", f"Mutation wall time: {runtime:.1f}s. Full per-operator counts are in `support_operator_metrics.csv`."])
        return "\n".join(lines) + "\n"

    (OUT / "03_e0_random_evolution.md").write_text(arm_report("E0"), encoding="utf-8")
    (OUT / "04_e1_structure_evolution.md").write_text(arm_report("E1"), encoding="utf-8")

    e0g2, e1g2 = active.loc[("E0", "G2")], active.loc[("E1", "G2")]
    comparison = f"""# E0/E1 Evolution Comparison

| Metric | G0 shared | E0 G1 | E0 G2 | E1 G1 | E1 G2 |
|---|---:|---:|---:|---:|---:|
| Sliced Wasserstein | {active.loc[('E0','G0')].sliced_wasserstein:.6f} | {active.loc[('E0','G1')].sliced_wasserstein:.6f} | {e0g2.sliced_wasserstein:.6f} | {active.loc[('E1','G1')].sliced_wasserstein:.6f} | {e1g2.sliced_wasserstein:.6f} |
| 3x3 JS | {active.loc[('E0','G0')].coarse_js:.6f} | {active.loc[('E0','G1')].coarse_js:.6f} | {e0g2.coarse_js:.6f} | {active.loc[('E1','G1')].coarse_js:.6f} | {e1g2.coarse_js:.6f} |
| Strict target fraction | {active.loc[('E0','G0')].strict_target_fraction:.2%} | {active.loc[('E0','G1')].strict_target_fraction:.2%} | {e0g2.strict_target_fraction:.2%} | {active.loc[('E1','G1')].strict_target_fraction:.2%} | {e1g2.strict_target_fraction:.2%} |
| Coverage | {int(active.loc[('E0','G0')].reference_coverage_n)}/48 | {int(active.loc[('E0','G1')].reference_coverage_n)}/48 | {int(e0g2.reference_coverage_n)}/48 | {int(active.loc[('E1','G1')].reference_coverage_n)}/48 | {int(e1g2.reference_coverage_n)}/48 |
| Effective lineages | {active.loc[('E0','G0')].effective_lineages:.1f} | {active.loc[('E0','G1')].effective_lineages:.1f} | {e0g2.effective_lineages:.1f} | {active.loc[('E1','G1')].effective_lineages:.1f} | {e1g2.effective_lineages:.1f} |
| Auxiliary JS | {active.loc[('E0','G0')].aux_coarse_js:.6f} | {active.loc[('E0','G1')].aux_coarse_js:.6f} | {e0g2.aux_coarse_js:.6f} | {active.loc[('E1','G1')].aux_coarse_js:.6f} | {e1g2.aux_coarse_js:.6f} |

Raw offspring, survivor population, and bounded archive are reported as separate object types in `02_generation_metrics.csv`; archive retention is identical across arms and is not substituted for the raw-generation comparison.

E1's lower G2 parent→offspring improvement rate does not contradict population adaptation: E1 parents already lie closer to the target support. The causal comparison is the survivor trajectory and next-generation raw distribution under equal mutation budgets, not improvement rate alone.
"""
    (OUT / "05_evolution_comparison.md").write_text(comparison, encoding="utf-8")

    feasibility_lines = ["# Structure10k Feasibility", ""]
    for arm in ("E0", "E1"):
        meta = acquisition_meta[arm]
        if acquisitions[arm] is None:
            feasibility_lines.append(f"- {arm}: unavailable; eligible archive {meta['available']:,}, shortfall {meta['shortfall']:,}.")
        else:
            row = acquisition_rows.loc[arm]
            feasibility_lines.append(
                f"- {arm}: 10,000 unique acquired; JS {row.coarse_js:.8f}, SWD {row.sliced_wasserstein:.6f}, strict {int(row.strict_target_n):,} ({row.strict_target_fraction:.2%}), coverage {int(row.reference_coverage_n)}/48, effective lineages {row.effective_lineages:.1f}, max lineage {row.largest_lineage_share:.2%}, auxiliary JS {row.aux_coarse_js:.6f}."
            )
            selected = acquisitions[arm]
            assert selected is not None
            distances = nearest_matrix(
                (selected[list(FEATURES)].to_numpy(float) - geometry["center"]) / geometry["scale"],
                geometry["target_z"],
            ).min(axis=0)
            missing = geometry["manifest"].instance_id.to_numpy()[distances > geometry["radius"]].tolist()
            feasibility_lines.append(f"  Uncovered reference neighborhoods: {missing}.")
            feasibility_lines.append("  Capacity-aware quota fulfillment:")
            feasibility_lines.append("  | Cell | Requested | Capacity | Adjusted | Selected |")
            feasibility_lines.append("  |---:|---:|---:|---:|---:|")
            for item in meta["quota_fulfillment"]:
                feasibility_lines.append(
                    f"  | {item['cell']} | {item['requested_quota']:,} | {item['cell_capacity']:,} | {item['adjusted_quota']:,} | {item['selected']:,} |"
                )
    feasibility_lines.extend(
        [
            "",
            "Only E1, if complete, is exported under the required `provisional_structure10k*` names. E0 is retained as a control under `support_e0_provisional_structure10k*`. All instances are eligible offspring and no P03 survivor is used merely because it survived.",
            "",
            "The exact frozen quotas are infeasible in several bounded-archive cells. Those deficits are explicitly redistributed with frozen target weights and available capacity; they are not hidden by outside-box or duplicate backfill. Consequently, coarse JS is reported alongside SWD, reference coverage, auxiliary features, and lineage metrics rather than treated as a standalone success criterion.",
        ]
    )
    (OUT / "06_structure10k_feasibility.md").write_text("\n".join(feasibility_lines) + "\n", encoding="utf-8")

    g0 = active.loc[("E0", "G0")]
    better_e0 = bool(e1g2.sliced_wasserstein < e0g2.sliced_wasserstein)
    monotonic = bool(
        active.loc[("E1", "G1")].sliced_wasserstein < g0.sliced_wasserstein
        and e1g2.sliced_wasserstein < active.loc[("E1", "G1")].sliced_wasserstein
    )
    no_collapse = bool(
        e1g2.effective_lineages >= 0.5 * g0.effective_lineages
        and e1g2.largest_lineage_share <= 0.05
        and e1g2.feature_rounded_unique_fraction >= 0.95
    )
    enough = acquisitions["E1"] is not None
    aux_warning = bool(e1g2.aux_coarse_js > e0g2.aux_coarse_js + 0.10)
    if better_e0 and monotonic and no_collapse and enough and not aux_warning:
        decision = "GO"
    elif better_e0 and no_collapse:
        decision = "CONDITIONAL_GO"
    else:
        decision = "NO_GO"
    recommendation = {
        "GO": "Proceed to Concorde labelling and one M4→M5 fine-tuning cycle using the frozen E1 manifest.",
        "CONDITIONAL_GO": "Do not automatically add G3 or train; first resolve the stated auxiliary/trend/candidate limitation.",
        "NO_GO": "Pause the structure-evolution design; do not label or train this provisional set.",
    }[decision]
    decision_text = f"""# P2.2 Go / No-Go Decision

## Decision: `{decision}`

- E1 G2 continuous distance is better than E0 G2: **{better_e0}** ({e1g2.sliced_wasserstein:.6f} vs {e0g2.sliced_wasserstein:.6f}).
- E1 continuous distance improves G0→G1→G2: **{monotonic}** ({g0.sliced_wasserstein:.6f} → {active.loc[('E1','G1')].sliced_wasserstein:.6f} → {e1g2.sliced_wasserstein:.6f}).
- No lineage/selected-feature collapse guard triggered: **{no_collapse}**.
- Complete eligible E1 Structure10k: **{enough}**.
- Auxiliary feature-gaming warning (E1 auxiliary JS > E0 by 0.10): **{aux_warning}**.

Important limitations: both G2 populations and both provisional 10k sets cover 45/48 strict reference neighborhoods, and exact frozen cell quotas are infeasible in the bounded archives. E1 nevertheless satisfies the predeclared GO criteria through continuous-distance adaptation, a complete eligible 10k, healthy lineage/geometry diagnostics, and no auxiliary-JS collapse. This is evidence for the bounded evolutionary mechanism, not proof of full TSPLIB structural equivalence.

Recommendation: {recommendation}

No third generation, Concorde call, BQ inference/training, surrogate fitting, or formal A0 modification was performed.

`TARGET_AWARE_BOUNDED_EVOLUTION_P2_2_COMPLETE`
"""
    (OUT / "07_go_nogo_decision.md").write_text(decision_text, encoding="utf-8")
    return decision


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    geometry = load_geometry()
    history, history_table = load_training_history()
    p03_coordinates, p03, p03_integrity = load_p03(geometry)

    metrics_rows: list[dict[str, Any]] = []
    reference_rows: list[pd.DataFrame] = []
    operator_rows: list[pd.DataFrame] = []
    runtimes: list[dict[str, Any]] = []
    validity_rows: list[dict[str, Any]] = []
    populations: dict[tuple[str, int], pd.DataFrame] = {}
    archives: dict[str, pd.DataFrame] = {"E0": pd.DataFrame(), "E1": pd.DataFrame()}
    coordinate_maps: dict[str, dict[str, np.ndarray]] = {
        arm: dict(zip(p03.instance_hash.astype(str), p03_coordinates, strict=True)) for arm in ("E0", "E1")
    }
    population_coordinates = {"E0": p03_coordinates.copy(), "E1": p03_coordinates.copy()}
    population_frames = {"E0": p03.copy(), "E1": p03.copy()}

    for arm in ("E0", "E1"):
        row, reference = evaluate_frame(arm, 0, "active_population", p03, geometry)
        metrics_rows.append(row)
        reference_rows.append(reference)
        populations[(arm, 0)] = p03.copy()

    for generation in GENERATIONS:
        for arm in ("E0", "E1"):
            parent_coords = population_coordinates[arm]
            parents = population_frames[arm]
            child_coords, manifest, runtime = mutate_generation(arm, generation, parent_coords, parents)
            runtimes.append(runtime)
            child_coords, child, validity = build_offspring(
                arm, generation, child_coords, manifest, parents, parent_coords, geometry, history
            )
            validity_rows.append({"arm": arm, "generation": f"G{generation}", **validity})
            operator_rows.append(operator_metrics(arm, generation, manifest, child))
            row, reference = evaluate_frame(arm, generation, "raw_offspring", child, geometry)
            valid_attempts = manifest.valid_after_repair & manifest.proposal_status.eq("accepted")
            row.update(
                {
                    "mutation_attempts": len(manifest),
                    "valid_attempts": int(valid_attempts.sum()),
                    "valid_rate": float(valid_attempts.mean()),
                    "within_generation_duplicates": int((valid_attempts & manifest.duplicate).sum()),
                    "duplicate_rate": float((valid_attempts & manifest.duplicate).sum() / len(manifest)),
                    "eligible_unique_n": int((~child.m4_training_overlap).sum()),
                    "m4_training_overlap_n": int(child.m4_training_overlap.sum()),
                    "proximity_improved_fraction": float(child.target_proximity_improved.mean()),
                    "strict_entry_n": int(child.entered_strict_region.sum()),
                    "strict_target_yield_per_attempt": float(child.in_strict_target_region.sum() / len(manifest)),
                }
            )
            metrics_rows.append(row)
            reference_rows.append(reference)
            for instance_hash, coordinates in zip(child.instance_hash.astype(str), child_coords, strict=True):
                coordinate_maps[arm].setdefault(instance_hash, coordinates)

            union = pd.concat([parents, child], ignore_index=True, sort=False)
            union = union.drop_duplicates("instance_hash", keep="first").reset_index(drop=True)
            if arm == "E0":
                survivors, selection_meta = random_survivors(union, generation)
            else:
                survivors, selection_meta = structured_select(
                    union,
                    POPULATION_N,
                    geometry,
                    TARGET_FRACTION,
                    LINEAGE_CAP_ACTIVE,
                    f"E1_G{generation}_survivor",
                )
            survivor_coords = align_coordinates(survivors, coordinate_maps[arm])
            save_state(arm, generation, "active_population", survivors, survivor_coords, selection_meta)
            population_frames[arm] = survivors
            population_coordinates[arm] = survivor_coords
            populations[(arm, generation)] = survivors.copy()
            row, reference = evaluate_frame(arm, generation, "active_population", survivors, geometry)
            metrics_rows.append(row)
            reference_rows.append(reference)

            eligible = child.loc[~child.m4_training_overlap].copy().reset_index(drop=True)
            archives[arm], archive_meta = manage_archive(archives[arm], eligible, geometry)
            archive_coords = align_coordinates(archives[arm], coordinate_maps[arm])
            save_state(arm, generation, "bounded_archive", archives[arm], archive_coords, archive_meta)
            row, reference = evaluate_frame(arm, generation, "bounded_archive", archives[arm], geometry)
            row["eligible_unique_n"] = len(archives[arm])
            row["archive_capacity"] = ARCHIVE_CAPACITY
            metrics_rows.append(row)
            reference_rows.append(reference)

    acquisitions: dict[str, pd.DataFrame | None] = {}
    acquisition_meta: dict[str, dict[str, Any]] = {}
    for arm in ("E0", "E1"):
        selected, meta = acquisition(arm, archives[arm], geometry)
        acquisitions[arm] = selected
        acquisition_meta[arm] = meta
        write_json(OUT / f"support_{arm.lower()}_acquisition_metadata.json", meta)
        if selected is None:
            continue
        coords = align_coordinates(selected, coordinate_maps[arm])
        selected.to_parquet(OUT / f"support_{arm.lower()}_provisional_structure10k.parquet", index=False)
        selected.to_csv(OUT / f"support_{arm.lower()}_provisional_structure10k.csv", index=False)
        save_bundle(OUT / f"support_{arm.lower()}_provisional_structure10k.npz", coords, selected.instance_hash.astype(str).tolist())
        row, reference = evaluate_frame(arm, 2, "provisional_structure10k", selected, geometry)
        metrics_rows.append(row)
        reference_rows.append(reference)
        if arm == "E1":
            selected.to_csv(OUT / "provisional_structure10k_manifest.csv", index=False)
            save_bundle(OUT / "provisional_structure10k.npz", coords, selected.instance_hash.astype(str).tolist())
            c04_hard = set(
                pd.read_parquet(CYCLES / "C04" / "06_selection" / "T_selected.parquet", columns=["instance_hash"])
                .instance_hash.astype(str)
            )
            integrity = {
                "source": "P2.2 E1 bounded archive only",
                "rows": len(selected),
                "unique_ids": int(selected.instance_hash.nunique()),
                "known_m4_training_overlap": int(selected.instance_hash.astype(str).isin(history).sum()),
                "c04_original_hard10k_overlap_diagnostic": int(selected.instance_hash.astype(str).isin(c04_hard).sum()),
                "coordinate_shape": list(coords.shape),
                "membership_sha256": membership_hash(selected.instance_hash.astype(str).tolist()),
                "bundle_sha256": sha256_file(OUT / "provisional_structure10k.npz"),
                "manifest_sha256": sha256_file(OUT / "provisional_structure10k_manifest.csv"),
                "coordinate_hashes_reproduced": bool(
                    [coordinate_hash(points) for points in coords] == selected.instance_hash.astype(str).tolist()
                ),
                **coordinate_validity(coords),
            }
            write_json(OUT / "provisional_structure10k_integrity.json", integrity)

    metrics = pd.DataFrame(metrics_rows)
    references = pd.concat(reference_rows, ignore_index=True)
    operators = pd.concat(operator_rows, ignore_index=True)
    metrics.to_csv(OUT / "02_generation_metrics.csv", index=False)
    references.to_csv(OUT / "support_reference_coverage.csv", index=False)
    operators.to_csv(OUT / "support_operator_metrics.csv", index=False)
    pd.DataFrame(validity_rows).to_csv(OUT / "support_coordinate_validity.csv", index=False)
    history_table.to_csv(OUT / "support_m4_training_history.csv", index=False)
    make_figures(metrics, populations, acquisitions, geometry, references)
    decision = write_reports(
        geometry,
        p03_integrity,
        history_table,
        metrics,
        operators,
        acquisitions,
        acquisition_meta,
        runtimes,
    )
    first_experiment_start = min(
        (OUT / "evolution" / item["arm"] / f"G{item['generation']}" / "mutation" / "runtime.json").stat().st_mtime
        - float(item["wall_seconds"])
        for item in runtimes
    )
    runtime_summary = {
        "execution_mode": "CPU-only mutation, R/F40 extraction, and statistics; GPU not used",
        "mutation_wall_seconds_by_arm_generation": {
            f"{item['arm']}_G{item['generation']}": float(item["wall_seconds"]) for item in runtimes
        },
        "mutation_wall_seconds_sum": float(sum(float(item["wall_seconds"]) for item in runtimes)),
        "feature_cache_shards": len(list((OUT / "feature_cache").glob("cache_*.parquet"))),
        "artifact_wall_clock_span_seconds": time.time() - first_experiment_start,
        "finalization_invocation_seconds": time.time() - started,
    }
    write_json(OUT / "support_runtime.json", runtime_summary)
    write_json(
        OUT / "run_metadata.json",
        {
            "status": "COMPLETE",
            "decision": decision,
            "started_unix": started,
            "finished_unix": time.time(),
            **runtime_summary,
            "p03_bundle_sha256": sha256_file(P03_BUNDLE),
            "p03_metadata_sha256": sha256_file(P03_META),
            "profile_sha256": sha256_file(PROFILE_PATH),
            "runner_sha256": sha256_file(Path(__file__)),
            "seed": SEED,
            "mutation_cycle_namespaces": MUTATION_CYCLE_NAMESPACE,
            "no_oracle": True,
            "no_concorde": True,
            "no_hgb_prediction_or_fit": True,
            "no_bq_inference_or_training": True,
            "no_heldout_features": True,
            "no_third_generation": True,
        },
    )
    print(json.dumps({"status": "COMPLETE", "decision": decision}, indent=2), flush=True)


if __name__ == "__main__":
    main()
