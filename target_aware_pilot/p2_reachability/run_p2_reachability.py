#!/usr/bin/env python3
"""P2.0 structure-guided parent evolution reachability audit.

This driver is deliberately isolated below target_aware_pilot/p2_reachability.
It reuses the frozen C04 candidate bundle, FOGA11_BASE_V1 worker, salesperson
extractor, and P1 target geometry.  It never invokes an oracle, Concorde,
surrogate fitting, or BQ training.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
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


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "target_aware_pilot" / "p2_reachability"
FIG = OUT / "figures"
P1 = ROOT / "target_aware_pilot"
RUN = ROOT / "runs" / "BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1"
C04 = RUN / "A0_FULL_BASE" / "cycles" / "C04"
BASE_UNION = C04 / "05_scoring" / "union" / "C_union.parquet"
BASE_BUNDLE = C04 / "05_scoring" / "union" / "C_union.npz"
BASE_FEATURES = C04 / "candidate_features.parquet"
MUTATION_WORKER = ROOT / "scripts" / "foga11_base_v1_mutation_worker.py"
F40_WORKER = ROOT / "scripts" / "f40_one_shard.py"
R_EXTRACTOR = ROOT / "src" / "features" / "extract_salesperson_raw_features.R"
SCALER = ROOT / "assets" / "qd" / "reference_cloud" / "formal_raw_scaler.json"
PROFILE = ROOT / "src" / "mutation" / "foga2019_operators.yaml"

FEATURES = ("spn_distance_varcoeff", "spn_fraction_shorter_mean_distance")
MISSING_TARGETS = ("d198", "eil51", "ts225")
RANDOM_SEED = 20261024
MUTATION_CYCLE = 5
PARENT_N = 10_000
COVERAGE_PARENT_N = 7_000
FRONTIER_PARENT_N = 3_000
LINEAGE_CAP_INITIAL = 12
FRONTIER_POOL_PER_TARGET = 5_000
FEATURE_SHARD_N = 250
# Independent R subprocesses; the host has ample CPU/RAM and the formal run is
# complete.  Eight remains conservative while avoiding a roughly hour-long
# two-arm extraction at the production pipeline's four-worker setting.
FEATURE_WORKERS = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def membership_hash(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_bundle(path: Path, coordinates: np.ndarray, hashes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        coordinates=np.asarray(coordinates, dtype=np.float64),
        instance_hash=np.asarray(hashes),
    )
    temporary.replace(path)


def load_bundle(path: Path) -> tuple[np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as payload:
        return np.asarray(payload["coordinates"], dtype=np.float64), payload["instance_hash"].astype(str).tolist()


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
    indices = []
    for column, edge in enumerate(edges):
        inside &= (values[:, column] >= edge[0]) & (values[:, column] <= edge[-1])
        indices.append(np.clip(np.digitize(values[:, column], edge[1:-1]), 0, 2))
    return 3 * indices[0] + indices[1], inside


def distribution_metrics(counts: np.ndarray, target_counts: np.ndarray) -> dict[str, float | int]:
    counts = np.asarray(counts, dtype=float)
    target_counts = np.asarray(target_counts, dtype=float)
    probability = counts / counts.sum() if counts.sum() else np.zeros(9)
    target_probability = target_counts / target_counts.sum()
    positive = probability[probability > 0]
    entropy = float(-(positive * np.log(positive)).sum()) if len(positive) else 0.0
    return {
        "n_in_coarse_target_box": int(counts.sum()),
        "occupied_cells": int(np.sum(counts > 0)),
        "effective_cells": float(np.exp(entropy)),
        "largest_cell_share": float(probability.max()) if counts.sum() else float("nan"),
        "js_divergence_vs_target": float(jensenshannon(target_probability, probability, base=2.0) ** 2),
    }


def load_geometry() -> dict[str, Any]:
    manifest = pd.read_csv(P1 / "03_tsplib48_reference_manifest.csv")
    heldout = json.loads((P1 / "p1_heldout22_names.json").read_text())
    if len(manifest) != 48 or manifest.instance_id.nunique() != 48:
        raise RuntimeError("P1 TSPLIB48 manifest integrity failure")
    if set(manifest.instance_id) & set(heldout["names"]):
        raise RuntimeError("held-out identity overlaps development reference")

    extracted = pd.read_csv(P1 / "p1_tsplib48_normalization_sensitivity.csv")
    target = extracted[extracted.coordinate_variant.eq("aspect")].copy()
    target = target.set_index("instance_id").loc[manifest.instance_id.astype(str)].reset_index()
    target_values = target[list(FEATURES)].to_numpy(float)
    center = np.median(target_values, axis=0)
    scale = np.quantile(target_values, 0.75, axis=0) - np.quantile(target_values, 0.25, axis=0)
    target_z = (target_values - center) / scale
    target_pair = nearest_matrix(target_z, target_z)
    np.fill_diagonal(target_pair, np.inf)
    radius = float(np.median(target_pair.min(axis=1)))
    edges = [
        np.asarray(
            [
                target_values[:, i].min(),
                np.quantile(target_values[:, i], 1 / 3),
                np.quantile(target_values[:, i], 2 / 3),
                target_values[:, i].max(),
            ]
        )
        for i in range(2)
    ]
    target_cell, target_inside = coarse_cells(target_values, edges)
    if not target_inside.all():
        raise RuntimeError("target cell construction failure")
    target_counts = np.bincount(target_cell, minlength=9)

    union = pd.read_parquet(BASE_UNION)
    features = pd.read_parquet(BASE_FEATURES, columns=["instance_hash", *FEATURES])
    if len(union) != 119_530 or union.instance_hash.nunique() != len(union):
        raise RuntimeError("C04 baseline is not the frozen 119,530 unique union")
    # C04 union already carries the QD subset, which includes distance CV but
    # not necessarily every selected F40 feature.  Keep one authoritative copy
    # of both dimensions from candidate_features.parquet and avoid pandas
    # suffixes changing the frozen feature names.
    union_metadata = union.drop(columns=[name for name in FEATURES if name in union.columns])
    candidates = union_metadata.merge(features, on="instance_hash", validate="one_to_one", sort=False)
    coordinates, hashes = load_bundle(BASE_BUNDLE)
    if hashes != union.instance_hash.astype(str).tolist() or len(coordinates) != len(union):
        raise RuntimeError("C04 union bundle/metadata alignment failure")

    values = candidates[list(FEATURES)].to_numpy(float)
    candidate_z = (values - center) / scale
    distance = nearest_matrix(candidate_z, target_z)
    nearest_reference = distance.argmin(axis=1)
    nearest_distance = distance[np.arange(len(distance)), nearest_reference]
    candidate_cell, candidate_inside = coarse_cells(values, edges)
    in_region = nearest_distance <= radius
    if int(in_region.sum()) != 8969:
        raise RuntimeError(f"P1 strict region changed: {int(in_region.sum())} != 8969")
    target_to_baseline = distance.min(axis=0)
    if int(np.sum(target_to_baseline <= radius)) != 45:
        raise RuntimeError("P1 target coverage changed")

    candidates["nearest_reference_index"] = nearest_reference
    candidates["nearest_reference"] = manifest.instance_id.to_numpy()[nearest_reference]
    candidates["target_distance"] = nearest_distance
    candidates["in_strict_target_region"] = in_region
    candidates["coarse_cell"] = candidate_cell
    candidates["inside_target_box"] = candidate_inside
    candidates["coord_index"] = np.arange(len(candidates))
    return {
        "manifest": manifest,
        "target": target,
        "target_values": target_values,
        "target_z": target_z,
        "center": center,
        "scale": scale,
        "radius": radius,
        "edges": edges,
        "target_cell": target_cell,
        "target_counts": target_counts,
        "candidates": candidates,
        "coordinates": coordinates,
        "target_to_baseline": target_to_baseline,
    }


def capacity_weighted_quotas(total: int, weights: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    """Weighted fair allocation with deterministic capacity redistribution."""
    weights = np.asarray(weights, dtype=float)
    capacity = np.asarray(capacity, dtype=int)
    quota = np.zeros(len(weights), dtype=int)
    for _ in range(total):
        active = np.flatnonzero((quota < capacity) & (weights > 0))
        if not len(active):
            raise RuntimeError("insufficient target-support capacity for coverage parent quota")
        score = quota[active] / weights[active]
        chosen = int(active[np.argmin(score)])
        quota[chosen] += 1
    return quota


def lineage_limited_take(
    ordered_indices: list[int],
    needed: int,
    candidates: pd.DataFrame,
    selected: set[int],
    lineage_counts: dict[str, int],
    initial_cap: int = LINEAGE_CAP_INITIAL,
) -> tuple[list[int], int]:
    taken: list[int] = []
    cap = initial_cap
    while len(taken) < needed:
        progress = False
        for index in ordered_indices:
            if index in selected:
                continue
            lineage = str(candidates.at[index, "lineage_root"])
            if lineage_counts.get(lineage, 0) >= cap:
                continue
            selected.add(index)
            taken.append(index)
            lineage_counts[lineage] = lineage_counts.get(lineage, 0) + 1
            progress = True
            if len(taken) == needed:
                break
        if len(taken) == needed:
            break
        if not progress and cap >= 128:
            break
        cap = min(128, cap * 2)
    return taken, cap


def structure_parent_selection(geometry: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates = geometry["candidates"]
    target_cell = geometry["target_cell"]
    target_counts = geometry["target_counts"]
    region = candidates.in_strict_target_region.to_numpy(bool)
    region_capacity = np.bincount(candidates.loc[region, "coarse_cell"].astype(int), minlength=9)
    coverage_quota = capacity_weighted_quotas(COVERAGE_PARENT_N, target_counts, region_capacity)
    selected: set[int] = set()
    lineage_counts: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    cap_used = LINEAGE_CAP_INITIAL

    # Candidate order is round-robin across references, with references having
    # fewer currently reachable candidates placed first.  Distance breaks ties.
    for cell in range(9):
        needed = int(coverage_quota[cell])
        if not needed:
            continue
        pool = candidates[region & candidates.coarse_cell.eq(cell).to_numpy()].copy()
        pool = pool.sort_values(["nearest_reference", "target_distance", "instance_hash"], kind="stable")
        pool["rank_within_reference"] = pool.groupby("nearest_reference", sort=False).cumcount()
        availability = pool.groupby("nearest_reference").size().to_dict()
        pool["reference_availability"] = pool.nearest_reference.map(availability)
        pool = pool.sort_values(
            ["rank_within_reference", "reference_availability", "target_distance", "instance_hash"],
            kind="stable",
        )
        taken, cap = lineage_limited_take(
            pool.index.astype(int).tolist(), needed, candidates, selected, lineage_counts
        )
        cap_used = max(cap_used, cap)
        if len(taken) != needed:
            raise RuntimeError(f"coverage cell {cell} filled {len(taken)}/{needed}")
        for index in taken:
            rows.append(
                {
                    "row_index": index,
                    "selection_component": "target_coverage",
                    "selection_stratum": f"coarse_cell_{cell}",
                    "assigned_target": candidates.at[index, "nearest_reference"],
                }
            )

    # Frontier parents are outside the frozen region.  For each uncovered
    # reference, use the 5,000 closest remaining candidates and round-robin
    # eight directions in the frozen standardized feature space.
    target_names = geometry["manifest"].instance_id.astype(str).tolist()
    target_z = geometry["target_z"]
    candidate_values = candidates[list(FEATURES)].to_numpy(float)
    candidate_z = (candidate_values - geometry["center"]) / geometry["scale"]
    frontier_quota = FRONTIER_PARENT_N // len(MISSING_TARGETS)
    frontier_caps: dict[str, int] = {}
    for name in MISSING_TARGETS:
        target_index = target_names.index(name)
        delta = candidate_z - target_z[target_index]
        direct_distance = np.sqrt(np.sum(delta * delta, axis=1))
        eligible = np.flatnonzero(~region & ~np.isin(np.arange(len(candidates)), np.fromiter(selected, int)))
        order = eligible[np.argsort(direct_distance[eligible], kind="stable")]
        order = order[: min(FRONTIER_POOL_PER_TARGET, len(order))]
        angle = np.arctan2(delta[order, 1], delta[order, 0])
        angle_bin = np.floor((angle + np.pi) / (2 * np.pi) * 8).astype(int).clip(0, 7)
        frontier = pd.DataFrame(
            {
                "index": order,
                "distance": direct_distance[order],
                "angle_bin": angle_bin,
            }
        ).sort_values(["angle_bin", "distance", "index"], kind="stable")
        frontier["rank_within_angle"] = frontier.groupby("angle_bin", sort=False).cumcount()
        frontier = frontier.sort_values(
            ["rank_within_angle", "distance", "angle_bin", "index"], kind="stable"
        )
        taken, cap = lineage_limited_take(
            frontier["index"].astype(int).tolist(),
            frontier_quota,
            candidates,
            selected,
            lineage_counts,
        )
        cap_used = max(cap_used, cap)
        frontier_caps[name] = cap
        if len(taken) != frontier_quota:
            raise RuntimeError(f"frontier {name} filled {len(taken)}/{frontier_quota}")
        for index in taken:
            rows.append(
                {
                    "row_index": index,
                    "selection_component": "frontier_exploration",
                    "selection_stratum": f"frontier_{name}",
                    "assigned_target": name,
                }
            )

    selection = pd.DataFrame(rows)
    if len(selection) != PARENT_N or selection.row_index.nunique() != PARENT_N:
        raise RuntimeError("structure-guided selection is not exactly 10,000 unique parents")
    selected_frame = selection.merge(
        candidates.reset_index(names="row_index"), on="row_index", validate="one_to_one", sort=False
    )
    metadata = {
        "coverage_parent_n": COVERAGE_PARENT_N,
        "frontier_parent_n": FRONTIER_PARENT_N,
        "coverage_quota_by_cell": coverage_quota.astype(int).tolist(),
        "region_capacity_by_cell": region_capacity.astype(int).tolist(),
        "target_count_by_cell": target_counts.astype(int).tolist(),
        "lineage_cap_initial": LINEAGE_CAP_INITIAL,
        "lineage_cap_max_used": cap_used,
        "selected_lineage_n": int(selected_frame.lineage_root.nunique()),
        "largest_lineage_count": int(selected_frame.lineage_root.value_counts().max()),
        "frontier_pool_per_target": FRONTIER_POOL_PER_TARGET,
        "frontier_quota_by_target": {name: frontier_quota for name in MISSING_TARGETS},
        "frontier_cap_used": frontier_caps,
    }
    return selected_frame, metadata


def random_parent_selection(geometry: dict[str, Any]) -> pd.DataFrame:
    candidates = geometry["candidates"]
    canonical = candidates.sort_values("instance_hash", kind="stable")
    chosen = np.random.default_rng(RANDOM_SEED).choice(len(canonical), PARENT_N, replace=False)
    output = canonical.iloc[chosen].copy().reset_index(drop=True)
    output["selection_component"] = "random_control"
    output["selection_stratum"] = "canonical_hash_uniform_without_replacement"
    output["assigned_target"] = output.nearest_reference
    return output


def prepare_parents(geometry: dict[str, Any]) -> dict[str, pd.DataFrame]:
    parent_root = OUT / "parents"
    parent_root.mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / "parent_selection_metadata.json"
    if all((parent_root / arm / "parents.parquet").is_file() for arm in ("E0", "E1")) and protocol_path.is_file():
        return {arm: pd.read_parquet(parent_root / arm / "parents.parquet") for arm in ("E0", "E1")}

    e0 = random_parent_selection(geometry)
    e1, metadata = structure_parent_selection(geometry)
    outputs = {"E0": e0, "E1": e1}
    base_coords = geometry["coordinates"]
    for arm, frame in outputs.items():
        directory = parent_root / arm
        directory.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(directory / "parents.parquet", index=False)
        indices = frame.coord_index.to_numpy(int)
        hashes = frame.instance_hash.astype(str).tolist()
        save_bundle(directory / "parents.npz", base_coords[indices], hashes)
        frame[["instance_hash", "selection_component", "selection_stratum", "assigned_target", "lineage_root", "target_distance", "coarse_cell"]].to_csv(
            directory / "parent_membership.csv", index=False
        )
    metadata.update(
        {
            "random_seed": RANDOM_SEED,
            "mutation_cycle_seed_namespace": MUTATION_CYCLE,
            "parent_n_each": PARENT_N,
            "operators": 11,
            "attempts_each": 110_000,
            "E0_membership_sha256": membership_hash(e0.instance_hash.astype(str).tolist()),
            "E1_membership_sha256": membership_hash(e1.instance_hash.astype(str).tolist()),
            "parent_overlap_n": int(len(set(e0.instance_hash) & set(e1.instance_hash))),
        }
    )
    write_json(protocol_path, metadata)
    return outputs


def run_mutation(arm: str) -> None:
    directory = OUT / "evolution" / arm / "mutation"
    runtime = directory / "FOGA11_BASE_V1_RUNTIME.json"
    if runtime.is_file() and (directory / "offspring_coordinates.npz").is_file() and (directory / "mutation_manifest.parquet").is_file():
        return
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(MUTATION_WORKER),
        "--parents",
        str(OUT / "parents" / arm / "parents.npz"),
        "--cycle",
        str(MUTATION_CYCLE),
        "--output",
        str(directory),
    ]
    print(f"[P2.0] starting {arm} mutation: 110,000 attempts", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def prepare_new_offspring(arm: str, geometry: dict[str, Any]) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    arm_root = OUT / "evolution" / arm
    new_bundle = arm_root / "new_unique_offspring.npz"
    new_meta_path = arm_root / "new_unique_offspring.parquet"
    cumulative_bundle = arm_root / "cumulative_candidate_pool.npz"
    cumulative_meta_path = arm_root / "cumulative_candidate_pool.parquet"
    if all(path.is_file() for path in (new_bundle, new_meta_path, cumulative_bundle, cumulative_meta_path)):
        coords, hashes = load_bundle(new_bundle)
        return coords, hashes, pd.read_parquet(new_meta_path)

    mutation = arm_root / "mutation"
    offspring_coords, offspring_hashes = load_bundle(mutation / "offspring_coordinates.npz")
    manifest = pd.read_parquet(mutation / "mutation_manifest.parquet")
    accepted = manifest.valid_after_repair & manifest.proposal_status.eq("accepted") & ~manifest.duplicate
    accepted_meta = manifest.loc[accepted].reset_index(drop=True)
    if accepted_meta.child_hash.astype(str).tolist() != offspring_hashes:
        raise RuntimeError(f"{arm} mutation bundle/manifest alignment failure")
    baseline_hashes = set(geometry["candidates"].instance_hash.astype(str))
    is_new = np.asarray([value not in baseline_hashes for value in offspring_hashes], dtype=bool)
    new_coords = offspring_coords[is_new]
    new_hashes = [value for value, keep in zip(offspring_hashes, is_new, strict=True) if keep]
    new_meta = accepted_meta.loc[is_new].reset_index(drop=True)
    new_meta["instance_hash"] = new_hashes
    new_meta["coordinate_origin"] = "p2_new_offspring"
    new_meta["arm"] = arm
    new_meta["coord_index"] = np.arange(len(new_meta))
    new_meta.to_parquet(new_meta_path, index=False)
    save_bundle(new_bundle, new_coords, new_hashes)

    base = geometry["candidates"].copy()
    base["p2_origin"] = "c04_baseline"
    added = new_meta.copy()
    added["p2_origin"] = f"{arm}_new_offspring"
    cumulative = pd.concat([base, added], ignore_index=True, sort=False)
    if cumulative.instance_hash.nunique() != len(cumulative):
        raise RuntimeError(f"{arm} cumulative pool is not unique")
    cumulative["p2_coord_index"] = np.arange(len(cumulative))
    cumulative.to_parquet(cumulative_meta_path, index=False)
    save_bundle(
        cumulative_bundle,
        np.concatenate([geometry["coordinates"], new_coords]),
        cumulative.instance_hash.astype(str).tolist(),
    )
    return new_coords, new_hashes, new_meta


def scaler_params() -> dict[str, tuple[float, float]]:
    payload = json.loads(SCALER.read_text())
    table = {item["feature_name"]: item for item in payload["parameters"]}
    return {name: (float(table[name]["center_median"]), float(table[name]["scale"])) for name in FEATURES}


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
        raise RuntimeError(f"feature shard failed rc={process.returncode}: {stderr.decode(errors='replace')[-2000:]}")
    return stdout.decode(errors="replace").strip()


def load_feature_cache() -> pd.DataFrame:
    files = sorted((OUT / "feature_cache").glob("cache_*.parquet"))
    if not files:
        return pd.DataFrame(columns=["instance_hash", *FEATURES])
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    frame = frame.drop_duplicates("instance_hash", keep="first")
    return frame


def extract_features(arm: str, coordinates: np.ndarray, hashes: list[str]) -> pd.DataFrame:
    final_path = OUT / "evolution" / arm / "new_unique_offspring_features.parquet"
    if final_path.is_file():
        frame = pd.read_parquet(final_path)
        if frame.instance_hash.astype(str).tolist() == hashes:
            return frame
    cache = load_feature_cache()
    known = set(cache.instance_hash.astype(str))
    missing_indices = [i for i, value in enumerate(hashes) if value not in known]
    if missing_indices:
        cache_dir = OUT / "feature_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        jobs = []
        params = scaler_params()
        for start in range(0, len(missing_indices), FEATURE_SHARD_N):
            indices = missing_indices[start : start + FEATURE_SHARD_N]
            block_hashes = [hashes[i] for i in indices]
            tag = membership_hash(block_hashes)[:16]
            jobs.append(
                {
                    "coordinates": coordinates[np.asarray(indices, int)],
                    "hashes": block_hashes,
                    "names": list(FEATURES),
                    "out": str(cache_dir / f"cache_{tag}.parquet"),
                    "rscript": shutil.which("Rscript") or "Rscript",
                    "r_script": str(R_EXTRACTOR),
                    "params": params,
                }
            )
        print(f"[P2.0] {arm} extracting {len(missing_indices)} new feature rows in {len(jobs)} shards", flush=True)
        done = 0
        with ThreadPoolExecutor(max_workers=min(FEATURE_WORKERS, len(jobs))) as executor:
            futures = [executor.submit(run_feature_job, job) for job in jobs]
            for future in as_completed(futures):
                future.result()
                done += 1
                if done == 1 or done == len(jobs) or done % 25 == 0:
                    print(f"[P2.0] {arm} feature shards {done}/{len(jobs)}", flush=True)
        cache = load_feature_cache()
    lookup = cache.set_index("instance_hash")
    missing = [value for value in hashes if value not in lookup.index]
    if missing:
        raise RuntimeError(f"feature cache missing {len(missing)} {arm} hashes")
    result = lookup.loc[hashes].reset_index()[["instance_hash", *FEATURES]]
    if np.isinf(result[list(FEATURES)].to_numpy(float)).any() or result[list(FEATURES)].isna().any().any():
        raise RuntimeError(f"{arm} selected features contain missing/infinite values")
    result.to_parquet(final_path, index=False)
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
        np.add.at(
            histogram,
            (batch_index, bins[:, :, 0].ravel(), bins[:, :, 1].ravel()),
            1.0,
        )
        magnitude = np.abs(np.fft.fftshift(np.fft.fft2(histogram, axes=(1, 2)), axes=(1, 2)))
        magnitude[:, 16, 16] = 0.0
        fft_values[start : start + len(points)] = magnitude[:, fft_mask].mean(axis=1)
        if (start // block) % 100 == 0:
            print(f"[P2.0] EvoReal diagnostics {min(start + block, len(coordinates))}/{len(coordinates)}", flush=True)
    return fft_values, nn_values


def add_child_metrics(
    arm: str,
    coordinates: np.ndarray,
    hashes: list[str],
    metadata: pd.DataFrame,
    features: pd.DataFrame,
    geometry: dict[str, Any],
) -> pd.DataFrame:
    diagnostic_path = OUT / "evolution" / arm / "new_offspring_diagnostics.parquet"
    if diagnostic_path.is_file():
        cached = pd.read_parquet(diagnostic_path)
        if cached.instance_hash.astype(str).tolist() == hashes:
            return cached
    frame = metadata.merge(features, on="instance_hash", validate="one_to_one", sort=False)
    values = frame[list(FEATURES)].to_numpy(float)
    child_z = (values - geometry["center"]) / geometry["scale"]
    distance = nearest_matrix(child_z, geometry["target_z"])
    nearest = distance.argmin(axis=1)
    frame["nearest_reference_index"] = nearest
    frame["nearest_reference"] = geometry["manifest"].instance_id.to_numpy()[nearest]
    frame["target_distance"] = distance[np.arange(len(distance)), nearest]
    frame["in_strict_target_region"] = frame.target_distance <= geometry["radius"]
    cell, inside = coarse_cells(values, geometry["edges"])
    frame["coarse_cell"] = cell
    frame["inside_target_box"] = inside
    parent_distance = geometry["candidates"].set_index("instance_hash").target_distance
    frame["parent_target_distance"] = frame.parent_hash.astype(str).map(parent_distance)
    if frame.parent_target_distance.isna().any():
        raise RuntimeError(f"{arm} child parent missing from C04 baseline")
    frame["target_distance_change"] = frame.target_distance - frame.parent_target_distance
    frame["target_proximity_improved"] = frame.target_distance_change < 0
    frame["entered_strict_region"] = (
        (frame.parent_target_distance > geometry["radius"]) & frame.in_strict_target_region
    )
    fft_energy, nn_ratio = evoreal_diagnostics(coordinates)
    frame["fft_energy"] = fft_energy
    frame["nn_ratio"] = nn_ratio
    frame.to_parquet(diagnostic_path, index=False)
    return frame


def auxiliary_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    target_aux = pd.read_csv(P1 / "p1_evoreal_feature_diagnostic.csv")[["fft_energy", "nn_ratio"]].to_numpy(float)
    edges = [
        np.asarray([target_aux[:, i].min(), np.quantile(target_aux[:, i], 1 / 3), np.quantile(target_aux[:, i], 2 / 3), target_aux[:, i].max()])
        for i in range(2)
    ]
    target_cell, _ = coarse_cells(target_aux, edges)
    values = frame[["fft_energy", "nn_ratio"]].to_numpy(float)
    cell, inside = coarse_cells(values, edges)
    target_counts = np.bincount(target_cell, minlength=9)
    counts = np.bincount(cell[inside], minlength=9)
    metrics = distribution_metrics(counts, target_counts)
    return {f"aux_{key}": value for key, value in metrics.items()} | {
        "aux_fft_median": float(frame.fft_energy.median()),
        "aux_nn_ratio_median": float(frame.nn_ratio.median()),
    }


def evaluate_arm(
    arm: str,
    child: pd.DataFrame,
    geometry: dict[str, Any],
    parents: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    manifest = pd.read_parquet(OUT / "evolution" / arm / "mutation" / "mutation_manifest.parquet")
    valid = manifest.valid_after_repair & manifest.proposal_status.eq("accepted")
    internal_unique = valid & ~manifest.duplicate
    values = child[list(FEATURES)].to_numpy(float)
    new_cell = child.coarse_cell.to_numpy(int)
    new_inside = child.inside_target_box.to_numpy(bool)
    new_counts = np.bincount(new_cell[new_inside], minlength=9)
    baseline = geometry["candidates"]
    base_cell = baseline.coarse_cell.to_numpy(int)
    base_inside = baseline.inside_target_box.to_numpy(bool)
    base_counts = np.bincount(base_cell[base_inside], minlength=9)
    cumulative_counts = base_counts + new_counts
    target_counts = geometry["target_counts"]
    new_distribution = distribution_metrics(new_counts, target_counts)
    cumulative_distribution = distribution_metrics(cumulative_counts, target_counts)

    child_z = (values - geometry["center"]) / geometry["scale"]
    distance = nearest_matrix(child_z, geometry["target_z"])
    target_to_new = distance.min(axis=0)
    target_to_cumulative = np.minimum(geometry["target_to_baseline"], target_to_new)
    new_region_n = int(child.in_strict_target_region.sum())
    cumulative_region_n = 8969 + new_region_n
    rounded_unique = int(np.unique(np.round(values, 8), axis=0).shape[0])
    summary: dict[str, Any] = {
        "arm": arm,
        "parent_n": len(parents),
        "parent_unique_n": int(parents.instance_hash.nunique()),
        "parent_lineage_n": int(parents.lineage_root.nunique()),
        "mutation_attempts": len(manifest),
        "valid_attempts": int(valid.sum()),
        "valid_rate": float(valid.mean()),
        "within_arm_duplicate_attempts": int((valid & manifest.duplicate).sum()),
        "internal_unique_offspring_n": int(internal_unique.sum()),
        "new_unique_offspring_n": len(child),
        "existing_pool_duplicate_n": int(internal_unique.sum() - len(child)),
        "new_unique_rate_per_attempt": float(len(child) / len(manifest)),
        "new_strict_target_region_unique_n": new_region_n,
        "new_target_region_fraction": float(new_region_n / len(child)),
        "cumulative_unique_candidates_n": int(119_530 + len(child)),
        "cumulative_strict_target_region_unique_n": cumulative_region_n,
        "new_target_neighborhood_coverage_n": int(np.sum(target_to_new <= geometry["radius"])),
        "cumulative_target_neighborhood_coverage_n": int(np.sum(target_to_cumulative <= geometry["radius"])),
        "target_nearest_distance_mean_new": float(target_to_new.mean()),
        "target_nearest_distance_mean_cumulative": float(target_to_cumulative.mean()),
        "target_nearest_distance_p90_cumulative": float(np.quantile(target_to_cumulative, 0.9)),
        "offspring_target_distance_change_mean": float(child.target_distance_change.mean()),
        "offspring_target_distance_change_median": float(child.target_distance_change.median()),
        "offspring_proximity_improved_fraction": float(child.target_proximity_improved.mean()),
        "offspring_entered_strict_region_n": int(child.entered_strict_region.sum()),
        "selected_feature_rounded_unique_n": rounded_unique,
        "selected_feature_rounded_unique_fraction": float(rounded_unique / len(child)),
        **{f"new_{key}": value for key, value in new_distribution.items()},
        **{f"cumulative_{key}": value for key, value in cumulative_distribution.items()},
        **auxiliary_metrics(child),
    }
    reference = geometry["manifest"][["instance_id", "evoreal_split"]].copy()
    reference["baseline_nearest_distance"] = geometry["target_to_baseline"]
    reference["new_offspring_nearest_distance"] = target_to_new
    reference["cumulative_nearest_distance"] = target_to_cumulative
    reference["baseline_covered"] = reference.baseline_nearest_distance <= geometry["radius"]
    reference["new_offspring_covered"] = reference.new_offspring_nearest_distance <= geometry["radius"]
    reference["cumulative_covered"] = reference.cumulative_nearest_distance <= geometry["radius"]
    reference["arm"] = arm

    operator_rows = []
    for operator, part in manifest.groupby("operator_id", sort=True):
        part_valid = part.valid_after_repair & part.proposal_status.eq("accepted")
        hashes = set(part.loc[part_valid & ~part.duplicate, "child_hash"].astype(str))
        new_part = child[child.instance_hash.astype(str).isin(hashes)]
        operator_rows.append(
            {
                "arm": arm,
                "operator_id": operator,
                "attempts": len(part),
                "valid_attempts": int(part_valid.sum()),
                "within_arm_unique": int((part_valid & ~part.duplicate).sum()),
                "new_unique_vs_c04": len(new_part),
                "new_strict_target_region": int(new_part.in_strict_target_region.sum()),
                "target_region_success_rate_per_attempt": float(new_part.in_strict_target_region.sum() / len(part)),
                "strict_region_entries": int(new_part.entered_strict_region.sum()),
                "mean_target_distance_change": float(new_part.target_distance_change.mean()),
                "proximity_improved_fraction": float(new_part.target_proximity_improved.mean()),
            }
        )
    return summary, reference, pd.DataFrame(operator_rows), new_counts, cumulative_counts


def result_rows(
    summary: dict[str, Any],
    reference: pd.DataFrame,
    new_counts: np.ndarray,
    cumulative_counts: np.ndarray,
    target_counts: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [{"record_type": "summary", **summary}]
    for row in reference.itertuples(index=False):
        rows.append(
            {
                "record_type": "reference_reachability",
                "arm": row.arm,
                "instance_id": row.instance_id,
                "evoreal_split": row.evoreal_split,
                "baseline_nearest_distance": row.baseline_nearest_distance,
                "new_offspring_nearest_distance": row.new_offspring_nearest_distance,
                "cumulative_nearest_distance": row.cumulative_nearest_distance,
                "baseline_covered": row.baseline_covered,
                "new_offspring_covered": row.new_offspring_covered,
                "cumulative_covered": row.cumulative_covered,
            }
        )
    for cell in range(9):
        rows.append(
            {
                "record_type": "coarse_3x3_cell",
                "arm": summary["arm"],
                "cell": cell,
                "target_count": int(target_counts[cell]),
                "new_offspring_count": int(new_counts[cell]),
                "cumulative_count": int(cumulative_counts[cell]),
                "new_offspring_share": float(new_counts[cell] / max(1, new_counts.sum())),
                "cumulative_share": float(cumulative_counts[cell] / max(1, cumulative_counts.sum())),
            }
        )
    return pd.DataFrame(rows)


def make_figures(
    geometry: dict[str, Any],
    diagnostics: dict[str, pd.DataFrame],
    references: pd.DataFrame,
    summaries: dict[str, dict[str, Any]],
    counts: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    target = geometry["target_values"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), sharex=True, sharey=True)
    rng = np.random.default_rng(RANDOM_SEED)
    for ax, arm in zip(axes, ("E0", "E1")):
        frame = diagnostics[arm]
        take = rng.choice(len(frame), min(30_000, len(frame)), replace=False)
        ax.hexbin(frame.iloc[take][FEATURES[0]], frame.iloc[take][FEATURES[1]], gridsize=45, mincnt=1, bins="log", cmap="Blues")
        ax.scatter(target[:, 0], target[:, 1], s=28, color="#d62728", edgecolor="black", linewidth=0.3, label="TSPLIB48")
        ax.set_title(f"{arm} new offspring")
        ax.set_xlabel(FEATURES[0].replace("spn_", ""))
    axes[0].set_ylabel(FEATURES[1].replace("spn_", ""))
    axes[0].legend()
    fig.suptitle("Equal-budget offspring structure distribution")
    fig.tight_layout()
    fig.savefig(FIG / "01_e0_e1_offspring_structure.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for arm, color in (("E0", "#4c78a8"), ("E1", "#f58518")):
        delta = diagnostics[arm].target_distance_change.to_numpy(float)
        axes[0].hist(np.clip(delta, -1, 1), bins=80, density=True, histtype="step", linewidth=1.7, color=color, label=arm)
        axes[1].bar(arm, summaries[arm]["offspring_proximity_improved_fraction"], color=color)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("child target distance - parent target distance (clipped)")
    axes[0].set_ylabel("density")
    axes[0].legend()
    axes[1].set_ylabel("fraction improved")
    axes[1].set_ylim(0, 1)
    fig.suptitle("Parent to offspring target-proximity change")
    fig.tight_layout()
    fig.savefig(FIG / "02_parent_offspring_distance_change.png", dpi=180)
    plt.close(fig)

    pivot = references.pivot(index="instance_id", columns="arm", values="cumulative_nearest_distance").loc[
        geometry["manifest"].instance_id
    ]
    baseline = references[references.arm.eq("E0")].set_index("instance_id").loc[pivot.index, "baseline_nearest_distance"]
    fig, ax = plt.subplots(figsize=(14, 5.5))
    x = np.arange(len(pivot))
    ax.plot(x, baseline, color="gray", linewidth=1.2, marker="o", markersize=2.5, label="C04 baseline")
    ax.plot(x, pivot.E0, color="#4c78a8", linewidth=1.2, marker="o", markersize=2.5, label="E0 cumulative")
    ax.plot(x, pivot.E1, color="#f58518", linewidth=1.2, marker="o", markersize=2.5, label="E1 cumulative")
    ax.axhline(geometry["radius"], color="black", linestyle="--", linewidth=0.9, label="strict radius")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=75, fontsize=7)
    ax.set_ylabel("nearest candidate distance")
    ax.set_title("TSPLIB48 neighborhood reachability")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "03_reference_coverage_comparison.png", dpi=180)
    plt.close(fig)

    target_probability = geometry["target_counts"] / geometry["target_counts"].sum()
    baseline = geometry["candidates"]
    base_counts = np.bincount(
        baseline.loc[baseline.inside_target_box, "coarse_cell"].astype(int), minlength=9
    )
    panels = [
        ("TSPLIB48", target_probability),
        ("C04", base_counts / base_counts.sum()),
        ("E0 cumulative", counts["E0"][1] / counts["E0"][1].sum()),
        ("E1 cumulative", counts["E1"][1] / counts["E1"][1].sum()),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, (title, probability) in zip(axes, panels):
        image = ax.imshow(probability.reshape(3, 3).T, origin="lower", cmap="viridis", vmin=0, vmax=max(p.max() for _, p in panels))
        for i in range(3):
            for j in range(3):
                ax.text(i, j, f"{probability[3*i+j]:.2f}", ha="center", va="center", color="white" if probability[3*i+j] > 0.3 else "black", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("distance CV tercile")
        ax.set_ylabel("fraction-shorter tercile")
    # Reserve a separate colorbar axis so it does not obscure the E1 panel.
    colorbar_axis = fig.add_axes([0.925, 0.16, 0.012, 0.64])
    fig.colorbar(image, cax=colorbar_axis)
    fig.suptitle("Frozen 3x3 structural distribution")
    fig.subplots_adjust(left=0.05, right=0.90, bottom=0.15, top=0.82, wspace=0.35)
    fig.savefig(FIG / "04_coarse_distribution_comparison.png", dpi=180)
    plt.close(fig)


def write_reports(
    geometry: dict[str, Any],
    parents: dict[str, pd.DataFrame],
    summaries: dict[str, dict[str, Any]],
    references: pd.DataFrame,
    operators: pd.DataFrame,
) -> str:
    meta = json.loads((OUT / "parent_selection_metadata.json").read_text())
    protocol = f"""# P2.0 Parent Selection Protocol

## Frozen inputs

- Source pool: C04 Parent+Child union, 119,530 unique instances.
- Target references: the P1-frozen TSPLIB48 development set. Held-out TSPLIB22 identities were checked only for disjointness and no held-out coordinates or features were loaded.
- Structure coordinates: `{FEATURES[0]}` and `{FEATURES[1]}` with the P1-frozen target median/IQR standardization.
- Strict radius: {geometry['radius']:.12f}, unchanged from P1.
- Parents per arm: 10,000 unique; operators: 11; attempts per arm: 110,000; mutation depth: 1; mutation cycle seed namespace: {MUTATION_CYCLE}.
- No BQ loss, predicted hardness, Oracle, Concorde, surrogate fitting, or BQ training is used.

## E0 random control

Candidates are sorted by instance hash and sampled uniformly without replacement using NumPy seed {RANDOM_SEED}. The resulting membership hash is `{meta['E0_membership_sha256']}`.

## E1 structure-guided selection

- 7,000 Target Coverage parents and 3,000 Frontier Exploration parents.
- Target Coverage cell weights are the frozen TSPLIB48 3x3 counts `{meta['target_count_by_cell']}`. Capacity-aware deterministic redistribution is applied because the strict-region availability is `{meta['region_capacity_by_cell']}`. Final parent quotas are `{meta['coverage_quota_by_cell']}`.
- Within each cell, selection round-robins across nearest TSPLIB references, placing currently scarce references first and then using target proximity. A global lineage cap starts at {LINEAGE_CAP_INITIAL} and is relaxed only if required; maximum used was {meta['lineage_cap_max_used']}.
- Frontier quota is fixed at 1,000 for each of `d198`, `eil51`, and `ts225`. It draws only non-region C04 candidates, from the 5,000 closest per target, round-robin across eight directions in standardized feature space.
- No real TSPLIB instance is ever used as a mutation parent.

E0/E1 share {meta['parent_overlap_n']} parents; shared parent/operator pairs therefore receive identical frozen child seeds. E1 membership hash is `{meta['E1_membership_sha256']}`.

All original A0 artifacts were read-only. New artifacts are isolated below this directory.
"""
    (OUT / "01_parent_selection_protocol.md").write_text(protocol, encoding="utf-8")

    e0, e1 = summaries["E0"], summaries["E1"]
    special = references[references.instance_id.isin(MISSING_TARGETS)].pivot(
        index="instance_id", columns="arm", values="cumulative_nearest_distance"
    )
    special_base = references[
        references.arm.eq("E0") & references.instance_id.isin(MISSING_TARGETS)
    ].set_index("instance_id").baseline_nearest_distance
    special_lines = "\n".join(
        f"- `{name}`: baseline {special_base[name]:.6f}, E0 {special.loc[name, 'E0']:.6f}, E1 {special.loc[name, 'E1']:.6f}."
        for name in MISSING_TARGETS
    )
    top = operators.sort_values(["arm", "new_strict_target_region"], ascending=[True, False]).groupby("arm").head(4)
    top_lines = "\n".join(
        f"- {row.arm} `{row.operator_id}`: {row.new_strict_target_region} target-region new unique; {row.strict_region_entries} entries from outside parents."
        for row in top.itertuples(index=False)
    )
    entry_top = operators.sort_values(["arm", "strict_region_entries"], ascending=[True, False]).groupby("arm").head(4)
    entry_lines = "\n".join(
        f"- {row.arm} `{row.operator_id}`: {row.strict_region_entries} strict-region entries from parents outside the region."
        for row in entry_top.itertuples(index=False)
    )
    comparison = f"""# P2.0 Target Reachability Comparison

## Equal-budget outcome

| Metric | E0 random | E1 structure-guided |
|---|---:|---:|
| Mutation attempts | {e0['mutation_attempts']:,} | {e1['mutation_attempts']:,} |
| New unique offspring | {e0['new_unique_offspring_n']:,} | {e1['new_unique_offspring_n']:,} |
| New strict-region unique | {e0['new_strict_target_region_unique_n']:,} | {e1['new_strict_target_region_unique_n']:,} |
| New strict-region fraction | {e0['new_target_region_fraction']:.3%} | {e1['new_target_region_fraction']:.3%} |
| Cumulative strict-region unique | {e0['cumulative_strict_target_region_unique_n']:,} | {e1['cumulative_strict_target_region_unique_n']:,} |
| Cumulative neighborhood coverage | {e0['cumulative_target_neighborhood_coverage_n']}/48 | {e1['cumulative_target_neighborhood_coverage_n']}/48 |
| Cumulative JS divergence | {e0['cumulative_js_divergence_vs_target']:.4f} | {e1['cumulative_js_divergence_vs_target']:.4f} |
| Cumulative largest cell share | {e0['cumulative_largest_cell_share']:.3%} | {e1['cumulative_largest_cell_share']:.3%} |
| Cumulative effective cells | {e0['cumulative_effective_cells']:.3f} | {e1['cumulative_effective_cells']:.3f} |
| Child proximity-improved fraction | {e0['offspring_proximity_improved_fraction']:.3%} | {e1['offspring_proximity_improved_fraction']:.3%} |
| Valid mutation rate | {e0['valid_rate']:.3%} | {e1['valid_rate']:.3%} |

Both cumulative and new-offspring-only metrics are reported; the comparison does not treat crossing 10,000 cumulative target-region candidates as sufficient by itself.

## Previously uncovered references

{special_lines}

## Leading operator contributions

By total new offspring retained in the strict region:

{top_lines}

By genuine entry from a parent outside the strict region:

{entry_lines}

## Auxiliary diagnostic

The unselected FFT Energy / NN-CV diagnostic has coarse target-box JS divergence {e0['aux_js_divergence_vs_target']:.4f} for E0 and {e1['aux_js_divergence_vs_target']:.4f} for E1. This diagnostic is not used for parent selection. FFT uses raw-count 32x32 occupancy and is node-count sensitive because all generated candidates have 100 nodes while TSPLIB48 spans multiple sizes; it is therefore interpreted only as a warning against claiming full structural equivalence.
"""
    (OUT / "04_target_reachability_comparison.md").write_text(comparison, encoding="utf-8")

    quantity = e1["new_strict_target_region_unique_n"] > e0["new_strict_target_region_unique_n"]
    reach = (
        e1["cumulative_target_neighborhood_coverage_n"] > e0["cumulative_target_neighborhood_coverage_n"]
        or e1["target_nearest_distance_mean_cumulative"] < e0["target_nearest_distance_mean_cumulative"]
    )
    distribution = (
        e1["cumulative_js_divergence_vs_target"] < e0["cumulative_js_divergence_vs_target"]
        and e1["cumulative_largest_cell_share"] <= e0["cumulative_largest_cell_share"] + 0.02
        and e1["cumulative_effective_cells"] >= 0.90 * e0["cumulative_effective_cells"]
    )
    validity = (
        e1["valid_rate"] >= e0["valid_rate"] - 0.01
        and e1["new_unique_rate_per_attempt"] >= e0["new_unique_rate_per_attempt"] - 0.02
        and e1["selected_feature_rounded_unique_fraction"] >= 0.95
    )
    if quantity and reach and distribution and validity:
        decision = "GO"
    elif quantity:
        decision = "CONDITIONAL GO"
    else:
        decision = "NO-GO"
    enough = e1["cumulative_strict_target_region_unique_n"] >= 10_000
    recommendation = (
        "Proceed to a separately frozen P2.1 second/third structure-evolution audit; do not infer that a final acquisition rule is already validated."
        if decision == "GO"
        else "Do not expand the evolution budget until the failed or conditional criterion is addressed."
    )
    decision_text = f"""# P2.0 Go / No-Go Decision

## Decision: {decision}

- E1 produced more strict target-region unique offspring than E0: **{quantity}** ({e1['new_strict_target_region_unique_n']:,} vs {e0['new_strict_target_region_unique_n']:,}).
- Neighborhood coverage / nearest-distance improvement criterion: **{reach}**.
- Coarse distribution improved without the frozen collapse guard being triggered: **{distribution}**.
- Validity and selected-feature diversity guard: **{validity}**.
- E1 cumulative pool contains at least 10,000 strict-region unique candidates: **{enough}** ({e1['cumulative_strict_target_region_unique_n']:,}). This capacity check is not used alone to determine success.

Recommendation: {recommendation}

The decision applies only to whether parent selection has reachability value. No Structure10k was acquired, no P2.1 evolution was run, and no Oracle, Concorde, surrogate retraining, or BQ training was started.

`STRUCTURE_GUIDED_PARENT_EVOLUTION_P2_0_COMPLETE`
"""
    (OUT / "06_go_nogo_decision.md").write_text(decision_text, encoding="utf-8")
    return decision


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    geometry = load_geometry()
    parents = prepare_parents(geometry)
    for arm in ("E0", "E1"):
        run_mutation(arm)

    diagnostics: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict[str, Any]] = {}
    references: list[pd.DataFrame] = []
    operators: list[pd.DataFrame] = []
    counts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for arm in ("E0", "E1"):
        coordinates, hashes, metadata = prepare_new_offspring(arm, geometry)
        features = extract_features(arm, coordinates, hashes)
        child = add_child_metrics(arm, coordinates, hashes, metadata, features, geometry)
        summary, reference, operator, new_counts, cumulative_counts = evaluate_arm(
            arm, child, geometry, parents[arm]
        )
        diagnostics[arm] = child
        summaries[arm] = summary
        references.append(reference)
        operators.append(operator)
        counts[arm] = (new_counts, cumulative_counts)
        result_rows(summary, reference, new_counts, cumulative_counts, geometry["target_counts"]).to_csv(
            OUT / ("02_e0_random_parent_results.csv" if arm == "E0" else "03_e1_structure_parent_results.csv"),
            index=False,
        )

    reference_frame = pd.concat(references, ignore_index=True)
    operator_frame = pd.concat(operators, ignore_index=True)
    operator_frame.to_csv(OUT / "05_operator_contribution.csv", index=False)
    reference_frame.to_csv(OUT / "support_reference_reachability.csv", index=False)
    make_figures(geometry, diagnostics, reference_frame, summaries, counts)
    decision = write_reports(geometry, parents, summaries, reference_frame, operator_frame)
    anchor_time = min(
        (OUT / "parent_selection_metadata.json").stat().st_mtime,
        (OUT / "evolution" / "E0" / "mutation" / "FOGA11_BASE_V1_RUNTIME.json").stat().st_mtime,
    )
    write_json(
        OUT / "run_metadata.json",
        {
            "status": "COMPLETE",
            "decision": decision,
            "finalization_invocation_seconds": time.time() - started,
            "artifact_wall_clock_span_seconds": time.time() - anchor_time,
            "baseline_union_sha256": sha256_file(BASE_UNION),
            "baseline_bundle_sha256": sha256_file(BASE_BUNDLE),
            "mutation_worker_sha256": sha256_file(MUTATION_WORKER),
            "mutation_profile_sha256": sha256_file(PROFILE),
            "p1_script_sha256": sha256_file(P1 / "run_p0_p1_audit.py"),
            "runner_sha256": sha256_file(Path(__file__)),
            "no_oracle": True,
            "no_concorde": True,
            "no_surrogate_fit": True,
            "no_bq_training": True,
        },
    )
    print(json.dumps({"status": "COMPLETE", "decision": decision, "summaries": summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()
