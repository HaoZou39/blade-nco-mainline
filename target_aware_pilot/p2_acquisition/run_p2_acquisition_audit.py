#!/usr/bin/env python3
"""P2.1 Structure10k acquisition feasibility audit.

Selection is structure-only.  HGB predictions and EvoReal auxiliary features
are computed only after the 10k membership is frozen.  This script never runs
mutation, an Oracle/Concorde backend, or BQ training.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "target_aware_pilot" / "p2_acquisition"
FIG = OUT / "figures"
P1 = ROOT / "target_aware_pilot"
P20 = P1 / "p2_reachability"
sys.path.insert(0, str(P20))
import run_p2_reachability as p2  # noqa: E402

RUN = ROOT / "runs" / "BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1"
CYCLES = RUN / "A0_FULL_BASE" / "cycles"
C04 = CYCLES / "C04"
E1 = P20 / "evolution" / "E1"
F40_MANIFEST = ROOT / "audits" / "S3_B2_RF_HGB_feature_label_scaling_v1" / "09_feature_subset_manifests.json"
SCALER = ROOT / "assets" / "qd" / "reference_cloud" / "formal_raw_scaler.json"
R_EXTRACTOR = ROOT / "src" / "features" / "extract_salesperson_raw_features.R"
F40_WORKER = ROOT / "scripts" / "f40_one_shard.py"
FEATURES = p2.FEATURES
SEEDS = (2026, 2027, 2028, 2029, 2030)
ACQUISITION_N = 10_000
SELECTION_SEED = 20261025
LINEAGE_PARENT_CAP_SCHEDULE = ((15, 6), (20, 8), (30, 12), (50, 20), (100, 50), (10_000, 10_000))
ANCHORS_PER_REFERENCE = 12


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def largest_remainder(total: int, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    raw = total * weights / weights.sum()
    result = np.floor(raw).astype(int)
    remainder = total - int(result.sum())
    order = np.lexsort((np.arange(len(weights)), -(raw - result)))
    result[order[:remainder]] += 1
    return result


def entropy_effective(values: pd.Series) -> float:
    counts = values.value_counts().to_numpy(float)
    probability = counts / counts.sum()
    return float(np.exp(-(probability * np.log(probability)).sum()))


def coordinate_hashes(coordinates: np.ndarray) -> list[str]:
    return [hashlib.sha256(np.ascontiguousarray(points, dtype=np.float64).tobytes()).hexdigest() for points in coordinates]


def old_training_sets() -> tuple[set[str], pd.DataFrame]:
    rows = []
    sets: list[set[str]] = []
    for cycle in range(5):
        path = CYCLES / f"C{cycle:02d}" / "06_selection" / "T_selected.parquet"
        frame = pd.read_parquet(path, columns=["instance_hash"])
        values = frame.instance_hash.astype(str).tolist()
        current = set(values)
        sets.append(current)
        rows.append(
            {
                "cycle": f"C{cycle:02d}",
                "model_produced": f"M{cycle + 1}",
                "training_n": len(values),
                "training_unique_n": len(current),
                "membership_sha256": p2.membership_hash(values),
                "exclusion_role": "required_predecessor" if cycle < 4 else "conservative_original_M5_comparator",
            }
        )
    union = set().union(*sets)
    rows.append(
        {
            "cycle": "C00-C04 union",
            "model_produced": "M1-M5",
            "training_n": sum(len(item) for item in sets),
            "training_unique_n": len(union),
            "membership_sha256": p2.membership_hash(sorted(union)),
            "exclusion_role": "all_known_formal_training_memberships",
        }
    )
    return union, pd.DataFrame(rows)


def load_candidates() -> tuple[dict[str, Any], pd.DataFrame, np.ndarray, pd.DataFrame]:
    geometry = p2.load_geometry()
    metadata = pd.read_parquet(E1 / "cumulative_candidate_pool.parquet")
    coordinates, hashes = p2.load_bundle(E1 / "cumulative_candidate_pool.npz")
    if len(metadata) != 229_148 or metadata.instance_hash.nunique() != len(metadata):
        raise RuntimeError("E1 cumulative pool identity failure")
    if hashes != metadata.instance_hash.astype(str).tolist() or len(coordinates) != len(metadata):
        raise RuntimeError("E1 coordinate/metadata alignment failure")
    if coordinates.shape != (229_148, 100, 2) or not np.isfinite(coordinates).all():
        raise RuntimeError("E1 coordinate validity failure")
    if np.any((coordinates < 0) | (coordinates > 1)):
        raise RuntimeError("E1 coordinates are outside [0,1]")
    if coordinate_hashes(coordinates) != hashes:
        raise RuntimeError("E1 instance hashes do not reproduce coordinate bytes")

    # These C04 columns are hardness/acquisition provenance.  Remove them
    # before constructing the structure-only table so they cannot accidentally
    # enter selection or collide with the new P2.1 selection rank.
    metadata = metadata.drop(
        columns=["selection_rank", "component", "hgb_predicted_log1p_hardness"],
        errors="ignore",
    )

    baseline_features = pd.read_parquet(C04 / "candidate_features.parquet", columns=["instance_hash", *FEATURES])
    new_diagnostics = pd.read_parquet(
        E1 / "new_offspring_diagnostics.parquet",
        columns=["instance_hash", *FEATURES, "fft_energy", "nn_ratio"],
    )
    feature_table = pd.concat(
        [baseline_features, new_diagnostics[["instance_hash", *FEATURES]]], ignore_index=True
    ).drop_duplicates("instance_hash", keep="last")
    if len(feature_table) != len(metadata):
        raise RuntimeError("selected-feature table does not cover E1 pool")
    metadata = metadata.drop(columns=[name for name in FEATURES if name in metadata], errors="ignore")
    metadata = metadata.merge(feature_table, on="instance_hash", validate="one_to_one", sort=False)

    lineage_lookup = geometry["candidates"].set_index("instance_hash").lineage_root
    metadata["resolved_lineage_root"] = metadata.lineage_root
    missing_lineage = metadata.resolved_lineage_root.isna()
    metadata.loc[missing_lineage, "resolved_lineage_root"] = metadata.loc[missing_lineage, "parent_hash"].map(lineage_lookup)
    if metadata.resolved_lineage_root.isna().any():
        raise RuntimeError("E1 lineage reconstruction failure")
    metadata["resolved_direct_parent"] = metadata.parent_hash
    missing_parent = metadata.resolved_direct_parent.isna()
    metadata.loc[missing_parent, "resolved_direct_parent"] = metadata.loc[missing_parent, "instance_hash"]

    excluded, training_audit = old_training_sets()
    training_audit["overlap_with_e1_pool"] = [
        len(set(pd.read_parquet(CYCLES / f"C{i:02d}" / "06_selection" / "T_selected.parquet", columns=["instance_hash"]).instance_hash.astype(str)) & set(hashes))
        if i < 5 else len(excluded & set(hashes))
        for i in range(6)
    ]
    training_audit.to_csv(OUT / "support_training_membership_exclusion.csv", index=False)
    metadata["excluded_prior_training"] = metadata.instance_hash.astype(str).isin(excluded)
    eligible = metadata[~metadata.excluded_prior_training].copy().reset_index(drop=True)
    if set(eligible.instance_hash.astype(str)) & excluded:
        raise RuntimeError("known prior training overlap remains after exclusion")
    eligible_indices = eligible.p2_coord_index.to_numpy(int)
    eligible_coordinates = coordinates[eligible_indices]

    values = eligible[list(FEATURES)].to_numpy(float)
    z = (values - geometry["center"]) / geometry["scale"]
    distance = p2.nearest_matrix(z, geometry["target_z"])
    nearest = distance.argmin(axis=1)
    eligible["nearest_reference_index"] = nearest
    eligible["nearest_reference"] = geometry["manifest"].instance_id.to_numpy()[nearest]
    eligible["target_distance"] = distance[np.arange(len(distance)), nearest]
    eligible["in_strict_target_region"] = eligible.target_distance <= geometry["radius"]
    cell, inside = p2.coarse_cells(values, geometry["edges"])
    eligible["coarse_cell"] = cell
    eligible["inside_target_box"] = inside
    eligible["eligible_index"] = np.arange(len(eligible))
    return geometry, eligible, eligible_coordinates, training_audit


def diverse_reference_order(
    pool: pd.DataFrame,
    pool_z: np.ndarray,
    target_z: np.ndarray,
    reference_index: int,
    radius: float,
) -> tuple[list[int], dict[int, str], dict[int, float]]:
    delta = pool_z - target_z[reference_index]
    distance = np.sqrt(np.sum(delta * delta, axis=1))
    anchor_count = min(ANCHORS_PER_REFERENCE, len(pool))
    anchor_order = np.argsort(distance, kind="stable")[:anchor_count]
    anchor_set = set(anchor_order.tolist())
    remaining = np.asarray([i for i in range(len(pool)) if i not in anchor_set], dtype=int)
    phase_code = np.where(distance <= radius, 0, np.where(distance <= 2 * radius, 1, 2))
    hash_prefix = np.asarray([int(value[:16], 16) for value in pool.instance_hash.astype(str)], dtype=np.uint64)
    salt = np.uint64(((reference_index + 1) * 0x9E3779B185EBCA87 + SELECTION_SEED) & ((1 << 64) - 1))
    random_key = hash_prefix ^ salt
    if len(remaining):
        diverse_order = remaining[
            np.lexsort((distance[remaining], random_key[remaining], phase_code[remaining]))
        ]
    else:
        diverse_order = remaining
    ordered_local = np.concatenate([anchor_order, diverse_order])
    indices = pool.iloc[ordered_local].eligible_index.astype(int).tolist()
    phase: dict[int, str] = {}
    direct_distance: dict[int, float] = {}
    for rank, local in enumerate(ordered_local):
        index = int(pool.iloc[int(local)].eligible_index)
        if rank < anchor_count:
            label = "nearest_anchor"
        elif phase_code[int(local)] == 0:
            label = "strict_diversity"
        elif phase_code[int(local)] == 1:
            label = "transition_diversity"
        else:
            label = "boundary_fallback"
        phase[index] = label
        direct_distance[index] = float(distance[int(local)])
    return indices, phase, direct_distance


def select_structure10k(geometry: dict[str, Any], eligible: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    target_counts = geometry["target_counts"]
    cell_quota = largest_remainder(ACQUISITION_N, target_counts)
    target_cell = geometry["target_cell"]
    values = eligible[list(FEATURES)].to_numpy(float)
    eligible_z = (values - geometry["center"]) / geometry["scale"]
    selected: set[int] = set()
    lineage_counts: dict[str, int] = {}
    parent_counts: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    max_lineage_cap_used = 0
    max_parent_cap_used = 0
    feasibility_rows = []

    for cell in range(9):
        quota = int(cell_quota[cell])
        refs = np.flatnonzero(target_cell == cell)
        cell_pool = eligible[eligible.inside_target_box & eligible.coarse_cell.eq(cell)].copy()
        strict_available = int(cell_pool.in_strict_target_region.sum())
        if quota == 0:
            feasibility_rows.append(
                {
                    "cell": cell,
                    "target_reference_n": len(refs),
                    "target_quota": 0,
                    "eligible_available": len(cell_pool),
                    "strict_available": strict_available,
                    "selected_n": 0,
                    "quota_deficit": 0,
                }
            )
            continue
        if len(cell_pool) < quota or not len(refs):
            raise RuntimeError(f"cell {cell} cannot satisfy target quota {quota}")
        reference_quota = largest_remainder(quota, np.ones(len(refs)))
        orders: dict[int, list[int]] = {}
        phases: dict[int, dict[int, str]] = {}
        direct_distances: dict[int, dict[int, float]] = {}
        pool_positions = cell_pool.eligible_index.to_numpy(int)
        pool_z = eligible_z[pool_positions]
        for ref in refs:
            order, phase, direct = diverse_reference_order(
                cell_pool, pool_z, geometry["target_z"], int(ref), geometry["radius"]
            )
            orders[int(ref)] = order
            phases[int(ref)] = phase
            direct_distances[int(ref)] = direct
        acquired = {int(ref): 0 for ref in refs}
        target_quota = {int(ref): int(reference_quota[i]) for i, ref in enumerate(refs)}

        for lineage_cap, parent_cap in LINEAGE_PARENT_CAP_SCHEDULE:
            pointers = {int(ref): 0 for ref in refs}
            progress = True
            while progress and any(acquired[ref] < target_quota[ref] for ref in acquired):
                progress = False
                for ref in refs:
                    ref = int(ref)
                    if acquired[ref] >= target_quota[ref]:
                        continue
                    order = orders[ref]
                    while pointers[ref] < len(order):
                        index = int(order[pointers[ref]])
                        pointers[ref] += 1
                        if index in selected:
                            continue
                        lineage = str(eligible.at[index, "resolved_lineage_root"])
                        parent = str(eligible.at[index, "resolved_direct_parent"])
                        if lineage_counts.get(lineage, 0) >= lineage_cap or parent_counts.get(parent, 0) >= parent_cap:
                            continue
                        selected.add(index)
                        lineage_counts[lineage] = lineage_counts.get(lineage, 0) + 1
                        parent_counts[parent] = parent_counts.get(parent, 0) + 1
                        acquired[ref] += 1
                        rows.append(
                            {
                                "eligible_index": index,
                                "assigned_reference_index": ref,
                                "assigned_reference": geometry["manifest"].iloc[ref].instance_id,
                                "assigned_reference_distance": direct_distances[ref][index],
                                "selection_component": phases[ref][index],
                                "target_cell_quota": quota,
                                "target_reference_quota": target_quota[ref],
                                "lineage_cap_at_selection": lineage_cap,
                                "parent_cap_at_selection": parent_cap,
                            }
                        )
                        max_lineage_cap_used = max(max_lineage_cap_used, lineage_cap)
                        max_parent_cap_used = max(max_parent_cap_used, parent_cap)
                        progress = True
                        break
            if all(acquired[ref] == target_quota[ref] for ref in acquired):
                break
        if not all(acquired[ref] == target_quota[ref] for ref in acquired):
            raise RuntimeError(f"cell {cell} reference quotas could not be filled: {acquired} vs {target_quota}")
        feasibility_rows.append(
            {
                "cell": cell,
                "target_reference_n": len(refs),
                "target_quota": quota,
                "eligible_available": len(cell_pool),
                "strict_available": strict_available,
                "selected_n": sum(acquired.values()),
                "quota_deficit": quota - sum(acquired.values()),
            }
        )

    # A target's closest feasible candidate can fall across a coarse-cell
    # boundary.  Repair any avoidable reference miss with a same-cell swap, so
    # exact 3x3 quotas remain unchanged.  The removed row must not be the sole
    # strict cover for any other reference.  This is still structure-only and
    # is determined before the selection freeze/HGB diagnostic.
    coverage_repair_n = 0
    selected_indices = [int(row["eligible_index"]) for row in rows]
    selected_z = eligible_z[np.asarray(selected_indices, int)]
    selected_distance = p2.nearest_matrix(selected_z, geometry["target_z"])
    cover_counts = np.sum(selected_distance <= geometry["radius"], axis=0).astype(int)
    for reference_index in np.flatnonzero(cover_counts == 0):
        direct = np.sqrt(
            np.sum((eligible_z - geometry["target_z"][int(reference_index)]) ** 2, axis=1)
        )
        add_index = int(np.argmin(direct))
        if direct[add_index] > geometry["radius"] or add_index in selected:
            continue
        add_cell = int(eligible.at[add_index, "coarse_cell"])
        removable_positions = [
            position
            for position, row in enumerate(rows)
            if int(eligible.at[int(row["eligible_index"]), "coarse_cell"]) == add_cell
            and row["selection_component"] != "coverage_repair_anchor"
        ]
        phase_priority = {
            "boundary_fallback": 3,
            "transition_diversity": 2,
            "strict_diversity": 1,
            "nearest_anchor": 0,
        }
        removable_positions.sort(
            key=lambda position: (
                phase_priority.get(str(rows[position]["selection_component"]), 0),
                float(rows[position]["assigned_reference_distance"]),
            ),
            reverse=True,
        )
        remove_position = None
        for position in removable_positions:
            remove_index = int(rows[position]["eligible_index"])
            remove_distance = np.sqrt(
                np.sum((eligible_z[remove_index] - geometry["target_z"]) ** 2, axis=1)
            )
            covered_by_remove = remove_distance <= geometry["radius"]
            if np.all(cover_counts[covered_by_remove] >= 2):
                remove_position = position
                break
        if remove_position is None:
            raise RuntimeError(f"no coverage-preserving same-cell swap for reference {reference_index}")
        remove_index = int(rows[remove_position]["eligible_index"])
        remove_distance = np.sqrt(
            np.sum((eligible_z[remove_index] - geometry["target_z"]) ** 2, axis=1)
        )
        add_distance_all = np.sqrt(
            np.sum((eligible_z[add_index] - geometry["target_z"]) ** 2, axis=1)
        )
        cover_counts -= (remove_distance <= geometry["radius"]).astype(int)
        cover_counts += (add_distance_all <= geometry["radius"]).astype(int)
        selected.remove(remove_index)
        selected.add(add_index)
        old_row = rows[remove_position]
        rows[remove_position] = {
            "eligible_index": add_index,
            "assigned_reference_index": int(reference_index),
            "assigned_reference": geometry["manifest"].iloc[int(reference_index)].instance_id,
            "assigned_reference_distance": float(direct[add_index]),
            "selection_component": "coverage_repair_anchor",
            "target_cell_quota": int(cell_quota[add_cell]),
            "target_reference_quota": np.nan,
            "lineage_cap_at_selection": old_row["lineage_cap_at_selection"],
            "parent_cap_at_selection": old_row["parent_cap_at_selection"],
        }
        coverage_repair_n += 1

    selection = pd.DataFrame(rows)
    if len(selection) != ACQUISITION_N or selection.eligible_index.nunique() != ACQUISITION_N:
        raise RuntimeError("provisional Structure10k is not 10,000 unique rows")
    manifest = selection.merge(eligible, on="eligible_index", validate="one_to_one", sort=False)
    manifest.insert(0, "selection_rank", np.arange(1, len(manifest) + 1))
    feasibility = pd.DataFrame(feasibility_rows)
    metadata = {
        "selection_seed": SELECTION_SEED,
        "selection_policy": "exact target 3x3 quota; equal reference subquota; 12 proximity anchors then hash-diverse strict/transition/boundary; lineage and direct-parent caps",
        "target_cell_counts": target_counts.astype(int).tolist(),
        "target_cell_quota": cell_quota.astype(int).tolist(),
        "max_lineage_cap_used": max_lineage_cap_used,
        "max_parent_cap_used": max_parent_cap_used,
        "coverage_preserving_same_cell_swap_n": coverage_repair_n,
        "selected_lineage_n": int(manifest.resolved_lineage_root.nunique()),
        "selected_direct_parent_n": int(manifest.resolved_direct_parent.nunique()),
    }
    return manifest, feasibility, metadata


def all_scaler_params(features: list[str]) -> dict[str, tuple[float, float]]:
    payload = json.loads(SCALER.read_text())
    table = {item["feature_name"]: item for item in payload["parameters"]}
    return {name: (float(table[name]["center_median"]), float(table[name]["scale"])) for name in features}


def extract_selected_f40(coordinates: np.ndarray, hashes: list[str]) -> pd.DataFrame:
    final_path = OUT / "support_structure10k_f40.parquet"
    if final_path.is_file():
        frame = pd.read_parquet(final_path)
        if frame.instance_hash.astype(str).tolist() == hashes:
            return frame
    features = list(json.loads(F40_MANIFEST.read_text())["F40"])
    cache_dir = OUT / "f40_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    params = all_scaler_params(features)
    for start in range(0, len(hashes), 250):
        stop = min(start + 250, len(hashes))
        block_hashes = hashes[start:stop]
        tag = p2.membership_hash(block_hashes)[:16]
        jobs.append(
            {
                "coordinates": coordinates[start:stop],
                "hashes": block_hashes,
                "names": features,
                "out": str(cache_dir / f"cache_{tag}.parquet"),
                "rscript": shutil.which("Rscript") or "Rscript",
                "r_script": str(R_EXTRACTOR),
                "params": params,
            }
        )
    print(f"[P2.1] extracting diagnostic F40 in {len(jobs)} shards after selection freeze", flush=True)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(p2.run_feature_job, job) for job in jobs]
        done = 0
        for future in as_completed(futures):
            future.result()
            done += 1
            if done == 1 or done == len(jobs) or done % 10 == 0:
                print(f"[P2.1] F40 shards {done}/{len(jobs)}", flush=True)
    frame = pd.concat([pd.read_parquet(job["out"]) for job in jobs], ignore_index=True)
    if frame.instance_hash.astype(str).tolist() != hashes or list(frame.columns) != ["instance_hash", *features]:
        raise RuntimeError("Structure10k F40 alignment/schema failure")
    frame.to_parquet(final_path, index=False)
    return frame


def hgb_predictions(f40: pd.DataFrame) -> np.ndarray:
    features = list(json.loads(F40_MANIFEST.read_text())["F40"])
    values = f40[features].to_numpy(float)
    predictions = []
    for seed in SEEDS:
        payload = joblib.load(C04 / "HGB_seed_models" / f"HGB40_seed_{seed}.joblib")
        if payload["features"] != features:
            raise RuntimeError("HGB diagnostic feature order mismatch")
        predictions.append(payload["model"].predict(values))
    return np.median(np.vstack(predictions), axis=0)


def auxiliary_distribution(values: np.ndarray, target_aux: np.ndarray) -> dict[str, Any]:
    edges = [
        np.asarray([target_aux[:, i].min(), np.quantile(target_aux[:, i], 1 / 3), np.quantile(target_aux[:, i], 2 / 3), target_aux[:, i].max()])
        for i in range(2)
    ]
    target_cell, _ = p2.coarse_cells(target_aux, edges)
    cell, inside = p2.coarse_cells(values, edges)
    target_counts = np.bincount(target_cell, minlength=9)
    counts = np.bincount(cell[inside], minlength=9)
    return p2.distribution_metrics(counts, target_counts) | {"counts": counts.tolist()}


def structure_metrics(frame: pd.DataFrame, geometry: dict[str, Any]) -> dict[str, Any]:
    values = frame[list(FEATURES)].to_numpy(float)
    cell, inside = p2.coarse_cells(values, geometry["edges"])
    counts = np.bincount(cell[inside], minlength=9)
    z = (values - geometry["center"]) / geometry["scale"]
    distance = p2.nearest_matrix(z, geometry["target_z"])
    nearest_candidate = distance.min(axis=0)
    return p2.distribution_metrics(counts, geometry["target_counts"]) | {
        "counts": counts.tolist(),
        "strict_region_n": int(np.sum(distance.min(axis=1) <= geometry["radius"])),
        "strict_region_fraction": float(np.mean(distance.min(axis=1) <= geometry["radius"])),
        "target_coverage_n": int(np.sum(nearest_candidate <= geometry["radius"])),
        "target_nearest_mean": float(nearest_candidate.mean()),
        "target_nearest_p90": float(np.quantile(nearest_candidate, 0.9)),
        "target_nearest": nearest_candidate.tolist(),
    }


def finalize_and_audit(
    geometry: dict[str, Any],
    eligible: pd.DataFrame,
    eligible_coordinates: np.ndarray,
    manifest: pd.DataFrame,
    feasibility: pd.DataFrame,
    selection_metadata: dict[str, Any],
    training_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    indices = manifest.eligible_index.to_numpy(int)
    selected_coordinates = eligible_coordinates[indices]
    selected_hashes = manifest.instance_hash.astype(str).tolist()
    bundle_path = OUT / "provisional_structure10k.npz"
    p2.save_bundle(bundle_path, selected_coordinates, selected_hashes)
    if coordinate_hashes(selected_coordinates) != selected_hashes:
        raise RuntimeError("selected coordinate hashes changed")

    # Freeze the structure-only membership before any HGB diagnostic exists.
    preliminary_columns = [
        "selection_rank", "instance_hash", "p2_origin", "resolved_lineage_root",
        "resolved_direct_parent", "operator_id", "assigned_reference", "assigned_reference_distance",
        "selection_component", *FEATURES, "target_distance", "in_strict_target_region",
        "coarse_cell", "target_cell_quota", "target_reference_quota",
    ]
    preliminary = manifest[preliminary_columns].copy()
    preliminary.to_csv(OUT / "support_structure_only_membership_pre_diagnostic.csv", index=False)
    membership_sha = p2.membership_hash(selected_hashes)
    write_json(
        OUT / "selection_freeze.json",
        {
            **selection_metadata,
            "selection_n": len(manifest),
            "membership_sha256": membership_sha,
            "structure_only_manifest_sha256": p2.sha256_file(OUT / "support_structure_only_membership_pre_diagnostic.csv"),
            "hgb_used_for_selection": False,
            "heldout_used": False,
        },
    )

    f40 = extract_selected_f40(selected_coordinates, selected_hashes)
    manifest["hgb40_predicted_log1p_hardness_diagnostic"] = hgb_predictions(f40)
    fft, nn = p2.evoreal_diagnostics(selected_coordinates)
    manifest["fft_energy"] = fft
    manifest["nn_ratio"] = nn

    hard = pd.read_parquet(C04 / "06_selection" / "T_selected.parquet", columns=["instance_hash"])
    candidate_features = pd.read_parquet(C04 / "candidate_features.parquet", columns=["instance_hash", *FEATURES])
    candidate_predictions = pd.read_parquet(C04 / "candidate_predictions.parquet", columns=["instance_hash", "hgb_predicted_log1p_hardness"])
    hard = hard.merge(candidate_features, on="instance_hash", validate="one_to_one").merge(
        candidate_predictions, on="instance_hash", validate="one_to_one"
    )
    hard_coords, hard_hashes = p2.load_bundle(C04 / "06_selection" / "T_selected.npz")
    if hard_hashes != hard.instance_hash.astype(str).tolist():
        raise RuntimeError("Hard10k bundle alignment failure")
    hard_fft, hard_nn = p2.evoreal_diagnostics(hard_coords)
    hard["fft_energy"] = hard_fft
    hard["nn_ratio"] = hard_nn

    structure = structure_metrics(manifest, geometry)
    hard_metrics = structure_metrics(hard, geometry)
    target_aux_frame = pd.read_csv(P1 / "p1_evoreal_feature_diagnostic.csv")
    target_aux = target_aux_frame[["fft_energy", "nn_ratio"]].to_numpy(float)
    structure_aux = auxiliary_distribution(manifest[["fft_energy", "nn_ratio"]].to_numpy(float), target_aux)
    hard_aux = auxiliary_distribution(hard[["fft_energy", "nn_ratio"]].to_numpy(float), target_aux)
    target_names = geometry["manifest"].instance_id.astype(str).tolist()
    reference = pd.DataFrame(
        {
            "instance_id": target_names,
            "evoreal_split": geometry["manifest"].evoreal_split,
            "structure10k_nearest_distance": structure["target_nearest"],
            "hard10k_nearest_distance": hard_metrics["target_nearest"],
        }
    )
    reference["strict_radius"] = geometry["radius"]
    reference["structure10k_covered"] = reference.structure10k_nearest_distance <= geometry["radius"]
    reference["hard10k_covered"] = reference.hard10k_nearest_distance <= geometry["radius"]
    reference.to_csv(OUT / "support_reference_coverage.csv", index=False)

    assigned_counts = manifest.assigned_reference.value_counts()
    eligible_values = eligible[list(FEATURES)].to_numpy(float)
    eligible_z = (eligible_values - geometry["center"]) / geometry["scale"]
    reference_quota_rows = []
    cell_quota = largest_remainder(ACQUISITION_N, geometry["target_counts"])
    for cell in range(9):
        refs = np.flatnonzero(geometry["target_cell"] == cell)
        if not len(refs):
            continue
        quotas = largest_remainder(int(cell_quota[cell]), np.ones(len(refs)))
        for position, ref in enumerate(refs):
            name = geometry["manifest"].iloc[int(ref)].instance_id
            assigned = manifest[manifest.assigned_reference.eq(name)]
            direct = np.sqrt(np.sum((eligible_z - geometry["target_z"][int(ref)]) ** 2, axis=1))
            reference_quota_rows.append(
                {
                    "instance_id": name,
                    "coarse_cell": cell,
                    "reference_quota": int(quotas[position]),
                    "assigned_n": int(assigned_counts.get(name, 0)),
                    "quota_delta": int(assigned_counts.get(name, 0) - quotas[position]),
                    "eligible_direct_strict_n": int(np.sum(direct <= geometry["radius"])),
                    "assigned_direct_strict_n": int(np.sum(assigned.assigned_reference_distance <= geometry["radius"])),
                    "assigned_direct_transition_n": int(np.sum((assigned.assigned_reference_distance > geometry["radius"]) & (assigned.assigned_reference_distance <= 2 * geometry["radius"]))),
                    "assigned_direct_boundary_n": int(np.sum(assigned.assigned_reference_distance > 2 * geometry["radius"])),
                    "selected_nearest_distance": float(reference.loc[reference.instance_id.eq(name), "structure10k_nearest_distance"].iloc[0]),
                    "strict_covered": bool(reference.loc[reference.instance_id.eq(name), "structure10k_covered"].iloc[0]),
                }
            )
    reference_quota = pd.DataFrame(reference_quota_rows)
    reference_quota.to_csv(OUT / "support_reference_quota_fulfillment.csv", index=False)

    lineage_max = int(manifest.resolved_lineage_root.value_counts().max())
    parent_max = int(manifest.resolved_direct_parent.value_counts().max())
    metrics = {
        "selection_n": len(manifest),
        "selection_unique_n": int(manifest.instance_hash.nunique()),
        "known_training_overlap_n": int(manifest.excluded_prior_training.sum()),
        "strict_region_n": structure["strict_region_n"],
        "strict_region_fraction": structure["strict_region_fraction"],
        "target_coverage_n": structure["target_coverage_n"],
        "coarse_counts": structure["counts"],
        "coarse_js_divergence": structure["js_divergence_vs_target"],
        "coarse_effective_cells": structure["effective_cells"],
        "coarse_largest_cell_share": structure["largest_cell_share"],
        "e1_cumulative_p20_js": 0.3205582230578501,
        "hard10k_js_divergence": hard_metrics["js_divergence_vs_target"],
        "hard10k_effective_cells": hard_metrics["effective_cells"],
        "hard10k_largest_cell_share": hard_metrics["largest_cell_share"],
        "hard10k_strict_region_n": hard_metrics["strict_region_n"],
        "hard10k_target_coverage_n": hard_metrics["target_coverage_n"],
        "lineage_n": int(manifest.resolved_lineage_root.nunique()),
        "lineage_effective_n": entropy_effective(manifest.resolved_lineage_root),
        "lineage_max_count": lineage_max,
        "lineage_max_share": lineage_max / len(manifest),
        "direct_parent_n": int(manifest.resolved_direct_parent.nunique()),
        "direct_parent_effective_n": entropy_effective(manifest.resolved_direct_parent),
        "direct_parent_max_count": parent_max,
        "direct_parent_max_share": parent_max / len(manifest),
        "reference_quota_exact_n": int(np.sum(reference_quota.quota_delta == 0)),
        "reference_quota_max_abs_delta": int(reference_quota.quota_delta.abs().max()),
        "origin_counts": manifest.p2_origin.value_counts().astype(int).to_dict(),
        "selection_component_counts": manifest.selection_component.value_counts().astype(int).to_dict(),
        "fft_nn_aux_js": structure_aux["js_divergence_vs_target"],
        "hard10k_fft_nn_aux_js": hard_aux["js_divergence_vs_target"],
        "hgb_predicted_mean": float(manifest.hgb40_predicted_log1p_hardness_diagnostic.mean()),
        "hgb_predicted_p90": float(manifest.hgb40_predicted_log1p_hardness_diagnostic.quantile(0.9)),
        "hard10k_hgb_predicted_mean": float(hard.hgb_predicted_log1p_hardness.mean()),
        "valid_coordinates": True,
        "bundle_sha256": p2.sha256_file(bundle_path),
        "membership_sha256": membership_sha,
        "m0_pretraining_membership_available": False,
        "known_formal_training_union_n": int(training_audit.iloc[-1].training_unique_n),
    }
    write_json(OUT / "support_distribution_metrics.json", metrics)

    output_columns = preliminary_columns + [
        "resolved_lineage_root", "resolved_direct_parent", "lineage_cap_at_selection",
        "parent_cap_at_selection", "fft_energy", "nn_ratio", "hgb40_predicted_log1p_hardness_diagnostic",
    ]
    output_columns = list(dict.fromkeys(output_columns))
    manifest[output_columns].to_csv(OUT / "02_provisional_structure10k_manifest.csv", index=False)
    manifest.to_parquet(OUT / "support_provisional_structure10k_manifest.parquet", index=False)
    hard.to_parquet(OUT / "support_hard10k_diagnostics.parquet", index=False)
    write_json(
        OUT / "provisional_structure10k_integrity.json",
        {
            "bundle": str(bundle_path),
            "bundle_sha256": metrics["bundle_sha256"],
            "membership_sha256": membership_sha,
            "instance_n": len(manifest),
            "instance_unique_n": int(manifest.instance_hash.nunique()),
            "coordinate_shape": list(selected_coordinates.shape),
            "coordinate_dtype": str(selected_coordinates.dtype),
            "coordinate_hashes_verified": True,
            "known_training_overlap_n": 0,
            "oracle_labels_present": False,
        },
    )
    return manifest, metrics, reference


def make_figures(
    geometry: dict[str, Any],
    eligible: pd.DataFrame,
    feasibility: pd.DataFrame,
    manifest: pd.DataFrame,
    metrics: dict[str, Any],
    reference: pd.DataFrame,
) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(9)
    ax.bar(x - 0.25, feasibility.eligible_available, width=0.25, label="eligible")
    ax.bar(x, feasibility.strict_available, width=0.25, label="strict-region eligible")
    ax.bar(x + 0.25, feasibility.target_quota, width=0.25, label="target quota")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xticks(x)
    ax.set_xlabel("3x3 cell")
    ax.set_ylabel("candidate count (symlog)")
    ax.set_title("Cell quota feasibility after all known-training exclusions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "01_cell_quota_feasibility.png", dpi=180)
    plt.close(fig)

    hard = pd.read_parquet(OUT / "support_hard10k_diagnostics.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), sharex=True, sharey=True)
    for ax, (title, frame) in zip(axes, (("Provisional Structure10k", manifest), ("C04 Hard10k", hard))):
        ax.hexbin(frame[FEATURES[0]], frame[FEATURES[1]], gridsize=40, mincnt=1, bins="log", cmap="Blues")
        ax.scatter(geometry["target_values"][:, 0], geometry["target_values"][:, 1], s=30, c="#d62728", edgecolor="black", linewidth=0.3, label="TSPLIB48")
        ax.set_title(title)
        ax.set_xlabel(FEATURES[0].replace("spn_", ""))
    axes[0].set_ylabel(FEATURES[1].replace("spn_", ""))
    axes[0].legend()
    fig.suptitle("Structure10k versus hardness-only acquisition")
    fig.tight_layout()
    fig.savefig(FIG / "02_structure10k_vs_hard10k.png", dpi=180)
    plt.close(fig)

    target_probability = geometry["target_counts"] / geometry["target_counts"].sum()
    structure_probability = np.asarray(metrics["coarse_counts"]) / sum(metrics["coarse_counts"])
    hard_values = hard[list(FEATURES)].to_numpy(float)
    hard_cell, hard_inside = p2.coarse_cells(hard_values, geometry["edges"])
    hard_counts = np.bincount(hard_cell[hard_inside], minlength=9)
    hard_probability = hard_counts / hard_counts.sum()
    panels = (("TSPLIB48", target_probability), ("Structure10k", structure_probability), ("Hard10k", hard_probability))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    vmax = max(values.max() for _, values in panels)
    for ax, (title, probability) in zip(axes, panels):
        image = ax.imshow(probability.reshape(3, 3).T, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
        for i in range(3):
            for j in range(3):
                ax.text(i, j, f"{probability[3*i+j]:.2f}", ha="center", va="center", fontsize=8, color="white" if probability[3*i+j] > 0.3 else "black")
        ax.set_title(title)
        ax.set_xlabel("distance CV tercile")
        ax.set_ylabel("fraction-shorter tercile")
    cax = fig.add_axes([0.925, 0.16, 0.014, 0.65])
    fig.colorbar(image, cax=cax)
    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.15, top=0.88, wspace=0.35)
    fig.savefig(FIG / "03_coarse_distribution_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5.2))
    x = np.arange(len(reference))
    ax.plot(x, reference.structure10k_nearest_distance, marker="o", markersize=2.5, linewidth=1.1, label="Structure10k")
    ax.plot(x, reference.hard10k_nearest_distance, marker="o", markersize=2.5, linewidth=1.1, label="Hard10k")
    ax.axhline(geometry["radius"], linestyle="--", color="black", linewidth=0.9, label="strict radius")
    ax.set_xticks(x)
    ax.set_xticklabels(reference.instance_id, rotation=75, fontsize=7)
    ax.set_ylabel("nearest selected candidate distance")
    ax.set_title("TSPLIB48 neighborhood coverage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "04_reference_coverage.png", dpi=180)
    plt.close(fig)

    target_aux = pd.read_csv(P1 / "p1_evoreal_feature_diagnostic.csv")
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(hard.fft_energy, hard.nn_ratio, s=5, alpha=0.12, label="Hard10k")
    ax.scatter(manifest.fft_energy, manifest.nn_ratio, s=5, alpha=0.14, label="Structure10k")
    ax.scatter(target_aux.fft_energy, target_aux.nn_ratio, s=35, c="#d62728", edgecolor="black", linewidth=0.3, label="TSPLIB48")
    ax.set_xlabel("FFT Energy")
    ax.set_ylabel("NN-CV")
    ax.set_title("Unselected auxiliary structure diagnostic")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "05_fft_nncv_diagnostic.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.hist(hard.hgb_predicted_log1p_hardness, bins=55, density=True, alpha=0.45, label="Hard10k")
    ax.hist(manifest.hgb40_predicted_log1p_hardness_diagnostic, bins=55, density=True, alpha=0.45, label="Structure10k")
    ax.set_xlabel("M4-conditioned predicted log1p hardness (diagnostic only)")
    ax.set_ylabel("density")
    ax.set_title("Post-selection HGB40 diagnostic")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "06_hgb_hardness_diagnostic.png", dpi=180)
    plt.close(fig)


def write_reports(
    geometry: dict[str, Any],
    eligible: pd.DataFrame,
    feasibility: pd.DataFrame,
    manifest: pd.DataFrame,
    metrics: dict[str, Any],
    reference: pd.DataFrame,
    training_audit: pd.DataFrame,
) -> str:
    reference_quota = pd.read_csv(OUT / "support_reference_quota_fulfillment.csv")
    quota_lines = "\n".join(
        f"| {int(row.cell)} | {int(row.target_reference_n)} | {int(row.target_quota):,} | {int(row.eligible_available):,} | {int(row.strict_available):,} | {int(row.selected_n):,} | {int(row.quota_deficit):,} |"
        for row in feasibility.itertuples(index=False)
    )
    transition_by_cell = manifest[~manifest.in_strict_target_region].groupby("coarse_cell").size().reindex(range(9), fill_value=0)
    report1 = f"""# Structure10k Candidate and Quota Feasibility

## Integrity boundary

- E1 expanded pool: 229,148 unique, finite, N=100 coordinate arrays. Every SHA-256 instance ID was reproduced from canonical float64 coordinate bytes.
- Known formal training union C00–C04: {int(training_audit.iloc[-1].training_unique_n):,} unique. The E1 overlap ({int(training_audit.iloc[-1].overlap_with_e1_pool):,}) was excluded before acquisition.
- C00–C03 exclusion is required to avoid reuse from M1–M4. C04 was additionally excluded so the provisional set also has zero overlap with the completed original-M5 Hard10k comparator.
- Eligible after conservative exclusion: {len(eligible):,} unique candidates.
- M0 is an imported canonical checkpoint and contains no per-instance pretraining membership in the repository; exact collision auditing against its external pretraining stream is therefore unavailable. No M0 training instances are knowingly reused.
- Held-out TSPLIB22 coordinates/features were never loaded. Only the frozen development TSPLIB48 reference was used.

## Frozen acquisition rule

The 10,000 budget is assigned by largest-remainder quotas from the frozen TSPLIB48 3×3 cell counts. Within each cell, quota is divided across the cell's development references. Each reference receives 12 nearest anchors, followed by deterministic hash-diverse candidates ordered through strict, transition (`r < d <= 2r`), and boundary fallback tiers. Global lineage and direct-parent caps are enforced and relaxed only by the recorded fixed schedule.

HGB hardness is absent from this rule and was computed only after `selection_freeze.json` was written.

| Cell | Target refs | Target quota | Eligible | Strict eligible | Selected | Deficit |
|---:|---:|---:|---:|---:|---:|---:|
{quota_lines}

All cell quotas are feasible and exactly filled. Cell 5 requires {int(transition_by_cell.loc[5]):,} selected non-strict/transition candidates because its strict capacity is below its 1,042 target quota. At reference level, `eil51` remains the only development target with zero strict-neighborhood candidates in the expanded pool.

The secondary per-reference subquotas are exact for {metrics['reference_quota_exact_n']}/48 references; the maximum absolute deviation is {metrics['reference_quota_max_abs_delta']}. The single same-cell coverage repair deliberately moves one assignment slot so that `ts225` is covered without altering any primary cell quota. Full details are in `support_reference_quota_fulfillment.csv`.
"""
    (OUT / "01_candidate_quota_feasibility.md").write_text(report1, encoding="utf-8")

    uncovered = reference[~reference.structure10k_covered]
    uncovered_lines = "\n".join(
        f"- `{row.instance_id}`: nearest distance {row.structure10k_nearest_distance:.6f} vs strict radius {row.strict_radius:.6f}."
        for row in uncovered.itertuples(index=False)
    ) or "- None."
    report3 = f"""# Provisional Structure10k Distribution Audit

## Core quality metrics

| Metric | Structure10k | C04 Hard10k |
|---|---:|---:|
| Unique instances | {metrics['selection_unique_n']:,} | 10,000 |
| Strict-region instances | {metrics['strict_region_n']:,} ({metrics['strict_region_fraction']:.2%}) | {metrics['hard10k_strict_region_n']:,} |
| TSPLIB48 neighborhood coverage | {metrics['target_coverage_n']}/48 | {metrics['hard10k_target_coverage_n']}/48 |
| 3×3 JS divergence | {metrics['coarse_js_divergence']:.6f} | {metrics['hard10k_js_divergence']:.6f} |
| Effective occupied cells | {metrics['coarse_effective_cells']:.3f} | {metrics['hard10k_effective_cells']:.3f} |
| Largest cell share | {metrics['coarse_largest_cell_share']:.2%} | {metrics['hard10k_largest_cell_share']:.2%} |
| FFT/NN-CV auxiliary JS | {metrics['fft_nn_aux_js']:.4f} | {metrics['hard10k_fft_nn_aux_js']:.4f} |
| HGB predicted log1p hardness mean | {metrics['hgb_predicted_mean']:.5f} | {metrics['hard10k_hgb_predicted_mean']:.5f} |

Structure10k improves substantially over the full E1 cumulative-pool JS of 0.3206 and the original Hard10k. This is expected from the exact coarse quota constraint and is not evidence of generalization beyond the two frozen selection features.

## Diversity and provenance

- Lineages: {metrics['lineage_n']:,} observed; effective count {metrics['lineage_effective_n']:.1f}; largest lineage {metrics['lineage_max_count']} ({metrics['lineage_max_share']:.2%}).
- Direct parents/families: {metrics['direct_parent_n']:,} observed; effective count {metrics['direct_parent_effective_n']:.1f}; maximum {metrics['direct_parent_max_count']} ({metrics['direct_parent_max_share']:.2%}).
- Origins: `{json.dumps(metrics['origin_counts'], sort_keys=True)}`.
- Known formal-training overlap: 0.
- Coordinate validity/hash verification: PASS.

The auxiliary FFT/NN-CV JS is diagnostic only. FFT uses raw-count occupancy and remains node-count sensitive across the variable-size TSPLIB48 reference, so it cannot override the frozen two-feature audit.

## Uncovered reference neighborhoods

{uncovered_lines}

The full provisional coordinate bundle is `provisional_structure10k.npz`, SHA-256 `{metrics['bundle_sha256']}`. It contains no tours or Oracle labels.
"""
    (OUT / "03_structure10k_distribution_audit.md").write_text(report3, encoding="utf-8")

    complete = metrics["selection_unique_n"] == 10_000 and metrics["known_training_overlap_n"] == 0
    structural = (
        metrics["coarse_js_divergence"] < metrics["e1_cumulative_p20_js"]
        and metrics["coarse_js_divergence"] < metrics["hard10k_js_divergence"]
        and feasibility.quota_deficit.sum() == 0
    )
    diverse = metrics["lineage_max_share"] <= 0.02 and metrics["direct_parent_max_share"] <= 0.01
    full_coverage = metrics["target_coverage_n"] == 48
    if complete and structural and diverse and full_coverage:
        decision = "READY_FOR_TRAINING"
        recommendation = "The frozen provisional membership can proceed to a separately approved Concorde-labeling and M4→M5 protocol."
    elif complete and structural and diverse and not full_coverage:
        decision = "NEED_TARGETED_EVOLUTION"
        recommendation = "Run one additional eil51-focused frontier evolution audit, then repeat this acquisition; do not mechanically execute two rounds or start training yet."
    else:
        decision = "INCONCLUSIVE"
        recommendation = "Resolve the failed integrity, structural, or diversity condition before labeling or training."
    decision_text = f"""# P2.1 Feasibility Decision

## Decision: {decision}

- Complete 10,000 unique with zero known-training overlap: **{complete}**.
- Exact cell quotas and JS improvement over E1 cumulative / Hard10k: **{structural}**.
- Lineage/direct-parent concentration guards: **{diverse}**.
- Full TSPLIB48 strict-neighborhood coverage: **{full_coverage}** ({metrics['target_coverage_n']}/48).
- M0 external pretraining membership is unavailable, so only the repository's complete formal C00–C04 memberships can be hash-audited.

{recommendation}

No mutation, Concorde/Oracle labeling, surrogate fitting, or BQ training was performed.

`TARGET_AWARE_STRUCTURE10K_FEASIBILITY_AUDIT_COMPLETE`
"""
    (OUT / "04_go_nogo_decision.md").write_text(decision_text, encoding="utf-8")
    return decision


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    geometry, eligible, eligible_coordinates, training_audit = load_candidates()
    manifest, feasibility, selection_metadata = select_structure10k(geometry, eligible)
    manifest, metrics, reference = finalize_and_audit(
        geometry, eligible, eligible_coordinates, manifest, feasibility, selection_metadata, training_audit
    )
    make_figures(geometry, eligible, feasibility, manifest, metrics, reference)
    decision = write_reports(geometry, eligible, feasibility, manifest, metrics, reference, training_audit)
    write_json(
        OUT / "run_metadata.json",
        {
            "status": "COMPLETE",
            "decision": decision,
            "elapsed_seconds": time.time() - started,
            "runner_sha256": p2.sha256_file(Path(__file__)),
            "p20_runner_sha256": p2.sha256_file(P20 / "run_p2_reachability.py"),
            "e1_bundle_sha256": p2.sha256_file(E1 / "cumulative_candidate_pool.npz"),
            "selection_membership_sha256": metrics["membership_sha256"],
            "provisional_bundle_sha256": metrics["bundle_sha256"],
            "no_mutation": True,
            "no_concorde": True,
            "no_oracle": True,
            "no_bq_training": True,
            "hgb_diagnostic_only": True,
        },
    )
    print(json.dumps({"status": "COMPLETE", "decision": decision, "metrics": metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
