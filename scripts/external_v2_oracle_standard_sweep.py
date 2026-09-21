#!/usr/bin/env python
"""Continue the external-v2 canonical Oracle sweep without fail-fast timeouts.

Frozen datasets and exact acceptance semantics are read-only. Per-instance hard
cases are recorded outside the exact-reference ledger and never accepted as exact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.external_v2 import (  # noqa: E402
    AUDIT_ROOT, DATA_ROOT, NEW_DATASETS, coordinate_hash, load_new_dataset,
    sha256_file, source_and_config_hashes,
)
from oracle.concorde import (  # noqa: E402
    CANONICAL_CONCORDE_SHA256, INTEGER_SCALE, _parse_tour,
    integer_distance_matrix, resolve_backend, solve_one, tour_cost_int,
    tour_hash, validate_tour_0based,
)

EXACT_SUCCESS = "EXACT_SUCCESS"
TIMEOUT_WITH_CANDIDATE = "TIMEOUT_WITH_CANDIDATE"
TIMEOUT_NO_CANDIDATE = "TIMEOUT_NO_CANDIDATE"
SOLVER_ERROR = "SOLVER_ERROR"
VERIFICATION_FAILURE = "VERIFICATION_FAILURE"
UNRESOLVED_STATUSES = {
    TIMEOUT_WITH_CANDIDATE, TIMEOUT_NO_CANDIDATE,
    SOLVER_ERROR, VERIFICATION_FAILURE,
}
TIMEOUT_SECONDS = 300.0
METRIC = "EXPLICIT_EUCLIDEAN_RINT_SCALE_1000000"

EXACT_COLUMNS = [
    "dataset", "index", "instance_id", "coordinate_hash", "instance_hash",
    "n_nodes", "status", "verified", "optimal_cost_int", "exact_reference_cost_int",
    "runtime_sec", "concorde_runtime_sec", "expert_tour", "exact_tour", "tour_hash",
    "binary_hash", "solver_invocation_config_sha256", "integer_scale", "metric",
    "verification_result", "oracle_tier", "timeout_sec",
]
ATTEMPT_COLUMNS = [
    "dataset", "index", "instance_id", "coordinate_hash", "n_nodes", "status",
    "standard_attempts", "attempt_runtimes_json", "total_concorde_wall_time_sec",
    "candidate_emitted", "candidate_legal", "candidate_cost_int", "candidate_tour_hash",
    "candidate_sol_sha256", "candidate_tour", "repeated_candidate_agreement",
    "standard_attempts_exhausted", "exact_reference_available", "candidate_is_exact",
    "nearest_neighbor_min", "nearest_neighbor_median", "nearest_neighbor_p90",
    "nearest_neighbor_max", "unique_x_count", "unique_y_count", "duplicate_node_count",
    "coordinate_min", "coordinate_max", "solver_invocation_config_sha256",
    "binary_hash", "integer_scale", "metric", "error", "diagnostic_figure_path",
    "final_standard_sweep_state",
]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=suffix, dir=path.parent)
    os.close(fd)
    try:
        if suffix == ".parquet":
            frame.to_parquet(temporary, index=False)
        elif suffix == ".csv":
            frame.to_csv(temporary, index=False)
        else:
            raise ValueError(path)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def solver_config(backend: dict[str, Any]) -> tuple[dict[str, Any], str]:
    value = {
        "backend": backend["backend"], "binary_hash": backend["binary_hash"],
        "invoke_mode": backend["invoke_mode"], "wsl_distro": backend["wsl_distro"],
        "timeout_seconds": TIMEOUT_SECONDS, "integer_scale": INTEGER_SCALE,
        "metric": METRIC, "canonical_input": "EXPLICIT/FULL_MATRIX/rint(Euclidean*1e6)",
    }
    digest = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return value, digest


def structural_diagnostics(coordinates: np.ndarray) -> dict[str, Any]:
    values = np.asarray(coordinates, dtype=np.float64)
    delta = values[:, None, :] - values[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=2))
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    unique_nodes = np.unique(values, axis=0).shape[0]
    return {
        "nearest_neighbor_min": float(nearest.min()),
        "nearest_neighbor_median": float(np.median(nearest)),
        "nearest_neighbor_p90": float(np.quantile(nearest, 0.90)),
        "nearest_neighbor_max": float(nearest.max()),
        "unique_x_count": int(np.unique(values[:, 0]).size),
        "unique_y_count": int(np.unique(values[:, 1]).size),
        "duplicate_node_count": int(len(values) - unique_nodes),
        "coordinate_min": float(values.min()),
        "coordinate_max": float(values.max()),
    }


def preflight_dataset(dataset_name: str) -> tuple[np.ndarray, list[dict[str, str]]]:
    coordinates, manifest = load_new_dataset(dataset_name)
    metadata_path = DATA_ROOT / dataset_name / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_shape = (1000, 500, 2) if dataset_name == "GaussianMixture500" else (1000, 100, 2)
    if coordinates.shape != expected_shape or coordinates.dtype != np.float32:
        raise RuntimeError(f"{dataset_name}: frozen shape/dtype mismatch")
    coordinate_path = DATA_ROOT / dataset_name / "coordinates.npy"
    if sha256_file(coordinate_path) != metadata["coordinates_file_sha256"]:
        raise RuntimeError(f"{dataset_name}: frozen coordinate file hash mismatch")
    if len(manifest) != 1000 or len({r["instance_id"] for r in manifest}) != 1000:
        raise RuntimeError(f"{dataset_name}: frozen manifest cardinality/ID mismatch")
    if sha256_file(DATA_ROOT / dataset_name / "instances.csv") != metadata["instance_manifest_sha256"]:
        raise RuntimeError(f"{dataset_name}: frozen instance manifest hash mismatch")
    for index, (coordinates_i, row) in enumerate(zip(coordinates, manifest, strict=True)):
        if int(row["index"]) != index or row["dataset"] != dataset_name:
            raise RuntimeError(f"{dataset_name}: manifest order mismatch at {index}")
        if coordinate_hash(coordinates_i) != row["coordinate_hash"]:
            raise RuntimeError(f"{dataset_name}: coordinate hash mismatch at {index}")
    return coordinates, manifest


def normalize_exact_ledger(
    dataset_name: str, coordinates: np.ndarray, manifest: list[dict[str, str]],
    backend: dict[str, Any], config_hash: str,
) -> pd.DataFrame:
    path = DATA_ROOT / dataset_name / "reference_ledger.parquet"
    if not path.is_file():
        return pd.DataFrame(columns=EXACT_COLUMNS)
    frame = pd.read_parquet(path)
    required = {"instance_id", "coordinate_hash", "instance_hash", "status", "verified",
                "optimal_cost_int", "runtime_sec", "expert_tour", "tour_hash", "binary_hash"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{dataset_name}: corrupted exact ledger missing {sorted(missing)}")
    if frame.instance_id.astype(str).duplicated().any() or frame.coordinate_hash.astype(str).duplicated().any():
        raise RuntimeError(f"{dataset_name}: corrupted exact ledger duplicate stable key")
    by_id = {row["instance_id"]: (index, row) for index, row in enumerate(manifest)}
    normalized: list[dict[str, Any]] = []
    for item in frame.to_dict("records"):
        instance_id = str(item["instance_id"])
        if instance_id not in by_id:
            raise RuntimeError(f"{dataset_name}: exact ledger unknown ID {instance_id}")
        index, manifest_row = by_id[instance_id]
        expected_hash = manifest_row["coordinate_hash"]
        if str(item["coordinate_hash"]) != expected_hash or str(item["instance_hash"]) != expected_hash:
            raise RuntimeError(f"{dataset_name}: exact ledger hash binding mismatch {instance_id}")
        if str(item["binary_hash"]) != CANONICAL_CONCORDE_SHA256:
            raise RuntimeError(f"{dataset_name}: exact ledger solver hash mismatch {instance_id}")
        if not bool(item["verified"]) or str(item["status"]) != "solved":
            raise RuntimeError(f"{dataset_name}: non-exact row found in exact ledger {instance_id}")
        tour = [int(x) for x in str(item["expert_tour"]).split()]
        legal, message = validate_tour_0based(tour, len(coordinates[index]))
        if not legal:
            raise RuntimeError(f"{dataset_name}: exact ledger illegal tour {instance_id}: {message}")
        computed = tour_cost_int(integer_distance_matrix(coordinates[index]), tour)
        if computed != int(item["optimal_cost_int"]) or tour_hash(tour) != str(item["tour_hash"]):
            raise RuntimeError(f"{dataset_name}: exact ledger tour/cost verification mismatch {instance_id}")
        runtime = float(item["runtime_sec"])
        normalized.append({
            "dataset": dataset_name, "index": index, "instance_id": instance_id,
            "coordinate_hash": expected_hash, "instance_hash": expected_hash,
            "n_nodes": len(coordinates[index]), "status": "solved", "verified": True,
            "optimal_cost_int": computed, "exact_reference_cost_int": computed,
            "runtime_sec": runtime, "concorde_runtime_sec": runtime,
            "expert_tour": " ".join(map(str, tour)), "exact_tour": " ".join(map(str, tour)),
            "tour_hash": tour_hash(tour), "binary_hash": backend["binary_hash"],
            "solver_invocation_config_sha256": config_hash, "integer_scale": INTEGER_SCALE,
            "metric": METRIC, "verification_result": "PASS",
            "oracle_tier": str(item["oracle_tier"]) if item.get("oracle_tier") not in (None, "") else "STANDARD",
            "timeout_sec": float(item["timeout_sec"]) if item.get("timeout_sec") not in (None, "") else TIMEOUT_SECONDS,
        })
    return pd.DataFrame(normalized, columns=EXACT_COLUMNS).sort_values("index", kind="stable").reset_index(drop=True)


def attempts_paths(dataset_name: str) -> tuple[Path, Path]:
    oracle_root = DATA_ROOT / dataset_name / "oracle"
    return oracle_root / "standard_attempts.parquet", oracle_root / "standard_attempts.csv"


def load_attempts(dataset_name: str, manifest: list[dict[str, str]]) -> pd.DataFrame:
    parquet_path, _ = attempts_paths(dataset_name)
    if not parquet_path.is_file():
        return pd.DataFrame(columns=ATTEMPT_COLUMNS)
    frame = pd.read_parquet(parquet_path)
    missing = set(ATTEMPT_COLUMNS).difference(frame.columns)
    if missing:
        raise RuntimeError(f"{dataset_name}: corrupted attempt ledger missing {sorted(missing)}")
    if frame.instance_id.astype(str).duplicated().any():
        raise RuntimeError(f"{dataset_name}: duplicate terminal attempt records")
    manifest_map = {r["instance_id"]: r["coordinate_hash"] for r in manifest}
    for row in frame.itertuples(index=False):
        if row.instance_id not in manifest_map or str(row.coordinate_hash) != manifest_map[row.instance_id]:
            raise RuntimeError(f"{dataset_name}: attempt ledger stable binding mismatch {row.instance_id}")
        if row.status not in UNRESOLVED_STATUSES | {EXACT_SUCCESS}:
            raise RuntimeError(f"{dataset_name}: invalid attempt status {row.status}")
    return frame.loc[:, ATTEMPT_COLUMNS].sort_values("index", kind="stable").reset_index(drop=True)


def persist_exact(dataset_name: str, frame: pd.DataFrame) -> None:
    frame = frame.copy()
    if "oracle_tier" not in frame.columns:
        frame["oracle_tier"] = "STANDARD"
    if "timeout_sec" not in frame.columns:
        frame["timeout_sec"] = TIMEOUT_SECONDS
    frame["oracle_tier"] = frame["oracle_tier"].fillna("STANDARD")
    frame["timeout_sec"] = frame["timeout_sec"].fillna(TIMEOUT_SECONDS)
    frame = frame.loc[:, EXACT_COLUMNS].sort_values("index", kind="stable").reset_index(drop=True)
    root = DATA_ROOT / dataset_name
    atomic_frame(root / "reference_ledger.parquet", frame)
    atomic_frame(root / "reference_ledger.csv", frame)


def persist_attempts(dataset_name: str, frame: pd.DataFrame) -> None:
    frame = frame.loc[:, ATTEMPT_COLUMNS].sort_values("index", kind="stable").reset_index(drop=True)
    parquet_path, csv_path = attempts_paths(dataset_name)
    atomic_frame(parquet_path, frame)
    atomic_frame(csv_path, frame)


def file_hash_if_present(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def import_0853_evidence(
    coordinates: np.ndarray, manifest: list[dict[str, str]], attempts: pd.DataFrame,
    backend: dict[str, Any], config_hash: str,
) -> pd.DataFrame:
    instance_id = "EvoReal_S1_100_0853"
    if instance_id in set(attempts.instance_id.astype(str)):
        return attempts
    index = 853
    row = manifest[index]
    retry_path = DATA_ROOT / "EvoReal_S1_100" / "ORACLE_RETRY_ONCE.json"
    failure_path = DATA_ROOT / "EvoReal_S1_100" / "ORACLE_FAILURE.json"
    if not retry_path.is_file() or not failure_path.is_file():
        raise RuntimeError("EvoReal_S1_100_0853 exhausted-attempt evidence is missing")
    retry = json.loads(retry_path.read_text(encoding="utf-8"))
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    expected_hash = row["coordinate_hash"]
    if retry.get("coordinate_hash") != expected_hash or failure.get("coordinate_hash") != expected_hash:
        raise RuntimeError("EvoReal_S1_100_0853 retry evidence hash mismatch")
    candidate_tour = [int(x) for x in retry["candidate_tour"]]
    legal, message = validate_tour_0based(candidate_tour, 100)
    if not legal:
        raise RuntimeError(f"EvoReal_S1_100_0853 stored candidate became illegal: {message}")
    computed = tour_cost_int(integer_distance_matrix(coordinates[index]), candidate_tour)
    if computed != int(retry["candidate_cost_int"]):
        raise RuntimeError("EvoReal_S1_100_0853 stored candidate cost mismatch")
    h = expected_hash
    first_sol = DATA_ROOT / "_oracle_work" / "EvoReal_S1_100" / h / f"{h}.sol"
    retry_sol = DATA_ROOT / "_oracle_retry_once" / "EvoReal_S1_100" / h / f"{h}.sol"
    first_sha, retry_sha = file_hash_if_present(first_sol), file_hash_if_present(retry_sol)
    agreement = first_sha is not None and first_sha == retry_sha
    diagnostics = structural_diagnostics(coordinates[index])
    imported = {
        "dataset": "EvoReal_S1_100", "index": index, "instance_id": instance_id,
        "coordinate_hash": expected_hash, "n_nodes": 100, "status": TIMEOUT_WITH_CANDIDATE,
        "standard_attempts": 2, "attempt_runtimes_json": json.dumps([300.0, 300.0]),
        "total_concorde_wall_time_sec": 600.0, "candidate_emitted": True,
        "candidate_legal": True, "candidate_cost_int": computed,
        "candidate_tour_hash": tour_hash(candidate_tour), "candidate_sol_sha256": retry_sha,
        "candidate_tour": " ".join(map(str, candidate_tour)),
        "repeated_candidate_agreement": agreement, "standard_attempts_exhausted": True,
        "exact_reference_available": False, "candidate_is_exact": False,
        **diagnostics, "solver_invocation_config_sha256": config_hash,
        "binary_hash": backend["binary_hash"], "integer_scale": INTEGER_SCALE,
        "metric": METRIC,
        "error": "two canonical 300 s attempts timed out; legal candidates are diagnostic only",
        "diagnostic_figure_path": retry.get("plot", "audits/external_benchmark_suite_v2/10_blocking_instance_EvoReal_S1_100_0853.png"),
        "final_standard_sweep_state": "UNRESOLVED_STANDARD_ATTEMPTS_EXHAUSTED",
    }
    result = pd.concat([attempts, pd.DataFrame([imported], columns=ATTEMPT_COLUMNS)], ignore_index=True)
    persist_attempts("EvoReal_S1_100", result)
    evidence = {
        "instance_id": instance_id, "coordinate_hash": expected_hash,
        "attempt_runtimes_sec": [300.0, 300.0], "attempt_statuses": ["TIMEOUT", "TIMEOUT"],
        "candidate_cost_int": computed, "candidate_tour_hash": tour_hash(candidate_tour),
        "first_candidate_sol_sha256": first_sha, "retry_candidate_sol_sha256": retry_sha,
        "candidates_match_exactly": agreement, "candidate_is_exact": False,
        "exact_reference_available": False, "diagnostic_figure_path": imported["diagnostic_figure_path"],
        "source_retry_record": retry_path.relative_to(ROOT).as_posix(),
        "source_failure_record": failure_path.relative_to(ROOT).as_posix(),
    }
    atomic_json(DATA_ROOT / "EvoReal_S1_100" / "oracle" / "unresolved" / "0853_imported_evidence.json", evidence)
    return result


def attempt_one(
    dataset_name: str, index: int, instance_id: str, expected_hash: str,
    coordinates: np.ndarray, backend: dict[str, Any], config_hash: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    work = DATA_ROOT / dataset_name / "oracle" / "work" / expected_hash
    start = time.perf_counter()
    oracle_row: dict[str, Any] | None = None
    status = SOLVER_ERROR
    error = ""
    candidate_emitted = False
    candidate_legal = False
    candidate_cost: int | None = None
    candidate_tour_hash: str | None = None
    candidate_sol_sha: str | None = None
    candidate_tour_text = ""
    try:
        oracle_row = solve_one(
            coordinates, expected_hash, work, backend, timeout=TIMEOUT_SECONDS,
        )
        status = EXACT_SUCCESS
    except subprocess.TimeoutExpired as exc:
        status = TIMEOUT_NO_CANDIDATE
        error = f"subprocess.TimeoutExpired after {exc.timeout} seconds"
    except RuntimeError as exc:
        error = str(exc)
        status = VERIFICATION_FAILURE if "verification failed" in error.lower() else SOLVER_ERROR
    except BaseException as exc:  # recorded per-instance; integrity checks remain outside this worker
        error = f"{type(exc).__name__}: {exc}"
        status = SOLVER_ERROR
    elapsed = time.perf_counter() - start

    if status != EXACT_SUCCESS:
        sol = work / f"{expected_hash}.sol"
        candidate_emitted = sol.is_file() and sol.stat().st_size > 0
        candidate_sol_sha = file_hash_if_present(sol)
        if candidate_emitted:
            try:
                candidate_tour = _parse_tour(sol, len(coordinates))
                candidate_legal, _ = validate_tour_0based(candidate_tour, len(coordinates))
                if candidate_legal:
                    candidate_cost = tour_cost_int(integer_distance_matrix(coordinates), candidate_tour)
                    candidate_tour_hash = tour_hash(candidate_tour)
                    candidate_tour_text = " ".join(map(str, candidate_tour))
                    if status == TIMEOUT_NO_CANDIDATE:
                        status = TIMEOUT_WITH_CANDIDATE
                    destination = DATA_ROOT / dataset_name / "oracle" / "unresolved" / "candidate_tours" / f"{instance_id}.sol"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(sol, destination)
            except (OSError, ValueError):
                candidate_legal = False

    diagnostics = structural_diagnostics(coordinates)
    attempt = {
        "dataset": dataset_name, "index": index, "instance_id": instance_id,
        "coordinate_hash": expected_hash, "n_nodes": len(coordinates), "status": status,
        "standard_attempts": 1, "attempt_runtimes_json": json.dumps([elapsed]),
        "total_concorde_wall_time_sec": elapsed, "candidate_emitted": candidate_emitted,
        "candidate_legal": candidate_legal, "candidate_cost_int": candidate_cost,
        "candidate_tour_hash": candidate_tour_hash, "candidate_sol_sha256": candidate_sol_sha,
        "candidate_tour": candidate_tour_text, "repeated_candidate_agreement": None,
        "standard_attempts_exhausted": status != EXACT_SUCCESS,
        "exact_reference_available": status == EXACT_SUCCESS, "candidate_is_exact": False,
        **diagnostics, "solver_invocation_config_sha256": config_hash,
        "binary_hash": backend["binary_hash"], "integer_scale": INTEGER_SCALE,
        "metric": METRIC, "error": error, "diagnostic_figure_path": "",
        "final_standard_sweep_state": "EXACT_VERIFIED" if status == EXACT_SUCCESS else "UNRESOLVED_STANDARD_ATTEMPT_COMPLETE",
    }
    if oracle_row is None:
        return attempt, None
    exact_tour = str(oracle_row["expert_tour"])
    exact = {
        "dataset": dataset_name, "index": index, "instance_id": instance_id,
        "coordinate_hash": expected_hash, "instance_hash": expected_hash,
        "n_nodes": len(coordinates), "status": "solved", "verified": True,
        "optimal_cost_int": int(oracle_row["optimal_cost_int"]),
        "exact_reference_cost_int": int(oracle_row["optimal_cost_int"]),
        "runtime_sec": float(oracle_row["runtime_sec"]),
        "concorde_runtime_sec": float(oracle_row["runtime_sec"]),
        "expert_tour": exact_tour, "exact_tour": exact_tour,
        "tour_hash": str(oracle_row["tour_hash"]), "binary_hash": str(oracle_row["binary_hash"]),
        "solver_invocation_config_sha256": config_hash, "integer_scale": INTEGER_SCALE,
        "metric": METRIC, "verification_result": "PASS",
        "oracle_tier": "STANDARD", "timeout_sec": TIMEOUT_SECONDS,
    }
    return attempt, exact


def write_unresolved(dataset_name: str, attempts: pd.DataFrame, exact: pd.DataFrame | None = None) -> None:
    exact_ids = set(exact.instance_id.astype(str)) if exact is not None and len(exact) else set()
    unresolved = attempts[
        attempts.status.isin(UNRESOLVED_STATUSES)
        & ~attempts.instance_id.astype(str).isin(exact_ids)
    ].copy()
    root = DATA_ROOT / dataset_name / "oracle" / "unresolved"
    atomic_frame(root / "unresolved_instances.csv", unresolved)
    atomic_json(root / "unresolved_details.json", unresolved.to_dict("records"))


def write_sweep_state(
    dataset_name: str, manifest: list[dict[str, str]], exact: pd.DataFrame, attempts: pd.DataFrame,
) -> None:
    exact_ids = set(exact.instance_id.astype(str))
    attempt_map = {str(r.instance_id): str(r.status) for r in attempts.itertuples(index=False)}
    rows = []
    for row in manifest:
        instance_id = row["instance_id"]
        if instance_id in exact_ids:
            status = EXACT_SUCCESS if attempt_map.get(instance_id) == EXACT_SUCCESS else "CACHE_EXACT_SUCCESS"
        else:
            status = attempt_map.get(instance_id, "NOT_YET_ATTEMPTED")
        rows.append({"dataset": dataset_name, "instance_id": instance_id,
                     "coordinate_hash": row["coordinate_hash"], "state": status})
    atomic_frame(DATA_ROOT / dataset_name / "oracle" / "sweep_state.csv", pd.DataFrame(rows))


def update_status(status: str, details: dict[str, Any]) -> None:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(AUDIT_ROOT / "ORACLE_STANDARD_SWEEP_STATUS.json", {"status": status, **details})
    strict_path = AUDIT_ROOT / "STRICT_VERIFICATION.json"
    strict = json.loads(strict_path.read_text(encoding="utf-8")) if strict_path.is_file() else {}
    strict["status"] = status
    strict["oracle_standard_sweep"] = details
    strict["blocking_causes"] = details.get("integrity_failures", [])
    strict["hard_case_references_pending"] = details.get("unresolved", [])
    atomic_json(strict_path, strict)


def dataset_counts(dataset_name: str) -> dict[str, Any]:
    exact_path = DATA_ROOT / dataset_name / "reference_ledger.parquet"
    exact = pd.read_parquet(exact_path) if exact_path.is_file() else pd.DataFrame()
    exact_ids = set(exact.instance_id.astype(str)) if len(exact) else set()
    attempts_path, _ = attempts_paths(dataset_name)
    attempts = pd.read_parquet(attempts_path) if attempts_path.is_file() else pd.DataFrame(columns=ATTEMPT_COLUMNS)
    if len(attempts):
        unresolved = attempts[
            attempts.status.isin(UNRESOLVED_STATUSES)
            & ~attempts.instance_id.astype(str).isin(exact_ids)
        ]
    else:
        unresolved = attempts
    return {"dataset": dataset_name, "exact": len(exact), "unresolved": len(unresolved),
            "not_yet_attempted": 1000 - len(exact) - len(unresolved),
            "unresolved_ids": unresolved.instance_id.astype(str).tolist() if len(unresolved) else []}


def run_dataset(
    dataset_name: str, workers: int, batch_size: int, backend: dict[str, Any], config_hash: str,
    validate_only: bool,
) -> dict[str, Any]:
    coordinates, manifest = preflight_dataset(dataset_name)
    exact = normalize_exact_ledger(dataset_name, coordinates, manifest, backend, config_hash)
    attempts = load_attempts(dataset_name, manifest)
    if dataset_name == "EvoReal_S1_100":
        attempts = import_0853_evidence(coordinates, manifest, attempts, backend, config_hash)
    exact_ids = set(exact.instance_id.astype(str))
    terminal_ids = set(attempts.instance_id.astype(str))
    live_unresolved = {
        str(r.instance_id) for r in attempts.itertuples(index=False)
        if r.status in UNRESOLVED_STATUSES and str(r.instance_id) not in exact_ids
    }
    if exact_ids & live_unresolved:
        raise RuntimeError(f"{dataset_name}: same instance appears exact and unresolved")
    pending = [i for i, row in enumerate(manifest) if row["instance_id"] not in exact_ids | terminal_ids]
    if validate_only:
        return {"dataset": dataset_name, "exact": len(exact),
                "unresolved": int(attempts.status.isin(UNRESOLVED_STATUSES).sum()),
                "pending": len(pending)}
    persist_exact(dataset_name, exact)
    persist_attempts(dataset_name, attempts)
    write_unresolved(dataset_name, attempts, exact)
    write_sweep_state(dataset_name, manifest, exact, attempts)
    print(f"{dataset_name}: exact={len(exact)} unresolved={int(attempts.status.isin(UNRESOLVED_STATUSES).sum())} pending={len(pending)}", flush=True)

    for batch_start in range(0, len(pending), batch_size):
        indices = pending[batch_start:batch_start + batch_size]
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(indices)))) as pool:
            futures = {}
            for index in indices:
                row = manifest[index]
                future = pool.submit(
                    attempt_one, dataset_name, index, row["instance_id"], row["coordinate_hash"],
                    coordinates[index], backend, config_hash,
                )
                futures[future] = index
            for future in as_completed(futures):
                attempt, exact_row = future.result()
                if exact_row is not None:
                    exact = pd.concat([exact, pd.DataFrame([exact_row], columns=EXACT_COLUMNS)], ignore_index=True)
                    if exact.instance_id.astype(str).duplicated().any():
                        raise RuntimeError(f"{dataset_name}: duplicate exact commit {exact_row['instance_id']}")
                    persist_exact(dataset_name, exact)
                attempts = pd.concat([attempts, pd.DataFrame([attempt], columns=ATTEMPT_COLUMNS)], ignore_index=True)
                if attempts.instance_id.astype(str).duplicated().any():
                    raise RuntimeError(f"{dataset_name}: duplicate attempt commit {attempt['instance_id']}")
                persist_attempts(dataset_name, attempts)
                if attempt["status"] != EXACT_SUCCESS:
                    print(f"BLOCKER {dataset_name}/{attempt['instance_id']} {attempt['status']}", flush=True)
        write_unresolved(dataset_name, attempts, exact)
        write_sweep_state(dataset_name, manifest, exact, attempts)
        errors = attempts[attempts.status == SOLVER_ERROR]
        if len(errors) >= 8:
            signatures = errors.error.astype(str).str.slice(0, 240).value_counts()
            if len(signatures) and int(signatures.iloc[0]) >= 8:
                raise RuntimeError(f"{dataset_name}: probable infrastructure-wide solver failure: {signatures.index[0]}")
        counts = dataset_counts(dataset_name)
        update_status("ORACLE_STANDARD_SWEEP_IN_PROGRESS", {
            "current_dataset": dataset_name, "dataset_progress": counts,
            "unresolved": counts["unresolved_ids"], "integrity_failures": [],
        })
        print(f"{dataset_name}: exact={counts['exact']} unresolved={counts['unresolved']} pending={counts['not_yet_attempted']}", flush=True)
    final = dataset_counts(dataset_name)
    if final["not_yet_attempted"] != 0 or final["exact"] + final["unresolved"] != 1000:
        raise RuntimeError(f"{dataset_name}: sweep state did not close: {final}")
    return final


def runtime_stats(frame: pd.DataFrame) -> dict[str, float | None]:
    if not len(frame):
        return {key: None for key in ("median", "p90", "p95", "p99", "max")}
    values = frame.concorde_runtime_sec.astype(float).to_numpy()
    return {"median": float(np.median(values)), "p90": float(np.quantile(values, .90)),
            "p95": float(np.quantile(values, .95)), "p99": float(np.quantile(values, .99)),
            "max": float(values.max())}


def finalize_reports(config_hash: str) -> tuple[str, list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    for dataset_name in NEW_DATASETS:
        exact = pd.read_parquet(DATA_ROOT / dataset_name / "reference_ledger.parquet")
        attempts = pd.read_parquet(attempts_paths(dataset_name)[0])
        exact_ids = set(exact.instance_id.astype(str))
        unresolved = attempts[
            attempts.status.isin(UNRESOLVED_STATUSES)
            & ~attempts.instance_id.astype(str).isin(exact_ids)
        ].copy()
        if len(exact) + len(unresolved) != 1000:
            raise RuntimeError(f"{dataset_name}: cannot finalize incomplete sweep")
        by_status = unresolved.status.value_counts().to_dict()
        summaries.append({
            "dataset": dataset_name, "total": 1000, "exact_success": len(exact),
            "exact_success_rate": len(exact) / 1000,
            "timeout_with_candidate": int(by_status.get(TIMEOUT_WITH_CANDIDATE, 0)),
            "timeout_no_candidate": int(by_status.get(TIMEOUT_NO_CANDIDATE, 0)),
            "solver_error": int(by_status.get(SOLVER_ERROR, 0)),
            "verification_failure": int(by_status.get(VERIFICATION_FAILURE, 0)),
            "unresolved": len(unresolved), "unresolved_rate": len(unresolved) / 1000,
            **{f"success_runtime_{k}_sec": v for k, v in runtime_stats(exact).items()},
        })
        blockers.extend(unresolved.to_dict("records"))
    blocker_frame = pd.DataFrame(blockers, columns=ATTEMPT_COLUMNS)
    atomic_frame(AUDIT_ROOT / "12_oracle_blockers.csv", blocker_frame)
    atomic_json(AUDIT_ROOT / "13_oracle_blockers_detailed.json", blockers)
    total_exact = sum(row["exact_success"] for row in summaries)
    total_unresolved = sum(row["unresolved"] for row in summaries)
    unresolved_ids = [str(row["instance_id"]) for row in blockers]
    if total_unresolved == 0:
        status = "EXTERNAL_BENCHMARK_SUITE_V2_READY_FOR_EVALUATION"
    else:
        hardcase_path = DATA_ROOT / "oracle" / "hardcase_escalation_attempts.parquet"
        if hardcase_path.is_file():
            hardcase = pd.read_parquet(hardcase_path)
            completed = set(hardcase.instance_id.astype(str)) if len(hardcase) else set()
            if "hardcase_escalation_exhausted" in hardcase.columns:
                completed = set(
                    hardcase.loc[
                        hardcase.hardcase_escalation_exhausted.astype(bool), "instance_id"
                    ].astype(str)
                )
            exhausted = bool(unresolved_ids) and set(unresolved_ids).issubset(completed)
            # Isolated unresolved exact references after a completed Tier-2 attempt are
            # not integrity failures; they wait for explicit human review.
            status = ("EXTERNAL_BENCHMARK_SUITE_V2_ONE_EXACT_REFERENCE_UNRESOLVED" if exhausted
                      else "EXTERNAL_BENCHMARK_SUITE_V2_PENDING_HARD_CASE_REFERENCES")
        else:
            status = "EXTERNAL_BENCHMARK_SUITE_V2_PENDING_HARD_CASE_REFERENCES"
    lines = ["# Standard Oracle sweep summary", "", "All frozen instances received their permitted canonical 300 s standard attempt, except previously exhausted EvoReal_S1_100_0853, whose two existing attempts were imported without rerun.", ""]
    for row in summaries:
        lines.extend([
            f"## {row['dataset']}", "",
            f"- Exact: {row['exact_success']}/1000 ({row['exact_success_rate']:.3%})",
            f"- Unresolved: {row['unresolved']}/1000 ({row['unresolved_rate']:.3%})",
            f"- Timeout with candidate: {row['timeout_with_candidate']}",
            f"- Timeout without candidate: {row['timeout_no_candidate']}",
            f"- Solver error: {row['solver_error']}",
            f"- Verification failure: {row['verification_failure']}",
            f"- Successful runtimes (s): median={row['success_runtime_median_sec']}, P90={row['success_runtime_p90_sec']}, P95={row['success_runtime_p95_sec']}, P99={row['success_runtime_p99_sec']}, max={row['success_runtime_max_sec']}", "",
        ])
    lines.extend([f"TOTAL exact: {total_exact}/4000", f"TOTAL unresolved: {total_unresolved}/4000", "", f"Status: `{status}`"])
    (AUDIT_ROOT / "11_oracle_standard_sweep_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    counts = {row["dataset"]: row["unresolved"] for row in summaries}
    if total_unresolved == 0:
        interpretation = "No hard-case tail was observed."
    elif total_unresolved == 1:
        interpretation = "The unresolved set is an isolated hard instance."
    elif total_unresolved <= 40:
        interpretation = "The unresolved set is a small hard tail (at most 1% of the 4000-instance suite)."
    else:
        affected = [name for name, count in counts.items() if count]
        interpretation = "The hard cases are systematic across: " + ", ".join(affected) + "."
    distribution = ["# Oracle blocker distribution", "", f"Total blockers: {total_unresolved}/4000.", ""]
    distribution.extend(f"- {name}: {count}/1000" for name, count in counts.items())
    distribution.extend(["", interpretation, "", "Repeated candidates are diagnostic and are not inferred to be optimal."])
    (AUDIT_ROOT / "14_oracle_blocker_distribution.md").write_text("\n".join(distribution) + "\n", encoding="utf-8")
    update_status(status, {"standard_sweep_complete": True, "dataset_summaries": summaries,
                           "total_exact": total_exact, "total_unresolved": total_unresolved,
                           "unresolved": unresolved_ids, "integrity_failures": [],
                           "solver_invocation_config_sha256": config_hash})
    return status, summaries


def print_final(status: str, summaries: list[dict[str, Any]]) -> None:
    print("STANDARD ORACLE SWEEP COMPLETE", flush=True)
    print("", flush=True)
    for row in summaries:
        print(f"{row['dataset']}:", flush=True)
        print(f"exact = {row['exact_success']}/1000", flush=True)
        print(f"unresolved = {row['unresolved']}", flush=True)
        print("", flush=True)
    exact = sum(row["exact_success"] for row in summaries)
    unresolved = sum(row["unresolved"] for row in summaries)
    print("TOTAL:", flush=True); print(f"exact = {exact}/4000", flush=True)
    print(f"unresolved = {unresolved}/4000", flush=True); print("", flush=True)
    print("STATUS:", flush=True); print(status, flush=True)
    blockers = pd.read_csv(AUDIT_ROOT / "12_oracle_blockers.csv")
    print("UNRESOLVED INSTANCE IDS:", flush=True)
    for instance_id in blockers.instance_id.astype(str).tolist():
        print(instance_id, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.workers != 8:
        raise RuntimeError("canonical standard sweep requires exactly 8 workers")
    source_and_config_hashes()
    backend = resolve_backend()
    if backend["binary_hash"] != CANONICAL_CONCORDE_SHA256 or backend["invoke_mode"] != "wsl":
        raise RuntimeError("canonical Concorde backend/invocation mismatch")
    config, config_hash = solver_config(backend)
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(AUDIT_ROOT / "ORACLE_STANDARD_SWEEP_CONFIG.json", config)
    plans = []
    if not args.validate_only:
        existing = [dataset_counts(name) for name in NEW_DATASETS]
        update_status("ORACLE_STANDARD_SWEEP_IN_PROGRESS", {
            "standard_sweep_complete": False, "dataset_progress": existing,
            "unresolved": [item for row in existing for item in row["unresolved_ids"]],
            "integrity_failures": [], "solver_invocation_config_sha256": config_hash,
        })
    try:
        for dataset_name in NEW_DATASETS:
            plans.append(run_dataset(dataset_name, args.workers, args.batch_size, backend, config_hash, args.validate_only))
        if args.validate_only:
            print(json.dumps({"status": "VALIDATED", "datasets": plans}, indent=2), flush=True)
            return 0
        status, summaries = finalize_reports(config_hash)
        print_final(status, summaries)
        return 0
    except BaseException as exc:
        if not args.validate_only:
            update_status("EXTERNAL_BENCHMARK_SUITE_V2_BLOCKED", {
                "integrity_failures": [f"{type(exc).__name__}: {exc}"],
                "traceback": traceback.format_exc(), "unresolved": [],
            })
        raise


if __name__ == "__main__":
    raise SystemExit(main())


