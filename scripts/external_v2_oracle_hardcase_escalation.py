#!/usr/bin/env python
"""Generic Tier-2 hard-case exact-reference escalation for EXTERNAL_BENCHMARK_SUITE_V2.

Eligible instances are discovered from persisted standard-attempt / exact ledgers.
The only solver change relative to the canonical 300 s protocol is timeout = 3600 s.
Each eligible instance receives at most one completed Tier-2 attempt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.external_v2 import (  # noqa: E402
    AUDIT_ROOT, DATA_ROOT, NEW_DATASETS, sha256_file, source_and_config_hashes,
)
from oracle.concorde import (  # noqa: E402
    CANONICAL_CONCORDE_SHA256, INTEGER_SCALE, _parse_tour, _reported,
    integer_distance_matrix, resolve_backend, tour_cost_int, tour_hash,
    validate_tour_0based, windows_to_wsl_path, write_explicit_tsp,
)

HARDCASE_EXACT_SUCCESS = "HARDCASE_EXACT_SUCCESS"
HARDCASE_TIMEOUT_WITH_CANDIDATE = "HARDCASE_TIMEOUT_WITH_CANDIDATE"
HARDCASE_TIMEOUT_NO_CANDIDATE = "HARDCASE_TIMEOUT_NO_CANDIDATE"
HARDCASE_SOLVER_ERROR = "HARDCASE_SOLVER_ERROR"
HARDCASE_VERIFICATION_FAILURE = "HARDCASE_VERIFICATION_FAILURE"
TERMINAL_HARDCASE = {
    HARDCASE_EXACT_SUCCESS, HARDCASE_TIMEOUT_WITH_CANDIDATE,
    HARDCASE_TIMEOUT_NO_CANDIDATE, HARDCASE_SOLVER_ERROR,
    HARDCASE_VERIFICATION_FAILURE,
}
TIMEOUT_UNRESOLVED = {"TIMEOUT_WITH_CANDIDATE", "TIMEOUT_NO_CANDIDATE"}
HARDCASE_TIMEOUT_SEC = 3600.0
STANDARD_TIMEOUT_SEC = 300.0
ORACLE_TIER = "HARD_CASE_ESCALATION"
METRIC = "EXPLICIT_EUCLIDEAN_RINT_SCALE_1000000"
STATUS_IN_PROGRESS = "HARD_CASE_EXACT_REFERENCE_IN_PROGRESS"
STATUS_READY = "EXTERNAL_BENCHMARK_SUITE_V2_READY_FOR_EVALUATION"
STATUS_ONE_UNRESOLVED = "EXTERNAL_BENCHMARK_SUITE_V2_ONE_EXACT_REFERENCE_UNRESOLVED"
STATUS_BLOCKED = "EXTERNAL_BENCHMARK_SUITE_V2_BLOCKED"

HARDCASE_COLUMNS = [
    "dataset", "index", "instance_id", "coordinate_hash", "n_nodes", "status",
    "oracle_tier", "timeout_sec", "tier2_attempts", "standard_attempts",
    "standard_timeout_sec", "exact_reference_available",
    "hardcase_escalation_exhausted", "verified", "verification_result",
    "solver_status", "exact_reference_cost_int", "exact_tour", "tour_hash",
    "candidate_emitted", "candidate_legal", "candidate_cost_int",
    "candidate_tour_hash", "candidate_sol_sha256", "candidate_tour",
    "runtime_sec", "concorde_runtime_sec", "solver_invocation_config_sha256",
    "binary_hash", "integer_scale", "metric", "stdout_path", "stderr_path",
    "work_dir", "error",
]
CANONICAL_EXACT_FIELDS = (
    "instance_id", "coordinate_hash", "instance_hash", "n_nodes", "status",
    "verified", "optimal_cost_int", "exact_reference_cost_int", "expert_tour",
    "exact_tour", "tour_hash", "binary_hash", "integer_scale", "metric",
)
FROZEN_DATASET_SHA256 = {
    "EvoReal_S1_100": "28c8e4c0b8f57f840e6b444e45a21c795993205faae4e27ffc9dcae1df55448f",
    "EvoReal_S2_100": "f393b1be940e09ff3f27c5818fb8b721dd240891d0cda22fedefc10e28c7256c",
    "EvoReal_S3_100": "d1f74c229e6efc609845f3570e58fa199158b5feaafc526792c2b3ee9dbab48f",
    "GaussianMixture500": "818dc8de29128231a725d2d9e3fdfb1f439c7d9054fe8c31124ed4fc9d1accd2",
}
POLICY_TEXT = """# Hard-case exact-reference escalation policy

The hard-case escalation policy was defined uniformly for any benchmark instance that failed to obtain an exact certificate after the standard 300-second Concorde protocol. The escalation changes only the Concorde time limit from 300 to 3600 seconds and allows one additional attempt. No dataset, solver parameter, metric, coordinate representation, or exact-reference acceptance criterion is changed.

The policy is independent of BQ model performance and is used solely to obtain exact benchmark reference solutions.

## Frozen tiers

### Tier 1 — Standard
Canonical Concorde; timeout = 300 s.

### Tier 2 — Hard-case escalation
For any benchmark instance that remains unresolved after exhausting the standard protocol:

- use the SAME canonical Concorde pipeline
- use the SAME input instance
- use the SAME coordinate representation
- use the SAME integer scaling
- use the SAME Euclidean metric
- use the SAME Concorde binary
- use the SAME command-line options
- use the SAME exact verification logic
- change ONLY timeout: 300 s -> 3600 s

Each unresolved instance receives AT MOST ONE Tier-2 run.

Eligibility is discovered from the persisted blocker/attempt ledger. The rule is generic and does not hard-code any instance ID.

## Exact-reference admission

Only `HARDCASE_EXACT_SUCCESS` may enter the exact-reference ledger.

A legal Hamiltonian candidate emitted without an optimality certificate is diagnostic only and is not an exact reference.

## Exhaustion

If the single 3600 s attempt does not produce a verified exact reference:

- `hardcase_escalation_exhausted = true`
- `exact_reference_available = false`
- the instance is retained in the benchmark
- no automatic replacement, deletion, heuristic substitution, or longer retry is performed
- scientific status is `EXTERNAL_BENCHMARK_SUITE_V2_ONE_EXACT_REFERENCE_UNRESOLVED`
- further action requires an explicit human decision

## Status language

- During Tier-2 execution: `HARD_CASE_EXACT_REFERENCE_IN_PROGRESS`
- If all references close: `EXTERNAL_BENCHMARK_SUITE_V2_READY_FOR_EVALUATION`
- If Tier-2 is exhausted but one reference remains unresolved: `EXTERNAL_BENCHMARK_SUITE_V2_ONE_EXACT_REFERENCE_UNRESOLVED`
- `EXTERNAL_BENCHMARK_SUITE_V2_BLOCKED` is reserved for genuine integrity/protocol failures (dataset hash mismatch, corrupted ledger, metric inconsistency, source provenance failure, canonical solver invocation corruption). An isolated unresolved exact-reference hard case is not an integrity failure.
"""


def load_sweep_module():
    path = Path(__file__).with_name("external_v2_oracle_standard_sweep.py")
    spec = importlib.util.spec_from_file_location("external_v2_oracle_standard_sweep", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    try:
        if pd.isna(value):
            return None
    except (ValueError, TypeError):
        pass
    return value


def hardcase_paths() -> tuple[Path, Path]:
    root = DATA_ROOT / "oracle"
    return root / "hardcase_escalation_attempts.parquet", root / "hardcase_escalation_attempts.csv"


def solver_config(backend: dict[str, Any], timeout_seconds: float) -> tuple[dict[str, Any], str]:
    value = {
        "backend": backend["backend"], "binary_hash": backend["binary_hash"],
        "invoke_mode": backend["invoke_mode"], "wsl_distro": backend["wsl_distro"],
        "timeout_seconds": timeout_seconds, "integer_scale": INTEGER_SCALE,
        "metric": METRIC, "canonical_input": "EXPLICIT/FULL_MATRIX/rint(Euclidean*1e6)",
    }
    digest = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return value, digest


def load_hardcase_ledger() -> pd.DataFrame:
    parquet_path, _ = hardcase_paths()
    if not parquet_path.is_file():
        return pd.DataFrame(columns=HARDCASE_COLUMNS)
    frame = pd.read_parquet(parquet_path)
    missing = set(HARDCASE_COLUMNS).difference(frame.columns)
    if missing:
        raise RuntimeError(f"corrupted hard-case ledger missing {sorted(missing)}")
    if len(frame) and frame.instance_id.astype(str).duplicated().any():
        raise RuntimeError("duplicate hard-case attempt records")
    return frame.loc[:, HARDCASE_COLUMNS].reset_index(drop=True)


def persist_hardcase_ledger(frame: pd.DataFrame) -> None:
    sweep = load_sweep_module()
    frame = frame.loc[:, HARDCASE_COLUMNS].reset_index(drop=True)
    parquet_path, csv_path = hardcase_paths()
    sweep.atomic_frame(parquet_path, frame)
    sweep.atomic_frame(csv_path, frame)


def dataset_tensor_sha256(dataset_name: str) -> str:
    import numpy as np
    from evaluation.external_v2 import dataset_tensor_hash
    coordinates = np.load(DATA_ROOT / dataset_name / "coordinates.npy", allow_pickle=False)
    return dataset_tensor_hash(coordinates)


def exact_ledger(dataset_name: str) -> pd.DataFrame:
    path = DATA_ROOT / dataset_name / "reference_ledger.parquet"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(path)


def live_unresolved(dataset_name: str, sweep) -> pd.DataFrame:
    exact_ids = set(exact_ledger(dataset_name).instance_id.astype(str))
    attempts_path, _ = sweep.attempts_paths(dataset_name)
    if not attempts_path.is_file():
        return pd.DataFrame(columns=sweep.ATTEMPT_COLUMNS)
    attempts = pd.read_parquet(attempts_path)
    if not len(attempts):
        return attempts
    return attempts[
        attempts.status.isin(sweep.UNRESOLVED_STATUSES)
        & ~attempts.instance_id.astype(str).isin(exact_ids)
    ].copy()


def snapshot_exact_identities() -> dict[str, Any]:
    snapshot: dict[str, Any] = {"dataset_file_sha256": {}, "rows": {}}
    for dataset_name in NEW_DATASETS:
        path = DATA_ROOT / dataset_name / "reference_ledger.parquet"
        snapshot["dataset_file_sha256"][dataset_name] = sha256_file(path)
        frame = pd.read_parquet(path)
        snapshot["rows"][dataset_name] = {
            str(row.instance_id): {
                "coordinate_hash": str(row.coordinate_hash),
                "tour_hash": str(row.tour_hash),
                "optimal_cost_int": int(row.optimal_cost_int),
            }
            for row in frame.itertuples(index=False)
        }
    return snapshot


def verify_frozen_hashes() -> list[str]:
    failures: list[str] = []
    for dataset_name in NEW_DATASETS:
        metadata = json.loads((DATA_ROOT / dataset_name / "metadata.json").read_text(encoding="utf-8"))
        coordinate_path = DATA_ROOT / dataset_name / "coordinates.npy"
        file_sha = sha256_file(coordinate_path)
        tensor_sha = dataset_tensor_sha256(dataset_name)
        manifest_sha = sha256_file(DATA_ROOT / dataset_name / "instances.csv")
        if file_sha != metadata["coordinates_file_sha256"]:
            failures.append(f"{dataset_name}: coordinates.npy hash mismatch")
        if tensor_sha != metadata["coordinate_tensor_sha256"]:
            failures.append(f"{dataset_name}: coordinate tensor hash mismatch")
        if tensor_sha != FROZEN_DATASET_SHA256[dataset_name]:
            failures.append(f"{dataset_name}: suite-manifest tensor hash mismatch")
        if manifest_sha != metadata["instance_manifest_sha256"]:
            failures.append(f"{dataset_name}: instances.csv hash mismatch")
    return failures


def preflight(sweep) -> dict[str, Any]:
    failures: list[str] = []
    failures.extend(verify_frozen_hashes())
    for dataset_name in NEW_DATASETS:
        try:
            sweep.preflight_dataset(dataset_name)
        except Exception as exc:
            failures.append(f"{dataset_name}: {type(exc).__name__}: {exc}")
    counts = {name: int(len(exact_ledger(name))) for name in NEW_DATASETS}
    hardcase = load_hardcase_ledger()
    hardcase_by_id = {
        str(row.instance_id): row for row in hardcase.itertuples(index=False)
    } if len(hardcase) else {}
    if counts["EvoReal_S2_100"] != 1000:
        failures.append(f"EvoReal_S2_100 exact={counts['EvoReal_S2_100']} expected 1000")
    if counts["EvoReal_S3_100"] != 1000:
        failures.append(f"EvoReal_S3_100 exact={counts['EvoReal_S3_100']} expected 1000")
    if counts["GaussianMixture500"] != 1000:
        failures.append(f"GaussianMixture500 exact={counts['GaussianMixture500']} expected 1000")

    s1 = exact_ledger("EvoReal_S1_100")
    s1_ids = set(s1.instance_id.astype(str)) if len(s1) else set()
    target = "EvoReal_S1_100_0853"
    if counts["EvoReal_S1_100"] == 1000:
        if target not in s1_ids:
            failures.append("EvoReal_S1_100 has 1000 exact rows but 0853 is absent")
        record = hardcase_by_id.get(target)
        if record is None or str(record.status) != HARDCASE_EXACT_SUCCESS:
            failures.append("EvoReal_S1_100/0853 is exact without a completed HARDCASE_EXACT_SUCCESS record")
    elif counts["EvoReal_S1_100"] == 999:
        if target in s1_ids:
            failures.append("exact ledger contains EvoReal_S1_100/0853 before certification")
        unresolved = live_unresolved("EvoReal_S1_100", sweep)
        unresolved_ids = unresolved.instance_id.astype(str).tolist() if len(unresolved) else []
        if unresolved_ids != [target]:
            failures.append(f"live unresolved set is {unresolved_ids}, expected ['{target}']")
        blocker_path = AUDIT_ROOT / "12_oracle_blockers.csv"
        if blocker_path.is_file():
            blockers = pd.read_csv(blocker_path)
            blocker_ids = blockers.instance_id.astype(str).tolist() if len(blockers) else []
            if blocker_ids != [target]:
                failures.append(f"blocker manifest is {blocker_ids}, expected ['{target}']")
        else:
            failures.append("missing 12_oracle_blockers.csv")
        if len(unresolved) == 1:
            row = unresolved.iloc[0]
            if int(row.standard_attempts) != 2 or not bool(row.standard_attempts_exhausted):
                failures.append("EvoReal_S1_100_0853 standard retry budget is not exhausted")
            if str(row.status) not in TIMEOUT_UNRESOLVED:
                failures.append(f"EvoReal_S1_100_0853 status is {row.status}, not a timeout state")
            if bool(row.exact_reference_available) or bool(row.candidate_is_exact):
                failures.append("EvoReal_S1_100_0853 candidate is incorrectly marked exact")
    else:
        failures.append(f"EvoReal_S1_100 exact={counts['EvoReal_S1_100']} expected 999 or 1000")

    suite_unresolved = []
    for name in NEW_DATASETS:
        suite_unresolved.extend(live_unresolved(name, sweep).instance_id.astype(str).tolist())
    if counts["EvoReal_S1_100"] == 999 and suite_unresolved != [target]:
        failures.append(f"suite live unresolved is {suite_unresolved}, expected ['{target}']")
    if counts["EvoReal_S1_100"] == 1000 and suite_unresolved:
        failures.append(f"suite live unresolved is {suite_unresolved}, expected empty after exact close")

    return {"failures": failures, "counts": counts, "unresolved": suite_unresolved}


def is_timeout_unresolved(status: str) -> bool:
    value = str(status)
    return value in TIMEOUT_UNRESOLVED or value.startswith("TIMEOUT")


def discover_eligible(sweep) -> list[dict[str, Any]]:
    hardcase = load_hardcase_ledger()
    completed = set()
    if len(hardcase):
        completed = set(
            hardcase.loc[hardcase.status.isin(TERMINAL_HARDCASE), "instance_id"].astype(str)
        )
    eligible: list[dict[str, Any]] = []
    for dataset_name in NEW_DATASETS:
        coordinates, manifest = sweep.preflight_dataset(dataset_name)
        exact_ids = set(exact_ledger(dataset_name).instance_id.astype(str))
        attempts = sweep.load_attempts(dataset_name, manifest)
        if not len(attempts):
            continue
        by_id = {row["instance_id"]: row for row in manifest}
        for rec in attempts.itertuples(index=False):
            instance_id = str(rec.instance_id)
            if instance_id in exact_ids:
                continue
            if not bool(rec.standard_attempts_exhausted):
                continue
            if not is_timeout_unresolved(str(rec.status)):
                continue
            if instance_id in completed:
                continue
            if instance_id not in by_id:
                raise RuntimeError(f"{dataset_name}: attempt ID {instance_id} missing from frozen manifest")
            if str(rec.coordinate_hash) != by_id[instance_id]["coordinate_hash"]:
                raise RuntimeError(f"{dataset_name}: attempt/manifest hash mismatch {instance_id}")
            eligible.append({
                "dataset": dataset_name,
                "index": int(rec.index),
                "instance_id": instance_id,
                "coordinate_hash": str(rec.coordinate_hash),
                "n_nodes": int(rec.n_nodes),
                "standard_attempts": int(rec.standard_attempts),
                "standard_status": str(rec.status),
                "coordinates": coordinates[int(rec.index)],
            })
    return eligible


def invoke_canonical_concorde(
    coordinates, instance_hash: str, work: Path, backend: dict[str, Any], timeout: float,
) -> dict[str, Any]:
    """Same binary, flags, input, metric, and verification as solve_one; logs are persisted."""
    work.mkdir(parents=True, exist_ok=True)
    dist = integer_distance_matrix(coordinates, INTEGER_SCALE)
    n = len(coordinates)
    tsp = work / f"{instance_hash}.tsp"
    sol = work / f"{instance_hash}.sol"
    stdout_path = work / f"{instance_hash}.stdout.log"
    stderr_path = work / f"{instance_hash}.stderr.log"
    write_explicit_tsp(tsp, dist, instance_hash)
    if backend["invoke_mode"] == "wsl":
        inner = (
            f"cd '{windows_to_wsl_path(work)}' && "
            f"'{windows_to_wsl_path(Path(backend['binary']))}' -x -o '{sol.name}' '{tsp.name}'"
        )
        cmd = ["wsl", "-d", backend["wsl_distro"], "--", "bash", "-lc", inner]
        cwd = None
    else:
        cmd = [backend["binary"], "-x", "-o", sol.name, tsp.name]
        cwd = str(work)
    start = time.perf_counter()
    timed_out = False
    proc_returncode = None
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout,
        )
        stdout, stderr = proc.stdout or "", proc.stderr or ""
        proc_returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout, stderr = as_text(exc.stdout), as_text(exc.stderr)
    elapsed = time.perf_counter() - start
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    text = stdout + "\n" + stderr
    return {
        "timed_out": timed_out, "elapsed": elapsed, "stdout": stdout, "stderr": stderr,
        "text": text, "sol": sol, "dist": dist, "n": n, "returncode": proc_returncode,
        "stdout_path": stdout_path, "stderr_path": stderr_path, "work": work,
    }


def archive_logs(dataset_name: str, instance_id: str, work: Path, instance_hash: str) -> None:
    destination = DATA_ROOT / dataset_name / "oracle" / "hardcase" / instance_id
    destination.mkdir(parents=True, exist_ok=True)
    for suffix in (".stdout.log", ".stderr.log", ".sol", ".res", ".tsp"):
        source = work / f"{instance_hash}{suffix}"
        if source.is_file():
            shutil.copy2(source, destination / source.name)
    extra = work / f"O{instance_hash}.res"
    if extra.is_file():
        shutil.copy2(extra, destination / extra.name)


def empty_hardcase_row(item: dict[str, Any], config_hash: str, backend: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": item["dataset"], "index": item["index"], "instance_id": item["instance_id"],
        "coordinate_hash": item["coordinate_hash"], "n_nodes": item["n_nodes"],
        "status": HARDCASE_SOLVER_ERROR, "oracle_tier": ORACLE_TIER,
        "timeout_sec": HARDCASE_TIMEOUT_SEC, "tier2_attempts": 1,
        "standard_attempts": item["standard_attempts"],
        "standard_timeout_sec": STANDARD_TIMEOUT_SEC, "exact_reference_available": False,
        "hardcase_escalation_exhausted": True, "verified": False,
        "verification_result": "FAIL", "solver_status": "error",
        "exact_reference_cost_int": None, "exact_tour": "", "tour_hash": None,
        "candidate_emitted": False, "candidate_legal": False, "candidate_cost_int": None,
        "candidate_tour_hash": None, "candidate_sol_sha256": None, "candidate_tour": "",
        "runtime_sec": None, "concorde_runtime_sec": None,
        "solver_invocation_config_sha256": config_hash, "binary_hash": backend["binary_hash"],
        "integer_scale": INTEGER_SCALE, "metric": METRIC, "stdout_path": "",
        "stderr_path": "", "work_dir": "", "error": "",
    }


def attempt_hardcase(
    item: dict[str, Any], backend: dict[str, Any], config_hash: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    work = DATA_ROOT / item["dataset"] / "oracle" / "hardcase_work" / item["coordinate_hash"]
    if work.exists():
        shutil.rmtree(work)
    row = empty_hardcase_row(item, config_hash, backend)
    row["work_dir"] = work.relative_to(ROOT).as_posix()
    raw = invoke_canonical_concorde(
        item["coordinates"], item["coordinate_hash"], work, backend, HARDCASE_TIMEOUT_SEC,
    )
    archive_logs(item["dataset"], item["instance_id"], work, item["coordinate_hash"])
    row["runtime_sec"] = float(raw["elapsed"])
    row["concorde_runtime_sec"] = float(raw["elapsed"])
    row["stdout_path"] = raw["stdout_path"].relative_to(ROOT).as_posix()
    row["stderr_path"] = raw["stderr_path"].relative_to(ROOT).as_posix()
    sol: Path = raw["sol"]
    candidate_emitted = sol.is_file() and sol.stat().st_size > 0
    row["candidate_emitted"] = candidate_emitted
    if candidate_emitted:
        row["candidate_sol_sha256"] = sha256_file(sol)
        try:
            tour = _parse_tour(sol, raw["n"])
            legal, message = validate_tour_0based(tour, raw["n"])
            row["candidate_legal"] = legal
            if legal:
                cost = tour_cost_int(raw["dist"], tour)
                row["candidate_cost_int"] = int(cost)
                row["candidate_tour_hash"] = tour_hash(tour)
                row["candidate_tour"] = " ".join(map(str, tour))
            else:
                row["error"] = message
        except (OSError, ValueError) as exc:
            row["candidate_legal"] = False
            row["error"] = f"{type(exc).__name__}: {exc}"

    if raw["timed_out"]:
        row["solver_status"] = "timeout"
        row["status"] = (
            HARDCASE_TIMEOUT_WITH_CANDIDATE if row["candidate_legal"]
            else HARDCASE_TIMEOUT_NO_CANDIDATE
        )
        row["error"] = (
            row["error"] or f"subprocess.TimeoutExpired after {HARDCASE_TIMEOUT_SEC} seconds"
        )
        row["hardcase_escalation_exhausted"] = True
        row["exact_reference_available"] = False
        return row, None

    if not candidate_emitted:
        row["status"] = HARDCASE_SOLVER_ERROR
        row["solver_status"] = "error"
        row["error"] = (
            f"Concorde {item['coordinate_hash']} exit={raw['returncode']}: {raw['text'][-500:]}"
        )
        return row, None

    if not row["candidate_legal"]:
        row["status"] = HARDCASE_VERIFICATION_FAILURE
        row["solver_status"] = "verification_failure"
        row["verification_result"] = "FAIL"
        return row, None

    stated = _reported(raw["text"])
    cost = int(row["candidate_cost_int"])
    verified = bool(row["candidate_legal"]) and (stated is None or stated == cost)
    if not verified:
        row["status"] = HARDCASE_VERIFICATION_FAILURE
        row["solver_status"] = "verification_failure"
        row["verification_result"] = "FAIL"
        row["error"] = f"Concorde verification failed: stated={stated}, cost={cost}"
        return row, None

    tour_text = str(row["candidate_tour"])
    row["status"] = HARDCASE_EXACT_SUCCESS
    row["solver_status"] = "solved"
    row["verified"] = True
    row["verification_result"] = "PASS"
    row["exact_reference_available"] = True
    row["hardcase_escalation_exhausted"] = False
    row["exact_reference_cost_int"] = cost
    row["exact_tour"] = tour_text
    row["tour_hash"] = str(row["candidate_tour_hash"])
    row["error"] = ""
    exact = {
        "dataset": item["dataset"], "index": item["index"],
        "instance_id": item["instance_id"], "coordinate_hash": item["coordinate_hash"],
        "instance_hash": item["coordinate_hash"], "n_nodes": item["n_nodes"],
        "status": "solved", "verified": True, "optimal_cost_int": cost,
        "exact_reference_cost_int": cost, "runtime_sec": float(raw["elapsed"]),
        "concorde_runtime_sec": float(raw["elapsed"]), "expert_tour": tour_text,
        "exact_tour": tour_text, "tour_hash": str(row["candidate_tour_hash"]),
        "binary_hash": backend["binary_hash"],
        "solver_invocation_config_sha256": config_hash, "integer_scale": INTEGER_SCALE,
        "metric": METRIC, "verification_result": "PASS",
        "oracle_tier": ORACLE_TIER, "timeout_sec": HARDCASE_TIMEOUT_SEC,
    }
    return row, exact


def commit_exact_success(sweep, dataset_name: str, exact_row: dict[str, Any]) -> None:
    coordinates, manifest = sweep.preflight_dataset(dataset_name)
    verify_exact_row(dataset_name, coordinates[int(exact_row["index"])], exact_row)
    exact = exact_ledger(dataset_name)
    if exact_row["instance_id"] in set(exact.instance_id.astype(str)):
        raise RuntimeError(f"{dataset_name}: refusing to duplicate exact row {exact_row['instance_id']}")
    if "oracle_tier" not in exact.columns:
        exact = exact.copy()
        exact["oracle_tier"] = "STANDARD"
    if "timeout_sec" not in exact.columns:
        exact = exact.copy()
        exact["timeout_sec"] = STANDARD_TIMEOUT_SEC
    exact = pd.concat([exact, pd.DataFrame([exact_row])], ignore_index=True)
    if exact.instance_id.astype(str).duplicated().any() or exact.coordinate_hash.astype(str).duplicated().any():
        raise RuntimeError(f"{dataset_name}: duplicate exact keys after hard-case commit")
    sweep.persist_exact(dataset_name, exact)
    attempts = sweep.load_attempts(dataset_name, manifest)
    exact = exact_ledger(dataset_name)
    sweep.write_unresolved(dataset_name, attempts, exact)
    sweep.write_sweep_state(dataset_name, manifest, exact, attempts)


def coverage() -> dict[str, dict[str, int]]:
    sweep = load_sweep_module()
    result = {}
    for name in NEW_DATASETS:
        exact_n = int(len(exact_ledger(name)))
        unresolved_n = int(len(live_unresolved(name, sweep)))
        result[name] = {"exact": exact_n, "unresolved": unresolved_n}
    result["TOTAL"] = {
        "exact": sum(result[name]["exact"] for name in NEW_DATASETS),
        "unresolved": sum(result[name]["unresolved"] for name in NEW_DATASETS),
    }
    return result


def update_suite_status(status: str, details: dict[str, Any]) -> None:
    sweep = load_sweep_module()
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {"status": status, **details}
    sweep.atomic_json(AUDIT_ROOT / "ORACLE_HARDCASE_ESCALATION_STATUS.json", payload)
    strict_path = AUDIT_ROOT / "STRICT_VERIFICATION.json"
    strict = json.loads(strict_path.read_text(encoding="utf-8")) if strict_path.is_file() else {}
    cov = coverage()
    strict["status"] = status
    strict["hard_case_references_pending"] = details.get("unresolved", [])
    strict["blocking_causes"] = details.get("integrity_failures", [])
    strict["oracle_hardcase_escalation"] = payload
    oracle_status = strict.setdefault("oracle_status", {})
    s1 = oracle_status.setdefault("EvoReal_S1_100", {})
    s1["count"] = cov["EvoReal_S1_100"]["exact"]
    s1["verified_count"] = cov["EvoReal_S1_100"]["exact"]
    s1["complete"] = cov["EvoReal_S1_100"]["exact"] == 1000 and cov["EvoReal_S1_100"]["unresolved"] == 0
    ledger_path = DATA_ROOT / "EvoReal_S1_100" / "reference_ledger.parquet"
    if ledger_path.is_file():
        s1["ledger_sha256"] = sha256_file(ledger_path)
    if status == STATUS_BLOCKED:
        s1["integrity_failure"] = True
    else:
        s1["integrity_failure"] = False
        if cov["EvoReal_S1_100"]["exact"] == 1000:
            s1.pop("failure", None)
    sweep.atomic_json(strict_path, strict)


def write_policy() -> None:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    (AUDIT_ROOT / "15_hardcase_escalation_policy.md").write_text(POLICY_TEXT, encoding="utf-8")


def write_integrity_report(cov: dict[str, dict[str, int]], hardcase: pd.DataFrame) -> None:
    lines = [
        "# Oracle integrity",
        "",
        "Canonical explicit integer-distance Concorde only; no LKH or heuristic fallback.",
        "",
        "An isolated unresolved exact-reference hard case is not an integrity failure.",
        "",
    ]
    for name in NEW_DATASETS:
        complete = cov[name]["exact"] == 1000 and cov[name]["unresolved"] == 0
        lines.append(
            f"- {name}: {cov[name]['exact']}/1000 cached; complete={complete}; "
            "stable binding by instance_id + coordinate_hash."
        )
        if name == "EvoReal_S1_100" and cov[name]["unresolved"]:
            lines.append(
                "  - Isolated exact-reference hard case: EvoReal_S1_100_0853 remains without an "
                "optimality certificate after standard 300 s attempts and one 3600 s Tier-2 attempt. "
                "This is not an integrity/protocol failure."
            )
        if name == "EvoReal_S1_100" and len(hardcase):
            row = hardcase[hardcase.instance_id.astype(str) == "EvoReal_S1_100_0853"]
            if len(row) and str(row.iloc[0].status) == HARDCASE_EXACT_SUCCESS:
                lines.append(
                    "  - EvoReal_S1_100_0853 admitted as exact via oracle_tier=HARD_CASE_ESCALATION "
                    f"(timeout_sec=3600, runtime_sec={row.iloc[0].runtime_sec})."
                )
    (AUDIT_ROOT / "05_oracle_integrity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reports(hardcase: pd.DataFrame, details: list[dict[str, Any]], final_status: str) -> None:
    sweep = load_sweep_module()
    write_policy()
    cov = coverage()
    sweep.atomic_frame(AUDIT_ROOT / "16_hardcase_escalation_results.csv", hardcase if len(hardcase) else pd.DataFrame(columns=HARDCASE_COLUMNS))
    sweep.atomic_json(AUDIT_ROOT / "17_hardcase_escalation_details.json", details)
    write_integrity_report(cov, hardcase)

    s1_row = None
    if len(hardcase):
        match = hardcase[hardcase.instance_id.astype(str) == "EvoReal_S1_100_0853"]
        if len(match):
            s1_row = match.iloc[0]
    cert = "NOT AVAILABLE"
    availability = False
    candidate_note = "n/a"
    runtime_note = "n/a"
    if s1_row is not None:
        availability = bool(s1_row.exact_reference_available)
        cert = "VERIFIED" if availability else f"NOT AVAILABLE ({s1_row.status})"
        runtime_note = str(s1_row.runtime_sec)
        if pd.notna(s1_row.candidate_cost_int):
            candidate_note = str(int(s1_row.candidate_cost_int))
        if availability and pd.notna(s1_row.exact_reference_cost_int):
            candidate_note = str(int(s1_row.exact_reference_cost_int))
    lines = [
        "# Final oracle reference status",
        "",
        "EvoReal_S1_100:",
        f"exact = {cov['EvoReal_S1_100']['exact']}/1000",
        f"unresolved = {cov['EvoReal_S1_100']['unresolved']}",
        "",
        "EvoReal_S2_100:",
        "exact = 1000/1000",
        "unresolved = 0",
        "",
        "EvoReal_S3_100:",
        "exact = 1000/1000",
        "unresolved = 0",
        "",
        "GaussianMixture500:",
        "exact = 1000/1000",
        "unresolved = 0",
        "",
        "TOTAL:",
        f"exact = {cov['TOTAL']['exact']}/4000",
        f"unresolved = {cov['TOTAL']['unresolved']}",
        "",
        "EvoReal_S1_100_0853:",
        "- standard attempts = 2",
        "- standard timeout = 300 s each",
        "- Tier-2 attempts = 1",
        "- Tier-2 timeout budget = 3600 s",
        f"- exact certification result = {cert}",
        f"- final exact availability = {availability}",
        f"- hard-case runtime_sec = {runtime_note}",
        f"- candidate/exact cost_int = {candidate_note}",
        "",
        f"Status: `{final_status}`",
        "",
        "No automatic instance removal, replacement, heuristic substitution, or further retry was performed.",
    ]
    (AUDIT_ROOT / "18_final_oracle_reference_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_exact_row(dataset_name: str, coordinates, item: dict[str, Any]) -> None:
    tour = [int(x) for x in str(item["expert_tour"]).split()]
    legal, message = validate_tour_0based(tour, int(item["n_nodes"]))
    if not legal:
        raise RuntimeError(f"{dataset_name}/{item['instance_id']}: illegal exact tour: {message}")
    computed = tour_cost_int(integer_distance_matrix(coordinates), tour)
    if computed != int(item["optimal_cost_int"]) or computed != int(item["exact_reference_cost_int"]):
        raise RuntimeError(f"{dataset_name}/{item['instance_id']}: exact cost mismatch")
    if tour_hash(tour) != str(item["tour_hash"]):
        raise RuntimeError(f"{dataset_name}/{item['instance_id']}: tour hash mismatch")
    if str(item["binary_hash"]) != CANONICAL_CONCORDE_SHA256:
        raise RuntimeError(f"{dataset_name}/{item['instance_id']}: solver hash mismatch")
    if not bool(item["verified"]) or str(item["status"]) != "solved":
        raise RuntimeError(f"{dataset_name}/{item['instance_id']}: non-exact row in exact ledger")
    if int(item["integer_scale"]) != INTEGER_SCALE or str(item["metric"]) != METRIC:
        raise RuntimeError(f"{dataset_name}/{item['instance_id']}: metric/scale mismatch")


def strict_post_verification(snapshot: dict[str, Any], launched: list[str]) -> list[str]:
    failures = verify_frozen_hashes()
    sweep = load_sweep_module()
    hardcase = load_hardcase_ledger()
    for dataset_name in NEW_DATASETS:
        path = DATA_ROOT / dataset_name / "reference_ledger.parquet"
        current_sha = sha256_file(path)
        frame = pd.read_parquet(path)
        if frame.instance_id.astype(str).duplicated().any():
            failures.append(f"{dataset_name}: duplicate instance_id in exact ledger")
        if frame.coordinate_hash.astype(str).duplicated().any():
            failures.append(f"{dataset_name}: duplicate coordinate_hash in exact ledger")
        previous = snapshot["rows"][dataset_name]
        for instance_id, fields in previous.items():
            match = frame[frame.instance_id.astype(str) == instance_id]
            if len(match) != 1:
                failures.append(f"{dataset_name}: previously exact {instance_id} missing after escalation")
                continue
            row = match.iloc[0]
            if str(row.coordinate_hash) != fields["coordinate_hash"]:
                failures.append(f"{dataset_name}/{instance_id}: coordinate_hash changed")
            if str(row.tour_hash) != fields["tour_hash"]:
                failures.append(f"{dataset_name}/{instance_id}: tour_hash changed")
            if int(row.optimal_cost_int) != fields["optimal_cost_int"]:
                failures.append(f"{dataset_name}/{instance_id}: cost changed")
        if dataset_name != "EvoReal_S1_100" and current_sha != snapshot["dataset_file_sha256"][dataset_name]:
            failures.append(f"{dataset_name}: exact ledger file hash changed")
        coordinates, _ = sweep.preflight_dataset(dataset_name)
        for item in frame.to_dict("records"):
            verify_exact_row(dataset_name, coordinates[int(item["index"])], item)
            if str(item["instance_id"]) == "EvoReal_S1_100_0853":
                if str(item.get("oracle_tier", "")) != ORACLE_TIER:
                    failures.append("0853 exact row missing HARD_CASE_ESCALATION tier")
                if float(item.get("timeout_sec", 0)) != HARDCASE_TIMEOUT_SEC:
                    failures.append("0853 exact row timeout_sec is not 3600")
    s1 = exact_ledger("EvoReal_S1_100")
    s1_ids = set(s1.instance_id.astype(str)) if len(s1) else set()
    if "EvoReal_S1_100_0853" in s1_ids:
        if not len(hardcase) or str(hardcase.iloc[0].status) != HARDCASE_EXACT_SUCCESS:
            failures.append("0853 entered exact ledger without HARDCASE_EXACT_SUCCESS")
    else:
        if len(hardcase) and str(hardcase.iloc[0].status) == HARDCASE_EXACT_SUCCESS:
            failures.append("HARDCASE_EXACT_SUCCESS did not enter the exact ledger")
    if len(hardcase):
        if float(hardcase.iloc[0].timeout_sec) != HARDCASE_TIMEOUT_SEC:
            failures.append("Tier-2 timeout_sec is not 3600")
        if int(hardcase.iloc[0].tier2_attempts) != 1:
            failures.append("Tier-2 attempt count is not 1")
        if len(hardcase) != 1:
            failures.append(f"hard-case ledger has {len(hardcase)} rows; expected at most one completed attempt")
        if str(hardcase.iloc[0].binary_hash) != CANONICAL_CONCORDE_SHA256:
            failures.append("hard-case binary hash mismatch")
        if int(hardcase.iloc[0].integer_scale) != INTEGER_SCALE or str(hardcase.iloc[0].metric) != METRIC:
            failures.append("hard-case metric/scale mismatch")
    if len(launched) > 1:
        failures.append(f"launched {len(launched)} Tier-2 jobs; expected at most one")
    return failures


def print_human(final_status: str, hardcase: pd.DataFrame) -> None:
    cov = coverage()
    print("HARD-CASE ESCALATION COMPLETE", flush=True)
    print("", flush=True)
    print("EvoReal_S1_100_0853:", flush=True)
    print("standard attempts: 2 x 300 s timeout", flush=True)
    if len(hardcase):
        row = hardcase.iloc[0]
        if str(row.status) == HARDCASE_EXACT_SUCCESS:
            print("hard-case attempt: SUCCESS", flush=True)
            print("exact reference: VERIFIED", flush=True)
            print(f"hard-case runtime: {row.runtime_sec}", flush=True)
            print(f"exact cost: {int(row.exact_reference_cost_int)}", flush=True)
        else:
            label = {
                HARDCASE_TIMEOUT_WITH_CANDIDATE: "TIMEOUT",
                HARDCASE_TIMEOUT_NO_CANDIDATE: "TIMEOUT",
                HARDCASE_SOLVER_ERROR: "ERROR",
                HARDCASE_VERIFICATION_FAILURE: "VERIFICATION FAILURE",
            }.get(str(row.status), str(row.status))
            print(f"hard-case attempt: {label}", flush=True)
            print("exact reference: NOT AVAILABLE", flush=True)
    print("", flush=True)
    if cov["TOTAL"]["exact"] == 4000:
        print("FINAL ORACLE COVERAGE:", flush=True)
        print(f"EvoReal S1: {cov['EvoReal_S1_100']['exact']}/1000", flush=True)
        print(f"EvoReal S2: {cov['EvoReal_S2_100']['exact']}/1000", flush=True)
        print(f"EvoReal S3: {cov['EvoReal_S3_100']['exact']}/1000", flush=True)
        print(f"GM500: {cov['GaussianMixture500']['exact']}/1000", flush=True)
        print("TOTAL: 4000/4000", flush=True)
    else:
        print("FINAL ORACLE COVERAGE:", flush=True)
        print(f"{cov['TOTAL']['exact']}/4000", flush=True)
    print("", flush=True)
    print("STATUS:", flush=True)
    print(final_status, flush=True)
    if final_status == STATUS_ONE_UNRESOLVED:
        print("", flush=True)
        print("STOP.", flush=True)
        print("Do not take any further scientific action automatically.", flush=True)


def details_from_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for row in frame.to_dict("records"):
        records.append({key: jsonable(value) for key, value in row.items()})
    return records


def final_status_from_coverage() -> str:
    cov = coverage()
    if cov["TOTAL"]["unresolved"] == 0 and cov["TOTAL"]["exact"] == 4000:
        return STATUS_READY
    hardcase = load_hardcase_ledger()
    if len(hardcase) and bool(hardcase.hardcase_escalation_exhausted.any()) and cov["TOTAL"]["unresolved"] == 1:
        return STATUS_ONE_UNRESOLVED
    if cov["TOTAL"]["unresolved"] > 0:
        return STATUS_ONE_UNRESOLVED if len(hardcase) else "EXTERNAL_BENCHMARK_SUITE_V2_PENDING_HARD_CASE_REFERENCES"
    return STATUS_READY


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    write_policy()
    source_and_config_hashes()
    sweep = load_sweep_module()
    backend = resolve_backend()
    if backend["binary_hash"] != CANONICAL_CONCORDE_SHA256 or backend["invoke_mode"] != "wsl":
        update_suite_status(STATUS_BLOCKED, {
            "integrity_failures": ["canonical Concorde backend/invocation mismatch"],
            "unresolved": [],
        })
        print("INTEGRITY MISMATCH: canonical Concorde backend/invocation mismatch", flush=True)
        return 2
    standard_config, standard_hash = solver_config(backend, STANDARD_TIMEOUT_SEC)
    hardcase_config, hardcase_hash = solver_config(backend, HARDCASE_TIMEOUT_SEC)
    if standard_config["timeout_seconds"] != STANDARD_TIMEOUT_SEC:
        raise RuntimeError("standard timeout drifted")
    mutated = {k: v for k, v in hardcase_config.items() if k != "timeout_seconds"}
    baseline = {k: v for k, v in standard_config.items() if k != "timeout_seconds"}
    if mutated != baseline:
        update_suite_status(STATUS_BLOCKED, {
            "integrity_failures": ["hard-case solver config changed a non-timeout field"],
            "unresolved": [],
        })
        print("INTEGRITY MISMATCH: non-timeout solver parameter changed", flush=True)
        return 2
    if hardcase_config["timeout_seconds"] != HARDCASE_TIMEOUT_SEC:
        update_suite_status(STATUS_BLOCKED, {
            "integrity_failures": ["Tier-2 timeout is not 3600 s"],
            "unresolved": [],
        })
        print("INTEGRITY MISMATCH: Tier-2 timeout is not 3600 s", flush=True)
        return 2
    sweep.atomic_json(AUDIT_ROOT / "ORACLE_HARDCASE_ESCALATION_CONFIG.json", hardcase_config)

    pre = preflight(sweep)
    if pre["failures"]:
        update_suite_status(STATUS_BLOCKED, {
            "integrity_failures": pre["failures"], "unresolved": pre["unresolved"],
        })
        print("INTEGRITY MISMATCH", flush=True)
        for item in pre["failures"]:
            print(item, flush=True)
        return 2

    snapshot = snapshot_exact_identities()
    sweep.atomic_json(DATA_ROOT / "oracle" / "hardcase_preflight_exact_snapshot.json", {
        "dataset_file_sha256": snapshot["dataset_file_sha256"],
        "counts": {name: len(rows) for name, rows in snapshot["rows"].items()},
        "standard_solver_invocation_config_sha256": standard_hash,
        "hardcase_solver_invocation_config_sha256": hardcase_hash,
    })
    eligible = discover_eligible(sweep)
    print(f"eligible_hardcase_instances={len(eligible)}", flush=True)
    for item in eligible:
        print(f"ELIGIBLE {item['dataset']}/{item['instance_id']} status={item['standard_status']}", flush=True)
    existing_hardcase = load_hardcase_ledger()
    already_completed = set(existing_hardcase.instance_id.astype(str)) if len(existing_hardcase) else set()
    if (
        pre["counts"]["EvoReal_S1_100"] == 999
        and not eligible
        and "EvoReal_S1_100_0853" not in already_completed
    ):
        update_suite_status(STATUS_BLOCKED, {
            "integrity_failures": [
                "generic discovery found no eligible instance while 0853 remains unresolved without a Tier-2 record"
            ],
            "unresolved": pre["unresolved"],
        })
        print("INTEGRITY MISMATCH: no eligible hard-case instance discovered", flush=True)
        return 2
    if args.validate_only:
        print(json.dumps({
            "status": "VALIDATED", "eligible": [
                {k: v for k, v in item.items() if k != "coordinates"} for item in eligible
            ],
            "preflight_counts": pre["counts"],
        }, indent=2), flush=True)
        return 0

    hardcase = load_hardcase_ledger()
    launched: list[str] = []
    details: list[dict[str, Any]] = details_from_frame(hardcase)
    if eligible:
        update_suite_status(STATUS_IN_PROGRESS, {
            "integrity_failures": [],
            "unresolved": [item["instance_id"] for item in eligible],
            "eligible_count": len(eligible),
            "timeout_sec": HARDCASE_TIMEOUT_SEC,
            "solver_invocation_config_sha256": hardcase_hash,
        })
    for item in eligible:
        print(
            f"LAUNCH TIER2 {item['dataset']}/{item['instance_id']} timeout={HARDCASE_TIMEOUT_SEC}",
            flush=True,
        )
        launched.append(item["instance_id"])
        attempt, exact_row = attempt_hardcase(item, backend, hardcase_hash)
        if len(hardcase):
            hardcase = pd.concat(
                [hardcase, pd.DataFrame([attempt], columns=HARDCASE_COLUMNS)], ignore_index=True,
            )
        else:
            hardcase = pd.DataFrame([attempt], columns=HARDCASE_COLUMNS)
        if hardcase.instance_id.astype(str).duplicated().any():
            raise RuntimeError(f"duplicate hard-case commit {attempt['instance_id']}")
        persist_hardcase_ledger(hardcase)
        if exact_row is not None:
            commit_exact_success(sweep, item["dataset"], exact_row)
        details = details_from_frame(hardcase)
        print(f"TIER2 RESULT {item['dataset']}/{item['instance_id']} {attempt['status']}", flush=True)

    post_failures = strict_post_verification(snapshot, launched)
    if post_failures:
        update_suite_status(STATUS_BLOCKED, {
            "integrity_failures": post_failures,
            "unresolved": live_unresolved("EvoReal_S1_100", sweep).instance_id.astype(str).tolist(),
            "traceback": "",
        })
        write_reports(hardcase, details, STATUS_BLOCKED)
        print("INTEGRITY MISMATCH AFTER ESCALATION", flush=True)
        for item in post_failures:
            print(item, flush=True)
        return 2

    status = final_status_from_coverage()
    cov = coverage()
    launched_ids = launched or (
        hardcase.instance_id.astype(str).tolist() if len(hardcase) else []
    )
    update_suite_status(status, {
        "integrity_failures": [],
        "unresolved": live_unresolved("EvoReal_S1_100", sweep).instance_id.astype(str).tolist()
        + live_unresolved("EvoReal_S2_100", sweep).instance_id.astype(str).tolist()
        + live_unresolved("EvoReal_S3_100", sweep).instance_id.astype(str).tolist()
        + live_unresolved("GaussianMixture500", sweep).instance_id.astype(str).tolist(),
        "coverage": cov,
        "launched": launched_ids,
        "timeout_sec": HARDCASE_TIMEOUT_SEC,
        "solver_invocation_config_sha256": hardcase_hash,
        "standard_solver_invocation_config_sha256": standard_hash,
    })
    write_reports(hardcase, details, status)
    print_human(status, hardcase)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        try:
            update_suite_status(STATUS_BLOCKED, {
                "integrity_failures": [f"{type(exc).__name__}: {exc}"],
                "traceback": traceback.format_exc(), "unresolved": [],
            })
        except Exception:
            pass
        raise
