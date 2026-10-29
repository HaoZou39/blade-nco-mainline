#!/usr/bin/env python3
"""Fail-closed, persistent recovery for one approved multi-seed A4 blocker.

The normal A4 labeler intentionally uses the historical implementation, whose
temporary work directory is removed after every call.  This recovery entry is
separate so that an explicitly approved exceptional solve can preserve its
Concorde master/restart state without changing the frozen training protocol.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from evaluation.external_v2 import coordinate_hash  # noqa: E402
from final_run.stage import atomic_json  # noqa: E402
from oracle import concorde as concorde_impl  # noqa: E402


RUN = ROOT / "runs" / "BLADE_FULL_PIPELINE_MULTISEED_V1"
CONFIRMATION = "I_APPROVE_THIS_SINGLE_INSTANCE_EXPANDED_EXACT_BUDGET"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def context(seed: int) -> dict[str, Any]:
    seed_root = RUN / f"seed_{seed}"
    blocker_path = seed_root / "BLOCKER.json"
    if not blocker_path.is_file():
        raise RuntimeError(f"missing fail-closed blocker: {blocker_path}")
    blocker = read_json(blocker_path)
    instance_hash = str(blocker["blocked_instance_hash"])
    index = int(blocker["blocked_instance_index_zero_based"])
    dataset = seed_root / "A4" / "datasets" / "evoreal50k_coordinates.npy"
    coordinates = np.load(dataset, mmap_mode="r", allow_pickle=False)
    if coordinates.shape != (50_000, 100, 2):
        raise RuntimeError(f"unexpected frozen A4 shape: {coordinates.shape}")
    xy = np.asarray(coordinates[index], dtype=np.float64)
    if coordinate_hash(xy) != instance_hash:
        raise RuntimeError("blocker hash/index does not match the frozen dataset")
    oracle = seed_root / "A4" / "A4_EVOREAL50K" / "oracle"
    record = oracle / "instance_records" / instance_hash[:2] / f"{instance_hash}.json"
    return {
        "seed_root": seed_root,
        "blocker_path": blocker_path,
        "blocker": blocker,
        "instance_hash": instance_hash,
        "index": index,
        "xy": xy,
        "oracle": oracle,
        "record": record,
    }


def inspect(seed: int) -> dict[str, Any]:
    ctx = context(seed)
    records = list((ctx["oracle"] / "instance_records").glob("*/*.json"))
    backend = concorde_impl.resolve_backend()
    result = {
        "status": "READY_FOR_EXPLICIT_APPROVAL" if not ctx["record"].is_file() else "ALREADY_LABELED",
        "seed": seed,
        "instance_hash": ctx["instance_hash"],
        "instance_index_zero_based": ctx["index"],
        "persisted_exact_record_count": len(records),
        "target_record_exists": ctx["record"].is_file(),
        "prior_consumed_seconds": float(ctx["blocker"]["attempt_timeout_seconds"]),
        "concorde_binary": backend["binary"],
        "concorde_binary_sha256": backend["binary_hash"],
        "required_confirmation": CONFIRMATION,
        "scientific_protocol_changed": False,
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def run_captured(command: list[str], directory: Path, timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    timed_out = False
    try:
        process = subprocess.run(
            command, cwd=directory, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
        stdout, stderr, returncode = process.stdout or "", process.stderr or "", process.returncode
    except subprocess.TimeoutExpired as error:
        timed_out = True
        stdout = error.stdout.decode("utf-8", "replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
        stderr = error.stderr.decode("utf-8", "replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
        returncode = None
    elapsed = time.perf_counter() - started
    write_text(directory / "stdout.txt", stdout)
    write_text(directory / "stderr.txt", stderr)
    return {
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_seconds": elapsed,
        "stdout": stdout,
        "stderr": stderr,
    }


def solve(seed: int, cumulative_timeout: float, confirmation: str) -> dict[str, Any]:
    if confirmation != CONFIRMATION:
        raise RuntimeError("expanded-budget confirmation does not match the required literal")
    ctx = context(seed)
    blocker = ctx["blocker"]
    prior = float(blocker["attempt_timeout_seconds"])
    if cumulative_timeout <= prior:
        raise RuntimeError("approved cumulative timeout must exceed the consumed 7200 seconds")
    remaining = cumulative_timeout - prior
    if ctx["record"].is_file():
        raise RuntimeError("target already has a persisted exact label; refusing to solve again")

    directory = ctx["oracle"] / "exact_recovery" / ctx["instance_hash"] / "attempt_01"
    result_path = directory / "RESULT.json"
    if result_path.is_file():
        result = read_json(result_path)
        if bool(result.get("verified")) and ctx["record"].is_file():
            print(json.dumps(result, indent=2), flush=True)
            return result
        raise RuntimeError("a completed unverified recovery attempt already exists; refusing another attempt")
    if directory.exists():
        raise RuntimeError(f"recovery directory already exists; refusing overwrite: {directory}")
    directory.mkdir(parents=True)

    xy = ctx["xy"]
    distances = concorde_impl.integer_distance_matrix(xy, concorde_impl.INTEGER_SCALE)
    tsp = directory / "instance.tsp"
    solution = directory / "optimal.sol"
    concorde_impl.write_explicit_tsp(tsp, distances, ctx["instance_hash"])
    coordinates_sha256 = hashlib.sha256(np.ascontiguousarray(xy).tobytes()).hexdigest()
    atomic_json(directory / "NODE_MAPPING.json", {
        "instance_hash": ctx["instance_hash"],
        "mapping": "identity",
        "input_coordinate_index_base": 0,
        "tsplib_node_index_base": 0,
        "input_index_to_tsplib_node": list(range(len(xy))),
        "distance_definition": "round(euclidean_distance * 1000000), numpy.rint",
        "integer_scale": concorde_impl.INTEGER_SCALE,
        "coordinates_sha256": coordinates_sha256,
    })
    backend = concorde_impl.resolve_backend()
    command = [backend["binary"], "-o", solution.name, tsp.name]
    atomic_json(directory / "ATTEMPT.json", {
        "seed": seed,
        "instance_hash": ctx["instance_hash"],
        "mode": "approved_independent_fresh_exact_solve_from_frozen_coordinates",
        "prior_attempt_state_available": False,
        "prior_consumed_seconds": prior,
        "approved_cumulative_timeout_seconds": cumulative_timeout,
        "current_attempt_timeout_seconds": remaining,
        "command": command,
        "preserve_concorde_state": True,
        "delete_state_flag_x_used": False,
        "concorde_binary_sha256": backend["binary_hash"],
        "started_utc": utc_now(),
    })
    execution = run_captured(command, directory, remaining)
    combined = execution["stdout"] + "\n" + execution["stderr"]
    reported = concorde_impl._reported(combined)
    validation: dict[str, Any] = {"verified": False, "reason": "optimal solution missing"}
    tour: list[int] | None = None
    if solution.is_file():
        try:
            tour = concorde_impl._parse_tour(solution, len(xy))
            legal, message = concorde_impl.validate_tour_0based(tour, len(xy))
            cost = concorde_impl.tour_cost_int(distances, tour)
            consistent = reported is not None and int(cost) == int(reported)
            validation = {
                "verified": bool(legal and consistent and not execution["timed_out"] and execution["returncode"] == 0),
                "hamiltonian": bool(legal),
                "validation_message": message,
                "recomputed_tour_cost_int": int(cost),
                "proven_optimal_objective_int": reported,
                "proof_tour_consistency": bool(consistent),
                "tour_hash": concorde_impl.tour_hash(tour),
            }
        except BaseException as error:
            validation = {"verified": False, "reason": repr(error)}

    cumulative = prior + float(execution["wall_seconds"])
    state_files = sorted(path.name for path in directory.iterdir() if path.is_file())
    result = {
        "status": "VERIFIED_OPTIMAL" if validation["verified"] else "FAIL_CLOSED",
        "seed": seed,
        "instance_hash": ctx["instance_hash"],
        "solver_wall_seconds": execution["wall_seconds"],
        "prior_consumed_seconds": prior,
        "cumulative_solver_wall_seconds": cumulative,
        "approved_cumulative_timeout_seconds": cumulative_timeout,
        "timed_out": execution["timed_out"],
        "returncode": execution["returncode"],
        "state_files": state_files,
        "master_present": (directory / "instance.mas").is_file(),
        "restart_present": (directory / "instance.res").is_file(),
        "node_mapping_present": (directory / "NODE_MAPPING.json").is_file(),
        **validation,
        "completed_utc": utc_now(),
    }
    atomic_json(result_path, result)
    if not bool(validation["verified"]) or tour is None:
        atomic_json(ctx["seed_root"] / "BLOCKER.json", {
            **blocker,
            "status": "BLOCKED_FAIL_CLOSED_AFTER_APPROVED_RECOVERY",
            "recovery_result": str(result_path),
            "cumulative_solver_wall_seconds": cumulative,
            "timestamp_utc": utc_now(),
        })
        print(json.dumps(result, indent=2), flush=True)
        return result

    row = {
        "instance_hash": ctx["instance_hash"],
        "status": "solved",
        "verified": True,
        "optimal_cost_int": int(validation["recomputed_tour_cost_int"]),
        "runtime_sec": cumulative,
        "expert_tour": " ".join(map(str, tour)),
        "tour_hash": validation["tour_hash"],
        "binary_hash": backend["binary_hash"],
    }
    atomic_json(ctx["record"], row)
    atomic_json(ctx["seed_root"] / "BLOCKER.json", {
        **blocker,
        "status": "RECOVERED_VERIFIED_OPTIMAL",
        "valid_exact_labels": 50_000,
        "recovery_result": str(result_path),
        "cumulative_solver_wall_seconds": cumulative,
        "timestamp_utc": utc_now(),
    })
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect_parser = sub.add_parser("inspect")
    inspect_parser.add_argument("--seed", type=int, required=True)
    solve_parser = sub.add_parser("solve")
    solve_parser.add_argument("--seed", type=int, required=True)
    solve_parser.add_argument("--approved-cumulative-timeout-seconds", type=float, required=True)
    solve_parser.add_argument("--confirm-expanded-budget", required=True)
    args = parser.parse_args()
    if args.command == "inspect":
        inspect(args.seed)
    else:
        result = solve(args.seed, args.approved_cumulative_timeout_seconds, args.confirm_expanded_budget)
        if not bool(result.get("verified")):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
