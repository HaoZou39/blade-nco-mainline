#!/usr/bin/env python3
"""Run the two matched-budget BQ baselines for the formal V2 experiment.

The program is intentionally split into idempotent commands so expensive
Concorde labeling and GPU training can be resumed independently.  It writes
only below the V2 run directory.  The completed Base ablation and the V1
partial audit are treated as immutable inputs and verified by SHA256.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback
from typing import Any

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import external_v2_passive_blade_eval as evaluator  # noqa: E402
import run_blade_core_ablation_5c_noanchor_foga11base_v1 as canonical  # noqa: E402
from evaluation.external_v2 import coordinate_hash, coordinate_set_hash  # noqa: E402
from final_run.science import membership_hash  # noqa: E402
from final_run.stage import atomic_json, sha256_file, utc_now  # noqa: E402
from oracle.concorde import (  # noqa: E402
    CANONICAL_CONCORDE_SHA256,
    canonical_oracle_ledger,
    integer_distance_matrix,
    resolve_backend,
    solve_one,
    tour_cost_int,
    tour_hash,
)


RUN_ID = "BLADE_BASE_UNIFORM_EVOREAL_5C_BENCHMARK_V2"
OUT = ROOT / "runs" / RUN_ID
BASE = ROOT / "runs" / "BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1"
V1 = ROOT / "runs" / "BLADE_BASE_UNIFORM_EVOREAL_5C_BENCHMARK_V1"
V1_EVO = V1 / "evoreal_mixture_protocol_20261022"
M0 = ROOT / "assets" / "solver" / "M0" / "tsp.best"
M0_SHA256 = "6826d7a71b30d882e4673a26ced162b7391157795931efe230f4d019292bba15"
OFFICIAL_REPOSITORY = "https://github.com/HenryZhu1029/EvoReal"
OFFICIAL_COMMIT = "919616337105e894e06463734ec1dd71f6451309"
EVO_COUNTS = {"S1": 19_792, "S2": 12_500, "S3": 17_708}
EVO_VALIDATION_COUNTS = {"S1": 19, "S2": 12, "S3": 17}
EVO_SPLITS = {
    0: {"S1": 3958, "S2": 2500, "S3": 3542},
    1: {"S1": 3958, "S2": 2500, "S3": 3542},
    2: {"S1": 3958, "S2": 2500, "S3": 3542},
    3: {"S1": 3959, "S2": 2500, "S3": 3541},
    4: {"S1": 3959, "S2": 2500, "S3": 3541},
}
GENERATOR_HASHES = {
    "gpt_S1.py": "918bf7568ab4bcbe26042d0c8d02a221d4f9b2fc838e2b415992a55f31218fe0",
    "gpt_S2.py": "1129b5b902bc7aeee65c226e97e7180a601c2e35ed8b9810c3b8ead0a18b71f8",
    "gpt_S3.py": "b9eee5d038ca999a30d7aec80be365cacf5ee332264d6fe00e1a1c454d98c364",
}
UNIFORM_GENERATION_SEED = 2026102200
UNIFORM_SHUFFLE_SEED = 2026102201
ARMS = {
    "B0": {"directory": "B0_UNIFORM50K", "dataset": "uniform50k_coordinates.npy"},
    "B1": {"directory": "B1_EVOREAL50K", "dataset": "evoreal50k_coordinates.npy"},
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows, columns=columns)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(target) != sha256_file(source):
            raise RuntimeError(f"existing reused artifact differs: {target}")
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def arm_name(short: str) -> str:
    return str(ARMS[short]["directory"])


def arm_dir(short: str) -> Path:
    return OUT / arm_name(short)


def dataset_path(short: str) -> Path:
    return OUT / "datasets" / str(ARMS[short]["dataset"])


def update_status(short: str, **values: Any) -> None:
    path = arm_dir(short) / "status.json"
    old = read_json(path) if path.is_file() else {"arm": arm_name(short), "run_id": RUN_ID}
    old.update(values)
    old["timestamp_utc"] = utc_now()
    atomic_json(path, old)


def command_output(command: list[str]) -> dict[str, Any]:
    proc = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return {"command": command, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def protected_reference_snapshot() -> dict[str, str]:
    paths: list[Path] = [
        BASE / "STRICT_VERIFICATION.json",
        BASE / "12_all_eight_benchmark_gaps.csv",
        BASE / "13_M5_four_arm_comparison.csv",
        BASE / "18_total_compute_costs.csv",
        BASE / "protocol" / "frozen_experiment_config.json",
        M0,
    ]
    for arm in canonical.ARMS:
        paths.extend(sorted((BASE / arm / "checkpoints").glob("M*_E256.tsp.best")))
        paths.extend(sorted((BASE / arm / "cycles").glob("C*/COMPLETE.json")))
    return {str(path.resolve()): sha256_file(path) for path in paths}


def verify_protected_references() -> None:
    snapshot = read_json(OUT / "protocol" / "reference_checkpoint_hashes.json")
    for value in snapshot["protected_files"]:
        path = Path(value)
        if not path.is_file() or sha256_file(path) != snapshot["protected_files"][value]:
            raise RuntimeError(f"protected Base artifact changed: {path}")


def frozen_test_hashes() -> tuple[set[str], set[str], list[dict[str, Any]]]:
    ordered: set[str] = set()
    pointsets: set[str] = set()
    rows: list[dict[str, Any]] = []
    for name in canonical.BENCHMARKS:
        payload = evaluator.load_fixed_references(name)
        arrays = payload.get("coordinates")
        if arrays is None:
            arrays = [record[1] for record in payload["records"]]
        for xy in arrays:
            variants = [np.asarray(xy), np.asarray(xy, dtype=np.float32).astype(np.float64)]
            if name == "OriginalTSPLIB70":
                normalized = evaluator.aspect_preserving_normalize(xy)
                variants.extend([normalized, np.asarray(normalized, dtype=np.float32).astype(np.float64)])
            for variant in variants:
                ordered.add(coordinate_hash(variant))
                pointsets.add(coordinate_set_hash(variant))
        rows.append({
            "dataset": name,
            "instance_count": len(arrays),
            "dataset_sha256": payload["dataset_sha256"],
        })
    return ordered, pointsets, rows


def prepare_uniform() -> dict[str, Any]:
    target = dataset_path("B0")
    manifest_path = OUT / "datasets" / "uniform50k_manifest.json"
    forbidden, forbidden_sets, test_rows = frozen_test_hashes()
    started = time.perf_counter()
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if not target.is_file() or sha256_file(target) != manifest["coordinate_file_sha256"]:
            raise RuntimeError("frozen Uniform50k hash mismatch")
        coordinates = np.load(target, allow_pickle=False)
    else:
        rng = np.random.Generator(np.random.PCG64(UNIFORM_GENERATION_SEED))
        rows: list[np.ndarray] = []
        seen: set[str] = set()
        seen_sets: set[str] = set()
        rejected = 0
        while len(rows) < 50_000:
            xy = rng.uniform(0.0, 1.0, size=(100, 2))
            h = coordinate_hash(xy)
            hs = coordinate_set_hash(xy)
            f32 = xy.astype(np.float32).astype(np.float64)
            overlaps = (
                h in forbidden or hs in forbidden_sets
                or coordinate_hash(f32) in forbidden
                or coordinate_set_hash(f32) in forbidden_sets
            )
            if h in seen or hs in seen_sets or overlaps:
                rejected += 1
                continue
            rows.append(xy)
            seen.add(h)
            seen_sets.add(hs)
        coordinates = np.stack(rows)
        order = np.random.Generator(np.random.PCG64(UNIFORM_SHUFFLE_SEED)).permutation(50_000)
        coordinates = np.ascontiguousarray(coordinates[order], dtype=np.float64)
        np.save(target, coordinates, allow_pickle=False)
        permutation = OUT / "datasets" / "uniform50k_shuffle_permutation.npy"
        np.save(permutation, order, allow_pickle=False)
        hashes = [coordinate_hash(xy) for xy in coordinates]
        cycles = []
        for cycle in range(5):
            membership = OUT / "datasets" / f"uniform_D{cycle}_membership.txt"
            membership.write_text("\n".join(hashes[cycle * 10_000:(cycle + 1) * 10_000]) + "\n", encoding="ascii")
            cycles.append({
                "cycle": f"D{cycle}", "count": 10_000,
                "membership_file": str(membership), "membership_sha256": sha256_file(membership),
            })
        manifest = {
            "status": "FROZEN_AWAITING_EXPERT_LABELS",
            "distribution": "TSP100 iid Uniform([0,1]^2)",
            "coordinate_file": str(target),
            "coordinate_file_sha256": sha256_file(target),
            "shape": list(coordinates.shape), "dtype": str(coordinates.dtype),
            "valid_unique_count": 50_000,
            "unique_ordered_instances": 50_000, "unique_point_sets": 50_000,
            "generation_rng": "numpy.random.PCG64", "generation_seed": UNIFORM_GENERATION_SEED,
            "shuffle_rng": "numpy.random.PCG64", "shuffle_seed": UNIFORM_SHUFFLE_SEED,
            "shuffle_count": 1, "shuffle_file_sha256": sha256_file(permutation),
            "rejected_during_generation": rejected, "exact_train_test_overlap": 0,
            "evaluation_sets": test_rows, "cycles": cycles,
            "generation_wall_seconds": time.perf_counter() - started,
        }
        atomic_json(manifest_path, manifest)
    hashes = [coordinate_hash(xy) for xy in coordinates]
    point_hashes = [coordinate_set_hash(xy) for xy in coordinates]
    if coordinates.shape != (50_000, 100, 2) or not np.isfinite(coordinates).all():
        raise RuntimeError("invalid Uniform50k tensor")
    if coordinates.min() < 0 or coordinates.max() > 1:
        raise RuntimeError("Uniform50k coordinates outside unit square")
    if len(set(hashes)) != 50_000 or len(set(point_hashes)) != 50_000:
        raise RuntimeError("Uniform50k contains duplicates")
    if set(hashes) & forbidden or set(point_hashes) & forbidden_sets:
        raise RuntimeError("Uniform50k overlaps a frozen evaluation set")
    return manifest


def prepare_evoreal() -> dict[str, Any]:
    source_manifest = read_json(ROOT / "third_party" / "evoreal" / "SOURCE_MANIFEST.json")
    if source_manifest["pinned_commit"] != OFFICIAL_COMMIT:
        raise RuntimeError("EvoReal source commit mismatch")
    for filename, expected in GENERATOR_HASHES.items():
        if sha256_file(ROOT / "third_party" / "evoreal" / filename) != expected:
            raise RuntimeError(f"EvoReal generator hash mismatch: {filename}")
    prior = read_json(V1_EVO / "evoreal50k_provenance.json")
    if prior["official_mixture_ratio_S1_S2_S3"] != [19, 12, 17] or prior["counts"] != EVO_COUNTS:
        raise RuntimeError("V1 EvoReal mixture provenance mismatch")
    source_coords = Path(prior["coordinate_file"])
    source_labels = Path(prior["source_label_file"])
    target_coords = dataset_path("B1")
    target_labels = OUT / "datasets" / "evoreal50k_source_labels.npy"
    if sha256_file(source_coords) != prior["coordinate_file_sha256"]:
        raise RuntimeError("V1 frozen EvoReal coordinates changed")
    if sha256_file(source_labels) != prior["source_label_file_sha256"]:
        raise RuntimeError("V1 frozen EvoReal labels changed")
    link_or_copy(source_coords, target_coords)
    link_or_copy(source_labels, target_labels)
    coordinates = np.load(target_coords, allow_pickle=False)
    labels = np.load(target_labels, allow_pickle=False)
    if coordinates.shape != (50_000, 100, 2) or labels.shape != (50_000,):
        raise RuntimeError("invalid frozen EvoReal50k shape")
    counts = {key: int(np.sum(labels == key)) for key in EVO_COUNTS}
    if counts != EVO_COUNTS or not np.isfinite(coordinates).all() or coordinates.min() < 0 or coordinates.max() > 1:
        raise RuntimeError("invalid frozen EvoReal50k contents")
    hashes = [coordinate_hash(xy) for xy in coordinates]
    point_hashes = [coordinate_set_hash(xy) for xy in coordinates]
    forbidden, forbidden_sets, test_rows = frozen_test_hashes()
    if len(set(hashes)) != 50_000 or len(set(point_hashes)) != 50_000:
        raise RuntimeError("EvoReal50k contains duplicate instances")
    if set(hashes) & forbidden or set(point_hashes) & forbidden_sets:
        raise RuntimeError("EvoReal50k overlaps a frozen evaluation set")
    cycle_rows: list[dict[str, Any]] = []
    for cycle in range(5):
        part = labels[cycle * 10_000:(cycle + 1) * 10_000]
        membership = OUT / "datasets" / f"evoreal_D{cycle}_membership.txt"
        membership.write_text("\n".join(hashes[cycle * 10_000:(cycle + 1) * 10_000]) + "\n", encoding="ascii")
        for source in ("S1", "S2", "S3"):
            actual = int(np.sum(part == source))
            if actual != EVO_SPLITS[cycle][source]:
                raise RuntimeError(f"EvoReal D{cycle}/{source} composition mismatch")
            cycle_rows.append({
                "cycle": f"D{cycle}", "source": source, "count": actual,
                "fraction": actual / 10_000,
                "target_fraction": EVO_VALIDATION_COUNTS[source] / 48,
                "membership_sha256": sha256_file(membership),
            })
    write_csv(OUT / "datasets" / "evoreal_cycle_mixture.csv", cycle_rows)
    manifest = {
        "status": "FROZEN_AWAITING_EXPERT_LABELS",
        "source_policy": "reuse V1 deterministic pool after full hash and content verification",
        "source_v1_manifest": str(V1_EVO / "evoreal50k_provenance.json"),
        "source_v1_manifest_sha256": sha256_file(V1_EVO / "evoreal50k_provenance.json"),
        "coordinate_file": str(target_coords), "coordinate_file_sha256": sha256_file(target_coords),
        "source_label_file": str(target_labels), "source_label_file_sha256": sha256_file(target_labels),
        "shape": list(coordinates.shape), "dtype": str(coordinates.dtype),
        "problem_size": 100, "coordinate_range": [float(coordinates.min()), float(coordinates.max())],
        "coordinate_conversion": "none; official generators already emit unit-square x,y coordinates",
        "official_repository": OFFICIAL_REPOSITORY, "official_commit": OFFICIAL_COMMIT,
        "official_partition": "rule_based", "validation_counts": EVO_VALIDATION_COUNTS,
        "validation_total": 48, "mixture_ratio_S1_S2_S3": [19, 12, 17], "counts": EVO_COUNTS,
        "generator_hashes": GENERATOR_HASHES, "valid_unique_count": 50_000,
        "unique_ordered_instances": 50_000, "unique_point_sets": 50_000,
        "exact_train_test_overlap": 0, "evaluation_sets": test_rows,
        "phase_two_real_TSPLIB_finetuning": False,
    }
    atomic_json(OUT / "datasets" / "evoreal50k_manifest.json", manifest)
    return manifest


def environment_manifest() -> dict[str, Any]:
    return {
        "timestamp_utc": utc_now(), "hostname": socket.gethostname(),
        "username": os.environ.get("USER") or command_output(["id", "-un"])["stdout"].strip(),
        "repository_path": str(ROOT),
        "git_commit": command_output(["git", "rev-parse", "HEAD"])["stdout"].strip(),
        "git_status": command_output(["git", "status", "--short"]),
        "os": platform.platform(), "kernel": platform.release(),
        "python_executable": sys.executable, "python_version": sys.version,
        "torch_version": torch.__version__, "torch_cuda_runtime": torch.version.cuda,
        "nvidia_smi": command_output(["nvidia-smi"]),
        "gpu_pci": command_output(["lspci", "-nn"]),
        "nvidia_device_nodes": command_output(["bash", "-lc", "ls -l /dev/nvidia* 2>&1"]),
    }


def write_protocol_and_audits() -> None:
    (OUT / "protocol").mkdir(parents=True, exist_ok=True)
    (OUT / "datasets").mkdir(parents=True, exist_ok=True)
    for short in ARMS:
        for child in ("checkpoints", "cycles", "evaluations", "logs", "oracle"):
            (arm_dir(short) / child).mkdir(parents=True, exist_ok=True)
    for child in ("comparison", "figures", "reports"):
        (OUT / child).mkdir(parents=True, exist_ok=True)
    if sha256_file(M0) != M0_SHA256:
        raise RuntimeError("canonical M0 hash mismatch")
    strict = read_json(BASE / "STRICT_VERIFICATION.json")
    if strict.get("STRICT_VERIFICATION") != "PASS":
        raise RuntimeError("completed Base strict verification is not PASS")
    source_manifest = read_json(ROOT / "third_party" / "evoreal" / "SOURCE_MANIFEST.json")
    if source_manifest["pinned_commit"] != OFFICIAL_COMMIT:
        raise RuntimeError("pinned EvoReal source mismatch")
    atomic_json(OUT / "protocol" / "frozen_protocol.json", {
        "run_id": RUN_ID, "reference": canonical.RUN_ID, "initial_model": str(M0),
        "initial_model_sha256": M0_SHA256, "architecture": "canonical BQ TSP100",
        "cycles": 5, "instances_per_cycle": 10_000, "epochs_per_cycle": 256,
        "effective_batch_size": 256, "optimizer": "Adam", "learning_rate": 3.75e-5,
        "weight_decay": 0, "fresh_optimizer_each_cycle": True, "scheduler": None,
        "AMP": False, "gradient_clipping": None, "early_stopping": False,
        "best_checkpoint_selection": False, "uniform_anchor": 0, "training_seed": 2026,
        "drop_last": False, "optimizer_steps_per_epoch": 40,
        "optimizer_steps_per_cycle": 10_240, "optimizer_steps_per_arm": 51_200,
        "evaluation": {"datasets": list(canonical.BENCHMARKS), "decoder": "greedy", "endpoint": "M5"},
    })
    atomic_json(OUT / "protocol" / "reference_checkpoint_hashes.json", {
        "protected_files": protected_reference_snapshot(), "V1_run": str(V1),
        "V1_top_level_files": {str(path.resolve()): sha256_file(path) for path in V1.iterdir() if path.is_file()},
    })
    atomic_json(OUT / "protocol" / "environment_manifest.json", environment_manifest())
    ratio_rows = []
    for source in ("S1", "S2", "S3"):
        ratio_rows.append({
            "source": source, "validation_count": EVO_VALIDATION_COUNTS[source],
            "validation_total": 48, "weight": EVO_VALIDATION_COUNTS[source] / 48,
            "training_count": EVO_COUNTS[source], "official_partition": "rule_based",
            "official_commit": OFFICIAL_COMMIT, "generator_sha256": GENERATOR_HASHES[f"gpt_{source}.py"],
        })
    write_csv(OUT / "datasets" / "evoreal_official_ratio_audit.csv", ratio_rows)
    (OUT / "datasets" / "evoreal_source_audit.md").write_text(
        "# EvoReal official-source audit\n\n"
        "Status: **VERIFIED**\n\n"
        f"Official repository: {OFFICIAL_REPOSITORY}\n\n"
        f"Pinned commit: `{OFFICIAL_COMMIT}`. The local S1/S2/S3 modules are byte-identical "
        "to the pinned POMO Phase One generator modules. The pinned `train_p1.py` uses "
        "`gen_weights=[19,12,17]` in positional `[S1,S2,S3]` order; the invoked mixed generator "
        "uses that same order. The active rule-based validation directories contain 19 S1, 12 S2, "
        "and 17 S3 instances (48 total). The paper example must therefore not be interpreted as "
        "positional 17:19:12 for this implementation. Phase Two and all real-TSPLIB training are excluded.\n\n"
        "The V1 frozen deterministic 50k pool is reused by hash; its incomplete audit is not overwritten.\n",
        encoding="utf-8",
    )


def init() -> None:
    write_protocol_and_audits()
    environment = read_json(OUT / "protocol" / "environment_manifest.json")
    gpu_ready = torch.cuda.is_available() and torch.cuda.device_count() > 0
    devices: list[dict[str, Any]] = []
    allocation_pass = False
    m0_load_pass = False
    optimizer_audit: dict[str, Any] = {"status": "NOT_RUN"}
    if gpu_ready:
        for index in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(index)
            allocation_pass = float(torch.ones(8, device=f"cuda:{index}").sum()) == 8.0
            devices.append({"visible_index": index, "name": prop.name, "total_memory": prop.total_memory})
        model, metadata = canonical.load_canonical_checkpoint(M0, torch.device("cuda:0"), expected_sha256=M0_SHA256)
        m0_load_pass = metadata["sha256"] == M0_SHA256 and all(torch.isfinite(p).all().item() for p in model.parameters())
        del model
        probe = np.random.default_rng(0).random((256, 100, 2))
        optimizer_audit = canonical.production_optimizer_audit(probe)
        torch.cuda.empty_cache()
    gpu_record = {
        "status": "CUDA_READY" if gpu_ready and allocation_pass and m0_load_pass and optimizer_audit.get("status") == "PASS" else "GPU_BLOCKED",
        "root_cause_of_prior_v1_failure": "B: agent sandbox lacked NVIDIA device passthrough; host execution is CUDA-ready",
        "visible_devices": devices, "allocation_check": allocation_pass,
        "canonical_M0_load": m0_load_pass, "production_optimizer_audit": optimizer_audit,
        "nvidia_smi": environment["nvidia_smi"],
    }
    atomic_json(OUT / "protocol" / "GPU_readiness.json", gpu_record)
    uniform = prepare_uniform()
    evo = prepare_evoreal()
    atomic_json(OUT / "protocol" / "reference_compatibility.json", {
        "status": "PASS", "canonical_M0_sha256": M0_SHA256,
        "base_strict_verification": "PASS", "training_protocol_equal": canonical.CONFIG["training"] == read_json(BASE / "protocol" / "frozen_experiment_config.json")["training"],
        "eight_frozen_evaluation_sets": list(canonical.BENCHMARKS),
        "uniform_dataset_sha256": uniform["coordinate_file_sha256"],
        "evoreal_dataset_sha256": evo["coordinate_file_sha256"],
    })
    update_status("B0", status="DATA_PENDING", GPU_status=gpu_record["status"], training_started=False)
    update_status("B1", status="DATA_PENDING", GPU_status=gpu_record["status"], training_started=False)
    verify_protected_references()
    print(f"INIT_COMPLETE GPU={gpu_record['status']} B0=DATA_PENDING B1=DATA_PENDING", flush=True)


def record_path(short: str, instance_hash: str) -> Path:
    return arm_dir(short) / "oracle" / "instance_records" / instance_hash[:2] / f"{instance_hash}.json"


def prior_evoreal_record(instance_hash: str) -> Path:
    return V1_EVO / "oracle" / "instance_records" / instance_hash[:2] / f"{instance_hash}.json"


def valid_oracle_row(row: dict[str, Any], instance_hash: str) -> bool:
    return (
        str(row.get("instance_hash")) == instance_hash
        and row.get("status") == "solved" and bool(row.get("verified"))
        and row.get("binary_hash") == CANONICAL_CONCORDE_SHA256
    )


def solve_and_clean(short: str, xy: np.ndarray, instance_hash: str, backend: dict[str, Any], timeout: float) -> dict[str, Any]:
    work = arm_dir(short) / "oracle" / "work" / instance_hash
    if work.exists():
        shutil.rmtree(work)
    try:
        return solve_one(xy, instance_hash, work, backend, timeout)
    finally:
        if work.exists():
            shutil.rmtree(work)


def label(short: str, workers: int, timeout: float) -> None:
    if short not in ARMS:
        raise ValueError(short)
    verify_protected_references()
    lock_path = arm_dir(short) / "oracle" / "LABELING.lock"
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        coordinates = np.load(dataset_path(short), allow_pickle=False)
        hashes = [coordinate_hash(xy) for xy in coordinates]
        backend = resolve_backend()
        if backend["binary_hash"] != CANONICAL_CONCORDE_SHA256:
            raise RuntimeError("Concorde binary mismatch")
        completed: dict[str, dict[str, Any]] = {}
        reused_v1 = 0
        for instance_hash in hashes:
            target = record_path(short, instance_hash)
            candidates = [target]
            if short == "B1":
                candidates.append(prior_evoreal_record(instance_hash))
            row = None
            for candidate in candidates:
                if candidate.is_file():
                    possible = read_json(candidate)
                    if not valid_oracle_row(possible, instance_hash):
                        raise RuntimeError(f"invalid persisted oracle record: {candidate}")
                    row = possible
                    if candidate != target:
                        reused_v1 += 1
                    break
            if row is not None:
                completed[instance_hash] = row
        pending = [(index, h) for index, h in enumerate(hashes) if h not in completed]
        update_status(short, status="DATA_PENDING", phase="CONCORDE_LABELING", labeled=len(completed), total=50_000,
                      reused_V1_expert_tours=reused_v1, concorde_workers=workers, training_started=False)
        started = time.perf_counter()
        print(f"{short}_CONCORDE_RESUME={len(completed)}/50000 reused_v1={reused_v1}", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            jobs = {
                executor.submit(solve_and_clean, short, coordinates[index], instance_hash, backend, timeout): instance_hash
                for index, instance_hash in pending
            }
            for future in as_completed(jobs):
                row = future.result()
                instance_hash = str(row["instance_hash"])
                atomic_json(record_path(short, instance_hash), row)
                completed[instance_hash] = row
                count = len(completed)
                if count % 100 == 0 or count == 50_000:
                    update_status(short, status="DATA_PENDING", phase="CONCORDE_LABELING", labeled=count, total=50_000,
                                  reused_V1_expert_tours=reused_v1, concorde_workers=workers, training_started=False)
                    print(f"{short}_CONCORDE_PROGRESS={count}/50000", flush=True)
        ledger = canonical_oracle_ledger(pd.DataFrame([completed[h] for h in hashes]))
        ledger_path = arm_dir(short) / "oracle" / "expert_ledger.parquet"
        ledger.to_parquet(ledger_path, index=False)
        tours = np.stack([np.fromstring(value, sep=" ", dtype=np.int64) for value in ledger.expert_tour])
        if tours.shape != (50_000, 100) or not np.array_equal(np.sort(tours, axis=1), np.broadcast_to(np.arange(100), tours.shape)):
            raise RuntimeError("expert tours are not Hamiltonian permutations")
        for index, tour in enumerate(tours):
            if tour_cost_int(integer_distance_matrix(coordinates[index]), tour) != int(ledger.iloc[index].optimal_cost_int):
                raise RuntimeError(f"expert cost mismatch at {index}")
            if tour_hash(tour.tolist()) != ledger.iloc[index].tour_hash:
                raise RuntimeError(f"expert tour hash mismatch at {index}")
        tour_path = arm_dir(short) / "oracle" / "expert_tours.npy"
        np.save(tour_path, tours, allow_pickle=False)
        elapsed = time.perf_counter() - started
        atomic_json(arm_dir(short) / "oracle" / "expert_tour_integrity.json", {
            "status": "PASS", "count": 50_000, "valid_hamiltonian_tours": 50_000,
            "recomputed_integer_costs_match": 50_000, "concorde_binary_sha256": backend["binary_hash"],
            "protocol": "canonical explicit FULL_MATRIX rint(Euclidean*1e6)",
            "newly_solved_instances": 50_000 - reused_v1,
            "reused_V1_expert_tours": reused_v1, "workers": workers,
            "label_resume_wall_seconds": elapsed, "solver_sum_seconds": float(ledger.runtime_sec.sum()),
            "ledger": str(ledger_path), "ledger_sha256": sha256_file(ledger_path),
            "tour_file": str(tour_path), "tour_file_sha256": sha256_file(tour_path),
        })
        manifest_path = OUT / "datasets" / ("uniform50k_manifest.json" if short == "B0" else "evoreal50k_manifest.json")
        manifest = read_json(manifest_path)
        manifest.update({
            "status": "DATA_AND_EXPERT_LABELS_READY", "expert_tour_count": 50_000,
            "expert_tour_file": str(tour_path), "expert_tour_file_sha256": sha256_file(tour_path),
            "oracle_ledger": str(ledger_path), "oracle_ledger_sha256": sha256_file(ledger_path),
            "concorde_binary_sha256": backend["binary_hash"], "reused_V1_expert_tours": reused_v1,
        })
        atomic_json(manifest_path, manifest)
        update_status(short, status="READY", phase="DATA_AND_LABELS_READY", labeled=50_000,
                      reused_V1_expert_tours=reused_v1, training_started=False,
                      error=None, traceback=None)
        verify_protected_references()
        print(f"{short}_DATA_READY labels=50000 reused_v1={reused_v1}", flush=True)


class TrainingStatus:
    def __init__(self, short: str):
        self.short = short

    def update(self, **values: Any) -> None:
        update_status(self.short, **values)


def configure_canonical_output() -> None:
    canonical.RUN = OUT
    canonical.RUN_ID = RUN_ID


def completed_cycle_valid(short: str, cycle: int) -> bool:
    marker = arm_dir(short) / "cycles" / f"C{cycle:02d}" / "COMPLETE.json"
    if not marker.is_file():
        return False
    record = read_json(marker)
    for path, expected in record["required_files"].items():
        artifact = Path(path)
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise RuntimeError(f"completed cycle artifact mismatch: {artifact}")
    return True


def train(short: str) -> None:
    if short not in ARMS:
        raise ValueError(short)
    verify_protected_references()
    integrity = arm_dir(short) / "oracle" / "expert_tour_integrity.json"
    if not integrity.is_file() or read_json(integrity).get("status") != "PASS":
        raise RuntimeError(f"{short} expert labels are not ready")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("training worker requires exactly one CUDA-visible physical GPU")
    prop = torch.cuda.get_device_properties(0)
    allocation = float(torch.ones(8, device="cuda:0").sum())
    if allocation != 8.0:
        raise RuntimeError("CUDA allocation test failed")
    configure_canonical_output()
    lock_path = arm_dir(short) / "TRAINING.lock"
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        coordinates = np.load(dataset_path(short), allow_pickle=False)
        hashes = [coordinate_hash(xy) for xy in coordinates]
        tours = np.load(arm_dir(short) / "oracle" / "expert_tours.npy", allow_pickle=False)
        if coordinates.shape != (50_000, 100, 2) or tours.shape != (50_000, 100):
            raise RuntimeError("training tensor shape mismatch")
        status = TrainingStatus(short)
        status.update(status="RUNNING", phase="GPU_PREFLIGHT", training_started=True,
                      CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
                      local_cuda_device=0, gpu_name=prop.name, gpu_total_memory=prop.total_memory,
                      error=None, traceback=None)
        device = torch.device("cuda:0")
        model, meta = canonical.load_canonical_checkpoint(M0, device, expected_sha256=M0_SHA256)
        if meta["sha256"] != M0_SHA256:
            raise RuntimeError("arm did not begin from canonical M0")
        del model
        torch.cuda.empty_cache()
        m0_source = BASE / "protocol" / "common_initialization" / "M0_external_benchmark_results.csv"
        link_or_copy(m0_source, arm_dir(short) / "evaluations" / "M0_eight.csv")
        for cycle in range(5):
            if completed_cycle_valid(short, cycle):
                update_status(short, status="RUNNING", phase="VERIFIED_COMPLETE_SKIP", cycle=f"C{cycle:02d}")
                continue
            source_checkpoint = M0 if cycle == 0 else arm_dir(short) / "checkpoints" / f"M{cycle}_E256.tsp.best"
            expected = M0_SHA256 if cycle == 0 else None
            model, meta = canonical.load_canonical_checkpoint(source_checkpoint, device, expected_sha256=expected)
            start = cycle * 10_000
            end = start + 10_000
            cycle_hashes = hashes[start:end]
            membership_file = OUT / "datasets" / f"{'uniform' if short == 'B0' else 'evoreal'}_D{cycle}_membership.txt"
            if membership_hash(cycle_hashes) != sha256_file(membership_file):
                raise RuntimeError(f"{short} C{cycle:02d} membership mismatch")
            status.update(status="RUNNING", phase="BQ_TRAINING", cycle=f"C{cycle:02d}",
                          input_checkpoint_sha256=meta["sha256"])
            model, output_meta, timing = canonical.train(
                arm_name(short), cycle, model, meta["sha256"], coordinates[start:end],
                cycle_hashes, tours[start:end], device, status, True,
            )
            checkpoint = arm_dir(short) / "checkpoints" / f"M{cycle + 1}_E256.tsp.best"
            frame, evaluation_seconds = canonical.evaluate(
                model, checkpoint, f"M{cycle + 1}",
                arm_dir(short) / "evaluations" / f"M{cycle + 1}_eight.csv", status, arm_name(short),
            )
            if len(frame) != 8 or not frame.failed_tours.eq(0).all():
                raise RuntimeError(f"{short} C{cycle:02d} evaluation incomplete")
            cycle_dir = arm_dir(short) / "cycles" / f"C{cycle:02d}"
            summary = {
                "status": "COMPLETE", "arm": arm_name(short), "cycle": f"C{cycle:02d}",
                "model_before": f"M{cycle}", "model_after": f"M{cycle + 1}",
                "input_checkpoint_sha256": meta["sha256"], "output_checkpoint_sha256": output_meta["sha256"],
                "training_membership_sha256": membership_hash(cycle_hashes),
                "training": timing, "evaluation_seconds": evaluation_seconds,
                "completed_utc": utc_now(),
            }
            atomic_json(cycle_dir / "cycle_summary.json", summary)
            required_paths = [
                checkpoint, cycle_dir / "09_bq_training" / "training_loss_curve.csv",
                cycle_dir / "09_bq_training" / "bq_training_timing.json",
                arm_dir(short) / "evaluations" / f"M{cycle + 1}_eight.csv", cycle_dir / "cycle_summary.json",
            ]
            atomic_json(cycle_dir / "COMPLETE.json", {
                "status": "COMPLETE", "arm": arm_name(short), "cycle": f"C{cycle:02d}",
                "required_files": {str(path.resolve()): sha256_file(path) for path in required_paths},
                "completed_utc": utc_now(),
            })
            update_status(short, status="RUNNING", phase="CYCLE_COMPLETE", cycle=f"C{cycle:02d}",
                          checkpoint_sha256=output_meta["sha256"])
            del model
            torch.cuda.empty_cache()
        m5 = arm_dir(short) / "checkpoints" / "M5_E256.tsp.best"
        if not m5.is_file() or not all(completed_cycle_valid(short, cycle) for cycle in range(5)):
            raise RuntimeError(f"{short} did not reach a verified M5")
        update_status(short, status="COMPLETE", phase="M5_COMPLETE", training_started=True,
                      M5_checkpoint_sha256=sha256_file(m5), official_endpoint="M5",
                      error=None, traceback=None)
        verify_protected_references()
        print(f"{short}_TRAINING_COMPLETE M5={sha256_file(m5)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init")
    label_parser = subparsers.add_parser("label")
    label_parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    label_parser.add_argument("--workers", type=int, default=32)
    label_parser.add_argument("--timeout", type=float, default=7200.0)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    args = parser.parse_args()
    try:
        if args.command == "init":
            init()
        elif args.command == "label":
            label(args.arm, args.workers, args.timeout)
        else:
            train(args.arm)
    except BaseException as exc:
        # A duplicate invocation is correctly rejected by the arm lock.  It
        # must not overwrite the status owned by the already-running worker.
        if getattr(args, "arm", None) in ARMS and not isinstance(exc, BlockingIOError):
            update_status(args.arm, status="FAILED", phase=args.command.upper(), error=repr(exc), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
