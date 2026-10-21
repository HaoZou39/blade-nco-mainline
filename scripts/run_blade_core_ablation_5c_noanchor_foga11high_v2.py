#!/usr/bin/env python
"""Four-arm matched BLADE core ablation with frozen FOGA11_HIGH_V1."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback
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
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import external_v2_passive_blade_eval as evaluator  # noqa: E402
import run_blade_v2 as base  # noqa: E402
import run_s4b_hgb40_union_qd as hgbcore  # noqa: E402
from bq.checkpoint import load_canonical_checkpoint  # noqa: E402
from bq.protocol import MICROBATCH_SIZE  # noqa: E402
from bq.supervised import create_optimizer, sample_suffix_length, supervised_optimizer_step  # noqa: E402
from bq.training_hardness import closed_expert_coordinates  # noqa: E402
from features.qd4 import QD4_FEATURE_NAMES  # noqa: E402
from final_run.science import hardness, membership_hash, qd_frame  # noqa: E402
from final_run.stage import atomic_json, atomic_stage, sha256_file, utc_now  # noqa: E402
from mutation import OPERATOR_NAMES, apply_operator_with_config  # noqa: E402
from mutation.parameters import load_named_profile  # noqa: E402
from surrogate_benchmark.models import HGB_CFG_CANONICAL, SEEDS  # noqa: E402


RUN_ID = "BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11HIGH_V2_BATCH256"
SEED_NAMESPACE = "BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11HIGH_V2"
RUN = ROOT / "runs" / RUN_ID
PROTOCOL = RUN / "protocol"
COMMON = PROTOCOL / "common_initialization"
SHARED_STATE = RUN / "_shared_state"
COMPARISON = RUN / "comparison"
FIGURES = RUN / "figures"
RUNTIME = ROOT / "runtime"
LOCK = RUNTIME / "blade_core_ablation_5c_foga11high_v2_batch256.lock"
STATUS = RUNTIME / "blade_core_ablation_5c_foga11high_v2_batch256_status.json"
PROFILE_NAME = "FOGA11_HIGH_V1"
PROFILE_PATH = ROOT / "src" / "mutation" / "foga11_high_v1.yaml"
BASE_PROFILE_PATH = ROOT / "src" / "mutation" / "foga2019_operators.yaml"
WORKER = ROOT / "scripts" / "foga11_high_v1_mutation_worker.py"
CANONICAL_MUTATION_PYTHON = Path("C:/ProgramData/anaconda3/python.exe")
M0 = ROOT / "assets" / "solver" / "M0" / "tsp.best"
M0_SHA = "6826d7a71b30d882e4673a26ced162b7391157795931efe230f4d019292bba15"
U0_SOURCE = ROOT / "runs" / "blade_v2_corrected_union_sanity_seed2026_v1" / "C00" / "01_inputs" / "parents.npz"
HIST_RANDOM = ROOT / "outputs" / "random_mut_iter_15round"
ARMS = ("A0_FULL", "A1_WITHOUT_QD", "A2_WITHOUT_SURROGATE", "A3_RANDOM")
HGB_ARMS = {"A0_FULL", "A1_WITHOUT_QD"}
QD_ARMS = {"A0_FULL", "A2_WITHOUT_SURROGATE"}
BENCHMARKS = tuple(evaluator.EIGHT)
HARD_N = 10_000
CALIBRATION_N = 10_000
EPOCHS = 256
LOGICAL_BATCH = 256
STEPS_PER_EPOCH = 40
STEPS_PER_CYCLE = 10_240
LR = 3.75e-5
CHECKPOINT_EPOCHS = {0, 256}
PROGRESS_EPOCHS = {64, 128, 192, 256}
NORM_COLS = [f"normalized_{name}" for name in QD4_FEATURE_NAMES]
COLORS = {"A0_FULL": "#1f77b4", "A1_WITHOUT_QD": "#ff7f0e", "A2_WITHOUT_SURROGATE": "#2ca02c", "A3_RANDOM": "#7f7f7f"}


CONFIG = {
    "schema_version": RUN_ID,
    "design_freeze": "BLADE_FOGA11_MUTATION_DESIGN_FROZEN_V1",
    "mutation_profile": PROFILE_NAME,
    "arms": {
        "A0_FULL": {"surrogate": "HGB40", "QD": True},
        "A1_WITHOUT_QD": {"surrogate": "HGB40", "QD": False},
        "A2_WITHOUT_SURROGATE": {"surrogate": None, "QD": True},
        "A3_RANDOM": {"surrogate": None, "QD": False},
    },
    "cycles": list(range(5)),
    "parent_n": HARD_N,
    "training_n": HARD_N,
    "parent_child_union": True,
    "surrogate": {"estimator": "HistGradientBoostingRegressor", **HGB_CFG_CANONICAL,
                  "seeds": list(SEEDS), "aggregate": "median", "features": "F40",
                  "target": "log1p_current_LBQ", "calibration_target_n": CALIBRATION_N},
    "QD": {"descriptors": list(QD4_FEATURE_NAMES), "K": 1000,
           "normalization": "frozen_deep_P1P99_clip"},
    "training": {"epochs": EPOCHS, "logical_batch_size": LOGICAL_BATCH,
                 "effective_batch_size": LOGICAL_BATCH,
                 "microbatch_size": MICROBATCH_SIZE, "optimizer_steps_per_epoch": STEPS_PER_EPOCH,
                 "optimizer_steps_per_cycle": STEPS_PER_CYCLE, "learning_rate": LR,
                 "optimizer": "fresh_Adam", "scheduler": None, "weight_decay": 0,
                 "gradient_clipping": None, "AMP": False, "seed": 2026,
                 "gradient_accumulation_steps": 1, "data_parallel_workers": 1,
                 "hard_occurrences_per_epoch": HARD_N, "uniform_anchor_n": 0,
                 "drop_last": False, "checkpoint": "E256_only"},
    "evaluation": {"datasets": list(BENCHMARKS), "decoder": "greedy", "passive_only": True},
    "automatic_M6": False,
}


class Halt(RuntimeError):
    pass


class Status:
    def __init__(self) -> None:
        self.value = {"run_id": RUN_ID, "pid": os.getpid(), "state": "RUNNING", "arm": None,
                      "cycle": None, "phase": "initializing", "latest_error": None}
        self.update()

    def update(self, **values: Any) -> None:
        self.value.update(values)
        self.value["timestamp"] = utc_now()
        atomic_json(STATUS, self.value)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def arm_dir(arm: str) -> Path:
    return RUN / arm


def cycle_dir(arm: str, cycle: int) -> Path:
    return arm_dir(arm) / "cycles" / f"C{cycle:02d}"


def model_path(arm: str, index: int) -> Path:
    return M0 if index == 0 else arm_dir(arm) / "checkpoints" / f"M{index}_E256.tsp.best"


def digest_values(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def deterministic_seed(*parts: object) -> int:
    material = "|".join([SEED_NAMESPACE, "2026", *map(str, parts)]).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def copy_or_link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def active_lock() -> dict[str, Any] | None:
    if not LOCK.is_file():
        return None
    value = read_json(LOCK)
    pid = int(value.get("pid", -1))
    if value.get("state") == "ACTIVE" and pid > 0:
        try:
            os.kill(pid, 0)
            return value
        except OSError:
            pass
    return None


def free_gib() -> float:
    return shutil.disk_usage(ROOT).free / 2**30


def configure_hgb_context(arm: str) -> None:
    hgbcore.RUN_ID = f"{RUN_ID}/{arm}"
    hgbcore.RUN_DIR = arm_dir(arm)
    hgbcore.CYCLES_DIR = arm_dir(arm) / "cycles"
    hgbcore.STATE_DIR = SHARED_STATE
    hgbcore.STATUS = STATUS
    hgbcore.HARD_N = HARD_N
    hgbcore.CALIBRATION_N = CALIBRATION_N
    hgbcore.CONFIG = CONFIG
    hgbcore.cycle_dir = lambda cycle: cycle_dir(arm, cycle)
    hgbcore.model_path = lambda index: model_path(arm, index)


def operator_profile_audit() -> dict[str, Any]:
    base_cfg = yaml.safe_load(BASE_PROFILE_PATH.read_text(encoding="utf-8"))
    profile = load_named_profile(PROFILE_NAME)
    high = {"implosion_mutation", "axis_projection_mutation"}
    base_ops = set(OPERATOR_NAMES) - high
    checks: dict[str, bool] = {
        "operator_count_11": len(OPERATOR_NAMES) == 11 and set(profile["operators"]) == set(OPERATOR_NAMES),
        "repair_exact_base": profile["repair"] == base_cfg["repair"],
        "nine_base_exact": all(profile["operators"][name] == base_cfg["operators"][name] for name in base_ops),
        "implosion_high_exact": profile["operators"]["implosion_mutation"] == {
            "radius": {"distribution": "uniform", "low": 0.20, "high": 0.45},
            "strength": {"distribution": "uniform", "low": 0.60, "high": 0.95}},
        "axis_high_exact": profile["operators"]["axis_projection_mutation"] == {
            "affected_fraction": {"distribution": "uniform", "low": 0.30, "high": 0.60},
            "axis": {"distribution": "choice", "values": ["x", "y"]},
            "target": {"distribution": "uniform", "low": 0.10, "high": 0.90},
            "jitter": {"distribution": "uniform", "low": 0.0005, "high": 0.01}},
    }
    with np.load(U0_SOURCE, allow_pickle=False) as payload:
        probe = np.asarray(payload["coordinates"][0], dtype=np.float64)
    trace: list[dict[str, Any]] = []
    for name in OPERATOR_NAMES:
        for index in range(32):
            seed = deterministic_seed("preflight", name, index)
            first = apply_operator_with_config(name, probe, np.random.default_rng(seed), profile)
            second = apply_operator_with_config(name, probe, np.random.default_rng(seed), profile)
            if first.final_output_hash != second.final_output_hash or first.sampled_parameters != second.sampled_parameters:
                checks["deterministic_replay"] = False
                break
            params = first.sampled_parameters
            row = {"operator_id": name, "seed": seed, "child_hash": first.final_output_hash,
                   "valid": first.valid_after_repair, "repair_count": first.repair_count}
            if name == "implosion_mutation":
                row.update(radius=float(params["radius"]), strength=float(params["strength"]))
            if name == "axis_projection_mutation":
                row.update(affected_fraction=float(params["affected_fraction"]), target=float(params["target"]),
                           jitter=float(params["jitter"]), axis=str(params["axis"]))
            trace.append(row)
    checks.setdefault("deterministic_replay", True)
    imp = [r for r in trace if r["operator_id"] == "implosion_mutation"]
    axis = [r for r in trace if r["operator_id"] == "axis_projection_mutation"]
    checks["implosion_samples_in_range"] = all(.20 <= r["radius"] <= .45 and .60 <= r["strength"] <= .95 for r in imp)
    checks["axis_samples_in_range"] = all(.30 <= r["affected_fraction"] <= .60 and .10 <= r["target"] <= .90
                                                 and .0005 <= r["jitter"] <= .01 and r["axis"] in {"x", "y"} for r in axis)
    checks["independent_nonconstant_samples"] = len({(r["radius"], r["strength"]) for r in imp}) > 30 and len({(r["affected_fraction"], r["target"], r["jitter"], r["axis"]) for r in axis}) > 30
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "profile_sha256": sha256_file(PROFILE_PATH), "base_sha256": sha256_file(BASE_PROFILE_PATH),
            "trace": trace, "two_high": sorted(high), "nine_base": sorted(base_ops)}


def dataloader_audit() -> dict[str, Any]:
    rng = np.random.default_rng(2026)
    rows = []
    suffixes = []
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(HARD_N)
        sizes = []
        for start in range(0, HARD_N, LOGICAL_BATCH):
            size = len(order[start:start + LOGICAL_BATCH])
            sizes.append(size)
            suffixes.append(sample_suffix_length(101, rng))
        rows.append({"epoch": epoch, "unique": len(np.unique(order)), "occurrences": len(order),
                     "steps": len(sizes), "first_batch": sizes[0], "last_batch": sizes[-1]})
    valid = all(r["unique"] == r["occurrences"] == HARD_N and r["steps"] == STEPS_PER_EPOCH
                and r["first_batch"] == LOGICAL_BATCH and r["last_batch"] == 16 for r in rows)
    effective = MICROBATCH_SIZE * (LOGICAL_BATCH // MICROBATCH_SIZE) * 1
    return {"status": "PASS" if valid and STEPS_PER_EPOCH * EPOCHS == STEPS_PER_CYCLE and effective == LOGICAL_BATCH else "FAIL",
            "batch_size": LOGICAL_BATCH, "microbatch_size": MICROBATCH_SIZE, "drop_last": False,
            "gradient_accumulation_steps": 1, "data_parallel_workers": 1,
            "effective_batch_size": effective, "full_batches_per_epoch": HARD_N // LOGICAL_BATCH,
            "final_partial_batch_size": HARD_N % LOGICAL_BATCH,
            "steps_per_epoch": STEPS_PER_EPOCH, "steps_per_cycle": STEPS_PER_CYCLE,
            "occurrences_per_epoch": HARD_N, "epoch_1": rows[0], "epoch_256": rows[-1],
            "suffix_schedule_sha256": hashlib.sha256(np.asarray(suffixes, dtype=np.int16).tobytes()).hexdigest()}


def production_optimizer_audit(ucoords: np.ndarray) -> dict[str, Any]:
    """Exercise the exact production optimizer/microbatch path for one 256-instance update."""
    device = base._device()
    if device.type != "cuda":
        return {"status": "FAIL", "reason": f"CUDA required, got {device}"}
    model, _ = load_canonical_checkpoint(M0, device, expected_sha256=M0_SHA)
    optimizer = create_optimizer(model, LR)
    closed = np.concatenate([ucoords[:LOGICAL_BATCH], ucoords[:LOGICAL_BATCH, :1]], axis=1)
    result = supervised_optimizer_step(model, optimizer, closed, closed.shape[1], device)
    audited = {
        "status": "PASS" if (
            result["n_instances"] == 256
            and result["microbatch_size"] == MICROBATCH_SIZE
            and result["n_microbatches"] == 4
            and result["optimizer_updates"] == 1
            and optimizer.param_groups[0]["lr"] == LR
            and optimizer.param_groups[0]["weight_decay"] == 0
        ) else "FAIL",
        "logical_batch_size": int(result["n_instances"]),
        "microbatch_size": int(result["microbatch_size"]),
        "microbatches_per_optimizer_update": int(result["n_microbatches"]),
        "optimizer_updates": int(result["optimizer_updates"]),
        "gradient_accumulation_steps": 1,
        "data_parallel_workers": 1,
        "effective_batch_size": int(result["n_instances"]),
        "optimizer": type(optimizer).__name__,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "weight_decay": float(optimizer.param_groups[0]["weight_decay"]),
        "loss_finite": bool(np.isfinite(result["training_loss"])),
    }
    del optimizer, model
    torch.cuda.empty_cache()
    return audited


def preflight(write: bool = True) -> dict[str, Any]:
    if sha256_file(M0) != M0_SHA:
        raise Halt("canonical M0 SHA256 mismatch")
    if not U0_SOURCE.is_file() or not WORKER.is_file() or not CANONICAL_MUTATION_PYTHON.is_file():
        raise Halt("required U0, mutation worker, or canonical mutation Python is missing")
    source = hgbcore.verify_sources()
    mutation = operator_profile_audit()
    loader = dataloader_audit()
    with np.load(U0_SOURCE, allow_pickle=False) as payload:
        ucoords = np.asarray(payload["coordinates"], dtype=np.float64)
        uhashes = payload["instance_hash"].astype(str).tolist()
    optimizer_audit = production_optimizer_audit(ucoords)
    checks = {
        "M0_exact": sha256_file(M0) == M0_SHA,
        "U0_10000_unique": len(ucoords) == len(uhashes) == len(set(uhashes)) == HARD_N,
        "profile_pass": mutation["status"] == "PASS",
        "dataloader_pass": loader["status"] == "PASS",
        "production_optimizer_pass": optimizer_audit["status"] == "PASS",
        "F40_exact": len(source["F40_names"]) == 40,
        "HGB_exact": source["HGB_config_sha256"] == hgbcore.HGB_CONFIG_HASH,
        "QD4_exact": len(QD4_FEATURE_NAMES) == 4,
        "four_arms": len(ARMS) == 4,
        "five_cycles": CONFIG["cycles"] == list(range(5)),
        "disk_minimum_20GiB": free_gib() >= 20.0,
        "cuda_available": torch.cuda.is_available(),
    }
    result = {"status": "PASS" if all(checks.values()) else "BLOCKED", "checks": checks,
              "M0_sha256": M0_SHA, "U0_n": len(uhashes), "U0_membership_sha256": membership_hash(uhashes),
              "profile_sha256": sha256_file(PROFILE_PATH), "mutation_audit": mutation,
              "dataloader_audit": loader, "production_optimizer_audit": optimizer_audit,
              "F40_sha256": source["F40_sha256"],
              "HGB_config_sha256": source["HGB_config_sha256"], "free_D_GiB": free_gib(),
              "cuda_available": torch.cuda.is_available(), "created_utc": utc_now()}
    if write:
        PROTOCOL.mkdir(parents=True, exist_ok=True)
        atomic_json(PROTOCOL / "preflight_integrity.json", result)
        atomic_json(PROTOCOL / "preflight_verification.json", result)
        write_csv(PROTOCOL / "preflight_mutation_trace.csv", pd.DataFrame(mutation["trace"]))
    if result["status"] != "PASS":
        raise Halt("preflight blocked: " + ", ".join(k for k, v in checks.items() if not v))
    return result


def initialize_protocol(pre: dict[str, Any], resume: bool) -> None:
    if RUN.exists() and not resume and (PROTOCOL / "frozen_experiment_config.json").is_file():
        raise FileExistsError(f"target exists: {RUN}; use --resume")
    for path in [PROTOCOL, COMMON, SHARED_STATE, COMPARISON, FIGURES, RUNTIME]:
        path.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        for relative in ["cycles", "checkpoints", "evaluations", "manifests"]:
            (arm_dir(arm) / relative).mkdir(parents=True, exist_ok=True)
    resolved = yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8"))
    frozen = deepcopy(CONFIG)
    frozen.update({"M0_sha256": M0_SHA, "U0_membership_sha256": pre["U0_membership_sha256"],
                   "mutation_profile_sha256": pre["profile_sha256"],
                   "runner_sha256": sha256_file(Path(__file__)), "worker_sha256": sha256_file(WORKER)})
    protocol_hash = hashlib.sha256(json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    target = PROTOCOL / "frozen_experiment_config.json"
    if target.is_file():
        old = read_json(target)
        if old["protocol_sha256"] != protocol_hash:
            lock = read_json(LOCK) if LOCK.is_file() else {}
            trigger_error = str(lock.get("error", ""))
            old_science = {k: v for k, v in old.items() if k not in {"protocol_sha256", "runner_sha256", "frozen_utc", "engineering_recovery"}}
            new_science = {k: v for k, v in frozen.items() if k != "runner_sha256"}
            recovery_id = (
                "RECOVERY_001_CHECKPOINT_FILENAME_PARSE"
                if "invalid literal for int() with base 10: '000.tsp'" in trigger_error
                else "RECOVERY_002_E256_ONLY_PROGRESS_POINTER"
                if "local variable 'checkpoint' referenced before assignment" in trigger_error
                else None
            )
            recovery_only = (
                resume
                and old_science == new_science
                and lock.get("state") == "FAILED"
                and recovery_id is not None
            )
            if not recovery_only:
                raise Halt("resume protocol/source hash mismatch")
            amendment = {
                "status": "PASS",
                "recovery_id": recovery_id,
                "scientific_configuration_changed": False,
                "checkpoint_weights_changed": False,
                "incomplete_training_restarted_from_epoch_zero": recovery_id == "RECOVERY_002_E256_ONLY_PROGRESS_POINTER",
                "old_runner_sha256": old.get("runner_sha256"),
                "new_runner_sha256": frozen["runner_sha256"],
                "old_protocol_sha256": old["protocol_sha256"],
                "new_protocol_sha256": protocol_hash,
                "trigger_error": trigger_error,
                "created_utc": utc_now(),
            }
            atomic_json(PROTOCOL / f"{recovery_id}.json", amendment)
            atomic_json(target, {**frozen, "protocol_sha256": protocol_hash,
                                 "frozen_utc": old.get("frozen_utc"),
                                 "engineering_recovery": amendment["recovery_id"]})
    else:
        atomic_json(target, {**frozen, "protocol_sha256": protocol_hash, "frozen_utc": utc_now()})
    (PROTOCOL / "resolved_FOGA11_HIGH_V1.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    atomic_json(PROTOCOL / "mutation_config_SHA256.json", {"profile": PROFILE_NAME, "sha256": pre["profile_sha256"]})
    atomic_json(PROTOCOL / "mutation_design_freeze_manifest.json", pre["mutation_audit"])
    atomic_json(PROTOCOL / "common_M0_manifest.json", {"path": str(M0), "sha256": M0_SHA})
    atomic_json(PROTOCOL / "random_reuse_compatibility.json", {
        "historical_reference": str(HIST_RANDOM), "RANDOM_REUSE_COMPATIBLE": "NO",
        "RANDOM_RERUN_REQUIRED": "YES", "reason": "historical trajectory predates FOGA11_HIGH_V1 and O0_HIGH"})
    atomic_json(PROTOCOL / "corrected_training_config.json", {
        **CONFIG["training"], "training_instances_per_cycle": HARD_N,
        "full_batches_per_epoch": 39, "final_partial_batch_size": 16,
        "optimizer_steps_per_arm": STEPS_PER_CYCLE * 5,
        "optimizer_steps_four_arms": STEPS_PER_CYCLE * 5 * len(ARMS)})
    atomic_json(PROTOCOL / "frozen_mutation_config.json", {
        "design_freeze": CONFIG["design_freeze"], "profile": PROFILE_NAME,
        "profile_sha256": sha256_file(PROFILE_PATH), "resolved_profile": resolved,
        "parents_per_cycle": HARD_N, "operators": len(OPERATOR_NAMES),
        "nominal_attempts_per_cycle": HARD_N * len(OPERATOR_NAMES), "mutation_depth": 1})
    atomic_json(PROTOCOL / "common_initialization.json", {
        "M0_path": str(M0), "M0_sha256": M0_SHA,
        "U0_path": str(U0_SOURCE), "U0_membership_sha256": pre["U0_membership_sha256"],
        "shared_across_all_arms": True, "M0_external_evaluation_policy": "evaluate_once_then_share"})
    invalid_archive = ROOT / "runs" / "BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11HIGH_V2_INVALID_BATCH128"
    atomic_json(PROTOCOL / "batch_size_correction_manifest.json", {
        "status": "NEW_BATCH256_CONFIG_VERIFIED",
        "invalid_archive": str(invalid_archive),
        "invalid_archive_marker_verified": (invalid_archive / "BLADE_ABLATION_BATCH128_RUN_INVALIDATED").is_file(),
        "old_effective_batch_size": 128, "new_effective_batch_size": LOGICAL_BATCH,
        "old_updates_per_epoch": 79, "new_updates_per_epoch": STEPS_PER_EPOCH,
        "old_updates_per_cycle": 20_224, "new_updates_per_cycle": STEPS_PER_CYCLE,
        "old_checkpoint_reuse_forbidden": True, "restart_all_arms_from_M0": True,
        "seed_namespace_preserved": SEED_NAMESPACE, "created_utc": utc_now()})
    atomic_json(LOCK, {"pid": os.getpid(), "host": socket.gethostname(), "state": "ACTIVE",
                       "profile_sha256": pre["profile_sha256"], "timestamp": utc_now()})


def run_mutation_worker(parents: Path, cycle: int, output: Path) -> None:
    command = [str(CANONICAL_MUTATION_PYTHON), str(WORKER), "--parents", str(parents),
               "--cycle", str(cycle), "--output", str(output)]
    process = subprocess.run(command, cwd=ROOT)
    if process.returncode:
        raise RuntimeError(f"FOGA11_HIGH_V1 worker failed with exit {process.returncode}")


def build_union(parent_coords: np.ndarray, parent: pd.DataFrame, offspring_coords: np.ndarray,
                offspring_hashes: list[str], manifest: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    offspring = qd_frame(offspring_coords, offspring_hashes)
    accepted = manifest.valid_after_repair & manifest.proposal_status.eq("accepted") & ~manifest.duplicate
    lineage = manifest.loc[accepted, ["parent_hash", "operator_id"]].reset_index(drop=True)
    if len(lineage) != len(offspring):
        raise RuntimeError("mutation/QD alignment mismatch")
    offspring["parent_hash"] = lineage.parent_hash.astype(str)
    offspring["operator_id"] = lineage.operator_id.astype(str)
    root_lookup = dict(zip(parent.instance_hash.astype(str), parent.lineage_root.astype(str), strict=True))
    offspring["lineage_root"] = offspring.parent_hash.map(root_lookup)
    offspring["candidate_age"] = 0
    offspring["source"] = "offspring"
    offspring["coordinate_origin"] = "offspring"
    offspring["origin_index"] = np.arange(len(offspring))
    p = parent.copy()
    p["candidate_age"] = p.candidate_age.astype(int) + 1
    p["source"] = "parent"
    p["coordinate_origin"] = "parent"
    p["origin_index"] = np.arange(len(p))
    union = pd.concat([p, offspring], ignore_index=True, sort=False).drop_duplicates("instance_hash", keep="first").reset_index(drop=True)
    coords = np.empty((len(union), 100, 2), dtype=np.float64)
    pmask = union.coordinate_origin.eq("parent").to_numpy()
    coords[pmask] = parent_coords[union.loc[pmask, "origin_index"].to_numpy(int)]
    coords[~pmask] = offspring_coords[union.loc[~pmask, "origin_index"].to_numpy(int)]
    union["coord_index"] = np.arange(len(union))
    union["is_previous_parent"] = union.instance_hash.astype(str).isin(set(parent.instance_hash.astype(str)))
    union["is_new_offspring"] = union.instance_hash.astype(str).isin(set(offspring_hashes))
    return coords, union, offspring


def prepare_common(status: Status | None, resume: bool) -> dict[str, Any]:
    stage = COMMON / "_stage_common_pool"
    inputs = {"U0": sha256_file(U0_SOURCE), "profile": sha256_file(PROFILE_PATH)}
    with atomic_stage(stage, "common_U0_O0_HIGH_C0_HIGH", inputs, resume=resume) as outputs:
        if outputs is not None:
            if status:
                status.update(phase="generate_common_O0_HIGH")
            copy_or_link(U0_SOURCE, COMMON / "U0.npz")
            with np.load(COMMON / "U0.npz", allow_pickle=False) as payload:
                ucoords = np.asarray(payload["coordinates"], dtype=np.float64)
                uhashes = payload["instance_hash"].astype(str).tolist()
            parent = qd_frame(ucoords, uhashes)
            parent["lineage_root"] = parent.instance_hash.astype(str)
            parent["candidate_age"] = 0
            parent.to_parquet(COMMON / "U0_qd.parquet", index=False)
            run_mutation_worker(COMMON / "U0.npz", 0, COMMON / "O0_HIGH_mutation")
            ocoords, ohashes, _ = base._load_bundle(COMMON / "O0_HIGH_mutation" / "offspring_coordinates.npz")
            manifest = pd.read_parquet(COMMON / "O0_HIGH_mutation" / "mutation_manifest.parquet")
            ccoords, union, offspring = build_union(ucoords, parent, ocoords, ohashes, manifest)
            base._save_bundle(COMMON / "C0_HIGH.npz", ccoords, union.instance_hash.astype(str).tolist())
            union.to_parquet(COMMON / "C0_HIGH.parquet", index=False)
            offspring.to_parquet(COMMON / "O0_HIGH_qd.parquet", index=False)
            sorted_idx = np.argsort(np.asarray(ohashes), kind="stable")
            chosen = sorted_idx[np.random.default_rng(2026).choice(len(sorted_idx), 500, replace=False)]
            bcoords = ocoords[chosen]
            bmeta = offspring.set_index("instance_hash", drop=False).loc[[ohashes[int(i)] for i in chosen]].reset_index(drop=True)
            bmeta["coord_index"] = np.arange(500)
            base._save_bundle(COMMON / "B0_HIGH_unlabeled.npz", bcoords, bmeta.instance_hash.astype(str).tolist())
            bmeta.to_parquet(COMMON / "B0_HIGH_unlabeled.parquet", index=False)
            outputs.extend([COMMON / "U0.npz", COMMON / "U0_qd.parquet",
                            COMMON / "O0_HIGH_mutation" / "offspring_coordinates.npz",
                            COMMON / "O0_HIGH_mutation" / "mutation_manifest.parquet",
                            COMMON / "O0_HIGH_mutation" / "FOGA11_HIGH_V1_RUNTIME.json",
                            COMMON / "O0_HIGH_qd.parquet", COMMON / "C0_HIGH.npz", COMMON / "C0_HIGH.parquet",
                            COMMON / "B0_HIGH_unlabeled.npz", COMMON / "B0_HIGH_unlabeled.parquet"])
    with np.load(COMMON / "U0.npz", allow_pickle=False) as payload:
        uhashes = payload["instance_hash"].astype(str).tolist()
    _, ohashes, _ = base._load_bundle(COMMON / "O0_HIGH_mutation" / "offspring_coordinates.npz")
    _, chashes, _ = base._load_bundle(COMMON / "C0_HIGH.npz")
    _, bhashes, _ = base._load_bundle(COMMON / "B0_HIGH_unlabeled.npz")
    result = {"U0_n": len(uhashes), "U0_membership_sha256": membership_hash(uhashes),
              "O0_HIGH_n": len(ohashes), "O0_HIGH_membership_sha256": membership_hash(ohashes),
              "C0_HIGH_n": len(chashes), "C0_HIGH_membership_sha256": membership_hash(chashes),
              "B0_HIGH_n": len(bhashes), "B0_HIGH_membership_sha256": membership_hash(bhashes),
              "profile_sha256": sha256_file(PROFILE_PATH)}
    atomic_json(PROTOCOL / "common_U0_manifest.json", {"n": len(uhashes), "membership_sha256": result["U0_membership_sha256"]})
    atomic_json(PROTOCOL / "common_O0_HIGH_manifest.json", {"n": len(ohashes), "membership_sha256": result["O0_HIGH_membership_sha256"],
                "runtime": read_json(COMMON / "O0_HIGH_mutation" / "FOGA11_HIGH_V1_RUNTIME.json")})
    atomic_json(PROTOCOL / "common_C0_HIGH_manifest.json", {"n": len(chashes), "membership_sha256": result["C0_HIGH_membership_sha256"]})
    atomic_json(PROTOCOL / "B0_HIGH_manifest.json", {"n": len(bhashes), "membership_sha256": result["B0_HIGH_membership_sha256"],
                "selection": "sort O0 hashes, np.random.default_rng(2026).choice without replacement"})
    return result


def load_shared_cache() -> pd.DataFrame:
    pieces = [base._reusable_oracle_cache()]
    path = SHARED_STATE / "shared_oracle_cache.parquet"
    if path.is_file():
        pieces.append(base._canonical_cache_frame(pd.read_parquet(path)))
    for ledger in RUN.glob("A*/cycles/C*/07_oracle/T/ledger.parquet"):
        pieces.append(base._canonical_cache_frame(pd.read_parquet(ledger)))
    b0 = COMMON / "B0_oracle" / "ledger.parquet"
    if b0.is_file():
        pieces.append(base._canonical_cache_frame(pd.read_parquet(b0)))
    return pd.concat(pieces, ignore_index=True).drop_duplicates("instance_hash", keep="last")


def refresh_shared_cache(ledger: pd.DataFrame) -> None:
    frame = pd.concat([load_shared_cache(), base._canonical_cache_frame(ledger)], ignore_index=True).drop_duplicates("instance_hash", keep="last")
    target = SHARED_STATE / "shared_oracle_cache.parquet"
    temp = target.with_suffix(".tmp.parquet")
    frame.to_parquet(temp, index=False)
    os.replace(temp, target)
    base.GLOBAL_CACHE = frame


def label_b0(device: torch.device, status: Status, resume: bool) -> None:
    stage = COMMON / "_stage_B0_label"
    _, bhashes, _ = base._load_bundle(COMMON / "B0_HIGH_unlabeled.npz")
    inputs = {"B0": membership_hash(bhashes), "M0": M0_SHA}
    with atomic_stage(stage, "B0_HIGH_oracle_and_M0_hardness", inputs, resume=resume) as outputs:
        if outputs is not None:
            status.update(phase="label_B0_HIGH_and_compute_M0_LBQ")
            model, _ = load_canonical_checkpoint(M0, device, expected_sha256=M0_SHA)
            coords, hashes, _ = base._load_bundle(COMMON / "B0_HIGH_unlabeled.npz")
            selected = pd.read_parquet(COMMON / "B0_HIGH_unlabeled.parquet")
            base.GLOBAL_CACHE = load_shared_cache()
            labeled, tours, _ = base._label_population(COMMON / "B0_oracle", selected, coords, model, device, None,
                                                        "B0_HIGH_frozen_uniform500_then_oracle", resume)
            labeled["first_cycle_observed"] = -1
            labeled["archive_provenance"] = "B0_HIGH"
            labeled.to_parquet(COMMON / "B0_HIGH_labeled.parquet", index=False)
            base._save_bundle(COMMON / "B0_HIGH_labeled.npz", coords, hashes, tours)
            refresh_shared_cache(pd.read_parquet(COMMON / "B0_oracle" / "ledger.parquet"))
            outputs.extend([COMMON / "B0_HIGH_labeled.parquet", COMMON / "B0_HIGH_labeled.npz",
                            COMMON / "B0_oracle" / "ledger.parquet", COMMON / "B0_oracle" / "oracle_summary.json"])
            del model
            torch.cuda.empty_cache()


def load_archive(arm: str, cycle: int) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    if cycle == 0:
        coords, hashes, tours = base._load_bundle(COMMON / "B0_HIGH_labeled.npz")
        meta = pd.read_parquet(COMMON / "B0_HIGH_labeled.parquet")
    else:
        coords, hashes, tours = base._load_bundle(cycle_dir(arm, cycle - 1) / "08_archive" / "A_next.npz")
        meta = pd.read_parquet(cycle_dir(arm, cycle - 1) / "08_archive" / "A_next.parquet")
    if tours is None or hashes != meta.instance_hash.astype(str).tolist():
        raise RuntimeError("archive alignment failure")
    return coords, meta, tours


def save_archive(arm: str, cycle: int, prior_coords: np.ndarray, prior: pd.DataFrame, prior_tours: np.ndarray,
                 selected_coords: np.ndarray, labeled: pd.DataFrame, tours: np.ndarray, resume: bool) -> pd.DataFrame:
    directory = cycle_dir(arm, cycle) / "08_archive"
    ph = prior.instance_hash.astype(str).tolist()
    sh = labeled.instance_hash.astype(str).tolist()
    with atomic_stage(directory, "branch_local_cumulative_archive", {"prior": membership_hash(ph), "selected": membership_hash(sh)}, resume=resume) as outputs:
        if outputs is not None:
            seen = set(ph)
            keep = [i for i, value in enumerate(sh) if value not in seen]
            meta = labeled.copy()
            meta["first_cycle_observed"] = cycle
            meta["archive_provenance"] = f"{arm}_C{cycle:02d}"
            out_meta = pd.concat([prior, meta.iloc[keep]], ignore_index=True, sort=False)
            out_coords = np.concatenate([prior_coords, selected_coords[keep]])
            out_tours = np.concatenate([prior_tours, tours[keep]])
            base._save_bundle(directory / "A_next.npz", out_coords, out_meta.instance_hash.astype(str).tolist(), out_tours)
            out_meta.to_parquet(directory / "A_next.parquet", index=False)
            atomic_json(directory / "archive_summary.json", {"archive_n": len(out_meta), "new_unique": len(keep)})
            outputs.extend([directory / "A_next.npz", directory / "A_next.parquet", directory / "archive_summary.json"])
    return pd.read_parquet(directory / "A_next.parquet")


def prepare_candidate(arm: str, cycle: int, status: Status, resume: bool) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cdir = cycle_dir(arm, cycle)
    cdir.mkdir(parents=True, exist_ok=True)
    if cycle == 0:
        coords, hashes, _ = base._load_bundle(COMMON / "C0_HIGH.npz")
        union = pd.read_parquet(COMMON / "C0_HIGH.parquet")
        parent = pd.read_parquet(COMMON / "U0_qd.parquet")
        offspring = pd.read_parquet(COMMON / "O0_HIGH_qd.parquet")
        for source, target in [(COMMON / "O0_HIGH_mutation" / "offspring_coordinates.npz", cdir / "03_mutation" / "offspring_coordinates.npz"),
                               (COMMON / "O0_HIGH_mutation" / "mutation_manifest.parquet", cdir / "03_mutation" / "mutation_manifest.parquet"),
                               (COMMON / "O0_HIGH_mutation" / "FOGA11_HIGH_V1_RUNTIME.json", cdir / "03_mutation" / "FOGA11_HIGH_V1_RUNTIME.json")]:
            copy_or_link(source, target)
        return coords, union, parent, offspring
    stage = cdir / "_stage_candidate"
    prev = cycle_dir(arm, cycle - 1) / "06_selection" / "P_next.npz"
    _, phashes, _ = base._load_bundle(prev)
    inputs = {"parents": membership_hash(phashes), "profile": sha256_file(PROFILE_PATH), "cycle": str(cycle)}
    with atomic_stage(stage, "FOGA11_HIGH_V1_then_parent_child_union", inputs, resume=resume) as outputs:
        if outputs is not None:
            status.update(arm=arm, cycle=f"C{cycle:02d}", phase="FOGA11_HIGH_V1_mutation")
            mutation_dir = cdir / "03_mutation"
            run_mutation_worker(prev, cycle, mutation_dir)
            pcoords, phashes, _ = base._load_bundle(prev)
            parent = pd.read_parquet(cycle_dir(arm, cycle - 1) / "06_selection" / "P_next.parquet")
            ocoords, ohashes, _ = base._load_bundle(mutation_dir / "offspring_coordinates.npz")
            manifest = pd.read_parquet(mutation_dir / "mutation_manifest.parquet")
            coords, union, offspring = build_union(pcoords, parent, ocoords, ohashes, manifest)
            (cdir / "04_qd").mkdir(parents=True, exist_ok=True)
            (cdir / "05_scoring" / "union").mkdir(parents=True, exist_ok=True)
            parent.to_parquet(cdir / "04_qd" / "parent_qd.parquet", index=False)
            offspring.to_parquet(cdir / "04_qd" / "offspring_qd.parquet", index=False)
            union.to_parquet(cdir / "05_scoring" / "union" / "C_union.parquet", index=False)
            base._save_bundle(cdir / "05_scoring" / "union" / "C_union.npz", coords, union.instance_hash.astype(str).tolist())
            outputs.extend([mutation_dir / "offspring_coordinates.npz", mutation_dir / "mutation_manifest.parquet",
                            mutation_dir / "FOGA11_HIGH_V1_RUNTIME.json", cdir / "04_qd" / "parent_qd.parquet",
                            cdir / "04_qd" / "offspring_qd.parquet", cdir / "05_scoring" / "union" / "C_union.parquet",
                            cdir / "05_scoring" / "union" / "C_union.npz"])
    coords, hashes, _ = base._load_bundle(cdir / "05_scoring" / "union" / "C_union.npz")
    union = pd.read_parquet(cdir / "05_scoring" / "union" / "C_union.parquet")
    return coords, union, pd.read_parquet(cdir / "04_qd" / "parent_qd.parquet"), pd.read_parquet(cdir / "04_qd" / "offspring_qd.parquet")


def hgb_scores(arm: str, cycle: int, model: torch.nn.Module, model_hash: str, device: torch.device,
               coords: np.ndarray, union: pd.DataFrame, source: dict[str, Any], resume: bool) -> tuple[np.ndarray, dict[str, Any]]:
    configure_hgb_context(arm)
    acoords, ameta, atours = load_archive(arm, cycle)
    hgbcore.prepare_archive_input(cycle, acoords, ameta, atours)
    b, calibration, _ = hgbcore.calibration_stage(cycle, model, model_hash, device, acoords, ameta, atours, resume)
    manifest = hgbcore.fit_hgb_stage(cycle, b, source, resume)
    scores, scoring = hgbcore.score_stage(cycle, coords, union, source, resume)
    return scores, {"calibration": calibration, "hgb": manifest, "scoring": scoring,
                    "archive_coords": acoords, "archive_meta": ameta, "archive_tours": atours}


def select_rows(arm: str, cycle: int, union: pd.DataFrame, scores: np.ndarray | None) -> pd.DataFrame:
    hashes = union.instance_hash.astype(str).to_numpy()
    if len(union) < HARD_N or union.instance_hash.astype(str).duplicated().any():
        raise RuntimeError("invalid unique candidate pool")
    if arm == "A0_FULL":
        assert scores is not None
        work = pd.DataFrame({"index": np.arange(len(union)), "niche": union.niche_id.astype(int), "score": scores, "hash": hashes})
        elite = work.sort_values(["niche", "score", "hash"], ascending=[True, False, True], kind="stable").drop_duplicates("niche")
        chosen = elite["index"].astype(int).tolist()
        chosen_set = set(chosen)
        fill = [int(i) for i in np.lexsort((hashes, -scores)) if int(i) not in chosen_set][:HARD_N - len(chosen)]
        indices = np.asarray(chosen + fill, dtype=int)
        components = ["qd_hgb_elite"] * len(chosen) + ["global_hgb_fill"] * len(fill)
    elif arm == "A1_WITHOUT_QD":
        assert scores is not None
        indices = np.lexsort((hashes, -scores))[:HARD_N]
        components = ["global_hgb"] * HARD_N
    elif arm == "A2_WITHOUT_SURROGATE":
        rng = np.random.default_rng(deterministic_seed(arm, f"C{cycle:02d}", "selection"))
        chosen = []
        for niche in sorted(union.niche_id.astype(int).unique()):
            options = np.flatnonzero(union.niche_id.to_numpy(int) == niche)
            chosen.append(int(rng.choice(options)))
        chosen_set = set(chosen)
        remaining = np.asarray([i for i in range(len(union)) if i not in chosen_set], dtype=int)
        fill = rng.choice(remaining, size=HARD_N - len(chosen), replace=False).astype(int).tolist()
        indices = np.asarray(chosen + fill, dtype=int)
        components = ["qd_uniform_representative"] * len(chosen) + ["uniform_fill"] * len(fill)
    else:
        rng = np.random.default_rng(deterministic_seed(arm, f"C{cycle:02d}", "selection"))
        indices = rng.choice(len(union), size=HARD_N, replace=False).astype(int)
        components = ["global_uniform"] * HARD_N
    selected = union.iloc[indices].copy().reset_index(drop=True)
    selected["component"] = components
    selected["selection_rank"] = np.arange(1, HARD_N + 1)
    selected["hgb_predicted_log1p_hardness"] = np.nan if scores is None else np.asarray(scores)[indices]
    if selected.instance_hash.astype(str).duplicated().any():
        raise RuntimeError("selection contains duplicates")
    return selected


def selection_stage(arm: str, cycle: int, coords: np.ndarray, union: pd.DataFrame,
                    scores: np.ndarray | None, resume: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    directory = cycle_dir(arm, cycle) / "06_selection"
    score_hash = "NONE" if scores is None else hashlib.sha256(np.ascontiguousarray(scores).tobytes()).hexdigest()
    with atomic_stage(directory, f"{arm}_frozen_selection", {"candidate": membership_hash(union.instance_hash.astype(str).tolist()), "scores": score_hash}, resume=resume) as outputs:
        if outputs is not None:
            selected = select_rows(arm, cycle, union, scores)
            selected.to_parquet(directory / "T_selected.parquet", index=False)
            selected.to_parquet(directory / "P_next.parquet", index=False)
            indices = selected.coord_index.to_numpy(int)
            base._save_bundle(directory / "T_selected.npz", coords[indices], selected.instance_hash.astype(str).tolist())
            base._save_bundle(directory / "P_next.npz", coords[indices], selected.instance_hash.astype(str).tolist())
            counts = selected.component.value_counts().to_dict()
            info = {"arm": arm, "cycle": f"C{cycle:02d}", "T_n": HARD_N, "P_n": HARD_N,
                    "P_equals_T": True, "membership_sha256": membership_hash(selected.instance_hash.astype(str).tolist()),
                    "component_counts": counts, "occupied_niches": int(selected.niche_id.nunique()),
                    "selection_seed": None if arm in HGB_ARMS else deterministic_seed(arm, f"C{cycle:02d}", "selection")}
            atomic_json(directory / "selection_summary.json", info)
            outputs.extend([directory / "T_selected.parquet", directory / "P_next.parquet",
                            directory / "T_selected.npz", directory / "P_next.npz", directory / "selection_summary.json"])
    return pd.read_parquet(directory / "T_selected.parquet"), read_json(directory / "selection_summary.json")


def label_selected(arm: str, cycle: int, model: torch.nn.Module, device: torch.device,
                   coords: np.ndarray, selected: pd.DataFrame, archive: pd.DataFrame | None,
                   resume: bool) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    directory = cycle_dir(arm, cycle) / "07_oracle" / "T"
    indices = selected.coord_index.to_numpy(int)
    tcoords = coords[indices]
    base.GLOBAL_CACHE = load_shared_cache()
    known = set(base.GLOBAL_CACHE.instance_hash.astype(str))
    labeled, tours, summary = base._label_population(directory, selected, tcoords, model, device, archive,
                                                      f"{arm}_T_frozen_then_oracle", resume)
    ledger = pd.read_parquet(directory / "ledger.parquet")
    refresh_shared_cache(ledger)
    hits = sum(h in known for h in selected.instance_hash.astype(str))
    summary = {**summary, "historical_or_shared_cache_hits": hits,
               "physical_new_concorde_calls": len(selected) - hits,
               "selection_frozen_before_oracle": True}
    atomic_json(directory / "oracle_budget.json", summary)
    return labeled, tours, summary


def save_checkpoint(model: torch.nn.Module, path: Path, arm: str, cycle: int, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"net": model.state_dict(), "epoch": epoch, "formal_scientific_checkpoint": epoch == EPOCHS,
                "metadata": {"experiment": RUN_ID, "arm": arm, "cycle": cycle, "seed": 2026,
                             "learning_rate": LR, "hard_unique_count": HARD_N,
                             "hard_occurrences_per_epoch": HARD_N, "logical_batch_size": LOGICAL_BATCH,
                             "effective_batch_size": LOGICAL_BATCH,
                             "gradient_accumulation_steps": 1, "data_parallel_workers": 1,
                             "checkpoint_policy": "E256_only",
                             "uniform_anchor_count": 0, "diagnostic_only": epoch != EPOCHS}}, path)


def checkpoint_epoch(path: Path) -> int:
    return int(path.name.split("_E")[-1].split(".")[0])


def recover_completed_training_report(directory: Path, official: Path, arm: str, cycle: int,
                                      inputs: dict[str, str], device: torch.device) -> None:
    done = directory / "STAGE_DONE.json"
    failed = directory / "STAGE_FAILED.json"
    curve_path = directory / "training_loss_curve.csv"
    checkpoints = [directory / f"M{cycle + 1}_E{epoch:03d}.tsp.best" for epoch in sorted(CHECKPOINT_EPOCHS)]
    if done.is_file() or not (failed.is_file() and official.is_file() and curve_path.is_file()):
        return
    failure = read_json(failed)
    if "invalid literal for int() with base 10" not in str(failure.get("exception", "")):
        return
    frame = pd.read_csv(curve_path)
    valid = (
        len(frame) == EPOCHS
        and frame.optimizer_updates.eq(STEPS_PER_EPOCH).all()
        and frame.hard_unique_instances.eq(HARD_N).all()
        and frame.hard_instances_seen.eq(HARD_N).all()
        and frame.anchor_instances_seen.eq(0).all()
        and all(path.is_file() for path in checkpoints)
        and sha256_file(checkpoints[-1]) == sha256_file(official)
    )
    if not valid:
        raise Halt("failed training stage is not eligible for reporting-only recovery")
    loaded, meta = load_canonical_checkpoint(official, device, expected_sha256=None)
    finite = all(torch.isfinite(parameter).all().item() for parameter in loaded.parameters())
    del loaded
    if not finite:
        raise Halt("reporting-only recovery found non-finite E256 checkpoint")
    checkpoint_path = directory / "checkpoint_hashes.csv"
    write_csv(checkpoint_path, pd.DataFrame([
        {"epoch": checkpoint_epoch(path), "path": str(path), "sha256": sha256_file(path),
         "official": checkpoint_epoch(path) == EPOCHS}
        for path in checkpoints
    ]))
    epoch_times = frame.epoch_wall_s.to_numpy(float)
    timing = {"hard_unique": HARD_N, "hard_occurrences_per_epoch": HARD_N, "uniform_anchor": 0,
              "logical_batch_size": LOGICAL_BATCH, "effective_batch_size": LOGICAL_BATCH,
              "microbatch_size": MICROBATCH_SIZE, "microbatches_per_optimizer_update": 4,
              "gradient_accumulation_steps": 1, "data_parallel_workers": 1,
              "drop_last": False, "full_batches_per_epoch": 39, "final_partial_batch_size": 16,
              "epochs": EPOCHS,
              "optimizer_steps_per_epoch": STEPS_PER_EPOCH,
              "optimizer_steps": int(frame.optimizer_updates.sum()),
              "cumulative_optimizer_steps": STEPS_PER_CYCLE * (cycle + 1),
              "training_loop_wall_s": float(epoch_times.sum()),
              "final_loss": float(frame.iloc[-1].mean_training_loss),
              "checkpoint_sha256": meta["sha256"], "model_finite": True,
              "gpu_peak_allocated_bytes": None,
              "audited_recovery": "RECOVERY_001_CHECKPOINT_FILENAME_PARSE"}
    timing_path = directory / "bq_training_timing.json"
    atomic_json(timing_path, timing)
    recovery_path = directory / "RECOVERY_001_CHECKPOINT_FILENAME_PARSE.json"
    atomic_json(recovery_path, {"status": "PASS", "scientific_training_reused": True,
                "checkpoint_weights_changed": False, "checkpoint_sha256": meta["sha256"],
                "epochs_verified": EPOCHS, "optimizer_steps_verified": int(frame.optimizer_updates.sum()),
                "arm": arm, "cycle": f"C{cycle:02d}", "created_utc": utc_now()})
    outputs = checkpoints + [official, curve_path, timing_path, checkpoint_path, recovery_path]
    records = {str(path.resolve()): {"path": str(path.resolve()), "bytes": path.stat().st_size,
                                    "sha256": sha256_file(path)} for path in outputs}
    atomic_json(done, {"stage": "BQ_10k_once_batch256_no_anchor_E256", "status": "COMPLETE",
                "input_hashes": inputs, "outputs": records,
                "start_timestamp": failure.get("start_timestamp"), "end_timestamp": utc_now(),
                "wall_seconds": failure.get("wall_seconds"),
                "audited_recovery": "RECOVERY_001_CHECKPOINT_FILENAME_PARSE"})


def train(arm: str, cycle: int, model: torch.nn.Module, model_hash: str, coordinates: np.ndarray,
          hashes: list[str], tours: np.ndarray, device: torch.device, status: Status, resume: bool) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    directory = cycle_dir(arm, cycle) / "09_bq_training"
    official = model_path(arm, cycle + 1)
    inputs = {"model": model_hash, "T": membership_hash(hashes), "lr": str(LR), "epochs": str(EPOCHS),
              "sampler": "one_permutation_10k_batch256_drop_last_false"}
    if resume:
        recover_completed_training_report(directory, official, arm, cycle, inputs, device)
    with atomic_stage(directory, "BQ_10k_once_batch256_no_anchor_E256", inputs, resume=resume) as outputs:
        if outputs is not None:
            if len(coordinates) != HARD_N or len(tours) != HARD_N or len(set(hashes)) != HARD_N:
                raise RuntimeError("training set size/uniqueness mismatch")
            closed = closed_expert_coordinates(coordinates, tours)
            rng = np.random.default_rng(2026)
            optimizer = create_optimizer(model, LR)
            history = []
            checkpoints = []
            e0 = directory / f"M{cycle + 1}_E000.tsp.best"
            save_checkpoint(model, e0, arm, cycle, 0)
            checkpoints.append(e0)
            latest_checkpoint = e0
            started = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for epoch in range(1, EPOCHS + 1):
                epoch_started = time.perf_counter()
                order = rng.permutation(HARD_N)
                losses = []
                updates = 0
                for start in range(0, HARD_N, LOGICAL_BATCH):
                    batch = closed[order[start:start + LOGICAL_BATCH]]
                    suffix = sample_suffix_length(batch.shape[1], rng)
                    result = supervised_optimizer_step(model, optimizer, batch, suffix, device)
                    losses.append(float(result["training_loss"]))
                    updates += 1
                if updates != STEPS_PER_EPOCH:
                    raise RuntimeError(f"optimizer step mismatch: {updates}")
                row = {"epoch": epoch, "mean_training_loss": float(np.mean(losses)),
                       "optimizer_updates": updates, "hard_unique_instances": HARD_N,
                       "hard_instances_seen": HARD_N, "anchor_instances_seen": 0,
                       "samples_processed": HARD_N, "batches_processed": updates,
                       "epoch_wall_s": time.perf_counter() - epoch_started}
                history.append(row)
                if epoch in CHECKPOINT_EPOCHS:
                    checkpoint = directory / f"M{cycle + 1}_E{epoch:03d}.tsp.best"
                    save_checkpoint(model, checkpoint, arm, cycle, epoch)
                    checkpoints.append(checkpoint)
                    latest_checkpoint = checkpoint
                if epoch in PROGRESS_EPOCHS:
                    print(f"[{RUN_ID}][{arm}][C{cycle:02d}] E{epoch} loss={row['mean_training_loss']:.8f}", flush=True)
                    status.update(arm=arm, cycle=f"C{cycle:02d}", phase=f"BQ_E{epoch}", latest_checkpoint=str(latest_checkpoint))
            copy_or_link(directory / f"M{cycle + 1}_E256.tsp.best", official)
            frame = pd.DataFrame(history)
            frame.to_csv(directory / "training_loss_curve.csv", index=False)
            timing = {"hard_unique": HARD_N, "hard_occurrences_per_epoch": HARD_N, "uniform_anchor": 0,
                      "logical_batch_size": LOGICAL_BATCH, "effective_batch_size": LOGICAL_BATCH,
                      "microbatch_size": MICROBATCH_SIZE, "microbatches_per_optimizer_update": 4,
                      "gradient_accumulation_steps": 1, "data_parallel_workers": 1,
                      "drop_last": False, "full_batches_per_epoch": 39, "final_partial_batch_size": 16,
                      "epochs": EPOCHS,
                      "optimizer_steps_per_epoch": STEPS_PER_EPOCH,
                      "optimizer_steps": int(frame.optimizer_updates.sum()),
                      "cumulative_optimizer_steps": STEPS_PER_CYCLE * (cycle + 1),
                      "training_loop_wall_s": time.perf_counter() - started,
                      "final_loss": float(frame.iloc[-1].mean_training_loss),
                      "checkpoint_sha256": sha256_file(official),
                      "model_finite": all(torch.isfinite(p).all().item() for p in model.parameters()),
                      "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0}
            atomic_json(directory / "bq_training_timing.json", timing)
            write_csv(directory / "checkpoint_hashes.csv", pd.DataFrame([
                {"epoch": checkpoint_epoch(p), "path": str(p), "sha256": sha256_file(p),
                 "official": checkpoint_epoch(p) == EPOCHS}
                for p in checkpoints]))
            outputs.extend(checkpoints + [official, directory / "training_loss_curve.csv",
                                           directory / "bq_training_timing.json", directory / "checkpoint_hashes.csv"])
    loaded, meta = load_canonical_checkpoint(official, device, expected_sha256=None)
    timing = read_json(directory / "bq_training_timing.json")
    if timing["optimizer_steps"] != STEPS_PER_CYCLE or meta["sha256"] != timing["checkpoint_sha256"] or not timing["model_finite"]:
        raise RuntimeError("completed training verification failed")
    return loaded, meta, timing


def evaluate(model: torch.nn.Module, checkpoint: Path, label: str, output: Path, status: Status, arm: str | None) -> tuple[pd.DataFrame, float]:
    checkpoint_hash = sha256_file(checkpoint)
    rows = []
    started = time.perf_counter()
    work = output.parent / "evaluation_work" / label
    for dataset in BENCHMARKS:
        status.update(arm=arm, phase=f"evaluate_{label}_{dataset}")
        directory = work / dataset
        summary_path = directory / "summary.json"
        if summary_path.is_file() and read_json(summary_path).get("checkpoint_sha256") == checkpoint_hash:
            rows.append(read_json(summary_path))
            continue
        directory.mkdir(parents=True, exist_ok=True)
        payload = evaluator.load_fixed_references(dataset)
        if dataset == "OriginalTSPLIB70":
            detail, result = evaluator.evaluate_original70(model, model.device if hasattr(model, "device") else next(model.parameters()).device, payload, checkpoint_hash)
        else:
            detail, result = evaluator.evaluate_tensor_dataset(model, next(model.parameters()).device, dataset, payload, checkpoint_hash)
        result.update({"model": label, "dataset": dataset, "checkpoint_sha256": checkpoint_hash,
                       "passive_only": True, "controls_training": False})
        detail.insert(0, "model", label)
        detail.insert(1, "dataset", dataset)
        detail.to_parquet(directory / "instance_metrics.parquet", index=False)
        atomic_json(summary_path, result)
        rows.append(result)
    frame = pd.DataFrame(rows)
    write_csv(output, frame)
    return frame, time.perf_counter() - started


def niche_metrics(values: pd.Series) -> dict[str, float]:
    counts = values.value_counts().to_numpy(float)
    p = counts / counts.sum()
    return {"occupied_niches": int(len(counts)), "N_eff": float(1 / np.sum(p * p)),
            "largest_niche_share": float(p.max())}


def lineage_metrics(values: pd.Series) -> dict[str, float]:
    counts = values.astype(str).value_counts().to_numpy(float)
    p = counts / counts.sum()
    return {"lineage_N_eff": float(1 / np.sum(p * p)), "largest_lineage_share": float(p.max())}


def mutation_statistics(arm: str, cycle: int) -> pd.DataFrame:
    manifest = pd.read_parquet(cycle_dir(arm, cycle) / "03_mutation" / "mutation_manifest.parquet")
    rows = []
    for operator in OPERATOR_NAMES:
        part = manifest[manifest.operator_id.eq(operator)]
        accepted = part.valid_after_repair & part.proposal_status.eq("accepted")
        row = {"arm": arm, "cycle": f"C{cycle:02d}", "operator_id": operator, "attempts": len(part),
               "valid": int(accepted.sum()), "unique": int((accepted & ~part.duplicate).sum()),
               "duplicate": int(part.duplicate.sum()), "repair_count": int(part.repair_count.sum()),
               "valid_rate": float(accepted.mean()), "unique_rate": float((accepted & ~part.duplicate).mean()),
               "wall_s": float(part.total_operator_s.sum())}
        for column in ["implosion_radius", "implosion_strength", "axis_affected_fraction", "axis_target", "axis_jitter"]:
            values = part[column].dropna().to_numpy(float)
            row[f"{column}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{column}_min"] = float(values.min()) if len(values) else np.nan
            row[f"{column}_max"] = float(values.max()) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def cycle_complete(arm: str, cycle: int) -> bool:
    marker = cycle_dir(arm, cycle) / "COMPLETE.json"
    if not marker.is_file():
        return False
    value = read_json(marker)
    for relative, expected in value["required_files"].items():
        path = cycle_dir(arm, cycle) / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise Halt(f"completed-cycle artifact conflict: {path}")
    return True


def run_cycle(arm: str, cycle: int, device: torch.device, source: dict[str, Any], status: Status, resume: bool) -> None:
    if cycle_complete(arm, cycle):
        status.update(arm=arm, cycle=f"C{cycle:02d}", phase="verified_complete_skip", latest_checkpoint=str(model_path(arm, cycle + 1)))
        return
    if free_gib() < 5:
        raise Halt(f"disk safety stop: {free_gib():.2f} GiB free")
    started = time.perf_counter()
    status.update(arm=arm, cycle=f"C{cycle:02d}", phase="load_model_and_candidate")
    model, meta = load_canonical_checkpoint(model_path(arm, cycle), device, expected_sha256=M0_SHA if cycle == 0 else None)
    coords, union, parent_qd, offspring_qd = prepare_candidate(arm, cycle, status, resume)
    hgb_info: dict[str, Any] = {}
    scores: np.ndarray | None = None
    if arm in HGB_ARMS:
        status.update(phase="refresh_archive_fit_HGB40_score_candidates")
        scores, hgb_info = hgb_scores(arm, cycle, model, meta["sha256"], device, coords, union, source, resume)
    status.update(phase="freeze_selection")
    selected, selection = selection_stage(arm, cycle, coords, union, scores, resume)
    archive_meta = hgb_info.get("archive_meta")
    status.update(phase="oracle_labels_and_current_hardness")
    labeled, tours, oracle = label_selected(arm, cycle, model, device, coords, selected, archive_meta, resume)
    tcoords = coords[selected.coord_index.to_numpy(int)]
    archive_post_n = None
    if arm in HGB_ARMS:
        archive_post = save_archive(arm, cycle, hgb_info["archive_coords"], hgb_info["archive_meta"], hgb_info["archive_tours"],
                                    tcoords, labeled, tours, resume)
        archive_post_n = len(archive_post)
    status.update(phase="BQ_E256_training")
    new_model, new_meta, training = train(arm, cycle, model, meta["sha256"], tcoords,
                                          selected.instance_hash.astype(str).tolist(), tours, device, status, resume)
    status.update(phase="passive_evaluation")
    evaluation, eval_s = evaluate(new_model, model_path(arm, cycle + 1), f"M{cycle + 1}",
                                  cycle_dir(arm, cycle) / "external_benchmark_results.csv", status, arm)
    qd = niche_metrics(selected.niche_id)
    lineage = lineage_metrics(selected.lineage_root)
    mutation = mutation_statistics(arm, cycle)
    write_csv(cycle_dir(arm, cycle) / "operator_statistics.csv", mutation)
    summary = {"arm": arm, "cycle": f"C{cycle:02d}", "model_before": f"M{cycle}", "model_after": f"M{cycle+1}",
               "candidate_n": len(union), "parent_n": len(parent_qd), "offspring_unique_n": len(offspring_qd),
               "T_n": HARD_N, "P_n": HARD_N, "P_equals_T": True,
               "selection": selection, "QD": qd, "lineage": lineage,
               "archive_post_n": archive_post_n, "oracle": oracle, "training": training,
               "evaluation_s": eval_s, "cycle_wall_s": time.perf_counter() - started,
               "effective_batch_size": LOGICAL_BATCH, "training_set_size": HARD_N,
               "epochs": EPOCHS, "minibatches_per_epoch": STEPS_PER_EPOCH,
               "optimizer_steps_per_cycle": STEPS_PER_CYCLE,
               "cumulative_optimizer_steps": STEPS_PER_CYCLE * (cycle + 1),
               "training_data_sha256": membership_hash(selected.instance_hash.astype(str).tolist()),
               "parent_membership_sha256": membership_hash(parent_qd.instance_hash.astype(str).tolist()),
               "training_membership_sha256": membership_hash(selected.instance_hash.astype(str).tolist()),
               "input_checkpoint_sha256": meta["sha256"], "output_checkpoint_sha256": new_meta["sha256"],
               "checkpoint_sha256": new_meta["sha256"], "profile_sha256": sha256_file(PROFILE_PATH),
               "benchmark_results_sha256": sha256_file(cycle_dir(arm, cycle) / "external_benchmark_results.csv"),
               "uniform_anchor_n": 0, "mutation_depth": 1, "nominal_mutation_attempts": 110_000}
    atomic_json(cycle_dir(arm, cycle) / "cycle_summary.json", summary)
    required = ["06_selection/T_selected.parquet", "06_selection/P_next.parquet", "07_oracle/T/labeled_bundle.npz",
                "09_bq_training/training_loss_curve.csv", "external_benchmark_results.csv", "operator_statistics.csv",
                "cycle_summary.json"]
    records = {relative: sha256_file(cycle_dir(arm, cycle) / relative) for relative in required}
    atomic_json(cycle_dir(arm, cycle) / "COMPLETE.json", {"status": "COMPLETE", "arm": arm,
                "cycle": f"C{cycle:02d}", "required_files": records, "completed_utc": utc_now()})
    status.update(phase="COMPLETE", latest_checkpoint=str(model_path(arm, cycle + 1)))
    print(f"{RUN_ID} {arm} C{cycle:02d} M{cycle}->M{cycle+1} COMPLETE", flush=True)
    del model, new_model
    torch.cuda.empty_cache()


def ensure_baseline(device: torch.device, status: Status) -> pd.DataFrame:
    output = COMMON / "M0_external_benchmark_results.csv"
    if output.is_file():
        frame = pd.read_csv(output)
        valid = (
            len(frame) == len(BENCHMARKS)
            and set(frame.dataset) == set(BENCHMARKS)
            and frame.model.astype(str).eq("M0").all()
            and frame.checkpoint_sha256.astype(str).eq(M0_SHA).all()
        )
        if valid:
            atomic_json(COMMON / "M0_EVALUATION_ONCE.json", {
                "status": "PASS", "policy": "evaluate_once_then_share_across_four_arms",
                "physical_evaluations_in_corrected_run": 0,
                "reused_verified_M0_results": True, "M0_sha256": M0_SHA,
                "result_sha256": sha256_file(output), "datasets": list(BENCHMARKS)})
            return frame
        raise Halt("existing M0 evaluation failed identity checks; refusing an automatic repeat")
    model, _ = load_canonical_checkpoint(M0, device, expected_sha256=M0_SHA)
    frame, _ = evaluate(model, M0, "M0", output, status, None)
    atomic_json(COMMON / "M0_EVALUATION_ONCE.json", {
        "status": "PASS", "policy": "evaluate_once_then_share_across_four_arms",
        "physical_evaluations_in_corrected_run": 1,
        "reused_verified_M0_results": False, "M0_sha256": M0_SHA,
        "result_sha256": sha256_file(output), "datasets": list(BENCHMARKS)})
    del model
    torch.cuda.empty_cache()
    return frame


def assemble_tables() -> dict[str, pd.DataFrame]:
    eval_rows = []
    m0 = pd.read_csv(COMMON / "M0_external_benchmark_results.csv")
    for arm in ARMS:
        base_frame = m0.copy()
        base_frame["arm"] = arm
        eval_rows.append(base_frame)
        for cycle in range(5):
            frame = pd.read_csv(cycle_dir(arm, cycle) / "external_benchmark_results.csv")
            frame["arm"] = arm
            eval_rows.append(frame)
    evaluations = pd.concat(eval_rows, ignore_index=True)
    all_gaps = evaluations[["arm", "model", "dataset", "mean_gap_percent", "checkpoint_sha256"]].copy()
    all_gaps["model_index"] = all_gaps.model.str.extract(r"(\d+)").astype(int)
    pivot = all_gaps.pivot_table(index=["arm", "model", "model_index"], columns="dataset", values="mean_gap_percent").reset_index()
    pivot["GAP8"] = pivot[list(BENCHMARKS)].mean(axis=1)
    mutation = pd.concat([mutation_statistics(arm, cycle) for arm in ARMS for cycle in range(5)], ignore_index=True)
    cycles = pd.DataFrame([read_json(cycle_dir(arm, cycle) / "cycle_summary.json") for arm in ARMS for cycle in range(5)])
    selection_rows = []
    diversity_rows = []
    training_rows = []
    candidate_rows = []
    cost_rows = []
    for row in cycles.to_dict("records"):
        selection_rows.append({"arm": row["arm"], "cycle": row["cycle"], **row["selection"]})
        diversity_rows.append({"arm": row["arm"], "cycle": row["cycle"], **row["QD"], **row["lineage"]})
        training_rows.append({"arm": row["arm"], "cycle": row["cycle"], **row["training"]})
        candidate_rows.append({"arm": row["arm"], "cycle": row["cycle"], "parent_n": row["parent_n"],
                               "offspring_unique_n": row["offspring_unique_n"], "candidate_n": row["candidate_n"]})
        cost_rows.append({"arm": row["arm"], "cycle": row["cycle"], "cycle_wall_s": row["cycle_wall_s"],
                          "training_s": row["training"]["training_loop_wall_s"], "evaluation_s": row["evaluation_s"],
                          "oracle_s": row["oracle"]["wall_seconds"]})
    selection = pd.DataFrame(selection_rows)
    diversity = pd.DataFrame(diversity_rows)
    training = pd.DataFrame(training_rows)
    candidates = pd.DataFrame(candidate_rows)
    costs = pd.DataFrame(cost_rows)
    m5 = pivot[pivot.model.eq("M5")].copy().sort_values("arm")
    arms = {row.arm: row for row in m5.itertuples()}
    factorial_rows = []
    for metric in [*BENCHMARKS, "GAP8"]:
        g11, g10, g01, g00 = [float(getattr(arms[a], metric)) for a in ARMS]
        factorial_rows.extend([
            {"metric": metric, "effect": "QD_with_surrogate_G11_minus_G10", "value_pp": g11 - g10},
            {"metric": metric, "effect": "QD_without_surrogate_G01_minus_G00", "value_pp": g01 - g00},
            {"metric": metric, "effect": "Surrogate_with_QD_G11_minus_G01", "value_pp": g11 - g01},
            {"metric": metric, "effect": "Surrogate_without_QD_G10_minus_G00", "value_pp": g10 - g00},
            {"metric": metric, "effect": "Interaction_G11_minus_G10_minus_G01_plus_G00", "value_pp": g11 - g10 - g01 + g00},
        ])
    factorial = pd.DataFrame(factorial_rows)
    protocol = pd.DataFrame([{"arm": arm, "surrogate": CONFIG["arms"][arm]["surrogate"] or "OFF",
                              "QD": CONFIG["arms"][arm]["QD"], "profile": PROFILE_NAME,
                              "cycles": 5, "training_n": HARD_N, "anchor_n": 0,
                              "steps_per_cycle": STEPS_PER_CYCLE} for arm in ARMS])
    tables = {"01_protocol_comparison.csv": protocol,
              "02_mutation_profile_verification.csv": pd.DataFrame([operator_profile_audit()["checks"]]),
              "03_operator_parameter_distributions.csv": mutation.filter(regex="arm|cycle|operator|radius|strength|axis_"),
              "04_operator_validity_uniqueness.csv": mutation,
              "05_common_initialization_integrity.csv": pd.DataFrame([read_json(PROTOCOL / "common_C0_HIGH_manifest.json")]),
              "06_training_budget_comparison.csv": training,
              "07_candidate_generation_trajectory.csv": candidates,
              "08_selection_statistics.csv": selection,
              "09_Original70_gap_trajectory.csv": pivot[["arm", "model", "model_index", "OriginalTSPLIB70"]],
              "10_TSPLIB100_gap_trajectory.csv": pivot[["arm", "model", "model_index", "TSPLIB100"]],
              "11_Uniform100_gap_trajectory.csv": pivot[["arm", "model", "model_index", "Uniform100"]],
              "12_all_eight_benchmark_gaps.csv": all_gaps,
              "13_M5_four_arm_comparison.csv": m5,
              "14_factorial_effects.csv": factorial,
              "15_QD_diversity_trajectory.csv": diversity[["arm", "cycle", "occupied_niches", "N_eff", "largest_niche_share"]],
              "16_lineage_diversity_trajectory.csv": diversity[["arm", "cycle", "lineage_N_eff", "largest_lineage_share"]],
              "17_mutation_costs.csv": mutation[["arm", "cycle", "operator_id", "wall_s"]],
              "18_total_compute_costs.csv": costs,
              "19_final_decision_summary.csv": m5[["arm", "OriginalTSPLIB70", "TSPLIB100", "Uniform100", "GAP8"]]}
    for name, frame in tables.items():
        write_csv(RUN / name, frame)
    return {**tables, "pivot": pivot, "factorial": factorial, "diversity": diversity, "costs": costs, "mutation": mutation}


def make_figures(tables: dict[str, pd.DataFrame]) -> None:
    pivot = tables["pivot"]
    def trajectory(metric: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        for arm in ARMS:
            part = pivot[pivot.arm.eq(arm)].sort_values("model_index")
            ax.plot(part.model_index, part[metric], marker="o", label=arm, color=COLORS[arm])
        ax.set(xlabel="Model", ylabel=f"{metric} mean gap (%)", xticks=range(6))
        ax.legend(fontsize=8); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIGURES / filename, dpi=170); plt.close(fig)
    trajectory("OriginalTSPLIB70", "fig01_Original70_four_arm_trajectory.png")
    trajectory("TSPLIB100", "fig02_TSPLIB100_four_arm_trajectory.png")
    trajectory("Uniform100", "fig03_Uniform100_four_arm_trajectory.png")
    trajectory("GAP8", "fig04_eight_set_average_trajectory.png")
    m5 = tables["13_M5_four_arm_comparison.csv"]
    fig, ax = plt.subplots(figsize=(8, 5)); ax.bar(m5.arm, m5.OriginalTSPLIB70, color=[COLORS[a] for a in m5.arm]); ax.tick_params(axis="x", rotation=18); ax.set_ylabel("Original70 M5 gap (%)"); fig.tight_layout(); fig.savefig(FIGURES / "fig05_Original70_M5_comparison.png", dpi=170); plt.close(fig)
    bench = m5.set_index("arm")[list(BENCHMARKS)].T
    fig, ax = plt.subplots(figsize=(11, 6)); bench.plot(kind="bar", ax=ax, color=[COLORS[a] for a in ARMS]); ax.set_ylabel("M5 mean gap (%)"); fig.tight_layout(); fig.savefig(FIGURES / "fig06_M5_all_benchmarks.png", dpi=170); plt.close(fig)
    diversity = tables["diversity"].copy(); diversity["cycle_index"] = diversity.cycle.str.extract(r"(\d+)").astype(int)
    for metric, filename in [("N_eff", "fig07_QD_niche_Neff.png"), ("lineage_N_eff", "fig08_lineage_Neff.png")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for arm in ARMS:
            part = diversity[diversity.arm.eq(arm)]; ax.plot(part.cycle_index, part[metric], marker="o", label=arm, color=COLORS[arm])
        ax.legend(fontsize=8); ax.set(xlabel="Cycle", ylabel=metric); fig.tight_layout(); fig.savefig(FIGURES / filename, dpi=170); plt.close(fig)
    mutation = tables["mutation"]
    for metric, filename, ylabel in [("unique_rate", "fig09_offspring_unique_rate.png", "Unique rate"),
                                      ("valid_rate", "fig10_FOGA11_operator_validity.png", "Valid rate"),
                                      ("wall_s", "fig11_mutation_time_by_operator.png", "Operator wall seconds")]:
        data = mutation.groupby(["arm", "operator_id"], as_index=False)[metric].mean().pivot(index="operator_id", columns="arm", values=metric)
        fig, ax = plt.subplots(figsize=(12, 6)); data.plot(kind="bar", ax=ax, color=[COLORS[a] for a in data.columns]); ax.set_ylabel(ylabel); ax.tick_params(axis="x", labelrotation=35); fig.tight_layout(); fig.savefig(FIGURES / filename, dpi=170); plt.close(fig)
    costs = tables["costs"].groupby("arm", as_index=False).cycle_wall_s.sum()
    fig, ax = plt.subplots(figsize=(8, 5)); ax.bar(costs.arm, costs.cycle_wall_s / 3600, color=[COLORS[a] for a in costs.arm]); ax.tick_params(axis="x", rotation=18); ax.set_ylabel("Total hours"); fig.tight_layout(); fig.savefig(FIGURES / "fig12_total_compute_cost_by_arm.png", dpi=170); plt.close(fig)


def write_workbook(tables: dict[str, pd.DataFrame]) -> None:
    path = RUN / f"{RUN_ID}.xlsx"
    sheets = {"Summary": tables["19_final_decision_summary.csv"], "Protocol": tables["01_protocol_comparison.csv"],
              "Mutation_Config": tables["03_operator_parameter_distributions.csv"], "Operator_Statistics": tables["04_operator_validity_uniqueness.csv"],
              "Original70": tables["09_Original70_gap_trajectory.csv"], "TSPLIB100": tables["10_TSPLIB100_gap_trajectory.csv"],
              "Uniform100": tables["11_Uniform100_gap_trajectory.csv"], "All_Benchmarks": tables["12_all_eight_benchmark_gaps.csv"],
              "Factorial": tables["14_factorial_effects.csv"], "Diversity": tables["15_QD_diversity_trajectory.csv"],
              "Compute_Cost": tables["18_total_compute_costs.csv"], "Integrity": tables["02_mutation_profile_verification.csv"]}
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)


def strict_verification(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    common = prepare_common(None, True)
    mutation = tables["mutation"]
    training = tables["06_training_budget_comparison.csv"]
    invalid_archive = ROOT / "runs" / "BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11HIGH_V2_INVALID_BATCH128"
    invalid_manifest = read_json(invalid_archive / "INVALID_RUN_MANIFEST.json")
    summaries = {(arm, cycle): read_json(cycle_dir(arm, cycle) / "cycle_summary.json")
                 for arm in ARMS for cycle in range(5)}
    selections = {(arm, cycle): read_json(cycle_dir(arm, cycle) / "06_selection" / "selection_summary.json")
                  for arm in ARMS for cycle in range(5)}
    old_checkpoint_hashes = {
        sha256_file(path) for path in invalid_archive.glob("A*/checkpoints/*.tsp.best")
    }
    corrected_checkpoint_hashes = {
        sha256_file(model_path(arm, index)) for arm in ARMS for index in range(1, 6)
    }
    chain_valid = all(
        summaries[(arm, cycle)]["input_checkpoint_sha256"] ==
        (M0_SHA if cycle == 0 else summaries[(arm, cycle - 1)]["output_checkpoint_sha256"])
        for arm in ARMS for cycle in range(5)
    )
    m0_once = read_json(COMMON / "M0_EVALUATION_ONCE.json")
    reuse_path = COMMON / "M0_REUSE_PROVENANCE.json"
    historical_m0_evaluations = int(read_json(reuse_path).get("historical_physical_evaluations", 0)) if reuse_path.is_file() else 0
    total_m0_evaluations = historical_m0_evaluations + int(m0_once["physical_evaluations_in_corrected_run"])
    checks = {
        "01_old_batch128_jobs_stopped": not (RUNTIME / "blade_core_ablation_5c_foga11high_v2.lock").exists(),
        "02_old_outputs_preserved_and_invalid": invalid_manifest["official_result_eligible"] is False and (invalid_archive / "BLADE_ABLATION_BATCH128_RUN_INVALIDATED").is_file(),
        "03_corrected_directory_separate": RUN.resolve() != invalid_archive.resolve(),
        "04_all_arms_restart_from_M0": all(summaries[(a, 0)]["model_before"] == "M0" for a in ARMS),
        "05_identical_M0_sha256": all(summaries[(a, 0)]["input_checkpoint_sha256"] == M0_SHA for a in ARMS),
        "06_effective_batch_size_256": training.effective_batch_size.eq(256).all(),
        "07_no_effective_batch_size_128": training.effective_batch_size.ne(128).all(),
        "08_training_set_10000": training.hard_unique.eq(10_000).all() and training.hard_occurrences_per_epoch.eq(10_000).all(),
        "09_uniform_anchor_zero": training.uniform_anchor.eq(0).all(),
        "10_epochs_256": training.epochs.eq(256).all(),
        "11_updates_per_epoch_40": training.optimizer_steps_per_epoch.eq(40).all(),
        "12_updates_per_cycle_10240": training.optimizer_steps.eq(10_240).all(),
        "13_updates_per_arm_51200": training.groupby("arm").optimizer_steps.sum().eq(51_200).all(),
        "14_gradient_accumulation_one": training.gradient_accumulation_steps.eq(1).all(),
        "15_last_partial_batch_retained": training.drop_last.eq(False).all() and training.final_partial_batch_size.eq(16).all(),
        "16_optimizer_and_lr_unchanged": CONFIG["training"]["optimizer"] == "fresh_Adam" and LR == 3.75e-5,
        "17_all_arms_FOGA11_HIGH_V1": all(summaries[(a, c)]["profile_sha256"] == sha256_file(PROFILE_PATH) for a in ARMS for c in range(5)),
        "18_implosion_HIGH_verified": operator_profile_audit()["checks"]["implosion_high_exact"],
        "19_axis_projection_HIGH_verified": operator_profile_audit()["checks"]["axis_high_exact"],
        "20_nine_base_operators_unchanged": operator_profile_audit()["checks"]["nine_base_exact"],
        "21_same_frozen_mutation_profile": len({summaries[(a, c)]["profile_sha256"] for a in ARMS for c in range(5)}) == 1,
        "22_parent_child_union_unchanged": tables["07_candidate_generation_trajectory.csv"].candidate_n.gt(10_000).all(),
        "23_same_common_U0_O0_C0": common["U0_n"] == 10_000 and common["C0_HIGH_n"] > 10_000 and all(sha256_file(cycle_dir(a, 0) / "03_mutation" / "offspring_coordinates.npz") == sha256_file(COMMON / "O0_HIGH_mutation" / "offspring_coordinates.npz") for a in ARMS),
        "24_QD4_CVT_K1000_unchanged": len(QD4_FEATURE_NAMES) == 4 and CONFIG["QD"]["K"] == 1000,
        "25_HGB40_F40_unchanged": CONFIG["surrogate"]["features"] == "F40" and len(hgbcore.verify_sources()["F40_names"]) == 40,
        "26_FULL_is_Elite_QD_P_equals_T": all(set(selections[("A0_FULL", c)]["component_counts"]).issubset({"qd_hgb_elite", "global_hgb_fill"}) and selections[("A0_FULL", c)]["P_equals_T"] for c in range(5)),
        "27_without_QD_is_global_HGB": all(set(selections[("A1_WITHOUT_QD", c)]["component_counts"]) == {"global_hgb"} for c in range(5)),
        "28_without_surrogate_is_random_niche_QD": all(set(selections[("A2_WITHOUT_SURROGATE", c)]["component_counts"]).issubset({"qd_uniform_representative", "uniform_fill"}) for c in range(5)),
        "29_random_is_global_uniform": all(set(selections[("A3_RANDOM", c)]["component_counts"]) == {"global_uniform"} for c in range(5)),
        "30_no_batch128_checkpoint_reused": corrected_checkpoint_hashes.isdisjoint(old_checkpoint_hashes),
        "31_no_contaminated_later_cycle_chain": chain_valid,
        "32_external_datasets_unchanged": set(tables["12_all_eight_benchmark_gaps.csv"].dataset) == set(BENCHMARKS),
        "33_M5_official": all(model_path(a, 5).is_file() for a in ARMS),
        "34_no_best_cycle_selection": CONFIG["training"]["checkpoint"] == "E256_only" and all(
            {p.name for p in (cycle_dir(a, c) / "09_bq_training").glob("M*_E*.tsp.best")} ==
            {f"M{c + 1}_E000.tsp.best", f"M{c + 1}_E256.tsp.best"}
            for a in ARMS for c in range(5)),
        "35_no_extra_seeds_or_cycles": len(ARMS) == 4 and all(not model_path(a, 6).exists() for a in ARMS),
        "36_no_mutation_parameter_retuning": read_json(PROTOCOL / "frozen_experiment_config.json")["mutation_profile_sha256"] == sha256_file(PROFILE_PATH),
        "37_no_benchmark_driven_adjustments": CONFIG["evaluation"]["passive_only"] is True,
        "38_cycle_manifests_and_hashes_verified": all(cycle_complete(a, c) for a in ARMS for c in range(5)),
        "39_M0_evaluated_exactly_once_and_shared": total_m0_evaluations == 1 and m0_once["policy"] == "evaluate_once_then_share_across_four_arms",
    }
    return {"STRICT_VERIFICATION": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "failed": [k for k, v in checks.items() if not v], "created_utc": utc_now()}


def finalize(elapsed_s: float) -> dict[str, Any]:
    tables = assemble_tables()
    make_figures(tables)
    write_workbook(tables)
    strict = strict_verification(tables)
    atomic_json(RUN / "STRICT_VERIFICATION.json", strict)
    m5 = tables["13_M5_four_arm_comparison.csv"]
    effects = tables["14_factorial_effects.csv"]
    report = f"""# BLADE-NCO CORE ABLATION BATCH256 CORRECTION COMPLETE

Frozen mutation design: `BLADE_FOGA11_MUTATION_DESIGN_FROZEN_V1`

Mutation profile: `{PROFILE_NAME}` (`{sha256_file(PROFILE_PATH)}`)

Implosion and Axis Projection are HIGH; the other nine operators are canonical BASE.

Each cycle uses 110,000 depth-1 attempts, Parent+Child Union, P=T=10,000, no Uniform anchor, effective batch size 256, and 10,240 optimizer steps.
The shared M0 benchmark suite was physically evaluated exactly once and reused across all four reporting trajectories.

## Primary M5 results

{m5[["arm", "OriginalTSPLIB70", "TSPLIB100", "Uniform100", "GAP8"]].to_markdown(index=False)}

## Factorial effects (percentage points; lower gap is better)

{effects[effects.metric.isin(["OriginalTSPLIB70", "GAP8"])].to_markdown(index=False)}

This is one matched evolutionary seed and supports descriptive causal attribution within the frozen run, not population-level significance. OriginalTSPLIB70 has influenced prior development and is not an untouched external holdout.

Strict verification: `{strict['STRICT_VERIFICATION']}`. Total wall time: {elapsed_s / 3600:.2f} hours.
"""
    (RUN / "final_report.md").write_text(report, encoding="utf-8")
    if strict["STRICT_VERIFICATION"] != "PASS":
        raise Halt("STRICT_VERIFICATION failed: " + ", ".join(strict["failed"]))
    marker = RUN / "BLADE_CORE_ABLATION_5C_BATCH256_COMPLETE"
    marker.write_text("BLADE_CORE_ABLATION_5C_BATCH256_COMPLETE\n", encoding="utf-8")
    return {"status": "PASS", "final_label": "BLADE-NCO CORE ABLATION BATCH256 CORRECTION COMPLETE",
            "strict": "PASS", "M5": m5.to_dict("records")}


def write_error(exc: BaseException, status: Status | None) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "ERROR_REPORT.md").write_text(
        f"# Core ablation incomplete\n\n- Exception: `{type(exc).__name__}: {exc}`\n- Resume: `D:\\conda_envs\\bq-nco\\python.exe scripts\\{Path(__file__).name} --resume`\n\n```text\n{traceback.format_exc()}\n```\n",
        encoding="utf-8")
    atomic_json(LOCK, {"pid": os.getpid(), "state": "FAILED", "error": repr(exc), "timestamp": utc_now()})
    if status:
        status.update(state="FAILED", phase="FAILED", latest_error=repr(exc))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--prepare-common", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    status: Status | None = None
    started = time.perf_counter()
    try:
        if args.preflight:
            print(json.dumps(preflight(True), indent=2), flush=True)
            return 0
        running = active_lock()
        if running and int(running.get("pid", -1)) != os.getpid():
            raise Halt(f"experiment already active: PID {running['pid']}")
        pre = preflight(True)
        initialize_protocol(pre, args.resume)
        status = Status()
        common = prepare_common(status, args.resume)
        if args.prepare_common:
            atomic_json(LOCK, {"pid": os.getpid(), "state": "COMMON_PREPARED", "timestamp": utc_now(), **common})
            status.update(state="COMMON_PREPARED", phase="COMMON_PREPARED")
            print(json.dumps({"status": "COMMON_PREPARED", **common}, indent=2), flush=True)
            return 0
        device = base._device()
        if device.type != "cuda":
            raise Halt(f"CUDA required, got {device}")
        label_b0(device, status, args.resume)
        ensure_baseline(device, status)
        source = hgbcore.verify_sources()
        for arm in ARMS:
            for cycle in range(5):
                run_cycle(arm, cycle, device, source, status, args.resume)
        status.update(arm=None, cycle=None, phase="final_reporting")
        result = finalize(time.perf_counter() - started)
        atomic_json(LOCK, {"pid": os.getpid(), "state": "COMPLETE", "timestamp": utc_now(), **result})
        status.update(state="COMPLETE", phase="COMPLETE", latest_error=None)
        print("BLADE-NCO CORE ABLATION BATCH256 CORRECTION COMPLETE", flush=True)
        print(json.dumps(result, indent=2), flush=True)
        return 0
    except BaseException as exc:
        write_error(exc, status)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
