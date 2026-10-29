#!/usr/bin/env python3
"""Audited, resumable five-seed BLADE full-pipeline reproduction.

This entrypoint composes the existing formal Base-ablation, EvoReal generation,
Concorde labeling, BQ training, and frozen evaluation implementations.  It does
not redefine any acquisition method.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import external_v2_passive_blade_eval as evaluator  # noqa: E402
import foga11_base_v1_mutation_worker as foga_worker  # noqa: E402
import prepare_evoreal50k_official_mix as evo_prepare  # noqa: E402
import run_blade_core_ablation_5c_noanchor_foga11base_v1 as core  # noqa: E402
import run_matched_budget_baselines_v2 as baseline  # noqa: E402
from bq.checkpoint import load_canonical_checkpoint  # noqa: E402
from evaluation.external_v2 import coordinate_hash, coordinate_set_hash  # noqa: E402
from final_run.science import membership_hash, qd_frame  # noqa: E402
from final_run.stage import atomic_json, sha256_file, utc_now  # noqa: E402
from mutation import OPERATOR_NAMES, apply_operator_with_config  # noqa: E402
from mutation.parameters import load_named_profile  # noqa: E402
from oracle.concorde import (  # noqa: E402
    CANONICAL_CONCORDE_SHA256,
    integer_distance_matrix,
    resolve_backend,
    solve_one,
    tour_cost_int,
)
from surrogate_benchmark.models import hgb_params  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402


RUN_ID = "BLADE_FULL_PIPELINE_MULTISEED_V1"
OUT = ROOT / "runs" / RUN_ID
HIST_BASE = ROOT / "runs" / "BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1"
HIST_EVO = ROOT / "runs" / "BLADE_BASE_UNIFORM_EVOREAL_5C_BENCHMARK_V2"
SEEDS = (2026, 1107, 2203, 3307, 4409)
NEW_SEEDS = (1107, 2203, 3307, 4409)
METHODS = {
    "A0_FULL_BLADE": "A0_FULL_BASE",
    "A1_WITHOUT_QD": "A1_WITHOUT_QD_BASE",
    "A2_WITHOUT_SURROGATE": "A2_WITHOUT_SURROGATE_BASE",
    "A3_RANDOM_FOGA": "A3_RANDOM_BASE",
    "A4_EVOREAL50K": "A4_EVOREAL50K",
}
CORE_ARMS = tuple(list(METHODS.values())[:4])
M0 = ROOT / "assets" / "solver" / "M0" / "tsp.best"
M0_SHA = "6826d7a71b30d882e4673a26ced162b7391157795931efe230f4d019292bba15"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def seed_root(seed: int) -> Path:
    return OUT / f"seed_{seed}"


def core_root(seed: int) -> Path:
    return seed_root(seed) / "A0_A3"


def a4_root(seed: int) -> Path:
    return seed_root(seed) / "A4"


def command_record(command: list[str]) -> dict[str, Any]:
    process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return {"command": command, "returncode": process.returncode,
            "stdout": process.stdout, "stderr": process.stderr}


def historical_m5() -> dict[str, float]:
    base = pd.read_csv(HIST_BASE / "13_M5_four_arm_comparison.csv")
    arm_column = "arm" if "arm" in base.columns else "method"
    gap_column = "GAP8" if "GAP8" in base.columns else "gap8"
    result = {
        method: float(base.loc[base[arm_column].astype(str).eq(arm), gap_column].iloc[0])
        for method, arm in list(METHODS.items())[:4]
    }
    evo = pd.read_csv(HIST_EVO / "B1_EVOREAL50K" / "evaluations" / "M5_eight.csv")
    result["A4_EVOREAL50K"] = float(evo.mean_gap_percent.mean())
    return result


def audit() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    strict = read_json(HIST_BASE / "STRICT_VERIFICATION.json")
    base_cfg = read_json(HIST_BASE / "protocol" / "frozen_experiment_config.json")
    evo_cfg = read_json(HIST_EVO / "protocol" / "frozen_protocol.json")
    evo_manifest = read_json(HIST_EVO / "datasets" / "evoreal50k_manifest.json")
    checkpoints: dict[str, Any] = {}
    for method, arm in METHODS.items():
        path = (HIST_EVO / "B1_EVOREAL50K" / "checkpoints" / "M5_E256.tsp.best"
                if method == "A4_EVOREAL50K"
                else HIST_BASE / arm / "checkpoints" / "M5_E256.tsp.best")
        checkpoints[method] = {"path": str(path), "sha256": sha256_file(path)}
    checks = {
        "historical_strict_verification_pass": strict.get("STRICT_VERIFICATION") == "PASS",
        "same_M0": base_cfg.get("M0_sha256") == evo_cfg.get("initial_model_sha256") == M0_SHA,
        "five_cycles": base_cfg.get("cycles") == list(range(5)) and evo_cfg.get("cycles") == 5,
        "ten_thousand_per_cycle": base_cfg["training"]["hard_occurrences_per_epoch"] == 10_000
        and evo_cfg.get("instances_per_cycle") == 10_000,
        "epochs_256": base_cfg["training"]["epochs"] == evo_cfg.get("epochs_per_cycle") == 256,
        "batch_256": base_cfg["training"]["effective_batch_size"] == evo_cfg.get("effective_batch_size") == 256,
        "updates_10240": base_cfg["training"]["optimizer_steps_per_cycle"] == evo_cfg.get("optimizer_steps_per_cycle") == 10_240,
        "updates_total_51200": evo_cfg.get("optimizer_steps_per_arm") == 51_200,
        "fresh_adam_no_scheduler": base_cfg["training"]["optimizer"] == "fresh_Adam"
        and evo_cfg.get("fresh_optimizer_each_cycle") is True and evo_cfg.get("scheduler") is None,
        "no_uniform_anchor": base_cfg["training"]["uniform_anchor_n"] == evo_cfg.get("uniform_anchor") == 0,
        "evoreal_unique_50000": evo_manifest.get("valid_unique_count") == 50_000,
        "evoreal_mix_19_12_17": evo_manifest.get("mixture_ratio_S1_S2_S3") == [19, 12, 17],
        "eight_frozen_benchmarks_historical": len(base_cfg["evaluation"]["datasets"]) == 8
        and base_cfg["evaluation"]["datasets"] == evo_cfg["evaluation"]["datasets"],
    }
    record = {
        "status": "PASS_WITH_RECORDED_QUALIFICATION" if all(checks.values()) else "BLOCKED",
        "tag": "BLADE_MULTISEED_HISTORICAL_AUDIT_PASSED" if all(checks.values()) else "BLADE_FULL_PIPELINE_MULTISEED_PHASE1_PARTIAL",
        "created_utc": now(), "S0_experiment_seed": 2026,
        "S0_evoreal_generator_master_seed": 20261022,
        "qualification": (
            "A4 is budget/M0/evaluation comparable but its historical generator uses the separately frozen "
            "master seed 20261022; it is not a common-random-number pairing with A0-A3. Historical evaluation "
            "contains eight frozen sets, so the reproduction keeps all eight (including the four named core sets)."
        ),
        "checks": checks,
        "code": {
            "A0_A3": str((ROOT / "scripts/run_blade_core_ablation_5c_noanchor_foga11base_v1.py").resolve()),
            "A4": str((ROOT / "scripts/run_matched_budget_baselines_v2.py").resolve()),
            "EvoReal_generator": str((ROOT / "scripts/prepare_evoreal50k_official_mix.py").resolve()),
        },
        "M0": {"path": str(M0), "sha256": sha256_file(M0)},
        "U0": {"path": str(core.U0_SOURCE), "membership_sha256": core.U0_MEMBERSHIP,
               "policy": "same immutable canonical Uniform10k roots for every seed"},
        "methods": deepcopy(core.CONFIG["arms"]) | {
            "A4_EVOREAL50K": {"surrogate": None, "QD": False, "FOGA11": False,
                               "data": "pinned official EvoReal S1:S2:S3=19:12:17 synthetic-only"}},
        "checkpoint_M5": checkpoints,
        "S0_M5_GAP8": historical_m5(),
        "historical_base_config": base_cfg,
        "historical_evoreal_config": evo_cfg,
    }
    atomic_json(OUT / "00_HISTORICAL_PROTOCOL_AUDIT.json", record)
    rows = "\n".join(f"- {key}: `{value}`" for key, value in checks.items())
    gaps = "\n".join(f"- {key}: {value:.9f}%" for key, value in record["S0_M5_GAP8"].items())
    write_text(OUT / "00_HISTORICAL_PROTOCOL_AUDIT.md", f"""# Historical protocol audit

Status: **{record['status']}**  
Tag: `{record['tag']}`

The historical implementation passes the matched-budget checks. S0 uses BQ/HGB/training seed 2026; historical A4 used its separately frozen EvoReal generator master seed 20261022. This is recorded as a pairing qualification, not hidden.

The real historical evaluator contains eight frozen benchmark sets. They are retained for GAP8 and the four named core sets remain directly reportable.

## Checks

{rows}

## S0 M5 GAP8

{gaps}
""")
    if record["status"] == "BLOCKED":
        raise RuntimeError("historical protocol audit failed")
    return record


def frozen_protocol() -> dict[str, Any]:
    record = {
        "run_id": RUN_ID, "status": "FROZEN", "created_utc": now(),
        "seeds": list(SEEDS), "historical_seed": 2026, "new_seeds": list(NEW_SEEDS),
        "methods": list(METHODS), "cycles": 5, "instances_per_cycle": 10_000,
        "epochs_per_cycle": 256, "effective_batch_size": 256,
        "optimizer_steps_per_cycle": 10_240, "optimizer_steps_per_method": 51_200,
        "learning_rate": 3.75e-5, "optimizer": "fresh Adam per cycle",
        "scheduler": None, "uniform_anchor": 0, "M0_sha256": M0_SHA,
        "U0_policy": "same immutable canonical Uniform10k roots; per-seed FOGA streams",
        "HGB_seed_stream": "[experiment_seed, ..., experiment_seed+4]",
        "calibration_seed_stream": "experiment_seed*100+cycle",
        "EvoReal_seed_stream": "official generator, master_seed=experiment_seed, SHA-derived substreams",
        "parallelism": "at most two GPU training workers; seeds completed sequentially",
        "evaluation": {"datasets": list(core.BENCHMARKS), "decoder": "greedy", "checkpoint_selection": False},
        "concorde": {"exact_only": True, "binary_sha256": CANONICAL_CONCORDE_SHA256,
                      "A0_A3_timeout_seconds": 300, "A4_timeout_seconds": 7200,
                      "warmstart_30000_exception_inherited": False},
        "no_test_tuning": True,
    }
    atomic_json(OUT / "protocol" / "frozen_multiseed_protocol.json", record)
    write_text(OUT / "01_FROZEN_MULTISEED_PROTOCOL.md", "# Frozen multi-seed protocol\n\n```json\n" + json.dumps(record, indent=2) + "\n```\n")
    return record


def preflight() -> dict[str, Any]:
    audit_record = audit()
    protocol = frozen_protocol()
    backend = resolve_backend()
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    generator_ok = all(sha256_file(ROOT / "third_party" / "evoreal" / name) == expected
                       for name, expected in baseline.GENERATOR_HASHES.items())
    checks = {
        "audit_pass": audit_record["status"].startswith("PASS"),
        "M0_hash": sha256_file(M0) == M0_SHA,
        "canonical_U0_present": core.U0_SOURCE.is_file(),
        "concorde_hash": backend["binary_hash"] == CANONICAL_CONCORDE_SHA256,
        "Rscript_available": shutil.which("Rscript") is not None,
        "two_cuda_devices": gpu_count >= 2,
        "free_disk_ge_500GiB": shutil.disk_usage(ROOT).free >= 500 * 2**30,
        "evoreal_generators_pinned": generator_ok,
        "historical_results_immutable_inputs": (HIST_BASE / "STRICT_VERIFICATION.json").is_file()
        and (HIST_EVO / "B1_EVOREAL50K/status.json").is_file(),
    }
    record = {"status": "PASS" if all(checks.values()) else "BLOCKED", "checks": checks,
              "created_utc": now(), "gpu_count": gpu_count,
              "free_disk_GiB": shutil.disk_usage(ROOT).free / 2**30,
              "concorde": backend, "protocol_sha256": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()}
    atomic_json(OUT / "02_PREFLIGHT_AND_SMOKE_TEST.json", {"preflight": record, "smoke": {"status": "PENDING"}})
    if record["status"] != "PASS":
        raise RuntimeError("preflight blocked: " + ", ".join(k for k, v in checks.items() if not v))
    print("BLADE_MULTISEED_PREFLIGHT_PASSED", flush=True)
    return record


def smoke() -> dict[str, Any]:
    pre = preflight()
    smoke_dir = OUT / "smoke_seed_1107"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    core.configure_experiment(smoke_dir / "core", 1107)
    with np.load(core.U0_SOURCE, allow_pickle=False) as payload:
        roots = np.asarray(payload["coordinates"], dtype=np.float64)
        root_hashes = payload["instance_hash"].astype(str).tolist()
    profile = load_named_profile(core.PROFILE_NAME)
    children = []
    mutation_hashes = []
    for operator in OPERATOR_NAMES:
        value = foga_worker.seeded_child_seed(1107, 0, root_hashes[0], operator)
        result = apply_operator_with_config(operator, roots[0], np.random.default_rng(value), profile)
        if not result.valid_after_repair:
            raise RuntimeError(f"smoke mutation invalid: {operator}")
        children.append(result.final_coordinates)
        mutation_hashes.append(result.final_output_hash)
    qd = qd_frame(np.stack(children), mutation_hashes)

    historical_union = pd.read_parquet(HIST_BASE / "protocol/common_initialization/C0_BASE.parquet")
    scores = np.linspace(-1.0, 1.0, len(historical_union))
    selections = {}
    for arm in CORE_ARMS:
        selected = core.select_rows(arm, 0, historical_union, scores if arm in core.HGB_ARMS else None)
        selections[arm] = {"n": len(selected), "unique": int(selected.instance_hash.nunique())}

    rng = np.random.default_rng(1107)
    x = rng.normal(size=(128, 40)); y = rng.normal(size=128)
    hgb = HistGradientBoostingRegressor(**hgb_params(1107)).fit(x, y)
    hgb_finite = bool(np.isfinite(hgb.predict(x)).all())

    evo_samples = {}
    for source in ("S1", "S2", "S3"):
        evo_prepare.seed_all(evo_prepare.derived_seed("smoke", 1107, source))
        values = evo_prepare.load_generator(source)(batch_size=2, problem_size=100).detach().cpu().numpy()
        evo_samples[source] = {"shape": list(values.shape), "finite": bool(np.isfinite(values).all())}

    oracle_work = smoke_dir / "concorde"
    oracle = solve_one(roots[0], root_hashes[0], oracle_work, resolve_backend(), timeout=300.0)
    oracle_cost = tour_cost_int(integer_distance_matrix(roots[0]), [int(x) for x in oracle["expert_tour"].split()])

    device = torch.device("cuda:0")
    core.seed_runtime()
    optimizer = core.production_optimizer_audit(roots)
    model, _ = load_canonical_checkpoint(M0, device, expected_sha256=M0_SHA)
    checkpoint = smoke_dir / "roundtrip.tsp.best"
    core.save_checkpoint(model, checkpoint, "SMOKE", 0, 0)
    _, roundtrip = load_canonical_checkpoint(checkpoint, device, expected_sha256=sha256_file(checkpoint))
    payload = evaluator.load_fixed_references("Uniform100")
    subset = {key: (value[:4] if key in {"coordinates", "identifiers", "optima", "hashes"} else value)
              for key, value in payload.items()}
    _, eval_summary = evaluator.evaluate_tensor_dataset(model, device, "Uniform100_SMOKE4", subset, roundtrip["sha256"])
    del model
    torch.cuda.empty_cache()

    checks = {
        "all_11_foga_operators_valid": len(children) == 11,
        "qd_features_finite": len(qd) == 11 and np.isfinite(qd[list(core.QD4_FEATURE_NAMES)].to_numpy()).all(),
        "five_method_paths_covered": len(selections) == 4 and len(evo_samples) == 3,
        "selection_10k_unique": all(v == {"n": 10_000, "unique": 10_000} for v in selections.values()),
        "hgb_fit_predict": hgb_finite,
        "evoreal_generators": all(v["shape"] == [2, 100, 2] and v["finite"] for v in evo_samples.values()),
        "concorde_exact_verified": oracle["verified"] and oracle_cost == oracle["optimal_cost_int"],
        "bq_one_optimizer_update": optimizer.get("status") == "PASS",
        "checkpoint_roundtrip": roundtrip["sha256"] == sha256_file(checkpoint),
        "frozen_evaluator_smoke": eval_summary["valid_tours"] == 4 and eval_summary["failed_tours"] == 0,
    }
    record = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "created_utc": now(), "mutation_hashes": mutation_hashes,
              "selections": selections, "evo_samples": evo_samples,
              "oracle": oracle, "optimizer": optimizer, "evaluation": eval_summary}
    atomic_json(OUT / "02_PREFLIGHT_AND_SMOKE_TEST.json", {"preflight": pre, "smoke": record,
                "tag": "BLADE_MULTISEED_PREFLIGHT_PASSED" if record["status"] == "PASS" else "BLADE_FULL_PIPELINE_MULTISEED_PHASE1_PARTIAL"})
    if record["status"] != "PASS":
        raise RuntimeError("smoke test failed")
    print("SMOKE_TEST_PASS", flush=True)
    return record


def configure_a4(seed: int) -> None:
    root = a4_root(seed)
    baseline.RUN_ID = f"{RUN_ID}/seed_{seed}/A4"
    baseline.OUT = root
    baseline.ARMS = {"B1": {"directory": "A4_EVOREAL50K", "dataset": "evoreal50k_coordinates.npy"}}
    core.configure_experiment(root, seed)
    baseline.canonical = core


def prepare_a4(seed: int) -> dict[str, Any]:
    configure_a4(seed)
    root = a4_root(seed)
    generator_root = root / "generator"
    evo_prepare.RUN_ID = f"{RUN_ID}_seed_{seed}_EVOREAL50K"
    evo_prepare.OUT = generator_root
    evo_prepare.DATA = generator_root / "data"
    evo_prepare.ORACLE = generator_root / "oracle"
    evo_prepare.MASTER_SEED = seed
    evo_prepare.write_official_audit()
    coordinates, labels, hashes = evo_prepare.load_or_build_pool()

    for relative in ("protocol", "datasets", "A4_EVOREAL50K/checkpoints", "A4_EVOREAL50K/cycles",
                     "A4_EVOREAL50K/evaluations", "A4_EVOREAL50K/logs", "A4_EVOREAL50K/oracle"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    baseline.link_or_copy(generator_root / "data/evoreal50k_coordinates.npy", baseline.dataset_path("B1"))
    baseline.link_or_copy(generator_root / "data/evoreal50k_source_labels.npy", root / "datasets/evoreal50k_source_labels.npy")
    for cycle in range(5):
        baseline.link_or_copy(generator_root / f"data/D{cycle}_membership.txt", root / f"datasets/evoreal_D{cycle}_membership.txt")
    forbidden, forbidden_sets, test_rows = baseline.frozen_test_hashes()
    point_hashes = [coordinate_set_hash(xy) for xy in coordinates]
    if len(set(hashes)) != 50_000 or len(set(point_hashes)) != 50_000:
        raise RuntimeError("new EvoReal pool is not unique")
    if set(hashes) & forbidden or set(point_hashes) & forbidden_sets:
        raise RuntimeError("new EvoReal pool overlaps frozen tests")
    manifest = {
        "status": "FROZEN_AWAITING_EXPERT_LABELS", "master_seed": seed,
        "coordinate_file": str(baseline.dataset_path("B1")),
        "coordinate_file_sha256": sha256_file(baseline.dataset_path("B1")),
        "source_label_file": str(root / "datasets/evoreal50k_source_labels.npy"),
        "source_label_file_sha256": sha256_file(root / "datasets/evoreal50k_source_labels.npy"),
        "shape": list(coordinates.shape), "dtype": str(coordinates.dtype),
        "problem_size": 100, "coordinate_range": [float(coordinates.min()), float(coordinates.max())],
        "official_repository": baseline.OFFICIAL_REPOSITORY, "official_commit": baseline.OFFICIAL_COMMIT,
        "official_partition": "rule_based", "validation_counts": baseline.EVO_VALIDATION_COUNTS,
        "validation_total": 48, "mixture_ratio_S1_S2_S3": [19, 12, 17],
        "counts": baseline.EVO_COUNTS, "generator_hashes": baseline.GENERATOR_HASHES,
        "valid_unique_count": 50_000, "unique_ordered_instances": 50_000,
        "unique_point_sets": 50_000, "exact_train_test_overlap": 0,
        "evaluation_sets": test_rows, "phase_two_real_TSPLIB_finetuning": False,
        "generation_provenance": str(generator_root / "evoreal50k_provenance.json"),
    }
    atomic_json(root / "datasets/evoreal50k_manifest.json", manifest)
    atomic_json(root / "protocol/reference_checkpoint_hashes.json", {
        "protected_files": baseline.protected_reference_snapshot(), "created_utc": now()})
    atomic_json(root / "protocol/frozen_protocol.json", {
        **frozen_protocol(), "run_id": baseline.RUN_ID, "experiment_seed": seed,
        "training_seed": seed, "generator_master_seed": seed, "method": "A4_EVOREAL50K"})
    baseline.update_status("B1", status="DATA_PENDING", phase="POOL_FROZEN", labeled=0,
                           total=50_000, training_started=False)
    print(f"A4_POOL_READY seed={seed} sha256={manifest['coordinate_file_sha256']}", flush=True)
    return manifest


def label_a4(seed: int, workers: int, timeout: float) -> None:
    configure_a4(seed)
    baseline.label("B1", workers, timeout)


def train_a4(seed: int) -> None:
    configure_a4(seed)
    core.seed_runtime()
    baseline.train("B1")


def run_logged(command: list[str], log_path: Path, env: dict[str, str] | None = None) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8", buffering=1)
    handle.write(f"\n[{now()}] COMMAND {json.dumps(command)}\n")
    process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                               env=env or os.environ.copy(), text=True)
    process._blade_log_handle = handle  # type: ignore[attr-defined]
    return process


def wait_group(processes: list[tuple[str, subprocess.Popen]], seed: int) -> None:
    next_report = time.monotonic() + 3600
    while True:
        active = [(name, process) for name, process in processes if process.poll() is None]
        if not active:
            break
        if time.monotonic() >= next_report:
            print(f"HOURLY_CHECK seed={seed} active={[name for name, _ in active]} utc={now()}", flush=True)
            next_report += 3600
        time.sleep(60)
    failures = []
    for name, process in processes:
        handle = getattr(process, "_blade_log_handle", None)
        if handle is not None:
            handle.close()
        if process.returncode:
            failures.append((name, process.returncode))
    if failures:
        raise RuntimeError(f"worker failures: {failures}")


def run_seed(seed: int, workers: int, timeout: float) -> None:
    if seed not in NEW_SEEDS:
        raise ValueError(f"formal new seed must be one of {NEW_SEEDS}")
    sroot = seed_root(seed)
    logs = OUT / "logs" / f"seed_{seed}"
    sroot.mkdir(parents=True, exist_ok=True)
    atomic_json(sroot / "RUN_STATUS.json", {"seed": seed, "status": "RUNNING", "started_utc": now()})
    python = sys.executable
    entry = str(Path(__file__).resolve())
    core_entry = str((ROOT / "scripts/run_blade_core_ablation_5c_noanchor_foga11base_v1.py").resolve())
    base_env = os.environ.copy()

    if not (a4_root(seed) / "datasets/evoreal50k_manifest.json").is_file():
        prepare_a4(seed)
    integrity = a4_root(seed) / "A4_EVOREAL50K/oracle/expert_tour_integrity.json"
    if not integrity.is_file():
        label_a4(seed, workers, timeout)

    common_cmd = [python, core_entry, "--output-dir", str(core_root(seed)),
                  "--experiment-seed", str(seed), "--prepare-baseline", "--resume"]
    env0 = base_env | {"CUDA_VISIBLE_DEVICES": "0"}
    common = run_logged(common_cmd, logs / "prepare_common_baseline.log", env0)
    wait_group([("prepare_common_baseline", common)], seed)

    groups = ((CORE_ARMS[0], CORE_ARMS[1]), (CORE_ARMS[2], CORE_ARMS[3]))
    for group_index, group in enumerate(groups):
        processes = []
        for gpu, arm in enumerate(group):
            env = base_env | {"CUDA_VISIBLE_DEVICES": str(gpu)}
            command = [python, core_entry, "--output-dir", str(core_root(seed)),
                       "--experiment-seed", str(seed), "--arms", arm, "--resume"]
            processes.append((arm, run_logged(command, logs / f"{arm}.log", env)))
        wait_group(processes, seed)

    a4_env = base_env | {"CUDA_VISIBLE_DEVICES": "0"}
    a4_process = run_logged([python, entry, "train-a4", "--seed", str(seed)], logs / "A4_EVOREAL50K.log", a4_env)
    wait_group([("A4_EVOREAL50K", a4_process)], seed)

    # The canonical core runner uses its provenance RUN_ID as the workbook
    # relative path.  Preserve that frozen runner byte-for-byte and create the
    # required parent directory here as a purely presentational recovery.
    (core_root(seed) / "BLADE_FULL_PIPELINE_MULTISEED_V1" / f"seed_{seed}").mkdir(parents=True, exist_ok=True)
    finalize = run_logged([python, core_entry, "--output-dir", str(core_root(seed)),
                           "--experiment-seed", str(seed), "--finalize", "--resume"],
                          logs / "finalize_A0_A3.log", env0)
    wait_group([("finalize_A0_A3", finalize)], seed)
    atomic_json(sroot / "RUN_STATUS.json", {"seed": seed, "status": "COMPLETE", "completed_utc": now(),
                "methods": 5, "cycles_per_method": 5})
    print(f"BLADE_MULTISEED_SEED{seed}_COMPLETE", flush=True)


def evaluation_rows(seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    historical = seed == 2026
    for method, arm in METHODS.items():
        if method == "A4_EVOREAL50K":
            root = HIST_EVO / "B1_EVOREAL50K" if historical else a4_root(seed) / "A4_EVOREAL50K"
            for model_index in range(6):
                path = (root / "evaluations" / ("M0_eight.csv" if model_index == 0 else f"M{model_index}_eight.csv"))
                if not path.is_file():
                    continue
                frame = pd.read_csv(path)
                checkpoint = M0 if model_index == 0 else root / "checkpoints" / f"M{model_index}_E256.tsp.best"
                timing_path = root / "cycles" / f"C{model_index-1:02d}" / "09_bq_training/bq_training_timing.json"
                timing = read_json(timing_path) if timing_path.is_file() else {}
                cycle_summary = root / "cycles" / f"C{model_index-1:02d}" / "cycle_summary.json"
                summary = read_json(cycle_summary) if cycle_summary.is_file() else {}
                gap8 = float(frame.mean_gap_percent.mean())
                for item in frame.to_dict("records"):
                    rows.append({"seed": seed, "method": method, "cycle": model_index,
                                 "benchmark": item["dataset"], "n_instances": item.get("instance_count"),
                                 "mean_gap_percent": item["mean_gap_percent"], "gap8": gap8,
                                 "checkpoint_hash": sha256_file(checkpoint),
                                 "training_updates": 0 if model_index == 0 else timing.get("optimizer_steps"),
                                 "oracle_time": None, "training_time": timing.get("training_loop_wall_s"),
                                 "total_wall_clock": summary.get("evaluation_seconds"), "status": "COMPLETE"})
        else:
            root = HIST_BASE if historical else core_root(seed)
            m0_path = root / "protocol/common_initialization/M0_external_benchmark_results.csv"
            for model_index in range(6):
                path = m0_path if model_index == 0 else root / arm / "cycles" / f"C{model_index-1:02d}" / "external_benchmark_results.csv"
                if not path.is_file():
                    continue
                frame = pd.read_csv(path)
                checkpoint = M0 if model_index == 0 else root / arm / "checkpoints" / f"M{model_index}_E256.tsp.best"
                cycle_summary = root / arm / "cycles" / f"C{model_index-1:02d}" / "cycle_summary.json"
                summary = read_json(cycle_summary) if cycle_summary.is_file() else {}
                training = summary.get("training", {})
                gap8 = float(frame.mean_gap_percent.mean())
                for item in frame.to_dict("records"):
                    rows.append({"seed": seed, "method": method, "cycle": model_index,
                                 "benchmark": item["dataset"], "n_instances": item.get("instance_count"),
                                 "mean_gap_percent": item["mean_gap_percent"], "gap8": gap8,
                                 "checkpoint_hash": sha256_file(checkpoint),
                                 "training_updates": 0 if model_index == 0 else training.get("optimizer_steps"),
                                 "oracle_time": summary.get("oracle", {}).get("wall_seconds"),
                                 "training_time": training.get("training_loop_wall_s"),
                                 "total_wall_clock": summary.get("cycle_wall_s"), "status": "COMPLETE"})
    return rows


def analyze() -> dict[str, Any]:
    completed = [2026] + [seed for seed in NEW_SEEDS if (seed_root(seed) / "RUN_STATUS.json").is_file()
                          and read_json(seed_root(seed) / "RUN_STATUS.json").get("status") == "COMPLETE"]
    all_rows = pd.DataFrame([row for seed in completed for row in evaluation_rows(seed)])
    if all_rows.empty:
        raise RuntimeError("no completed results")
    for index, seed in enumerate(NEW_SEEDS, start=3):
        frame = all_rows[all_rows.seed.eq(seed)]
        if not frame.empty:
            write_frame(OUT / f"{index:02d}_SEED_{seed}_RESULTS.csv", frame)
    m5 = all_rows[all_rows.cycle.eq(5)].drop_duplicates(["seed", "method"])[["seed", "method", "gap8"]]
    write_frame(OUT / "05_FIVE_SEED_COMPARISON.csv", m5)
    stats = m5.groupby("method").gap8.agg(["count", "mean", "std"]).reset_index()
    full = m5[m5.method.eq("A0_FULL_BLADE")].set_index("seed").gap8
    paired = []
    rng = np.random.default_rng(2026)
    for method in list(METHODS)[1:]:
        other = m5[m5.method.eq(method)].set_index("seed").gap8
        common = full.index.intersection(other.index)
        delta = (full.loc[common] - other.loc[common]).to_numpy(float)
        if len(delta):
            draws = rng.choice(delta, size=(20_000, len(delta)), replace=True).mean(axis=1)
            paired.append({"comparison": f"A0_FULL_BLADE - {method}", "n_seeds": len(delta),
                           "mean_delta_gap": float(delta.mean()), "sd_delta_gap": float(delta.std(ddof=1)) if len(delta) > 1 else None,
                           "ci95_low": float(np.quantile(draws, .025)), "ci95_high": float(np.quantile(draws, .975)),
                           "full_win_count": int(np.sum(delta < 0)), "ties": int(np.sum(delta == 0))})
    paired_frame = pd.DataFrame(paired)
    write_frame(OUT / "06_PAIRED_STATISTICS.csv", paired_frame)
    costs = all_rows[all_rows.cycle.gt(0)].drop_duplicates(["seed", "method", "cycle"])[
        ["seed", "method", "cycle", "training_updates", "oracle_time", "training_time", "total_wall_clock"]]
    write_frame(OUT / "07_COMPUTE_COST_AUDIT.csv", costs)
    write_frame(OUT / "analysis/all_raw_results.csv", all_rows)
    write_frame(OUT / "analysis/method_summary.csv", stats)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = OUT / "figures"; figures.mkdir(parents=True, exist_ok=True)
    trajectories = all_rows.drop_duplicates(["seed", "method", "cycle"])
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, part in trajectories.groupby("method"):
        mean = part.groupby("cycle").gap8.mean()
        ax.plot(mean.index, mean.values, marker="o", label=method)
    ax.set_xlabel("Model cycle"); ax.set_ylabel("GAP8 (%)"); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(figures / "M0_M5_GAP8_trajectories.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for position, method in enumerate(METHODS):
        values = m5[m5.method.eq(method)].set_index("seed").gap8
        ax.scatter(np.full(len(values), position), values.values, label=method)
    ax.set_xticks(range(len(METHODS)), list(METHODS), rotation=25, ha="right"); ax.set_ylabel("M5 GAP8 (%)")
    fig.tight_layout(); fig.savefig(figures / "M5_five_method_seed_comparison.png", dpi=180); plt.close(fig)

    full_wins = paired_frame.set_index("comparison").full_win_count.to_dict() if len(paired_frame) else {}
    status = "COMPLETE" if len(completed) == 5 and len(m5) == 25 else "PARTIAL"
    markdown = f"""# Multi-seed analysis

Status: **{status}**

- Completed seeds: {completed}
- Complete M5 method/seed cells: {len(m5)}/25
- Evaluation: historical eight-set GAP8, with every constituent dataset retained in the raw result tables.
- Statistical unit: independent full-pipeline seed.
- Historical qualification: S0 A4 used generator master seed 20261022 while its BQ training seed was 2026.

## Method summary

{stats.to_markdown(index=False)}

## Paired Full-BLADE differences

Negative ΔGAP means Full BLADE is better.

{paired_frame.to_markdown(index=False) if len(paired_frame) else 'No paired results yet.'}

The five-seed result is interpreted only after all 25 M5 cells pass integrity checks. No held-out result was used for training or checkpoint selection.
"""
    write_text(OUT / "08_MULTISEED_ANALYSIS.md", markdown)
    record = {"run_id": RUN_ID, "status": status, "completed_seeds": completed,
              "completed_seed_count": len(completed), "M5_cells": len(m5),
              "methods_per_seed": m5.groupby("seed").size().to_dict(),
              "full_win_counts": full_wins, "updated_utc": now(),
              "tag": "BLADE_FULL_PIPELINE_MULTISEED_COMPLETE" if status == "COMPLETE" else "BLADE_FULL_PIPELINE_MULTISEED_PHASE1_PARTIAL"}
    atomic_json(OUT / "RUN_STATUS.json", record)
    return record


def orchestrate(workers: int, timeout: float) -> None:
    smoke()
    for seed in NEW_SEEDS:
        status_path = seed_root(seed) / "RUN_STATUS.json"
        if status_path.is_file() and read_json(status_path).get("status") == "COMPLETE":
            print(f"BLADE_MULTISEED_SEED{seed}_ALREADY_COMPLETE", flush=True)
            analyze()
            continue
        run_seed(seed, workers, timeout)
        analyze()
    result = analyze()
    print(json.dumps(result, indent=2), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit")
    sub.add_parser("preflight")
    sub.add_parser("smoke")
    pa4 = sub.add_parser("prepare-a4"); pa4.add_argument("--seed", type=int, required=True)
    la4 = sub.add_parser("label-a4"); la4.add_argument("--seed", type=int, required=True)
    la4.add_argument("--workers", type=int, default=32); la4.add_argument("--timeout", type=float, default=7200.0)
    ta4 = sub.add_parser("train-a4"); ta4.add_argument("--seed", type=int, required=True)
    rs = sub.add_parser("run-seed"); rs.add_argument("--seed", type=int, required=True)
    rs.add_argument("--workers", type=int, default=32); rs.add_argument("--timeout", type=float, default=7200.0)
    sub.add_parser("analyze")
    orch = sub.add_parser("orchestrate"); orch.add_argument("--workers", type=int, default=32)
    orch.add_argument("--timeout", type=float, default=7200.0)
    args = parser.parse_args()
    try:
        if args.command == "audit": audit()
        elif args.command == "preflight": preflight()
        elif args.command == "smoke": smoke()
        elif args.command == "prepare-a4": prepare_a4(args.seed)
        elif args.command == "label-a4": label_a4(args.seed, args.workers, args.timeout)
        elif args.command == "train-a4": train_a4(args.seed)
        elif args.command == "run-seed": run_seed(args.seed, args.workers, args.timeout)
        elif args.command == "analyze": print(json.dumps(analyze(), indent=2))
        else: orchestrate(args.workers, args.timeout)
        return 0
    except BaseException as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        atomic_json(OUT / "LAST_ERROR.json", {"status": "FAILED", "command": args.command,
                    "error": repr(exc), "traceback": traceback.format_exc(), "timestamp_utc": now()})
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
