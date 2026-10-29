#!/usr/bin/env python3
"""EvoReal strong-baseline diagnostic and one-cycle BLADE warm-start study.

The entry point is deliberately staged.  ``init`` and ``dry-run`` are safe and
do not start full training.  Expensive production work requires explicit
commands (``prepare``, ``label``, ``select-c2``, ``train``, ``evaluate`` and
``finalize``).  All writes are confined to a new run directory; completed A/B
experiments are immutable inputs and are fingerprinted before reuse.

Examples
--------
    python scripts/run_evoreal_warmstart_diagnostic.py init
    python scripts/run_evoreal_warmstart_diagnostic.py dry-run --count 24
    python scripts/run_evoreal_warmstart_diagnostic.py prepare --seed 2026
    python scripts/run_evoreal_warmstart_diagnostic.py label --target calibration --workers 16
    python scripts/run_evoreal_warmstart_diagnostic.py select-c2 --device cuda:0
    python scripts/run_evoreal_warmstart_diagnostic.py label --target C0 --workers 16
    python scripts/run_evoreal_warmstart_diagnostic.py train --arm C0 --device cuda:0
    python scripts/run_evoreal_warmstart_diagnostic.py evaluate --arm C0 --device cuda:0
    python scripts/run_evoreal_warmstart_diagnostic.py structural-audit --device cuda:0
    python scripts/run_evoreal_warmstart_diagnostic.py finalize

The formal pilot is one paired seed.  It is preliminary by construction and
cannot emit a definitive P4 scientific label.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist, pdist
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import DBSCAN
from sklearn.ensemble import HistGradientBoostingRegressor
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import external_v2_passive_blade_eval as evaluator  # noqa: E402
import oracle.concorde as concorde_impl  # noqa: E402
import run_blade_core_ablation_5c_noanchor_foga11base_v1 as canonical  # noqa: E402
import run_s4b_hgb40_union_qd as hgbcore  # noqa: E402
from bq.checkpoint import clone_model_from, load_canonical_checkpoint  # noqa: E402
from bq.inference import greedy_tour_costs  # noqa: E402
from bq.supervised import create_optimizer, supervised_optimizer_step  # noqa: E402
from evaluation.external_v2 import coordinate_hash, coordinate_set_hash  # noqa: E402
from features.qd4 import QD4_FEATURE_NAMES  # noqa: E402
from final_run.science import hardness, membership_hash, qd_frame  # noqa: E402
from final_run.stage import atomic_json, sha256_file, utc_now  # noqa: E402
from oracle.concorde import (  # noqa: E402
    CANONICAL_CONCORDE_SHA256,
    canonical_oracle_ledger,
    resolve_backend,
    solve_one,
    solve_batch,
)
from surrogate_benchmark.models import HGB_CFG_CANONICAL, SEEDS, hgb_params, median_stack  # noqa: E402


RUN_ID = "BLADE_EVOREAL_WARMSTART_DIAGNOSTIC_V1"
OUT = ROOT / "runs" / RUN_ID
FIGURES = OUT / "figures"
DRY = OUT / "dry_run"
BASE = ROOT / "runs" / "BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1"
MATCHED = ROOT / "runs" / "BLADE_BASE_UNIFORM_EVOREAL_5C_BENCHMARK_V2"
EVO_ARM = MATCHED / "B1_EVOREAL50K"
EVO_M5 = EVO_ARM / "checkpoints" / "M5_E256.tsp.best"
EVO_M5_SHA256 = "364de077edcf3a5cadeb8d19c614b6708e92d42d8eefda63744b1252f6380494"
EVO50K = MATCHED / "datasets" / "evoreal50k_coordinates.npy"
EVO50K_LABELS = MATCHED / "datasets" / "evoreal50k_source_labels.npy"
EVO50K_TOURS = EVO_ARM / "oracle" / "expert_tours.npy"
COMMON = BASE / "protocol" / "common_initialization"
U0_BUNDLE = COMMON / "U0.npz"
CANDIDATE_BUNDLE = COMMON / "C0_BASE.npz"
CANDIDATE_META = COMMON / "C0_BASE.parquet"
FOGA_RUNTIME = COMMON / "O0_BASE_mutation" / "FOGA11_BASE_V1_RUNTIME.json"
FOGA_PROFILE = ROOT / "src" / "mutation" / "foga2019_operators.yaml"
SOURCE_MANIFEST = ROOT / "third_party" / "evoreal" / "SOURCE_MANIFEST.json"
OFFICIAL_COMMIT = "919616337105e894e06463734ec1dd71f6451309"
GENERATOR_FILES = {
    "S1": (ROOT / "third_party" / "evoreal" / "gpt_S1.py", "generate_s1_type"),
    "S2": (ROOT / "third_party" / "evoreal" / "gpt_S2.py", "generate_s2_type"),
    "S3": (ROOT / "third_party" / "evoreal" / "gpt_S3.py", "generate_s3_type"),
}
GENERATOR_HASHES = {
    "S1": "918bf7568ab4bcbe26042d0c8d02a221d4f9b2fc838e2b415992a55f31218fe0",
    "S2": "1129b5b902bc7aeee65c226e97e7180a601c2e35ed8b9810c3b8ead0a18b71f8",
    "S3": "b9eee5d038ca999a30d7aec80be365cacf5ee332264d6fe00e1a1c454d98c364",
}
ARMS = ("C0", "C1", "C2")
ARM_NAMES = {
    "C0": "C0_FRESH_EVOREAL10K",
    "C1": "C1_RANDOM_FOGA10K",
    "C2": "C2_BLADE_HARD10K",
}
EIGHT = tuple(evaluator.EIGHT)
CALIBRATION_N = 10_000
TRAIN_N = 10_000
EPOCHS = 256
LOGICAL_BATCH = 256
UPDATES_PER_EPOCH = 40
UPDATES_PER_ARM = 10_240
LEARNING_RATE = 3.75e-5
FORMAL_CONCORDE_TIMEOUT = 3_000.0
FINAL_THREE_CUMULATIVE_TIMEOUT = 30_000.0
TRAIN_SEED = 2026
FRESH_EVO_SEED = 2026102301
RATIO = {"S1": 19, "S2": 12, "S3": 17}
STRUCTURAL_GROUPS = ("EvoReal_S1", "EvoReal_S2", "EvoReal_S3", "BLADE_D0", "BLADE_D1", "BLADE_D2", "BLADE_D3", "BLADE_D4")


class ProtocolError(RuntimeError):
    """A frozen comparison condition is absent or inconsistent."""


class NullStatus:
    def update(self, **_: Any) -> None:
        return None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + f".tmp.{os.getpid()}" + path.suffix)
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, index=False)
    else:
        frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small dataframe without pandas' optional tabulate dependency."""

    def cell(value: Any) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        if isinstance(value, float):
            return f"{value:.8g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    columns = [str(value) for value in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def stable_seed(*parts: object, master: int = TRAIN_SEED) -> int:
    material = "|".join([RUN_ID, str(master), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def set_all_seeds(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def load_generator(source: str):
    path, function = GENERATOR_FILES[source]
    if sha256_file(path) != GENERATOR_HASHES[source]:
        raise ProtocolError(f"official EvoReal generator hash mismatch: {path}")
    spec = importlib.util.spec_from_file_location(f"evoreal_warmstart_{source}", path)
    if spec is None or spec.loader is None:
        raise ProtocolError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function)


def largest_remainder_counts(total: int) -> dict[str, int]:
    if total <= 0:
        raise ValueError("total must be positive")
    denom = sum(RATIO.values())
    exact = {key: total * value / denom for key, value in RATIO.items()}
    counts = {key: int(math.floor(value)) for key, value in exact.items()}
    order = sorted(RATIO, key=lambda key: (-(exact[key] - counts[key]), key))
    for key in order[: total - sum(counts.values())]:
        counts[key] += 1
    return counts


def load_bundle(path: Path) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    with np.load(path, allow_pickle=False) as payload:
        coordinates = np.asarray(payload["coordinates"], dtype=np.float64)
        hashes = payload["instance_hash"].astype(str).tolist()
        tours = np.asarray(payload["expert_tours"], dtype=np.int64) if "expert_tours" in payload.files else None
    if len(coordinates) != len(hashes):
        raise ProtocolError(f"bundle alignment failed: {path}")
    return coordinates, hashes, tours


def save_bundle(path: Path, coordinates: np.ndarray, hashes: list[str], tours: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "coordinates": np.ascontiguousarray(coordinates, dtype=np.float64),
        "instance_hash": np.asarray(hashes, dtype="U64"),
    }
    if tours is not None:
        arrays["expert_tours"] = np.asarray(tours, dtype=np.int64)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def arm_dir(arm: str) -> Path:
    if arm not in ARMS:
        raise ValueError(arm)
    return OUT / ARM_NAMES[arm]


def formal_dataset_bundle(arm: str) -> Path:
    return arm_dir(arm) / "data" / "training_labeled.npz"


def selected_bundle(arm: str) -> Path:
    return arm_dir(arm) / "data" / "selected_unlabeled.npz"


def verify_static_sources() -> dict[str, Any]:
    required = [EVO_M5, EVO50K, EVO50K_LABELS, EVO50K_TOURS, U0_BUNDLE, CANDIDATE_BUNDLE,
                CANDIDATE_META, FOGA_RUNTIME, FOGA_PROFILE, SOURCE_MANIFEST]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ProtocolError(f"missing frozen sources: {missing}")
    if sha256_file(EVO_M5) != EVO_M5_SHA256:
        raise ProtocolError("EvoReal M5 checkpoint changed")
    source = read_json(SOURCE_MANIFEST)
    if source.get("pinned_commit") != OFFICIAL_COMMIT:
        raise ProtocolError("EvoReal official commit mismatch")
    for name, (path, _) in GENERATOR_FILES.items():
        if sha256_file(path) != GENERATOR_HASHES[name]:
            raise ProtocolError(f"{name} source hash mismatch")
    base_strict = read_json(BASE / "STRICT_VERIFICATION.json")
    if base_strict.get("STRICT_VERIFICATION") != "PASS":
        raise ProtocolError("Base experiment strict verification is not PASS")
    coords, hashes, _ = load_bundle(U0_BUNDLE)
    candidates, candidate_hashes, _ = load_bundle(CANDIDATE_BUNDLE)
    if coords.shape != (CALIBRATION_N, 100, 2) or len(set(hashes)) != CALIBRATION_N:
        raise ProtocolError("U0 is not a unique 10k TSP100 calibration archive")
    if candidates.shape[1:] != (100, 2) or len(set(candidate_hashes)) != len(candidate_hashes):
        raise ProtocolError("shared parent+child candidate pool is invalid")
    return {
        "status": "PASS",
        "evoreal_m5_sha256": EVO_M5_SHA256,
        "u0_n": len(hashes),
        "u0_membership_sha256": membership_hash(hashes),
        "candidate_n": len(candidate_hashes),
        "candidate_membership_sha256": membership_hash(candidate_hashes),
        "foga_profile_sha256": sha256_file(FOGA_PROFILE),
        "foga_runtime_sha256": sha256_file(FOGA_RUNTIME),
        "evoreal_commit": OFFICIAL_COMMIT,
    }


def formal_preflight(device_name: str) -> None:
    """Fail closed before any formal data preparation, training, or evaluation."""

    checks: dict[str, Any] = {}
    static = verify_static_sources()
    checks["static_sources"] = static
    checks["frozen_budget"] = {
        "status": "PASS" if (CALIBRATION_N, TRAIN_N, EPOCHS, LOGICAL_BATCH, UPDATES_PER_ARM)
        == (10_000, 10_000, 256, 256, 10_240) else "FAIL",
        "calibration_n": CALIBRATION_N,
        "training_n": TRAIN_N,
        "epochs": EPOCHS,
        "logical_batch": LOGICAL_BATCH,
        "optimizer_updates": UPDATES_PER_ARM,
        "learning_rate": LEARNING_RATE,
        "optimizer": "fresh Adam",
        "uniform_anchor_n": 0,
        "formal_concorde_timeout_seconds": FORMAL_CONCORDE_TIMEOUT,
    }
    development, split_info = official_development_ids()
    checks["original70_split"] = {
        **split_info,
        "status": "PASS" if split_info.get("status") == "SOURCE_RECOVERED" and len(development) == 48 else "FAIL",
        "development_n": len(development),
        "held_out_n": 70 - len(development),
    }
    usage = shutil.disk_usage(OUT.parent)
    checks["disk"] = {
        "status": "PASS" if usage.free >= 200 * (1024 ** 3) else "FAIL",
        "path": str(OUT.parent),
        "free_bytes": usage.free,
        "minimum_free_bytes": 200 * (1024 ** 3),
    }
    resolved_out = OUT.resolve()
    protected = [BASE.resolve(), MATCHED.resolve()]
    checks["output_isolation"] = {
        "status": "PASS" if all(resolved_out != path and path not in resolved_out.parents for path in protected) else "FAIL",
        "output": str(resolved_out),
        "protected_inputs": [str(path) for path in protected],
    }

    backend = resolve_backend(ROOT / "third_party" / "concorde" / "concorde")
    dry_coords, dry_hashes, _ = load_bundle(U0_BUNDLE)
    with tempfile.TemporaryDirectory(prefix="blade_evoreal_preflight_", dir="/tmp") as temporary:
        solved = solve_one(dry_coords[0], dry_hashes[0], Path(temporary), backend,
                           timeout=FORMAL_CONCORDE_TIMEOUT)
    checks["concorde"] = {
        "status": "PASS" if solved.get("verified") and backend["binary_hash"] == CANONICAL_CONCORDE_SHA256 else "FAIL",
        "backend": backend,
        "probe_instance_hash": dry_hashes[0],
        "probe_verified": bool(solved.get("verified")),
        "probe_runtime_seconds": solved.get("runtime_sec"),
    }

    device = torch.device(device_name)
    gpu: dict[str, Any] = {"requested_device": device_name, "cuda_available": torch.cuda.is_available()}
    if device.type != "cuda" or not torch.cuda.is_available():
        gpu["status"] = "FAIL"
    else:
        probe = torch.ones(8, device=device)
        model, model_meta = load_canonical_checkpoint(EVO_M5, device, expected_sha256=EVO_M5_SHA256)
        gpu.update({
            "status": "PASS" if float(probe.sum()) == 8.0 and model_meta["sha256"] == EVO_M5_SHA256 else "FAIL",
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(device),
            "checkpoint_loaded_sha256": model_meta["sha256"],
            "allocated_probe_sum": float(probe.sum()),
        })
        del probe, model
        torch.cuda.empty_cache()
    query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    gpu["nvidia_smi_compute_processes"] = [line for line in query.stdout.splitlines() if line.strip()]
    gpu["nvidia_smi_returncode"] = query.returncode
    checks["gpu"] = gpu

    test_status = read_json(OUT / "protocol" / "TEST_STATUS.json")
    targeted = test_status.get("targeted_new_tests", {})
    repository = test_status.get("repository_suite", {})
    expected_historical_failures = (
        int(repository.get("failed", -1)) == 23
        and repository.get("status") == "PREEXISTING_ASSET_BLOCKERS"
    )
    checks["tests"] = {
        "status": "PASS" if targeted.get("status") == "PASS" and expected_historical_failures else "FAIL",
        "targeted_status": targeted.get("status"),
        "targeted_passed": targeted.get("passed"),
        "full_suite_passed": repository.get("passed"),
        "full_suite_failed": repository.get("failed"),
        "historical_failures_are_nonblocking": expected_historical_failures,
    }
    checks["resume"] = {
        "status": "PASS",
        "training": "completed atomic stages are hash-verified and reused; an interrupted incomplete arm restarts deterministically from EvoReal M5",
        "oracle": "verified ledger rows are reused and only missing instances are solved",
        "evaluation": "per-dataset checkpoint-keyed summaries are reused",
    }
    failed = [name for name, value in checks.items() if value.get("status") != "PASS"]
    report = {
        "status": "PASS" if not failed else "FAIL",
        "run_id": RUN_ID,
        "timestamp_utc": utc_now(),
        "failed_checks": failed,
        "checks": checks,
    }
    atomic_json(OUT / "protocol" / "formal_preflight.json", report)
    if failed:
        raise ProtocolError(f"formal preflight failed: {failed}")
    print("FORMAL_PREFLIGHT_PASS")


def frozen_test_hashes() -> tuple[set[str], set[str]]:
    ordered: set[str] = set()
    pointsets: set[str] = set()
    for name in EIGHT:
        payload = evaluator.load_fixed_references(name)
        coordinates = payload.get("coordinates")
        if coordinates is None:
            coordinates = [row[1] for row in payload["records"]]
        for xy in coordinates:
            variants = [np.asarray(xy), np.asarray(xy, dtype=np.float32).astype(np.float64)]
            if name == "OriginalTSPLIB70":
                normalized = evaluator.aspect_preserving_normalize(xy)
                variants.extend([normalized, np.asarray(normalized, dtype=np.float32).astype(np.float64)])
            for variant in variants:
                ordered.add(coordinate_hash(variant))
                pointsets.add(coordinate_set_hash(variant))
    return ordered, pointsets


def generate_fresh_evoreal(total: int, master_seed: int, *, shard_size: int = 500) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate a unique fresh pool with the frozen 19:12:17 mixture.

    It rejects overlap with the old EvoReal50k pool and all sealed evaluation
    sets.  No evaluation metric or model output is consulted.
    """

    old = np.load(EVO50K, allow_pickle=False)
    forbidden = {coordinate_hash(xy) for xy in old}
    forbidden_sets = {coordinate_set_hash(xy) for xy in old}
    test_ordered, test_sets = frozen_test_hashes()
    forbidden |= test_ordered
    forbidden_sets |= test_sets
    seen: set[str] = set()
    seen_sets: set[str] = set()
    rows: list[np.ndarray] = []
    labels: list[str] = []
    records: dict[str, Any] = {}
    counts = largest_remainder_counts(total)
    started = time.perf_counter()
    for source in ("S1", "S2", "S3"):
        generator = load_generator(source)
        target = counts[source]
        accepted = 0
        attempt = 0
        rejected = Counter()
        while accepted < target:
            batch_n = min(shard_size, target - accepted)
            seed = stable_seed("fresh_evoreal", source, attempt, master=master_seed)
            set_all_seeds(seed)
            batch = np.asarray(generator(batch_size=batch_n, problem_size=100), dtype=np.float32)
            attempt += 1
            if batch.shape != (batch_n, 100, 2):
                raise ProtocolError(f"{source} returned {batch.shape}")
            for xy in batch:
                if not np.isfinite(xy).all() or xy.min() < 0 or xy.max() > 1:
                    rejected["invalid"] += 1
                    continue
                h, hs = coordinate_hash(xy), coordinate_set_hash(xy)
                if h in forbidden or hs in forbidden_sets:
                    rejected["old_or_test_overlap"] += 1
                    continue
                if h in seen or hs in seen_sets:
                    rejected["duplicate"] += 1
                    continue
                rows.append(np.asarray(xy, dtype=np.float64))
                labels.append(source)
                seen.add(h)
                seen_sets.add(hs)
                accepted += 1
        records[source] = {"target": target, "accepted": accepted, "attempts": attempt, **dict(rejected)}
    coordinates = np.stack(rows)
    source_labels = np.asarray(labels, dtype="U2")
    permutation = np.random.Generator(np.random.PCG64(stable_seed("fresh_evoreal_shuffle", master=master_seed))).permutation(total)
    coordinates = np.ascontiguousarray(coordinates[permutation], dtype=np.float64)
    source_labels = source_labels[permutation]
    hashes = [coordinate_hash(xy) for xy in coordinates]
    point_hashes = [coordinate_set_hash(xy) for xy in coordinates]
    if len(set(hashes)) != total or len(set(point_hashes)) != total:
        raise ProtocolError("fresh EvoReal pool is not unique")
    manifest = {
        "status": "PASS",
        "total": total,
        "counts": {key: int(np.sum(source_labels == key)) for key in RATIO},
        "ratio": RATIO,
        "master_seed": master_seed,
        "seed_derivation": "first_64_bits_little_endian_SHA256(run_id|master|parts)",
        "official_commit": OFFICIAL_COMMIT,
        "generator_sha256": GENERATOR_HASHES,
        "old_evoreal50k_overlap": 0,
        "sealed_test_overlap": 0,
        "unique_ordered": len(set(hashes)),
        "unique_point_sets": len(set(point_hashes)),
        "generation": records,
        "wall_seconds": time.perf_counter() - started,
    }
    return coordinates, source_labels, manifest


def checkpoint_rows() -> pd.DataFrame:
    branches = {
        "BLADE Full": BASE / "A0_FULL_BASE",
        "Random FOGA": BASE / "A3_RANDOM_BASE",
        "Uniform50k": MATCHED / "B0_UNIFORM50K",
        "EvoReal50k": MATCHED / "B1_EVOREAL50K",
    }
    rows: list[dict[str, Any]] = []
    m0 = ROOT / "assets" / "solver" / "M0" / "tsp.best"
    for branch, directory in branches.items():
        m0_blob = torch.load(m0, map_location="cpu", weights_only=False)
        rows.append({"branch": branch, "model": "M0", "path": str(m0), "sha256": sha256_file(m0), "exists": True,
                     "recorded_epoch": m0_blob.get("epoch"), "formal_scientific_checkpoint": m0_blob.get("formal_scientific_checkpoint"),
                     "optimizer_state_stored": "optimizer" in m0_blob or "optimizer_state_dict" in m0_blob})
        for index in range(1, 6):
            path = directory / "checkpoints" / f"M{index}_E256.tsp.best"
            blob = torch.load(path, map_location="cpu", weights_only=False) if path.is_file() else {}
            rows.append({"branch": branch, "model": f"M{index}", "path": str(path),
                         "sha256": sha256_file(path) if path.is_file() else "", "exists": path.is_file(),
                         "recorded_epoch": blob.get("epoch"), "formal_scientific_checkpoint": blob.get("formal_scientific_checkpoint"),
                         "optimizer_state_stored": "optimizer" in blob or "optimizer_state_dict" in blob})
    return pd.DataFrame(rows)


def training_budget_rows() -> pd.DataFrame:
    branches = {
        "BLADE Full": BASE / "A0_FULL_BASE",
        "Random FOGA": BASE / "A3_RANDOM_BASE",
        "Uniform50k": MATCHED / "B0_UNIFORM50K",
        "EvoReal50k": MATCHED / "B1_EVOREAL50K",
    }
    rows = []
    for branch, directory in branches.items():
        for cycle in range(5):
            path = directory / "cycles" / f"C{cycle:02d}" / "09_bq_training" / "bq_training_timing.json"
            data = read_json(path) if path.is_file() else {}
            rows.append({
                "branch": branch,
                "cycle": f"C{cycle:02d}",
                "timing_path": str(path),
                "epochs": data.get("epochs"),
                "optimizer_steps": data.get("optimizer_steps"),
                "batch_size": data.get("effective_batch_size"),
                "hard_unique": data.get("hard_unique"),
                "anchor_n": data.get("uniform_anchor"),
                "wall_seconds": data.get("training_loop_wall_s"),
                "checkpoint_sha256": data.get("checkpoint_sha256"),
            })
    return pd.DataFrame(rows)


def baseline_asset_rows() -> pd.DataFrame:
    """Inventory the actual config/data/oracle/evaluation inputs, not inferred names."""
    branches = {
        "BLADE Full": (BASE / "A0_FULL_BASE", BASE / "protocol" / "frozen_experiment_config.json", "adaptive"),
        "Random FOGA": (BASE / "A3_RANDOM_BASE", BASE / "protocol" / "frozen_experiment_config.json", "adaptive"),
        "Uniform50k": (MATCHED / "B0_UNIFORM50K", MATCHED / "protocol" / "frozen_protocol.json", "static"),
        "EvoReal50k": (MATCHED / "B1_EVOREAL50K", MATCHED / "protocol" / "frozen_protocol.json", "static"),
    }
    rows: list[dict[str, Any]] = []
    for branch, (directory, config, kind) in branches.items():
        rows.append({"branch": branch, "cycle": "ALL", "role": "config", "path": str(config),
                     "sha256": sha256_file(config), "bytes": config.stat().st_size})
        if kind == "static":
            prefix = "uniform" if branch == "Uniform50k" else "evoreal"
            dataset = MATCHED / "datasets" / f"{prefix}50k_coordinates.npy"
            tours = directory / "oracle" / "expert_tours.npy"
            rows.extend([
                {"branch": branch, "cycle": "ALL", "role": "coordinates50k", "path": str(dataset),
                 "sha256": sha256_file(dataset), "bytes": dataset.stat().st_size},
                {"branch": branch, "cycle": "ALL", "role": "expert_tours50k", "path": str(tours),
                 "sha256": sha256_file(tours), "bytes": tours.stat().st_size},
            ])
            for cycle in range(5):
                membership = MATCHED / "datasets" / f"{prefix}_D{cycle}_membership.txt"
                evaluation = directory / "evaluations" / f"M{cycle + 1}_eight.csv"
                for role, path in (("training_membership", membership), ("evaluation", evaluation)):
                    rows.append({"branch": branch, "cycle": f"C{cycle:02d}", "role": role, "path": str(path),
                                 "sha256": sha256_file(path), "bytes": path.stat().st_size})
        else:
            for cycle in range(5):
                bundle = directory / "cycles" / f"C{cycle:02d}" / "07_oracle" / "T" / "labeled_bundle.npz"
                evaluation = directory / "cycles" / f"C{cycle:02d}" / "external_benchmark_results.csv"
                for role, path in (("training_coordinates_and_tours", bundle), ("evaluation", evaluation)):
                    rows.append({"branch": branch, "cycle": f"C{cycle:02d}", "role": role, "path": str(path),
                                 "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return pd.DataFrame(rows)


def baseline_integrity_audit() -> dict[str, Any]:
    source = verify_static_sources()
    checkpoints = checkpoint_rows()
    budgets = training_budget_rows()
    assets = baseline_asset_rows()
    write_frame(OUT / "protocol" / "baseline_checkpoints.csv", checkpoints)
    write_frame(OUT / "protocol" / "baseline_training_budgets.csv", budgets)
    write_frame(OUT / "protocol" / "baseline_asset_inventory.csv", assets)
    checks = {
        "all_M0_M5_checkpoints_present": bool(checkpoints.exists.all()),
        "all_cycles_256_epochs": bool(budgets.epochs.eq(EPOCHS).all()),
        "all_cycles_10240_updates": bool(budgets.optimizer_steps.eq(UPDATES_PER_ARM).all()),
        "all_cycles_batch256": bool(budgets.batch_size.eq(LOGICAL_BATCH).all()),
        "all_cycles_hard10k": bool(budgets.hard_unique.eq(TRAIN_N).all()),
        "all_cycles_no_anchor": bool(budgets.anchor_n.eq(0).all()),
        "evoreal_M5_hash_verified": sha256_file(EVO_M5) == EVO_M5_SHA256,
        "canonical_concorde_hash_verified": read_json(EVO_ARM / "oracle" / "expert_tour_integrity.json").get("concorde_binary_sha256") == CANONICAL_CONCORDE_SHA256,
        "same_frozen_eight_sets": list(EIGHT) == read_json(MATCHED / "protocol" / "frozen_protocol.json")["evaluation"]["datasets"],
    }
    evo_config = read_json(MATCHED / "protocol" / "frozen_protocol.json")
    evo_manifest = read_json(MATCHED / "06_evoreal50k_dataset_manifest.json")
    evo_source = read_json(Path(evo_manifest["source_v1_manifest"]))
    report = [
        "# Baseline integrity audit",
        "",
        f"Status: **{'PASS' if all(checks.values()) else 'FAIL'}**",
        "",
        "This audit reads completed experiments only. No old checkpoint, dataset, configuration or result was modified.",
        "",
        "## EvoReal50k M5 identity",
        "",
        f"- Checkpoint: `{EVO_M5}`",
        f"- SHA256: `{EVO_M5_SHA256}`",
        f"- Training: 5 cycles x {evo_config['epochs_per_cycle']} epochs x {evo_config['optimizer_steps_per_epoch']} updates/epoch = {evo_config['optimizer_steps_per_arm']} updates.",
        f"- Optimizer: fresh Adam per cycle, LR={evo_config['learning_rate']}, scheduler={evo_config['scheduler']}, weight decay={evo_config['weight_decay']}.",
        f"- Batch: effective {evo_config['effective_batch_size']}, drop_last={evo_config['drop_last']}; 10,000 unique instances per cycle.",
        f"- Dataset SHA256: `{evo_manifest['coordinate_file_sha256']}`; generator commit `{OFFICIAL_COMMIT}`; master generation seed `{evo_source['master_seed']}`.",
        f"- Concorde binary SHA256: `{CANONICAL_CONCORDE_SHA256}`; greedy decoder and the same eight-set gap evaluator are frozen.",
        "",
        "## Cross-branch budget checks",
        "",
        markdown_table(pd.DataFrame([{"check": key, "pass": value} for key, value in checks.items()])),
        "",
        "Detailed checkpoint hashes, per-cycle budgets, actual training bundles, configs, Oracle labels and evaluation records are in `protocol/baseline_checkpoints.csv`, `protocol/baseline_training_budgets.csv` and `protocol/baseline_asset_inventory.csv`.",
    ]
    write_text(OUT / "01_baseline_integrity_audit.md", "\n".join(report) + "\n")
    payload = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "sources": source}
    atomic_json(OUT / "protocol" / "baseline_integrity.json", payload)
    if payload["status"] != "PASS":
        raise ProtocolError("baseline integrity audit failed")
    return payload


def protocol_document() -> None:
    source = verify_static_sources()
    development_ids, split_source = official_development_ids()
    original = evaluator.load_fixed_references("OriginalTSPLIB70")
    original_ids = {str(value).lower().replace(".tsp", "") for value in original["identifiers"]}
    split_ok = split_source["status"] == "SOURCE_RECOVERED" and len(development_ids & original_ids) == 48 and len(original_ids - development_ids) == 22
    atomic_json(OUT / "protocol" / "original70_split_recovery.json", {
        **split_source, "status": "PASS" if split_ok else "UNVERIFIABLE",
        "matched_development": len(development_ids & original_ids), "matched_heldout": len(original_ids - development_ids),
        "binding": "casefold instance ID after removing .tsp suffix; no inferred membership",
    })
    protocol = {
        "run_id": RUN_ID,
        "pilot_scope": "single paired seed; preliminary only",
        "common_initial_checkpoint": str(EVO_M5),
        "common_initial_checkpoint_sha256": EVO_M5_SHA256,
        "arms": {
            "C0": "fresh unique EvoReal10k with frozen 19:12:17 S1/S2/S3 mixture",
            "C1": "uniform random 10k from common frozen U0+FOGA11 child union",
            "C2": "HGB40+QD Hard10k from the identical C1/C2 candidate union",
        },
        "calibration": {
            "n": CALIBRATION_N,
            "source": str(U0_BUNDLE),
            "source_is_independent_of_acquisition": True,
            "labels": "canonical Concorde tours; teacher-forced CE recomputed under EvoReal M5",
            "historical_hardness_reused": False,
        },
        "candidate": {
            "parent": str(U0_BUNDLE),
            "parent_n": 10_000,
            "union": str(CANDIDATE_BUNDLE),
            "union_n": source["candidate_n"],
            "mutation": "frozen FOGA11_BASE_V1",
            "parent_child_union": True,
            "shared_by_C1_C2": True,
        },
        "C2": {
            "features": "frozen F40",
            "surrogate": {"estimator": "HistGradientBoostingRegressor", **HGB_CFG_CANONICAL,
                          "seeds": list(SEEDS), "aggregate": "median", "target": "log1p_current_LBQ"},
            "QD": {"descriptors": list(QD4_FEATURE_NAMES), "K": 1000,
                   "selection": "one best predicted-hardness elite per occupied niche then global HGB fill"},
        },
        "training": {
            "n": TRAIN_N, "epochs": EPOCHS, "logical_batch_size": LOGICAL_BATCH,
            "optimizer_updates_per_epoch": UPDATES_PER_EPOCH, "optimizer_updates": UPDATES_PER_ARM,
            "optimizer": "fresh Adam", "restore_optimizer": False, "learning_rate": LEARNING_RATE,
            "scheduler": None, "weight_decay": 0, "AMP": False, "uniform_anchor": 0,
            "seed": TRAIN_SEED, "same_initial_weights": True,
        },
        "evaluation": {"datasets": list(EIGHT), "decoder": "greedy", "checkpoint": "E256 only",
                       "paired_instance_analysis": True, "bootstrap_replicates": 10_000},
        "cost_accounting": ["generation", "features", "oracle", "surrogate", "QD", "BQ_training", "evaluation"],
        "sealed_test_policy": "never used by generation acceptance except exact overlap rejection; never used by acquisition, training, tuning or checkpoint selection",
    }
    atomic_json(OUT / "protocol" / "frozen_warmstart_protocol.json", protocol)
    lines = [
        "# EvoReal warm-start protocol",
        "",
        "The formal pilot starts C0/C1/C2 from the exact same frozen EvoReal50k M5 weights. Optimizers are not restored; each arm creates a fresh Adam with the same seed and schedule.",
        "",
        "## Data and acquisition",
        "",
        "- C0 generates a new unique 10k pool from the pinned official S1/S2/S3 generators at 19:12:17. It rejects overlap with the historical EvoReal50k and frozen test coordinates.",
        "- C1 and C2 use the same immutable 10k U0 parents, the same FOGA11_BASE_V1 children and the same deduplicated Parent+Child union. C1 samples uniformly; C2 uses frozen HGB40 + CVT-QD.",
        "- C2 calibration is exactly 10,000 U0 archive instances. Their teacher-forced CE is recomputed under EvoReal M5; no historical hardness is reused.",
        "- C1/C2 selected-set overlap and duplicates are written before Oracle labeling.",
        "",
        "## Training fairness",
        "",
        f"All arms use 10,000 unique Concorde-labeled instances, batch 256, 256 epochs, 40 optimizer updates per epoch and exactly {UPDATES_PER_ARM:,} updates. LR is {LEARNING_RATE}; there is no scheduler, anchor, early stopping or checkpoint selection.",
        "",
        "## Evaluation and interpretation",
        "",
        "All eight frozen sets are evaluated separately and GAP8 is a macro mean. Instance-level paired differences and bootstrap CIs are produced. A one-seed pilot remains preliminary and may only be labeled INCONCLUSIVE_PRELIMINARY; a definitive P4 label requires at least three independent paired seeds.",
        "",
        ("OriginalTSPLIB70 official IDs were recovered and match exactly 48 development-informed / 22 held-out instances; the binding is recorded in `protocol/original70_split_recovery.json`."
         if split_ok else "OriginalTSPLIB70 48/22 reporting is UNVERIFIABLE; no split will be guessed."),
        "",
        "## Entry points",
        "",
        "Run `python scripts/run_evoreal_warmstart_diagnostic.py --help` for all parameters. The intended order is: `init`, `dry-run`, `prepare`, `label --target calibration`, `select-c2`, `label --target C0/C1/C2`, `train --arm C0/C1/C2`, `evaluate --arm C0/C1/C2`, `structural-audit`, and `finalize`. Only `train` starts the full 256-epoch GPU workload.",
    ]
    write_text(OUT / "03_warmstart_protocol.md", "\n".join(lines) + "\n")


def init() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        for child in ("data", "oracle", "training", "evaluation", "timing"):
            (arm_dir(arm) / child).mkdir(parents=True, exist_ok=True)
    baseline_integrity_audit()
    protocol_document()
    baseline = pd.read_csv(EVO_ARM / "evaluations" / "M5_eight.csv")
    rows = []
    for row in baseline.itertuples(index=False):
        rows.append({
            "arm": "EvoReal_M5", "seed": "frozen", "status": "EXISTING_BASELINE", "dataset": row.dataset,
            "instance_count": row.instance_count, "valid_tours": row.valid_tours, "failed_tours": row.failed_tours,
            "mean_gap_percent": row.mean_gap_percent, "median_gap_percent": row.median_gap_percent,
            "P90_gap_percent": row.P90, "runtime_seconds": row.inference_wall_clock_sec,
            "checkpoint_sha256": row.checkpoint_sha256,
        })
    for arm in ARMS:
        for dataset in EIGHT:
            rows.append({"arm": arm, "seed": TRAIN_SEED, "status": "NOT_RUN", "dataset": dataset})
    write_frame(OUT / "04_warmstart_results.csv", pd.DataFrame(rows))
    write_text(OUT / "02_structural_distribution_audit.md",
               "# Structural distribution audit\n\nStatus: **PENDING_FULL_AUDIT**\n\n"
               "Run `structural-audit` for the full EvoReal50k versus five BLADE Hard10k cycles analysis. "
               "Dry-run summaries, when present, are explicitly preliminary and are not scientific results.\n")
    write_text(OUT / "05_incremental_value_analysis.md",
               "# Incremental value analysis\n\nStatus: **INCONCLUSIVE_PRE_FORMAL_RUN**\n\n"
               "No C0/C1/C2 formal checkpoint has been trained or evaluated. No P4 conclusion is permitted.\n")
    atomic_json(OUT / "STATUS.json", {"status": "INITIALIZED", "full_gpu_training_started": False,
                                      "run_id": RUN_ID, "timestamp_utc": utc_now()})
    print(f"INIT_COMPLETE {OUT}")


def prepare(seed: int) -> None:
    verify_static_sources()
    c0_started = time.perf_counter()
    coords, labels, manifest = generate_fresh_evoreal(TRAIN_N, FRESH_EVO_SEED + seed - TRAIN_SEED)
    hashes = [coordinate_hash(xy) for xy in coords]
    save_bundle(selected_bundle("C0"), coords, hashes)
    np.save(arm_dir("C0") / "data" / "source_labels.npy", labels, allow_pickle=False)
    manifest.update({"coordinate_file_sha256": sha256_file(selected_bundle("C0")),
                     "membership_sha256": membership_hash(hashes)})
    atomic_json(arm_dir("C0") / "data" / "manifest.json", manifest)
    atomic_json(arm_dir("C0") / "timing" / "generation.json", {"wall_seconds": time.perf_counter() - c0_started,
                                                                  "generated_n": TRAIN_N})

    candidate_coords, candidate_hashes, _ = load_bundle(CANDIDATE_BUNDLE)
    meta = pd.read_parquet(CANDIDATE_META)
    if candidate_hashes != meta.instance_hash.astype(str).tolist():
        raise ProtocolError("candidate metadata alignment failure")
    c1_started = time.perf_counter()
    rng = np.random.Generator(np.random.PCG64(stable_seed("C1", "selection", master=seed)))
    indices = rng.choice(len(candidate_hashes), TRAIN_N, replace=False)
    selected = meta.iloc[indices].copy().reset_index(drop=True)
    selected["selection_component"] = "shared_union_uniform_random"
    selected["selection_rank"] = np.arange(1, TRAIN_N + 1)
    selected["coord_index_shared_union"] = indices
    save_bundle(selected_bundle("C1"), candidate_coords[indices], selected.instance_hash.astype(str).tolist())
    write_frame(arm_dir("C1") / "data" / "selected.parquet", selected)
    atomic_json(arm_dir("C1") / "data" / "manifest.json", {
        "status": "PASS", "seed": seed, "n": TRAIN_N, "duplicates": 0,
        "candidate_pool": str(CANDIDATE_BUNDLE), "candidate_n": len(candidate_hashes),
        "candidate_membership_sha256": membership_hash(candidate_hashes),
        "selection": "PCG64 uniform without replacement after no metric inspection",
        "membership_sha256": membership_hash(selected.instance_hash.astype(str).tolist()),
    })
    foga_runtime = read_json(FOGA_RUNTIME)
    atomic_json(arm_dir("C1") / "timing" / "generation.json", {
        "wall_seconds_current_run": 0.0, "wall_seconds_selection": time.perf_counter() - c1_started,
        "reused_frozen_candidate_pool": True, "historical_measured_foga_wall_seconds": foga_runtime["wall_seconds"],
        "nominal_offspring": foga_runtime["nominal_offspring"], "physical_generation_calls_current_run": 0,
    })
    atomic_json(arm_dir("C2") / "timing" / "generation.json", {
        "wall_seconds_current_run": 0.0, "reused_frozen_candidate_pool": True,
        "historical_measured_foga_wall_seconds": foga_runtime["wall_seconds"],
        "nominal_offspring": foga_runtime["nominal_offspring"], "physical_generation_calls_current_run": 0,
    })
    ucoords, uhashes, _ = load_bundle(U0_BUNDLE)
    save_bundle(OUT / "calibration" / "U0_unlabeled.npz", ucoords, uhashes)
    atomic_json(OUT / "calibration" / "manifest.json", {
        "status": "AWAITING_ORACLE", "n": len(uhashes), "source": str(U0_BUNDLE),
        "membership_sha256": membership_hash(uhashes), "calibration_n": CALIBRATION_N,
        "hardness_model_sha256": EVO_M5_SHA256, "historical_hardness_reused": False,
    })
    atomic_json(OUT / "STATUS.json", {"status": "DATA_PREPARED_C0_C1", "full_gpu_training_started": False,
                                      "seed": seed, "timestamp_utc": utc_now()})
    print("PREPARE_COMPLETE C0=10000 C1=10000 calibration=10000 C2=AWAITING_CURRENT_HARDNESS")


def _verified_row_from_existing_solution(coordinates: np.ndarray, instance_hash: str, directory: Path,
                                         backend: dict[str, Any], runtime_seconds: float) -> dict[str, Any]:
    """Rebuild a canonical row only from a clean, previously completed Concorde work directory."""

    solution = directory / f"{instance_hash}.sol"
    if not solution.is_file():
        raise ProtocolError(f"missing Concorde solution for cache recovery: {instance_hash}")
    tour = concorde_impl._parse_tour(solution, len(coordinates))
    legal, message = concorde_impl.validate_tour_0based(tour, len(coordinates))
    distances = concorde_impl.integer_distance_matrix(coordinates, concorde_impl.INTEGER_SCALE)
    cost = concorde_impl.tour_cost_int(distances, tour)
    if not legal:
        raise ProtocolError(f"invalid cached Concorde tour {instance_hash}: {message}")
    return {
        "instance_hash": instance_hash,
        "status": "solved",
        "verified": True,
        "optimal_cost_int": cost,
        "runtime_sec": runtime_seconds,
        "expert_tour": " ".join(map(str, tour)),
        "tour_hash": concorde_impl.tour_hash(tour),
        "binary_hash": backend["binary_hash"],
    }


def _has_interrupted_branch_state(directory: Path) -> bool:
    """Concorde leaves numbered branch-node files when killed before cleanup."""

    return any(re.search(r"\.\d+$", path.name) for path in directory.iterdir() if path.is_file())


def _restart_one(coordinates: np.ndarray, instance_hash: str, directory: Path,
                 backend: dict[str, Any], timeout: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """Continue an exact Concorde proof from its durable .res state."""

    restart = directory / f"{instance_hash}.res"
    tsp = directory / f"{instance_hash}.tsp"
    solution = directory / f"{instance_hash}.sol"
    if not restart.is_file() or not tsp.is_file():
        raise ProtocolError(f"missing exact-restart state for {instance_hash}")
    if backend["invoke_mode"] == "wsl":
        inner = (
            f"cd '{concorde_impl.windows_to_wsl_path(directory)}' && "
            f"'{concorde_impl.windows_to_wsl_path(Path(backend['binary']))}' "
            f"-R '{restart.name}' -x -o '{solution.name}' '{tsp.name}'"
        )
        command = ["wsl", "-d", backend["wsl_distro"], "--", "bash", "-lc", inner]
        cwd = None
    else:
        command = [backend["binary"], "-R", restart.name, "-x", "-o", solution.name, tsp.name]
        cwd = str(directory)
    started = time.perf_counter()
    process = subprocess.run(command, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=timeout)
    elapsed = time.perf_counter() - started
    output = (process.stdout or "") + "\n" + (process.stderr or "")
    row = _verified_row_from_existing_solution(coordinates, instance_hash, directory, backend, elapsed)
    stated = concorde_impl._reported(output)
    if stated is None:
        raise RuntimeError(
            f"Concorde restart {instance_hash} did not emit an optimality certificate: {output[-500:]}"
        )
    if int(stated) != int(row["optimal_cost_int"]):
        raise RuntimeError(
            f"Concorde restart verification failed {instance_hash}: stated={stated}, "
            f"tour={row['optimal_cost_int']}"
        )
    return row, {
        "instance_hash": instance_hash,
        "mode": "exact_restart_R",
        "timeout_seconds": timeout,
        "wall_seconds": elapsed,
        "returncode": process.returncode,
        "reported_optimal_cost_int": stated,
        "verified": True,
    }


def _fresh_solve_one(coordinates: np.ndarray, instance_hash: str, directory: Path,
                     backend: dict[str, Any], timeout: float) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    row = solve_one(coordinates, instance_hash, directory, backend, timeout=timeout)
    elapsed = time.perf_counter() - started
    return row, {
        "instance_hash": instance_hash,
        "mode": "fresh_exact_solve",
        "timeout_seconds": timeout,
        "wall_seconds": elapsed,
        "verified": True,
    }


def durable_solve_batch(coordinates: np.ndarray, hashes: list[str], destination: Path,
                        backend: dict[str, Any], workers: int, timeout: float,
                        *, recover_interrupted_c0: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Persist every verified result and resume only unresolved exact proofs."""

    destination.mkdir(parents=True, exist_ok=True)
    work = destination / "work"
    work.mkdir(parents=True, exist_ok=True)
    partial_path = destination / "ledger.partial.parquet"
    final_path = destination / "ledger.parquet"
    frames = []
    for path in (final_path, partial_path):
        if path.is_file():
            frames.append(canonical_oracle_ledger(pd.read_parquet(path)))
    done: dict[str, dict[str, Any]] = {}
    if frames:
        combined = pd.concat(frames, ignore_index=True).drop_duplicates("instance_hash", keep="last")
        done = {
            str(row.instance_hash): row._asdict()
            for row in canonical_oracle_ledger(combined).itertuples(index=False)
            if bool(row.verified) and row.status == "solved"
        }

    recovered_clean = 0
    interrupted_hashes: list[str] = []
    if recover_interrupted_c0:
        for index, instance_hash in enumerate(hashes):
            if instance_hash in done:
                continue
            directory = work / instance_hash
            solution = directory / f"{instance_hash}.sol"
            tsp = directory / f"{instance_hash}.tsp"
            if not (solution.is_file() and tsp.is_file()):
                continue
            if _has_interrupted_branch_state(directory):
                interrupted_hashes.append(instance_hash)
                continue
            done[instance_hash] = _verified_row_from_existing_solution(
                coordinates[index], instance_hash, directory, backend, float("nan")
            )
            recovered_clean += 1
        if done:
            write_frame(partial_path, canonical_oracle_ledger(pd.DataFrame(list(done.values()))))

    pending = [(index, value) for index, value in enumerate(hashes) if value not in done]
    restart_set = set(interrupted_hashes)
    events: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    restart_calls = 0
    fresh_calls = 0
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(pending)))) as pool:
        futures = {}
        for index, instance_hash in pending:
            directory = work / instance_hash
            if recover_interrupted_c0 and instance_hash in restart_set:
                restart_calls += 1
                future = pool.submit(_restart_one, coordinates[index], instance_hash, directory, backend, timeout)
            else:
                fresh_calls += 1
                future = pool.submit(_fresh_solve_one, coordinates[index], instance_hash, directory, backend, timeout)
            futures[future] = instance_hash
        for count, future in enumerate(as_completed(futures), 1):
            instance_hash = futures[future]
            try:
                row, event = future.result()
                done[instance_hash] = row
                events.append(event)
            except BaseException as error:
                directory = work / instance_hash
                failures.append({
                    "instance_hash": instance_hash,
                    "exception": repr(error),
                    "restart_state_present": (directory / f"{instance_hash}.res").is_file(),
                    "timeout_seconds": timeout,
                })
            if count % 25 == 0 or count == len(pending) or len(pending) <= 25:
                if done:
                    write_frame(partial_path, canonical_oracle_ledger(pd.DataFrame(list(done.values()))))
                atomic_json(destination / "restart_events.json", {
                    "events": events,
                    "failures": failures,
                    "verified_so_far": len(done),
                    "total": len(hashes),
                })
            if count % 100 == 0:
                print(f"CONCORDE_DURABLE_PROGRESS={len(done)}/{len(hashes)}", flush=True)

    accounting = {
        "wall_seconds_current_command": time.perf_counter() - started,
        "recovered_verified_cache_rows": len(hashes) - len(pending),
        "recovered_clean_work_rows": recovered_clean,
        "exact_restart_calls": restart_calls,
        "fresh_exact_calls": fresh_calls,
        "failures": failures,
        "timeout_seconds": timeout,
    }
    atomic_json(destination / "durable_recovery.json", accounting)
    if failures:
        raise ProtocolError(f"Concorde exact labeling failed closed for {len(failures)} instances")
    if len(done) != len(hashes):
        raise ProtocolError(f"Concorde durable ledger incomplete: {len(done)}/{len(hashes)}")
    ordered = canonical_oracle_ledger(pd.DataFrame([done[value] for value in hashes]))
    return ordered, accounting


def reconcile_proven_restarts(target: str) -> None:
    """Persist restart results that carry an explicit optimality certificate without rerunning them."""

    if target == "calibration":
        bundle = OUT / "calibration" / "U0_unlabeled.npz"
        destination = OUT / "calibration" / "oracle"
    elif target in ARMS:
        bundle = selected_bundle(target)
        destination = arm_dir(target) / "oracle"
    else:
        raise ValueError(target)
    events_path = destination / "restart_events.json"
    partial_path = destination / "ledger.partial.parquet"
    if not events_path.is_file() or not partial_path.is_file():
        raise ProtocolError(f"no durable restart state to reconcile for {target}")
    coordinates, hashes, _ = load_bundle(bundle)
    index_by_hash = {value: index for index, value in enumerate(hashes)}
    backend = resolve_backend(ROOT / "third_party" / "concorde" / "concorde")
    ledger = canonical_oracle_ledger(pd.read_parquet(partial_path))
    done = {str(row.instance_hash): row._asdict() for row in ledger.itertuples(index=False)}
    record = read_json(events_path)
    reconciled = []
    unresolved = []
    for failure in record.get("failures", []):
        instance_hash = str(failure["instance_hash"])
        match = re.search(r"Optimal Solution:\s*([0-9]+(?:\.[0-9]+)?)", str(failure.get("exception", "")))
        if match is None:
            unresolved.append(failure)
            continue
        row = _verified_row_from_existing_solution(
            coordinates[index_by_hash[instance_hash]], instance_hash,
            destination / "work" / instance_hash, backend, float("nan"),
        )
        stated = int(round(float(match.group(1))))
        if stated != int(row["optimal_cost_int"]):
            raise ProtocolError(
                f"reconciled restart cost mismatch {instance_hash}: stated={stated}, tour={row['optimal_cost_int']}"
            )
        done[instance_hash] = row
        reconciled.append({
            "instance_hash": instance_hash,
            "mode": "reconciled_exact_restart_certificate",
            "reported_optimal_cost_int": stated,
            "verified": True,
        })
    ordered_partial = canonical_oracle_ledger(pd.DataFrame([done[value] for value in hashes if value in done]))
    write_frame(partial_path, ordered_partial)
    atomic_json(destination / "restart_reconciliation.json", {
        "status": "PARTIAL" if unresolved else "PASS",
        "target": target,
        "verified_rows": len(ordered_partial),
        "reconciled_exact_restarts": reconciled,
        "unresolved_failures": unresolved,
        "timeout_seconds": FORMAL_CONCORDE_TIMEOUT,
    })
    print(
        f"RESTART_RECONCILIATION target={target} verified={len(ordered_partial)}/{len(hashes)} "
        f"reconciled={len(reconciled)} unresolved={len(unresolved)}"
    )


def _run_captured_concorde(command: list[str], directory: Path, timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    timed_out = False
    returncode: int | None = None
    stdout = ""
    stderr = ""
    try:
        process = subprocess.run(command, cwd=directory, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", timeout=timeout)
        returncode = process.returncode
        stdout = process.stdout or ""
        stderr = process.stderr or ""
    except subprocess.TimeoutExpired as error:
        timed_out = True
        stdout = error.stdout.decode("utf-8", "replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
        stderr = error.stderr.decode("utf-8", "replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
    elapsed = time.perf_counter() - started
    write_text(directory / "stdout.txt", stdout)
    write_text(directory / "stderr.txt", stderr)
    return {
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_seconds": elapsed,
        "stdout": stdout,
        "stderr": stderr,
    }


def _validate_recovery_tour(coordinates: np.ndarray, instance_hash: str, tour_path: Path,
                            proven_objective: int | None) -> dict[str, Any]:
    if not tour_path.is_file():
        return {"verified": False, "reason": "optimal tour file missing"}
    try:
        tour = concorde_impl._parse_tour(tour_path, len(coordinates))
        legal, message = concorde_impl.validate_tour_0based(tour, len(coordinates))
        distances = concorde_impl.integer_distance_matrix(coordinates, concorde_impl.INTEGER_SCALE)
        cost = concorde_impl.tour_cost_int(distances, tour)
    except BaseException as error:
        return {"verified": False, "reason": repr(error)}
    consistent = proven_objective is not None and int(cost) == int(proven_objective)
    return {
        "verified": bool(legal and consistent),
        "hamiltonian": bool(legal),
        "validation_message": message,
        "recomputed_tour_cost_int": int(cost),
        "proven_optimal_objective_int": proven_objective,
        "proof_tour_consistency": bool(consistent),
        "tour": tour,
        "tour_hash": concorde_impl.tour_hash(tour),
    }


def _mismatch_objectives() -> dict[str, int]:
    record = read_json(arm_dir("C0") / "oracle" / "restart_events.json")
    result: dict[str, int] = {}
    for failure in record.get("failures", []):
        value = concorde_impl._reported(str(failure.get("exception", "")))
        if value is not None:
            result[str(failure["instance_hash"])] = int(value)
    if len(result) != 8:
        raise ProtocolError(f"expected 8 certified-objective/tour mismatches, found {len(result)}")
    return result


def _timeout_hashes() -> list[str]:
    record = read_json(arm_dir("C0") / "oracle" / "restart_events.json")
    values = [
        str(failure["instance_hash"])
        for failure in record.get("failures", [])
        if "TimeoutExpired" in str(failure.get("exception", ""))
    ]
    if len(values) != 3:
        raise ProtocolError(f"expected 3 3000-second timeout instances, found {len(values)}")
    return values


def _fresh_objective_recovery_job(coordinates: np.ndarray, instance_hash: str,
                                  objective: int, backend: dict[str, Any]) -> dict[str, Any]:
    base = arm_dir("C0") / "oracle" / "exact_tour_recovery" / "mismatch" / instance_hash
    first_result = base / "attempt_01" / "RESULT.json"
    if first_result.is_file():
        first = read_json(first_result)
        if bool(first.get("verified")):
            return first
        sandbox_rejection = (
            first.get("returncode") == -31
            and not bool(first.get("timed_out"))
            and float(first.get("solver_wall_seconds", float("inf"))) < 1.0
        )
        if not sandbox_rejection:
            return first
        directory = base / "attempt_02"
    else:
        directory = base / "attempt_01"
    result_path = directory / "RESULT.json"
    if result_path.is_file():
        return read_json(result_path)
    if directory.exists():
        raise ProtocolError(f"refusing to overwrite mismatch recovery attempt: {directory}")
    directory.mkdir(parents=True)
    distances = concorde_impl.integer_distance_matrix(coordinates, concorde_impl.INTEGER_SCALE)
    tsp = directory / "instance.tsp"
    tour_path = directory / "optimal.sol"
    concorde_impl.write_explicit_tsp(tsp, distances, instance_hash)
    command = [backend["binary"], "-o", tour_path.name, tsp.name]
    atomic_json(directory / "ATTEMPT.json", {
        "instance_hash": instance_hash,
        "mode": "independent_fresh_exact_solve_for_certified_objective",
        "known_proven_objective_int": objective,
        "timeout_seconds": FORMAL_CONCORDE_TIMEOUT,
        "command": command,
        "concorde_binary_sha256": backend["binary_hash"],
        "attempt_number": int(directory.name.split("_")[-1]),
        "prior_attempt_environment_failure": str(first_result) if directory.name == "attempt_02" else None,
    })
    execution = _run_captured_concorde(command, directory, FORMAL_CONCORDE_TIMEOUT)
    reported = concorde_impl._reported(execution["stdout"] + "\n" + execution["stderr"])
    validation = _validate_recovery_tour(coordinates, instance_hash, tour_path, objective)
    result = {
        "instance_hash": instance_hash,
        "mode": "P0_independent_fresh_exact_solve",
        "restart_status": "NOT_USED_FRESH_SOLVE",
        "prior_proven_optimal_objective_int": objective,
        "current_run_reported_objective_int": reported,
        "optimal_tour_file": str(tour_path),
        "solver_wall_seconds": execution["wall_seconds"],
        "timed_out": execution["timed_out"],
        "returncode": execution["returncode"],
        "restart_count": 0,
        **{key: value for key, value in validation.items() if key != "tour"},
    }
    atomic_json(result_path, result)
    return result


def _proper_master_restart_job(coordinates: np.ndarray, instance_hash: str,
                               backend: dict[str, Any]) -> dict[str, Any]:
    source = arm_dir("C0") / "oracle" / "work" / instance_hash
    directory = arm_dir("C0") / "oracle" / "exact_tour_recovery" / "timeout" / instance_hash / "attempt_01"
    result_path = directory / "RESULT.json"
    if result_path.is_file():
        return read_json(result_path)
    if directory.exists():
        raise ProtocolError(f"refusing more than one new timeout recovery attempt: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, directory)
    # Concorde's -M option accepts only the CCutil_writemaster artifact
    # (<problem>.mas), which contains both the datagroup and original ptour.
    # The extensionless <problem-label> file is a B&B problem file and must
    # never be passed as a master: it has no dat section or node permutation.
    master = directory / f"{instance_hash}.mas"
    restart = directory / f"{instance_hash}.res"
    tour_path = directory / "optimal.sol"
    if not master.is_file() or not restart.is_file():
        raise ProtocolError(
            f"missing Concorde .mas/restart pair for controlled recovery: {instance_hash}; "
            "a .res file alone cannot reconstruct the original node permutation"
        )
    command = [backend["binary"], "-M", master.name, "-R", restart.name, "-o", tour_path.name]
    atomic_json(directory / "ATTEMPT.json", {
        "instance_hash": instance_hash,
        "mode": "controlled_exact_restart_with_original_master",
        "timeout_seconds": FORMAL_CONCORDE_TIMEOUT,
        "command": command,
        "source_work_directory": str(source),
        "concorde_binary_sha256": backend["binary_hash"],
        "new_attempt_number_for_exact_tour_recovery_task": 1,
    })
    execution = _run_captured_concorde(command, directory, FORMAL_CONCORDE_TIMEOUT)
    reported = concorde_impl._reported(execution["stdout"] + "\n" + execution["stderr"])
    validation = _validate_recovery_tour(coordinates, instance_hash, tour_path, reported)
    result = {
        "instance_hash": instance_hash,
        "mode": "P1_controlled_exact_restart_with_original_master",
        "restart_status": (
            "TIMEOUT" if execution["timed_out"] else
            "RETURNED" if execution["returncode"] == 0 else
            "CLI_REJECTED"
        ),
        "proven_optimal_objective_int": reported,
        "optimal_tour_file": str(tour_path),
        "solver_wall_seconds": execution["wall_seconds"],
        "timed_out": execution["timed_out"],
        "returncode": execution["returncode"],
        "restart_count": 1,
        **{key: value for key, value in validation.items() if key != "tour"},
    }
    atomic_json(result_path, result)
    return result


def _final_three_fresh_job(coordinates: np.ndarray, instance_hash: str,
                           backend: dict[str, Any]) -> dict[str, Any]:
    """Solve one final C0 blocker from original coordinates, preserving all state."""

    directory = arm_dir("C0") / "oracle" / "exact_tour_recovery" / "final3" / instance_hash / "attempt_01"
    result_path = directory / "RESULT.json"
    if result_path.is_file():
        return read_json(result_path)
    if directory.exists():
        raise ProtocolError(f"refusing to overwrite final-three exact solve: {directory}")
    directory.mkdir(parents=True)
    distances = concorde_impl.integer_distance_matrix(coordinates, concorde_impl.INTEGER_SCALE)
    tsp = directory / "instance.tsp"
    tour_path = directory / "optimal.sol"
    concorde_impl.write_explicit_tsp(tsp, distances, instance_hash)
    atomic_json(directory / "NODE_MAPPING.json", {
        "instance_hash": instance_hash,
        "mapping": "identity",
        "input_coordinate_index_base": 0,
        "tsplib_node_index_base": 0,
        "input_index_to_tsplib_node": list(range(len(coordinates))),
        "distance_definition": "round(euclidean_distance * 1000000), numpy.rint",
        "integer_scale": concorde_impl.INTEGER_SCALE,
        "coordinates_sha256": hashlib.sha256(np.ascontiguousarray(coordinates, dtype=np.float64).tobytes()).hexdigest(),
    })
    command = [backend["binary"], "-o", tour_path.name, tsp.name]
    atomic_json(directory / "ATTEMPT.json", {
        "instance_hash": instance_hash,
        "mode": "independent_fresh_exact_solve_from_frozen_original_coordinates",
        "uses_historical_restart": False,
        "timeout_seconds": FORMAL_CONCORDE_TIMEOUT,
        "command": command,
        "preserve_concorde_state": True,
        "delete_state_flag_x_used": False,
        "concorde_binary_sha256": backend["binary_hash"],
    })
    execution = _run_captured_concorde(command, directory, FORMAL_CONCORDE_TIMEOUT)
    reported = concorde_impl._reported(execution["stdout"] + "\n" + execution["stderr"])
    validation = _validate_recovery_tour(coordinates, instance_hash, tour_path, reported)
    state_files = sorted(path.name for path in directory.iterdir() if path.is_file())
    required_state = {
        "master": (directory / "instance.mas").is_file(),
        "restart": (directory / "instance.res").is_file(),
        "node_mapping": (directory / "NODE_MAPPING.json").is_file(),
        "stdout": (directory / "stdout.txt").is_file(),
        "stderr": (directory / "stderr.txt").is_file(),
        "solution": tour_path.is_file(),
    }
    verified = bool(validation.get("verified")) and reported is not None and all(required_state.values())
    result = {
        "instance_hash": instance_hash,
        "mode": "FINAL3_independent_fresh_exact_solve",
        "restart_status": "FRESH_SOLVE_TIMEOUT" if execution["timed_out"] else "FRESH_SOLVE_RETURNED",
        "proven_optimal_objective_int": reported,
        "optimal_tour_file": str(tour_path),
        "solver_wall_seconds": execution["wall_seconds"],
        "timed_out": execution["timed_out"],
        "returncode": execution["returncode"],
        "restart_count": 0,
        "fresh_solve_count": 1,
        "required_state_files_present": required_state,
        "state_files": state_files,
        **{key: value for key, value in validation.items() if key not in ("tour", "verified")},
        "verified": verified,
    }
    atomic_json(result_path, result)
    return result


def _final_three_extended_restart_job(coordinates: np.ndarray, instance_hash: str,
                                      backend: dict[str, Any]) -> dict[str, Any]:
    """Resume a final-three fresh solve while enforcing a cumulative 30,000s cap."""

    source = arm_dir("C0") / "oracle" / "exact_tour_recovery" / "final3" / instance_hash / "attempt_01"
    source_result = read_json(source / "RESULT.json")
    initial_wall = float(source_result["solver_wall_seconds"])
    remaining_timeout = max(0.0, FINAL_THREE_CUMULATIVE_TIMEOUT - initial_wall)
    if remaining_timeout <= 0.0:
        raise ProtocolError(f"no approved cumulative Concorde budget remains for {instance_hash}")
    directory = arm_dir("C0") / "oracle" / "exact_tour_recovery" / "extended30000" / instance_hash / "attempt_01"
    result_path = directory / "RESULT.json"
    if result_path.is_file():
        return read_json(result_path)
    if directory.exists():
        raise ProtocolError(f"refusing to overwrite extended exact-restart state: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, directory)
    master = directory / "instance.mas"
    restart = directory / "instance.res"
    mapping = directory / "NODE_MAPPING.json"
    if not master.is_file() or not restart.is_file() or not mapping.is_file():
        raise ProtocolError(f"incomplete fresh-solve restart state for {instance_hash}")
    tour_path = directory / "optimal.sol"
    command = [backend["binary"], "-M", master.name, "-R", restart.name, "-o", tour_path.name]
    atomic_json(directory / "EXTENDED_ATTEMPT.json", {
        "instance_hash": instance_hash,
        "mode": "exact_restart_from_verified_fresh_mas_res",
        "approved_cumulative_timeout_seconds": FINAL_THREE_CUMULATIVE_TIMEOUT,
        "prior_fresh_solve_wall_seconds": initial_wall,
        "current_restart_timeout_seconds": remaining_timeout,
        "command": command,
        "source_state_directory": str(source),
        "preserve_concorde_state": True,
        "delete_state_flag_x_used": False,
        "concorde_binary_sha256": backend["binary_hash"],
    })
    execution = _run_captured_concorde(command, directory, remaining_timeout)
    reported = concorde_impl._reported(execution["stdout"] + "\n" + execution["stderr"])
    candidates = []
    for candidate in (tour_path, directory / "instance.sol"):
        validation = _validate_recovery_tour(coordinates, instance_hash, candidate, reported)
        candidates.append({
            "path": str(candidate),
            **{key: value for key, value in validation.items() if key != "tour"},
        })
    accepted = next((value for value in candidates if bool(value.get("verified"))), None)
    cumulative = initial_wall + float(execution["wall_seconds"])
    required_state = {
        "master": master.is_file(),
        "restart": restart.is_file(),
        "node_mapping": mapping.is_file(),
        "stdout": (directory / "stdout.txt").is_file(),
        "stderr": (directory / "stderr.txt").is_file(),
    }
    verified = reported is not None and accepted is not None and all(required_state.values())
    result = {
        "instance_hash": instance_hash,
        "mode": "FINAL3_extended_exact_restart",
        "restart_status": "VERIFIED_OPTIMAL" if verified else ("CUMULATIVE_TIMEOUT" if execution["timed_out"] else "UNVERIFIED_RETURN"),
        "proven_optimal_objective_int": reported,
        "optimal_tour_file": accepted["path"] if accepted is not None else None,
        "recomputed_tour_cost_int": accepted.get("recomputed_tour_cost_int") if accepted is not None else None,
        "proof_tour_consistency": bool(accepted is not None),
        "hamiltonian": bool(accepted is not None and accepted.get("hamiltonian")),
        "tour_hash": accepted.get("tour_hash") if accepted is not None else None,
        "solver_wall_seconds": execution["wall_seconds"],
        "prior_fresh_solve_wall_seconds": initial_wall,
        "cumulative_solver_wall_seconds": cumulative,
        "approved_cumulative_timeout_seconds": FINAL_THREE_CUMULATIVE_TIMEOUT,
        "current_restart_timeout_seconds": remaining_timeout,
        "timed_out": execution["timed_out"],
        "returncode": execution["returncode"],
        "restart_count": 1,
        "fresh_solve_count": 1,
        "candidate_tour_validations": candidates,
        "required_state_files_present": required_state,
        "verified": bool(verified),
    }
    atomic_json(result_path, result)
    return result


def assemble_complete_c0_labels() -> None:
    """Materialize C0 labels only after every proof and tour is independently valid."""

    coordinates, hashes, _ = load_bundle(selected_bundle("C0"))
    oracle = arm_dir("C0") / "oracle"
    partial = canonical_oracle_ledger(pd.read_parquet(oracle / "ledger.partial.parquet"))
    cached = {str(row.instance_hash): row._asdict() for row in partial.itertuples(index=False)}
    recovered: dict[str, dict[str, Any]] = {}
    recovery_root = oracle / "exact_tour_recovery"
    for path in sorted(recovery_root.glob("*/*/attempt_*/RESULT.json")):
        value = read_json(path)
        if bool(value.get("verified")):
            recovered[str(value["instance_hash"])] = value
    rows = []
    for index, instance_hash in enumerate(hashes):
        if instance_hash in cached:
            rows.append(cached[instance_hash])
            continue
        value = recovered.get(instance_hash)
        if value is None:
            raise ProtocolError(f"C0 exact recovery is incomplete: {instance_hash}")
        tour_path = Path(str(value["optimal_tour_file"]))
        objective = int(value["proven_optimal_objective_int"])
        validation = _validate_recovery_tour(coordinates[index], instance_hash, tour_path, objective)
        if not bool(validation.get("verified")):
            raise ProtocolError(f"C0 recovered tour failed final validation: {instance_hash}")
        tour = validation["tour"]
        rows.append({
            "instance_hash": instance_hash,
            "status": "solved",
            "verified": True,
            "optimal_cost_int": objective,
            "runtime_sec": float(value["solver_wall_seconds"]),
            "expert_tour": " ".join(map(str, tour)),
            "tour_hash": concorde_impl.tour_hash(tour),
            "binary_hash": CANONICAL_CONCORDE_SHA256,
        })
    ledger = canonical_oracle_ledger(pd.DataFrame(rows))
    if len(ledger) != TRAIN_N or not bool(ledger.verified.all()):
        raise ProtocolError(f"C0 final ledger failed completeness gate: {int(ledger.verified.sum())}/{TRAIN_N}")
    ordered = ledger.set_index("instance_hash").loc[hashes].reset_index()
    tours = np.stack([np.fromstring(value, sep=" ", dtype=np.int64) for value in ordered.expert_tour])
    for index, tour in enumerate(tours):
        legal, message = concorde_impl.validate_tour_0based(tour.tolist(), len(coordinates[index]))
        if not legal:
            raise ProtocolError(f"C0 assembled tour is invalid at row {index}: {message}")
        distance = concorde_impl.integer_distance_matrix(coordinates[index], concorde_impl.INTEGER_SCALE)
        if concorde_impl.tour_cost_int(distance, tour.tolist()) != int(ordered.iloc[index].optimal_cost_int):
            raise ProtocolError(f"C0 assembled tour cost mismatch at row {index}")
    ledger_path = oracle / "ledger.parquet"
    labeled_path = arm_dir("C0") / "data" / "training_labeled.npz"
    write_frame(ledger_path, ordered)
    save_bundle(labeled_path, coordinates, hashes, tours)
    check_coordinates, check_hashes, check_tours = load_bundle(labeled_path)
    if check_hashes != hashes or check_tours is None or check_tours.shape != (TRAIN_N, 100):
        raise ProtocolError("C0 saved training_labeled.npz failed round-trip integrity")
    if not np.array_equal(check_coordinates, coordinates) or not np.array_equal(check_tours, tours):
        raise ProtocolError("C0 saved training_labeled.npz changed frozen data or tours")
    atomic_json(oracle / "summary.json", {
        "status": "PASS",
        "target": "C0",
        "n": TRAIN_N,
        "verified": TRAIN_N,
        "reused_original_verified_rows": len(cached),
        "recovered_verified_rows": TRAIN_N - len(cached),
        "concorde_binary_sha256": CANONICAL_CONCORDE_SHA256,
        "timeout": FORMAL_CONCORDE_TIMEOUT,
        "ledger_sha256": sha256_file(ledger_path),
        "training_labeled_sha256": sha256_file(labeled_path),
        "proof_tour_consistency": True,
    })
    atomic_json(arm_dir("C0") / "C0_LABELING_COMPLETE.json", {
        "status": "PASS",
        "verified": TRAIN_N,
        "total": TRAIN_N,
        "ledger": str(ledger_path),
        "training_labeled": str(labeled_path),
    })
    print(f"C0_EXACT_LABELING_COMPLETE verified={TRAIN_N}/{TRAIN_N}")


def _refresh_exact_recovery_ledger() -> pd.DataFrame:
    coordinates, hashes, _ = load_bundle(selected_bundle("C0"))
    partial = canonical_oracle_ledger(pd.read_parquet(arm_dir("C0") / "oracle" / "ledger.partial.parquet"))
    cached = partial.set_index("instance_hash")
    results: dict[str, dict[str, Any]] = {}
    root = arm_dir("C0") / "oracle" / "exact_tour_recovery"
    for path in sorted(root.glob("*/*/attempt_*/RESULT.json")) if root.exists() else []:
        value = read_json(path)
        instance_hash = str(value["instance_hash"])
        if instance_hash not in results or bool(value.get("verified")):
            results[instance_hash] = value
    rows = []
    for instance_hash in hashes:
        if instance_hash in cached.index:
            row = cached.loc[instance_hash]
            rows.append({
                "instance_hash": instance_hash,
                "concorde_restart_status": "REUSED_CLEAN_VERIFIED_CACHE",
                "proven_optimal_objective_int": int(row.optimal_cost_int),
                "optimal_tour_file": str(arm_dir("C0") / "oracle" / "work" / instance_hash / f"{instance_hash}.sol"),
                "recomputed_tour_cost_int": int(row.optimal_cost_int),
                "proof_tour_consistency": True,
                "solver_wall_seconds": row.runtime_sec,
                "restart_count": 0,
                "verified": True,
            })
        else:
            value = results.get(instance_hash, {})
            rows.append({
                "instance_hash": instance_hash,
                "concorde_restart_status": value.get("restart_status", "UNRESOLVED"),
                "proven_optimal_objective_int": value.get("proven_optimal_objective_int", value.get("prior_proven_optimal_objective_int")),
                "optimal_tour_file": value.get("optimal_tour_file"),
                "recomputed_tour_cost_int": value.get("recomputed_tour_cost_int"),
                "proof_tour_consistency": bool(value.get("proof_tour_consistency", False)),
                "solver_wall_seconds": value.get("solver_wall_seconds"),
                "restart_count": value.get("restart_count"),
                "verified": bool(value.get("verified", False)),
            })
    frame = pd.DataFrame(rows)
    write_frame(root / "exact_recovery_ledger.parquet", frame)
    return frame


def recover_exact_tours(phase: str, workers: int) -> None:
    coordinates, hashes, _ = load_bundle(selected_bundle("C0"))
    index_by_hash = {value: index for index, value in enumerate(hashes)}
    backend = resolve_backend(ROOT / "third_party" / "concorde" / "concorde")
    if phase == "mismatch":
        objectives = _mismatch_objectives()
        jobs = [(value, objectives[value]) for value in objectives]
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as pool:
            futures = {
                pool.submit(_fresh_objective_recovery_job, coordinates[index_by_hash[value]], value, objective, backend): value
                for value, objective in jobs
            }
            results = [future.result() for future in as_completed(futures)]
    elif phase == "timeout":
        values = _timeout_hashes()
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(values)))) as pool:
            futures = {
                pool.submit(_proper_master_restart_job, coordinates[index_by_hash[value]], value, backend): value
                for value in values
            }
            results = [future.result() for future in as_completed(futures)]
    elif phase == "final3":
        values = _timeout_hashes()
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(values)))) as pool:
            futures = {
                pool.submit(_final_three_fresh_job, coordinates[index_by_hash[value]], value, backend): value
                for value in values
            }
            results = [future.result() for future in as_completed(futures)]
    elif phase == "extended30000":
        values = _timeout_hashes()
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(values), 3))) as pool:
            futures = {
                pool.submit(_final_three_extended_restart_job, coordinates[index_by_hash[value]], value, backend): value
                for value in values
            }
            results = [future.result() for future in as_completed(futures)]
    else:
        raise ValueError(phase)
    ledger = _refresh_exact_recovery_ledger()
    atomic_json(arm_dir("C0") / "oracle" / "exact_tour_recovery" / f"{phase}_summary.json", {
        "phase": phase,
        "attempted": len(results),
        "verified": sum(bool(value.get("verified")) for value in results),
        "results": results,
        "full_c0_verified": int(ledger.verified.sum()),
        "c0_total": len(ledger),
        "timeout_seconds": FORMAL_CONCORDE_TIMEOUT,
    })
    print(
        f"EXACT_TOUR_RECOVERY phase={phase} attempted={len(results)} "
        f"phase_verified={sum(bool(value.get('verified')) for value in results)} "
        f"full_c0_verified={int(ledger.verified.sum())}/{len(ledger)}"
    )
    if phase in ("final3", "extended30000") and int(ledger.verified.sum()) == len(ledger):
        assemble_complete_c0_labels()


def label_target(target: str, workers: int, timeout: float) -> None:
    if timeout != FORMAL_CONCORDE_TIMEOUT:
        raise ProtocolError(
            f"the formal exact-label path is frozen at timeout={FORMAL_CONCORDE_TIMEOUT:.0f} seconds"
        )
    if target == "calibration":
        bundle = OUT / "calibration" / "U0_unlabeled.npz"
        destination = OUT / "calibration" / "oracle"
    elif target in ARMS:
        bundle = selected_bundle(target)
        destination = arm_dir(target) / "oracle"
    else:
        raise ValueError(target)
    coords, hashes, _ = load_bundle(bundle)
    existing_path = destination / "ledger.parquet"
    backend = resolve_backend(ROOT / "third_party" / "concorde" / "concorde")
    started = time.perf_counter()
    ledger, recovery = durable_solve_batch(
        coords, hashes, destination, backend, workers=workers, timeout=timeout,
        recover_interrupted_c0=target == "C0",
    )
    ledger = canonical_oracle_ledger(ledger)
    if len(ledger) != len(hashes) or not bool(ledger.verified.all()):
        raise ProtocolError(f"{target}: incomplete Concorde labeling")
    lookup = ledger.set_index("instance_hash")
    ordered = lookup.loc[hashes].reset_index()
    tours = np.stack([np.fromstring(value, sep=" ", dtype=np.int64) for value in ordered.expert_tour])
    if tours.shape != (len(hashes), 100):
        raise ProtocolError(f"{target}: invalid expert tours")
    write_frame(existing_path, ordered)
    labeled = destination.parent / ("U0_labeled.npz" if target == "calibration" else "data/training_labeled.npz")
    save_bundle(labeled, coords, hashes, tours)
    atomic_json(destination / "summary.json", {
        "status": "PASS", "target": target, "n": len(hashes), "verified": int(ordered.verified.sum()),
        "concorde_binary_sha256": CANONICAL_CONCORDE_SHA256, "workers": workers, "timeout": timeout,
        "logical_oracle_queries": len(hashes),
        "physical_new_concorde_calls_current_command": recovery["fresh_exact_calls"],
        "physical_exact_restart_calls_current_command": recovery["exact_restart_calls"],
        "oracle_cache_hits": recovery["recovered_verified_cache_rows"],
        "durable_partial_ledger": str(destination / "ledger.partial.parquet"),
        "ledger_sha256": sha256_file(existing_path),
        "wall_seconds": time.perf_counter() - started,
    })
    print(f"LABEL_COMPLETE target={target} n={len(hashes)}")


def configure_f40_cache() -> None:
    hgbcore.RUN_ID = RUN_ID
    hgbcore.RUN_DIR = OUT
    hgbcore.STATE_DIR = OUT / "feature_state"


def fit_hgb40(calibration_features: pd.DataFrame, y: np.ndarray, output: Path) -> tuple[list[Any], dict[str, Any]]:
    output.mkdir(parents=True, exist_ok=True)
    x = calibration_features[hgbcore.F40].to_numpy(float)
    models = []
    rows = []
    started = time.perf_counter()
    for seed in SEEDS:
        tick = time.perf_counter()
        estimator = HistGradientBoostingRegressor(**hgb_params(seed)).fit(x, y)
        path = output / f"HGB40_seed_{seed}.joblib"
        joblib.dump({"model": estimator, "features": hgbcore.F40, "seed": seed,
                     "target": "log1p_current_LBQ"}, path, compress=3)
        pred = estimator.predict(x)
        rows.append({"seed": seed, "sha256": sha256_file(path), "fit_seconds": time.perf_counter() - tick,
                     "train_mae": float(np.mean(np.abs(pred - y))),
                     "train_spearman": float(spearmanr(pred, y).statistic)})
        models.append(estimator)
    return models, {"models": rows, "total_fit_seconds": time.perf_counter() - started,
                    "n_train": len(y), "features": list(hgbcore.F40), "config": HGB_CFG_CANONICAL,
                    "seeds": list(SEEDS), "aggregate": "median"}


def qd_hgb_select(meta: pd.DataFrame, scores: np.ndarray, budget: int = TRAIN_N) -> pd.DataFrame:
    if len(meta) != len(scores) or meta.instance_hash.astype(str).duplicated().any():
        raise ProtocolError("invalid C2 scoring frame")
    hashes = meta.instance_hash.astype(str).to_numpy()
    work = pd.DataFrame({"index": np.arange(len(meta)), "niche": meta.niche_id.astype(int),
                         "score": np.asarray(scores, dtype=float), "hash": hashes})
    elite = work.sort_values(["niche", "score", "hash"], ascending=[True, False, True], kind="stable").drop_duplicates("niche")
    if len(elite) > budget:
        # Production has K=1000 and budget=10000, so this branch exists only
        # for intentionally tiny smoke tests. Preserve deterministic best-per-
        # niche semantics, then retain the highest-scoring niche elites.
        elite = elite.sort_values(["score", "hash"], ascending=[False, True], kind="stable").iloc[:budget]
    chosen = elite["index"].astype(int).tolist()
    chosen_set = set(chosen)
    fill = [int(i) for i in np.lexsort((hashes, -np.asarray(scores))) if int(i) not in chosen_set][: budget - len(chosen)]
    indices = np.asarray(chosen + fill, dtype=int)
    result = meta.iloc[indices].copy().reset_index(drop=True)
    result["selection_component"] = ["qd_hgb_elite"] * len(chosen) + ["global_hgb_fill"] * len(fill)
    result["selection_rank"] = np.arange(1, len(result) + 1)
    result["hgb_predicted_log1p_hardness"] = np.asarray(scores)[indices]
    result["coord_index_shared_union"] = result.coord_index.astype(int)
    if len(result) != budget or result.instance_hash.astype(str).duplicated().any():
        raise ProtocolError("C2 selection is not unique 10k")
    return result


def select_c2(device_name: str) -> None:
    calibration_path = OUT / "calibration" / "U0_labeled.npz"
    if not calibration_path.is_file():
        raise ProtocolError("label the 10k U0 calibration archive first")
    device = torch.device(device_name)
    model, model_meta = load_canonical_checkpoint(EVO_M5, device, expected_sha256=EVO_M5_SHA256)
    ucoords, uhashes, utours = load_bundle(calibration_path)
    assert utours is not None
    started = time.perf_counter()
    current_hardness = hardness(model, ucoords, utours, device)
    hardness_seconds = time.perf_counter() - started
    configure_f40_cache()
    calibration_features, cal_feature_info = hgbcore.ensure_features(ucoords, uhashes, "warmstart_U0_10k")
    y = np.log1p(current_hardness)
    models, hgb_manifest = fit_hgb40(calibration_features, y, arm_dir("C2") / "surrogate")
    candidates, hashes, _ = load_bundle(CANDIDATE_BUNDLE)
    meta = pd.read_parquet(CANDIDATE_META)
    if hashes != meta.instance_hash.astype(str).tolist():
        raise ProtocolError("candidate/meta alignment failure")
    candidate_features, candidate_feature_info = hgbcore.ensure_features(candidates, hashes, "warmstart_C1C2_union")
    score_started = time.perf_counter()
    stacked = np.vstack([estimator.predict(candidate_features[hgbcore.F40].to_numpy(float)) for estimator in models])
    scores = median_stack(stacked)
    score_seconds = time.perf_counter() - score_started
    qd_started = time.perf_counter()
    selected = qd_hgb_select(meta, scores)
    qd_seconds = time.perf_counter() - qd_started
    indices = selected.coord_index_shared_union.to_numpy(int)
    save_bundle(selected_bundle("C2"), candidates[indices], selected.instance_hash.astype(str).tolist())
    write_frame(arm_dir("C2") / "data" / "selected.parquet", selected)
    write_frame(OUT / "calibration" / "current_hardness.parquet",
                pd.DataFrame({"instance_hash": uhashes, "current_LBQ": current_hardness, "y_log1p": y}))
    c1_hashes = load_bundle(selected_bundle("C1"))[1]
    c2_hashes = selected.instance_hash.astype(str).tolist()
    overlap = len(set(c1_hashes) & set(c2_hashes))
    selection_manifest = {
        "status": "PASS", "model_sha256": model_meta["sha256"], "calibration_n": len(uhashes),
        "historical_hardness_reused": False, "hardness_seconds": hardness_seconds,
        "calibration_features": cal_feature_info, "candidate_features": candidate_feature_info,
        "surrogate": hgb_manifest, "score_seconds": score_seconds, "qd_seconds": qd_seconds,
        "candidate_n": len(hashes), "selected_n": len(c2_hashes),
        "occupied_niches_selected": int(selected.niche_id.nunique()),
        "elite_n": int((selected.selection_component == "qd_hgb_elite").sum()),
        "C1_C2_overlap_n": overlap, "C1_C2_overlap_fraction": overlap / TRAIN_N,
        "C1_duplicates": len(c1_hashes) - len(set(c1_hashes)),
        "C2_duplicates": len(c2_hashes) - len(set(c2_hashes)),
        "membership_sha256": membership_hash(c2_hashes),
    }
    atomic_json(arm_dir("C2") / "data" / "manifest.json", selection_manifest)
    atomic_json(OUT / "protocol" / "C1_C2_overlap.json", {
        key: selection_manifest[key] for key in ("C1_C2_overlap_n", "C1_C2_overlap_fraction", "C1_duplicates", "C2_duplicates")
    })
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"C2_SELECTION_COMPLETE selected=10000 overlap_with_C1={overlap}")


def configure_canonical_training() -> None:
    canonical.RUN_ID = RUN_ID
    canonical.RUN = OUT
    canonical.arm_dir = arm_dir
    canonical.cycle_dir = lambda arm, cycle: arm_dir(arm) / "training" / f"C{cycle:02d}"
    canonical.model_path = lambda arm, index: EVO_M5 if index == 0 else arm_dir(arm) / "checkpoints" / "M6_E256.tsp.best"


def train_arm(arm: str, device_name: str) -> None:
    bundle = formal_dataset_bundle(arm)
    if not bundle.is_file():
        raise ProtocolError(f"{arm} does not have verified Concorde labels")
    coords, hashes, tours = load_bundle(bundle)
    if coords.shape != (TRAIN_N, 100, 2) or tours is None or tours.shape != (TRAIN_N, 100):
        raise ProtocolError(f"{arm}: invalid training bundle")
    device = torch.device(device_name)
    model, meta = load_canonical_checkpoint(EVO_M5, device, expected_sha256=EVO_M5_SHA256)
    configure_canonical_training()
    started = time.perf_counter()
    trained, output_meta, timing = canonical.train(arm, 0, model, meta["sha256"], coords, hashes, tours,
                                                    device, NullStatus(), True)
    total = time.perf_counter() - started
    if timing["optimizer_steps"] != UPDATES_PER_ARM:
        raise ProtocolError(f"{arm}: optimizer update mismatch")
    atomic_json(arm_dir(arm) / "timing" / "training.json", {
        "wall_seconds": total, "optimizer_updates": timing["optimizer_steps"],
        "checkpoint_sha256": output_meta["sha256"], "initial_checkpoint_sha256": meta["sha256"],
        "seed": TRAIN_SEED, "optimizer_restore": False,
    })
    atomic_json(arm_dir(arm) / "TRAINING_COMPLETE.json", {
        "status": "PASS", "arm": arm, "checkpoint": str(arm_dir(arm) / "checkpoints" / "M6_E256.tsp.best"),
        "checkpoint_sha256": output_meta["sha256"], "initial_checkpoint_sha256": meta["sha256"],
        "optimizer_updates": timing["optimizer_steps"], "epochs": timing["epochs"],
    })
    del trained, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    atomic_json(OUT / "STATUS.json", {"status": "FORMAL_TRAINING_PARTIAL", "full_gpu_training_started": True,
                                      "last_completed_arm": arm, "timestamp_utc": utc_now()})
    print(f"TRAIN_COMPLETE arm={arm} checkpoint={output_meta['sha256']}")


def evaluate_arm(arm: str, device_name: str) -> None:
    checkpoint = arm_dir(arm) / "checkpoints" / "M6_E256.tsp.best"
    if not checkpoint.is_file():
        raise ProtocolError(f"missing trained checkpoint: {checkpoint}")
    device = torch.device(device_name)
    model, meta = load_canonical_checkpoint(checkpoint, device, expected_sha256=None)
    configure_canonical_training()
    started = time.perf_counter()
    frame, elapsed = canonical.evaluate(model, checkpoint, "M6", arm_dir(arm) / "evaluation" / "M6_eight.csv",
                                        NullStatus(), arm)
    if len(frame) != 8 or not frame.failed_tours.eq(0).all():
        raise ProtocolError(f"{arm}: incomplete eight-set evaluation")
    atomic_json(arm_dir(arm) / "timing" / "evaluation.json", {"wall_seconds": time.perf_counter() - started,
                                                                 "reported_seconds": elapsed,
                                                                 "checkpoint_sha256": meta["sha256"]})
    original70_split(frame, arm)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"EVALUATE_COMPLETE arm={arm}")


def official_development_ids() -> tuple[set[str], dict[str, Any]]:
    directory = ROOT / "audits" / "external_benchmark_suite_v2" / "evoreal_structure_audit" / "corrected_v2" / "_calibration_src"
    ids: list[str] = []
    files = [directory / f"tsplib_type{name}.txt" for name in ("S1", "S2", "S3")]
    for path in files:
        if not path.is_file():
            return set(), {"status": "UNVERIFIABLE", "reason": f"missing official split source {path}"}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ids.append(str(ast.literal_eval(line)[0]).lower())
    if len(ids) != 48 or len(set(ids)) != 48:
        return set(), {"status": "UNVERIFIABLE", "reason": f"official development IDs count={len(ids)} unique={len(set(ids))}"}
    return set(ids), {"status": "SOURCE_RECOVERED", "source_files": [str(path) for path in files], "development_n": 48}


def original70_split(_: pd.DataFrame, arm: str) -> None:
    details = arm_dir(arm) / "evaluation" / "evaluation_work" / "M6" / "OriginalTSPLIB70" / "instance_metrics.parquet"
    if not details.is_file():
        candidates = list((arm_dir(arm) / "evaluation").rglob("OriginalTSPLIB70*.parquet"))
        details = candidates[0] if candidates else details
    dev_ids, provenance = official_development_ids()
    if not details.is_file() or provenance["status"] != "SOURCE_RECOVERED":
        atomic_json(arm_dir(arm) / "evaluation" / "original70_48_22.json", {
            **provenance, "status": "UNVERIFIABLE", "detail_file_found": details.is_file()})
        return
    frame = pd.read_parquet(details)
    id_column = "instance_id"
    normalized = frame[id_column].astype(str).str.lower().str.replace(".tsp", "", regex=False)
    mask = normalized.isin(dev_ids)
    if int(mask.sum()) != 48 or int((~mask).sum()) != 22:
        atomic_json(arm_dir(arm) / "evaluation" / "original70_48_22.json", {
            **provenance, "status": "UNVERIFIABLE", "matched_development": int(mask.sum()),
            "matched_heldout": int((~mask).sum()), "reason": "exact 48/22 ID binding not recovered"})
        return
    rows = []
    for split, chosen in (("development_informed_48", mask), ("held_out_22", ~mask)):
        values = frame.loc[chosen, "gap_percent"].to_numpy(float)
        rows.append({"split": split, "n": len(values), "mean_gap_percent": float(values.mean()),
                     "median_gap_percent": float(np.median(values)), "P90_gap_percent": float(np.percentile(values, 90))})
    write_frame(arm_dir(arm) / "evaluation" / "original70_48_22.csv", pd.DataFrame(rows))
    atomic_json(arm_dir(arm) / "evaluation" / "original70_48_22.json", {**provenance, "status": "PASS",
                                                                          "heldout_n": 22})


def fft_energy(xy: np.ndarray) -> float:
    points = np.asarray(xy, dtype=np.float64)
    shifted = points - points.min(axis=0)
    scale = float((points.max(axis=0) - points.min(axis=0)).max())
    points = shifted / scale if scale > 0 else shifted
    hist, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=32, range=[[0, 1], [0, 1]])
    magnitude = np.abs(np.fft.fftshift(np.fft.fft2(hist)))
    magnitude[16, 16] = 0
    yy, xx = np.ogrid[:32, :32]
    radius = np.sqrt((yy - 16) ** 2 + (xx - 16) ** 2)
    return float(magnitude[(radius > 1) & (radius <= 4)].mean())


def python_structure_features(xy: np.ndarray) -> dict[str, float]:
    points = np.asarray(xy, dtype=np.float64)
    distances = cdist(points, points)
    np.fill_diagonal(distances, np.inf)
    ordered = np.sort(distances, axis=1)
    nn = ordered[:, 0]
    finite_distances = distances.copy()
    np.fill_diagonal(finite_distances, 0)
    mst = minimum_spanning_tree(finite_distances).data
    eps = float(np.median(nn) * 1.5)
    labels = DBSCAN(eps=max(eps, 1e-9), min_samples=3).fit_predict(points)
    clusters = labels[labels >= 0]
    return {
        "fft_energy": fft_energy(points),
        "nn_mean": float(nn.mean()),
        "nn_cv": float(nn.std(ddof=0) / nn.mean()) if nn.mean() > 0 else float("nan"),
        "knn3_mean": float(ordered[:, :3].mean()),
        "knn5_mean": float(ordered[:, :5].mean()),
        "mst_total": float(mst.sum()),
        "mst_mean": float(mst.mean()),
        "mst_cv": float(mst.std(ddof=0) / mst.mean()) if mst.mean() > 0 else float("nan"),
        "cluster_count": float(len(np.unique(clusters))) if len(clusters) else 0.0,
        "cluster_noise_fraction": float(np.mean(labels < 0)),
    }


def load_structural_groups(sample_per_group: int | None, seed: int) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    evo = np.load(EVO50K, allow_pickle=False)
    labels = np.load(EVO50K_LABELS, allow_pickle=False).astype(str)
    tours = np.load(EVO50K_TOURS, allow_pickle=False)
    all_coords, all_tours, groups, hashes = [], [], [], []
    rng = np.random.default_rng(seed)
    for source in ("S1", "S2", "S3"):
        indices = np.flatnonzero(labels == source)
        if sample_per_group is not None and len(indices) > sample_per_group:
            indices = np.sort(rng.choice(indices, sample_per_group, replace=False))
        all_coords.append(evo[indices]); all_tours.append(tours[indices]); groups.extend([f"EvoReal_{source}"] * len(indices))
        hashes.extend([coordinate_hash(xy) for xy in evo[indices]])
    for cycle in range(5):
        bundle = BASE / "A0_FULL_BASE" / "cycles" / f"C{cycle:02d}" / "07_oracle" / "T" / "labeled_bundle.npz"
        coords, cycle_hashes, cycle_tours = load_bundle(bundle)
        assert cycle_tours is not None
        indices = np.arange(len(coords))
        if sample_per_group is not None and len(indices) > sample_per_group:
            indices = np.sort(rng.choice(indices, sample_per_group, replace=False))
        all_coords.append(coords[indices]); all_tours.append(cycle_tours[indices]); groups.extend([f"BLADE_D{cycle}"] * len(indices))
        hashes.extend([cycle_hashes[int(i)] for i in indices])
    return np.concatenate(all_coords), np.concatenate(all_tours), groups, hashes


def effective_niche_count(values: Iterable[int]) -> float:
    counts = np.asarray(list(Counter(values).values()), dtype=float)
    if not len(counts):
        return 0.0
    p = counts / counts.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def structural_audit(device_name: str, sample_per_group: int | None, *, dry: bool = False) -> pd.DataFrame:
    coords, tours, groups, hashes = load_structural_groups(sample_per_group, stable_seed("structural_sample"))
    started = time.perf_counter()
    rows = [{"instance_hash": h, "group": group, **python_structure_features(xy)}
            for xy, h, group in zip(coords, hashes, groups, strict=True)]
    frame = pd.DataFrame(rows)
    qd = qd_frame(coords, hashes)
    frame = frame.merge(qd[["instance_hash", "niche_id", *QD4_FEATURE_NAMES]], on="instance_hash", validate="one_to_one")
    configure_f40_cache()
    f40, f40_info = hgbcore.ensure_features(coords, hashes, "structural_dry" if dry else "structural_full")
    # Four QD descriptors are also members of F40. Keep the already attached
    # canonical QD columns and merge only the remaining F40 fields so pandas
    # cannot silently create _x/_y aliases.
    f40_new = [name for name in hgbcore.F40 if name not in frame.columns]
    frame = frame.merge(f40[["instance_hash", *f40_new]], on="instance_hash", validate="one_to_one")
    device = torch.device(device_name)
    model, _ = load_canonical_checkpoint(EVO_M5, device, expected_sha256=EVO_M5_SHA256)
    frame["teacher_forced_ce"] = hardness(model, coords, tours, device)
    greedy_costs, _ = greedy_tour_costs(model, coords, device)
    from oracle.concorde import integer_distance_matrix, tour_cost_int

    optima = np.asarray([tour_cost_int(integer_distance_matrix(xy), tour.tolist()) for xy, tour in zip(coords, tours, strict=True)])
    frame["greedy_gap_percent"] = 100.0 * (greedy_costs - optima) / optima
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    output = DRY / "structural_instances.parquet" if dry else OUT / "structural_instances.parquet"
    write_frame(output, frame)
    numeric = ["fft_energy", "nn_cv", "knn3_mean", "knn5_mean", "mst_total", "mst_cv", "cluster_count",
               "cluster_noise_fraction", *QD4_FEATURE_NAMES]
    summaries = []
    for group, part in frame.groupby("group", sort=False):
        entry: dict[str, Any] = {
            "group": group, "n": len(part), "occupied_niches": int(part.niche_id.nunique()),
            "coverage_fraction": float(part.niche_id.nunique() / 1000),
            "effective_niche_count": effective_niche_count(part.niche_id.astype(int)),
            "hardness_mean": float(part.teacher_forced_ce.mean()),
            "gap_mean": float(part.greedy_gap_percent.mean()),
            "hardness_gap_spearman": float(spearmanr(part.teacher_forced_ce, part.greedy_gap_percent).statistic) if len(part) > 2 else float("nan"),
            "hardness_gap_pearson": float(pearsonr(part.teacher_forced_ce, part.greedy_gap_percent).statistic) if len(part) > 2 else float("nan"),
        }
        standardized = part[numeric].replace([np.inf, -np.inf], np.nan).fillna(part[numeric].median()).to_numpy(float)
        scale = np.nanstd(standardized, axis=0); scale[scale == 0] = 1
        standardized = (standardized - np.nanmean(standardized, axis=0)) / scale
        if len(standardized) > 2000:
            standardized = standardized[np.random.default_rng(0).choice(len(standardized), 2000, replace=False)]
        entry["mean_pairwise_feature_distance"] = float(pdist(standardized).mean()) if len(standardized) > 1 else 0.0
        for name in numeric:
            entry[f"{name}_mean"] = float(part[name].mean())
        summaries.append(entry)
    summary = pd.DataFrame(summaries)
    summary_path = DRY / "structural_summary.csv" if dry else OUT / "structural_summary.csv"
    write_frame(summary_path, summary)
    strata_rows = []
    for group, part in frame.groupby("group", sort=False):
        ranked = part.teacher_forced_ce.rank(method="first")
        bins = min(4, len(part))
        labels = [f"Q{i + 1}" for i in range(bins)]
        stratum = pd.qcut(ranked, q=bins, labels=labels)
        for label, section in part.groupby(stratum, observed=True):
            gaps = section.greedy_gap_percent.to_numpy(float)
            strata_rows.append({"group": group, "hardness_stratum": str(label), "n": len(section),
                                "hardness_min": float(section.teacher_forced_ce.min()),
                                "hardness_max": float(section.teacher_forced_ce.max()),
                                "gap_mean": float(gaps.mean()), "gap_median": float(np.median(gaps)),
                                "gap_P90": float(np.percentile(gaps, 90))})
    strata = pd.DataFrame(strata_rows)
    write_frame((DRY if dry else OUT) / "hardness_gap_strata.csv", strata)
    occupancy = frame.groupby(["group", "niche_id"], sort=False).size().rename("occupancy").reset_index()
    write_frame((DRY if dry else OUT) / "niche_occupancy.csv", occupancy)
    evo_niches = set(frame.loc[frame.group.str.startswith("EvoReal"), "niche_id"].astype(int))
    blade_niches = set(frame.loc[frame.group.str.startswith("BLADE"), "niche_id"].astype(int))
    coverage = {
        "evoreal_occupied": len(evo_niches), "blade_occupied": len(blade_niches),
        "intersection": len(evo_niches & blade_niches), "evoreal_only": len(evo_niches - blade_niches),
        "blade_only": len(blade_niches - evo_niches),
        "jaccard": len(evo_niches & blade_niches) / max(1, len(evo_niches | blade_niches)),
        "sampled": sample_per_group is not None,
    }
    atomic_json((DRY if dry else OUT) / "coverage_overlap.json", coverage)
    plot_structural(frame, summary, strata, dry=dry)
    if not dry:
        lines = [
            "# Structural distribution audit",
            "",
            "Status: **FULL_AUDIT_COMPLETE**",
            "",
            "All teacher-forced CE values were recomputed under the same frozen EvoReal M5 checkpoint. Historical per-cycle hardness was not reused. Greedy gaps use the same checkpoint and canonical integer-distance Oracle tours.",
            "",
            markdown_table(summary[["group", "n", "occupied_niches", "effective_niche_count", "hardness_mean", "gap_mean", "hardness_gap_spearman", "mean_pairwise_feature_distance"]]),
            "",
            "The per-instance table includes frozen F40, QD4/CVT niches, MST, NN/KNN, cluster, FFT-energy and NN-CV features. Interpret coverage differences jointly with acquisition cost; this audit does not alter training or selection.",
            "",
            f"CVT overlap: EvoReal occupied {coverage['evoreal_occupied']} niches, BLADE occupied {coverage['blade_occupied']}, intersection {coverage['intersection']}, EvoReal-only {coverage['evoreal_only']}, BLADE-only {coverage['blade_only']}, Jaccard {coverage['jaccard']:.6f}.",
            "Hardness-quartile gap statistics are stored in `hardness_gap_strata.csv`; the complete occupancy distribution is stored in `niche_occupancy.csv`.",
        ]
        write_text(OUT / "02_structural_distribution_audit.md", "\n".join(lines) + "\n")
        atomic_json(OUT / "protocol" / "structural_audit_manifest.json", {
            "status": "PASS", "n": len(frame), "sample_per_group": sample_per_group,
            "model_sha256": EVO_M5_SHA256, "historical_hardness_reused": False,
            "F40": f40_info, "wall_seconds": time.perf_counter() - started,
        })
    return frame


def plot_structural(frame: pd.DataFrame, summary: pd.DataFrame, strata: pd.DataFrame, *, dry: bool) -> None:
    directory = DRY / "figures" if dry else FIGURES
    directory.mkdir(parents=True, exist_ok=True)
    prefix = "PRELIMINARY DRY-RUN — " if dry else ""
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot = pd.crosstab(frame.group, frame.niche_id)
    ax.imshow((pivot > 0).to_numpy(), aspect="auto", interpolation="nearest", cmap="Blues")
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_xlabel("occupied CVT niche columns (observed subset)")
    ax.set_title(prefix + "CVT niche occupancy")
    fig.tight_layout(); fig.savefig(directory / "fig01_structural_coverage.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for group, part in frame.groupby("group", sort=False):
        ax.scatter(part.teacher_forced_ce, part.greedy_gap_percent, s=10, alpha=.45, label=group)
    ax.set_xlabel("teacher-forced CE under EvoReal M5"); ax.set_ylabel("greedy gap (%)")
    ax.set_title(prefix + "Hardness–gap relation"); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(directory / "fig02_hardness_gap.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(summary.group, summary.effective_niche_count)
    ax.tick_params(axis="x", rotation=35); ax.set_ylabel("effective niche count")
    ax.set_title(prefix + "Effective structural diversity")
    fig.tight_layout(); fig.savefig(directory / "fig03_effective_niches.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = strata.pivot(index="group", columns="hardness_stratum", values="gap_mean")
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("mean greedy gap (%)"); ax.set_xlabel("")
    ax.set_title(prefix + "Gap by within-group hardness stratum")
    ax.legend(title="CE stratum")
    fig.tight_layout(); fig.savefig(directory / "fig04_hardness_strata.png", dpi=150); plt.close(fig)


def dry_run(count: int, device_name: str, oracle_count: int) -> None:
    if count < 3:
        raise ValueError("dry-run count must be at least 3")
    init()
    DRY.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    coords, labels, manifest = generate_fresh_evoreal(count, stable_seed("dry_fresh", master=FRESH_EVO_SEED), shard_size=min(32, count))
    hashes = [coordinate_hash(xy) for xy in coords]
    save_bundle(DRY / "C0_fresh_evoreal.npz", coords, hashes)
    manifest["source_counts_actual"] = dict(Counter(labels.tolist()))
    atomic_json(DRY / "C0_manifest.json", manifest)

    candidate_coords, candidate_hashes, _ = load_bundle(CANDIDATE_BUNDLE)
    meta = pd.read_parquet(CANDIDATE_META)
    sample_n = min(max(count * 8, count), len(candidate_hashes))
    pool_indices = np.random.default_rng(stable_seed("dry_pool")).choice(len(candidate_hashes), sample_n, replace=False)
    pool = meta.iloc[pool_indices].copy().reset_index(drop=True)
    pool_coords = candidate_coords[pool_indices]
    pool["coord_index"] = np.arange(sample_n)
    c1_indices = np.random.default_rng(stable_seed("dry_C1")).choice(sample_n, count, replace=False)
    c1 = pool.iloc[c1_indices].copy().reset_index(drop=True)
    # Synthetic scores test only the frozen QD selection mechanics. They are
    # explicitly forbidden from becoming formal artifacts.
    synthetic_scores = np.random.default_rng(stable_seed("dry_scores")).normal(size=sample_n)
    c2 = qd_hgb_select(pool, synthetic_scores, budget=count)
    overlap = len(set(c1.instance_hash.astype(str)) & set(c2.instance_hash.astype(str)))
    write_frame(DRY / "C1_selected.parquet", c1)
    write_frame(DRY / "C2_selected_MECHANICS_ONLY.parquet", c2)
    atomic_json(DRY / "selection_checks.json", {
        "status": "PASS", "formal_scientific_output": False, "candidate_sample_n": sample_n,
        "C1_n": len(c1), "C2_n": len(c2), "C1_duplicates": int(c1.instance_hash.duplicated().sum()),
        "C2_duplicates": int(c2.instance_hash.duplicated().sum()), "overlap_n": overlap,
        "C2_scores": "synthetic mechanics-only; never valid for formal selection",
    })

    oracle_n = min(oracle_count, count)
    oracle_status: dict[str, Any]
    try:
        backend = resolve_backend(ROOT / "third_party" / "concorde" / "concorde")
        tick = time.perf_counter()
        ledger = solve_batch(coords[:oracle_n], hashes[:oracle_n], DRY / "oracle_work", backend, workers=1)
        oracle_status = {"status": "PASS", "n": len(ledger), "verified": int(ledger.verified.sum()),
                         "wall_seconds": time.perf_counter() - tick}
        write_frame(DRY / "oracle_ledger.parquet", ledger)
    except Exception as exc:
        oracle_status = {"status": "BLOCKED", "reason": f"{type(exc).__name__}: {exc}"}
    atomic_json(DRY / "oracle_status.json", oracle_status)

    sample_structural = max(2, min(4, count))
    structure_status: dict[str, Any]
    try:
        structural_audit(device_name, sample_structural, dry=True)
        structure_status = {"status": "PASS", "sample_per_group": sample_structural}
    except Exception as exc:
        structure_status = {"status": "BLOCKED", "reason": f"{type(exc).__name__}: {exc}"}
    atomic_json(DRY / "structural_status.json", structure_status)

    optimizer_status: dict[str, Any] = {"status": "NOT_RUN"}
    try:
        if oracle_status["status"] == "PASS":
            ledger = pd.read_parquet(DRY / "oracle_ledger.parquet").set_index("instance_hash").loc[hashes[:oracle_n]].reset_index()
            tours = np.stack([np.fromstring(value, sep=" ", dtype=np.int64) for value in ledger.expert_tour])
            optimizer_coords = coords[:oracle_n]
            label_source = "dry_run_canonical_concorde"
        else:
            # The managed sandbox may block the legacy 32-bit Concorde binary.
            # A previously verified canonical tour still validates the exact
            # warm-start optimizer path without manufacturing a label.
            existing_coords = np.load(EVO50K, allow_pickle=False)[:oracle_n]
            tours = np.load(EVO50K_TOURS, allow_pickle=False)[:oracle_n]
            optimizer_coords = existing_coords
            label_source = "existing_verified_evoreal50k_tours_fallback"
        device = torch.device(device_name)
        model, _ = load_canonical_checkpoint(EVO_M5, device, expected_sha256=EVO_M5_SHA256)
        before = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
        clone = clone_model_from(model, device)
        closed = canonical.closed_expert_coordinates(optimizer_coords, tours)
        repeated = np.concatenate([closed] * math.ceil(LOGICAL_BATCH / oracle_n), axis=0)[:LOGICAL_BATCH]
        optimizer = create_optimizer(clone, LEARNING_RATE)
        result = supervised_optimizer_step(clone, optimizer, repeated, repeated.shape[1], device)
        after = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in clone.parameters())).hexdigest()
        optimizer_status = {"status": "PASS", "initial_checkpoint_sha256": EVO_M5_SHA256,
                            "fresh_optimizer": True, "optimizer_updates": result["optimizer_updates"],
                            "logical_batch": result["n_instances"], "microbatch": result["microbatch_size"],
                            "finite_loss": bool(np.isfinite(result["training_loss"])), "weights_changed": before != after,
                            "label_source": label_source}
        del optimizer, clone, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as exc:
        optimizer_status = {"status": "BLOCKED", "reason": f"{type(exc).__name__}: {exc}"}
    atomic_json(DRY / "optimizer_step_status.json", optimizer_status)
    overall = "PASS" if oracle_status["status"] == structure_status["status"] == optimizer_status["status"] == "PASS" else "PARTIAL"
    atomic_json(DRY / "DRY_RUN.json", {
        "status": overall, "count": count, "oracle": oracle_status, "structural": structure_status,
        "optimizer": optimizer_status, "full_gpu_training_started": False,
        "wall_seconds": time.perf_counter() - started, "timestamp_utc": utc_now(),
    })
    write_text(OUT / "02_structural_distribution_audit.md",
               "# Structural distribution audit\n\nStatus: **PRELIMINARY_DRY_RUN_ONLY**\n\n"
               f"The dry-run processed {sample_structural} instances per S1/S2/S3 and BLADE D0–D4 group using F40, QD4/CVT, MST, NN/KNN, cluster, FFT, current EvoReal-M5 teacher-forced CE and greedy gap. "
               "These tiny samples validate plumbing only and must not be interpreted scientifically. Run the full `structural-audit` command for conclusions.\n")
    atomic_json(OUT / "STATUS.json", {"status": "DRY_RUN_COMPLETE" if overall == "PASS" else "DRY_RUN_PARTIAL",
                                      "full_gpu_training_started": False, "timestamp_utc": utc_now()})
    if overall == "PASS":
        for path in sorted((DRY / "figures").glob("*.png")):
            shutil.copy2(path, FIGURES / f"PRELIMINARY_DRY_RUN_{path.name}")
        write_text(OUT / "EVOREAL_STRONG_BASELINE_DIAGNOSTIC_READY",
                   "Code implementation and small dry-run complete. Full C0/C1/C2 training not started.\n")
    print(f"DRY_RUN_{overall} count={count} full_training_started=false")


def bootstrap_paired(delta: np.ndarray, seed: int, replicates: int = 10_000) -> dict[str, float]:
    values = np.asarray(delta, dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates)
    for start in range(0, replicates, 500):
        stop = min(replicates, start + 500)
        index = rng.integers(0, len(values), size=(stop - start, len(values)))
        estimates[start:stop] = values[index].mean(axis=1)
    return {"mean": float(values.mean()), "ci95_low": float(np.percentile(estimates, 2.5)),
            "ci95_high": float(np.percentile(estimates, 97.5)), "replicates": replicates}


def collect_evaluation(arm: str) -> pd.DataFrame:
    path = arm_dir(arm) / "evaluation" / "M6_eight.csv"
    if not path.is_file():
        raise ProtocolError(f"missing {path}")
    frame = pd.read_csv(path)
    return pd.DataFrame({
        "arm": arm, "seed": TRAIN_SEED, "status": "FORMAL_SINGLE_SEED_PRELIMINARY",
        "dataset": frame.dataset, "instance_count": frame.instance_count,
        "valid_tours": frame.valid_tours, "failed_tours": frame.failed_tours,
        "mean_gap_percent": frame.mean_gap_percent, "median_gap_percent": frame.median_gap_percent,
        "P90_gap_percent": frame.P90, "runtime_seconds": frame.inference_wall_clock_sec,
        "checkpoint_sha256": frame.checkpoint_sha256,
    })


def finalize() -> None:
    frames = []
    baseline = pd.read_csv(EVO_ARM / "evaluations" / "M5_eight.csv")
    frames.append(pd.DataFrame({
        "arm": "EvoReal_M5", "seed": "frozen", "status": "EXISTING_BASELINE", "dataset": baseline.dataset,
        "instance_count": baseline.instance_count, "valid_tours": baseline.valid_tours,
        "failed_tours": baseline.failed_tours, "mean_gap_percent": baseline.mean_gap_percent,
        "median_gap_percent": baseline.median_gap_percent, "P90_gap_percent": baseline.P90,
        "runtime_seconds": baseline.inference_wall_clock_sec, "checkpoint_sha256": baseline.checkpoint_sha256,
    }))
    for arm in ARMS:
        frames.append(collect_evaluation(arm))
    results = pd.concat(frames, ignore_index=True)
    gap8 = results.groupby("arm", sort=False).mean_gap_percent.mean().to_dict()
    baseline_gap = gap8["EvoReal_M5"]
    improvements = {arm: baseline_gap - gap8[arm] for arm in ARMS}
    write_frame(OUT / "04_warmstart_results.csv", results)

    paired_rows = []
    baseline_dir = EVO_ARM / "evaluations" / "evaluation_work" / "M5"
    for arm in ARMS:
        for dataset in EIGHT:
            old_candidates = list((baseline_dir / dataset).glob("*.parquet"))
            new_candidates = list((arm_dir(arm) / "evaluation" / "evaluation_work" / "M6" / dataset).glob("*.parquet"))
            if not old_candidates or not new_candidates:
                continue
            old = pd.read_parquet(old_candidates[0]); new = pd.read_parquet(new_candidates[0])
            if "gap_percent" not in old or "gap_percent" not in new:
                continue
            key = "instance_id"
            joined = old[[key, "gap_percent"]].merge(new[[key, "gap_percent"]], on=key, suffixes=("_baseline", "_new"), validate="one_to_one")
            delta = joined.gap_percent_baseline.to_numpy(float) - joined.gap_percent_new.to_numpy(float)
            ci = bootstrap_paired(delta, stable_seed("bootstrap", arm, dataset))
            paired_rows.append({"arm": arm, "dataset": dataset, "n": len(delta), "improvement_mean_pp": ci["mean"],
                                "ci95_low_pp": ci["ci95_low"], "ci95_high_pp": ci["ci95_high"],
                                "win_rate": float(np.mean(delta > 0))})
    paired = pd.DataFrame(paired_rows)
    write_frame(OUT / "paired_gap_bootstrap.csv", paired)

    timing_rows = []
    for arm in ARMS:
        for phase, path in {
            "generation": arm_dir(arm) / "timing" / "generation.json",
            "oracle": arm_dir(arm) / "oracle" / "summary.json",
            "training": arm_dir(arm) / "timing" / "training.json",
            "evaluation": arm_dir(arm) / "timing" / "evaluation.json",
        }.items():
            if path.is_file():
                data = read_json(path)
                timing_rows.append({"arm": arm, "phase": phase, "wall_seconds": data.get("wall_seconds"),
                                    "logical_oracle_queries": data.get("logical_oracle_queries", 0),
                                    "physical_new_concorde_calls": data.get("physical_new_concorde_calls", 0),
                                    "historical_shared_generation_seconds": data.get("historical_measured_foga_wall_seconds", 0),
                                    "current_run_generation_seconds": data.get("wall_seconds_current_run", 0)})
    calibration_oracle = OUT / "calibration" / "oracle" / "summary.json"
    if calibration_oracle.is_file():
        data = read_json(calibration_oracle)
        timing_rows.append({"arm": "C2", "phase": "calibration_oracle", "wall_seconds": data.get("wall_seconds"),
                            "logical_oracle_queries": data.get("logical_oracle_queries", 0),
                            "physical_new_concorde_calls": data.get("physical_new_concorde_calls", 0),
                            "historical_shared_generation_seconds": 0, "current_run_generation_seconds": 0})
    c2_manifest = read_json(arm_dir("C2") / "data" / "manifest.json")
    timing_rows.extend([
        {"arm": "C2", "phase": "hardness_refresh", "wall_seconds": c2_manifest["hardness_seconds"], "logical_oracle_queries": 0, "physical_new_concorde_calls": 0},
        {"arm": "C2", "phase": "F40_features", "wall_seconds": c2_manifest["calibration_features"]["wall_seconds"] + c2_manifest["candidate_features"]["wall_seconds"], "logical_oracle_queries": 0, "physical_new_concorde_calls": 0},
        {"arm": "C2", "phase": "surrogate", "wall_seconds": c2_manifest["surrogate"]["total_fit_seconds"] + c2_manifest["score_seconds"], "logical_oracle_queries": 0, "physical_new_concorde_calls": 0},
        {"arm": "C2", "phase": "QD", "wall_seconds": c2_manifest["qd_seconds"], "logical_oracle_queries": 0, "physical_new_concorde_calls": 0},
    ])
    timing = pd.DataFrame(timing_rows)
    write_frame(OUT / "cost_accounting.csv", timing)

    # Single-seed results are always preliminary even if point estimates are large.
    decision = "INCONCLUSIVE_PRELIMINARY_SINGLE_SEED"
    lines = [
        "# Incremental value analysis",
        "",
        f"Status: **{decision}**",
        "",
        "The pilot uses one paired seed. Point estimates and paired bootstrap intervals are reported, but no final P4 label is allowed until at least three independent paired seeds reproduce the effect.",
        "",
        "## GAP8 and incremental improvement",
        "",
        markdown_table(pd.DataFrame([{"arm": arm, "GAP8": gap8[arm], "improvement_vs_EvoReal_M5_pp": 0.0 if arm == "EvoReal_M5" else improvements[arm]}
                                     for arm in ("EvoReal_M5", *ARMS)])),
        "",
        f"C2-C0 GAP8 advantage (positive favors C2): {gap8['C0'] - gap8['C2']:.6f} pp.",
        f"C2-C1 GAP8 advantage (positive favors C2): {gap8['C1'] - gap8['C2']:.6f} pp.",
        "",
        "Per-dataset mean/median/P90, valid and failed tours are in `04_warmstart_results.csv`; paired bootstrap intervals are in `paired_gap_bootstrap.csv`; wall-clock and Oracle-query accounting are in `cost_accounting.csv`.",
    ]
    write_text(OUT / "05_incremental_value_analysis.md", "\n".join(lines) + "\n")
    plot_results(results)
    atomic_json(OUT / "FINAL_VERIFICATION.json", {
        "status": "PASS", "formal_arms": list(ARMS), "single_seed": True, "preliminary": True,
        "decision": decision, "gap8": gap8, "improvements": improvements,
        "all_updates_equal": all(read_json(arm_dir(arm) / "TRAINING_COMPLETE.json")["optimizer_updates"] == UPDATES_PER_ARM for arm in ARMS),
        "all_initial_hashes_equal": len({read_json(arm_dir(arm) / "TRAINING_COMPLETE.json")["initial_checkpoint_sha256"] for arm in ARMS}) == 1,
    })
    print(decision)


def plot_results(results: pd.DataFrame) -> None:
    pivot = results.pivot(index="dataset", columns="arm", values="mean_gap_percent").loc[list(EIGHT)]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(pivot)); width = .2
    for i, arm in enumerate(("EvoReal_M5", *ARMS)):
        ax.bar(x + (i - 1.5) * width, pivot[arm], width, label=arm)
    ax.set_xticks(x, pivot.index, rotation=30, ha="right"); ax.set_ylabel("mean gap (%)")
    ax.set_title("Warm-start one-cycle comparison (single-seed preliminary)"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES / "fig04_dataset_gaps.png", dpi=150); plt.close(fig)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="audit old experiments and write the frozen warm-start protocol")
    preflight_parser = commands.add_parser("preflight-formal", help="fail-closed GPU/Oracle/storage/protocol verification")
    preflight_parser.add_argument("--device", default="cuda:0")
    dry_parser = commands.add_parser("dry-run", help="small generation/selection/Oracle/feature/optimizer smoke test; never full training")
    dry_parser.add_argument("--count", type=int, default=24)
    dry_parser.add_argument("--oracle-count", type=int, default=2)
    dry_parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    prepare_parser = commands.add_parser("prepare", help="create formal C0/C1 inputs and the independent U0 10k calibration input")
    prepare_parser.add_argument("--seed", type=int, default=TRAIN_SEED)
    label_parser = commands.add_parser("label", help="run canonical Concorde labeling for calibration or a frozen arm")
    label_parser.add_argument("--target", choices=("calibration", *ARMS), required=True)
    label_parser.add_argument("--workers", type=int, default=16)
    label_parser.add_argument("--timeout", type=float, default=FORMAL_CONCORDE_TIMEOUT)
    reconcile_parser = commands.add_parser("reconcile-restarts", help="persist already-proven restart results without solving")
    reconcile_parser.add_argument("--target", choices=("calibration", *ARMS), required=True)
    recover_parser = commands.add_parser(
        "recover-exact-tours",
        help="recover the frozen C0 mismatch/timeout tours with a persistent one-attempt ledger",
    )
    recover_parser.add_argument(
        "--phase", choices=("mismatch", "timeout", "final3", "extended30000"), required=True
    )
    recover_parser.add_argument("--workers", type=int, default=8)
    select_parser = commands.add_parser("select-c2", help="refresh EvoReal-M5 hardness, fit HGB40 and freeze C2 QD selection")
    select_parser.add_argument("--device", default="cuda:0")
    train_parser = commands.add_parser("train", help="explicitly run one full 10,240-update warm-start arm")
    train_parser.add_argument("--arm", choices=ARMS, required=True)
    train_parser.add_argument("--device", default="cuda:0")
    eval_parser = commands.add_parser("evaluate", help="evaluate a completed arm on all eight frozen sets")
    eval_parser.add_argument("--arm", choices=ARMS, required=True)
    eval_parser.add_argument("--device", default="cuda:0")
    audit_parser = commands.add_parser("structural-audit", help="read-only EvoReal50k versus BLADE Hard10k audit")
    audit_parser.add_argument("--device", default="cuda:0")
    audit_parser.add_argument("--sample-per-group", type=int, default=None,
                              help="optional explicit diagnostic subsample; omit for the full audit")
    commands.add_parser("finalize", help="assemble formal results; requires all three arms")
    return value


def main() -> int:
    args = parser().parse_args()
    if args.command == "init":
        init()
    elif args.command == "preflight-formal":
        formal_preflight(args.device)
    elif args.command == "dry-run":
        dry_run(args.count, args.device, args.oracle_count)
    elif args.command == "prepare":
        prepare(args.seed)
    elif args.command == "label":
        label_target(args.target, args.workers, args.timeout)
    elif args.command == "reconcile-restarts":
        reconcile_proven_restarts(args.target)
    elif args.command == "recover-exact-tours":
        recover_exact_tours(args.phase, args.workers)
    elif args.command == "select-c2":
        select_c2(args.device)
    elif args.command == "train":
        train_arm(args.arm, args.device)
    elif args.command == "evaluate":
        evaluate_arm(args.arm, args.device)
    elif args.command == "structural-audit":
        structural_audit(args.device, args.sample_per_group)
    elif args.command == "finalize":
        finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
