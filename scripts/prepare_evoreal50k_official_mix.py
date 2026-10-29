#!/usr/bin/env python3
"""Prepare and exactly label the paper-aligned EvoReal50k pool.

This script deliberately stops after data/label preparation. It never imports or
invokes BQ training code. The frozen mixture is the verified official rule-based
Phase One ratio S1:S2:S3 = 19:12:17.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.external_v2 import coordinate_hash, coordinate_set_hash
from final_run.stage import atomic_json, sha256_file, utc_now
from oracle.concorde import (
    canonical_oracle_ledger,
    integer_distance_matrix,
    resolve_backend,
    solve_one,
    tour_cost_int,
    tour_hash,
)

RUN_ID = "BLADE_BASE_UNIFORM_EVOREAL_5C_BENCHMARK_V1"
OUT = ROOT / "runs" / RUN_ID / "evoreal_mixture_protocol_20261022"
DATA = OUT / "data"
ORACLE = OUT / "oracle"
SOURCE_ROOT = ROOT / "third_party" / "evoreal"
OFFICIAL_REPOSITORY = "https://github.com/HenryZhu1029/EvoReal"
OFFICIAL_COMMIT = "919616337105e894e06463734ec1dd71f6451309"
M0 = ROOT / "assets" / "solver" / "M0" / "tsp.best"
M0_SHA256 = "6826d7a71b30d882e4673a26ced162b7391157795931efe230f4d019292bba15"
CONCORDE_SHA256 = "38a647d3f04196231c725e90d23c2c883c0a23899f9bab8a257ae6f777d1dec0"
COUNTS = {"S1": 19_792, "S2": 12_500, "S3": 17_708}
VALIDATION_COUNTS = {"S1": 19, "S2": 12, "S3": 17}
SPLIT_COUNTS = {
    "D0": {"S1": 3958, "S2": 2500, "S3": 3542},
    "D1": {"S1": 3958, "S2": 2500, "S3": 3542},
    "D2": {"S1": 3958, "S2": 2500, "S3": 3542},
    "D3": {"S1": 3959, "S2": 2500, "S3": 3541},
    "D4": {"S1": 3959, "S2": 2500, "S3": 3541},
}
GENERATOR_FILES = {
    "S1": ("gpt_S1.py", "generate_s1_type", "918bf7568ab4bcbe26042d0c8d02a221d4f9b2fc838e2b415992a55f31218fe0"),
    "S2": ("gpt_S2.py", "generate_s2_type", "1129b5b902bc7aeee65c226e97e7180a601c2e35ed8b9810c3b8ead0a18b71f8"),
    "S3": ("gpt_S3.py", "generate_s3_type", "b9eee5d038ca999a30d7aec80be365cacf5ee332264d6fe00e1a1c454d98c364"),
}
MASTER_SEED = 20261022
GEN_SHARD = 500
CONCORDE_TIMEOUT_SECONDS = 7200.0
ORACLE_WORKERS = 32


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_generator(source: str):
    filename, function_name, expected = GENERATOR_FILES[source]
    path = SOURCE_ROOT / filename
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{source} generator hash mismatch: {actual}")
    spec = importlib.util.spec_from_file_location(f"evoreal_official_{source}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)


def derived_seed(*parts: object) -> int:
    material = "|".join([RUN_ID, str(MASTER_SEED), *map(str, parts)]).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def frozen_test_hashes() -> tuple[set[str], set[str], list[dict[str, Any]]]:
    import external_v2_passive_blade_eval as evaluator

    ordered: set[str] = set()
    pointsets: set[str] = set()
    rows: list[dict[str, Any]] = []
    for name in evaluator.EIGHT:
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
            "frozen_dataset_sha256": payload["dataset_sha256"],
        })
    return ordered, pointsets, rows


def write_official_audit() -> None:
    source_rows = []
    for source, (filename, function_name, expected) in GENERATOR_FILES.items():
        source_rows.append({
            "source": source,
            "validation_category_count": VALIDATION_COUNTS[source],
            "official_weight": VALIDATION_COUNTS[source] / 48,
            "evoreal50k_count": COUNTS[source],
            "generator_file": filename,
            "generator_function": function_name,
            "generator_sha256": expected,
        })
    write_csv(
        OUT / "S1_S2_S3_counts.csv",
        source_rows,
        ["source", "validation_category_count", "official_weight", "evoreal50k_count",
         "generator_file", "generator_function", "generator_sha256"],
    )
    text = f"""# Official EvoReal generator-mixture audit

Status: **VERIFIED**

The official repository was inspected at pinned commit `{OFFICIAL_COMMIT}`. In
`POMO-EvoReal-2phase_finetune/TSP/POMO_TSP_p1/train_p1.py`, the Phase One
configuration sets `gen_weights = [19,12,17]` and documents the positional order
as `[S1,S2,S3]` for the rule-based TSPLIB division. The invoked
`evolved_gen/mixed_generator.py` uses the same default and calls
`random.choices(["S1","S2","S3"], weights=gen_weights)`.

The official rule-based validation directories contain exactly 19 S1 files, 12
S2 files, and 17 S3 files (48 total). The alternative spectral partition has
15:16:17, but it is not the active Phase One setting in the pinned official
`train_p1.py`.

Therefore the verified implementation order is **S1:S2:S3 = 19:12:17**. The
17:19:12 sequence is the same three numerals in a different order and is not the
positional order used by the pinned code. For a 50,000-instance frozen pool,
largest-remainder allocation gives S1=19,792, S2=12,500, S3=17,708.

The three local generator files are byte-identical to the pinned official files.
Their hashes are recorded in `S1_S2_S3_counts.csv` and
`evoreal50k_provenance.json`.

No pre-existing EvoReal50k training pool was found in the accessible project,
home data directories, shell history, or mounted data volumes. Frozen 1,000-item
S1/S2/S3 evaluation sets were found and explicitly excluded. This preparation
therefore generates a new deterministic pool from the pinned original generators,
as authorized on 2026-10-22; it does not claim byte identity with an unpublished
or unavailable historical pool.

Phase Two and real-TSPLIB fine-tuning are excluded. BQ training is intentionally
not started by this preparation program. Human review is required after exact
Concorde labeling.
"""
    (OUT / "official_generator_mixture_audit.md").write_text(text, encoding="utf-8")


def generate_source(source: str, forbidden: set[str], forbidden_sets: set[str], global_seen: set[str], global_sets: set[str]) -> np.ndarray:
    target = COUNTS[source]
    target_path = DATA / f"{source}_coordinates.npy"
    record_path = DATA / f"{source}_generation.json"
    if record_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if sha256_file(target_path) != record["sha256"]:
            raise RuntimeError(f"frozen {source} file hash mismatch")
        result = np.load(target_path, allow_pickle=False)
        for xy in result:
            global_seen.add(coordinate_hash(xy))
            global_sets.add(coordinate_set_hash(xy))
        return result

    generator = load_generator(source)
    accepted: list[np.ndarray] = []
    rejected = {"duplicate": 0, "test_overlap": 0, "invalid": 0}
    attempt = 0
    while sum(len(block) for block in accepted) < target:
        remaining = target - sum(len(block) for block in accepted)
        batch_size = min(GEN_SHARD, remaining + 16)
        seed = derived_seed("generation", source, attempt)
        seed_all(seed)
        batch = generator(batch_size=batch_size, problem_size=100).detach().cpu().numpy().astype(np.float32, copy=False)
        keep = []
        for xy in batch:
            if xy.shape != (100, 2) or not np.isfinite(xy).all() or xy.min() < 0 or xy.max() > 1:
                rejected["invalid"] += 1
                continue
            h, hs = coordinate_hash(xy), coordinate_set_hash(xy)
            if h in forbidden or hs in forbidden_sets:
                rejected["test_overlap"] += 1
                continue
            if h in global_seen or hs in global_sets:
                rejected["duplicate"] += 1
                continue
            keep.append(xy)
            global_seen.add(h)
            global_sets.add(hs)
            if sum(len(block) for block in accepted) + len(keep) == target:
                break
        if keep:
            accepted.append(np.stack(keep))
        attempt += 1
    result = np.ascontiguousarray(np.concatenate(accepted)[:target], dtype=np.float32)
    np.save(target_path, result, allow_pickle=False)
    atomic_json(record_path, {
        "source": source,
        "count": target,
        "shape": list(result.shape),
        "dtype": str(result.dtype),
        "master_seed": MASTER_SEED,
        "per_attempt_seed_derivation": "first_32_bits_sha256(run_id|master_seed|generation|source|attempt)",
        "attempts": attempt,
        "rejected": rejected,
        "sha256": sha256_file(target_path),
    })
    return result


def build_pool() -> tuple[np.ndarray, np.ndarray, list[str]]:
    DATA.mkdir(parents=True, exist_ok=True)
    forbidden, forbidden_sets, test_rows = frozen_test_hashes()
    atomic_json(OUT / "frozen_eight_evaluation_sets.json", {"datasets": test_rows})
    global_seen: set[str] = set()
    global_sets: set[str] = set()
    by_source = {source: generate_source(source, forbidden, forbidden_sets, global_seen, global_sets) for source in ("S1", "S2", "S3")}

    source_orders = {}
    for source, values in by_source.items():
        rng = np.random.Generator(np.random.PCG64(derived_seed("source_shuffle", source)))
        source_orders[source] = rng.permutation(len(values))
    offsets = {source: 0 for source in by_source}
    cycle_blocks: list[np.ndarray] = []
    cycle_labels: list[np.ndarray] = []
    cycle_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    for cycle_index, (cycle, composition) in enumerate(SPLIT_COUNTS.items()):
        blocks, labels = [], []
        for source in ("S1", "S2", "S3"):
            begin, count = offsets[source], composition[source]
            indices = source_orders[source][begin:begin + count]
            offsets[source] += count
            blocks.append(by_source[source][indices])
            labels.extend([source] * count)
        coordinates = np.concatenate(blocks)
        labels_array = np.asarray(labels, dtype="U2")
        rng = np.random.Generator(np.random.PCG64(derived_seed("cycle_shuffle", cycle)))
        order = rng.permutation(10_000)
        coordinates, labels_array = coordinates[order], labels_array[order]
        cycle_blocks.append(coordinates)
        cycle_labels.append(labels_array)
        hashes = [coordinate_hash(xy) for xy in coordinates]
        membership_path = DATA / f"{cycle}_membership.txt"
        membership_path.write_text("\n".join(hashes) + "\n", encoding="ascii")
        np.save(DATA / f"{cycle}_indices.npy", np.arange(cycle_index * 10_000, (cycle_index + 1) * 10_000), allow_pickle=False)
        for source in ("S1", "S2", "S3"):
            cycle_rows.append({
                "cycle": cycle,
                "source": source,
                "count": composition[source],
                "fraction": composition[source] / 10_000,
                "target_fraction": VALIDATION_COUNTS[source] / 48,
                "absolute_fraction_error": abs(composition[source] / 10_000 - VALIDATION_COUNTS[source] / 48),
            })
        membership_rows.append({"cycle": cycle, "count": 10_000, "membership_sha256": sha256_file(membership_path)})
    if offsets != COUNTS:
        raise RuntimeError(f"split allocation mismatch: {offsets}")
    pool = np.ascontiguousarray(np.concatenate(cycle_blocks), dtype=np.float32)
    labels = np.concatenate(cycle_labels)
    pool_path = DATA / "evoreal50k_coordinates.npy"
    labels_path = DATA / "evoreal50k_source_labels.npy"
    np.save(pool_path, pool, allow_pickle=False)
    np.save(labels_path, labels, allow_pickle=False)
    write_csv(OUT / "evoreal50k_cycle_composition.csv", cycle_rows,
              ["cycle", "source", "count", "fraction", "target_fraction", "absolute_fraction_error"])
    hashes = [coordinate_hash(xy) for xy in pool]
    if len(set(hashes)) != 50_000 or len({coordinate_set_hash(xy) for xy in pool}) != 50_000:
        raise RuntimeError("final pool is not unique")
    if set(hashes) & forbidden or {coordinate_set_hash(xy) for xy in pool} & forbidden_sets:
        raise RuntimeError("final pool overlaps a frozen evaluation set")
    provenance = {
        "status": "GENERATED_FROZEN_AWAITING_EXPERT_LABELS",
        "run_id": RUN_ID,
        "created_utc": utc_now(),
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "official_partition": "rule_based",
        "official_validation_counts_S1_S2_S3": VALIDATION_COUNTS,
        "official_mixture_ratio_S1_S2_S3": [19, 12, 17],
        "counts": COUNTS,
        "total": 50_000,
        "problem_size": 100,
        "coordinate_range": [float(pool.min()), float(pool.max())],
        "finite_coordinates": bool(np.isfinite(pool).all()),
        "unique_ordered_instances": len(set(hashes)),
        "unique_point_sets": len({coordinate_set_hash(xy) for xy in pool}),
        "exact_train_test_overlap": 0,
        "existing_historical_pool": {"path": None, "status": "NOT_FOUND"},
        "regeneration_authorization": "User instruction dated 2026-10-22: continue; generate proportionally; use longer Concorde timeout; stop before training.",
        "master_seed": MASTER_SEED,
        "generators": {source: {"path": str(SOURCE_ROOT / values[0]), "function": values[1], "sha256": values[2]} for source, values in GENERATOR_FILES.items()},
        "coordinate_file": str(pool_path),
        "coordinate_file_sha256": sha256_file(pool_path),
        "source_label_file": str(labels_path),
        "source_label_file_sha256": sha256_file(labels_path),
        "cycle_memberships": membership_rows,
        "BQ_frozen_training_budget": {
            "canonical_M0_sha256": M0_SHA256,
            "cycles": 5,
            "instances_per_cycle": 10_000,
            "epochs_per_cycle": 256,
            "effective_batch_size": 256,
            "optimizer_updates_per_cycle": 10_240,
            "learning_rate": 3.75e-5,
            "optimizer": "fresh Adam per cycle",
            "phase_two_real_TSPLIB_finetuning": False,
        },
        "training_started": False,
        "human_review_required_before_training": True,
    }
    atomic_json(OUT / "evoreal50k_provenance.json", provenance)
    return pool, labels, hashes


def load_or_build_pool() -> tuple[np.ndarray, np.ndarray, list[str]]:
    record_path = OUT / "evoreal50k_provenance.json"
    if not record_path.is_file():
        return build_pool()
    record = json.loads(record_path.read_text(encoding="utf-8"))
    pool_path = Path(record["coordinate_file"])
    labels_path = Path(record["source_label_file"])
    if sha256_file(pool_path) != record["coordinate_file_sha256"] or sha256_file(labels_path) != record["source_label_file_sha256"]:
        raise RuntimeError("frozen EvoReal50k pool hash mismatch")
    pool = np.load(pool_path, allow_pickle=False)
    labels = np.load(labels_path, allow_pickle=False)
    hashes = [coordinate_hash(xy) for xy in pool]
    if pool.shape != (50_000, 100, 2) or labels.shape != (50_000,):
        raise RuntimeError("frozen pool shape mismatch")
    return pool, labels, hashes


def instance_record_path(instance_hash: str) -> Path:
    return ORACLE / "instance_records" / instance_hash[:2] / f"{instance_hash}.json"


def solve_and_clean(coordinates: np.ndarray, instance_hash: str, backend: dict[str, Any]) -> dict[str, Any]:
    work = ORACLE / "work" / instance_hash
    if work.exists():
        shutil.rmtree(work)
    try:
        return solve_one(coordinates, instance_hash, work, backend, CONCORDE_TIMEOUT_SECONDS)
    finally:
        if work.exists():
            shutil.rmtree(work)


def migrate_certified_shards(expected_hashes: set[str]) -> int:
    migrated = 0
    for path in sorted(ORACLE.glob("labels_*.parquet")):
        certificate = path.with_suffix(".json")
        if not certificate.is_file() or sha256_file(path) != json.loads(certificate.read_text(encoding="utf-8"))["sha256"]:
            raise RuntimeError(f"legacy label shard is not certified: {path}")
        frame = canonical_oracle_ledger(pd.read_parquet(path))
        for row in frame.to_dict(orient="records"):
            instance_hash = str(row["instance_hash"])
            if instance_hash not in expected_hashes:
                raise RuntimeError(f"legacy label does not belong to frozen pool: {instance_hash}")
            record_path = instance_record_path(instance_hash)
            if record_path.is_file():
                if json.loads(record_path.read_text(encoding="utf-8")) != row:
                    raise RuntimeError(f"instance record conflict: {record_path}")
            else:
                atomic_json(record_path, row)
                migrated += 1
    return migrated


def label_pool(pool: np.ndarray, hashes: list[str], backend: dict[str, Any]) -> None:
    ORACLE.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    migrate_certified_shards(set(hashes))
    completed: dict[str, dict[str, Any]] = {}
    pending: list[tuple[int, str]] = []
    for index, instance_hash in enumerate(hashes):
        record_path = instance_record_path(instance_hash)
        if record_path.is_file():
            row = json.loads(record_path.read_text(encoding="utf-8"))
            if row.get("instance_hash") != instance_hash or not row.get("verified") or row.get("status") != "solved" or row.get("binary_hash") != CONCORDE_SHA256:
                raise RuntimeError(f"invalid persisted instance record: {record_path}")
            completed[instance_hash] = row
        else:
            pending.append((index, instance_hash))
    print(f"EVOREAL_CONCORDE_RESUME={len(completed)}/50000", flush=True)
    with ThreadPoolExecutor(max_workers=ORACLE_WORKERS) as executor:
        jobs = {
            executor.submit(solve_and_clean, pool[index], instance_hash, backend): instance_hash
            for index, instance_hash in pending
        }
        for future in as_completed(jobs):
            row = future.result()
            instance_hash = str(row["instance_hash"])
            atomic_json(instance_record_path(instance_hash), row)
            completed[instance_hash] = row
            count = len(completed)
            if count % 100 == 0 or count == 50_000:
                atomic_json(OUT / "preparation_status.json", {
                    "status": "LABELING",
                    "labeled": count,
                    "total": 50_000,
                    "concorde_workers": ORACLE_WORKERS,
                    "concorde_timeout_seconds_per_instance": CONCORDE_TIMEOUT_SECONDS,
                    "training_started": False,
                    "timestamp_utc": utc_now(),
                })
                print(f"EVOREAL_CONCORDE_PROGRESS={count}/50000", flush=True)
    ledger = canonical_oracle_ledger(pd.DataFrame([completed[instance_hash] for instance_hash in hashes]))
    ledger_path = ORACLE / "evoreal50k_expert_ledger.parquet"
    ledger.to_parquet(ledger_path, index=False)
    tours = np.stack([np.fromstring(value, sep=" ", dtype=np.int64) for value in ledger.expert_tour])
    if tours.shape != (50_000, 100) or not np.array_equal(np.sort(tours, axis=1), np.broadcast_to(np.arange(100), tours.shape)):
        raise RuntimeError("expert tours are not Hamiltonian permutations")
    for index, tour in enumerate(tours):
        if tour_cost_int(integer_distance_matrix(pool[index]), tour) != int(ledger.iloc[index].optimal_cost_int):
            raise RuntimeError(f"expert cost mismatch at {index}")
        if tour_hash(tour.tolist()) != ledger.iloc[index].tour_hash:
            raise RuntimeError(f"expert tour hash mismatch at {index}")
    tour_path = DATA / "evoreal50k_expert_tours.npy"
    np.save(tour_path, tours, allow_pickle=False)
    record = json.loads((OUT / "evoreal50k_provenance.json").read_text(encoding="utf-8"))
    record.update({
        "status": "DATA_AND_EXPERT_LABELS_READY_HUMAN_REVIEW_REQUIRED",
        "expert_tour_file": str(tour_path),
        "expert_tour_file_sha256": sha256_file(tour_path),
        "expert_tour_count": 50_000,
        "concorde_binary_sha256": CONCORDE_SHA256,
        "concorde_timeout_seconds_per_instance": CONCORDE_TIMEOUT_SECONDS,
        "concorde_workers": ORACLE_WORKERS,
        "oracle_wall_seconds": time.perf_counter() - started,
        "oracle_solver_sum_seconds": float(ledger.runtime_sec.sum()),
        "oracle_ledger": str(ledger_path),
        "oracle_ledger_sha256": sha256_file(ledger_path),
        "instance_record_count": len(completed),
        "training_started": False,
        "human_review_required_before_training": True,
    })
    atomic_json(OUT / "evoreal50k_provenance.json", record)
    atomic_json(OUT / "preparation_status.json", {
        "status": "READY_FOR_HUMAN_REVIEW_BEFORE_TRAINING",
        "completion_label": "EVOREAL50K_DATA_PREPARATION_COMPLETE",
        "labeled": 50_000,
        "training_started": False,
        "timestamp_utc": utc_now(),
    })
    (OUT / "EVOREAL50K_DATA_PREPARATION_COMPLETE").write_text(
        "READY_FOR_HUMAN_REVIEW_BEFORE_TRAINING\n", encoding="ascii")


def verify_preflight() -> dict[str, Any]:
    if sha256_file(M0) != M0_SHA256:
        raise RuntimeError("canonical M0 hash mismatch")
    backend = resolve_backend()
    if backend["binary_hash"] != CONCORDE_SHA256:
        raise RuntimeError("Concorde hash mismatch")
    manifest = json.loads((SOURCE_ROOT / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest["pinned_commit"] != OFFICIAL_COMMIT:
        raise RuntimeError("EvoReal pinned commit mismatch")
    for source in GENERATOR_FILES:
        load_generator(source)
    if sum(COUNTS.values()) != 50_000:
        raise RuntimeError("mixture total mismatch")
    for cycle, composition in SPLIT_COUNTS.items():
        if sum(composition.values()) != 10_000:
            raise RuntimeError(f"{cycle} total mismatch")
    return backend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock_path = OUT / "PREPARATION.lock"
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(OUT / "preparation_status.json", {
            "status": "RUNNING",
            "pid": os.getpid(),
            "training_started": False,
            "timestamp_utc": utc_now(),
        })
        write_official_audit()
        backend = verify_preflight()
        pool, _labels, hashes = load_or_build_pool()
        if args.generate_only:
            atomic_json(OUT / "preparation_status.json", {
                "status": "GENERATED_AWAITING_EXPERT_LABELS",
                "training_started": False,
                "timestamp_utc": utc_now(),
            })
            return
        label_pool(pool, hashes, backend)


if __name__ == "__main__":
    main()
