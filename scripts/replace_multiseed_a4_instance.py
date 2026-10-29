#!/usr/bin/env python3
"""Replace one explicitly authorized, unsolved A4 EvoReal instance.

This is an audited protocol-deviation utility.  It preserves the source class,
pool index, cycle position, mixture counts, all existing exact labels, and the
official generator implementation.  The first valid candidate from a new
deterministic replacement RNG stream is accepted before any oracle call.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import prepare_evoreal50k_official_mix as evo  # noqa: E402
import run_matched_budget_baselines_v2 as baseline  # noqa: E402
from evaluation.external_v2 import coordinate_hash, coordinate_set_hash  # noqa: E402
from final_run.stage import atomic_json, sha256_file  # noqa: E402


RUN = ROOT / "runs" / "BLADE_FULL_PIPELINE_MULTISEED_V1"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def atomic_lines(path: Path, values: list[str]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text("\n".join(values) + "\n", encoding="ascii")
    os.replace(temporary, path)


def replacement_seed(run_id: str, seed: int, old_hash: str, attempt: int) -> int:
    material = "|".join([
        run_id, str(seed), "user_authorized_single_replacement", old_hash, str(attempt),
    ]).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def replace(seed: int, expected_old_hash: str) -> dict[str, Any]:
    seed_root = RUN / f"seed_{seed}"
    a4 = seed_root / "A4"
    blocker_path = seed_root / "BLOCKER.json"
    blocker = read_json(blocker_path) if blocker_path.is_file() else {}

    generator_pool_path = a4 / "generator/data/evoreal50k_coordinates.npy"
    dataset_pool_path = a4 / "datasets/evoreal50k_coordinates.npy"
    generator_labels_path = a4 / "generator/data/evoreal50k_source_labels.npy"
    dataset_labels_path = a4 / "datasets/evoreal50k_source_labels.npy"
    generator_pool = np.load(generator_pool_path, allow_pickle=False)
    dataset_pool = np.load(dataset_pool_path, allow_pickle=False)
    generator_labels = np.load(generator_labels_path, allow_pickle=False)
    dataset_labels = np.load(dataset_labels_path, allow_pickle=False)
    if not np.array_equal(generator_pool, dataset_pool):
        raise RuntimeError("generator and formal A4 pools differ before replacement")
    if not np.array_equal(generator_labels, dataset_labels):
        raise RuntimeError("generator and formal A4 source labels differ")
    if generator_pool.shape != (50_000, 100, 2) or generator_labels.shape != (50_000,):
        raise RuntimeError("unexpected A4 pool shape")
    matching = [i for i, xy in enumerate(generator_pool) if coordinate_hash(xy) == expected_old_hash]
    if len(matching) != 1:
        raise RuntimeError(f"expected exactly one pool row for replacement hash, found {len(matching)}")
    index = matching[0]
    if (
        str(blocker.get("blocked_instance_hash", expected_old_hash)) == expected_old_hash
        and "blocked_instance_index_zero_based" in blocker
        and int(blocker["blocked_instance_index_zero_based"]) != index
    ):
        raise RuntimeError("recorded blocker index does not match the current pool")
    target_record = a4 / "A4_EVOREAL50K/oracle/instance_records" / expected_old_hash[:2] / f"{expected_old_hash}.json"
    if target_record.is_file():
        raise RuntimeError("refusing to replace an instance that already has a persisted exact label")
    policy = RUN / "protocol/AUTOMATIC_EVOREAL_TIMEOUT_REPLACEMENT_POLICY.json"
    explicitly_blocked = str(blocker.get("blocked_instance_hash")) == expected_old_hash
    if not explicitly_blocked and not policy.is_file():
        raise RuntimeError("replacement lacks a per-instance blocker and the global user authorization policy")
    old_xy = np.asarray(generator_pool[index], dtype=np.float32)
    if coordinate_hash(old_xy) != expected_old_hash:
        raise RuntimeError("blocked hash does not match the indexed coordinates")
    source = str(generator_labels[index])
    if source not in {"S1", "S2", "S3"}:
        raise RuntimeError(f"invalid source label: {source}")

    existing_hashes = {coordinate_hash(xy) for xy in generator_pool}
    existing_sets = {coordinate_set_hash(xy) for xy in generator_pool}
    forbidden, forbidden_sets, _ = baseline.frozen_test_hashes()
    run_id = f"BLADE_FULL_PIPELINE_MULTISEED_V1_seed_{seed}_EVOREAL50K"
    evo.RUN_ID = run_id
    evo.MASTER_SEED = seed
    generator = evo.load_generator(source)
    rejected = {"invalid": 0, "pool_duplicate": 0, "test_overlap": 0}
    new_xy: np.ndarray | None = None
    accepted_seed: int | None = None
    accepted_attempt: int | None = None
    accepted_rank: int | None = None
    for attempt in range(10_000):
        stream_seed = replacement_seed(run_id, seed, expected_old_hash, attempt)
        evo.seed_all(stream_seed)
        batch = generator(batch_size=16, problem_size=100).detach().cpu().numpy().astype(np.float32, copy=False)
        for rank, candidate in enumerate(batch):
            if candidate.shape != (100, 2) or not np.isfinite(candidate).all() or candidate.min() < 0 or candidate.max() > 1:
                rejected["invalid"] += 1
                continue
            ordered = coordinate_hash(candidate)
            pointset = coordinate_set_hash(candidate)
            if ordered in forbidden or pointset in forbidden_sets:
                rejected["test_overlap"] += 1
                continue
            if ordered in existing_hashes or pointset in existing_sets:
                rejected["pool_duplicate"] += 1
                continue
            new_xy = np.ascontiguousarray(candidate, dtype=np.float32)
            accepted_seed = stream_seed
            accepted_attempt = attempt
            accepted_rank = rank
            break
        if new_xy is not None:
            break
    if new_xy is None:
        raise RuntimeError("no valid deterministic replacement candidate found")
    new_hash = coordinate_hash(new_xy)
    new_pointset_hash = coordinate_set_hash(new_xy)

    cycle = index // 10_000
    cycle_position = index % 10_000
    generator_membership = a4 / f"generator/data/D{cycle}_membership.txt"
    dataset_membership = a4 / f"datasets/evoreal_D{cycle}_membership.txt"
    generator_lines = generator_membership.read_text(encoding="ascii").splitlines()
    dataset_lines = dataset_membership.read_text(encoding="ascii").splitlines()
    if generator_lines != dataset_lines or len(generator_lines) != 10_000:
        raise RuntimeError("formal and generator cycle memberships differ")
    if generator_lines[cycle_position] != expected_old_hash:
        raise RuntimeError("cycle membership does not contain the old hash at its fixed position")

    deviation = a4 / "protocol/authorized_instance_replacements" / expected_old_hash
    if deviation.exists():
        raise RuntimeError(f"replacement audit directory already exists: {deviation}")
    deviation.mkdir(parents=True)
    atomic_npy(deviation / "old_coordinates.npy", old_xy)
    atomic_npy(deviation / "new_coordinates.npy", new_xy)

    old_pool_sha = sha256_file(generator_pool_path)
    updated_pool = np.array(generator_pool, copy=True)
    updated_pool[index] = new_xy
    updated_lines = list(generator_lines)
    updated_lines[cycle_position] = new_hash
    atomic_npy(generator_pool_path, updated_pool)
    atomic_npy(dataset_pool_path, updated_pool)
    atomic_lines(generator_membership, updated_lines)
    atomic_lines(dataset_membership, updated_lines)
    new_pool_sha = sha256_file(generator_pool_path)
    if sha256_file(dataset_pool_path) != new_pool_sha:
        raise RuntimeError("formal and generator pools differ after replacement")
    new_membership_sha = sha256_file(generator_membership)
    if sha256_file(dataset_membership) != new_membership_sha:
        raise RuntimeError("formal and generator memberships differ after replacement")

    hashes = [coordinate_hash(xy) for xy in updated_pool]
    pointsets = [coordinate_set_hash(xy) for xy in updated_pool]
    if len(set(hashes)) != 50_000 or len(set(pointsets)) != 50_000:
        raise RuntimeError("replacement violated pool uniqueness")
    if set(hashes) & forbidden or set(pointsets) & forbidden_sets:
        raise RuntimeError("replacement introduced frozen-test overlap")

    replacement_record = {
        "authorization": "User instruction: skip this instance and regenerate one.",
        "authorization_scope": f"single unsolved seed-{seed} A4 instance",
        "scientific_protocol_deviation": True,
        "selection_policy": "first valid candidate before any Concorde call",
        "seed": seed,
        "pool_index_zero_based": index,
        "cycle": f"D{cycle}",
        "cycle_position_zero_based": cycle_position,
        "source_class_preserved": source,
        "old_instance_hash": expected_old_hash,
        "old_pointset_hash": coordinate_set_hash(old_xy),
        "new_instance_hash": new_hash,
        "new_pointset_hash": new_pointset_hash,
        "replacement_rng_derivation": "first_32_bits_sha256(run_id|seed|user_authorized_single_replacement|old_hash|attempt)",
        "replacement_attempt": accepted_attempt,
        "replacement_batch_rank": accepted_rank,
        "replacement_rng_seed": accepted_seed,
        "rejected_before_acceptance": rejected,
        "old_pool_sha256": old_pool_sha,
        "new_pool_sha256": new_pool_sha,
        "updated_cycle_membership_sha256": new_membership_sha,
        "mixture_counts_unchanged": {value: int((generator_labels == value).sum()) for value in ("S1", "S2", "S3")},
        "valid_unique_count": 50_000,
        "unique_point_sets": 50_000,
        "exact_train_test_overlap": 0,
        "existing_exact_labels_reused": 49_999,
        "new_exact_label_status": "PENDING",
        "created_utc": utc_now(),
    }
    atomic_json(deviation / "REPLACEMENT_AUDIT.json", replacement_record)

    provenance_path = a4 / "generator/evoreal50k_provenance.json"
    provenance = read_json(provenance_path)
    provenance["status"] = "USER_AUTHORIZED_SINGLE_REPLACEMENT_AWAITING_EXPERT_LABEL"
    provenance["coordinate_file_sha256"] = new_pool_sha
    provenance["cycle_memberships"][cycle]["membership_sha256"] = new_membership_sha
    provenance.setdefault("authorized_instance_replacements", []).append(replacement_record)
    provenance["updated_utc"] = utc_now()
    atomic_json(provenance_path, provenance)

    manifest_path = a4 / "datasets/evoreal50k_manifest.json"
    manifest = read_json(manifest_path)
    manifest["status"] = "USER_AUTHORIZED_SINGLE_REPLACEMENT_AWAITING_EXPERT_LABEL"
    manifest["coordinate_file_sha256"] = new_pool_sha
    manifest.setdefault("authorized_instance_replacements", []).append(replacement_record)
    manifest["updated_utc"] = utc_now()
    atomic_json(manifest_path, manifest)

    atomic_json(blocker_path, {
        **blocker,
        "status": "USER_AUTHORIZED_INSTANCE_REPLACEMENT_APPLIED",
        "failure": blocker.get("failure", "Concorde timeout under the frozen per-call limit"),
        "blocked_instance_hash": expected_old_hash,
        "blocked_instance_index_zero_based": index,
        "protocol_deviation_authorized": True,
        "old_instance_hash": expected_old_hash,
        "replacement_instance_hash": new_hash,
        "replacement_audit": str(deviation / "REPLACEMENT_AUDIT.json"),
        "valid_exact_labels": 49_999,
        "required_exact_labels": 50_000,
        "replacement_exact_label_status": "PENDING",
        "timestamp_utc": utc_now(),
    })
    print(json.dumps(replacement_record, indent=2), flush=True)
    return replacement_record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--old-hash", required=True)
    args = parser.parse_args()
    replace(args.seed, args.old_hash)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
