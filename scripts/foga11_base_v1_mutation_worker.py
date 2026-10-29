#!/usr/bin/env python
"""Deterministic CPU worker for the frozen FOGA11_BASE_V1 profile."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from final_run.science import cycle_child_seed, membership_hash  # noqa: E402
from final_run.stage import sha256_file  # noqa: E402
from mutation import OPERATOR_NAMES, apply_operator_with_config  # noqa: E402
from mutation.parameters import load_named_profile  # noqa: E402


PROFILE_NAME = "FOGA11_BASE_V1"
PROFILE_PATH = ROOT / "src" / "mutation" / "foga2019_operators.yaml"


def save_bundle(path: Path, coordinates: np.ndarray, hashes: list[str]) -> None:
    np.savez_compressed(
        path,
        coordinates=np.asarray(coordinates, dtype=np.float64),
        instance_hash=np.asarray(hashes),
    )


def scalar_trace(operator: str, parameters: dict[str, object]) -> dict[str, object]:
    row: dict[str, object] = {
        "implosion_radius": np.nan,
        "implosion_strength": np.nan,
        "axis_affected_fraction": np.nan,
        "axis_target": np.nan,
        "axis_jitter": np.nan,
        "axis_choice": None,
    }
    if operator == "implosion_mutation":
        row["implosion_radius"] = float(parameters["radius"])
        row["implosion_strength"] = float(parameters["strength"])
    elif operator == "axis_projection_mutation":
        row["axis_affected_fraction"] = float(parameters["affected_fraction"])
        row["axis_target"] = float(parameters["target"])
        row["axis_jitter"] = float(parameters["jitter"])
        row["axis_choice"] = str(parameters["axis"])
    return row


def seeded_child_seed(experiment_seed: int, cycle: int, parent_hash: str, operator_id: str) -> int:
    """Preserve the audited S0 stream and define the frozen multi-seed extension."""
    if experiment_seed == 2026:
        return cycle_child_seed(cycle, parent_hash, operator_id)
    material = f"BLADE_FINAL_20CYCLE_V1|{experiment_seed}|C{cycle:02d}|{parent_hash}|{operator_id}|FOGA11".encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def generate(parents: np.ndarray, parent_hashes: list[str], cycle: int,
             experiment_seed: int = 2026) -> tuple[np.ndarray, pd.DataFrame]:
    if len(parents) != 10_000 or len(parent_hashes) != 10_000 or len(set(parent_hashes)) != 10_000:
        raise ValueError("FOGA11_BASE_V1 requires exactly 10,000 unique parents")
    profile = load_named_profile(PROFILE_NAME)
    coordinates: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for parent_index, (parent, parent_hash) in enumerate(zip(parents, parent_hashes, strict=True)):
        for operator_id in OPERATOR_NAMES:
            seed = seeded_child_seed(experiment_seed, cycle, parent_hash, operator_id)
            result = apply_operator_with_config(operator_id, parent, np.random.default_rng(seed), profile)
            accepted = result.proposal_status == "accepted" and result.valid_after_repair
            duplicate = bool(accepted and result.final_output_hash in seen)
            row: dict[str, object] = {
                "parent_index": parent_index,
                "parent_hash": parent_hash,
                "operator_id": operator_id,
                "mutation_profile": PROFILE_NAME,
                "mutation_depth": 1,
                "child_seed": np.uint64(seed),
                "child_hash": result.final_output_hash,
                "raw_child_hash": result.raw_output_hash,
                "n_affected": int(result.n_affected),
                "valid_before_repair": bool(result.valid_before_repair),
                "valid_after_repair": bool(result.valid_after_repair),
                "repair_count": int(result.repair_count),
                "repair_fraction": float(result.repair_fraction),
                "proposal_status": result.proposal_status,
                "failure_reason": result.failure_reason,
                "duplicate": duplicate,
                "cycle": cycle,
                **scalar_trace(operator_id, dict(result.sampled_parameters)),
            }
            row.update({key: float(value) for key, value in result.timing.items()})
            rows.append(row)
            if accepted and not duplicate:
                seen.add(result.final_output_hash)
                coordinates.append(np.asarray(result.final_coordinates, dtype=np.float64))
        if (parent_index + 1) % 1000 == 0:
            print(f"[FOGA11_BASE_V1][C{cycle:02d}] parents {parent_index + 1}/10000", flush=True)
    return np.stack(coordinates), pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parents", type=Path, required=True)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-seed", type=int, default=2026)
    args = parser.parse_args()
    with np.load(args.parents, allow_pickle=False) as payload:
        parents = np.asarray(payload["coordinates"], dtype=np.float64)
        parent_hashes = payload["instance_hash"].astype(str).tolist()
    started = time.perf_counter()
    coordinates, manifest = generate(parents, parent_hashes, args.cycle, args.experiment_seed)
    accepted = manifest.valid_after_repair & manifest.proposal_status.eq("accepted") & ~manifest.duplicate
    offspring_hashes = manifest.loc[accepted, "child_hash"].astype(str).tolist()
    args.output.mkdir(parents=True, exist_ok=True)
    save_bundle(args.output / "offspring_coordinates.npz", coordinates, offspring_hashes)
    manifest.to_parquet(args.output / "mutation_manifest.parquet", index=False)
    operator_counts = manifest.groupby("operator_id", sort=True).size().astype(int).to_dict()
    runtime = {
        "status": "PASS",
        "cycle": args.cycle,
        "experiment_seed": args.experiment_seed,
        "python_executable": sys.executable,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "profile_name": PROFILE_NAME,
        "profile_sha256": sha256_file(PROFILE_PATH),
        "parent_membership_sha256": membership_hash(parent_hashes),
        "offspring_membership_sha256": membership_hash(offspring_hashes),
        "nominal_offspring": len(manifest),
        "valid_unique_offspring": len(offspring_hashes),
        "operator_attempts": operator_counts,
        "mutation_depth": 1,
        "seed_derivation": (
            "historical audited cycle_child_seed" if args.experiment_seed == 2026
            else "first_64_bits_le_sha256(BLADE_FINAL_20CYCLE_V1|experiment_seed|cycle|parent_hash|operator|FOGA11)"
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    if len(manifest) != 110_000 or set(operator_counts.values()) != {10_000}:
        raise RuntimeError(f"mutation budget mismatch: {operator_counts}")
    (args.output / "FOGA11_BASE_V1_RUNTIME.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
