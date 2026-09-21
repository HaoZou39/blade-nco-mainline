#!/usr/bin/env python
"""Canonical CPU worker for byte-stable formal FOGA mutation stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from final_run.science import generate_offspring, membership_hash


def save_bundle(path: Path, coordinates: np.ndarray, hashes: list[str]) -> None:
    np.savez_compressed(
        path,
        coordinates=np.asarray(coordinates, dtype=np.float64),
        instance_hash=np.asarray(hashes),
    )


def verify_c00(manifest: pd.DataFrame) -> None:
    audited = pd.read_parquet(
        ROOT / "audits/uniform_rooted_qd_bootstrap_v1/02_offspring_manifest.parquet",
        columns=["child_hash", "raw_child_hash", "child_seed"],
    )
    checks = (
        len(manifest) == 110_000,
        np.array_equal(manifest.child_hash.astype(str), audited.child_hash.astype(str)),
        np.array_equal(manifest.raw_child_hash.astype(str), audited.raw_child_hash.astype(str)),
        np.array_equal(manifest.child_seed.astype(np.uint64), audited.child_seed.astype(np.uint64)),
    )
    if not all(checks):
        raise RuntimeError(f"canonical worker C00 replay failure: {checks}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parents", type=Path, required=True)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.parents, allow_pickle=False) as payload:
        parents = np.asarray(payload["coordinates"], dtype=np.float64)
        parent_hashes = payload["instance_hash"].astype(str).tolist()
    coordinates, manifest = generate_offspring(parents, parent_hashes, args.cycle)
    if args.cycle == 0:
        verify_c00(manifest)
    accepted = manifest.valid_after_repair & manifest.proposal_status.eq("accepted") & ~manifest.duplicate
    offspring_hashes = manifest.loc[accepted, "child_hash"].astype(str).tolist()
    args.output.mkdir(parents=True, exist_ok=True)
    save_bundle(args.output / "offspring_coordinates.npz", coordinates, offspring_hashes)
    manifest.to_parquet(args.output / "mutation_manifest.parquet", index=False)
    (args.output / "CANONICAL_MUTATION_RUNTIME.json").write_text(json.dumps({
        "status": "PASS",
        "cycle": args.cycle,
        "python_executable": sys.executable,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "parent_membership_sha256": membership_hash(parent_hashes),
        "nominal_offspring": len(manifest),
        "valid_unique_offspring": len(offspring_hashes),
        "c00_audited_hash_replay": "PASS" if args.cycle == 0 else "NOT_APPLICABLE",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
