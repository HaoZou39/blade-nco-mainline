#!/usr/bin/env python3
"""Incrementally extract the frozen formal F40 representation for P6-G1.

This is a read-only audit with respect to all historical experiment artifacts.
Only the P6-G1 output directory is written. Existing P6-C0 F40 shards are used
as immutable cache hits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
P6B = ROOT / "target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution"
P6C0_CACHE = ROOT / "target_aware_pilot/p6c0_training_value_mechanism_audit/full_f40_cache"
P03 = ROOT / "runs/BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1/A0_FULL_BASE/cycles/C03/06_selection/P_next.npz"
EVOREAL = ROOT / "runs/BLADE_BASE_UNIFORM_EVOREAL_5C_BENCHMARK_V2/datasets/evoreal50k_coordinates.npy"
EVOREAL_FEATURES = ROOT / "target_aware_pilot/p6e0_evoreal_structural_representation_audit/features/EvoReal50k.parquet"
F40_MANIFEST = ROOT / "audits/S3_B2_RF_HGB_feature_label_scaling_v1/09_feature_subset_manifests.json"
SCALER = ROOT / "assets/qd/reference_cloud/formal_raw_scaler.json"
WORKER = ROOT / "scripts/f40_one_shard.py"
R_EXTRACTOR = ROOT / "src/features/extract_salesperson_raw_features.R"
REPLICATES = ("Original", "S1", "S2")
ARMS = ("E0", "E1")
F40 = tuple(json.loads(F40_MANIFEST.read_text())["F40"])


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def stage_sources() -> list[tuple[str, Path, str]]:
    rows: list[tuple[str, Path, str]] = [("P03", P03, "npz")]
    for replicate in REPLICATES:
        rows.append((f"{replicate}_G1_RAW", P6B / replicate / "shared_G1/mutation/offspring_coordinates.npz", "npz"))
        for arm in ARMS:
            rows.append((f"{replicate}_{arm}_G2_RAW", P6B / replicate / arm / "G2/mutation/offspring_coordinates.npz", "npz"))
    rows.append(("EvoReal50k", EVOREAL, "npy"))
    return rows


def load_stage(path: Path, kind: str) -> tuple[np.ndarray, list[str]]:
    if kind == "npz":
        with np.load(path, allow_pickle=False) as z:
            return np.asarray(z["coordinates"], np.float64), z["instance_hash"].astype(str).tolist()
    coordinates = np.load(path, mmap_mode="r")
    hashes = pd.read_parquet(EVOREAL_FEATURES, columns=["instance_hash"]).instance_hash.astype(str).tolist()
    if len(coordinates) != len(hashes):
        raise RuntimeError("EvoReal coordinates/features length mismatch")
    return coordinates, hashes


def scaler_params() -> dict[str, tuple[float, float]]:
    payload = json.loads(SCALER.read_text())
    values = {x["feature_name"]: (float(x["center_median"]), float(x["scale"])) for x in payload["parameters"]}
    missing = [x for x in F40 if x not in values]
    if missing:
        raise RuntimeError(f"F40 absent from scaler: {missing}")
    return values


def shard_token(hashes: list[str]) -> str:
    return hashlib.sha256("\n".join(hashes).encode()).hexdigest()[:20]


def run_job(job: dict[str, Any]) -> str:
    process = subprocess.Popen(
        [sys.executable, str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(pickle.dumps(job), timeout=900)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise RuntimeError(f"F40 shard timeout: {job['out']}")
    if process.returncode:
        raise RuntimeError(f"F40 shard failed: {stderr.decode(errors='replace')[-3000:]}")
    return stdout.decode(errors="replace").strip()


def read_existing_hashes() -> set[str]:
    existing: set[str] = set()
    for cache in (P6C0_CACHE, OUT / "f40_cache"):
        for path in sorted(cache.glob("f40_*.parquet")):
            existing.update(pd.read_parquet(path, columns=["instance_hash"]).instance_hash.astype(str))
    return existing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=250)
    args = parser.parse_args()
    cache = OUT / "f40_cache"
    cache.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    (OUT / "manifests").mkdir(exist_ok=True)

    existing = read_existing_hashes()
    params = scaler_params()
    rscript = shutil.which("Rscript") or "/home/haozou/micromamba/envs/blade-r/bin/Rscript"
    audit_rows: list[dict[str, Any]] = []
    all_started = time.perf_counter()
    print(f"[P6-G1] pre-existing unique F40={len(existing):,}", flush=True)

    for stage, source, kind in stage_sources():
        coordinates, hashes = load_stage(source, kind)
        if len(hashes) != len(set(hashes)):
            raise RuntimeError(f"non-unique stage hashes: {stage}")
        missing_idx = [i for i, h in enumerate(hashes) if h not in existing]
        jobs: list[dict[str, Any]] = []
        for start in range(0, len(missing_idx), args.shard_size):
            idx = missing_idx[start:start + args.shard_size]
            shard_hashes = [hashes[i] for i in idx]
            jobs.append({
                "coordinates": np.asarray(coordinates[idx], np.float64),
                "hashes": shard_hashes,
                "names": list(F40),
                "out": str(cache / f"f40_{shard_token(shard_hashes)}.parquet"),
                "rscript": rscript,
                "r_script": str(R_EXTRACTOR),
                "params": params,
            })
        started = time.perf_counter()
        print(f"[P6-G1] {stage}: n={len(hashes):,} hits={len(hashes)-len(missing_idx):,} missing={len(missing_idx):,} shards={len(jobs):,}", flush=True)
        if jobs:
            with ThreadPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
                futures = [pool.submit(run_job, job) for job in jobs]
                for done, future in enumerate(as_completed(futures), 1):
                    future.result()
                    if done == 1 or done % 50 == 0 or done == len(jobs):
                        elapsed = time.perf_counter() - started
                        print(f"[P6-G1] {stage}: shards {done}/{len(jobs)} elapsed={elapsed:.1f}s", flush=True)
            existing.update(hashes[i] for i in missing_idx)
        audit_rows.append({
            "stage": stage, "source": str(source.relative_to(ROOT)), "n": len(hashes),
            "unique_n": len(set(hashes)), "cache_hits_before": len(hashes) - len(missing_idx),
            "newly_extracted": len(missing_idx), "shards": len(jobs),
            "wall_s": time.perf_counter() - started,
        })
        pd.DataFrame(audit_rows).to_csv(OUT / "manifests/f40_stage_coverage.csv", index=False)

    # A final membership check catches interrupted/corrupt extraction before analysis.
    final_hashes = read_existing_hashes()
    for stage, source, kind in stage_sources():
        _, hashes = load_stage(source, kind)
        missing = set(hashes) - final_hashes
        if missing:
            raise RuntimeError(f"F40 final coverage failure {stage}: {len(missing)} missing")
    result = {
        "status": "PASS", "f40_n": len(F40), "workers": args.workers,
        "shard_size": args.shard_size, "stages": audit_rows,
        "total_wall_s": time.perf_counter() - all_started,
        "formal_preprocessing": "asinh((raw-center_median)/scale)",
        "immutable_cache": str(P6C0_CACHE.relative_to(ROOT)),
    }
    write_json(OUT / "logs/f40_extraction.json", result)
    print(f"[P6-G1] COMPLETE total_wall={result['total_wall_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
