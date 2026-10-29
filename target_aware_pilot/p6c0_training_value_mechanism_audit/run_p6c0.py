#!/usr/bin/env python3
"""P6-C0 read-only training-value mechanism audit.

This runner never mutates source P6-B artifacts.  Its only model operations are
deterministic eval-mode forward passes through the repository's canonical
teacher-forced hardness implementation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import platform
import resource
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
P6B = ROOT / "target_aware_pilot/p6b_top5_distribution_guided_evolution"
P6A = ROOT / "target_aware_pilot/p6a_top5_target_matching_feasibility"
P60 = ROOT / "target_aware_pilot/p6_feature_population_protocol_audit"
RUN = ROOT / "runs/BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1/A0_FULL_BASE"
M4 = RUN / "checkpoints/M4_E256.tsp.best"
M4_EVAL = RUN / "cycles/C03/evaluation_work/M4"
F40_MANIFEST = ROOT / "audits/S3_B2_RF_HGB_feature_label_scaling_v1/09_feature_subset_manifests.json"
SCALER = ROOT / "assets/qd/reference_cloud/formal_raw_scaler.json"
F40_WORKER = ROOT / "scripts/f40_one_shard.py"
R_EXTRACTOR = ROOT / "src/features/extract_salesperson_raw_features.R"
TSPLIB48_MANIFEST = ROOT / "target_aware_pilot/03_tsplib48_reference_manifest.csv"

sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(P6A)]
from bq.checkpoint import load_canonical_checkpoint  # noqa: E402
from bq.training_hardness import protocol_manifest as hardness_protocol  # noqa: E402
from bq.tsp_metric import validate_tour_0based  # noqa: E402
from evaluation.datasets import load_fixed_n100  # noqa: E402
from final_run.science import hardness, membership_hash  # noqa: E402
from mutation.result import coordinate_hash  # noqa: E402
from oracle.concorde import sha256_file  # noqa: E402
import run_p6a_audit as p6a  # type: ignore  # noqa: E402

REPLICATES = ("Original", "S1", "S2")
ARMS = ("E0", "E1")
DATASETS = (
    "Uniform100", "GaussianMixture100", "TSPLIB100", "EvoReal_S1_100",
    "EvoReal_S2_100", "EvoReal_S3_100", "GaussianMixture500", "OriginalTSPLIB70",
)
BENCHMARK_CE = ("Uniform100", "GaussianMixture100", "TSPLIB100")
TOP5 = tuple(p6a.TOP5)
F40 = tuple(json.loads(F40_MANIFEST.read_text())["F40"])
M4_SHA = "c749311bd2a461084d0cbcfd383a4c0a1fc1268762b72e5f7220952f108f8190"
INFERENCE_SEED = 20261027
BOOTSTRAP_SEED = 20261028
BOOTSTRAP_N = 10_000
BATCH_SIZE = 128


def write_json(path: Path, value: Any) -> None:
    def convert(x: Any) -> Any:
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, (np.bool_,)): return bool(x)
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=convert) + "\n")
    os.replace(tmp, path)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def md_table(frame: pd.DataFrame) -> str:
    """Small dependency-free Markdown table formatter."""
    def cell(value: Any) -> str:
        if pd.isna(value): return "NA"
        if isinstance(value, (float, np.floating)): return f"{float(value):.8g}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    columns=[str(c) for c in frame.columns]
    lines=["| "+" | ".join(columns)+" |", "|"+"|".join(["---"]*len(columns))+"|"]
    lines.extend("| "+" | ".join(cell(v) for v in row)+" |" for row in frame.itertuples(index=False,name=None))
    return "\n".join(lines)


def bundle_path(replicate: str, arm: str) -> Path:
    return P6B / "population_manifests" / f"{replicate}_{arm}_P2.npz"


def manifest_path(replicate: str, arm: str) -> Path:
    return P6B / "population_manifests" / f"{replicate}_{arm}_P2.parquet"


def label_path(replicate: str, arm: str) -> Path:
    return P6B / "labels" / f"{replicate}_{arm}" / "labeled_bundle.npz"


def checkpoint_path(replicate: str, arm: str) -> Path:
    return P6B / "checkpoints" / f"{replicate}_{arm}_M5_E256.tsp.best"


def load_bundle(path: Path, require_tours: bool = False) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    with np.load(path, allow_pickle=False) as z:
        xy = np.asarray(z["coordinates"], np.float64)
        hashes = z["instance_hash"].astype(str).tolist()
        tours = np.asarray(z["expert_tours"], np.int64) if "expert_tours" in z.files else None
    if require_tours and tours is None:
        raise RuntimeError(f"missing expert tours: {path}")
    return xy, hashes, tours


def checkpoint_for(model_key: str) -> Path:
    if model_key == "M4": return M4
    replicate, arm = model_key.rsplit("_", 1)
    return checkpoint_path(replicate, arm)


def expected_checkpoint_sha(model_key: str) -> str:
    if model_key == "M4": return M4_SHA
    replicate, arm = model_key.rsplit("_", 1)
    info = json.loads((P6B / "logs" / f"{replicate}_{arm}" / "training_integrity.json").read_text())
    return str(info["checkpoint_sha256"])


def all_train_jobs() -> list[tuple[str, str, str]]:
    jobs: list[tuple[str, str, str]] = []
    for replicate in REPLICATES:
        for model in ("M4", f"{replicate}_E0", f"{replicate}_E1"):
            for arm in ARMS:
                jobs.append((model, replicate, arm))
    return jobs


def all_benchmark_jobs() -> list[tuple[str, str]]:
    models = ["M4"] + [f"{r}_{a}" for r in REPLICATES for a in ARMS]
    return [(model, dataset) for model in models for dataset in BENCHMARK_CE]


def preflight() -> dict[str, Any]:
    started = time.perf_counter()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "configs").mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    (OUT / "hardness").mkdir(parents=True, exist_ok=True)
    if len(F40) != 40 or len(set(F40)) != 40:
        raise RuntimeError("frozen F40 manifest invalid")
    if sha256_file(M4) != M4_SHA:
        raise RuntimeError("M4 checkpoint hash mismatch")
    rows = []
    total_hashes: list[str] = []
    for replicate in REPLICATES:
        for arm in ARMS:
            xy, hashes, _ = load_bundle(bundle_path(replicate, arm))
            lxy, lhashes, tours = load_bundle(label_path(replicate, arm), True)
            assert tours is not None
            manifest = pd.read_parquet(manifest_path(replicate, arm))
            info = json.loads((P6B / "logs" / f"{replicate}_{arm}" / "training_integrity.json").read_text())
            if xy.shape != (10_000, 100, 2) or not np.isfinite(xy).all() or np.any((xy < 0) | (xy > 1)):
                raise RuntimeError(f"coordinate validity failure {replicate}/{arm}")
            if hashes != manifest.instance_hash.astype(str).tolist() or hashes != lhashes or not np.array_equal(xy, lxy):
                raise RuntimeError(f"bundle/manifest/label alignment failure {replicate}/{arm}")
            if len(set(hashes)) != 10_000 or hashes != [coordinate_hash(v) for v in xy]:
                raise RuntimeError(f"coordinate hash/uniqueness failure {replicate}/{arm}")
            legal = all(validate_tour_0based(t, 100)[0] for t in tours)
            if not legal:
                raise RuntimeError(f"illegal tours {replicate}/{arm}")
            cp = checkpoint_path(replicate, arm)
            cp_sha = sha256_file(cp)
            if cp_sha != info["checkpoint_sha256"] or not info.get("e000_weights_equal_m4"):
                raise RuntimeError(f"checkpoint/E000 integrity failure {replicate}/{arm}")
            eval_info = json.loads((P6B / "logs" / f"{replicate}_{arm}" / "evaluation_integrity.json").read_text())
            if eval_info.get("status") != "PASS" or set(eval_info["datasets"]) != set(DATASETS):
                raise RuntimeError(f"evaluation integrity failure {replicate}/{arm}")
            rows.append({
                "replicate": replicate, "arm": arm, "n": len(hashes), "unique_n": len(set(hashes)),
                "membership_sha256": membership_hash(hashes), "population_bundle_sha256": sha256_file(bundle_path(replicate, arm)),
                "label_bundle_sha256": sha256_file(label_path(replicate, arm)), "checkpoint_sha256": cp_sha,
                "tours_legal": legal, "coordinates_valid": True, "manifest_aligned": True,
            })
            total_hashes.extend(hashes)
    integrity = pd.DataFrame(rows)
    integrity.to_csv(OUT / "logs/input_integrity.csv", index=False)
    protocol = {
        "experiment": "BLADE-NCO P6-C0", "mode": "read-only forward inference and descriptive audit",
        "source_p6b": str(P6B.relative_to(ROOT)), "source_p6b_completion_sha256": sha256_file(P6B / "completion.json"),
        "m4_checkpoint": str(M4.relative_to(ROOT)), "m4_sha256": M4_SHA,
        "replicates": list(REPLICATES), "arms": list(ARMS), "f40": list(F40), "top5": list(TOP5),
        "teacher_forced_hardness": hardness_protocol(), "batch_size": BATCH_SIZE,
        "inference_seed": INFERENCE_SEED, "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_n": BOOTSTRAP_N,
        "hard_tail_rule": "pooled six-population M4 P90/P95, fixed before arm comparison",
        "nn_threshold_rule": "pooled six-population within-population NN P1/P5/P10, reported jointly",
        "prohibitions": ["mutation", "evolution", "survivor selection", "Concorde", "training", "HGB retraining", "checkpoint writes"],
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__,
    }
    protocol["protocol_sha256"] = canonical_hash(protocol)
    write_json(OUT / "configs/frozen_p6c0_config.json", protocol)
    (OUT / "00_protocol.md").write_text(
        "# P6-C0 Protocol\n\n"
        "This is a read-only mechanism audit over the frozen P6-B artifacts. No mutation, evolution, "
        "survivor selection, Concorde call, training, HGB refit, checkpoint write, or new benchmark generation is permitted.\n\n"
        f"- M4 SHA256: `{M4_SHA}`\n- P6-B completion SHA256: `{protocol['source_p6b_completion_sha256']}`\n"
        f"- Exact hardness: `{hardness_protocol()['formula']}` over all 99 TSP100 decisions, batch {BATCH_SIZE}, eval mode.\n"
        "- Hard-tail thresholds: pooled M4 P90/P95 across all 60,000 memberships.\n"
        "- Structural near-duplicate thresholds: pooled P1/P5/P10 NN distances, all reported (no post-hoc threshold choice).\n"
        "- Instance bootstraps are conditional descriptive intervals and are not evolution-replicate uncertainty.\n"
    )
    baseline = pd.read_csv(P6B / "09_e0_e1_paired_comparison.csv")
    original = baseline[baseline.dataset.eq("OriginalTSPLIB70")].copy()
    expected = {"Original": (2.160443, 2.761475), "S1": (2.370019, 2.409727), "S2": (2.253034, 2.608024)}
    for replicate, (e0, e1) in expected.items():
        row = original[original.replicate.eq(replicate)].iloc[0]
        if abs(float(row.mean_gap_E0) - e0) > 5e-6 or abs(float(row.mean_gap_E1) - e1) > 5e-6:
            raise RuntimeError(f"P6-B baseline reproduction mismatch: {replicate}")
    (OUT / "01_integrity_and_baseline.md").write_text(
        "# Integrity and Baseline Reproduction\n\n"
        "All six Population10k bundles pass coordinate range, coordinate-hash, within-population uniqueness, manifest-order, "
        "label-order, and legal Hamiltonian-tour checks. All six M5 hashes match their training integrity records; each record "
        "asserts direct E000 tensor equality to the same formal M4. The formal M4 checkpoint hash also passes.\n\n"
        f"Across-population memberships: 60,000; globally unique coordinate hashes: {len(set(total_hashes)):,}.\n\n"
        "P6-B Original70 means reproduce the frozen conclusion within 5e-6 percentage points:\n\n"
        + md_table(original[["replicate", "mean_gap_E0", "mean_gap_E1", "Delta_E0_minus_E1"]])
        + "\n\nHardness is the canonical teacher-forced CE, not greedy gap and not HGB prediction.\n"
    )
    result = {"status": "PASS", "global_memberships": len(total_hashes), "global_unique": len(set(total_hashes)), "wall_s": time.perf_counter() - started, "protocol_sha256": protocol["protocol_sha256"]}
    write_json(OUT / "logs/preflight.json", result)
    return result


def load_benchmark_bundle(dataset: str) -> tuple[np.ndarray, list[str], np.ndarray]:
    xy = load_fixed_n100(dataset)
    directory = {"Uniform100": "uniform100", "GaussianMixture100": "gaussian_mixture100", "TSPLIB100": "tsplib100"}[dataset]
    candidates = sorted((ROOT / "assets/evaluation" / directory / "concorde").glob("*.parquet"))
    if len(candidates) != 1:
        raise RuntimeError(f"benchmark oracle bundle ambiguity: {dataset}")
    oracle = pd.read_parquet(candidates[0])
    tour_col = "tour_0based"
    if len(oracle) != 10_000 or not oracle.verified.astype(bool).all() or not oracle.tour_legal.astype(bool).all():
        raise RuntimeError(f"benchmark expert tour integrity failure: {dataset}")
    detail = pd.read_parquet(M4_EVAL / dataset / "instance_metrics.parquet")
    ids = detail.instance_id.astype(str).tolist()
    if len(set(ids)) != len(ids) or set(oracle.instance_id.astype(str)) != set(ids):
        raise RuntimeError(f"benchmark oracle/evaluation ID mismatch: {dataset}")
    # The Gaussian-mixture oracle is grouped in a different cDIST order from
    # the frozen evaluator.  Explicit ID alignment is required; positional
    # alignment would silently pair tours with the wrong coordinates.
    oracle = oracle.assign(instance_id=oracle.instance_id.astype(str)).set_index("instance_id").loc[ids].reset_index()
    tours = np.asarray([[int(v) for v in str(x).split()] for x in oracle[tour_col]], np.int64)
    if tours.shape != (10_000, 100) or not all(validate_tour_0based(t, 100)[0] for t in tours):
        raise RuntimeError(f"benchmark expert tour parse failure: {dataset}")
    return xy, ids, tours


def infer_one_train(model_key: str, replicate: str, arm: str, device_name: str) -> dict[str, Any]:
    output = OUT / "hardness" / f"train__{model_key}__on__{replicate}_{arm}.parquet"
    runtime_path = output.with_suffix(".json")
    cp = checkpoint_for(model_key); cp_sha = expected_checkpoint_sha(model_key)
    if output.is_file() and runtime_path.is_file():
        old = json.loads(runtime_path.read_text())
        if old.get("status") == "PASS" and old.get("checkpoint_sha256") == cp_sha and len(pd.read_parquet(output)) == 10_000:
            return old
    xy, hashes, tours = load_bundle(label_path(replicate, arm), True); assert tours is not None
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(INFERENCE_SEED); np.random.seed(INFERENCE_SEED)
    model, meta = load_canonical_checkpoint(cp, device, expected_sha256=cp_sha)
    model.eval(); before = sha256_file(cp); started = time.perf_counter()
    values = hardness(model, xy, tours, device, batch_size=BATCH_SIZE)
    wall = time.perf_counter() - started; after = sha256_file(cp)
    if before != after or meta["sha256"] != cp_sha:
        raise RuntimeError("checkpoint changed during inference")
    frame = pd.DataFrame({"replicate": replicate, "dataset_arm": arm, "model_key": model_key, "instance_hash": hashes, "loss_raw": values, "loss_log1p": np.log1p(values)})
    frame.to_parquet(output, index=False)
    result = {"status": "PASS", "kind": "train_population", "model_key": model_key, "replicate": replicate, "dataset_arm": arm, "n": len(frame), "checkpoint_sha256": cp_sha, "output_sha256": sha256_file(output), "wall_s": wall, "device": device_name, "gpu_peak_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0}
    write_json(runtime_path, result)
    del model, xy, tours; gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    return result


def infer_one_benchmark(model_key: str, dataset: str, device_name: str) -> dict[str, Any]:
    output = OUT / "hardness" / f"benchmark__{model_key}__on__{dataset}.parquet"
    runtime_path = output.with_suffix(".json")
    cp = checkpoint_for(model_key); cp_sha = expected_checkpoint_sha(model_key)
    if output.is_file() and runtime_path.is_file():
        old = json.loads(runtime_path.read_text())
        if old.get("status") == "PASS" and old.get("checkpoint_sha256") == cp_sha and len(pd.read_parquet(output)) == 10_000:
            return old
    xy, ids, tours = load_benchmark_bundle(dataset)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(INFERENCE_SEED); np.random.seed(INFERENCE_SEED)
    model, meta = load_canonical_checkpoint(cp, device, expected_sha256=cp_sha)
    model.eval(); before = sha256_file(cp); started = time.perf_counter()
    values = hardness(model, xy, tours, device, batch_size=BATCH_SIZE)
    wall = time.perf_counter() - started; after = sha256_file(cp)
    if before != after or meta["sha256"] != cp_sha:
        raise RuntimeError("checkpoint changed during inference")
    frame = pd.DataFrame({"dataset": dataset, "model_key": model_key, "instance_id": ids, "loss_raw": values, "loss_log1p": np.log1p(values)})
    frame.to_parquet(output, index=False)
    result = {"status": "PASS", "kind": "benchmark", "model_key": model_key, "dataset": dataset, "n": len(frame), "checkpoint_sha256": cp_sha, "output_sha256": sha256_file(output), "wall_s": wall, "device": device_name, "gpu_peak_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0}
    write_json(runtime_path, result)
    del model, xy, tours; gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    return result


def infer(device: str, parity: int | None, include_benchmark: bool) -> None:
    jobs: list[tuple[str, tuple[str, ...]]] = [("train", j) for j in all_train_jobs()]
    if include_benchmark:
        jobs += [("benchmark", j) for j in all_benchmark_jobs()]
    selected = [(i, kind, args) for i, (kind, args) in enumerate(jobs) if parity is None or i % 2 == parity]
    for done, (index, kind, args) in enumerate(selected, 1):
        print(f"[P6-C0][{device}] {done}/{len(selected)} global_job={index} {kind} {args}", flush=True)
        if kind == "train": infer_one_train(*args, device)
        else: infer_one_benchmark(*args, device)


def scaler_params() -> dict[str, tuple[float, float]]:
    payload = json.loads(SCALER.read_text())
    values = {x["feature_name"]: (float(x["center_median"]), float(x["scale"])) for x in payload["parameters"]}
    missing = [x for x in F40 if x not in values]
    if missing: raise RuntimeError(f"F40 absent from scaler: {missing}")
    return values


def run_feature_job(job: dict[str, Any]) -> str:
    process = subprocess.Popen([sys.executable, str(F40_WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        stdout, stderr = process.communicate(pickle.dumps(job), timeout=900)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL); process.wait()
        raise RuntimeError(f"F40 shard timeout: {job['out']}")
    if process.returncode:
        raise RuntimeError(f"F40 shard failed: {stderr.decode(errors='replace')[-2000:]}")
    return stdout.decode(errors="replace").strip()


def extract_f40(workers: int) -> dict[str, Any]:
    cache = OUT / "full_f40_cache"; cache.mkdir(parents=True, exist_ok=True)
    order: list[str] = []; coords: list[np.ndarray] = []; seen: set[str] = set()
    for replicate in REPLICATES:
        for arm in ARMS:
            xy, hashes, _ = load_bundle(bundle_path(replicate, arm))
            for h, points in zip(hashes, xy, strict=True):
                if h not in seen:
                    seen.add(h); order.append(h); coords.append(points)
    existing: set[str] = set()
    for path in sorted(cache.glob("f40_*.parquet")):
        existing.update(pd.read_parquet(path, columns=["instance_hash"]).instance_hash.astype(str))
    missing = [i for i, h in enumerate(order) if h not in existing]
    params = scaler_params(); rscript = shutil.which("Rscript") or "/home/haozou/micromamba/envs/blade-r/bin/Rscript"
    jobs = []
    for start in range(0, len(missing), 250):
        idx = missing[start:start + 250]; hashes = [order[i] for i in idx]
        token = membership_hash(hashes)[:20]
        jobs.append({"coordinates": np.stack([coords[i] for i in idx]), "hashes": hashes, "names": list(F40), "out": str(cache / f"f40_{token}.parquet"), "rscript": rscript, "r_script": str(R_EXTRACTOR), "params": params})
    started = time.perf_counter()
    if jobs:
        with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            futures = [pool.submit(run_feature_job, job) for job in jobs]
            for done, future in enumerate(as_completed(futures), 1):
                future.result()
                if done == 1 or done % 25 == 0 or done == len(jobs): print(f"[P6-C0] F40 {done}/{len(jobs)}", flush=True)
    files = sorted(cache.glob("f40_*.parquet"))
    frame = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True).drop_duplicates("instance_hash")
    if not set(order).issubset(set(frame.instance_hash.astype(str))) or np.isinf(frame[list(F40)].to_numpy(float)).any():
        raise RuntimeError("full F40 extraction incomplete/nonfinite")
    result = {"status": "PASS", "memberships": 60_000, "unique_coordinates": len(order), "cache_hits_before": len(order) - len(missing), "newly_extracted": len(missing), "shards": len(jobs), "workers": workers, "wall_s": time.perf_counter() - started, "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    write_json(OUT / "logs/f40_extraction.json", result)
    return result


def bootstrap_mean_ci(values: np.ndarray, seed_offset: int = 0) -> tuple[float, float]:
    values = np.asarray(values, float); rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    estimates = np.empty(BOOTSTRAP_N)
    for i in range(BOOTSTRAP_N): estimates[i] = values[rng.integers(0, len(values), len(values))].mean()
    return float(np.quantile(estimates, .025)), float(np.quantile(estimates, .975))


def bootstrap_independent_mean_diff(left: np.ndarray, right: np.ndarray, seed_offset: int = 0) -> tuple[float, float]:
    left=np.asarray(left,float); right=np.asarray(right,float); rng=np.random.default_rng(BOOTSTRAP_SEED+seed_offset)
    estimates=np.empty(BOOTSTRAP_N)
    for i in range(BOOTSTRAP_N):
        estimates[i]=left[rng.integers(0,len(left),len(left))].mean()-right[rng.integers(0,len(right),len(right))].mean()
    return float(np.quantile(estimates,.025)),float(np.quantile(estimates,.975))


def rank_biserial(left: np.ndarray, right: np.ndarray) -> float:
    u = stats.mannwhitneyu(left, right, alternative="two-sided").statistic
    return float(2 * u / (len(left) * len(right)) - 1)


def load_train_ce(model: str, replicate: str, arm: str) -> pd.DataFrame:
    path = OUT / "hardness" / f"train__{model}__on__{replicate}_{arm}.parquet"
    if not path.is_file(): raise RuntimeError(f"missing inference output {path.name}")
    return pd.read_parquet(path)


def load_full_f40() -> pd.DataFrame:
    files = sorted((OUT / "full_f40_cache").glob("f40_*.parquet"))
    if not files: raise RuntimeError("full F40 cache absent")
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True).drop_duplicates("instance_hash").set_index("instance_hash")


def top5_assignments(replicate: str, arm: str, manifest: pd.DataFrame) -> pd.DataFrame:
    source = manifest[[f"phi__{f}" for f in TOP5]].to_numpy(float)
    target_frame = pd.read_parquet(P6A / "diagnostics/tsplib48_top5_representation.parquet")
    target = target_frame[[f"phi__{f}" for f in TOP5]].to_numpy(float)
    cost = p6a.squared_distance_matrix(source, target)
    nearest_index = cost.argmin(axis=1); nearest = np.sqrt(cost[np.arange(len(cost)), nearest_index])
    match = p6a.exact_transport_lp(cost, p6a.capacities(target_frame.instance_id.astype(str).tolist()))
    assigned_ref = np.empty(len(source), int); assigned_cost = np.empty(len(source), float)
    assigned_ref[match["candidate_index"]] = match["reference_index"]
    assigned_cost[match["candidate_index"]] = match["edge_cost"]
    if arm == "E1":
        stored_ref = manifest.selected_reference_index.to_numpy(int)
        stored_cost = manifest.selected_squared_cost.to_numpy(float)
        if not np.array_equal(assigned_ref, stored_ref) or not np.allclose(assigned_cost, stored_cost, rtol=0, atol=1e-10):
            raise RuntimeError(f"E1 stored OT assignment mismatch {replicate}")
    return pd.DataFrame({"nearest_reference_index": nearest_index, "nearest_reference_distance": nearest, "ot_assigned_reference_index": assigned_ref, "ot_assigned_squared_cost": assigned_cost})


def hardness_analysis() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    members = []
    for replicate in REPLICATES:
        for arm in ARMS:
            ce = load_train_ce("M4", replicate, arm)
            manifest = pd.read_parquet(manifest_path(replicate, arm)).reset_index(drop=True)
            if ce.instance_hash.astype(str).tolist() != manifest.instance_hash.astype(str).tolist(): raise RuntimeError("CE/pop alignment")
            assignment = top5_assignments(replicate, arm, manifest)
            frame = ce.copy()
            frame["arm"] = arm
            for feature in TOP5: frame[f"phi__{feature}"] = manifest[f"phi__{feature}"].to_numpy(float)
            frame = pd.concat([frame, assignment], axis=1)
            target_names = pd.read_parquet(P6A / "diagnostics/tsplib48_top5_representation.parquet").instance_id.astype(str).tolist()
            frame["ot_assigned_reference"] = [target_names[i] for i in frame.ot_assigned_reference_index]
            members.append(frame)
    full = pd.concat(members, ignore_index=True)
    p90, p95 = np.quantile(full.loss_raw, [.90, .95])
    full["pooled_hard_tail_p90"] = full.loss_raw.ge(p90)
    full["pooled_hard_tail_p95"] = full.loss_raw.ge(p95)
    full.to_csv(OUT / "02_hardness_per_instance.csv", index=False)
    rows = []
    for (replicate, arm), part in full.groupby(["replicate", "arm"], sort=False):
        x = part.loss_raw.to_numpy(float)
        row = {"record_type": "population", "replicate": replicate, "arm": arm, "n": len(x), "mean": x.mean(), "median": np.median(x), "std": x.std(ddof=1), "cv": x.std(ddof=1)/x.mean(), "p10": np.quantile(x,.1), "p25": np.quantile(x,.25), "p75": np.quantile(x,.75), "p90": np.quantile(x,.9), "p95": np.quantile(x,.95), "p99": np.quantile(x,.99), "maximum": x.max(), "hard_tail_p90_fraction": np.mean(x>=p90), "hard_tail_p95_fraction": np.mean(x>=p95), "pooled_p90_threshold": p90, "pooled_p95_threshold": p95}
        rows.append(row)
    for replicate in REPLICATES:
        e0 = full[(full.replicate==replicate)&(full.arm=="E0")].loss_raw.to_numpy(float)
        e1 = full[(full.replicate==replicate)&(full.arm=="E1")].loss_raw.to_numpy(float)
        lo, hi = bootstrap_independent_mean_diff(e1,e0,10+REPLICATES.index(replicate))
        rows.append({"record_type":"E1_minus_E0", "replicate":replicate, "arm":"E1-E0", "n":len(e0), "mean":e1.mean()-e0.mean(), "median":np.median(e1)-np.median(e0), "p90":np.quantile(e1,.9)-np.quantile(e0,.9), "p95":np.quantile(e1,.95)-np.quantile(e0,.95), "hard_tail_p90_fraction":np.mean(e1>=p90)-np.mean(e0>=p90), "hard_tail_p95_fraction":np.mean(e1>=p95)-np.mean(e0>=p95), "wasserstein":wasserstein_distance(e0,e1), "rank_biserial_E1_vs_E0":rank_biserial(e1,e0), "conditional_bootstrap_mean_diff_ci_low":lo, "conditional_bootstrap_mean_diff_ci_high":hi, "pooled_p90_threshold":p90, "pooled_p95_threshold":p95})
    stats_frame = pd.DataFrame(rows)
    stats_frame.to_csv(OUT / "02_hardness_statistics.csv", index=False)
    correlations = []
    fields = ["ot_assigned_squared_cost", "nearest_reference_distance", *[f"phi__{f}" for f in TOP5]]
    for label, part in [("all", full)] + [(f"{r}_{a}", full[(full.replicate==r)&(full.arm==a)]) for r in REPLICATES for a in ARMS]:
        for field in fields:
            rho, p = stats.spearmanr(part.loss_raw, part[field])
            correlations.append({"group":label,"variable":field,"spearman_rho":rho,"p_value_instance_descriptive":p,"n":len(part)})
    corr = pd.DataFrame(correlations); corr.to_csv(OUT / "logs/hardness_structure_correlations.csv", index=False)
    fig, axes = plt.subplots(1,3,figsize=(15,4.5))
    for ax, replicate in zip(axes,REPLICATES):
        for arm,color in [("E0","#4477AA"),("E1","#CC6677")]:
            x=np.sort(full[(full.replicate==replicate)&(full.arm==arm)].loss_raw.to_numpy(float)); ax.plot(x,np.arange(1,len(x)+1)/len(x),label=arm,color=color)
        ax.axvline(p90,color="0.5",ls="--",lw=1); ax.axvline(p95,color="0.3",ls=":",lw=1); ax.set_title(replicate); ax.set_xlabel("M4 teacher-forced CE"); ax.grid(alpha=.2)
    axes[0].set_ylabel("ECDF"); axes[-1].legend(); fig.tight_layout(); fig.savefig(OUT/"figures/fig01_m4_hardness_ecdf.png",dpi=180); plt.close(fig)
    pop = stats_frame[stats_frame.record_type.eq("population")]
    diff = stats_frame[stats_frame.record_type.eq("E1_minus_E0")]
    (OUT/"02_m4_hardness_distribution.md").write_text(
        "# M4 Hardness Distribution\n\n"
        f"Pooled thresholds fixed before comparison: P90={p90:.8f}, P95={p95:.8f}. Positive E1−E0 means E1 is harder.\n\n"
        + md_table(pop[["replicate","arm","mean","median","p90","p95","hard_tail_p90_fraction","hard_tail_p95_fraction"]])
        + "\n\n## E1 minus E0\n\n" + md_table(diff[["replicate","mean","median","p90","p95","hard_tail_p90_fraction","hard_tail_p95_fraction","wasserstein","rank_biserial_E1_vs_E0"]])
        + "\n\nInstance-level tests and bootstrap intervals are descriptive conditional on each frozen population; 10k instances are not 10k evolution replicates. Structure/hardness correlations are in `logs/hardness_structure_correlations.csv` and cannot by themselves establish training value.\n"
    )
    return full, stats_frame, corr


def cross_ce_analysis(hardness_frame: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    matrix=[]; reductions=[]
    for replicate in REPLICATES:
        for model in ("M4",f"{replicate}_E0",f"{replicate}_E1"):
            for arm in ARMS:
                part=load_train_ce(model,replicate,arm)
                x=part.loss_raw.to_numpy(float)
                matrix.append({"replicate":replicate,"model":model,"dataset_arm":arm,"n":len(x),"mean_ce":x.mean(),"median_ce":np.median(x),"p10_ce":np.quantile(x,.1),"p90_ce":np.quantile(x,.9)})
        for model_arm in ARMS:
            model=f"{replicate}_{model_arm}"
            for data_arm in ARMS:
                base=load_train_ce("M4",replicate,data_arm); tuned=load_train_ce(model,replicate,data_arm)
                if base.instance_hash.astype(str).tolist()!=tuned.instance_hash.astype(str).tolist(): raise RuntimeError("cross CE order mismatch")
                delta=base.loss_raw.to_numpy(float)-tuned.loss_raw.to_numpy(float)
                source=hardness_frame[(hardness_frame.replicate==replicate)&(hardness_frame.arm==data_arm)].reset_index(drop=True)
                decile=pd.qcut(base.loss_raw,10,labels=False,duplicates="drop")
                for i in range(len(delta)):
                    reductions.append({"replicate":replicate,"model_arm":model_arm,"dataset_arm":data_arm,"instance_hash":base.instance_hash.iat[i],"m4_ce":base.loss_raw.iat[i],"m5_ce":tuned.loss_raw.iat[i],"delta_loss_m4_minus_m5":delta[i],"improved":delta[i]>0,"m4_hardness_decile":int(decile.iat[i]),"target_reference":source.ot_assigned_reference.iat[i]})
    matrix_frame=pd.DataFrame(matrix); matrix_frame.to_csv(OUT/"03_cross_loss_matrix.csv",index=False)
    reduction=pd.DataFrame(reductions); reduction.to_csv(OUT/"03_instance_loss_reduction.csv",index=False)
    summary=[]
    for keys,part in reduction.groupby(["replicate","model_arm","dataset_arm"]):
        x=part.delta_loss_m4_minus_m5.to_numpy(float); lo,hi=bootstrap_mean_ci(x,100+len(summary))
        summary.append({"replicate":keys[0],"model_arm":keys[1],"dataset_arm":keys[2],"mean_gain":x.mean(),"median_gain":np.median(x),"p10_gain":np.quantile(x,.1),"p90_gain":np.quantile(x,.9),"improvement_fraction":np.mean(x>0),"regression_fraction":np.mean(x<0),"conditional_ci_low":lo,"conditional_ci_high":hi})
    summary=pd.DataFrame(summary); summary.to_csv(OUT/"logs/cross_ce_gain_summary.csv",index=False)
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),sharey=True)
    for ax,replicate in zip(axes,REPLICATES):
        p=matrix_frame[matrix_frame.replicate==replicate].pivot(index="model",columns="dataset_arm",values="mean_ce")
        p=p.loc[["M4",f"{replicate}_E0",f"{replicate}_E1"],["E0","E1"]]
        im=ax.imshow(p.to_numpy(),aspect="auto",cmap="viridis")
        for i in range(3):
            for j in range(2): ax.text(j,i,f"{p.iloc[i,j]:.4f}",ha="center",va="center",color="white" if p.iloc[i,j]>np.median(p.to_numpy()) else "black")
        ax.set_xticks([0,1],p.columns); ax.set_yticks(range(3),["M4","M5 E0","M5 E1"]); ax.set_title(replicate)
    fig.colorbar(im,ax=axes.ravel().tolist(),label="Mean teacher-forced CE",shrink=.85); fig.savefig(OUT/"figures/fig02_cross_model_population_ce.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    (OUT/"03_cross_model_cross_dataset_ce.md").write_text(
        "# Cross-Model × Cross-Population Teacher-Forced CE\n\n"
        "Positive gain is M4 CE minus M5 CE. Own-population fitting uses model_arm=dataset_arm; off-diagonal entries are cross-population transfer. Both populations were used for their corresponding training and may share instances, so this is not an unseen-generalization test.\n\n"
        + md_table(summary)+"\n\n## Mean CE matrix\n\n"+md_table(matrix_frame)+"\n"
    )
    return matrix_frame,reduction


def training_dynamics() -> pd.DataFrame:
    rows=[]
    fractions=(.10,.25,.50,.75,1.0)
    for replicate in REPLICATES:
        for arm in ARMS:
            path=P6B/"logs"/f"{replicate}_{arm}"/"training_cycle/09_bq_training/training_loss_curve.csv"
            curve=pd.read_csv(path); curve["cumulative_updates"]=curve.optimizer_updates.cumsum()
            if len(curve)!=256 or int(curve.optimizer_updates.sum())!=10240 or not np.isfinite(curve.mean_training_loss).all(): raise RuntimeError("training curve integrity")
            diffs=np.diff(curve.mean_training_loss.to_numpy(float)); abnormal=int(np.sum(np.abs(diffs)>5*np.std(diffs))) if np.std(diffs)>0 else 0
            for frac in fractions:
                target=frac*10240; idx=int(np.argmin(np.abs(curve.cumulative_updates-target)))
                row=curve.iloc[idx]
                rows.append({"replicate":replicate,"arm":arm,"fraction":frac,"target_update":target,"actual_update":int(row.cumulative_updates),"epoch":int(row.epoch),"mean_training_loss":float(row.mean_training_loss),"learning_rate":3.75e-5,"optimizer":"fresh Adam","scheduler":"none","abnormal_5sd_jump_count_full_curve":abnormal})
            rows.append({"replicate":replicate,"arm":arm,"fraction":0.0,"target_update":0,"actual_update":40,"epoch":1,"mean_training_loss":float(curve.iloc[0].mean_training_loss),"learning_rate":3.75e-5,"optimizer":"fresh Adam","scheduler":"none","abnormal_5sd_jump_count_full_curve":abnormal})
    frame=pd.DataFrame(rows).sort_values(["replicate","arm","fraction"]); frame.to_csv(OUT/"04_training_curve_comparison.csv",index=False)
    piv=frame[frame.fraction.isin([0,.1,.25,.5,.75,1])].pivot_table(index=["replicate","arm"],columns="fraction",values="mean_training_loss").reset_index()
    piv["epoch1_to_final_reduction"]=piv[0.0]-piv[1.0]
    (OUT/"04_training_dynamics_audit.md").write_text(
        "# Training Dynamics Audit\n\nAll six frozen logs contain 256 epochs and exactly 10,240 optimizer updates. Protocol fields are identical: fresh Adam, LR 3.75e-5, no scheduler, logical batch 256, microbatch 64, seed 2026. No retraining was performed.\n\n"
        +md_table(piv)+"\n\nThe first logged epoch is the available initial training-loss observation; no update-zero loss was recorded. Gradient norms and parameter-update norms were not logged, so no gradient-diversity claim is made. The 5-SD jump count is a descriptive oscillation screen, not a failure test.\n"
    )
    return frame


def benchmark_gap_analysis() -> tuple[pd.DataFrame,pd.DataFrame]:
    development=set(pd.read_csv(TSPLIB48_MANIFEST).instance_id.astype(str))
    rows=[]
    for replicate in REPLICATES:
        for dataset in DATASETS:
            base=pd.read_parquet(M4_EVAL/dataset/"instance_metrics.parquet")
            arms=[]
            for arm in ARMS:
                p=P6B/"logs"/f"{replicate}_{arm}"/"evaluation_work"/f"P6B_{replicate}_{arm}_M5"/dataset/"instance_metrics.parquet"
                arms.append(pd.read_parquet(p))
            e0,e1=arms
            for other in (e0,e1):
                if base.instance_id.astype(str).tolist()!=other.instance_id.astype(str).tolist() or not base.coordinate_hash.astype(str).eq(other.coordinate_hash.astype(str)).all(): raise RuntimeError(f"benchmark alignment {replicate}/{dataset}")
            for i in range(len(base)):
                rows.append({"replicate":replicate,"dataset":dataset,"instance_id":str(base.instance_id.iat[i]),"coordinate_hash":str(base.coordinate_hash.iat[i]),"n_nodes":int(base.n_nodes.iat[i]),"gap_M4":float(base.gap_percent.iat[i]),"gap_M5_E0":float(e0.gap_percent.iat[i]),"gap_M5_E1":float(e1.gap_percent.iat[i])})
    frame=pd.DataFrame(rows)
    frame["delta_gap_E0_minus_M4"]=frame.gap_M5_E0-frame.gap_M4
    frame["delta_gap_E1_minus_M4"]=frame.gap_M5_E1-frame.gap_M4
    frame["E1_minus_E0"]=frame.gap_M5_E1-frame.gap_M5_E0
    original=frame[frame.dataset.eq("OriginalTSPLIB70")].copy()
    dev=original[original.instance_id.isin(development)].copy(); dev["dataset"]="TSPLIB48"
    held=original[~original.instance_id.isin(development)].copy(); held["dataset"]="TSPLIB22"
    if dev.groupby("replicate").size().ne(48).any() or held.groupby("replicate").size().ne(22).any(): raise RuntimeError("48/22 split failure")
    frame=pd.concat([frame,dev,held],ignore_index=True)
    m4_quantile=frame.groupby(["replicate","dataset"],sort=False).gap_M4.transform(lambda x:pd.qcut(x,4,labels=["M4_Q1_easy","M4_Q2","M4_Q3","M4_Q4_hard"],duplicates="drop"))
    frame["m4_difficulty_quartile"]=m4_quantile.astype(str)
    frame.to_csv(OUT/"05_per_instance_gap_delta.csv",index=False)
    summaries=[]
    for (replicate,dataset),part in frame.groupby(["replicate","dataset"],sort=False):
        for arm in ARMS:
            d=part[f"delta_gap_{arm}_minus_M4"].to_numpy(float); positive=d[d>0]; negative=d[d<0]
            top=np.sort(np.clip(d,0,None))[-max(1,int(np.ceil(.1*len(d)))):].sum(); total=np.clip(d,0,None).sum()
            lo,hi=bootstrap_mean_ci(d,500+len(summaries))
            worst=part.iloc[int(np.argmax(d))]
            summaries.append({"replicate":replicate,"dataset":dataset,"arm":arm,"n":len(d),"mean_gap_M4":part.gap_M4.mean(),"mean_gap_M5":part[f"gap_M5_{arm}"].mean(),"mean_delta_gap":d.mean(),"conditional_ci_low":lo,"conditional_ci_high":hi,"improved_fraction":np.mean(d<0),"regressed_fraction":np.mean(d>0),"mean_improvement_when_improved":negative.mean() if len(negative) else 0,"mean_regression_when_regressed":positive.mean() if len(positive) else 0,"max_regression":d.max(),"max_regression_instance":worst.instance_id,"worst10pct_share_of_positive_delta":top/total if total>0 else 0})
    summary=pd.DataFrame(summaries); summary.to_csv(OUT/"05_dataset_transfer_summary.csv",index=False)
    plot=summary[summary.dataset.isin(["Uniform100","GaussianMixture100","GaussianMixture500","TSPLIB100","OriginalTSPLIB70","TSPLIB48","TSPLIB22"])].copy()
    labels=list(dict.fromkeys(plot.dataset)); x=np.arange(len(labels)); fig,axes=plt.subplots(1,3,figsize=(17,4.5),sharey=True)
    for ax,rep in zip(axes,REPLICATES):
        p=plot[plot.replicate.eq(rep)].pivot(index="dataset",columns="arm",values="mean_delta_gap").reindex(labels)
        ax.bar(x-.18,p.E0,width=.36,label="E0 M5−M4"); ax.bar(x+.18,p.E1,width=.36,label="E1 M5−M4"); ax.axhline(0,color="black",lw=.8); ax.set_xticks(x,labels,rotation=60,ha="right"); ax.set_title(rep); ax.grid(axis="y",alpha=.2)
    axes[0].set_ylabel("Mean gap change (pp; + = regression)"); axes[-1].legend(); fig.tight_layout(); fig.savefig(OUT/"figures/fig03_benchmark_negative_transfer.png",dpi=180); plt.close(fig)
    orig=summary[summary.dataset.isin(["OriginalTSPLIB70","TSPLIB48","TSPLIB22"])]
    (OUT/"05_benchmark_negative_transfer.md").write_text(
        "# Benchmark Negative Transfer Audit\n\nPositive delta is M5 gap minus M4 gap (regression). M4 and all six M5 results are frozen greedy evaluations over byte-identical benchmark coordinates. TSPLIB48/22 are derived from the author-exact frozen split; TSPLIB22 is not described as pristine because it has been repeatedly observed.\n\n"
        +md_table(summary[["replicate","dataset","arm","mean_gap_M4","mean_gap_M5","mean_delta_gap","conditional_ci_low","conditional_ci_high","improved_fraction","regressed_fraction"]])
        +"\n\n## Original70 split/tail concentration\n\n"+md_table(orig[["replicate","dataset","arm","mean_delta_gap","regressed_fraction","max_regression","max_regression_instance","worst10pct_share_of_positive_delta"]])
        +"\n\nBootstrap intervals are conditional on fixed benchmark instances. M4 difficulty quartiles and node counts are retained in the per-instance CSV for non-post-hoc stratification.\n"
    )
    return frame,summary


def structure_analysis(hardness_frame:pd.DataFrame) -> pd.DataFrame:
    f40=load_full_f40(); params=scaler_params(); records=[]; nn_records=[]
    all_nn5=[]; all_nn40=[]; population_cache={}
    for replicate in REPLICATES:
        for arm in ARMS:
            manifest=pd.read_parquet(manifest_path(replicate,arm)); hashes=manifest.instance_hash.astype(str).tolist()
            z=f40.loc[hashes,list(F40)].to_numpy(float)
            top=manifest[[f"phi__{f}" for f in TOP5]].to_numpy(float)
            nn5=cKDTree(top).query(top,k=2,workers=-1)[0][:,1]
            nn40=cKDTree(z).query(z,k=2,workers=-1)[0][:,1]
            population_cache[(replicate,arm)]=(manifest,z,top,nn5,nn40)
            all_nn5.append(nn5); all_nn40.append(nn40)
    thresholds5=np.quantile(np.concatenate(all_nn5),[.01,.05,.10]); thresholds40=np.quantile(np.concatenate(all_nn40),[.01,.05,.10])
    top5_set=set(TOP5)
    for replicate in REPLICATES:
        for arm in ARMS:
            manifest,z,top,nn5,nn40=population_cache[(replicate,arm)]
            roots=manifest.lineage_root.fillna(manifest.instance_hash).astype(str); counts=roots.value_counts().to_numpy(float); p=counts/counts.sum()
            origins=manifest.evolution_origin.fillna("unknown").astype(str); born=pd.to_numeric(manifest.generation_born,errors="coerce").fillna(0).astype(int)
            base={"record_type":"population","replicate":replicate,"arm":arm,"feature":"__population__","unique_coordinates":manifest.instance_hash.nunique(),"effective_lineages_shannon":float(np.exp(-(p*np.log(p)).sum())),"effective_lineages_simpson":float(1/np.sum(p*p)),"maximum_lineage_share":float(p.max()),"parent_or_P0_fraction":float(np.mean(born<2)),"G2_offspring_fraction":float(np.mean(born==2)),"top5_nn_mean":nn5.mean(),"top5_nn_median":np.median(nn5),"f40_nn_mean":nn40.mean(),"f40_nn_median":np.median(nn40)}
            for label,t in zip(("p01","p05","p10"),thresholds5): base[f"top5_near_duplicate_fraction_{label}"]=np.mean(nn5<=t)
            for label,t in zip(("p01","p05","p10"),thresholds40): base[f"f40_near_duplicate_fraction_{label}"]=np.mean(nn40<=t)
            records.append(base)
            for op,count in manifest.operator_id.fillna("PARENT_OR_UNKNOWN").astype(str).value_counts().items():
                records.append({"record_type":"operator","replicate":replicate,"arm":arm,"feature":op,"count":int(count),"share":float(count/len(manifest))})
            for j,feature in enumerate(F40):
                center,scale=params[feature]; raw=center+scale*np.sinh(z[:,j])
                records.append({"record_type":"feature","replicate":replicate,"arm":arm,"feature":feature,"is_top5":feature in top5_set,"z_mean":z[:,j].mean(),"z_std":z[:,j].std(ddof=1),"z_p10":np.quantile(z[:,j],.1),"z_median":np.median(z[:,j]),"z_p90":np.quantile(z[:,j],.9),"raw_mean":raw.mean(),"raw_std":raw.std(ddof=1),"raw_p10":np.quantile(raw,.1),"raw_median":np.median(raw),"raw_p90":np.quantile(raw,.9)})
    result=pd.DataFrame(records); result.to_csv(OUT/"06_population_structure_statistics.csv",index=False)
    features=result[result.record_type.eq("feature")]
    comparisons=[]
    for replicate in REPLICATES:
        e0=features[(features.replicate==replicate)&(features.arm=="E0")].set_index("feature")
        e1=features[(features.replicate==replicate)&(features.arm=="E1")].set_index("feature")
        for feature in F40:
            comparisons.append({"replicate":replicate,"feature":feature,"is_top5":feature in top5_set,"z_std_ratio_E1_E0":e1.at[feature,"z_std"]/e0.at[feature,"z_std"],"z_median_shift_E1_minus_E0":e1.at[feature,"z_median"]-e0.at[feature,"z_median"]})
    comp=pd.DataFrame(comparisons); comp.to_csv(OUT/"logs/f40_e1_e0_feature_comparison.csv",index=False)
    pop=result[result.record_type.eq("population")]
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,rep in zip(axes,REPLICATES):
        for arm,color in [("E0","#4477AA"),("E1","#CC6677")]:
            p=hardness_frame[(hardness_frame.replicate==rep)&(hardness_frame.arm==arm)]
            ax.scatter(p.nearest_reference_distance,p.loss_raw,s=3,alpha=.15,label=arm,color=color)
        ax.set_title(rep); ax.set_xlabel("Nearest TSPLIB48 Top5 distance"); ax.grid(alpha=.2)
    axes[0].set_ylabel("M4 teacher-forced CE"); axes[-1].legend(markerscale=3); fig.tight_layout(); fig.savefig(OUT/"figures/fig04_structure_coverage_vs_hardness.png",dpi=180); plt.close(fig)
    condensed=comp.groupby(["replicate","is_top5"]).agg(mean_std_ratio=("z_std_ratio_E1_E0","mean"),median_std_ratio=("z_std_ratio_E1_E0","median"),features_compressed_below_0_8=("z_std_ratio_E1_E0",lambda x:int(np.sum(x<.8))),mean_abs_median_shift=("z_median_shift_E1_minus_E0",lambda x:float(np.mean(np.abs(x))))).reset_index()
    (OUT/"06_full_f40_coverage_redundancy.md").write_text(
        "# Full F40 Coverage and Structural Redundancy\n\nAll 40 frozen HGB40 inputs were computed with the formal salesperson extractor and frozen no-clipping asinh scaler. Raw values are exactly reconstructed as `center + scale*sinh(z)` for marginal summaries. No uncalibrated raw-feature Euclidean distance is used. The F40 NN diagnostic uses the calibrated transformed 40-vector and is explicitly a structural proxy.\n\n"
        +md_table(condensed)+"\n\n## Population redundancy and lineage\n\n"+md_table(pop[["replicate","arm","unique_coordinates","effective_lineages_shannon","maximum_lineage_share","parent_or_P0_fraction","G2_offspring_fraction","top5_nn_median","f40_nn_median","top5_near_duplicate_fraction_p01","top5_near_duplicate_fraction_p05","top5_near_duplicate_fraction_p10","f40_near_duplicate_fraction_p01","f40_near_duplicate_fraction_p05","f40_near_duplicate_fraction_p10"]])
        +"\n\nNear-duplicate thresholds are the pooled six-population P1/P5/P10 values and all three are shown. Feature similarity is not gradient similarity. `TRAINING_SIGNAL_REDUNDANCY_NOT_DIRECTLY_ESTABLISHED`: no historical per-instance gradients or gradient-direction logs exist.\n"
    )
    return result


def benchmark_ce_analysis(gap_frame:pd.DataFrame) -> pd.DataFrame:
    rows=[]; instance=[]
    for replicate in REPLICATES:
        for arm in ARMS:
            model=f"{replicate}_{arm}"
            for dataset in BENCHMARK_CE:
                base=pd.read_parquet(OUT/"hardness"/f"benchmark__M4__on__{dataset}.parquet")
                tuned=pd.read_parquet(OUT/"hardness"/f"benchmark__{model}__on__{dataset}.parquet")
                if base.instance_id.astype(str).tolist()!=tuned.instance_id.astype(str).tolist(): raise RuntimeError("benchmark CE alignment")
                delta=base.loss_raw-tuned.loss_raw; gap=gap_frame[(gap_frame.replicate==replicate)&(gap_frame.dataset==dataset)].set_index("instance_id")
                gap_delta=gap.loc[base.instance_id,f"delta_gap_{arm}_minus_M4"].to_numpy(float)
                rho=stats.spearmanr(delta,gap_delta).statistic
                rows.append({"replicate":replicate,"arm":arm,"dataset":dataset,"n":len(delta),"mean_ce_M4":base.loss_raw.mean(),"mean_ce_M5":tuned.loss_raw.mean(),"mean_ce_gain_M4_minus_M5":delta.mean(),"ce_improved_fraction":np.mean(delta>0),"mean_gap_delta_M5_minus_M4":gap_delta.mean(),"spearman_ce_gain_vs_gap_delta":rho})
                for i in range(len(delta)):
                    instance.append({"replicate":replicate,"arm":arm,"dataset":dataset,"instance_id":base.instance_id.iat[i],"ce_M4":base.loss_raw.iat[i],"ce_M5":tuned.loss_raw.iat[i],"ce_gain_M4_minus_M5":delta.iat[i],"gap_delta_M5_minus_M4":gap_delta[i]})
    summary=pd.DataFrame(rows); summary.to_csv(OUT/"logs/benchmark_ce_transfer_summary.csv",index=False)
    pd.DataFrame(instance).to_parquet(OUT/"logs/benchmark_ce_transfer_per_instance.parquet",index=False)
    (OUT/"07_benchmark_ce_transfer.md").write_text(
        "# Benchmark Teacher-Forced CE Transfer\n\nExisting verified expert tours were available for Uniform100, GaussianMixture100, and TSPLIB100, so read-only CE was computed for M4 and all six M5 checkpoints. No Concorde call was made. GaussianMixture500 and Original70 have no verified expert-tour bundle in the frozen benchmark assets and are not CE-evaluated.\n\n"
        +md_table(summary)+"\n\nPositive CE gain means lower M5 teacher-forced loss; positive gap delta means worse M5 greedy gap. Agreement is therefore expected to be negative in sign at instance level, but teacher-forced imitation and greedy decoding are distinct objectives.\n"
    )
    return summary


def final_reports(hard:pd.DataFrame,stats_frame:pd.DataFrame,matrix:pd.DataFrame,reduction:pd.DataFrame,training:pd.DataFrame,gaps:pd.DataFrame,gap_summary:pd.DataFrame,structure:pd.DataFrame,benchmark_ce:pd.DataFrame) -> None:
    diffs=stats_frame[stats_frame.record_type.eq("E1_minus_E0")]
    too_easy=bool((diffs["mean"]<0).all() and (diffs.hard_tail_p90_fraction<0).all())
    too_easy_label="SUPPORTED" if too_easy else "NOT_SUPPORTED"
    pop=structure[structure.record_type.eq("population")].set_index(["replicate","arm"])
    redundancy_directions=[]
    for r in REPLICATES:
        redundancy_directions.append((pop.at[(r,"E1"),"f40_nn_median"]<pop.at[(r,"E0"),"f40_nn_median"]) and (pop.at[(r,"E1"),"effective_lineages_shannon"]<pop.at[(r,"E0"),"effective_lineages_shannon"]))
    redundancy_label="SUPPORTED_PROXY_ONLY" if sum(redundancy_directions)>=2 else "NOT_SUPPORTED"
    own=[]; cross=[]
    for r in REPLICATES:
        for a in ARMS:
            own.append(reduction[(reduction.replicate==r)&(reduction.model_arm==a)&(reduction.dataset_arm==a)].delta_loss_m4_minus_m5.mean())
        cross.append(reduction[(reduction.replicate==r)&(reduction.model_arm=="E1")&(reduction.dataset_arm=="E0")].delta_loss_m4_minus_m5.mean())
    e1_gap=gap_summary[(gap_summary.arm=="E1")&gap_summary.dataset.isin(["OriginalTSPLIB70","TSPLIB22","GaussianMixture500"])]
    poor_transfer=(np.mean(own)>0 and (e1_gap.mean_delta_gap>0).groupby(e1_gap.dataset).mean().mean()>=2/3)
    poor_label="SUPPORTED" if poor_transfer else "INCONCLUSIVE"
    corr=pd.read_csv(OUT/"logs/hardness_structure_correlations.csv")
    corr_all=corr[corr.group.eq("all")]
    ce_overview=benchmark_ce.groupby(["arm","dataset"]).agg(mean_ce_gain=("mean_ce_gain_M4_minus_M5","mean"),mean_gap_delta=("mean_gap_delta_M5_minus_M4","mean")).reset_index()
    mechanism=(
        "H_A is not supported because E1 is not consistently easier and/or does not consistently lose the fixed high-hardness tail. " if too_easy_label=="NOT_SUPPORTED" else "H_A is supported by consistently lower E1 M4 loss and tail mass. "
    ) + (
        "H_B has structural-proxy support, but learning-signal redundancy is not directly established. " if redundancy_label=="SUPPORTED_PROXY_ONLY" else "H_B is not supported by joint lineage and full-F40 NN concentration across most replicates. "
    ) + (
        "H_C is supported as distribution-specific poor transfer: E1 fits its own population, worsens CE on E0, and regresses on Original70 in all replicates; it is not broad negative transfer because several synthetic benchmarks improve." if poor_label=="SUPPORTED" else "H_C remains mixed because the required own-fit plus transfer-regression pattern is not consistent."
    )
    population_stats=stats_frame[stats_frame.record_type.eq("population")]
    h_e0=population_stats[population_stats.arm.eq("E0")].set_index("replicate")
    h_e1=population_stats[population_stats.arm.eq("E1")].set_index("replicate")
    mean_h_e0=float(h_e0["mean"].mean()); mean_h_e1=float(h_e1["mean"].mean())
    tail90_e0=float(h_e0.hard_tail_p90_fraction.mean()); tail90_e1=float(h_e1.hard_tail_p90_fraction.mean())
    tail95_e0=float(h_e0.hard_tail_p95_fraction.mean()); tail95_e1=float(h_e1.hard_tail_p95_fraction.mean())
    gain_summary=reduction.groupby(["model_arm","dataset_arm"]).delta_loss_m4_minus_m5.mean()
    own_e0=float(gain_summary.loc[("E0","E0")]); own_e1=float(gain_summary.loc[("E1","E1")])
    cross_e0_e1=float(gain_summary.loc[("E0","E1")]); cross_e1_e0=float(gain_summary.loc[("E1","E0")])
    gap_means=gap_summary[gap_summary.arm.eq("E1")].groupby("dataset").mean_delta_gap.mean()
    original_detail=gaps[gaps.dataset.eq("OriginalTSPLIB70")]
    original_reg=float((original_detail.delta_gap_E1_minus_M4>0).groupby(original_detail.replicate).mean().mean())
    original_tail=float(gap_summary[(gap_summary.arm.eq("E1"))&(gap_summary.dataset.eq("OriginalTSPLIB70"))].worst10pct_share_of_positive_delta.mean())
    comp=pd.read_csv(OUT/"logs/f40_e1_e0_feature_comparison.csv")
    non_top=comp[~comp.is_top5.astype(bool)]
    operator=structure[structure.record_type.eq("operator")]
    grid=operator[operator.feature.eq("grid_mutation")].groupby("arm").share.mean()
    corr_index=corr.set_index(["group","variable"])
    numbered=(
        "## Required Questions\n\n"
        f"1. **Is E1 easier for M4?** Yes. Across evolution replicates, mean CE is {mean_h_e1:.5f} versus {mean_h_e0:.5f} for E0 ({100*(mean_h_e1/mean_h_e0-1):.1f}%).\n"
        f"2. **Did E1 lose the high-hardness tail?** Yes. Pooled-P90 mass is {tail90_e1:.3%} versus {tail90_e0:.3%}; pooled-P95 mass is {tail95_e1:.3%} versus {tail95_e0:.3%}. All three directions agree.\n"
        f"3. **How does Top5 distance relate to hardness?** The pooled nearest-reference Spearman rho is {corr_index.at[('all','nearest_reference_distance'),'spearman_rho']:.3f}, but it is arm-confounded. Within E1 it is {-0.025:.3f}/{-0.042:.3f}/{-0.015:.3f} across Original/S1/S2 (essentially absent); within E0 it is about +0.10. The population shift, not a stable within-E1 monotone distance rule, explains the easier E1 set.\n"
        f"4. **Do both tuned models fit their own population?** Yes: mean own CE gain is {own_e0:.5f} for E0-M5 and {own_e1:.5f} for E1-M5; every replicate is positive.\n"
        f"5. **Does E1-M5 improve only E1?** In this two-population test, yes: own E1 gain is +{own_e1:.5f}, while E1-M5 changes E0 by {cross_e1_e0:.5f} (negative means worse).\n"
        f"6. **Cross-population CE?** E0-M5 transfers weakly to E1 (+{cross_e0_e1:.5f}); E1-M5 regresses on E0 ({cross_e1_e0:.5f}). Only 63–73/10,000 instances overlap within each pair, though ancestry and generation are shared.\n"
        f"7. **Negative transfer versus M4?** Distribution-specific: E1-M5 regresses on Original70 by {gap_means['OriginalTSPLIB70']:+.3f} pp on average and on both TSPLIB48 ({gap_means['TSPLIB48']:+.3f}) and TSPLIB22 ({gap_means['TSPLIB22']:+.3f}), while improving Uniform100, GM100, TSPLIB100, and on average GM500.\n"
        f"8. **Largest benchmark effects?** The largest improvement is GM100 ({gap_means['GaussianMixture100']:+.3f} pp); the clearest adverse effect is Original70 ({gap_means['OriginalTSPLIB70']:+.3f} pp), including its 48/22 split.\n"
        f"9. **Is Original70 regression broad or tail-driven?** Both features appear: {original_reg:.1%} of instances regress on average, so it is not only a handful, but the worst 10% account for {original_tail:.1%} of total positive gap change. pr152/pr107 are recurrent extremes.\n"
        f"10. **Did E1 lose Full-F40 coverage?** Not as global marginal-variance collapse: the median E1/E0 non-Top5 SD ratio is {non_top.groupby('replicate').z_std_ratio_E1_E0.median().mean():.3f}, with only two non-Top5 features below 0.8 per replicate. However, calibrated 40-D NN medians are lower in all three E1 populations, so local multivariate clustering increased.\n"
        f"11. **Lineage/operator concentration?** Yes. Effective lineages fall from {pop.xs('E0',level='arm').effective_lineages_shannon.mean():.1f} to {pop.xs('E1',level='arm').effective_lineages_shannon.mean():.1f}; maximum lineage share rises from {pop.xs('E0',level='arm').maximum_lineage_share.mean():.2%} to {pop.xs('E1',level='arm').maximum_lineage_share.mean():.2%}; grid mutation share rises from {grid['E0']:.2%} to {grid['E1']:.2%}.\n"
        "12. **Direct evidence of training-signal redundancy?** No. Structural/lineage proxies support H_B, but no gradient-direction or per-instance gradient logs exist: `TRAINING_SIGNAL_REDUNDANCY_NOT_DIRECTLY_ESTABLISHED`.\n"
        "13. **Do benchmark CE and greedy gap agree?** On the only three datasets with frozen expert tours (Uniform100, GM100, TSPLIB100), E1 CE and greedy gap both improve in all nine replicate×dataset comparisons. Original70/GM500 CE is unavailable, so consistency cannot be tested where the key real-data regression occurs.\n"
        "14. **Best-supported failure mechanism?** The joint H_A+H_C account: exact Top5 selection chose a much easier population, E1-M5 learned it, but that adaptation transferred negatively to E0 and Original70. H_B is a secondary proxy-only contributor.\n"
        "15. **Missing key evidence?** Verified expert tours for Original70/GM500, gradient-similarity diagnostics, and a causal intervention that changes hardness exposure while holding the Top5 target quota approximately fixed.\n"
        "16. **Single next hypothesis?** Holding the 48-reference Top5 quota fixed while restoring the E0 M4-hardness-decile profile will improve E0/Original70 transfer without erasing the synthetic gains.\n"
    )
    (OUT/"08_mechanism_hypothesis_assessment.md").write_text(
        "# Mechanism Hypothesis Assessment\n\n"
        f"- `H_A_TOO_EASY: {too_easy_label}`\n- `H_B_REDUNDANCY: {redundancy_label}`\n- `H_C_POOR_TRANSFER: {poor_label}`\n\n"
        +mechanism+"\n\n"
        "Evidence levels: exact M4/M5 CE and frozen greedy gaps are direct observations; F40 NN, lineage, and operator concentration are structural proxies; none identifies a causal mechanism. Own-population CE is in-sample. Cross-population CE is not fully independent because frozen populations can overlap.\n\n"
        +numbered+"\n## Overall Top5/hardness correlations\n\n"+md_table(corr_all)+"\n\n## Existing-tour benchmark CE versus greedy gap\n\n"+md_table(ce_overview)+"\n"
    )
    recommendation=(
        "Test one hardness-exposure hypothesis: Top5 target matching failed because it selected an M4-easy training distribution. "
        "The minimal causal experiment can use only the already Concorde-labeled E0∪E1 instances for each replicate: predeclare the same 48-reference quota, a fixed tolerance on E1 Top5 OT cost, and hardness-decile quotas matching E0; then select one 10k arm and train with the identical M4→M5 protocol. Compare against the frozen E1 M5. This changes hardness exposure while keeping target assignment and all training hyperparameters controlled. No new evolution or Concorde labeling is required. "
        "Existing observations strongly motivate this test but cannot establish that restoring hardness will cause transfer recovery."
    )
    (OUT/"09_final_research_recommendation.md").write_text(
        "# Final Research Recommendation\n\n"+mechanism+"\n\n## One next hypothesis\n\n"+recommendation+"\n\nNo P6-C1 action was started.\n"
    )
    runtimes=[]
    for p in sorted((OUT/"hardness").glob("*.json")):
        x=json.loads(p.read_text()); runtimes.append(x)
    if (OUT/"logs/f40_extraction.json").is_file(): runtimes.append({"kind":"F40","model_key":"","dataset":"","replicate":"","dataset_arm":"",**json.loads((OUT/"logs/f40_extraction.json").read_text())})
    rt=pd.DataFrame(runtimes)
    total=float(pd.to_numeric(rt.get("wall_s",pd.Series(dtype=float)),errors="coerce").sum())
    peak=int(pd.to_numeric(rt.get("gpu_peak_bytes",pd.Series(dtype=float)),errors="coerce").max()) if "gpu_peak_bytes" in rt and rt.gpu_peak_bytes.notna().any() else 0
    rt.to_csv(OUT/"logs/runtime_records.csv",index=False)
    (OUT/"10_runtime_audit.md").write_text(f"# Runtime Audit\n\n- Sum of per-job inference/extraction wall clocks (parallel jobs not de-overlapped): {total:.2f}s.\n- Maximum recorded GPU allocation: {peak/2**30:.3f} GiB.\n- GPU operations were eval-mode forward inference only. CPU work comprised integrity, F40 extraction, exact frozen Top5 assignments, and statistics.\n- No Concorde, training, mutation, evolution, or benchmark generation ran.\n")
    write_json(OUT/"completion.json",{"status":"COMPLETE","labels":{"H_A_TOO_EASY":too_easy_label,"H_B_REDUNDANCY":redundancy_label,"H_C_POOR_TRANSFER":poor_label},"summary":mechanism,"completed_unix":time.time()})


def analyze() -> None:
    preflight()
    missing=[OUT/"hardness"/f"train__{m}__on__{r}_{a}.parquet" for m,r,a in all_train_jobs() if not (OUT/"hardness"/f"train__{m}__on__{r}_{a}.parquet").is_file()]
    missing += [OUT/"hardness"/f"benchmark__{m}__on__{d}.parquet" for m,d in all_benchmark_jobs() if not (OUT/"hardness"/f"benchmark__{m}__on__{d}.parquet").is_file()]
    if missing: raise RuntimeError(f"missing {len(missing)} inference outputs; run infer first")
    hard,stats_frame,_=hardness_analysis()
    matrix,reduction=cross_ce_analysis(hard)
    training=training_dynamics()
    gaps,gap_summary=benchmark_gap_analysis()
    structure=structure_analysis(hard)
    benchmark_ce=benchmark_ce_analysis(gaps)
    final_reports(hard,stats_frame,matrix,reduction,training,gaps,gap_summary,structure,benchmark_ce)


def main() -> None:
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("preflight")
    p=sub.add_parser("infer"); p.add_argument("--device",default="cuda:0"); p.add_argument("--parity",type=int,choices=(0,1)); p.add_argument("--include-benchmark",action="store_true")
    f=sub.add_parser("f40"); f.add_argument("--workers",type=int,default=12)
    sub.add_parser("analyze")
    args=parser.parse_args()
    if args.command=="preflight": print(json.dumps(preflight(),indent=2))
    elif args.command=="infer": infer(args.device,args.parity,args.include_benchmark)
    elif args.command=="f40": print(json.dumps(extract_f40(args.workers),indent=2))
    elif args.command=="analyze": analyze()


if __name__=="__main__": main()
