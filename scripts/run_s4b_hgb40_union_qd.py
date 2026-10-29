#!/usr/bin/env python
"""S4-B: frozen HGB40 + Parent/Child Union + QD, M0 through ten cycles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import signal
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil
import socket
import sys
import time
import traceback
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import external_v2_passive_blade_eval as evaluator
import run_blade_v2 as base
import run_s3b2_rf_hgb_feature_label as s3b2
from bq.checkpoint import load_canonical_checkpoint
from features.qd4 import QD4_FEATURE_NAMES
from final_run.science import hardness, load_anchor, membership_hash, qd_frame, select_population, summary_stats
from final_run.stage import atomic_json, atomic_stage, sha256_file, utc_now, verify_stage
from mutation import OPERATOR_NAMES
from surrogate_benchmark.models import HGB_CFG_CANONICAL, SEEDS, hgb_params, median_stack
from s4b_reporting import BENCHMARKS, build_reports


RUN_ID = "S4_B_HGB40_union_QD_10cycle_M0_v1"
RUN_DIR = ROOT / "runs" / RUN_ID
CYCLES_DIR = RUN_DIR / "cycles"
STATE_DIR = RUN_DIR / "_state"
RUNTIME = ROOT / "runtime"
LOCK = RUNTIME / "s4_b_hgb40_union_qd.lock"
STATUS = RUNTIME / "s4_b_hgb40_union_qd_status.json"
M0 = ROOT / "assets" / "solver" / "M0" / "tsp.best"
M0_SHA = "6826d7a71b30d882e4673a26ced162b7391157795931efe230f4d019292bba15"
CURRENT_FULL = ROOT / "runs" / "blade_v2_corrected_union_sanity_seed2026_v1"
FROZEN_C00 = ROOT / "runs" / "blade_v2_full_seed2026" / "cycles" / "C00"
S3B1 = ROOT / "audits" / "S3_B1_RF4_HGB4_label_budget_scaling_v1"
S3B2 = ROOT / "audits" / "S3_B2_RF_HGB_feature_label_scaling_v1"
S3D = ROOT / "audits" / "S3_D_final_independent_surrogate_validation_v1"
F40_MANIFEST = S3B2 / "09_feature_subset_manifests.json"
F40_HASHES = S3B2 / "10_feature_subset_hashes.json"
SCALER = ROOT / "assets" / "qd" / "reference_cloud" / "formal_raw_scaler.json"
R_SCRIPT = ROOT / "src" / "features" / "extract_salesperson_raw_features.R"
CENTROIDS = ROOT / "assets" / "qd" / "cvt" / "centroids.csv"
QD_NORM = ROOT / "assets" / "qd" / "normalization" / "normalization_A_P1P99.json"
ANCHOR_COORDS = ROOT / "assets" / "retention" / "uniform10k" / "fixed_uniform_anchor_10k_coordinates.npz"
ANCHOR_LABELS = ROOT / "assets" / "retention" / "uniform10k" / "fixed_uniform_anchor_10k_labels.parquet"
B0_SOURCE = CURRENT_FULL / "C00" / "07_oracle" / "B0"
C0_SOURCE = CURRENT_FULL / "C00" / "05_scoring" / "union"
HARD_N = 10_000
CALIBRATION_N = 10_000
EXPERIMENT_SEED = 2026
LR = 3.75e-5
EPOCHS = 256
NORM_COLS = [f"normalized_{name}" for name in QD4_FEATURE_NAMES]
F40 = json.loads(F40_MANIFEST.read_text(encoding="utf-8"))["F40"]
HGB_CONFIG_HASH = hashlib.sha256(json.dumps(HGB_CFG_CANONICAL, sort_keys=True).encode()).hexdigest()
KNOWN_FAILED_SOURCE_DIGEST = "23bdde60d505c01ad3d6c6a6de93bcda5f1f89ea42ee69d1649bf7254259688c"
NATIVE_MISSING_POLICY = "native_HGB_missing_routing_no_imputation"


CONFIG = {
    "schema_version": "S4_B_HGB40_UNION_QD_10CYCLE_M0_V1",
    "run_id": RUN_ID,
    "start_model": "M0", "cycles": list(range(10)), "final_model": "M10",
    "surrogate": {"estimator": "HistGradientBoostingRegressor", **HGB_CFG_CANONICAL,
                  "seeds": SEEDS, "aggregate": "median", "features": "F40",
                  "target": "log1p_LBQ", "calibration_target_n": CALIBRATION_N},
    "candidate": {"C00": "exact_corrected_unique_U0_union_O0",
                  "later": "unique_previous_P_union_current_O"},
    "selection": {"T": "HGB_Top10000", "P": "QD_centroid_nearest_plus_HGB_fill", "budget": HARD_N},
    "QD": {"descriptors": list(QD4_FEATURE_NAMES), "K": 1000, "normalization": "frozen_P1P99_clip"},
    "mutation": {"family": "FOGA11", "operators": list(OPERATOR_NAMES), "nominal_offspring": 110000},
    "training": {"hard_n": HARD_N, "uniform_anchor_n": HARD_N, "learning_rate": LR, "epochs": EPOCHS,
                 "optimizer": "fresh_Adam", "scheduler": None, "weight_decay": 0,
                 "gradient_clipping": None, "AMP": False, "seed": 2026, "checkpoint": "E256"},
    "evaluation": {"datasets": BENCHMARKS, "passive_only": True, "controls_training": False},
}
TRAINING_CONFIG_HASH = hashlib.sha256(
    json.dumps(CONFIG["training"], sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


class InvalidSource(RuntimeError):
    pass


class EventStatus:
    def __init__(self) -> None:
        self.value = {"run_id": RUN_ID, "pid": os.getpid(), "cycle": None, "phase": "initializing",
                      "latest_error": None, "recovery_state": "NONE", "timestamp": utc_now()}
        atomic_json(STATUS, self.value)

    def update(self, **values: Any) -> None:
        self.value.update(values); self.value["timestamp"] = utc_now(); atomic_json(STATUS, self.value)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha_list(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(map(str, values)) + "\n").encode()).hexdigest()


def cycle_dir(cycle: int) -> Path:
    return CYCLES_DIR / f"C{cycle:02d}"


def model_path(index: int) -> Path:
    return M0 if index == 0 else cycle_dir(index - 1) / "09_bq_training" / f"M{index}_E256.tsp.best"


base.FULL_DIR = RUN_DIR
base.GP_DIR = RUN_DIR
base._cycle_dir = lambda arm, cycle: cycle_dir(cycle)
base._model_path = lambda arm, index: model_path(index)


def load_model(index: int, device: torch.device):
    return load_canonical_checkpoint(model_path(index), device, expected_sha256=M0_SHA if index == 0 else None)


def source_files() -> list[Path]:
    paths = [
        M0, F40_MANIFEST, F40_HASHES, SCALER, R_SCRIPT, CENTROIDS, QD_NORM,
        ANCHOR_COORDS, ANCHOR_LABELS, B0_SOURCE / "labeled.parquet", B0_SOURCE / "labeled_bundle.npz",
        C0_SOURCE / "C_union.parquet", C0_SOURCE / "C_union.npz",
        CURRENT_FULL / "C00" / "03_mutation" / "offspring_coordinates.npz",
        CURRENT_FULL / "C00" / "03_mutation" / "mutation_manifest.parquet",
        CURRENT_FULL / "C00" / "04_qd" / "parent_qd.parquet",
        CURRENT_FULL / "C00" / "04_qd" / "offspring_qd.parquet",
        S3B1 / "STRICT_VERIFICATION.json", S3B2 / "STRICT_VERIFICATION.json", S3D / "STRICT_VERIFICATION.json",
        S3D / "24_surrogate_freeze_recommendation.md", ROOT / "src" / "mutation" / "__init__.py",
        ROOT / "src" / "final_run" / "science.py", ROOT / "src" / "bq" / "adaptation.py",
        ROOT / "scripts" / "external_v2_passive_blade_eval.py",
        ROOT / "scripts" / "run_s3b2_rf_hgb_feature_label.py", Path(__file__).resolve(),
        ROOT / "scripts" / "s4b_reporting.py",
    ]
    return paths


def fingerprints(paths: list[Path]) -> tuple[dict[str, str], str]:
    mapping: dict[str, str] = {}
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file(): raise FileNotFoundError(path)
        rel = path.relative_to(ROOT).as_posix(); value = sha256_file(path)
        mapping[rel] = value; digest.update(rel.encode()); digest.update(value.encode())
    return mapping, digest.hexdigest()


def free_gib() -> float:
    return shutil.disk_usage(RUN_DIR.parent if RUN_DIR.parent.exists() else ROOT).free / 2**30


def verify_sources() -> dict[str, Any]:
    if sha256_file(M0) != M0_SHA: raise InvalidSource("M0 hash mismatch")
    for audit in (S3B1, S3B2, S3D):
        strict = read_json(audit / "STRICT_VERIFICATION.json")
        if strict.get("all_required_true") is not True: raise InvalidSource(f"source audit not PASS: {audit.name}")
    freeze = (S3D / "24_surrogate_freeze_recommendation.md").read_text(encoding="utf-8")
    if "HGB" not in freeze or "40" not in freeze: raise InvalidSource("S3-D HGB40 freeze recommendation missing")
    fhash = read_json(F40_HASHES)["F40"]
    if len(F40) != 40 or len(set(F40)) != 40 or sha_list(F40) != fhash:
        raise InvalidSource("F40 manifest/hash mismatch")
    b0 = pd.read_parquet(B0_SOURCE / "labeled.parquet")
    bcoords, bhashes, btours = base._load_bundle(B0_SOURCE / "labeled_bundle.npz")
    if len(b0) != 500 or btours is None or bhashes != b0.instance_hash.astype(str).tolist():
        raise InvalidSource("initial B0/A0 alignment mismatch")
    ccoords, chashes, _ = base._load_bundle(C0_SOURCE / "C_union.npz")
    cframe = pd.read_parquet(C0_SOURCE / "C_union.parquet")
    if len(chashes) != 119781 or chashes != cframe.instance_hash.astype(str).tolist():
        raise InvalidSource("corrected C0 identity mismatch")
    if not Path(s3b2.rscript()).is_file(): raise InvalidSource("Rscript unavailable")
    mapping, digest = fingerprints(source_files())
    return {
        "status": "PASS", "source_fingerprints": mapping, "source_fingerprint_digest": digest,
        "M0_sha256": M0_SHA, "A0_n": 500, "A0_membership_sha256": membership_hash(bhashes),
        "C0_n": len(chashes), "C0_membership_sha256": membership_hash(chashes),
        "C0_bundle_sha256": sha256_file(C0_SOURCE / "C_union.npz"),
        "F40_sha256": fhash, "F40_names": F40, "HGB_config_sha256": HGB_CONFIG_HASH,
        "Rscript": s3b2.rscript(), "free_D_GiB": free_gib(),
    }


def source_unchanged(source: dict[str, Any]) -> bool:
    return fingerprints(source_files())[1] == source["source_fingerprint_digest"]


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists(): return
    try: os.link(source, target)
    except OSError: shutil.copy2(source, target)


def initialize(source: dict[str, Any], resume: bool) -> None:
    if not resume and RUN_DIR.exists(): raise FileExistsError(f"target exists: {RUN_DIR}; use --resume")
    if free_gib() < 7.5: raise RuntimeError(f"insufficient D: free space for formal launch: {free_gib():.2f} GiB")
    RUN_DIR.mkdir(parents=True, exist_ok=True); CYCLES_DIR.mkdir(exist_ok=True); STATE_DIR.mkdir(exist_ok=True); RUNTIME.mkdir(exist_ok=True)
    if LOCK.is_file():
        old = read_json(LOCK); command = base._pid_command(int(old.get("pid", -1)))
        if "run_s4b_hgb40_union_qd.py" in command: raise RuntimeError(f"active S4-B process: {old['pid']}")
        if not resume and old.get("state") not in {"FAILED", "COMPLETE"}: raise RuntimeError("stale lock requires --resume")
    protocol_hash = hashlib.sha256(json.dumps(CONFIG, sort_keys=True, default=str).encode()).hexdigest()
    frozen = RUN_DIR / "00_run_config.json"
    if frozen.is_file():
        frozen_config = read_json(frozen)
        if frozen_config["protocol_sha256"] != protocol_hash:
            raise RuntimeError("resume protocol mismatch")
        if frozen_config["source_fingerprint_digest"] != source["source_fingerprint_digest"]:
            lock = read_json(LOCK) if LOCK.is_file() else {}
            eligible = (
                resume
                and frozen_config["source_fingerprint_digest"] == KNOWN_FAILED_SOURCE_DIGEST
                and lock.get("state") == "FAILED"
                and "spn_nng_3_strong_components" in str(lock.get("error", ""))
            )
            if not eligible:
                raise RuntimeError("resume source mismatch")
            prior_source = read_json(RUN_DIR / "02_source_integrity.json")
            recovery = {
                "recovery_id": "RECOVERY_001_NATIVE_HGB_MISSING",
                "old_source_fingerprint_digest": frozen_config["source_fingerprint_digest"],
                "new_source_fingerprint_digest": source["source_fingerprint_digest"],
                "failed_error": lock.get("error"),
                "affected_instance_hash": "a0f3320130e4eb49bb153d7ac846d730a4ce674bcc93ed106a18cb61b20c3635",
                "affected_features": ["spn_nng_3_strong_components_sd", "spn_nng_3_strong_components_skew", "spn_nng_3_strong_components_varcoeff"],
                "policy": NATIVE_MISSING_POLICY, "imputation": False, "F40_changed": False,
                "HGB_config_changed": False, "prior_source_fingerprints": prior_source["source_fingerprints"],
                "new_source_fingerprints": source["source_fingerprints"], "timestamp": utc_now(),
            }
            atomic_json(RUN_DIR / "SOURCE_RECOVERY_001_NATIVE_HGB_MISSING.json", recovery)
            frozen_config["source_fingerprint_digest"] = source["source_fingerprint_digest"]
            frozen_config["audited_recovery"] = recovery["recovery_id"]
            atomic_json(frozen, frozen_config)
    else:
        atomic_json(frozen, {**CONFIG, "protocol_sha256": protocol_hash,
                             "source_fingerprint_digest": source["source_fingerprint_digest"], "created_utc": utc_now()})
    atomic_json(RUN_DIR / "02_source_integrity.json", source)
    protocol = """# S4-B frozen HGB40 Union + QD rollout\n\nExact M0; cycles C00-C09; branch-local A0=B0(500); current-solver hardness refresh; frozen F40/HGB; five-seed median; exact corrected C0; later unique(P union O); HGB Top10k T; frozen QD representative + HGB fill P; fixed FOGA-11; hard10k + fixed Uniform10k BQ at LR 3.75e-5 for E256; all external benchmarks passive. Historical GP output is descriptive only.\n"""
    (RUN_DIR / "01_protocol.md").write_text(protocol, encoding="utf-8")
    atomic_json(LOCK, {"pid": os.getpid(), "host": socket.gethostname(), "state": "ACTIVE",
                       "timestamp": utc_now(), "command": " ".join(sys.argv)})

    model, meta = load_model(0, torch.device("cpu"))
    m0_manifest = {"model": "M0", "checkpoint": str(M0), "sha256": meta["sha256"], "parent_sha256": None,
                   "architecture": model.__class__.__name__, "parameter_count": int(sum(p.numel() for p in model.parameters())),
                   "inference_config": "frozen external_v2 greedy decoding", "cycle": None}
    atomic_json(RUN_DIR / "03_M0_manifest.json", m0_manifest); del model

    a0_dir = STATE_DIR / "A0"; a0_dir.mkdir(exist_ok=True)
    link_or_copy(B0_SOURCE / "labeled_bundle.npz", a0_dir / "A0.npz")
    if not (a0_dir / "A0.parquet").is_file():
        frame = pd.read_parquet(B0_SOURCE / "labeled.parquet").copy()
        frame["first_cycle_observed"] = -1; frame["archive_provenance"] = "M0_INITIAL_B0"
        frame.to_parquet(a0_dir / "A0.parquet", index=False)
    atomic_json(RUN_DIR / "04_initial_archive_manifest.json", {
        "definition": "exact Oracle-labeled initial B0 available at M0 before evolutionary acquisition",
        "n": 500, "membership_sha256": source["A0_membership_sha256"],
        "bundle_sha256": sha256_file(a0_dir / "A0.npz"), "no_future_trajectory_data": True,
    })
    atomic_json(RUN_DIR / "05_F40_manifest.json", {"names": F40, "ordered_sha256": source["F40_sha256"],
                 "source": str(F40_MANIFEST), "preprocessing": "salesperson_1.0.0_frozen_arcsinh", "ranking_recomputed": False})
    atomic_json(RUN_DIR / "06_QD_manifest.json", {"descriptors": list(QD4_FEATURE_NAMES), "K": 1000,
                 "centroids": str(CENTROIDS), "centroids_sha256": sha256_file(CENTROIDS),
                 "normalization": str(QD_NORM), "normalization_sha256": sha256_file(QD_NORM), "clipping": True})
    atomic_json(RUN_DIR / "07_mutation_manifest.json", {"family": "FOGA11", "operators": list(OPERATOR_NAMES),
                 "operator_count": len(OPERATOR_NAMES), "nominal_offspring": 110000,
                 "implementation_sha256": sha256_file(ROOT / "src" / "mutation" / "__init__.py")})
    with np.load(ANCHOR_COORDS, allow_pickle=False) as payload:
        anchor_hashes = [str(x) for x in payload["coordinate_hash"].tolist()]
    atomic_json(RUN_DIR / "08_uniform_anchor_manifest.json", {"n": len(anchor_hashes),
                 "membership_sha256": membership_hash(anchor_hashes), "coordinates_sha256": sha256_file(ANCHOR_COORDS),
                 "labels_sha256": sha256_file(ANCHOR_LABELS), "fixed_across_cycles": True})


def stage_seconds(directory: Path) -> float:
    done = directory / "STAGE_DONE.json"
    return float(read_json(done)["wall_seconds"]) if done.is_file() else 0.0


def load_feature_cache() -> pd.DataFrame:
    directory = STATE_DIR / "feature_cache"; directory.mkdir(parents=True, exist_ok=True)
    files = sorted(directory.glob("cache_*.parquet"))
    if not files: return pd.DataFrame(columns=["instance_hash", *F40])
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    frame["instance_hash"] = frame.instance_hash.astype(str)
    if frame.instance_hash.duplicated().any(): raise RuntimeError("feature cache duplicate hash")
    if np.isinf(frame[F40].to_numpy(float)).any(): raise RuntimeError("feature cache contains infinite values")
    return frame


def extract_f40_shard(payload: dict[str, Any]) -> str:
    """Frozen S3-B2 extraction with explicit native-HGB NaN handling."""
    out = Path(payload["out"])
    coords = np.asarray(payload["coordinates"], dtype=np.float64)
    hashes = list(payload["hashes"])
    names = list(payload["names"])
    # Feature work directories are shared across ablation arms and the shard
    # filenames encode only positional ranges.  A file left by another arm is
    # reusable only when its ordered instance hashes match this payload.
    if out.is_file():
        try:
            cached_hashes = pd.read_parquet(out, columns=["instance_hash"]).instance_hash.astype(str).tolist()
        except Exception:
            cached_hashes = []
        if cached_hashes == hashes:
            return str(out)
    with s3b2.tempfile.TemporaryDirectory(prefix="s4b_f40_") as temporary:
        work = Path(temporary)
        inp, raw, schema = work / "coordinates.csv", work / "features.csv", work / "schema.csv"
        with inp.open("w", newline="", encoding="utf-8") as handle:
            writer = s3b2.csv.writer(handle, lineterminator="\n")
            writer.writerow(["state_id", "node_idx", "x", "y"])
            for instance_hash, points in zip(hashes, coords, strict=True):
                for node, (x, y) in enumerate(points):
                    writer.writerow([instance_hash, node, repr(float(x)), repr(float(y))])
        process = s3b2.subprocess.run(
            [payload["rscript"], str(R_SCRIPT), str(inp), str(raw), str(schema)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900,
        )
        if process.returncode or not raw.is_file():
            raise RuntimeError(f"F40 extraction failed: {process.stderr[-1500:]}")
        data = pd.read_csv(raw)
        if data.state_id.astype(str).tolist() != hashes: raise RuntimeError("F40 alignment failure")
        if "feature_error" in data.columns and data.feature_error.notna().any():
            raise RuntimeError("F40 feature_error reported")
        missing = [name for name in names if name not in data.columns]
        if missing: raise RuntimeError(f"F40 missing columns {missing[:8]}")
        frame = data[["state_id", *names]].rename(columns={"state_id": "instance_hash"})
        params = payload["params"]
        for name in names:
            center, scale = params[name]
            frame[name] = np.arcsinh((frame[name].to_numpy(float) - center) / scale)
        values = frame[names].to_numpy(float)
        if np.isinf(values).any():
            bad = np.isinf(values)
            counts = {name: int(bad[:, i].sum()) for i, name in enumerate(names) if bad[:, i].any()}
            raise RuntimeError(f"infinite F40 features {counts}")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp, index=False)
    tmp.replace(out)
    return str(out)


def _run_one_f40_shard(job: dict[str, Any]) -> str:
    """Run one shard in a fresh interpreter so a wedged worker can be killed."""
    worker = Path(__file__).resolve().parent / "f40_one_shard.py"
    last_error = "unknown"
    for attempt in range(3):
        proc = subprocess.Popen(
            [sys.executable, str(worker)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            out, err = proc.communicate(pickle.dumps(job), timeout=900)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            last_error = f"timed out: {job['out']}"
            continue
        if proc.returncode == 0:
            return out.decode("utf-8", errors="replace").strip()
        last_error = f"rc={proc.returncode} {job['out']}: {err.decode('utf-8', errors='replace')[-1500:]}"
        if proc.returncode > 0:
            break
    raise RuntimeError(f"F40 shard failed {last_error}")


def _run_f40_pool(jobs: list[dict[str, Any]], tag: str) -> None:
    """Extract shards with separate processes and only four running at once.

    A process-pool queue inside the training process deadlocks: after a worker
    exits, the parent still holds both pipe ends and blocks forever in
    pipe_write. Independent subprocesses are killed on timeout instead.
    """
    if not jobs:
        return
    total = len(jobs)
    done = 0
    with ThreadPoolExecutor(max_workers=min(4, total)) as executor:
        futures = [executor.submit(_run_one_f40_shard, job) for job in jobs]
        for future in as_completed(futures):
            future.result()
            done += 1
            if done == 1 or done == total or done % 25 == 0:
                print(f"[S4-B] F40 {tag} shards {done}/{total}", flush=True)


def ensure_features(coordinates: np.ndarray, hashes: list[str], tag: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter(); cache = load_feature_cache(); known = set(cache.instance_hash.astype(str))
    missing_indices = [i for i, value in enumerate(hashes) if value not in known]
    new_frame = pd.DataFrame(columns=["instance_hash", *F40])
    if missing_indices:
        work = STATE_DIR / "feature_work" / tag; work.mkdir(parents=True, exist_ok=True)
        params = s3b2.scaler_params(); jobs = []
        for start in range(0, len(missing_indices), 250):
            block = missing_indices[start:start + 250]
            jobs.append({"coordinates": coordinates[np.asarray(block, int)], "hashes": [hashes[i] for i in block],
                         "names": F40, "out": str(work / f"shard_{start:07d}_{start+len(block):07d}.parquet"),
                         "rscript": s3b2.rscript(), "r_script": str(R_SCRIPT), "params": params})
        _run_f40_pool(jobs, tag)
        new_frame = pd.concat([pd.read_parquet(job["out"]) for job in jobs], ignore_index=True)
        expected = [hashes[i] for i in missing_indices]
        if new_frame.instance_hash.astype(str).tolist() != expected: raise RuntimeError("F40 extraction alignment failure")
        shard = STATE_DIR / "feature_cache" / f"cache_{tag}_{membership_hash(expected)[:12]}.parquet"
        new_frame.to_parquet(shard, index=False)
        resolved = work.resolve(); root = RUN_DIR.resolve()
        if str(resolved).startswith(str(root)): shutil.rmtree(resolved)
        cache = pd.concat([cache, new_frame], ignore_index=True)
    lookup = cache.set_index("instance_hash")
    missing = [h for h in hashes if h not in lookup.index]
    if missing: raise RuntimeError(f"feature cache missing {len(missing)} hashes")
    result = lookup.loc[hashes].reset_index()[["instance_hash", *F40]]
    values = result[F40].to_numpy(float)
    if np.isinf(values).any(): raise RuntimeError("candidate F40 contains infinite values")
    nan_mask = np.isnan(values)
    missing_by_feature = {name: int(nan_mask[:, i].sum()) for i, name in enumerate(F40) if nan_mask[:, i].any()}
    info = {"requested_n": len(hashes), "cache_hits": len(hashes) - len(missing_indices),
            "newly_extracted": len(missing_indices), "wall_seconds": time.perf_counter() - started,
            "cache_size": len(cache), "hash_safe": not cache.instance_hash.duplicated().any(),
            "native_missing_values": int(nan_mask.sum()),
            "native_missing_instances": int(nan_mask.any(axis=1).sum()),
            "missing_by_feature": missing_by_feature,
            "missing_policy": NATIVE_MISSING_POLICY}
    atomic_json(STATE_DIR / "feature_cache_manifest.json", {**info,
                "shards": {p.name: sha256_file(p) for p in sorted((STATE_DIR / "feature_cache").glob("cache_*.parquet"))}})
    return result, info


def archive_pre(cycle: int) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    if cycle == 0:
        coords, hashes, tours = base._load_bundle(STATE_DIR / "A0" / "A0.npz")
        meta = pd.read_parquet(STATE_DIR / "A0" / "A0.parquet")
    else:
        coords, hashes, tours = base._load_bundle(cycle_dir(cycle - 1) / "08_archive" / "A_next.npz")
        meta = pd.read_parquet(cycle_dir(cycle - 1) / "08_archive" / "A_next.parquet")
    if tours is None or hashes != meta.instance_hash.astype(str).tolist(): raise RuntimeError("archive pre alignment")
    return coords, meta, tours


def parent_pre(cycle: int) -> tuple[np.ndarray, pd.DataFrame]:
    if cycle == 0:
        coords, hashes, _ = base._load_bundle(CURRENT_FULL / "C00" / "01_inputs" / "parents.npz")
        frame = pd.read_parquet(CURRENT_FULL / "C00" / "04_qd" / "parent_qd.parquet")
        frame["candidate_age"] = 0
    else:
        coords, hashes, _ = base._load_bundle(cycle_dir(cycle - 1) / "06_selection" / "P_next.npz")
        frame = pd.read_parquet(cycle_dir(cycle - 1) / "06_selection" / "P_next.parquet")
    if len(hashes) != 10000 or hashes != frame.instance_hash.astype(str).tolist(): raise RuntimeError("parent alignment")
    return coords, frame


def prepare_archive_input(cycle: int, coords: np.ndarray, meta: pd.DataFrame, tours: np.ndarray) -> None:
    directory = cycle_dir(cycle) / "01_inputs"; directory.mkdir(parents=True, exist_ok=True)
    base._save_bundle(directory / "archive_in.npz", coords, meta.instance_hash.astype(str).tolist(), tours)
    meta.to_parquet(directory / "archive_in.parquet", index=False)
    atomic_json(cycle_dir(cycle) / "A_archive_pre_manifest.json", {
        "cycle": f"C{cycle:02d}", "archive_n": len(meta),
        "membership_sha256": membership_hash(meta.instance_hash.astype(str).tolist()),
        "bundle_sha256": sha256_file(directory / "archive_in.npz"), "branch_local": True})


def calibration_stage(cycle: int, model: torch.nn.Module, model_hash: str, device: torch.device,
                      archive_coords: np.ndarray, archive_meta: pd.DataFrame, archive_tours: np.ndarray,
                      resume: bool) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    cdir = cycle_dir(cycle); stage = cdir / "_stages" / "01_calibration"
    archive_hashes = archive_meta.instance_hash.astype(str).tolist()
    sample_seed = EXPERIMENT_SEED * 100 + cycle
    inputs = {"archive": membership_hash(archive_hashes), "model": model_hash,
              "seed": str(sample_seed), "target_n": str(CALIBRATION_N)}
    with atomic_stage(stage, "branch_archive_sample_and_current_hardness_refresh", inputs, resume=resume) as outputs:
        if outputs is not None:
            order = np.argsort(np.asarray(archive_hashes), kind="stable")
            if len(order) <= CALIBRATION_N: chosen = order
            else: chosen = order[np.random.default_rng(sample_seed).choice(len(order), CALIBRATION_N, replace=False)]
            hashes = [archive_hashes[int(i)] for i in chosen]
            membership = pd.DataFrame({"draw_order": np.arange(len(chosen)), "instance_hash": hashes,
                                       "sample_seed": sample_seed, "archive_n": len(archive_meta)})
            membership.to_csv(cdir / "B_calibration_membership.csv", index=False)
            bcoords, btours = archive_coords[chosen], archive_tours[chosen]
            refresh_started = time.perf_counter(); values = hardness(model, bcoords, btours, device)
            refresh_s = time.perf_counter() - refresh_started
            features, feature_info = ensure_features(bcoords, hashes, f"C{cycle:02d}_B")
            frame = membership.copy(); frame["current_LBQ"] = values; frame["y_log1p"] = np.log1p(values)
            frame.to_parquet(cdir / "B_current_hardness.parquet", index=False)
            features.to_parquet(cdir / "_state_B_features.parquet", index=False)
            info = {"archive_n": len(archive_meta), "calibration_n": len(frame), "sample_seed": sample_seed,
                    "membership_sha256": membership_hash(hashes), "refresh_wall_seconds": refresh_s,
                    "refresh_throughput_per_s": len(frame) / max(refresh_s, 1e-12),
                    "hardness": summary_stats(values), "feature_cache": feature_info,
                    "policy": "all_archive" if len(archive_meta) <= CALIBRATION_N else "canonical_sort_uniform_random10k"}
            atomic_json(cdir / "_calibration_info.json", info)
            outputs.extend([cdir / "B_calibration_membership.csv", cdir / "B_current_hardness.parquet",
                            cdir / "_state_B_features.parquet", cdir / "_calibration_info.json"])
    return (pd.read_parquet(cdir / "B_current_hardness.parquet"), read_json(cdir / "_calibration_info.json"),
            read_json(cdir / "_calibration_info.json")["feature_cache"])


def fit_hgb_stage(cycle: int, b: pd.DataFrame, source: dict[str, Any], resume: bool) -> dict[str, Any]:
    cdir = cycle_dir(cycle); stage = cdir / "_stages" / "02_hgb_fit"; model_dir = cdir / "HGB_seed_models"
    inputs = {"B": membership_hash(b.instance_hash.astype(str).tolist()), "F40": source["F40_sha256"],
              "config": source["HGB_config_sha256"]}
    with atomic_stage(stage, "fit_five_frozen_HGB40_models", inputs, resume=resume) as outputs:
        if outputs is not None:
            features = pd.read_parquet(cdir / "_state_B_features.parquet")
            if features.instance_hash.astype(str).tolist() != b.instance_hash.astype(str).tolist(): raise RuntimeError("B F40 alignment")
            x, y = features[F40].to_numpy(float), b.y_log1p.to_numpy(float)
            model_dir.mkdir(parents=True, exist_ok=True); rows = []; predictions = []
            for seed in SEEDS:
                started = time.perf_counter(); estimator = HistGradientBoostingRegressor(**hgb_params(seed)).fit(x, y)
                pred = estimator.predict(x); path = model_dir / f"HGB40_seed_{seed}.joblib"
                joblib.dump({"model": estimator, "features": F40, "target": "log1p_LBQ",
                             "params": hgb_params(seed), "seed": seed}, path, compress=3)
                rows.append({"seed": seed, "fit_s": time.perf_counter() - started, "path": str(path),
                             "sha256": sha256_file(path), "train_mae": float(np.mean(np.abs(pred - y))),
                             "train_spearman": float(spearmanr(pred, y).statistic)})
                predictions.append(pred); outputs.append(path)
            median = median_stack(np.vstack(predictions))
            manifest = {"estimator": "HistGradientBoostingRegressor", "params": HGB_CFG_CANONICAL,
                        "config_sha256": source["HGB_config_sha256"], "seeds": SEEDS, "aggregate": "median",
                        "target": "log1p_LBQ", "features": F40, "F40_sha256": source["F40_sha256"],
                        "n_train": len(b), "seed_models": rows, "fit_total_s": sum(r["fit_s"] for r in rows),
                        "median_train_mae": float(np.mean(np.abs(median - y))),
                        "median_train_spearman": float(spearmanr(median, y).statistic),
                        "ranking_recomputed": False, "operator_feature": False, "retuned": False,
                        "cross_family_ensemble": False}
            atomic_json(cdir / "HGB_model_manifest.json", manifest); outputs.append(cdir / "HGB_model_manifest.json")
    return read_json(cdir / "HGB_model_manifest.json")


def load_hgb_models(cycle: int) -> list[Any]:
    return [joblib.load(cycle_dir(cycle) / "HGB_seed_models" / f"HGB40_seed_{seed}.joblib")["model"] for seed in SEEDS]


def prepare_candidate(cycle: int, resume: bool) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cdir = cycle_dir(cycle); stage = cdir / "_stages" / "03_candidate_union"
    parent_coords, parent = parent_pre(cycle); parent_hashes = parent.instance_hash.astype(str).tolist()
    if cycle == 0:
        o_source = CURRENT_FULL / "C00" / "03_mutation"
        for name in ("offspring_coordinates.npz", "mutation_manifest.parquet", "CANONICAL_MUTATION_RUNTIME.json"):
            link_or_copy(o_source / name, cdir / "03_mutation" / name)
    offspring_coords, offspring_hashes, _ = base._load_bundle(cdir / "03_mutation" / "offspring_coordinates.npz")
    manifest = pd.read_parquet(cdir / "03_mutation" / "mutation_manifest.parquet")
    inputs = {"parents": membership_hash(parent_hashes), "offspring": membership_hash(offspring_hashes),
              "policy": "exact_corrected_C0" if cycle == 0 else "unique_parent_offspring_union"}
    with atomic_stage(stage, "construct_exact_parent_offspring_union", inputs, resume=resume) as outputs:
        if outputs is not None:
            qdir = cdir / "04_qd"; udir = cdir / "05_scoring" / "union"; qdir.mkdir(parents=True, exist_ok=True); udir.mkdir(parents=True, exist_ok=True)
            if cycle == 0:
                for name in ("parent_qd.parquet", "offspring_qd.parquet"):
                    link_or_copy(CURRENT_FULL / "C00" / "04_qd" / name, qdir / name)
                link_or_copy(C0_SOURCE / "C_union.npz", udir / "C_union.npz")
                link_or_copy(C0_SOURCE / "C_union.parquet", udir / "C_union.parquet")
                coordinates, hashes, _ = base._load_bundle(udir / "C_union.npz"); union = pd.read_parquet(udir / "C_union.parquet")
                parent_qd = pd.read_parquet(qdir / "parent_qd.parquet"); offspring_qd = pd.read_parquet(qdir / "offspring_qd.parquet")
            else:
                parent_qd = parent[["instance_hash", "niche_id", *QD4_FEATURE_NAMES, *NORM_COLS, "candidate_age"]].copy()
                offspring_qd = qd_frame(offspring_coords, offspring_hashes)
                accepted = manifest.valid_after_repair & manifest.proposal_status.eq("accepted") & ~manifest.duplicate
                lineage = manifest.loc[accepted, ["parent_hash", "operator_id"]].reset_index(drop=True)
                if len(lineage) != len(offspring_qd): raise RuntimeError("mutation/QD alignment")
                offspring_qd["parent_hash"] = lineage.parent_hash; offspring_qd["operator_id"] = lineage.operator_id
                offspring_qd["candidate_age"] = 0; offspring_qd["source"] = "offspring"
                parent_qd.to_parquet(qdir / "parent_qd.parquet", index=False); offspring_qd.to_parquet(qdir / "offspring_qd.parquet", index=False)
                p = parent_qd.copy(); o = offspring_qd.copy(); p["coordinate_origin"] = "previous_parent"; o["coordinate_origin"] = "new_offspring"
                p["origin_index"] = np.arange(len(p)); o["origin_index"] = np.arange(len(o))
                union = pd.concat([p, o], ignore_index=True, sort=False).drop_duplicates("instance_hash", keep="first").reset_index(drop=True)
                pset, oset = set(parent_hashes), set(offspring_hashes)
                union["is_previous_parent"] = union.instance_hash.astype(str).isin(pset)
                union["is_new_offspring"] = union.instance_hash.astype(str).isin(oset)
                ages = dict(zip(parent_hashes, parent_qd.candidate_age.astype(int), strict=True))
                union["candidate_age"] = [ages[h] + 1 if h in ages else 0 for h in union.instance_hash.astype(str)]
                coordinates = np.empty((len(union), 100, 2), dtype=np.float64)
                pmask = union.coordinate_origin.eq("previous_parent").to_numpy()
                coordinates[pmask] = parent_coords[union.loc[pmask, "origin_index"].to_numpy(int)]
                coordinates[~pmask] = offspring_coords[union.loc[~pmask, "origin_index"].to_numpy(int)]
                union["coord_index"] = np.arange(len(union)); hashes = union.instance_hash.astype(str).tolist()
                union.to_parquet(udir / "C_union.parquet", index=False); base._save_bundle(udir / "C_union.npz", coordinates, hashes)
            if hashes != union.instance_hash.astype(str).tolist() or len(hashes) != len(set(hashes)): raise RuntimeError("union identity")
            union.to_csv(cdir / "candidate_manifest.csv", index=False)
            union[["instance_hash", "niche_id", *QD4_FEATURE_NAMES, *NORM_COLS]].to_csv(cdir / "QD_assignment.csv", index=False)
            accounting = {"parent_n": len(parent_hashes), "offspring_n": len(offspring_hashes), "candidate_n": len(union),
                          "overlap_n": len(set(parent_hashes) & set(offspring_hashes)),
                          "duplicate_removal_n": len(parent_hashes) + len(offspring_hashes) - len(union)}
            atomic_json(cdir / "_candidate_accounting.json", accounting)
            outputs.extend([udir / "C_union.npz", udir / "C_union.parquet", qdir / "parent_qd.parquet", qdir / "offspring_qd.parquet",
                            cdir / "candidate_manifest.csv", cdir / "QD_assignment.csv", cdir / "_candidate_accounting.json"])
    coords, hashes, _ = base._load_bundle(cdir / "05_scoring" / "union" / "C_union.npz")
    union = pd.read_parquet(cdir / "05_scoring" / "union" / "C_union.parquet")
    return coords, union, pd.read_parquet(cdir / "04_qd" / "parent_qd.parquet"), pd.read_parquet(cdir / "04_qd" / "offspring_qd.parquet"), read_json(cdir / "_candidate_accounting.json")


def score_stage(cycle: int, coords: np.ndarray, union: pd.DataFrame, source: dict[str, Any], resume: bool) -> tuple[np.ndarray, dict[str, Any]]:
    cdir = cycle_dir(cycle); stage = cdir / "_stages" / "04_features_scores"
    hashes = union.instance_hash.astype(str).tolist(); inputs = {"candidate": membership_hash(hashes), "F40": source["F40_sha256"], "HGB": sha256_file(cdir / "HGB_model_manifest.json")}
    with atomic_stage(stage, "F40_cache_and_five_seed_full_pool_scoring", inputs, resume=resume) as outputs:
        if outputs is not None:
            features, cache_info = ensure_features(coords, hashes, f"C{cycle:02d}_C")
            features.to_parquet(cdir / "candidate_features.parquet", index=False)
            x = features[F40].to_numpy(float); rows = {"instance_hash": hashes}; preds = []
            score_started = time.perf_counter()
            for seed, model in zip(SEEDS, load_hgb_models(cycle), strict=True):
                value = np.asarray(model.predict(x), float); rows[f"seed_{seed}"] = value; preds.append(value)
            median = median_stack(np.vstack(preds)); score_s = time.perf_counter() - score_started
            rows["hgb_predicted_log1p_hardness"] = median
            pd.DataFrame(rows).to_parquet(cdir / "candidate_predictions.parquet", index=False)
            order = np.lexsort((np.asarray(hashes), -median)); ranks = np.empty(len(hashes), int); ranks[order] = np.arange(1, len(hashes) + 1)
            percentile = (len(hashes) - ranks) / max(1, len(hashes) - 1)
            pd.DataFrame({"instance_hash": hashes, "rank": ranks, "percentile_high": percentile}).to_parquet(cdir / "candidate_ranks.parquet", index=False)
            atomic_json(cdir / "_score_info.json", {"score_wall_seconds": score_s, "feature_cache": cache_info,
                        "prediction_finite": bool(np.isfinite(median).all()), "tie_break": "instance_hash_ascending"})
            outputs.extend([cdir / "candidate_features.parquet", cdir / "candidate_predictions.parquet",
                            cdir / "candidate_ranks.parquet", cdir / "_score_info.json"])
    pred = pd.read_parquet(cdir / "candidate_predictions.parquet")
    if pred.instance_hash.astype(str).tolist() != hashes: raise RuntimeError("prediction alignment")
    return pred.hgb_predicted_log1p_hardness.to_numpy(float), read_json(cdir / "_score_info.json")


def selection_stage(cycle: int, coords: np.ndarray, union: pd.DataFrame, scores: np.ndarray, resume: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cdir = cycle_dir(cycle); stage = cdir / "_stages" / "05_selection"; directory = cdir / "06_selection"
    inputs = {"candidate": membership_hash(union.instance_hash.astype(str).tolist()),
              "scores": hashlib.sha256(np.ascontiguousarray(scores).tobytes()).hexdigest()}
    with atomic_stage(stage, "freeze_HGB_Top10k_and_QD_HGB_parents", inputs, resume=resume) as outputs:
        if outputs is not None:
            directory.mkdir(parents=True, exist_ok=True)
            t = base._global_topk(union, scores, HARD_N).rename(columns={"gp_predicted_log1p_hardness": "hgb_predicted_log1p_hardness"})
            t["component"] = "global_hgb"
            p = select_population(union, scores, HARD_N, f"HGB40_C{cycle:02d}").rename(columns={"gp_predicted_log1p_hardness": "hgb_predicted_log1p_hardness"})
            p["component"] = p.component.replace({"gp_fill": "hgb_fill"})
            t.to_parquet(directory / "T_selected.parquet", index=False); p.to_parquet(directory / "P_next.parquet", index=False)
            base._population_bundle(coords, t, directory / "T_selected.npz"); base._population_bundle(coords, p, directory / "P_next.npz")
            t[["instance_hash", "component", "hgb_predicted_log1p_hardness"]].to_csv(cdir / "T_hard_top10k_membership.csv", index=False)
            p[["instance_hash", "component", "niche_id", "centroid_distance", "hgb_predicted_log1p_hardness", "candidate_age"]].to_csv(cdir / "P_parent_membership.csv", index=False)
            qn = int(p.component.eq("centroid_floor").sum()); gn = int(p.component.eq("hgb_fill").sum())
            atomic_json(cdir / "_selection_info.json", {"T_n": len(t), "P_n": len(p), "Q_n": qn, "G_n": gn,
                        "T_membership_sha256": membership_hash(t.instance_hash.astype(str).tolist()),
                        "P_membership_sha256": membership_hash(p.instance_hash.astype(str).tolist()),
                        "T_frozen_before_oracle": True})
            outputs.extend([directory / "T_selected.parquet", directory / "P_next.parquet", directory / "T_selected.npz", directory / "P_next.npz",
                            cdir / "T_hard_top10k_membership.csv", cdir / "P_parent_membership.csv", cdir / "_selection_info.json"])
    return pd.read_parquet(directory / "T_selected.parquet"), pd.read_parquet(directory / "P_next.parquet"), read_json(cdir / "_selection_info.json")


def oracle_archive_stage(cycle: int, model: torch.nn.Module, device: torch.device, union_coords: np.ndarray,
                         selected_t: pd.DataFrame, archive_coords: np.ndarray, archive_meta: pd.DataFrame,
                         archive_tours: np.ndarray, resume: bool) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any], pd.DataFrame]:
    cdir = cycle_dir(cycle); t_indices = selected_t.coord_index.to_numpy(int); tcoords = union_coords[t_indices]
    labeled, tours, oracle = base._label_population(cdir / "07_oracle" / "T", selected_t, tcoords, model, device, archive_meta,
                                                     "S4B_T_frozen_then_oracle_and_current_hardness", resume)
    # Keep the atomic oracle-stage artifact byte-for-byte unchanged so its
    # completion manifest remains valid on resume. Provenance belongs only in
    # the branch-local archive copy assembled below.
    labeled_for_archive = labeled.copy()
    labeled_for_archive["first_cycle_observed"] = cycle
    labeled_for_archive["archive_provenance"] = f"S4B_T_C{cycle:02d}"
    ledger = pd.read_parquet(cdir / "07_oracle" / "T" / "ledger.parquet")
    ledger.to_csv(cdir / "selected_expert_tour_accounting.csv", index=False)
    exact = labeled[["instance_hash", "current_hardness", "hgb_predicted_log1p_hardness"]].copy()
    exact.to_parquet(cdir / "selected_exact_hardness.parquet", index=False)
    work = cdir / "07_oracle" / "T" / "work"
    if work.exists() and str(work.resolve()).startswith(str(RUN_DIR.resolve())): shutil.rmtree(work)
    anew_coords, anew_meta, anew_tours = base._save_archive(cdir / "08_archive", archive_coords, archive_meta,
                                                             tcoords, labeled_for_archive, tours, resume)
    atomic_json(cdir / "archive_post_manifest.json", {"archive_n": len(anew_meta),
                "membership_sha256": membership_hash(anew_meta.instance_hash.astype(str).tolist()),
                "new_unique": len(anew_meta) - len(archive_meta), "branch_local": True,
                "bundle_sha256": sha256_file(cdir / "08_archive" / "A_next.npz")})
    return labeled, tours, oracle, anew_meta


def evaluate_external(index: int, model: torch.nn.Module, device: torch.device, output: Path, status: EventStatus) -> tuple[pd.DataFrame, float]:
    checkpoint_hash = sha256_file(model_path(index)); rows = []; started = time.perf_counter(); work = output.parent / "evaluation_work" / f"M{index}"
    for dataset in BENCHMARKS:
        status.update(cycle=f"M{index}", phase=f"passive_eval_{dataset}", latest_checkpoint=str(model_path(index)))
        directory = work / dataset; summary_path = directory / "summary.json"
        if summary_path.is_file() and read_json(summary_path).get("checkpoint_sha256") == checkpoint_hash:
            rows.append(read_json(summary_path)); continue
        directory.mkdir(parents=True, exist_ok=True); payload = evaluator.load_fixed_references(dataset)
        if dataset == "OriginalTSPLIB70": detail, result = evaluator.evaluate_original70(model, device, payload, checkpoint_hash)
        else: detail, result = evaluator.evaluate_tensor_dataset(model, device, dataset, payload, checkpoint_hash)
        result.update({"model": f"M{index}", "dataset": dataset, "checkpoint_sha256": checkpoint_hash,
                       "passive_only": True, "controls_training": False})
        detail.insert(0, "model", f"M{index}"); detail.insert(1, "dataset", dataset)
        detail.to_parquet(directory / "instance_metrics.parquet", index=False); atomic_json(summary_path, result); rows.append(result)
    frame = pd.DataFrame(rows); frame.to_csv(output, index=False); return frame, time.perf_counter() - started


def evaluate_anchor(index: int, model: torch.nn.Module, device: torch.device, output: Path) -> tuple[pd.DataFrame, float]:
    started = time.perf_counter(); coords, tours = load_anchor(); values = hardness(model, coords, tours, device)
    anchor_dir = STATE_DIR / "anchor"; anchor_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame({"index": np.arange(len(values)), "LBQ": values}); detail.to_parquet(anchor_dir / f"M{index}.parquet", index=False)
    base_values = pd.read_parquet(anchor_dir / "M0.parquet").LBQ.to_numpy(float) if index else values
    prev_values = pd.read_parquet(anchor_dir / f"M{index-1}.parquet").LBQ.to_numpy(float) if index else values
    row = {"model": f"M{index}", "mean_LBQ": float(values.mean()), "median_LBQ": float(np.median(values)),
           "P90_LBQ": float(np.percentile(values, 90)), "delta_mean_vs_M0": float(values.mean() - base_values.mean()),
           "delta_mean_vs_previous": float(values.mean() - prev_values.mean()),
           "fraction_improved_vs_previous": float(np.mean(values < prev_values)),
           "fraction_worsened_vs_previous": float(np.mean(values > prev_values)),
           "anchor_sha256": sha256_file(ANCHOR_COORDS), "passive_only": True}
    frame = pd.DataFrame([row]); frame.to_csv(output, index=False); return frame, time.perf_counter() - started


def baseline_evaluation(device: torch.device, status: EventStatus, resume: bool) -> None:
    stage = STATE_DIR / "stages" / "M0_baseline"; inputs = {"M0": M0_SHA, "anchor": sha256_file(ANCHOR_COORDS)}
    with atomic_stage(stage, "passive_M0_external_and_anchor_baseline", inputs, resume=resume) as outputs:
        if outputs is not None:
            model, _ = load_model(0, device); model.eval()
            ext = STATE_DIR / "M0_external_benchmark_results.csv"; anc = STATE_DIR / "M0_uniform_anchor_results.csv"
            evaluate_external(0, model, device, ext, status); evaluate_anchor(0, model, device, anc)
            outputs.extend([ext, anc, STATE_DIR / "anchor" / "M0.parquet"]); del model; torch.cuda.empty_cache()


def operator_audit(cycle: int, selected_t: pd.DataFrame, selected_p: pd.DataFrame, parent_qd: pd.DataFrame, offspring_qd: pd.DataFrame) -> pd.DataFrame:
    manifest = pd.read_parquet(cycle_dir(cycle) / "03_mutation" / "mutation_manifest.parquet")
    tset, pset = set(selected_t.instance_hash.astype(str)), set(selected_p.instance_hash.astype(str)); old_niches = set(parent_qd.niche_id.astype(int))
    qmap = dict(zip(offspring_qd.instance_hash.astype(str), offspring_qd.niche_id.astype(int)))
    rows = []
    for operator in OPERATOR_NAMES:
        part = manifest[manifest.operator_id.eq(operator)]; valid = part.valid_after_repair & part.proposal_status.eq("accepted") & ~part.duplicate
        hashes = part.loc[valid, "child_hash"].astype(str).tolist()
        rows.append({"Cycle": f"C{cycle:02d}", "operator_id": operator, "attempted": len(part),
                     "valid": int((part.valid_after_repair & part.proposal_status.eq("accepted")).sum()),
                     "unique": len(hashes), "invalid": int((~(part.valid_after_repair & part.proposal_status.eq("accepted"))).sum()),
                     "fraction_entering_T": len(set(hashes) & tset) / max(1, len(hashes)),
                     "fraction_entering_P": len(set(hashes) & pset) / max(1, len(hashes)),
                     "fraction_new_niche": sum(qmap.get(h) not in old_niches for h in hashes) / max(1, len(hashes))})
    frame = pd.DataFrame(rows); frame.to_csv(cycle_dir(cycle) / "operator_trajectory.csv", index=False); return frame


def mutation_next(cycle: int, selected_p: pd.DataFrame, resume: bool) -> tuple[dict[str, Any], float]:
    started = time.perf_counter(); cdir = cycle_dir(cycle)
    if cycle == 9:
        info = {"status": "FINAL_CYCLE_REACHED_NO_NEXT_MUTATION", "next_cycle": None, "operator_count": 11}
    else:
        p_bundle = cdir / "06_selection" / "P_next.npz"; hashes = selected_p.instance_hash.astype(str).tolist()
        coords, ohashes, manifest = base._prepare_offspring("FULL", cycle + 1, p_bundle, hashes, resume)
        info = {"status": "COMPLETE", "next_cycle": f"C{cycle+1:02d}", "offspring_n": len(ohashes),
                "membership_sha256": membership_hash(ohashes), "manifest_sha256": sha256_file(cycle_dir(cycle + 1) / "03_mutation" / "mutation_manifest.parquet"),
                "bundle_sha256": sha256_file(cycle_dir(cycle + 1) / "03_mutation" / "offspring_coordinates.npz"), "operator_count": 11}
    atomic_json(cdir / "mutation_output_manifest.json", info); return info, time.perf_counter() - started


def verify_cycle_complete(cycle: int) -> bool:
    marker = cycle_dir(cycle) / "COMPLETE.json"
    if not marker.is_file(): return False
    data = read_json(marker)
    for name, record in data["required_files"].items():
        path = cycle_dir(cycle) / name
        if not path.is_file() or sha256_file(path) != record["sha256"]: raise RuntimeError(f"completed cycle artifact conflict: {path}")
    return True


def complete_cycle(cycle: int) -> None:
    required = [
        "A_archive_pre_manifest.json", "B_calibration_membership.csv", "B_current_hardness.parquet",
        "HGB_model_manifest.json", "candidate_manifest.csv", "candidate_features.parquet",
        "candidate_predictions.parquet", "candidate_ranks.parquet", "T_hard_top10k_membership.csv",
        "P_parent_membership.csv", "QD_assignment.csv", "selected_expert_tour_accounting.csv",
        "selected_exact_hardness.parquet", "BQ_training_config.json", "BQ_training_log.csv",
        "M_next_checkpoint_manifest.json", "external_benchmark_results.csv", "uniform_anchor_results.csv",
        "archive_post_manifest.json", "mutation_output_manifest.json", "cycle_costs.json",
    ]
    records = {}
    for name in required:
        path = cycle_dir(cycle) / name
        if not path.is_file(): raise RuntimeError(f"required cycle artifact missing: {path}")
        records[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    atomic_json(cycle_dir(cycle) / "COMPLETE.json", {"status": "COMPLETE", "cycle": f"C{cycle:02d}",
                "required_files": records, "completed_utc": utc_now()})


def run_cycle(cycle: int, device: torch.device, source: dict[str, Any], resume: bool, status: EventStatus) -> None:
    if verify_cycle_complete(cycle):
        status.update(cycle=f"C{cycle:02d}", phase="verified_complete_skip", latest_checkpoint=str(model_path(cycle + 1)))
        return
    if free_gib() < 1.5: raise RuntimeError(f"disk safety stop before C{cycle:02d}: only {free_gib():.2f} GiB free")
    started = time.perf_counter(); cdir = cycle_dir(cycle); cdir.mkdir(parents=True, exist_ok=True)
    status.update(cycle=f"C{cycle:02d}", phase="load_M_and_branch_archive")
    model, model_meta = load_model(cycle, device); model.eval()
    acoords, ameta, atours = archive_pre(cycle); prepare_archive_input(cycle, acoords, ameta, atours)

    status.update(phase="archive_sample_current_hardness_refresh")
    b, calibration, b_cache = calibration_stage(cycle, model, model_meta["sha256"], device, acoords, ameta, atours, resume)
    status.update(phase="fit_HGB40_five_seeds")
    hgb_manifest = fit_hgb_stage(cycle, b, source, resume)
    status.update(phase="construct_parent_offspring_union")
    union_coords, union, parent_qd, offspring_qd, candidate_info = prepare_candidate(cycle, resume)
    status.update(phase="F40_candidate_cache_and_HGB_score")
    scores, score_info = score_stage(cycle, union_coords, union, source, resume)
    status.update(phase="freeze_T_and_P")
    selected_t, selected_p, selection_info = selection_stage(cycle, union_coords, union, scores, resume)

    status.update(phase="T_oracle_and_exact_current_hardness")
    labeled, t_tours, oracle, archive_post = oracle_archive_stage(cycle, model, device, union_coords, selected_t,
                                                                  acoords, ameta, atours, resume)
    true_h = labeled.current_hardness.to_numpy(float); predicted = labeled.hgb_predicted_log1p_hardness.to_numpy(float)
    rho = float(spearmanr(predicted, true_h).statistic) if np.ptp(predicted) > 0 else 0.0

    status.update(phase="BQ_256E_training")
    new_model, new_meta, bq = base._train_bq("FULL", cycle, model, model_meta["sha256"],
                                              union_coords[selected_t.coord_index.to_numpy(int)],
                                              selected_t.instance_hash.astype(str).tolist(), t_tours, device, resume, status)
    base._repair(cdir / "10_post_adaptation", new_model, new_meta["sha256"],
                 union_coords[selected_t.coord_index.to_numpy(int)], selected_t.instance_hash.astype(str).tolist(),
                 t_tours, true_h, device, 2026 + cycle, resume)
    shutil.copy2(cdir / "09_bq_training" / "training_loss_curve.csv", cdir / "BQ_training_log.csv")
    atomic_json(cdir / "BQ_training_config.json", CONFIG["training"])
    checkpoint_manifest = {"model": f"M{cycle+1}", "cycle": f"C{cycle:02d}", "checkpoint": str(model_path(cycle + 1)),
                           "sha256": new_meta["sha256"], "parent_sha256": model_meta["sha256"],
                           "hard_set_sha256": selection_info["T_membership_sha256"],
                           "uniform_anchor_sha256": sha256_file(ANCHOR_COORDS), "training_config_sha256": TRAINING_CONFIG_HASH,
                           "wall_seconds": bq["total_wall_s"]}
    atomic_json(cdir / "M_next_checkpoint_manifest.json", checkpoint_manifest)

    status.update(phase="passive_external_and_uniform_anchor_evaluation")
    _, external_s = evaluate_external(cycle + 1, new_model, device, cdir / "external_benchmark_results.csv", status)
    _, anchor_s = evaluate_anchor(cycle + 1, new_model, device, cdir / "uniform_anchor_results.csv")
    operator_audit(cycle, selected_t, selected_p, parent_qd, offspring_qd)
    status.update(cycle=f"C{cycle:02d}", phase="generate_next_offspring" if cycle < 9 else "final_cycle_no_next_mutation")
    mutation_info, mutation_s = mutation_next(cycle, selected_p, resume)

    old_parent_set = set(parent_qd.instance_hash.astype(str)); pset = set(selected_p.instance_hash.astype(str))
    diversity = {"C": base._niche_metrics(union.niche_id.to_numpy(int)),
                 "T": base._niche_metrics(selected_t.niche_id.to_numpy(int)),
                 "P": base._niche_metrics(selected_p.niche_id.to_numpy(int)),
                 "O": base._niche_metrics(offspring_qd.niche_id.to_numpy(int))}
    costs = {
        "archive_refresh_s": float(calibration["refresh_wall_seconds"]),
        "feature_extraction_s": float(b_cache["wall_seconds"] + score_info["feature_cache"]["wall_seconds"]),
        "hgb_fit_s": float(hgb_manifest["fit_total_s"]), "candidate_score_s": float(score_info["score_wall_seconds"]),
        "qd_parent_s": stage_seconds(cdir / "_stages" / "03_candidate_union") + stage_seconds(cdir / "_stages" / "05_selection"),
        "concorde_s": float(oracle["wall_seconds"]), "exact_hardness_s": float(oracle["wall_seconds"]),
        "bq_training_s": float(bq["training_loop_wall_s"]), "external_eval_s": external_s,
        "uniform_anchor_eval_s": anchor_s, "mutation_s": mutation_s,
        "total_cycle_s": time.perf_counter() - started,
    }
    atomic_json(cdir / "cycle_costs.json", costs)
    bhashes = b.instance_hash.astype(str).tolist(); thashes = selected_t.instance_hash.astype(str).tolist()
    summary = {
        "cycle": cycle, "model_before": f"M{cycle}", "model_after": f"M{cycle+1}",
        "archive_pre_n": len(ameta), "calibration_n": len(b), "calibration_target_n": CALIBRATION_N,
        "calibration": {"archive_n": len(ameta), "calibration_n": len(b), "mean_current_H": calibration["hardness"]["mean"],
                        "P90_H": calibration["hardness"]["P90"], "P99_H": calibration["hardness"]["P99"],
                        "occupied_QD_niches": int(ameta.loc[ameta.instance_hash.astype(str).isin(bhashes), "niche_id"].nunique()),
                        "N_eff": base._niche_metrics(ameta.loc[ameta.instance_hash.astype(str).isin(bhashes), "niche_id"].to_numpy(int))["N_eff"],
                        "HGB_fit_time": hgb_manifest["fit_total_s"]},
        "candidate_n": len(union), "parent_count_in_candidate": candidate_info["parent_n"], "candidate_policy": "unique_parent_offspring_union",
        "T_n": len(selected_t), "P_n": len(selected_p), "Q_n": selection_info["Q_n"], "G_n": selection_info["G_n"],
        "survivor_fraction": len(old_parent_set & pset) / 10000, "selected_hardness": summary_stats(true_h),
        "predicted_T": summary_stats(predicted), "selection_diagnostics": {"spearman_pred_true": rho},
        "diversity": diversity, "oracle": {**oracle, "selection_frozen_before_oracle": True},
        "archive_post_n": len(archive_post), "archive_branch_only": True, "archive_update_after_T_freeze": True,
        "no_future_archive_injection": True, "current_hardness_refreshed": True,
        # B is the pre-cycle archive. Any B/T overlap is historical information,
        # not leakage; leakage means a T item absent from A_pre entered B before fit.
        "B_frozen_before_fit": True,
        "same_cycle_leakage_n": len(set(bhashes) & (set(thashes) - set(ameta.instance_hash.astype(str)))),
        "calibration_policy_ok": len(b) == min(CALIBRATION_N, len(ameta)),
        "C00_exact_corrected_union": cycle != 0 or (len(union) == 119781 and membership_hash(union.instance_hash.astype(str).tolist()) == source["C0_membership_sha256"]),
        "hgb": {"estimator": "HistGradientBoostingRegressor", "target": "log1p_LBQ", "config_sha256": source["HGB_config_sha256"],
                "seeds": SEEDS, "aggregate": "median", "F40_sha256": source["F40_sha256"], "ranking_recomputed": False,
                "operator_feature": False, "retuned": False, "cross_family_ensemble": False},
        "QD": {"descriptors": "QD4", "K": 1000, "normalization_frozen": True, "retuned": False},
        "mutation": {"operator_count": 11, "frozen": True, "retuned": False, **mutation_info},
        "selection": {"T": "HGB_Top10000", "P": "QD_representatives_plus_HGB_fill", "age_cap": None,
                      "forced_offspring_quota": None, "post_truth_reselection": False},
        "training_protocol": {"fresh_adam": True, "scheduler": None, "weight_decay": 0, "gradient_clipping": None, "AMP": False, "seed": 2026},
        "bq": bq, "feature_cache": score_info["feature_cache"], "costs": costs,
        "benchmark_controls_training": False, "resume_identity_preserved": True,
    }
    atomic_json(cdir / "cycle_summary.json", summary); complete_cycle(cycle)
    status.update(cycle=f"C{cycle:02d}", phase="COMPLETE", latest_checkpoint=str(model_path(cycle + 1)),
                  last_valid_artifact=str(cdir / "COMPLETE.json"))
    print(f"S4-B PROGRESS C{cycle:02d} M{cycle}->M{cycle+1} COMPLETE", flush=True)
    del new_model, model; torch.cuda.empty_cache()


def preflight() -> dict[str, Any]:
    source = verify_sources()
    return {"status": "PASS", "M0_sha256": source["M0_sha256"], "A0_n": source["A0_n"],
            "A0_membership_sha256": source["A0_membership_sha256"], "C0_n": source["C0_n"],
            "C0_membership_sha256": source["C0_membership_sha256"], "F40_sha256": source["F40_sha256"],
            "HGB_config_sha256": source["HGB_config_sha256"], "Rscript": source["Rscript"],
            "free_D_GiB": free_gib(), "target_exists": RUN_DIR.exists(), "cuda_available": torch.cuda.is_available()}


def smoke_test() -> dict[str, Any]:
    source = verify_sources(); device = base._device()
    if device.type != "cuda": raise RuntimeError("CUDA required")
    model, _ = load_model(0, device); coords, hashes, tours = base._load_bundle(B0_SOURCE / "labeled_bundle.npz")
    h = hardness(model, coords[:2], tours[:2], device)
    import tempfile
    with tempfile.TemporaryDirectory(prefix="s4b_smoke_") as temporary:
        out = Path(temporary) / "f.parquet"
        s3b2._extract_shard({"coordinates": coords[:2], "hashes": hashes[:2], "names": F40, "out": str(out),
                             "rscript": s3b2.rscript(), "params": s3b2.scaler_params()})
        feature = pd.read_parquet(out)
    x = np.tile(feature[F40].to_numpy(float), (20, 1)); y = np.tile(np.log1p(h), 20)
    estimator = HistGradientBoostingRegressor(**hgb_params(2026)).fit(x, y); pred = estimator.predict(x)
    return {"status": "PASS", "device": str(device), "hardness": h.tolist(), "F40_shape": list(feature[F40].shape),
            "HGB_finite": bool(np.isfinite(pred).all()), "source_fingerprint_digest": source["source_fingerprint_digest"]}


def write_error(exc: BaseException, status: EventStatus | None) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    label = "HGB_UNION_QD_10CYCLE_ROLLOUT_INCOMPLETE"
    (RUN_DIR / label).write_text(label + "\n", encoding="utf-8")
    (RUN_DIR / "ERROR_REPORT.md").write_text(f"# S4-B error\n\n- Label: `{label}`\n- Exception: `{type(exc).__name__}: {exc}`\n- Resume: `D:\\conda_envs\\bq-nco\\python.exe scripts\\run_s4b_hgb40_union_qd.py --resume`\n\n```text\n{traceback.format_exc()}\n```\n", encoding="utf-8")
    if LOCK.is_file():
        lock = read_json(LOCK); lock.update({"state": "FAILED", "error": repr(exc), "timestamp": utc_now()}); atomic_json(LOCK, lock)
    if status: status.update(phase="FAILED", latest_error=repr(exc), recovery_state="MANUAL_RESUME_REQUIRED")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--preflight", action="store_true"); parser.add_argument("--smoke-test", action="store_true"); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(preflight(), indent=2), flush=True); return 0
    if args.smoke_test:
        print(json.dumps(smoke_test(), indent=2), flush=True); return 0
    status: EventStatus | None = None
    try:
        source = verify_sources(); initialize(source, args.resume); status = EventStatus(); device = base._device()
        if device.type != "cuda": raise RuntimeError(f"CUDA required, got {device}")
        print(f"S4-B START device={device} resume={args.resume} A0=500 C00_calibration=500", flush=True)
        baseline_evaluation(device, status, args.resume)
        for cycle in range(10): run_cycle(cycle, device, source, args.resume, status)
        if not source_unchanged(source): raise RuntimeError("frozen source changed during run")
        status.update(cycle=None, phase="final_reporting_and_60_check_audit")
        result = build_reports(ROOT, RUN_DIR, source)
        if result["status"] != "PASS": raise RuntimeError("final strict verification failed")
        atomic_json(LOCK, {"pid": os.getpid(), "state": "COMPLETE", "timestamp": utc_now(), **result})
        status.update(phase="COMPLETE", latest_checkpoint=str(model_path(10)), latest_error=None)
        print("S4-B HGB40 UNION + QD 10-CYCLE ROLLOUT FROM M0 COMPLETE", flush=True)
        print(result["primary_label"], flush=True)
        return 0
    except BaseException as exc:
        write_error(exc, status); traceback.print_exc(); return 1


if __name__ == "__main__":
    raise SystemExit(main())
