"""Canonical HGB20 / GP20 feature extraction (salesperson + frozen arcsinh scaler)."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(r"D:\NCO\BLADE-NCO-mainline")
MANIFEST_PATH = ROOT / "configs" / "hgb20_features.yaml"
CYCLE0_MANIFEST = ROOT / "audits" / "cycle0_initialization_validation_v1" / "00_protocol" / "gp20_features.json"
SCALER_ASSET = ROOT / "assets" / "qd" / "reference_cloud" / "formal_raw_scaler.json"
R_SCRIPT = ROOT / "src" / "features" / "extract_salesperson_raw_features.R"
P0_GP20_CACHE = ROOT / "audits" / "cycle0_initialization_validation_v1" / "07_hgb_controls" / "gp20_features.parquet"
BATCH = 250


def load_hgb20_manifest() -> dict:
    yaml_spec = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    json_spec = json.loads(CYCLE0_MANIFEST.read_text(encoding="utf-8"))
    yaml_features = list(yaml_spec["features"])
    json_features = list(json_spec["features"])
    if yaml_features != json_features:
        raise RuntimeError("S2_B_HGB20_FEATURE_MANIFEST_AMBIGUOUS: yaml vs cycle0 json differ")
    if len(yaml_features) != 20:
        raise RuntimeError(f"HGB20 manifest length {len(yaml_features)} != 20")
    forbidden_tokens = (
        "operator", "lineage", "cycle", "solver", "checkpoint", "l_bq", "hardness",
        "gp_pred", "niche_id", "tsplib", "uniform100",
    )
    lowered = [n.lower() for n in yaml_features]
    hits = [n for n in lowered if any(tok in n for tok in forbidden_tokens)]
    if hits:
        raise RuntimeError(f"HGB20 manifest contains forbidden names: {hits}")
    h = hashlib.sha256()
    with MANIFEST_PATH.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return {
        "name": "HGB20",
        "status": yaml_spec.get("status"),
        "n_features": 20,
        "features": yaml_features,
        "source_file": str(MANIFEST_PATH.relative_to(ROOT)).replace("\\", "/"),
        "source_audit": str(CYCLE0_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "also_used_by": [
            "src/final_run/science.py extract_gp20 / train_hgb_controls",
            "scripts/run_cycle0_models_and_evaluation.py",
            "configs/surrogate.yaml controls.HGB20.features=GP20",
        ],
        "implementation": {
            "raw": "src/features/extract_salesperson_raw_features.R (salesperson 1.0.0)",
            "transform": "arcsinh((raw - center_median) / scale) using assets/qd/reference_cloud/formal_raw_scaler.json",
            "fit_on_p0": False,
        },
        "solver_free_structural": True,
        "yaml_json_identical": True,
        "manifest_sha256": h.hexdigest(),
        "acquisition_driver": False,
    }


def _rscript() -> str:
    configured = os.environ.get("BLADE_RSCRIPT")
    if configured:
        return configured
    found = shutil.which("Rscript")
    if found:
        return found
    standard = Path(r"C:\Program Files\R\R-4.6.0\bin\x64\Rscript.exe")
    if standard.is_file():
        return str(standard)
    raise FileNotFoundError("Rscript was not found; set BLADE_RSCRIPT")


def _scaler_params() -> dict[str, tuple[float, float]]:
    payload = json.loads(SCALER_ASSET.read_text(encoding="utf-8"))
    return {
        item["feature_name"]: (float(item["center_median"]), float(item["scale"]))
        for item in payload["parameters"]
    }


def extract_gp20_batch(
    coordinates: np.ndarray,
    hashes: list[str],
    features: list[str],
    require_finite: bool = True,
) -> pd.DataFrame:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
        raise ValueError(f"expected (B,N,2), got {coordinates.shape}")
    if len(hashes) != len(coordinates):
        raise ValueError("hash/coordinate length mismatch")
    params = _scaler_params()
    missing = [name for name in features if name not in params]
    if missing:
        raise RuntimeError(f"scaler missing HGB20 features {missing}")
    frames = []
    rscript = _rscript()
    for start in range(0, len(hashes), BATCH):
        stop = min(len(hashes), start + BATCH)
        with tempfile.TemporaryDirectory(prefix="s2b_gp20_") as temporary:
            work = Path(temporary)
            inp = work / "coordinates.csv"
            raw = work / "features.csv"
            schema = work / "schema.csv"
            with inp.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(["state_id", "node_idx", "x", "y"])
                for instance_hash, points in zip(hashes[start:stop], coordinates[start:stop], strict=True):
                    for node, (x, y) in enumerate(points):
                        writer.writerow([instance_hash, node, repr(float(x)), repr(float(y))])
            process = subprocess.run(
                [rscript, str(R_SCRIPT), str(inp), str(raw), str(schema)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if process.returncode or not raw.is_file():
                raise RuntimeError(f"GP20 extraction failed: {process.stderr[-1000:]}")
            data = pd.read_csv(raw)
            if data.state_id.astype(str).tolist() != hashes[start:stop]:
                raise RuntimeError("GP20 feature alignment failure")
            missing_cols = [name for name in features if name not in data.columns]
            if missing_cols:
                raise RuntimeError(f"GP20 raw output missing features {missing_cols}")
            frame = pd.DataFrame({"instance_hash": data.state_id.astype(str)})
            for name in features:
                center, scale = params[name]
                frame[name] = np.arcsinh((data[name].to_numpy(float) - center) / scale)
            values = frame[features].to_numpy(float)
            if not np.isfinite(values).all():
                bad = ~np.isfinite(values)
                counts = {name: int(bad[:, i].sum()) for i, name in enumerate(features) if bad[:, i].any()}
                n_rows = int(bad.any(axis=1).sum())
                # Canonical science.extract_gp20 does not drop these rows. sklearn HGB
                # accepts NaN as missing but rejects Inf, so map non-finite -> NaN.
                if require_finite:
                    raise RuntimeError(
                        f"GP20 produced non-finite transformed features: rows={n_rows} counts={counts}"
                    )
                frame[features] = np.where(np.isfinite(values), values, np.nan)
            if "feature_error" in data.columns and data["feature_error"].notna().any() and require_finite:
                raise RuntimeError("GP20 raw extraction reported feature_error")
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _worker(payload: dict) -> str:
    features = payload["features"]
    out = Path(payload["out"])
    if out.is_file():
        return str(out)
    coords = np.asarray(payload["coordinates"], dtype=np.float64)
    hashes = list(payload["hashes"])
    frame = extract_gp20_batch(
        coords, hashes, features, require_finite=bool(payload.get("require_finite", True))
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp, index=False)
    tmp.replace(out)
    return str(out)


def _init_worker() -> None:
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def extract_gp20_cached(
    coordinates: np.ndarray,
    hashes: list[str],
    cache_path: Path,
    features: list[str],
    workers: int = 1,
    require_finite: bool = True,
) -> tuple[pd.DataFrame, dict]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if cache_path.is_file():
        frame = pd.read_parquet(cache_path)
        frame["instance_hash"] = frame["instance_hash"].astype(str)
        if frame["instance_hash"].tolist() != list(hashes):
            raise RuntimeError(f"cached GP20 hash order mismatch: {cache_path}")
        missing = [c for c in features if c not in frame.columns]
        if missing:
            raise RuntimeError(f"cached GP20 missing {missing}")
        elapsed = time.perf_counter() - started
        return frame[["instance_hash", *features]].reset_index(drop=True), {
            "wall_s": elapsed, "n": len(hashes), "cached": True, "path": str(cache_path),
        }
    shard_dir = cache_path.parent / (cache_path.stem + "_shards")
    shard_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for start in range(0, len(hashes), BATCH):
        stop = min(len(hashes), start + BATCH)
        shard = shard_dir / f"shard_{start:07d}_{stop:07d}.parquet"
        jobs.append({
            "coordinates": np.asarray(coordinates[start:stop], dtype=np.float64),
            "hashes": list(hashes[start:stop]),
            "features": features,
            "out": str(shard),
            "require_finite": bool(require_finite),
        })
    n_jobs = len(jobs)
    done = 0
    if workers <= 1 or n_jobs == 1:
        for job in jobs:
            _worker(job)
            done += 1
            if done == 1 or done == n_jobs or done % 20 == 0:
                print(f"[GP20] {cache_path.name} {done}/{n_jobs} shards", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
            futures = [pool.submit(_worker, job) for job in jobs]
            for fut in as_completed(futures):
                fut.result()
                done += 1
                if done == 1 or done == n_jobs or done % 20 == 0:
                    print(f"[GP20] {cache_path.name} {done}/{n_jobs} shards workers={workers}", flush=True)
    parts = []
    for start in range(0, len(hashes), BATCH):
        stop = min(len(hashes), start + BATCH)
        shard = shard_dir / f"shard_{start:07d}_{stop:07d}.parquet"
        parts.append(pd.read_parquet(shard))
    frame = pd.concat(parts, ignore_index=True)
    if frame["instance_hash"].astype(str).tolist() != list(hashes):
        raise RuntimeError("concatenated GP20 hash order mismatch")
    tmp = cache_path.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp, index=False)
    tmp.replace(cache_path)
    elapsed = time.perf_counter() - started
    values = frame[features].to_numpy(float)
    nan_rows = int((~np.isfinite(values) | pd.isna(frame[features]).to_numpy()).any(axis=1).sum()) if len(frame) else 0
    return frame[["instance_hash", *features]].reset_index(drop=True), {
        "wall_s": elapsed, "n": len(hashes), "cached": False, "path": str(cache_path),
        "workers": workers, "batch": BATCH, "require_finite": bool(require_finite),
        "rows_with_nan": nan_rows,
    }
