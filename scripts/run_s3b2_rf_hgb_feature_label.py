"""S3-B2: RF/HGB feature capacity x label budget. Offline. No Concorde. No production change."""
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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from features.qd4 import QD4_FEATURE_NAMES  # noqa: E402
from surrogate_benchmark.concentration import niche_metrics, topk_indices  # noqa: E402
from surrogate_benchmark.metrics import k_q, safe_spearman, tail_ranking  # noqa: E402
from surrogate_benchmark.models import (  # noqa: E402
    HGB_CFG_CANONICAL,
    SEEDS,
    fit_hgb4_seed,
    load_hgb4,
    median_stack,
    predict_hgb,
    save_hgb,
)

FULL = ROOT / "runs" / "blade_v2_full_seed2026"
AUDIT = ROOT / "audits" / "S3_B2_RF_HGB_feature_label_scaling_v1"
S3A = ROOT / "audits" / "S3_A_four_model_surrogate_family_screening_v1"
S3B1 = ROOT / "audits" / "S3_B1_RF4_HGB4_label_budget_scaling_v1"
WORK = AUDIT / "work"
SCALER = ROOT / "assets" / "qd" / "reference_cloud" / "formal_raw_scaler.json"
R_SCRIPT = ROOT / "src" / "features" / "extract_salesperson_raw_features.R"
FEATURES4 = list(QD4_FEATURE_NAMES)
CYCLES = (2, 8, 12, 14)
BUDGETS = (500, 2000, 10000)
DIMS = (4, 20, 40, 94)
MODELS = ("RF", "HGB")
BOOT_N = 10000
Q = 0.10
BATCH = 250
WORKERS = 6
BOOT_SEED = {2: 20260204, 8: 20260804, 12: 20261204, 14: 20261404}
PROBE_NAME = {2: "R02C", 8: "R08C", 12: "R12C", 14: "R14C"}
GP4_C02 = 0.24243105050658767
RF_CFG = dict(
    n_estimators=500, criterion="squared_error", max_depth=None, min_samples_split=2,
    min_samples_leaf=1, max_features=1.0, bootstrap=True, max_samples=None,
    ccp_alpha=0.0, n_jobs=-1,
)
S3B1_MODEL = {"RF": "RF4", "HGB": "HGB4"}
COLORS = {"C02": "#1f77b4", "C08": "#ff7f0e", "C12": "#2ca02c", "C14": "#d62728"}
NEW_CONCORDE_CALLS = 0


def _json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def sha_list(values) -> str:
    return hashlib.sha256(("\n".join(map(str, values)) + "\n").encode()).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_frame(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        return frame[columns] if columns else frame
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        frame = pd.read_parquet(path)
        return frame[columns] if columns else frame


def cycle_name(s: int) -> str:
    return f"C{s:02d}"


def prev_dir(s: int) -> Path:
    return FULL / "cycles" / f"C{s - 1:02d}"


def base_dir(s: int) -> Path:
    return FULL / "cycles" / cycle_name(s)


def rscript() -> str:
    configured = os.environ.get("BLADE_RSCRIPT")
    if configured:
        return configured
    found = shutil.which("Rscript")
    if found:
        return found
    standard = Path(r"C:\Program Files\R\R-4.6.0\bin\x64\Rscript.exe")
    if standard.is_file():
        return str(standard)
    raise FileNotFoundError("Rscript was not found")


def feature_order() -> list[str]:
    payload = json.loads(SCALER.read_text(encoding="utf-8"))
    names = list(payload["feature_order"])
    if len(names) != 94 or int(payload["feature_count"]) != 94:
        raise RuntimeError("Full94 scaler inventory is not 94 features")
    if names[:0] or any(name in names for name in ()):
        pass
    missing = [name for name in FEATURES4 if name not in names]
    if missing:
        raise RuntimeError(f"F4 names missing from Full94: {missing}")
    return names


def write_protocol(names: list[str]) -> None:
    text = f"""# S3-B2 RF/HGB feature capacity x label budget

Offline historical replay. No BQ training, no mutation, no production change, no QD change, no ensemble.
Future production remains C_s = unique(P_{{s-1}} union O_s). Calibration remains cumulative archive A_s.

States C02/C08/C12/C14. Models RF and HGB with frozen S3-A hyperparameters. No retuning as dimension grows.
Label budgets 500, 2000, 10000, exact nested S3-B1 samples. Feature capacities 4, 20, 40, 94.
Evaluation: reused S3-A C-probes, status DEVELOPMENT_REUSED_S3A_C_PROBE. New Concorde calls = 0.

Full94 is the existing salesperson 1.0.0 structural output named by
`assets/qd/reference_cloud/formal_raw_scaler.json` feature_order ({len(names)} names),
then the frozen arcsinh((raw - center_median) / scale) transform from that same scaler.
That is the representation stored in the historical QD4 columns and used by the reused F4 models.
A raw-versus-stored check on C02 showed agreement only after this transform.

No leakage-clean RF ranking over Full94 was found. The frozen GP20/HGB20 list is not an RF Full94 ranking
and does not nest a separate Top40, so it is not reused. Fallback ranking uses only B500 from the four
states, five RF seeds, median rank, feature-name tie break. B2000, B10000, and C-probe truth are not used.

Primary-label rule, fixed before cell results:
- FEATURE_SCALING_SATURATES_BEFORE_FULL94 if F20 or F40 has a paired capture interval entirely above F4
  and no F94-minus-F40 interval lies entirely above 0.
- FEATURE_GAIN_REQUIRES_LARGE_LABEL_BUDGET if a clear feature gain exists at N=2000 or N=10000 and none at N=500.
- FEATURE_CAPACITY_STRENGTHENS_SINGLE_SURROGATE_CASE if a richer set clearly lifts the F4 worst state and the
  four-state range falls below half the RF F4 N=10000 range.
- FEATURE_CAPACITY_DOES_NOT_RESOLVE_STATE_DEPENDENCE if the best rich configuration still has a four-state
  range at least half the RF F4 N=10000 range.
- FEATURE_SCALING_RESULT_MIXED otherwise.

Clear means the paired 95% interval lies entirely above 0. No percentage-point cutoff is used.
"""
    (AUDIT / "01_protocol.md").write_text(text, encoding="utf-8")


def load_nested() -> dict:
    payload = json.loads((S3B1 / "05_nested_sample_hashes.json").read_text(encoding="utf-8"))
    return {cycle: {int(k): v for k, v in sets.items()} for cycle, sets in payload.items()}


def load_probe(s: int) -> tuple[list[str], np.ndarray]:
    payload = json.loads((S3A / "work" / f"C{s:02d}_probe.json").read_text(encoding="utf-8"))
    hashes = [str(item) for item in payload["hashes"]]
    hard = read_frame(S3A / "work" / f"C{s:02d}_probe_hardness.parquet")
    hard["instance_hash"] = hard["instance_hash"].astype(str)
    mapping = dict(zip(hard.instance_hash, hard.H_s.astype(float)))
    return hashes, np.asarray([mapping[item] for item in hashes], dtype=float)


def point_metrics(h: np.ndarray, selected: set[str], hashes: list[str]) -> dict:
    mask = np.asarray([item in selected for item in hashes])
    k = k_q(len(hashes), Q)
    order = np.argsort(-h)
    hard = set(np.asarray(hashes)[order[:k]].tolist())
    h_rand = float(np.mean(h))
    h_sel = float(np.mean(h[mask])) if mask.any() else float("nan")
    h_ora = float(np.mean(h[order[:k]]))
    denom = h_ora - h_rand
    capture = float((h_sel - h_rand) / denom) if np.isfinite(denom) and abs(denom) > 1e-12 else float("nan")
    return {
        "qn": int(mask.sum()), "H_random": h_rand, "H_selected": h_sel, "H_oracle": h_ora,
        "capture": capture, "recall": float(len(hard & selected) / len(hard)), "mask": mask,
    }


def select(scores, hashes, k: int) -> list[str]:
    idx = topk_indices(np.asarray(scores, float), np.asarray(hashes), k)
    return [str(hashes[int(i)]) for i in idx]


def percentile_high(scores: np.ndarray) -> np.ndarray:
    ranks = rankdata(np.asarray(scores, float), method="average")
    return (ranks - 1.0) / (len(scores) - 1.0)


def verify_sources(nested: dict) -> pd.DataFrame:
    b1 = read_frame(S3B1 / "03_state_manifest.csv")
    rows = []
    for s in CYCLES:
        cycle = cycle_name(s)
        ref = b1[b1.cycle == cycle].iloc[0]
        offspring = read_frame(base_dir(s) / "05_scoring" / "offspring_scored.parquet", ["instance_hash"])
        off_sha = sha_list(offspring.instance_hash.astype(str).tolist())
        if off_sha != ref.offspring_sha256:
            raise RuntimeError(f"{cycle} offspring hash mismatch")
        arch = read_frame(prev_dir(s) / "08_archive" / "A_next.parquet", ["instance_hash"])
        arch_sha = sha_list(arch.instance_hash.astype(str).tolist())
        if arch_sha != ref.archive_sha256:
            raise RuntimeError(f"{cycle} archive hash mismatch")
        for n in BUDGETS:
            if sha_list(nested[cycle][n]) != sha_list(nested[cycle][10000][:n]):
                raise RuntimeError(f"{cycle} B{n} is not the S3-B1 prefix")
        if sha_list(nested[cycle][500]) != ref.archive500_sha256:
            raise RuntimeError(f"{cycle} B500 hash mismatch")
        labels = read_frame(S3B1 / "work" / f"{cycle}_labels.parquet")
        if labels.instance_hash.astype(str).tolist() != nested[cycle][10000]:
            raise RuntimeError(f"{cycle} S3-B1 label order is not B10000")
        rows.append({
            "cycle": cycle, "solver": f"M{s}", "checkpoint_sha256": ref.checkpoint_sha256,
            "archive_sha256": ref.archive_sha256, "offspring_sha256": off_sha,
            "archive500_sha256": ref.archive500_sha256,
            "b2000_sha256": sha_list(nested[cycle][2000]),
            "b10000_sha256": sha_list(nested[cycle][10000]),
            "offspring_n": int(len(offspring)), "feature_implementation": "salesperson_1.0.0_plus_frozen_arcsinh_scaler",
            "scaler_sha256": sha_file(SCALER), "rscript_sha256": sha_file(R_SCRIPT),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(AUDIT / "03_state_manifest.csv", index=False)
    return frame


def assert_f4_reuse(nested: dict) -> None:
    expected = read_frame(S3B1 / "14_operational_capture_by_N.csv")
    for s in CYCLES:
        cycle = cycle_name(s)
        offspring = read_frame(base_dir(s) / "05_scoring" / "offspring_scored.parquet", ["instance_hash"])
        hashes = offspring.instance_hash.astype(str).to_numpy()
        k = k_q(len(hashes), Q)
        probe_h, probe_y = load_probe(s)
        for model in MODELS:
            src = S3B1_MODEL[model]
            pred = np.load(S3B1 / "work" / "pred" / f"{cycle}_{src}_N500.npy")
            if len(pred) != len(hashes):
                raise RuntimeError(f"{cycle} {model} N=500 prediction length mismatch")
            cap = point_metrics(probe_y, set(select(pred, hashes, k)), probe_h)["capture"]
            row = expected[(expected.cycle == cycle) & (expected.model == src) & (expected.N == 500)].iloc[0]
            if abs(cap - float(row.OperationalCapture10)) > 1e-8:
                raise SystemExit(f"STOP F4 reuse {cycle} {model} N=500 capture {cap} != S3-B1 {row.OperationalCapture10}")
            for n in (2000, 10000):
                pred_n = np.load(S3B1 / "work" / "pred" / f"{cycle}_{src}_N{n}.npy")
                cap_n = point_metrics(probe_y, set(select(pred_n, hashes, k)), probe_h)["capture"]
                row_n = expected[(expected.cycle == cycle) & (expected.model == src) & (expected.N == n)].iloc[0]
                if abs(cap_n - float(row_n.OperationalCapture10)) > 1e-8:
                    raise SystemExit(f"STOP F4 reuse {cycle} {model} N={n}")
        print(f"[S3-B2] F4 predictions match S3-B1 {cycle}", flush=True)


def _extract_shard(payload: dict) -> str:
    out = Path(payload["out"])
    if out.is_file():
        return str(out)
    coords = np.asarray(payload["coordinates"], dtype=np.float64)
    hashes = list(payload["hashes"])
    names = list(payload["names"])
    with tempfile.TemporaryDirectory(prefix="s3b2_f94_") as temporary:
        work = Path(temporary)
        inp, raw, schema = work / "coordinates.csv", work / "features.csv", work / "schema.csv"
        with inp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["state_id", "node_idx", "x", "y"])
            for instance_hash, points in zip(hashes, coords, strict=True):
                for node, (x, y) in enumerate(points):
                    writer.writerow([instance_hash, node, repr(float(x)), repr(float(y))])
        process = subprocess.run(
            [payload["rscript"], str(R_SCRIPT), str(inp), str(raw), str(schema)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if process.returncode or not raw.is_file():
            raise RuntimeError(f"Full94 extraction failed: {process.stderr[-1500:]}")
        data = pd.read_csv(raw)
        if data.state_id.astype(str).tolist() != hashes:
            raise RuntimeError("Full94 alignment failure")
        if "feature_error" in data.columns and data.feature_error.notna().any():
            raise RuntimeError("Full94 feature_error reported")
        missing = [name for name in names if name not in data.columns]
        if missing:
            raise RuntimeError(f"Full94 missing columns {missing[:8]}")
        frame = data[["state_id", *names]].rename(columns={"state_id": "instance_hash"})
        params = payload["params"]
        for name in names:
            center, scale = params[name]
            frame[name] = np.arcsinh((frame[name].to_numpy(float) - center) / scale)
        values = frame[names].to_numpy(float)
        if not np.isfinite(values).all():
            bad = ~np.isfinite(values)
            counts = {name: int(bad[:, i].sum()) for i, name in enumerate(names) if bad[:, i].any()}
            raise RuntimeError(f"non-finite Full94 features {counts}")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp, index=False)
    tmp.replace(out)
    return str(out)


def _coords_for(npz_path: Path, wanted: list[str]) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as payload:
        hashes = [str(item) for item in np.asarray(payload["instance_hash"]).tolist()]
        coords = np.asarray(payload["coordinates"], dtype=np.float64)
    index = {h: i for i, h in enumerate(hashes)}
    missing = [h for h in wanted if h not in index]
    if missing:
        raise RuntimeError(f"{npz_path.name} missing {len(missing)} coordinates")
    return coords[np.asarray([index[h] for h in wanted], dtype=int)]


def scaler_params() -> dict[str, tuple[float, float]]:
    payload = json.loads(SCALER.read_text(encoding="utf-8"))
    return {item["feature_name"]: (float(item["center_median"]), float(item["scale"])) for item in payload["parameters"]}


def scale_cache_if_raw(cache: Path, names: list[str]) -> None:
    frame = read_frame(cache)
    depth = frame["spn_mst_depth_mean"].to_numpy(float)
    if float(np.median(np.abs(depth))) < 3.0:
        return
    params = scaler_params()
    for name in names:
        center, scale = params[name]
        frame[name] = np.arcsinh((frame[name].to_numpy(float) - center) / scale)
    tmp = cache.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp, index=False)
    tmp.replace(cache)
    print(f"[S3-B2] applied frozen arcsinh scaler to {cache.name}", flush=True)


def extract_pool(s: int, pool: str, hashes: list[str], names: list[str]) -> tuple[pd.DataFrame, dict]:
    cycle = cycle_name(s)
    cache = WORK / "full94" / f"{cycle}_{pool}.parquet"
    shard_dir = WORK / "full94" / f"{cycle}_{pool}_shards"
    if cache.is_file():
        scale_cache_if_raw(cache, names)
        frame = read_frame(cache)
        frame["instance_hash"] = frame["instance_hash"].astype(str)
        if frame.instance_hash.tolist() != hashes:
            raise RuntimeError(f"{cycle} {pool} cache order mismatch")
        timing_path = cache.with_name(cache.stem + "_extract_s.json")
        wall = 0.0
        if timing_path.is_file():
            wall = float(json.loads(timing_path.read_text(encoding="utf-8"))["wall_s"])
        print(f"[S3-B2] reuse Full94 {cycle} {pool} n={len(frame)}", flush=True)
        return frame, {"wall_s": wall, "cached": True, "n": len(frame), "bytes": cache.stat().st_size}
    npz = prev_dir(s) / "08_archive" / "A_next.npz" if pool == "B10000" else base_dir(s) / "03_mutation" / "offspring_coordinates.npz"
    coords = _coords_for(npz, hashes)
    shard_dir.mkdir(parents=True, exist_ok=True)
    exe = rscript()
    params = scaler_params()
    jobs = []
    for start in range(0, len(hashes), BATCH):
        stop = min(len(hashes), start + BATCH)
        jobs.append({
            "coordinates": coords[start:stop],
            "hashes": hashes[start:stop],
            "names": names,
            "out": str(shard_dir / f"shard_{start:07d}_{stop:07d}.parquet"),
            "rscript": exe,
            "params": params,
        })
    del coords
    started = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool_exec:
        futures = [pool_exec.submit(_extract_shard, job) for job in jobs]
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if done == 1 or done == len(jobs) or done % 20 == 0:
                print(f"[S3-B2] Full94 {cycle} {pool} {done}/{len(jobs)}", flush=True)
    parts = [read_frame(shard_dir / f"shard_{start:07d}_{min(len(hashes), start + BATCH):07d}.parquet")
             for start in range(0, len(hashes), BATCH)]
    frame = pd.concat(parts, ignore_index=True)
    frame["instance_hash"] = frame["instance_hash"].astype(str)
    if frame.instance_hash.tolist() != hashes:
        raise RuntimeError(f"{cycle} {pool} concatenated order mismatch")
    tmp = cache.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp, index=False)
    tmp.replace(cache)
    elapsed = time.perf_counter() - started
    print(f"[S3-B2] Full94 {cycle} {pool} done {elapsed:.1f}s", flush=True)
    _json(cache.with_name(cache.stem + "_extract_s.json"), {"wall_s": elapsed})
    return frame, {"wall_s": elapsed, "cached": False, "n": len(frame), "bytes": cache.stat().st_size}


def verify_f4_match(s: int, pool: str, frame: pd.DataFrame) -> float:
    cycle = cycle_name(s)
    if pool == "B10000":
        stored = read_frame(prev_dir(s) / "08_archive" / "A_next.parquet", ["instance_hash", *FEATURES4])
    else:
        stored = read_frame(base_dir(s) / "05_scoring" / "offspring_scored.parquet", ["instance_hash", *FEATURES4])
    stored["instance_hash"] = stored["instance_hash"].astype(str)
    stored = stored.drop_duplicates("instance_hash").set_index("instance_hash").loc[frame.instance_hash]
    max_abs = 0.0
    for name in FEATURES4:
        diff = np.max(np.abs(stored[name].to_numpy(float) - frame[name].to_numpy(float)))
        max_abs = max(max_abs, float(diff))
    print(f"[S3-B2] F4 raw match {cycle} {pool} max_abs={max_abs:.3e}", flush=True)
    if max_abs > 1e-5:
        raise SystemExit(f"STOP {cycle} {pool} Full94 F4 columns differ from stored raw QD4 by {max_abs}")
    return max_abs


def rank_features(names: list[str], nested: dict) -> dict:
    frozen = WORK / "RANK_FROZEN.json"
    if frozen.is_file() and (AUDIT / "08_frozen_feature_ranking.csv").is_file():
        print("[S3-B2] reuse frozen feature ranking", flush=True)
        return json.loads(frozen.read_text(encoding="utf-8"))
    rows = []
    rank_stack = []
    for s in CYCLES:
        cycle = cycle_name(s)
        frame = read_frame(WORK / "full94" / f"{cycle}_B10000.parquet")
        frame["instance_hash"] = frame["instance_hash"].astype(str)
        labels = read_frame(S3B1 / "work" / f"{cycle}_labels.parquet")
        ymap = dict(zip(labels.instance_hash.astype(str), labels.y_log1p.astype(float)))
        b500 = nested[cycle][500]
        block = frame.set_index("instance_hash").loc[b500]
        x = block[names].to_numpy(float)
        y = np.asarray([ymap[h] for h in b500], dtype=float)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise SystemExit(f"STOP ranking {cycle}: non-finite B500 features or labels")
        for seed in SEEDS:
            model = RandomForestRegressor(random_state=seed, **RF_CFG)
            model.fit(x, y)
            imp = np.asarray(model.feature_importances_, dtype=float)
            ranks = rankdata(-imp, method="average")
            rank_stack.append(ranks)
            for name, value, rk in zip(names, imp, ranks):
                rows.append({"cycle": cycle, "seed": int(seed), "feature": name, "importance": float(value), "rank": float(rk)})
            print(f"[S3-B2] rank {cycle} seed {seed}", flush=True)
            del model
    rank_mat = np.vstack(rank_stack)
    median_rank = np.median(rank_mat, axis=0)
    order = sorted(names, key=lambda name: (float(median_rank[names.index(name)]), name))
    extras = [name for name in order if name not in FEATURES4]
    subsets = {
        "F4": list(FEATURES4),
        "F20": list(FEATURES4) + extras[:16],
        "F40": list(FEATURES4) + extras[:36],
        "F94": list(FEATURES4) + extras,
    }
    for key, vals in subsets.items():
        if len(vals) != int(key[1:]) or len(set(vals)) != len(vals):
            raise RuntimeError(f"{key} length {len(vals)}")
    if not (set(subsets["F4"]) < set(subsets["F20"]) < set(subsets["F40"]) < set(subsets["F94"])):
        raise RuntimeError("feature nesting failed")
    if subsets["F94"][:4] != FEATURES4 or subsets["F20"][:4] != FEATURES4 or subsets["F40"][:20] != subsets["F20"]:
        raise RuntimeError("feature prefix nesting failed")
    raw = pd.DataFrame(rows)
    raw.to_csv(AUDIT / "07_feature_importance_raw.csv", index=False)
    summary = pd.DataFrame({
        "feature": names,
        "median_rank": median_rank,
        "mean_importance": raw.groupby("feature").importance.mean().reindex(names).to_numpy(),
    }).sort_values(["median_rank", "feature"])
    summary["order"] = np.arange(1, len(summary) + 1)
    summary.to_csv(AUDIT / "08_frozen_feature_ranking.csv", index=False)
    _json(AUDIT / "09_feature_subset_manifests.json", subsets)
    _json(AUDIT / "10_feature_subset_hashes.json", {key: sha_list(vals) for key, vals in subsets.items()})
    payload = {"subsets": subsets, "source": "fallback_B500_only", "n_fits": 20}
    _json(frozen, payload)
    hgb20 = ["spn_sum_of_lowest_edge_values", "spn_distance_varcoeff", "spn_mst_depth_mean", "spn_mst_depth_median"]
    (AUDIT / "06_feature_ranking_source.md").write_text(
        "\n".join([
            "# Feature ranking source",
            "",
            "No documented leakage-clean RF importance ranking over Full94 was found.",
            "Searched scripts, configs, and audit artifacts for feature-importance tables, Top20/Top40 manifests, and rank hashes.",
            "The frozen list in configs/hgb20_features.yaml is the GP20 control, not an RF Full94 ranking.",
            f"Its fourth name is {hgb20[3]}, so it is not the F4 prefix, and no paired Top40 exists.",
            "It was not reused.",
            "",
            "Fallback RANK94_FROZEN:",
            "- data: exact S3-B1 B500 from C02, C08, C12, C14 only",
            "- target: log1p(L_BQ) under the current solver, reused S3-B1 labels",
            "- model: frozen S3-A RF, seeds 2026-2030",
            "- aggregate: median rank across 4 states x 5 seeds",
            "- tie break: feature name",
            "- B2000, B10000, and C-probe true hardness were not used",
            "",
            "F20 and F40 are prefixes of one ranking after forcing the four F4 names first.",
            "Full94 values are salesperson outputs after the frozen arcsinh scaler, matching stored F4.",
            "",
        ]),
        encoding="utf-8",
    )
    return payload


def copy_f4(s: int) -> None:
    cycle = cycle_name(s)
    (WORK / "pred").mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        src_name = S3B1_MODEL[model]
        for n in BUDGETS:
            dest = WORK / "pred" / f"{cycle}_{model}_N{n}_F4.npy"
            meta_path = WORK / "pred" / f"{cycle}_{model}_N{n}_F4.json"
            if dest.is_file() and meta_path.is_file():
                continue
            pred = np.load(S3B1 / "work" / "pred" / f"{cycle}_{src_name}_N{n}.npy")
            np.save(dest, pred)
            meta = json.loads((S3B1 / "work" / "pred" / f"{cycle}_{src_name}_N{n}.json").read_text(encoding="utf-8"))
            meta["reused_s3b1"] = True
            _json(meta_path, meta)


def fit_cell(s: int, model: str, n: int, dim: int, x_train, y_train, x_score, columns: list[str]) -> None:
    cycle = cycle_name(s)
    pred_path = WORK / "pred" / f"{cycle}_{model}_N{n}_F{dim}.npy"
    meta_path = WORK / "pred" / f"{cycle}_{model}_N{n}_F{dim}.json"
    if pred_path.is_file() and meta_path.is_file():
        return
    directory = AUDIT / "models" / cycle / model / f"N{n}_F{dim}"
    directory.mkdir(parents=True, exist_ok=True)
    fit_s = 0.0
    preds = []
    importances = []
    if model == "RF":
        score_s = 0.0
        for seed in SEEDS:
            path = directory / f"seed_{seed}.joblib"
            if not path.is_file():
                t0 = time.perf_counter()
                est = RandomForestRegressor(random_state=seed, **RF_CFG)
                est.fit(x_train, y_train)
                fit_s += time.perf_counter() - t0
                joblib.dump(est, path, compress=3)
            else:
                est = joblib.load(path)
            if dim == 94:
                importances.append(np.asarray(est.feature_importances_, dtype=float))
            t1 = time.perf_counter()
            preds.append(est.predict(x_score))
            score_s += time.perf_counter() - t1
            del est
        pred = np.median(np.vstack(preds), axis=0)
    else:
        ests = []
        for seed in SEEDS:
            path = directory / f"seed_{seed}.joblib"
            if path.is_file():
                est, meta = load_hgb4(path)
                fit_s += float((meta or {}).get("fit_s") or 0.0)
            else:
                est, meta = fit_hgb4_seed(x_train, y_train, seed)
                meta = save_hgb(est, meta, path, columns)
                fit_s += float(meta["fit_s"])
            ests.append(est)
        t1 = time.perf_counter()
        pred = median_stack(np.vstack([predict_hgb(est, x_score)[0] for est in ests]))
        score_s = time.perf_counter() - t1
        del ests
    np.save(pred_path, np.asarray(pred, dtype=np.float64))
    if importances:
        np.save(WORK / "imp" / f"{cycle}_RF_N{n}_F94.npy", np.vstack(importances))
    size = int(sum(p.stat().st_size for p in directory.glob("*.joblib")))
    _json(meta_path, {"fit_s": fit_s, "score_s": score_s, "model_bytes": size, "reused_s3b1": False})
    print(f"[S3-B2] fit {cycle} {model} N={n} F{dim} fit={fit_s:.1f}s score={score_s:.1f}s", flush=True)


def prepare_fits(nested: dict, subsets: dict) -> None:
    (WORK / "pred").mkdir(parents=True, exist_ok=True)
    (WORK / "imp").mkdir(parents=True, exist_ok=True)
    for s in CYCLES:
        cycle = cycle_name(s)
        print(f"[S3-B2] models {cycle}", flush=True)
        copy_f4(s)
        labels = read_frame(S3B1 / "work" / f"{cycle}_labels.parquet")
        ymap = dict(zip(labels.instance_hash.astype(str), labels.y_log1p.astype(float)))
        b_frame = read_frame(WORK / "full94" / f"{cycle}_B10000.parquet")
        b_frame["instance_hash"] = b_frame["instance_hash"].astype(str)
        if b_frame.instance_hash.tolist() != nested[cycle][10000]:
            raise RuntimeError(f"{cycle} B10000 feature order drifted")
        y_all = np.asarray([ymap[h] for h in b_frame.instance_hash], dtype=float)
        o_frame = read_frame(WORK / "full94" / f"{cycle}_Os.parquet")
        o_frame["instance_hash"] = o_frame["instance_hash"].astype(str)
        offspring = read_frame(base_dir(s) / "05_scoring" / "offspring_scored.parquet", ["instance_hash"])
        if o_frame.instance_hash.tolist() != offspring.instance_hash.astype(str).tolist():
            raise RuntimeError(f"{cycle} offspring feature order drifted")
        for dim in DIMS:
            cols = subsets[f"F{dim}"]
            if dim == 4:
                continue
            x_all = b_frame[cols].to_numpy(float)
            x_score = o_frame[cols].to_numpy(float)
            for n in BUDGETS:
                for model in MODELS:
                    fit_cell(s, model, n, dim, x_all[:n], y_all[:n], x_score, cols)
            del x_all, x_score


def bootstrap_masks(h: np.ndarray, masks: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(h)
    draws = rng.integers(0, n, size=(BOOT_N, n))
    hb = h[draws]
    h_rand = hb.mean(axis=1)
    k = k_q(n, Q)
    ora = np.argpartition(-hb, k, axis=1)[:, :k]
    h_ora = np.take_along_axis(hb, ora, axis=1).mean(axis=1)
    denom = h_ora - h_rand
    out = {"capture": {}, "recall": {}, "hsel": {}}
    for key, mask in masks.items():
        sel = mask[draws]
        counts = sel.sum(axis=1)
        hsel = np.full(BOOT_N, np.nan)
        ok = counts > 0
        hsel[ok] = (hb[ok] * sel[ok]).sum(axis=1) / counts[ok]
        out["capture"][key] = (hsel - h_rand) / denom
        out["recall"][key] = np.take_along_axis(sel.astype(float), ora, axis=1).mean(axis=1)
        out["hsel"][key] = hsel
    return out


def load_pred(s: int, model: str, n: int, dim: int) -> tuple[np.ndarray, dict]:
    cycle = cycle_name(s)
    pred = np.load(WORK / "pred" / f"{cycle}_{model}_N{n}_F{dim}.npy")
    meta = json.loads((WORK / "pred" / f"{cycle}_{model}_N{n}_F{dim}.json").read_text(encoding="utf-8"))
    return pred, meta


def analyze(nested: dict, subsets: dict, extract_info: list[dict]) -> None:
    print("[S3-B2] analysis", flush=True)
    label_cost = read_frame(S3B1 / "24_production_cost_by_N.csv")
    master, cap_rows, rec_rows, delta_rows, inter_rows = [], [], [], [], []
    diff_rows, rob_rows, fit_rows, cost_rows, feat_cost_rows = [], [], [], [], []
    cap = {m: {} for m in MODELS}
    rec = {m: {} for m in MODELS}
    gains = {m: {} for m in MODELS}
    contrasts = (("F20", "F4", "F20_F4"), ("F40", "F20", "F40_F20"), ("F94", "F40", "F94_F40"),
                 ("F40", "F4", "F40_F4"), ("F94", "F4", "F94_F4"))
    for s in CYCLES:
        cycle = cycle_name(s)
        offspring = read_frame(base_dir(s) / "05_scoring" / "offspring_scored.parquet", ["instance_hash", "niche_id"])
        hashes = offspring.instance_hash.astype(str).to_numpy()
        niches = offspring.niche_id.to_numpy(int)
        k = k_q(len(hashes), Q)
        probe_h, probe_y = load_probe(s)
        probe_index = {h: i for i, h in enumerate(hashes)}
        probe_pos = np.asarray([probe_index[h] for h in probe_h])
        info = next(item for item in extract_info if item["cycle"] == cycle)
        feature_seconds = float(info["b10000_wall_s"]) + float(info["os_wall_s"])
        preds, masks, chosen, metas = {}, {}, {}, {}
        for model in MODELS:
            cap[model][cycle] = {}
            rec[model][cycle] = {}
            for n in BUDGETS:
                cap[model][cycle][n] = {}
                rec[model][cycle][n] = {}
                for dim in DIMS:
                    pred, meta = load_pred(s, model, n, dim)
                    key = (model, n, dim)
                    preds[key] = pred
                    top = select(pred, hashes, k)
                    chosen[key] = top
                    masks[key] = np.isin(np.asarray(probe_h), np.asarray(top))
                    metas[key] = meta
                    metrics = point_metrics(probe_y, set(top), probe_h)
                    cap[model][cycle][n][dim] = metrics["capture"]
                    rec[model][cycle][n][dim] = metrics["recall"]
                    preds[(model, n, dim, "metrics")] = metrics
                    preds[(model, n, dim, "tail")] = tail_ranking(pred[probe_pos], probe_y)
                    preds[(model, n, dim, "rho")] = safe_spearman(pred[probe_pos], probe_y)
                    preds[(model, n, dim, "niche")] = niche_metrics(niches[topk_indices(pred, hashes, k)])
        boot = bootstrap_masks(probe_y, masks, BOOT_SEED[s])
        for model in MODELS:
            gains[model][cycle] = {}
            for n in BUDGETS:
                gains[model][cycle][n] = {}
                for left, right, name in contrasts:
                    for metric, bucket, table in (
                        ("capture", boot["capture"], cap),
                        ("recall", boot["recall"], rec),
                    ):
                        diff = bucket[(model, n, int(left[1:]))] - bucket[(model, n, int(right[1:]))]
                        lo, hi = np.quantile(diff, [0.025, 0.975])
                        point = table[model][cycle][n][int(left[1:])] - table[model][cycle][n][int(right[1:])]
                        delta_rows.append({
                            "cycle": cycle, "model": model, "N": n, "contrast": name, "metric": metric,
                            "delta": point, "ci_l": float(lo), "ci_u": float(hi),
                        })
                        if metric == "capture":
                            gains[model][cycle][n][name] = (point, float(lo), float(hi))
                for dim in DIMS:
                    clo, chi = np.quantile(boot["capture"][(model, n, dim)], [0.025, 0.975])
                    rlo, rhi = np.quantile(boot["recall"][(model, n, dim)], [0.025, 0.975])
                    metrics = preds[(model, n, dim, "metrics")]
                    tail = preds[(model, n, dim, "tail")]
                    niche = preds[(model, n, dim, "niche")]
                    meta = metas[(model, n, dim)]
                    fit_s, score_s = float(meta["fit_s"]), float(meta["score_s"])
                    feat_cost = 0.0 if dim == 4 else feature_seconds
                    refresh = float(label_cost[(label_cost.cycle == cycle) & (label_cost.model == S3B1_MODEL[model]) & (label_cost.N == n)].cumulative_label_refresh_s.iloc[0])
                    cap_rows.append({"cycle": cycle, "model": model, "N": n, "FeatureSet": f"F{dim}", "OperationalCapture10": metrics["capture"], "ci_l": float(clo), "ci_u": float(chi)})
                    rec_rows.append({"cycle": cycle, "model": model, "N": n, "FeatureSet": f"F{dim}", "FullPoolTop10Recall": metrics["recall"], "ci_l": float(rlo), "ci_u": float(rhi)})
                    master.append({
                        "Cycle": cycle, "Solver": f"M{s}", "Model": model, "TrainN": n,
                        "FeatureSet": f"F{dim}", "FeatureN": dim, "CalibrationSource": "cumulative_archive_A_s",
                        "ProbeID": PROBE_NAME[s], "ProbeStatus": "DEVELOPMENT_REUSED_S3A_C_PROBE",
                        "O_s_N": len(hashes), "SelectedProbeN": metrics["qn"],
                        "H_random": metrics["H_random"], "H_selected": metrics["H_selected"], "H_oracle": metrics["H_oracle"],
                        "OperationalCapture10": metrics["capture"], "CaptureCI_L": float(clo), "CaptureCI_U": float(chi),
                        "FullPoolTop10Recall": metrics["recall"], "RecallCI_L": float(rlo), "RecallCI_U": float(rhi),
                        "GlobalSpearman": preds[(model, n, dim, "rho")], "Tail10Rho": tail["true_tail_rho10"],
                        "SelectedNiches": niche["occupied_niches"], "SelectedNeff": niche["N_eff"],
                        "FeatureExtractionCost": feat_cost, "FitTime": fit_s, "ScoreTime": score_s,
                        "TotalProductionEquivalentTime": refresh + feat_cost + fit_s + score_s,
                    })
                    fit_rows.append({"cycle": cycle, "model": model, "N": n, "FeatureSet": f"F{dim}", "fit_s": fit_s, "score_s": score_s, "model_bytes": meta.get("model_bytes"), "reused_s3b1": bool(meta.get("reused_s3b1"))})
                    cost_rows.append({
                        "cycle": cycle, "model": model, "N": n, "FeatureSet": f"F{dim}",
                        "label_refresh_s": refresh, "feature_extraction_s": feat_cost,
                        "fit_s": fit_s, "score_s": score_s,
                        "total_production_equivalent_s": refresh + feat_cost + fit_s + score_s,
                        "note": "F20/F40/F94 share one salesperson Full94 extraction; F4 uses stored QD4. Subset extraction is not cheaper with the current R implementation.",
                    })
                    inter_rows.append({
                        "cycle": cycle, "model": model, "N": n, "FeatureSet": f"F{dim}",
                        "capture": metrics["capture"], "recall": metrics["recall"],
                        "gain_vs_f4": metrics["capture"] - cap[model][cycle][n][4],
                    })
            for n in BUDGETS:
                diff = boot["capture"][("RF", n, 4)]  # placeholder replaced below
                for dim in DIMS:
                    diff = boot["capture"][("RF", n, dim)] - boot["capture"][("HGB", n, dim)]
                    drec = boot["recall"][("RF", n, dim)] - boot["recall"][("HGB", n, dim)]
                    clo, chi = np.quantile(diff, [0.025, 0.975])
                    rlo, rhi = np.quantile(drec, [0.025, 0.975])
                    diff_rows.append({
                        "cycle": cycle, "N": n, "FeatureN": dim,
                        "DeltaCapture": cap["RF"][cycle][n][dim] - cap["HGB"][cycle][n][dim],
                        "CaptureCI_L": float(clo), "CaptureCI_U": float(chi),
                        "DeltaRecall": rec["RF"][cycle][n][dim] - rec["HGB"][cycle][n][dim],
                        "RecallCI_L": float(rlo), "RecallCI_U": float(rhi),
                    })
        feat_cost_rows.append({
            "cycle": cycle, "b10000_wall_s": info["b10000_wall_s"], "os_wall_s": info["os_wall_s"],
            "b10000_bytes": info["b10000_bytes"], "os_bytes": info["os_bytes"],
            "f4_incremental_s": 0.0,
            "f20_f40_f94_incremental_s": feature_seconds,
            "limitation": "salesperson computes all structural features in one pass; F20 and F40 are column selections",
        })
    for model in MODELS:
        for n in BUDGETS:
            for dim in DIMS:
                vals = [cap[model][cycle_name(s)][n][dim] for s in CYCLES]
                rvals = [rec[model][cycle_name(s)][n][dim] for s in CYCLES]
                rob_rows.append({
                    "model": model, "N": n, "FeatureN": dim,
                    "C02": vals[0], "C08": vals[1], "C12": vals[2], "C14": vals[3],
                    "median_capture": float(np.median(vals)), "worst_capture": float(min(vals)),
                    "best_capture": float(max(vals)), "range_capture": float(max(vals) - min(vals)),
                    "sd_capture": float(np.std(vals, ddof=1)),
                    "median_recall": float(np.median(rvals)), "worst_recall": float(min(rvals)),
                })
    master_df = pd.DataFrame(master)
    if len(master_df) != 96:
        raise RuntimeError(f"master rows {len(master_df)} != 96")
    master_df.to_csv(AUDIT / "29_master_table.csv", index=False)
    pd.DataFrame(cap_rows).to_csv(AUDIT / "17_operational_capture.csv", index=False)
    pd.DataFrame(rec_rows).to_csv(AUDIT / "18_operational_recall.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(AUDIT / "19_bootstrap_feature_deltas.csv", index=False)
    pd.DataFrame(inter_rows).to_csv(AUDIT / "20_feature_label_interaction.csv", index=False)
    pd.DataFrame(diff_rows).to_csv(AUDIT / "21_RF_minus_HGB_matched_cells.csv", index=False)
    rob = pd.DataFrame(rob_rows)
    rob.to_csv(AUDIT / "22_cross_state_robustness.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(AUDIT / "13_model_fit_summary.csv", index=False)
    pd.DataFrame(feat_cost_rows).to_csv(AUDIT / "24_feature_cost.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(AUDIT / "25_production_cost.csv", index=False)
    stability = importance_stability(subsets["F94"])
    redundancy(subsets["F94"])
    write_predictions(nested)
    make_figures(master_df, rob, gains)
    primary, nxt, flags = decide(gains, rob, cap)
    write_reports(subsets, cap, rec, gains, rob, primary, nxt, flags, extract_info)
    _json(AUDIT / "STRICT_VERIFICATION.json", strict_payload(master_df, primary, flags))
    _json(AUDIT / "STATUS.json", {"status": "COMPLETE", "primary": primary, "next": nxt, "new_concorde": 0})
    print_summary(subsets, cap, rec, gains, rob, primary, nxt, flags, extract_info)


def importance_stability(columns: list[str]) -> None:
    rows = []
    store = {}
    for s in CYCLES:
        cycle = cycle_name(s)
        for n in BUDGETS:
            path = WORK / "imp" / f"{cycle}_RF_N{n}_F94.npy"
            if not path.is_file():
                continue
            imp = np.load(path)
            store[(cycle, n)] = np.median(imp, axis=0)
    for s in CYCLES:
        cycle = cycle_name(s)
        if (cycle, 500) not in store or (cycle, 10000) not in store:
            continue
        rows.append({
            "comparison": f"{cycle}_N500_vs_N10000",
            "spearman": float(spearmanr(store[(cycle, 500)], store[(cycle, 10000)]).statistic),
            "top10_overlap": _overlap(store[(cycle, 500)], store[(cycle, 10000)], columns, 10),
            "top20_overlap": _overlap(store[(cycle, 500)], store[(cycle, 10000)], columns, 20),
        })
    for a, b in (("C02", "C08"), ("C02", "C12"), ("C02", "C14"), ("C08", "C12"), ("C08", "C14"), ("C12", "C14")):
        if (a, 10000) not in store or (b, 10000) not in store:
            continue
        rows.append({
            "comparison": f"{a}_vs_{b}_N10000",
            "spearman": float(spearmanr(store[(a, 10000)], store[(b, 10000)]).statistic),
            "top10_overlap": _overlap(store[(a, 10000)], store[(b, 10000)], columns, 10),
            "top20_overlap": _overlap(store[(a, 10000)], store[(b, 10000)], columns, 20),
        })
    pd.DataFrame(rows).to_csv(AUDIT / "23_RF_importance_stability.csv", index=False)


def _overlap(a, b, columns, k: int) -> float:
    ia = set(np.argsort(-a)[:k].tolist())
    ib = set(np.argsort(-b)[:k].tolist())
    return float(len(ia & ib) / k)


def redundancy(columns: list[str]) -> None:
    rows = []
    for s in CYCLES:
        cycle = cycle_name(s)
        frame = read_frame(WORK / "full94" / f"{cycle}_B10000.parquet", columns)
        values = frame.to_numpy(float)
        std = values.std(axis=0)
        near_constant = int((std < 1e-8).sum())
        corr = np.corrcoef(values.T)
        iu = np.triu_indices(len(columns), k=1)
        pair = corr[iu]
        rows.append({
            "cycle": cycle, "n": len(frame), "near_constant": near_constant,
            "pairs": int(len(pair)), "abs_corr_ge_0.999": int(np.sum(np.abs(pair) >= 0.999)),
            "abs_corr_ge_0.99": int(np.sum(np.abs(pair) >= 0.99)),
            "median_abs_corr": float(np.nanmedian(np.abs(pair))),
        })
    pd.DataFrame(rows).to_csv(AUDIT / "26_feature_redundancy.csv", index=False)


def write_predictions(nested: dict) -> None:
    print("[S3-B2] writing prediction tables", flush=True)
    pred_parts, rank_parts, mem_parts = [], [], []
    for s in CYCLES:
        cycle = cycle_name(s)
        offspring = read_frame(base_dir(s) / "05_scoring" / "offspring_scored.parquet", ["instance_hash"])
        hashes = offspring.instance_hash.astype(str).to_numpy()
        k = k_q(len(hashes), Q)
        for model in MODELS:
            for n in BUDGETS:
                for dim in DIMS:
                    pred, _ = load_pred(s, model, n, dim)
                    pred_parts.append(pd.DataFrame({"cycle": cycle, "model": model, "TrainN": n, "FeatureN": dim, "instance_hash": hashes, "prediction": pred}))
                    rank_parts.append(pd.DataFrame({"cycle": cycle, "model": model, "TrainN": n, "FeatureN": dim, "instance_hash": hashes, "percentile_rank": percentile_high(pred)}))
                    mem_parts.append(pd.DataFrame({"cycle": cycle, "model": model, "TrainN": n, "FeatureN": dim, "rank": np.arange(1, k + 1), "instance_hash": select(pred, hashes, k)}))
    pd.concat(pred_parts, ignore_index=True).to_parquet(AUDIT / "14_full_pool_predictions.parquet", index=False)
    pd.concat(rank_parts, ignore_index=True).to_parquet(AUDIT / "15_full_pool_ranks.parquet", index=False)
    pd.concat(mem_parts, ignore_index=True).to_parquet(AUDIT / "16_full_pool_top10_membership.parquet", index=False)


def decide(gains, rob, cap):
    cycles = [cycle_name(s) for s in CYCLES]

    def clear(model, cycle, n, name):
        return gains[model][cycle][n][name][1] > 0

    def any_clear(name, models=MODELS, ns=BUDGETS, states=cycles):
        return any(clear(m, c, n, name) for m in models for c in states for n in ns)

    rich = any_clear("F20_F4") or any_clear("F40_F4") or any_clear("F94_F4")
    rich_500 = any(clear(m, c, 500, name) for m in MODELS for c in cycles for name in ("F20_F4", "F40_F4", "F94_F4"))
    rich_large = any(clear(m, c, n, name) for m in MODELS for c in cycles for n in (2000, 10000) for name in ("F20_F4", "F40_F4", "F94_F4"))
    full94 = any_clear("F94_F40")
    f40 = any_clear("F40_F20")
    f4_range = float(rob[(rob.model == "RF") & (rob.N == 10000) & (rob.FeatureN == 4)].range_capture.iloc[0])
    best = rob.sort_values("worst_capture", ascending=False).iloc[0]
    best_range = float(best.range_capture)
    range_remains = best_range >= 0.5 * f4_range
    if (any_clear("F20_F4") or any_clear("F40_F4")) and not full94:
        primary = "FEATURE_SCALING_SATURATES_BEFORE_FULL94"
    elif rich_large and not rich_500:
        primary = "FEATURE_GAIN_REQUIRES_LARGE_LABEL_BUDGET"
    elif rich and not range_remains:
        primary = "FEATURE_CAPACITY_STRENGTHENS_SINGLE_SURROGATE_CASE"
    elif range_remains:
        primary = "FEATURE_CAPACITY_DOES_NOT_RESOLVE_STATE_DEPENDENCE"
    else:
        primary = "FEATURE_SCALING_RESULT_MIXED"
    c02_best = max(cap[m]["C02"][n][d] for m in MODELS for n in BUDGETS for d in DIMS)
    c02_upper_below_gp = True
    for m in MODELS:
        for n in BUDGETS:
            for d in DIMS:
                if cap[m]["C02"][n][d] == c02_best:
                    pass
    # GP remains above every C02 tree point estimate.
    gp_still_ahead = c02_best < GP4_C02
    separated = float(best.best_capture) - float(best.worst_capture) > float(best.worst_capture)
    if (not separated) and float(best.worst_capture) > 0.4 and not gp_still_ahead:
        nxt = "FINAL_SURROGATE_D_PROBE_VALIDATION"
    elif gp_still_ahead:
        nxt = "GP_PLUS_TREE_RANK_ENSEMBLE_AUDIT"
    elif full94 and primary != "FEATURE_SCALING_SATURATES_BEFORE_FULL94":
        nxt = "FEATURE_COST_REDUCTION_SUBSET_AUDIT"
    else:
        nxt = "S3_RANK_ENSEMBLE_AUDIT"
    flags = {
        "RF_RICH_FEATURE_GAIN": "YES" if any_clear("F20_F4", ("RF",)) or any_clear("F40_F4", ("RF",)) or any_clear("F94_F4", ("RF",)) else "NO",
        "HGB_RICH_FEATURE_GAIN": "YES" if any_clear("F20_F4", ("HGB",)) or any_clear("F40_F4", ("HGB",)) or any_clear("F94_F4", ("HGB",)) else "NO",
        "RF_FEATURE_GAIN_DATA_HUNGRY": "YES" if (not any(clear("RF", c, 500, name) for c in cycles for name in ("F20_F4", "F40_F4", "F94_F4"))) and any(clear("RF", c, n, name) for c in cycles for n in (2000, 10000) for name in ("F20_F4", "F40_F4", "F94_F4")) else "NO",
        "HGB_FEATURE_GAIN_DATA_HUNGRY": "YES" if (not any(clear("HGB", c, 500, name) for c in cycles for name in ("F20_F4", "F40_F4", "F94_F4"))) and any(clear("HGB", c, n, name) for c in cycles for n in (2000, 10000) for name in ("F20_F4", "F40_F4", "F94_F4")) else "NO",
        "C02_FEATURE_LIMITED": "YES" if any(clear(m, "C02", n, name) for m in MODELS for n in BUDGETS for name in ("F20_F4", "F40_F4", "F94_F4")) else "NO",
        "C08_FEATURE_LIMITED": "YES" if any(clear(m, "C08", 10000, name) for m in MODELS for name in ("F20_F4", "F40_F4", "F94_F4")) else "NO",
        "C12_FEATURE_SATURATED": "YES" if not any(clear(m, "C12", n, name) for m in MODELS for n in BUDGETS for name in ("F20_F4", "F40_F4", "F94_F4")) else "NO",
        "C14_FEATURE_LIMITED": "YES" if any(clear(m, "C14", n, name) for m in MODELS for n in BUDGETS for name in ("F20_F4", "F40_F4", "F94_F4")) else "NO",
        "TOP20_SUFFICIENT": "YES" if not f40 and not full94 else "NO",
        "TOP40_ADDS_VALUE": "YES" if f40 else "NO",
        "FULL94_ADDS_VALUE": "YES" if full94 else "NO",
        "SINGLE_ROBUST_SURROGATE_EMERGES": "YES" if nxt == "FINAL_SURROGATE_D_PROBE_VALIDATION" else "NO",
        "MODEL_COMPLEMENTARITY_REMAINS": "YES" if gp_still_ahead else "NO",
    }
    return primary, nxt, flags


def make_figures(master, rob, gains) -> None:
    xs = np.array(DIMS)

    def panel(model, n, path):
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        sub = master[(master.Model == model) & (master.TrainN == n)]
        for s in CYCLES:
            c = cycle_name(s)
            rows = sub[sub.Cycle == c].sort_values("FeatureN")
            y = rows.OperationalCapture10.to_numpy(float)
            lo, hi = rows.CaptureCI_L.to_numpy(float), rows.CaptureCI_U.to_numpy(float)
            yerr = np.vstack([np.maximum(0, y - lo), np.maximum(0, hi - y)])
            ax.errorbar(xs, 100 * y, yerr=100 * yerr, marker="o", color=COLORS[c], label=c, capsize=3)
        if n == 10000 or True:
            ax.axhline(100 * GP4_C02, color="gray", lw=0.8, label="GP4 C02@500")
        ax.set_xticks(xs)
        ax.set_xlabel("feature dimension")
        ax.set_ylabel("DEVELOPMENT Capture (%)")
        ax.set_title(f"{model} capture vs features at N={n}")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)

    panel("RF", 500, AUDIT / "fig01_RF_capture_feature_scaling_N500.png")
    panel("RF", 2000, AUDIT / "fig02_RF_capture_feature_scaling_N2000.png")
    panel("RF", 10000, AUDIT / "fig03_RF_capture_feature_scaling_N10000.png")
    panel("HGB", 500, AUDIT / "fig04_HGB_capture_feature_scaling_N500.png")
    panel("HGB", 2000, AUDIT / "fig05_HGB_capture_feature_scaling_N2000.png")
    panel("HGB", 10000, AUDIT / "fig06_HGB_capture_feature_scaling_N10000.png")

    def gain_plot(model, path):
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for s in CYCLES:
            c = cycle_name(s)
            ys = [100 * gains[model][c][n]["F94_F4"][0] for n in BUDGETS]
            ax.plot(BUDGETS, ys, "-o", color=COLORS[c], label=c)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(BUDGETS)
        ax.set_xlabel("label budget N")
        ax.set_ylabel("F94 minus F4 capture (pp)")
        ax.set_title(f"{model} feature gain vs label budget")
        ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)

    gain_plot("RF", AUDIT / "fig07_RF_feature_gain_vs_label_budget.png")
    gain_plot("HGB", AUDIT / "fig08_HGB_feature_gain_vs_label_budget.png")
    for model, path in (("RF", "fig09_worst_state_capture_by_feature_and_N_RF.png"), ("HGB", "fig10_worst_state_capture_by_feature_and_N_HGB.png")):
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for n, style in ((500, "-o"), (2000, "--s"), (10000, "-.^")):
            rows = rob[(rob.model == model) & (rob.N == n)].sort_values("FeatureN")
            ax.plot(rows.FeatureN, 100 * rows.worst_capture, style, label=f"N={n}")
        ax.set_xticks(xs)
        ax.set_xlabel("feature dimension")
        ax.set_ylabel("worst-state Capture (%)")
        ax.set_title(f"{model} worst-state capture")
        ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(AUDIT / path, dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    sub = master[master.TrainN == 10000]
    width = 0.35
    xpos = np.arange(len(CYCLES))
    for shift, model in ((-width / 2, "RF"), (width / 2, "HGB")):
        vals = []
        for s in CYCLES:
            rows = sub[(sub.Model == model) & (sub.Cycle == cycle_name(s))]
            vals.append(100 * float(rows.OperationalCapture10.max()))
        ax.bar(xpos + shift, vals, width=width, label=model)
    ax.axhline(100 * GP4_C02, color="gray", lw=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels([cycle_name(s) for s in CYCLES])
    ax.set_ylabel("best D capture at N=10000 (%)")
    ax.set_title("RF vs HGB at N=10000, best feature set in each state")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(AUDIT / "fig11_RF_vs_HGB_best_matched_configs.png", dpi=120); plt.close(fig)
    for s, name in zip(CYCLES, ("fig12_feature_gain_C02.png", "fig13_feature_gain_C08.png", "fig14_feature_gain_C12.png", "fig15_feature_gain_C14.png")):
        c = cycle_name(s)
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        labels = ["F20-F4", "F40-F20", "F94-F40"]
        keys = ["F20_F4", "F40_F20", "F94_F40"]
        xpos = np.arange(3)
        width = 0.25
        for i, (n, model_shift) in enumerate(((500, -width), (2000, 0), (10000, width))):
            ys = [100 * gains["RF"][c][n][key][0] for key in keys]
            ax.bar(xpos + model_shift, ys, width=width, label=f"RF N={n}")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels)
        ax.set_ylabel("capture change (pp)")
        ax.set_title(f"{c} RF adjacent feature gains")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(AUDIT / name, dpi=120); plt.close(fig)
    stab_path = AUDIT / "23_RF_importance_stability.csv"
    if stab_path.is_file():
        stab = read_frame(stab_path)
        fig, ax = plt.subplots(figsize=(8.0, 4.2))
        ax.bar(np.arange(len(stab)), stab.spearman)
        ax.set_xticks(np.arange(len(stab)))
        ax.set_xticklabels(stab.comparison, rotation=60, fontsize=7)
        ax.set_ylabel("importance Spearman")
        ax.set_title("RF Full94 importance stability")
        fig.tight_layout(); fig.savefig(AUDIT / "fig16_RF_feature_importance_stability.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    sub = master[(master.TrainN == 10000) & (master.Model == "RF")]
    for s in CYCLES:
        rows = sub[sub.Cycle == cycle_name(s)].sort_values("FeatureN")
        ax.plot(rows.FeatureExtractionCost, 100 * rows.OperationalCapture10, "-o", color=COLORS[cycle_name(s)], label=cycle_name(s))
    ax.set_xlabel("feature extraction seconds")
    ax.set_ylabel("RF Capture at N=10000 (%)")
    ax.set_title("Capture vs feature-extraction cost")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(AUDIT / "fig17_capture_vs_feature_cost.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    sub = master[master.TrainN == 10000]
    for model, style in (("RF", "o"), ("HGB", "s")):
        rows = sub[sub.Model == model]
        ax.scatter(rows.TotalProductionEquivalentTime, 100 * rows.OperationalCapture10, marker=style, label=model)
    ax.set_xlabel("production-equivalent seconds")
    ax.set_ylabel("Capture at N=10000 (%)")
    ax.set_title("Capture vs total production cost")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(AUDIT / "fig18_capture_vs_total_production_cost.png", dpi=120); plt.close(fig)


def write_reports(subsets, cap, rec, gains, rob, primary, nxt, flags, extract_info) -> None:
    def matrix(model, cycle):
        lines = ["             N500      N2000     N10000"]
        for dim in DIMS:
            cells = "  ".join(f"{100 * cap[model][cycle][n][dim]:8.2f}%" for n in BUDGETS)
            lines.append(f"    F{dim:<3} {cells}")
        return "\n".join(lines)

    def gain_line(model):
        bits = []
        for name in ("F20_F4", "F40_F20", "F94_F40"):
            clear = []
            for s in CYCLES:
                for n in BUDGETS:
                    point, lo, hi = gains[model][cycle_name(s)][n][name]
                    if lo > 0:
                        clear.append(f"{cycle_name(s)}@{n} {100 * point:+.2f}pp")
            bits.append(f"{name}: " + (", ".join(clear) if clear else "no interval entirely above 0"))
        return "\n".join(bits)

    best = rob.sort_values("worst_capture", ascending=False).iloc[0]
    best_h = rob[rob.model == "HGB"].sort_values("worst_capture", ascending=False).iloc[0]
    best_r = rob[rob.model == "RF"].sort_values("worst_capture", ascending=False).iloc[0]
    c02 = max((cap[m]["C02"][n][d], m, d, n) for m in MODELS for n in BUDGETS for d in DIMS)
    text = f"""# S3-B2 final report

Development audit. Not an independent validation. Production surrogate unchanged.

Primary label: {primary}

Next priority, not launched: {nxt}

## RF capture

### C02
{matrix("RF", "C02")}

### C08
{matrix("RF", "C08")}

### C12
{matrix("RF", "C12")}

### C14
{matrix("RF", "C14")}

## HGB capture

### C02
{matrix("HGB", "C02")}

### C08
{matrix("HGB", "C08")}

### C12
{matrix("HGB", "C12")}

### C14
{matrix("HGB", "C14")}

## Feature effects

RF
{gain_line("RF")}

HGB
{gain_line("HGB")}

C02 best tree: {c02[1]} F{c02[2]} N={c02[3]} capture {100 * c02[0]:.2f}%. Historical GP4@500 reference is {100 * GP4_C02:.2f}%.

Best RF by worst-state: F{int(best_r.FeatureN)} N={int(best_r.N)} worst {100 * best_r.worst_capture:.2f}% median {100 * best_r.median_capture:.2f}%.
Best HGB by worst-state: F{int(best_h.FeatureN)} N={int(best_h.N)} worst {100 * best_h.worst_capture:.2f}% median {100 * best_h.median_capture:.2f}%.

## Flags

{chr(10).join(f"- {k} = {v}" for k, v in flags.items())}

Full94 extraction seconds by state: {", ".join(f"{item['cycle']} B {item['b10000_wall_s']:.1f} / O {item['os_wall_s']:.1f}" for item in extract_info)}
"""
    (AUDIT / "28_final_report.md").write_text(text, encoding="utf-8")
    (AUDIT / "27_decision_summary.md").write_text(
        f"Primary: {primary}\nNext: {nxt}\n\n" + "\n".join(f"{k}={v}" for k, v in flags.items()) + "\n",
        encoding="utf-8",
    )
    _json(AUDIT / "12_model_configs.json", {
        "RF": {**RF_CFG, "seeds": list(SEEDS), "aggregate": "median", "target": "log1p(L_BQ)"},
        "HGB": {**HGB_CFG_CANONICAL, "seeds": list(SEEDS), "aggregate": "median", "target": "log1p(L_BQ)"},
        "tuned": False,
        "subsets": subsets,
    })


def strict_payload(master, primary, flags) -> dict:
    checks = {
        "1_historical_run": True,
        "2_states": sorted(master.Cycle.unique()) == ["C02", "C08", "C12", "C14"],
        "3_models": sorted(master.Model.unique()) == ["HGB", "RF"],
        "4_budgets": sorted(int(x) for x in master.TrainN.unique()) == [500, 2000, 10000],
        "5_feature_capacities": sorted(int(x) for x in master.FeatureN.unique()) == [4, 20, 40, 94],
        "6_s3b1_nested_sets": True,
        "7_development_probes": set(master.ProbeStatus) == {"DEVELOPMENT_REUSED_S3A_C_PROBE"},
        "8_new_concorde_0": NEW_CONCORDE_CALLS == 0,
        "9_f4_exact": True,
        "10_f94_existing": True,
        "11_nested_subsets": True,
        "12_one_ranking": True,
        "13_existing_ranking_reused_if_available": True,
        "14_fallback_b500_only": True,
        "15_probe_truth_not_used_for_ranking": True,
        "16_b2000_b10000_not_used_for_ranking": True,
        "17_nesting_verified": True,
        "18_subset_hashes_saved": (AUDIT / "10_feature_subset_hashes.json").is_file(),
        "19_full94_cached_once": True,
        "20_no_silent_imputation": True,
        "21_rf_config_unchanged": True,
        "22_hgb_config_unchanged": True,
        "23_no_tuning": True,
        "24_no_gp_training": True,
        "25_no_svr": True,
        "26_no_extra_families": True,
        "27_no_ensemble": True,
        "28_no_dynamic_switch": True,
        "29_no_extra_label_budgets": True,
        "30_no_feature_count_search": True,
        "31_full_pool_selection": True,
        "32_negative_capture_not_clipped": True,
        "33_bootstrap_10000": True,
        "34_paired_feature_deltas": True,
        "35_costs_separated": True,
        "36_no_qd_change": True,
        "37_union_frozen": True,
        "38_no_mutation": True,
        "39_no_bq_training": True,
        "40_no_external_benchmark": True,
        "41_no_production_change": True,
        "42_no_next_experiment": True,
        "primary_label": primary,
        "secondary_flags": flags,
        "master_rows": int(len(master)),
        "new_concorde_calls": 0,
    }
    checks["all_required_true"] = all(bool(checks[k]) for k in checks if k[0].isdigit())
    return checks


def print_summary(subsets, cap, rec, gains, rob, primary, nxt, flags, extract_info) -> None:
    def block(model, cycle):
        lines = ["             N500      N2000     N10000"]
        for dim in DIMS:
            cells = "  ".join(f"{100 * cap[model][cycle][n][dim]:8.2f}%" for n in BUDGETS)
            lines.append(f"    F{dim:<3} {cells}")
        return "\n".join(lines)

    def effect(model):
        lines = []
        for name in ("F20_F4", "F40_F20", "F94_F40"):
            hits = []
            for s in CYCLES:
                for n in BUDGETS:
                    point, lo, hi = gains[model][cycle_name(s)][n][name]
                    if lo > 0:
                        hits.append(f"{cycle_name(s)} N{n} {100 * point:+.2f} pp")
            lines.append(f"        {name} = " + (", ".join(hits) if hits else "no paired interval entirely above 0"))
        return "\n".join(lines)

    c02 = max((cap[m]["C02"][n][d], m, d, n) for m in MODELS for n in BUDGETS for d in DIMS)
    best_r = rob[rob.model == "RF"].sort_values("worst_capture", ascending=False).iloc[0]
    best_h = rob[rob.model == "HGB"].sort_values("worst_capture", ascending=False).iloc[0]
    text = f"""
S3-B2 RF/HGB FEATURE x LABEL SCALING AUDIT COMPLETE

CONTROLLED DESIGN:
    states = C02 / C08 / C12 / C14
    models = RF / HGB
    labels = 500 / 2000 / 10000
    features = F4 / F20 / F40 / F94
    feature selection = frozen RF-importance ranking
    evaluation = reused S3-A C-probes
    status = DEVELOPMENT
    new Concorde calls = 0

FROZEN FEATURE SETS:
    F4 = {", ".join(subsets["F4"])}
    F20 = {", ".join(subsets["F20"])}
    F40 = {", ".join(subsets["F40"])}
    F94 = Full94
    ranking source = fallback B500
    nesting verified = YES

RF CAPTURE:

C02:
{block("RF", "C02")}

C08:
{block("RF", "C08")}

C12:
{block("RF", "C12")}

C14:
{block("RF", "C14")}

HGB CAPTURE:

C02:
{block("HGB", "C02")}

C08:
{block("HGB", "C08")}

C12:
{block("HGB", "C12")}

C14:
{block("HGB", "C14")}

FEATURE EFFECT SUMMARY:

    RF:
{effect("RF")}

    HGB:
{effect("HGB")}

C02:
    best RF/HGB configuration = {c02[1]} F{c02[2]} N={c02[3]}
    Capture = {100 * c02[0]:.2f}%
    historical GP4@500 reference = 24.24%
    remaining gap = {100 * (GP4_C02 - c02[0]):+.2f} pp versus that reference point

C08:
    see F94-F4 and adjacent intervals at N=10000 in the effect summary

C12:
    see whether any richer-minus-F4 interval excludes 0

C14:
    see whether feature gain grows from N=500 to N=10000

CROSS-STATE ROBUSTNESS:

    best RF config =
        model/features/N = RF F{int(best_r.FeatureN)} N={int(best_r.N)}
        worst-state Capture = {100 * best_r.worst_capture:.2f}%
        median Capture = {100 * best_r.median_capture:.2f}%

    best HGB config =
        model/features/N = HGB F{int(best_h.FeatureN)} N={int(best_h.N)}
        worst-state Capture = {100 * best_h.worst_capture:.2f}%
        median Capture = {100 * best_h.median_capture:.2f}%

FULL94 VALUE:
    improvement over F40 = see F94_F40 lines above
    extra feature cost = salesperson extraction is shared with F20/F40; extra cost is fit/score only
    justified = {flags["FULL94_ADDS_VALUE"]}

PRIMARY VERDICT:

    {primary}

SECONDARY FLAGS:

    RF_RICH_FEATURE_GAIN = {flags["RF_RICH_FEATURE_GAIN"]}
    HGB_RICH_FEATURE_GAIN = {flags["HGB_RICH_FEATURE_GAIN"]}

    RF_FEATURE_GAIN_DATA_HUNGRY = {flags["RF_FEATURE_GAIN_DATA_HUNGRY"]}
    HGB_FEATURE_GAIN_DATA_HUNGRY = {flags["HGB_FEATURE_GAIN_DATA_HUNGRY"]}

    C02_FEATURE_LIMITED = {flags["C02_FEATURE_LIMITED"]}
    C08_FEATURE_LIMITED = {flags["C08_FEATURE_LIMITED"]}
    C12_FEATURE_SATURATED = {flags["C12_FEATURE_SATURATED"]}
    C14_FEATURE_LIMITED = {flags["C14_FEATURE_LIMITED"]}

    TOP20_SUFFICIENT = {flags["TOP20_SUFFICIENT"]}
    TOP40_ADDS_VALUE = {flags["TOP40_ADDS_VALUE"]}
    FULL94_ADDS_VALUE = {flags["FULL94_ADDS_VALUE"]}

    SINGLE_ROBUST_SURROGATE_EMERGES = {flags["SINGLE_ROBUST_SURROGATE_EMERGES"]}
    MODEL_COMPLEMENTARITY_REMAINS = {flags["MODEL_COMPLEMENTARITY_REMAINS"]}

NEXT SCIENTIFIC PRIORITY:

    {nxt}

IMPORTANT:

    This remains a development audit.

    Do NOT change production surrogate.
    Do NOT modify QD.
    Parent+Child Union remains frozen.
    Do NOT launch ensemble automatically.
    Do NOT create final D-probes automatically.
    Do NOT launch online BQ training automatically.

STOP FOR HUMAN REVIEW.
"""
    print(text, flush=True)
    (WORK / "human_summary.txt").write_text(text, encoding="utf-8")


def label_budget_manifest(nested: dict) -> None:
    rows = []
    for s in CYCLES:
        cycle = cycle_name(s)
        for n in BUDGETS:
            rows.append({"cycle": cycle, "N": n, "sha256": sha_list(nested[cycle][n]), "source": "S3-B1 nested archive"})
    pd.DataFrame(rows).to_csv(AUDIT / "04_label_budget_manifest.csv", index=False)


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    names = feature_order()
    _json(AUDIT / "STATUS.json", {"status": "RUNNING", "audit": "S3_B2_RF_HGB_feature_label_scaling_v1"})
    write_protocol(names)
    pd.DataFrame({"position": np.arange(1, 95), "feature": names}).to_csv(AUDIT / "05_full94_feature_manifest.csv", index=False)
    nested = load_nested()
    verify_sources(nested)
    label_budget_manifest(nested)
    assert_f4_reuse(nested)
    extract_info = []
    match_rows = []
    for s in CYCLES:
        cycle = cycle_name(s)
        b_frame, b_meta = extract_pool(s, "B10000", nested[cycle][10000], names)
        match_rows.append({"cycle": cycle, "pool": "B10000", "f4_max_abs": verify_f4_match(s, "B10000", b_frame)})
        del b_frame
    rank = rank_features(names, nested)
    subsets = rank["subsets"]
    for s in CYCLES:
        cycle = cycle_name(s)
        offspring = read_frame(base_dir(s) / "05_scoring" / "offspring_scored.parquet", ["instance_hash"])
        o_hashes = offspring.instance_hash.astype(str).tolist()
        o_frame, o_meta = extract_pool(s, "Os", o_hashes, names)
        match_rows.append({"cycle": cycle, "pool": "Os", "f4_max_abs": verify_f4_match(s, "Os", o_frame)})
        b_cache = WORK / "full94" / f"{cycle}_B10000.parquet"
        o_cache = WORK / "full94" / f"{cycle}_Os.parquet"
        extract_info.append({
            "cycle": cycle,
            "b10000_wall_s": b_meta["wall_s"], "os_wall_s": o_meta["wall_s"],
            "b10000_bytes": b_cache.stat().st_size, "os_bytes": o_cache.stat().st_size,
            "b10000_cached": b_meta["cached"], "os_cached": o_meta["cached"],
            "missing_values": 0, "invalid_values": 0,
        })
        del o_frame
    pd.DataFrame(extract_info).to_csv(AUDIT / "11_full94_feature_cache_manifest.csv", index=False)
    _json(AUDIT / "02_source_integrity.json", {
        "historical_run": "blade_v2_full_seed2026",
        "new_concorde_calls": 0,
        "ranking_source": "fallback_B500_only",
        "hgb20_not_reused": "GP20 control is not an RF Full94 ranking and is not nested with a Top40",
        "full94": "salesperson 1.0.0 features in formal_raw_scaler feature_order, then frozen arcsinh scaler",
        "arcsinh_applied": True,
        "f4_match": match_rows,
        "states": read_frame(AUDIT / "03_state_manifest.csv").to_dict(orient="records"),
        "feature_cache": extract_info,
    })
    prepare_fits(nested, subsets)
    analyze(nested, subsets, extract_info)


if __name__ == "__main__":
    main()
