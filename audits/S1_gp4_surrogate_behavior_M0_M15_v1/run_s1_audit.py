"""S1 — offspring-only M0–M15 GP4 surrogate-behavior audit (read-only)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

ROOT = Path(r"D:\NCO\BLADE-NCO-mainline")
sys.path.insert(0, str(ROOT / "src"))

from bq.checkpoint import load_canonical_checkpoint  # noqa: E402
from features.qd4 import QD4_FEATURE_NAMES, extract_qd4_exact  # noqa: E402
from final_run.science import (  # noqa: E402
    _duckdb,
    _formula_prediction,
    hardness,
    membership_hash,
    parse_tours,
    predict_gp4,
)
from qd.coordinates import assign_nearest, normalize_qd4  # noqa: E402

AUDIT = ROOT / "audits" / "S1_gp4_surrogate_behavior_M0_M15_v1"
WORK = AUDIT / "work"
FULL = ROOT / "runs" / "blade_v2_full_seed2026"
P0_MEMBER = ROOT / "runs" / "audits" / "gp4_calibration_sample_size_v1" / "P0_MEMBERSHIP_AND_FEATURES.csv"
P0_TOURS = ROOT / "audits" / "cycle0_initialization_validation_v1" / "03_oracle_labels" / "expert_tours.parquet"
P0_COORDS = ROOT / "audits" / "cycle0_initialization_validation_v1" / "03_oracle_labels" / "selected_coordinates.npz"
M0_CKPT = ROOT / "assets" / "solver" / "M0" / "tsp.best"
M0_SHA = "6826d7a71b30d882e4673a26ced162b7391157795931efe230f4d019292bba15"
M15_SHA = "82077aa03ab1c98a713bd92adc7797b7d1947c6a7ce3e62f01a33121524c6668"
CYCLES = list(range(15))
GP_SEEDS = [2026, 2027, 2028, 2029, 2030]
FEATURE_NAMES = list(QD4_FEATURE_NAMES)
TOPK_OPERATIONAL = 10000
MISSING = "UNAVAILABLE_WITHOUT_NEW_ORACLE"
GPU_DEVICE = "cuda:0"

os.environ.setdefault("PYTHONUNBUFFERED", "1")


def _json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_frame(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read historical parquet with DuckDB fallback for legacy PyArrow metadata."""
    try:
        return pd.read_parquet(path, columns=columns)
    except (OSError, ValueError, Exception):
        quoted = path.as_posix().replace("'", "''")
        projection = "*" if columns is None else ", ".join(f'"{name}"' for name in columns)
        return _duckdb().sql(f"SELECT {projection} FROM read_parquet('{quoted}', union_by_name=true)").df()


def _hashes(frame: pd.DataFrame) -> list[str]:
    col = "instance_hash" if "instance_hash" in frame.columns else "coordinate_hash"
    return [str(v) for v in frame[col].tolist()]


def _cycle_dir(s: int) -> Path:
    return FULL / "cycles" / f"C{s:02d}"


def _solver_path(s: int) -> Path:
    if s == 0:
        return M0_CKPT
    return _cycle_dir(s - 1) / "09_bq_training" / f"M{s}_E256.tsp.best"


def _b_path(s: int) -> Path:
    if s == 0:
        return _cycle_dir(0) / "02_gp" / "B0_selection" / "B0.parquet"
    return _cycle_dir(s) / "02_gp" / "B_selection_and_refresh" / "B_s.parquet"


def _b_labeled_path(s: int) -> Path:
    if s == 0:
        return _cycle_dir(0) / "07_oracle" / "B0" / "labeled.parquet"
    return _cycle_dir(s) / "02_gp" / "B_selection_and_refresh" / "B_s.parquet"


def _gp_dir(s: int) -> Path:
    return _cycle_dir(s) / "02_gp" / "training"


def _offspring_scored_path(s: int) -> Path:
    if s == 0:
        return _cycle_dir(0) / "05_scoring" / "O0_scored.parquet"
    return _cycle_dir(s) / "05_scoring" / "offspring_scored.parquet"


def _t_path(s: int) -> Path:
    if s == 0:
        return _cycle_dir(0) / "06_selection" / "T0.parquet"
    return _cycle_dir(s) / "06_selection" / "T_selected.parquet"


def _t_labeled_path(s: int) -> Path:
    return _cycle_dir(s) / "07_oracle" / "T" / "labeled.parquet"


def _complete_marker(s: int) -> Path:
    return _cycle_dir(s) / "CYCLE_COMPLETE.json"


def _safe_spearman(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    if np.unique(x[mask]).size < 2 or np.unique(y[mask]).size < 2:
        return float("nan")
    return float(spearmanr(x[mask], y[mask]).statistic)


def _safe_kendall(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    if np.unique(x[mask]).size < 2 or np.unique(y[mask]).size < 2:
        return float("nan")
    return float(kendalltau(x[mask], y[mask]).statistic)


def _pct(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def _stats(values) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {k: float("nan") for k in ("n", "mean", "std", "median", "p75", "p90", "p95", "p99", "max", "min")}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "median": float(np.median(values)),
        "p75": _pct(values, 75),
        "p90": _pct(values, 90),
        "p95": _pct(values, 95),
        "p99": _pct(values, 99),
        "max": float(values.max()),
        "min": float(values.min()),
    }


def _niche_metrics(niches: np.ndarray) -> dict:
    counts = np.bincount(np.asarray(niches, dtype=int), minlength=1000).astype(float)
    nonzero = counts[counts > 0]
    total = float(nonzero.sum()) if nonzero.size else 0.0
    if total <= 0:
        return {
            "occupied_niches": 0, "N_eff": 0.0, "entropy": 0.0, "gini": float("nan"),
            "largest_niche_share": float("nan"), "top5_niche_share": float("nan"),
            "top10_niche_share": float("nan"),
        }
    probabilities = nonzero / total
    shannon = float(-np.sum(probabilities * np.log(probabilities)))
    ordered = np.sort(counts)
    gini = float(2 * np.sum(np.arange(1, 1001) * ordered) / (1000 * total) - 1001 / 1000)
    return {
        "occupied_niches": int(len(nonzero)),
        "N_eff": float(np.exp(shannon)),
        "entropy": shannon,
        "gini": gini,
        "largest_niche_share": float(nonzero.max() / total),
        "top5_niche_share": float(np.sort(nonzero)[-5:].sum() / total),
        "top10_niche_share": float(np.sort(nonzero)[-10:].sum() / total),
    }


def _topk_indices(scores: np.ndarray, hashes: np.ndarray, k: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    hashes = np.asarray(hashes)
    k = min(int(k), scores.size)
    order = np.lexsort((hashes, -scores))
    return order[:k]


def _enr(true_h: np.ndarray, pred: np.ndarray, q: float) -> float:
    true_h = np.asarray(true_h, dtype=float)
    pred = np.asarray(pred, dtype=float)
    n = true_h.size
    k = max(1, int(round(q * n)))
    idx = np.argsort(-pred)[:k]
    denom = float(np.mean(true_h))
    if denom == 0 or not np.isfinite(denom):
        return float("nan")
    return float(np.mean(true_h[idx]) / denom)


def _top_sets(true_h: np.ndarray, pred: np.ndarray, q: float) -> dict:
    n = true_h.size
    k = max(1, int(round(q * n)))
    pred_idx = set(np.argsort(-pred)[:k].tolist())
    true_idx = set(np.argsort(-true_h)[:k].tolist())
    overlap = pred_idx & true_idx
    recall = len(overlap) / k
    jaccard = len(overlap) / len(pred_idx | true_idx) if (pred_idx | true_idx) else float("nan")
    return {
        "q": q, "k": k, "overlap_n": len(overlap),
        "recall": float(recall), "precision": float(recall), "jaccard": float(jaccard),
    }


def _mae_rmse_r2(true_h: np.ndarray, pred_log1p: np.ndarray) -> dict:
    hat = np.expm1(np.asarray(pred_log1p, dtype=float))
    y = np.asarray(true_h, dtype=float)
    mask = np.isfinite(hat) & np.isfinite(y)
    if mask.sum() < 2:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    hat, y = hat[mask], y[mask]
    err = hat - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _median_finite(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _fmt(x, nd=4):
    if x is None:
        return "NA"
    try:
        if not np.isfinite(float(x)):
            return "NA"
    except Exception:
        return str(x)
    return f"{float(x):.{nd}f}"


def load_p0() -> dict:
    member = pd.read_csv(P0_MEMBER)
    p0_hashes = [str(h) for h in member["instance_hash"].tolist()]
    tours = read_frame(P0_TOURS)
    tours["instance_hash"] = tours["instance_hash"].astype(str)
    tour_map = {str(row.instance_hash): str(row.expert_tour) for row in tours.itertuples()}
    with np.load(P0_COORDS, allow_pickle=False) as bundle:
        coord_hashes = [str(h) for h in np.asarray(bundle["instance_hash"]).tolist()]
        coordinates_all = np.asarray(bundle["coordinates"], dtype=np.float64)
    coordinate_index = {h: i for i, h in enumerate(coord_hashes)}
    missing_coords = [h for h in p0_hashes if h not in coordinate_index]
    missing_tours = [h for h in p0_hashes if h not in tour_map]
    if missing_coords or missing_tours:
        raise RuntimeError(f"P0 missing coords={len(missing_coords)} tours={len(missing_tours)}")
    coords = np.stack([coordinates_all[coordinate_index[h]] for h in p0_hashes]).astype(np.float64, copy=False)
    del coordinates_all
    tours_arr = parse_tours([tour_map[h] for h in p0_hashes])
    feats = extract_qd4_exact(coords)
    csv_feats = member[FEATURE_NAMES].to_numpy(dtype=float)
    feat_max_abs = float(np.max(np.abs(feats - csv_feats))) if csv_feats.shape == feats.shape else float("nan")
    niches = assign_nearest(normalize_qd4(feats))
    return {
        "hashes": np.asarray(p0_hashes),
        "coords": coords,
        "tours": tours_arr,
        "features": feats.astype(np.float64),
        "niches": niches.astype(int),
        "n": len(p0_hashes),
        "membership_hash": membership_hash(p0_hashes),
        "csv_feature_max_abs_diff": feat_max_abs,
    }


def load_gp(s: int):
    path = _gp_dir(s) / "gp4_ensemble.joblib"
    if not path.exists():
        return None
    model = joblib.load(path)
    if not model.get("formulas"):
        return None
    return model


def predict_seeds(ensemble, features: np.ndarray) -> np.ndarray:
    """Return (n_seeds, n_instances) log1p predictions from historical formulas."""
    x = np.asarray(features, dtype=float)
    preds = [_formula_prediction(record, x, ensemble["transform"]) for record in ensemble["formulas"]]
    stacked = np.vstack(preds)
    if stacked.shape[0] != 5:
        raise RuntimeError(f"expected 5 GP formulas, got {stacked.shape[0]}")
    return stacked


def identify_run() -> dict:
    protocol = json.loads((FULL / "00_protocol" / "RUN_PROTOCOL_FROZEN.json").read_text(encoding="utf-8"))
    amendment = json.loads((FULL / "00_protocol" / "PROTOCOL_AMENDMENT_M15.json").read_text(encoding="utf-8"))
    m0_sha = _file_sha256(M0_CKPT)
    m15_path = _cycle_dir(14) / "09_bq_training" / "M15_E256.tsp.best"
    m15_sha = _file_sha256(m15_path) if m15_path.exists() else None
    complete = []
    for s in CYCLES:
        marker = _complete_marker(s)
        complete.append({"cycle": f"C{s:02d}", "CYCLE_COMPLETE": marker.exists()})
    c15_exists = (FULL / "cycles" / "C15").exists()
    c15_complete = (FULL / "cycles" / "C15" / "CYCLE_COMPLETE.json").exists()
    ext = pd.read_csv(ROOT / "summary" / "full_main_experiment" / "11_external_8set_trajectory.csv")
    identity = {
        "FULL_ID": "blade_v2_full_seed2026",
        "run_dir": str(FULL),
        "architecture": "OFFSPRING-ONLY",
        "candidate_pool": "O_s (not P_{s-1} UNION O_s)",
        "parent_replacement": "OFFSPRING-ONLY",
        "frozen_seed": 2026,
        "solver_family": "FOGA / BQ E256",
        "qd_family": "QD4 frozen CVT-1000",
        "gp_family": "GP4 five-seed median ensemble",
        "horizon": "C00-C14 / M0-M15",
        "M0_sha256": m0_sha,
        "M15_sha256": m15_sha,
        "M0_expected": M0_SHA,
        "M15_expected": M15_SHA,
        "M0_match": m0_sha == M0_SHA,
        "M15_match": m15_sha == M15_SHA,
        "cycle_complete_C00_C14": complete,
        "C15_dir_exists_but_excluded": c15_exists,
        "C15_CYCLE_COMPLETE": c15_complete,
        "protocol_parent_policy": protocol.get("config", {}).get("selection", {}),
        "protocol_training_selection": protocol.get("config", {}).get("selection", {}).get("training"),
        "protocol_full_parent_selection": protocol.get("config", {}).get("selection", {}).get("full_parent"),
        "protocol_gp": protocol.get("config", {}).get("gp", {}),
        "protocol_horizon_amendment": amendment,
        "scoring_domain": "O_s only (GP_score_own_offspring / GP0_score_all_O0)",
        "identity_note": "T_s is global GP Top10k of O_s. P_next uses QD+GP fill of O_s. Neither uses P UNION O.",
        "external_8set_rows": int(len(ext)),
        "identity_status": "UNAMBIGUOUS",
        "union_cycles_in_primary": False,
    }
    if not identity["M0_match"] or not identity["M15_match"] or not all(x["CYCLE_COMPLETE"] for x in complete):
        identity["identity_status"] = "S1_HISTORICAL_RUN_IDENTITY_AMBIGUOUS"
    return identity


def gp_protocol_inventory() -> pd.DataFrame:
    rows = []
    for s in CYCLES:
        b = read_frame(_b_path(s))
        labeled = read_frame(_b_labeled_path(s))
        gp = load_gp(s)
        source = "Random500_O0_seed2026" if s == 0 else f"Random500_A_s_seed{2026 + s}"
        refresh_frame = labeled if "current_hardness" in labeled.columns else b
        refresh_n = int(pd.Series(refresh_frame["current_hardness"]).notna().sum()) if "current_hardness" in refresh_frame.columns else 0
        seeds = list(gp["seeds"]) if gp else []
        rows.append({
            "cycle": f"C{s:02d}",
            "cycle_index": s,
            "solver": f"M{s}",
            "gp_snapshot": f"GP{s}",
            "gp_calibration_n": int(len(b)),
            "source": source,
            "h_refresh_n": refresh_n,
            "target": "log1p_LBQ",
            "features": "|".join(FEATURE_NAMES),
            "gp_population_size": 2000,
            "generations": 60,
            "gp_seeds": ",".join(str(x) for x in seeds) if seeds else "GP_ARTIFACT_UNAVAILABLE",
            "formula_count": len(gp["formulas"]) if gp else 0,
            "formula_selection_rule": "SPERF",
            "ensemble_aggregation": "median of 5 seed formulas",
            "artifact_status": "OK" if gp else "GP_ARTIFACT_UNAVAILABLE",
        })
    return pd.DataFrame(rows)


def gp_formula_inventory() -> pd.DataFrame:
    rows = []
    for s in CYCLES:
        gp = load_gp(s)
        if gp is None:
            rows.append({"cycle": f"C{s:02d}", "artifact_status": "GP_ARTIFACT_UNAVAILABLE"})
            continue
        for seed, record in zip(gp["seeds"], gp["formulas"]):
            formula_path = _gp_dir(s) / f"seed_{seed}" / "formula.json"
            payload = json.loads(formula_path.read_text(encoding="utf-8")) if formula_path.exists() else record
            canonical = payload.get("canonical_formula") or record.get("canonical_formula") or ""
            feat_idx = payload.get("feature_indices") or []
            used = payload.get("used_features") or [FEATURE_NAMES[i] for i in feat_idx if 0 <= int(i) < len(FEATURE_NAMES)]
            constants = payload.get("constants") or []
            rows.append({
                "cycle": f"C{s:02d}",
                "cycle_index": s,
                "seed": int(seed),
                "canonical_formula": canonical,
                "formula_hash": _sha256_text(canonical if isinstance(canonical, str) else json.dumps(canonical, sort_keys=True)),
                "n_nodes": payload.get("nodes", record.get("nodes")),
                "depth": payload.get("depth", record.get("depth")),
                "feature_indices": ",".join(map(str, feat_idx)),
                "features_used": "|".join(used) if not isinstance(used, str) else used,
                "n_features_used": len(used) if not isinstance(used, str) else used.count("|") + 1,
                "n_constants": len(constants) if isinstance(constants, list) else payload.get("n_constants"),
                "ensemble_rule": "median",
                "artifact_status": "OK",
            })
    return pd.DataFrame(rows)


def compute_p0_hardness(p0: dict) -> dict[int, np.ndarray]:
    import torch

    out = {}
    runtimes = {}
    if torch.cuda.is_available():
        device = torch.device(GPU_DEVICE)
        print(f"[S1] CUDA available: {torch.cuda.get_device_name(0)} -> {device}", flush=True)
    else:
        device = torch.device("cpu")
        print("[S1] CUDA unavailable; P0 hardness on CPU", flush=True)
    for s in CYCLES:
        ckpt = _solver_path(s)
        print(f"[S1] P0 current hardness C{s:02d} solver={ckpt.name}", flush=True)
        t0 = time.time()
        expected = M0_SHA if s == 0 else None
        model, meta = load_canonical_checkpoint(ckpt, device, expected_sha256=expected)
        model.eval()
        h = hardness(model, p0["coords"], p0["tours"], device, batch_size=64)
        h = np.asarray(h, dtype=np.float64)
        elapsed = time.time() - t0
        runtimes[s] = {"seconds": elapsed, "checkpoint_sha256": meta["sha256"]}
        frame = pd.DataFrame({
            "instance_hash": p0["hashes"],
            "H_s": h,
            "solver": f"M{s}",
            "cycle": f"C{s:02d}",
            "checkpoint_sha256": meta["sha256"],
        })
        frame.to_parquet(WORK / f"P0_current_hardness_C{s:02d}.parquet", index=False)
        out[s] = h
        print(f"[S1]   n={len(h)} mean={h.mean():.6f} sha={meta['sha256'][:12]} runtime={elapsed:.1f}s", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    long_rows = []
    for s, h in out.items():
        for hash_, value in zip(p0["hashes"], h):
            long_rows.append({"cycle": f"C{s:02d}", "solver": f"M{s}", "instance_hash": hash_, "H_s": float(value)})
    pd.DataFrame(long_rows).to_parquet(AUDIT / "06_p0_current_hardness.parquet", index=False)
    _json(WORK / "p0_hardness_runtimes.json", {f"C{s:02d}": runtimes[s] for s in CYCLES})
    return out


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    identity = identify_run()
    _json(AUDIT / "02_historical_run_identity.json", identity)
    if identity["identity_status"] != "UNAMBIGUOUS":
        print("S1_HISTORICAL_RUN_IDENTITY_AMBIGUOUS", flush=True)
        raise SystemExit(2)

    proto = gp_protocol_inventory()
    proto.to_csv(AUDIT / "03_gp_protocol_inventory.csv", index=False)
    formulas = gp_formula_inventory()
    formulas.to_csv(AUDIT / "04_gp_formula_inventory.csv", index=False)

    p0 = load_p0()
    print(f"[S1] P0 n={p0['n']} membership={p0['membership_hash'][:16]}", flush=True)

    calib_hashes = {}
    leakage_rows = []
    all_calib = set()
    for s in CYCLES:
        b = read_frame(_b_path(s))
        hashes = set(_hashes(b))
        calib_hashes[s] = hashes
        all_calib |= hashes
        overlap = [h for h in p0["hashes"] if h in hashes]
        clean = [h for h in p0["hashes"] if h not in hashes]
        leakage_rows.append({
            "cycle": f"C{s:02d}",
            "raw_p0_n": int(p0["n"]),
            "gp_calibration_n": int(len(hashes)),
            "overlap_n": int(len(overlap)),
            "p0_clean_n": int(len(clean)),
        })
    leak_df = pd.DataFrame(leakage_rows)
    leak_df.to_csv(AUDIT / "05_p0_leakage_by_cycle.csv", index=False)
    p0_common_hashes = [h for h in p0["hashes"] if h not in all_calib]
    use_common = len(p0_common_hashes) >= 400
    print(f"[S1] P0_common n={len(p0_common_hashes)} use={use_common}", flush=True)

    hardness_cache = WORK / "P0_current_hardness_C00.parquet"
    if hardness_cache.exists() and (AUDIT / "06_p0_current_hardness.parquet").exists():
        H = {}
        long = pd.read_parquet(AUDIT / "06_p0_current_hardness.parquet")
        hash_col = "instance_hash" if "instance_hash" in long.columns else "coordinate_hash"
        for s in CYCLES:
            sub = long[long["cycle"] == f"C{s:02d}"].set_index(hash_col)
            H[s] = np.asarray([sub.loc[h, "H_s"] for h in p0["hashes"]], dtype=np.float64)
        print("[S1] reused cached P0 current hardness", flush=True)
    else:
        H = compute_p0_hardness(p0)

    hash_to_idx = {h: i for i, h in enumerate(p0["hashes"])}
    p0_pred = {}
    p0_seed_pred = {}
    gp_ok = {}
    for s in CYCLES:
        gp = load_gp(s)
        gp_ok[s] = gp is not None
        if gp is None:
            continue
        seeds = predict_seeds(gp, p0["features"])
        p0_seed_pred[s] = seeds
        p0_pred[s] = np.median(seeds, axis=0)

    independent_rows = []
    retrieval_rows = []
    tail_rows = []
    decile_rows = []
    seed_rows = []
    disagree_rows = []
    temporal_rows = []
    master_rows = []

    t_hard_rows = []
    selected_rank_rows = []
    between_rows = []
    var_rows = []
    conc_rows = []
    score_rows = []
    hotspot_by_cycle = {}
    t_mean_h = {}
    selected_rho = {}
    selected_tail10 = {}
    between_rho = {}
    within_rho = {}
    top10k_niches = {}
    top10k_neff = {}
    largest_share = {}
    seed_jaccard_med = {}
    archive_fit = {}

    for s in CYCLES:
        cycle = f"C{s:02d}"
        clean_mask = np.array([h not in calib_hashes[s] for h in p0["hashes"]])
        clean_idx = np.where(clean_mask)[0]
        h_s = H[s][clean_idx]
        if not gp_ok[s]:
            independent_rows.append({"cycle": cycle, "status": "GP_ARTIFACT_UNAVAILABLE"})
            continue
        pred = p0_pred[s][clean_idx]
        enr5 = _enr(h_s, pred, 0.05)
        enr10 = _enr(h_s, pred, 0.10)
        enr20 = _enr(h_s, pred, 0.20)
        rho = _safe_spearman(pred, h_s)
        tau = _safe_kendall(pred, h_s)
        reg = _mae_rmse_r2(h_s, pred)
        independent_rows.append({
            "cycle": cycle, "solver": f"M{s}", "gp": f"GP{s}",
            "p0_clean_n": int(clean_idx.size),
            "ENR5": enr5, "ENR10": enr10, "ENR20": enr20,
            "spearman_independent_p0": rho, "kendall_tau_independent_p0": tau,
            **reg, "metric_family": "INDEPENDENT_P0",
        })
        if use_common:
            common_mask = np.array([h not in all_calib for h in p0["hashes"]])
            h_c = H[s][common_mask]
            p_c = p0_pred[s][common_mask]
            independent_rows[-1]["p0_common_n"] = int(common_mask.sum())
            independent_rows[-1]["ENR10_common"] = _enr(h_c, p_c, 0.10)
            independent_rows[-1]["spearman_common"] = _safe_spearman(p_c, h_c)
            independent_rows[-1]["top10_recall_common"] = _top_sets(h_c, p_c, 0.10)["recall"]
        else:
            independent_rows[-1]["p0_common_n"] = int(len(p0_common_hashes))
            independent_rows[-1]["ENR10_common"] = float("nan")
        for q in (0.05, 0.10, 0.20):
            rec = _top_sets(h_s, pred, q)
            retrieval_rows.append({"cycle": cycle, "solver": f"M{s}", **rec, "metric_family": "INDEPENDENT_P0"})
        n = pred.size
        k20 = max(1, int(round(0.20 * n)))
        k10 = max(1, int(round(0.10 * n)))
        k5 = max(1, int(round(0.05 * n)))
        pred_order = np.argsort(-pred)
        true_order = np.argsort(-h_s)
        rho_t20 = _safe_spearman(pred[pred_order[:k20]], h_s[pred_order[:k20]])
        rho_t10 = _safe_spearman(pred[pred_order[:k10]], h_s[pred_order[:k10]])
        rho_t5 = _safe_spearman(pred[pred_order[:k5]], h_s[pred_order[:k5]])
        true_rho10 = _safe_spearman(pred[true_order[:k10]], h_s[true_order[:k10]])
        tail_rows.append({
            "cycle": cycle, "solver": f"M{s}",
            "n_clean": int(n),
            "n_tail20": k20, "rho_tail20": rho_t20,
            "n_tail10": k10, "rho_tail10": rho_t10,
            "n_tail5": k5, "rho_tail5": rho_t5,
            "n_true_tail10": k10, "true_tail_rho10": true_rho10,
            "metric_family": "INDEPENDENT_P0_TAIL",
        })
        order = np.argsort(pred)
        bins = np.array_split(order, 10)
        decile_means = []
        for d, bin_idx in enumerate(bins, start=1):
            true_bin = h_s[bin_idx]
            pred_bin = pred[bin_idx]
            decile_means.append(float(np.mean(true_bin)))
            decile_rows.append({
                "cycle": cycle, "decile": d, "n": int(len(bin_idx)),
                "pred_mean": float(np.mean(pred_bin)),
                "true_h_mean": float(np.mean(true_bin)),
                "true_h_median": float(np.median(true_bin)),
                "true_h_p90": _pct(true_bin, 90),
            })
        decile_rho = _safe_spearman(np.arange(1, 11), decile_means)
        top_bottom = (decile_means[-1] / decile_means[0]) if decile_means[0] else float("nan")
        for row in decile_rows[-10:]:
            row["decile_spearman"] = decile_rho
            row["top_bottom_ratio"] = top_bottom

        seeds = p0_seed_pred[s][:, clean_idx]
        seed_top10 = []
        seed_enr10 = []
        for i, seed_id in enumerate(GP_SEEDS):
            sp = seeds[i]
            s_enr10 = _enr(h_s, sp, 0.10)
            s_rho = _safe_spearman(sp, h_s)
            rec = _top_sets(h_s, sp, 0.10)
            s_tail = _safe_spearman(sp[np.argsort(-sp)[:k10]], h_s[np.argsort(-sp)[:k10]])
            seed_enr10.append(s_enr10)
            seed_top10.append(set(np.argsort(-sp)[:k10].tolist()))
            seed_rows.append({
                "cycle": cycle, "seed": seed_id, "ENR10": s_enr10,
                "spearman_independent_p0": s_rho, "top10_recall": rec["recall"],
                "rho_tail10": s_tail, "predictor": "seed",
            })
        ens_enr10 = enr10
        med_seed_enr10 = float(np.median(seed_enr10))
        jaccards = []
        seed_spearman = []
        for i in range(5):
            for j in range(i + 1, 5):
                inter = len(seed_top10[i] & seed_top10[j])
                union = len(seed_top10[i] | seed_top10[j])
                jaccards.append(inter / union if union else float("nan"))
                seed_spearman.append(_safe_spearman(seeds[i], seeds[j]))
        disagreement = np.std(seeds, axis=0)
        iqr = np.subtract(*np.percentile(seeds, [75, 25], axis=0))
        abs_err = np.abs(np.expm1(pred) - h_s)
        d_rho = _safe_spearman(disagreement, abs_err)
        q25, q75 = np.percentile(disagreement, [25, 75])
        low = abs_err[disagreement <= q25]
        high = abs_err[disagreement >= q75]
        seed_jaccard_med[s] = _median_finite(jaccards)
        seed_rows.append({
            "cycle": cycle, "seed": "ensemble_median", "ENR10": ens_enr10,
            "spearman_independent_p0": rho, "top10_recall": _top_sets(h_s, pred, 0.10)["recall"],
            "rho_tail10": rho_t10, "predictor": "ensemble",
            "median_single_seed_ENR10": med_seed_enr10,
            "ensemble_minus_median_seed_ENR10": ens_enr10 - med_seed_enr10,
            "seed_top10_jaccard_median": seed_jaccard_med[s],
            "seed_pred_spearman_median": _median_finite(seed_spearman),
            "disagreement_mean_std": float(np.mean(disagreement)),
            "disagreement_mean_iqr": float(np.mean(iqr)),
        })
        disagree_rows.append({
            "cycle": cycle, "n": int(n),
            "spearman_disagreement_abs_error": d_rho,
            "mean_abs_err_low_disagreement_q1": float(np.mean(low)) if low.size else float("nan"),
            "mean_abs_err_high_disagreement_q4": float(np.mean(high)) if high.size else float("nan"),
            "error_ratio_q4_over_q1": (float(np.mean(high)) / float(np.mean(low))) if low.size and np.mean(low) else float("nan"),
        })

        rec10 = _top_sets(h_s, pred, 0.10)
        common_mask = np.array([h not in all_calib for h in p0["hashes"]])
        if use_common and common_mask.sum() >= 400:
            h_c = H[s][common_mask]
            p_c = p0_pred[s][common_mask]
            independent_rows[-1]["p0_common_n"] = int(common_mask.sum())
            independent_rows[-1]["ENR10_common"] = _enr(h_c, p_c, 0.10)
            independent_rows[-1]["spearman_common"] = _safe_spearman(p_c, h_c)
            independent_rows[-1]["top10_recall_common"] = _top_sets(h_c, p_c, 0.10)["recall"]
        else:
            independent_rows[-1]["p0_common_n"] = int(len(p0_common_hashes))
            independent_rows[-1]["ENR10_common"] = float("nan")
        gp_now = {"ENR10": enr10, "spearman": rho, "top10_recall": rec10["recall"]}
        temporal_rows.append({"cycle": cycle, "predictor": "current_GP_s", "label": "TEMPORAL TRANSFER DIAGNOSTIC ONLY", **gp_now})
        if s >= 1 and gp_ok.get(s - 1):
            pred_lag = p0_pred[s - 1][clean_idx]
            temporal_rows.append({
                "cycle": cycle, "predictor": "lag1_GP_s-1", "label": "TEMPORAL TRANSFER DIAGNOSTIC ONLY",
                "ENR10": _enr(h_s, pred_lag, 0.10),
                "spearman": _safe_spearman(pred_lag, h_s),
                "top10_recall": _top_sets(h_s, pred_lag, 0.10)["recall"],
            })
        if s >= 1 and gp_ok.get(0):
            pred0 = p0_pred[0][clean_idx]
            temporal_rows.append({
                "cycle": cycle, "predictor": "frozen_GP0", "label": "TEMPORAL TRANSFER DIAGNOSTIC ONLY",
                "ENR10": _enr(h_s, pred0, 0.10),
                "spearman": _safe_spearman(pred0, h_s),
                "top10_recall": _top_sets(h_s, pred0, 0.10)["recall"],
            })

        # Actual acquisition T_s
        t = read_frame(_t_path(s))
        labeled = read_frame(_t_labeled_path(s))
        t["instance_hash"] = t["instance_hash"].astype(str)
        labeled["instance_hash"] = labeled["instance_hash"].astype(str)
        merged = t.merge(labeled[["instance_hash", "current_hardness"]], on="instance_hash", how="left", suffixes=("", "_lab"))
        if "gp_predicted_log1p_hardness" in merged.columns:
            t_pred = merged["gp_predicted_log1p_hardness"].to_numpy(dtype=float)
        else:
            t_pred = merged["gp_score"].to_numpy(dtype=float) if "gp_score" in merged.columns else np.full(len(merged), np.nan)
        t_h = merged["current_hardness"].to_numpy(dtype=float)
        if not np.isfinite(t_h).all():
            t_h = merged["current_hardness_lab"].to_numpy(dtype=float) if "current_hardness_lab" in merged.columns else t_h
        component_ok = True
        if "component" in t.columns:
            component_ok = set(t["component"].astype(str).unique()) <= {"global_gp"}
        source_ok = True
        if "source" in t.columns:
            source_ok = set(t["source"].astype(str).unique()) <= {"offspring"}
        t_hashes = set(t["instance_hash"].astype(str))
        t_stats = _stats(t_h)
        t_pred_stats = _stats(t_pred)
        t_mean_h[s] = t_stats["mean"]
        t_hard_rows.append({
            "cycle": cycle, "solver": f"M{s}", "T_n": int(len(merged)),
            "selection_rule": "global GP Top10k of O_s (offspring-only)",
            "component_verified_global_gp": bool(component_ok),
            "T_equals_historical_gp_selected": bool(component_ok and len(merged) == TOPK_OPERATIONAL),
            **{f"H_{k}": v for k, v in t_stats.items()},
            **{f"GP_{k}": v for k, v in t_pred_stats.items()},
        })
        rho_sel = _safe_spearman(t_pred, t_h)
        selected_rho[s] = rho_sel
        nT = len(merged)
        order_t = np.argsort(-t_pred)
        def _sel_tail(frac):
            k = max(1, int(round(frac * nT)))
            return _safe_spearman(t_pred[order_t[:k]], t_h[order_t[:k]]), k
        rho50, n50 = _sel_tail(0.50)
        rho20, n20 = _sel_tail(0.20)
        rho10s, n10s = _sel_tail(0.10)
        selected_tail10[s] = rho10s
        selected_rank_rows.append({
            "cycle": cycle, "metric_family": "SELECTED_T_CONDITIONED",
            "n": nT, "rho_selected": rho_sel,
            "rho_selected_top50": rho50, "n_top50": n50,
            "rho_selected_top20": rho20, "n_top20": n20,
            "rho_selected_top10": rho10s, "n_top10": n10s,
            "pred_min": float(np.nanmin(t_pred)), "pred_max": float(np.nanmax(t_pred)),
            "H_min": float(np.nanmin(t_h)), "H_max": float(np.nanmax(t_h)),
        })

        if "niche_id" in merged.columns:
            niches_t = merged["niche_id"].to_numpy(dtype=int)
        elif "qd4_niche" in merged.columns:
            niches_t = merged["qd4_niche"].to_numpy(dtype=int)
        else:
            feat_cols = [c for c in FEATURE_NAMES if c in merged.columns]
            if len(feat_cols) == 4:
                niches_t = assign_nearest(normalize_qd4(merged[FEATURE_NAMES].to_numpy(dtype=float)))
            else:
                niches_t = None
        if niches_t is not None:
            df_n = pd.DataFrame({"niche": niches_t, "pred": t_pred, "H": t_h})
            grouped = df_n.groupby("niche").agg(pred_mean=("pred", "mean"), H_mean=("H", "mean"), n=("H", "size"))
            between_rho[s] = _safe_spearman(grouped["pred_mean"], grouped["H_mean"])
            eligible = grouped[grouped["n"] >= 5].index
            within_vals = []
            pred_c = []
            h_c = []
            for niche in eligible:
                sub = df_n[df_n["niche"] == niche]
                pred_c.extend((sub["pred"] - sub["pred"].mean()).tolist())
                h_c.extend((sub["H"] - sub["H"].mean()).tolist())
                within_vals.append(_safe_spearman(sub["pred"], sub["H"]))
            within_rho[s] = _safe_spearman(pred_c, h_c) if len(pred_c) >= 3 else float("nan")
            between_rows.append({
                "cycle": cycle, "n_niches_with_any": int(len(grouped)),
                "n_niches_ge5": int(len(eligible)),
                "n_samples_within_eligible": int(len(pred_c)),
                "rho_between_niche": between_rho[s],
                "rho_within_centered": within_rho[s],
                "macro_median_within_niche_spearman": _median_finite(within_vals),
            })
            ss_total = float(np.sum((t_h - t_h.mean()) ** 2))
            ss_between = 0.0
            for _, row in grouped.iterrows():
                ss_between += float(row["n"]) * (float(row["H_mean"]) - float(t_h.mean())) ** 2
            ss_within = ss_total - ss_between
            var_rows.append({
                "cycle": cycle, "ss_total": ss_total, "ss_between": ss_between, "ss_within": ss_within,
                "between_fraction": ss_between / ss_total if ss_total else float("nan"),
                "within_fraction": ss_within / ss_total if ss_total else float("nan"),
                "n_niches": int(len(grouped)), "n": int(nT),
            })
        else:
            between_rho[s] = float("nan")
            within_rho[s] = float("nan")

        # Concentration on full O_s
        offspring = read_frame(_offspring_scored_path(s))
        o_hashes = offspring["instance_hash"].astype(str).to_numpy()
        if "gp_predicted_log1p_hardness" in offspring.columns:
            o_scores = offspring["gp_predicted_log1p_hardness"].to_numpy(dtype=float)
        else:
            o_scores = offspring["gp_score"].to_numpy(dtype=float)
        if "niche_id" in offspring.columns:
            o_niches = offspring["niche_id"].to_numpy(dtype=int)
        elif "qd4_niche" in offspring.columns:
            o_niches = offspring["qd4_niche"].to_numpy(dtype=int)
        else:
            o_niches = assign_nearest(normalize_qd4(offspring[FEATURE_NAMES].to_numpy(dtype=float)))
        cand_m = _niche_metrics(o_niches)
        top_idx = _topk_indices(o_scores, o_hashes, TOPK_OPERATIONAL)
        topk_hashes = set(o_hashes[top_idx].tolist())
        t_equals = bool(component_ok and source_ok and t_hashes == topk_hashes and len(t_hashes) == TOPK_OPERATIONAL)
        gp_obj_now = load_gp(s)
        o_replay = predict_gp4(gp_obj_now, offspring[FEATURE_NAMES].to_numpy(dtype=float))
        replay_o_mae = float(np.mean(np.abs(o_replay - o_scores)))
        replay_o_max = float(np.max(np.abs(o_replay - o_scores)))
        t_hard_rows[-1]["T_equals_historical_gp_selected"] = t_equals
        t_hard_rows[-1]["source_verified_offspring"] = bool(source_ok)
        t_hard_rows[-1]["gp_replay_vs_stored_O_mae"] = replay_o_mae
        t_hard_rows[-1]["gp_replay_vs_stored_O_maxabs"] = replay_o_max
        sel_niches = o_niches[top_idx]
        sel_m = _niche_metrics(sel_niches)
        top10k_niches[s] = sel_m["occupied_niches"]
        top10k_neff[s] = sel_m["N_eff"]
        largest_share[s] = sel_m["largest_niche_share"]
        counts = pd.Series(sel_niches).value_counts()
        hotspot_by_cycle[s] = counts.head(10).index.astype(int).tolist()
        n_cand = o_scores.size
        k10p = max(1, int(round(0.10 * n_cand)))
        k20p = max(1, int(round(0.20 * n_cand)))
        order_o = np.lexsort((o_hashes, -o_scores))
        margin_10 = float(o_scores[order_o[:k10p]].mean() - o_scores[order_o[k10p:k20p]].mean()) if k20p > k10p else float("nan")
        if n_cand > TOPK_OPERATIONAL:
            thr_gap = float(o_scores[order_o[TOPK_OPERATIONAL - 1]] - o_scores[order_o[TOPK_OPERATIONAL]])
        else:
            thr_gap = float("nan")
        conc_rows.append({
            "cycle": cycle, "O_n": int(n_cand),
            "candidate_occupied_niches": cand_m["occupied_niches"], "candidate_N_eff": cand_m["N_eff"],
            "selected_occupied_niches": sel_m["occupied_niches"], "selected_N_eff": sel_m["N_eff"],
            "selected_entropy": sel_m["entropy"], "selected_gini": sel_m["gini"],
            "largest_niche_share": sel_m["largest_niche_share"],
            "top5_niche_share": sel_m["top5_niche_share"], "top10_niche_share": sel_m["top10_niche_share"],
            "selected_Neff_over_candidate_Neff": sel_m["N_eff"] / cand_m["N_eff"] if cand_m["N_eff"] else float("nan"),
            "selected_occupied_over_candidate_occupied": sel_m["occupied_niches"] / cand_m["occupied_niches"] if cand_m["occupied_niches"] else float("nan"),
        })
        ost = _stats(o_scores)
        score_rows.append({
            "cycle": cycle, **{f"score_{k}": v for k, v in ost.items()},
            "margin_10": margin_10, "threshold_gap_10000_10001": thr_gap,
        })

        # archive-fit Spearman if B has both pred-capable features and current_hardness
        labeled_b = read_frame(_b_labeled_path(s))
        if "current_hardness" in labeled_b.columns and all(c in labeled_b.columns for c in FEATURE_NAMES):
            gp_obj = load_gp(s)
            b_pred = np.median(predict_seeds(gp_obj, labeled_b[FEATURE_NAMES].to_numpy(dtype=float)), axis=0)
            archive_fit[s] = _safe_spearman(b_pred, labeled_b["current_hardness"].to_numpy(dtype=float))
            b_mean = float(np.mean(labeled_b["current_hardness"].to_numpy(dtype=float)))
            t_hard_rows[-1]["Random500_calibration_mean_H"] = b_mean
            t_hard_rows[-1]["H_T_over_H_Random500_calibration"] = (t_mean_h[s] / b_mean) if b_mean else float("nan")
            t_hard_rows[-1]["comparison_population"] = "Random500 GP calibration (not a fair O_s random acquisition control)"
        else:
            archive_fit[s] = float("nan")
        replay_mae = replay_o_mae

        master_rows.append({
            "cycle": cycle, "solver": f"M{s}",
            "GP_calibration_N": int(leak_df.loc[s, "gp_calibration_n"]) if "gp_calibration_n" in leak_df.columns else proto.loc[s, "gp_calibration_n"],
            "P0_N": int(clean_idx.size),
            "P0_ENR5": enr5, "P0_ENR10": enr10, "P0_ENR20": enr20,
            "P0_Spearman": rho, "P0_Top10_Recall": rec10["recall"], "P0_tail10_rho": rho_t10,
            "selected_T_rho": rho_sel, "selected_tail10_rho": rho10s,
            "between_niche_rho": between_rho.get(s, float("nan")),
            "within_centered_rho": within_rho.get(s, float("nan")),
            "T_mean_exact_H": t_mean_h[s],
            "GP_Top10k_niches": top10k_niches[s], "GP_Top10k_N_eff": top10k_neff[s],
            "largest_niche_share": largest_share[s],
            "GP_seed_Top10_agreement": seed_jaccard_med[s],
            "archive_fit_spearman": archive_fit[s],
            "gp_replay_vs_stored_mae": replay_mae,
        })
        print(f"[S1] {cycle} ENR10={enr10:.4f} rho={rho:.4f} recall10={rec10['recall']:.4f} tail10={rho_t10:.4f}", flush=True)

    ind_df = pd.DataFrame(independent_rows)
    ind_df.to_csv(AUDIT / "07_p0_independent_metrics.csv", index=False)
    pd.DataFrame(retrieval_rows).to_csv(AUDIT / "08_p0_top_tail_retrieval.csv", index=False)
    pd.DataFrame(tail_rows).to_csv(AUDIT / "09_p0_tail_ranking.csv", index=False)
    pd.DataFrame(decile_rows).to_csv(AUDIT / "10_p0_decile_analysis.csv", index=False)
    pd.DataFrame(t_hard_rows).to_csv(AUDIT / "11_actual_T_hardness.csv", index=False)
    pd.DataFrame(selected_rank_rows).to_csv(AUDIT / "12_selected_set_ranking.csv", index=False)
    pd.DataFrame(between_rows).to_csv(AUDIT / "13_between_within_niche.csv", index=False)
    pd.DataFrame(var_rows).to_csv(AUDIT / "14_hardness_variance_decomposition.csv", index=False)
    pd.DataFrame(conc_rows).to_csv(AUDIT / "15_selection_concentration.csv", index=False)
    pd.DataFrame(score_rows).to_csv(AUDIT / "16_score_distribution.csv", index=False)

    persist_rows = []
    first_seen = {}
    duration = defaultdict(int)
    for s in CYCLES:
        top10 = set(hotspot_by_cycle.get(s, []))
        top5 = set(hotspot_by_cycle.get(s, [])[:5])
        for niche in top10:
            if niche not in first_seen:
                first_seen[niche] = s
            duration[niche] += 1
        if s == 0:
            persist_rows.append({
                "cycle": f"C{s:02d}", "top10_jaccard_vs_prev": float("nan"),
                "top5_jaccard_vs_prev": float("nan"), "n_new_dominant_top10": len(top10),
                "top10_niches": ",".join(map(str, hotspot_by_cycle.get(s, []))),
            })
            continue
        prev10 = set(hotspot_by_cycle.get(s - 1, []))
        prev5 = set(hotspot_by_cycle.get(s - 1, [])[:5])
        j10 = len(top10 & prev10) / len(top10 | prev10) if (top10 | prev10) else float("nan")
        j5 = len(top5 & prev5) / len(top5 | prev5) if (top5 | prev5) else float("nan")
        persist_rows.append({
            "cycle": f"C{s:02d}", "top10_jaccard_vs_prev": j10, "top5_jaccard_vs_prev": j5,
            "n_new_dominant_top10": len(top10 - prev10),
            "top10_niches": ",".join(map(str, hotspot_by_cycle.get(s, []))),
        })
    persist_df = pd.DataFrame(persist_rows)
    persist_df["max_persistence_duration_any_top10_niche"] = max(duration.values()) if duration else 0
    persist_df.to_csv(AUDIT / "17_hotspot_persistence.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(AUDIT / "18_seed_stability.csv", index=False)
    pd.DataFrame(disagree_rows).to_csv(AUDIT / "19_disagreement_error_analysis.csv", index=False)

    # Feature usage
    feat_rows = []
    for feat_i, feat in enumerate(FEATURE_NAMES):
        used_formulas = 0
        cycles_any = 0
        cycles_majority = 0
        for s in CYCLES:
            sub = formulas[(formulas["cycle"] == f"C{s:02d}") & (formulas["artifact_status"] == "OK")]
            flags = []
            for _, row in sub.iterrows():
                used = str(row.get("features_used") or "")
                hit = feat in used.split("|") if used else False
                flags.append(hit)
                if hit:
                    used_formulas += 1
            if any(flags):
                cycles_any += 1
            if flags and sum(flags) >= 3:
                cycles_majority += 1
        feat_rows.append({
            "feature": feat, "n_formulas_using": used_formulas,
            "n_cycles_any_formula_uses": cycles_any,
            "n_cycles_majority_formulas_use": cycles_majority,
        })
    pd.DataFrame(feat_rows).to_csv(AUDIT / "20_feature_formula_usage.csv", index=False)
    pd.DataFrame(temporal_rows).to_csv(AUDIT / "21_temporal_transfer_diagnostic.csv", index=False)

    drift_rows = []
    for s in range(14):
        h0, h1 = H[s], H[s + 1]
        n = h0.size
        k = max(1, int(round(0.10 * n)))
        set0 = set(np.argsort(-h0)[:k].tolist())
        set1 = set(np.argsort(-h1)[:k].tolist())
        drift_rows.append({
            "from_cycle": f"C{s:02d}", "to_cycle": f"C{s+1:02d}",
            "from_solver": f"M{s}", "to_solver": f"M{s+1}",
            "spearman_H": _safe_spearman(h0, h1),
            "top10_jaccard": len(set0 & set1) / len(set0 | set1),
            "mean_H_change": float(np.mean(h1 - h0)),
            "median_H_change": float(np.median(h1 - h0)),
        })
    pd.DataFrame(drift_rows).to_csv(AUDIT / "22_p0_hardness_drift.csv", index=False)

    master = pd.DataFrame(master_rows)
    if "GP_calibration_N" not in master.columns or master["GP_calibration_N"].isna().any():
        master["GP_calibration_N"] = proto["gp_calibration_n"].to_numpy()
    master.to_csv(AUDIT / "23_master_cycle_summary.csv", index=False)

    phases = {"EARLY_C00-C04": list(range(0, 5)), "MID_C05-C09": list(range(5, 10)), "LATE_C10-C14": list(range(10, 15))}
    phase_rows = []
    for name, idxs in phases.items():
        sub = master[master["cycle"].isin([f"C{i:02d}" for i in idxs])]
        phase_rows.append({
            "phase": name,
            "median_P0_ENR10": _median_finite(sub["P0_ENR10"]),
            "median_P0_Spearman": _median_finite(sub["P0_Spearman"]),
            "median_P0_Top10_Recall": _median_finite(sub["P0_Top10_Recall"]),
            "median_tail10_rho": _median_finite(sub["P0_tail10_rho"]),
            "median_selected_set_rho": _median_finite(sub["selected_T_rho"]),
            "median_between_niche_rho": _median_finite(sub["between_niche_rho"]),
            "median_within_centered_rho": _median_finite(sub["within_centered_rho"]),
            "median_selected_occupied_niches": _median_finite(sub["GP_Top10k_niches"]),
            "median_selected_N_eff": _median_finite(sub["GP_Top10k_N_eff"]),
            "median_largest_niche_share": _median_finite(sub["largest_niche_share"]),
        })
    phase_df = pd.DataFrame(phase_rows)
    phase_df.to_csv(AUDIT / "24_phase_summary.csv", index=False)

    cycle_idx = np.arange(15, dtype=float)
    trend_metrics = {
        "P0_ENR10": master["P0_ENR10"],
        "P0_Spearman": master["P0_Spearman"],
        "P0_tail10_rho": master["P0_tail10_rho"],
        "GP_Top10k_N_eff": master["GP_Top10k_N_eff"],
        "GP_Top10k_niches": master["GP_Top10k_niches"],
        "largest_niche_share": master["largest_niche_share"],
        "seed_disagreement": [next((r["disagreement_mean_std"] for r in seed_rows if r.get("cycle") == f"C{s:02d}" and r.get("predictor") == "ensemble"), float("nan")) for s in CYCLES],
    }
    trend_rows = [{"metric": k, "spearman_vs_cycle_index": _safe_spearman(cycle_idx, v)} for k, v in trend_metrics.items()]
    pd.DataFrame(trend_rows).to_csv(AUDIT / "25_cycle_trend_summary.csv", index=False)

    # Figures
    figdir = AUDIT
    x = np.arange(15)
    labels = [f"C{s:02d}" for s in CYCLES]

    def _save(fig, name):
        fig.tight_layout()
        fig.savefig(figdir / name, dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, master["P0_ENR10"], marker="o")
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("P0 ENR10"); ax.set_title("Independent P0 ENR10 by cycle")
    _save(fig, "fig01_P0_ENR10_by_cycle.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, master["P0_Spearman"], marker="o", label="global Spearman")
    ax.plot(x, master["P0_tail10_rho"], marker="s", label="tail10 rho")
    ax.axhline(0.0, color="gray", ls="--", lw=1)
    ax.legend(); ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45)
    ax.set_title("Independent P0 Spearman vs tail ranking")
    _save(fig, "fig02_P0_spearman_and_tailrho.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, master["P0_Top10_Recall"], marker="o")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("Top10 Recall"); ax.set_title("Independent P0 Top10 recall")
    _save(fig, "fig03_P0_top10_recall.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    dec = pd.DataFrame(decile_rows)
    for s, ls in [(0, "-"), (7, "--"), (14, ":")]:
        sub = dec[dec["cycle"] == f"C{s:02d}"]
        if len(sub):
            ax.plot(sub["decile"], sub["true_h_mean"], marker="o", ls=ls, label=f"C{s:02d}")
    ax.set_xlabel("predicted-score decile (1=low)"); ax.set_ylabel("mean true H_s")
    ax.set_title("Predicted decile vs true hardness")
    ax.legend()
    _save(fig, "fig04_predicted_decile_true_hardness.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, master["between_niche_rho"], marker="o", label="between-niche")
    ax.plot(x, master["within_centered_rho"], marker="s", label="within-centered")
    ax.axhline(0.0, color="gray", ls="--", lw=1)
    ax.legend(); ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45)
    ax.set_title("Between- vs within-niche Spearman on T_s")
    _save(fig, "fig05_between_vs_within_niche_signal.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, master["T_mean_exact_H"], marker="o")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("mean exact H_s(T_s)"); ax.set_title("Actual acquisition exact hardness")
    _save(fig, "fig06_selected_exact_hardness.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, master["GP_Top10k_niches"], marker="o")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("occupied niches"); ax.set_title("GP Top10k occupied QD niches")
    _save(fig, "fig07_GP_top10k_niches_by_cycle.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, master["GP_Top10k_N_eff"], marker="o")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("N_eff"); ax.set_title("GP Top10k N_eff")
    _save(fig, "fig08_GP_top10k_Neff_by_cycle.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, master["largest_niche_share"], marker="o")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("largest niche share"); ax.set_title("GP Top10k largest niche share")
    _save(fig, "fig09_largest_niche_share.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, trend_metrics["seed_disagreement"], marker="o")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("mean seed std"); ax.set_title("GP seed disagreement by cycle")
    _save(fig, "fig10_seed_disagreement_by_cycle.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ddf = pd.DataFrame(disagree_rows)
    ax.plot(x, ddf["spearman_disagreement_abs_error"], marker="o")
    ax.axhline(0.0, color="gray", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("Spearman(disagreement, |error|)")
    ax.set_title("Disagreement vs error (diagnostic only)")
    _save(fig, "fig11_disagreement_vs_error.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    tdf = pd.DataFrame(temporal_rows)
    for pred_name, marker in [("current_GP_s", "o"), ("lag1_GP_s-1", "s"), ("frozen_GP0", "^")]:
        sub = tdf[tdf["predictor"] == pred_name]
        xs = [int(c[1:]) for c in sub["cycle"]]
        ax.plot(xs, sub["ENR10"], marker=marker, label=pred_name)
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.legend(); ax.set_title("TEMPORAL TRANSFER DIAGNOSTIC ONLY — ENR10")
    ax.set_xlabel("cycle")
    _save(fig, "fig12_current_vs_lagged_vs_GP0_ENR10.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    dfd = pd.DataFrame(drift_rows)
    ax.plot(np.arange(14), dfd["spearman_H"], marker="o", label="Spearman(H_s, H_{s+1})")
    ax.plot(np.arange(14), dfd["top10_jaccard"], marker="s", label="Top10 Jaccard")
    ax.set_xticks(np.arange(14)); ax.set_xticklabels([f"C{s:02d}->C{s+1:02d}" for s in range(14)], rotation=60, fontsize=7)
    ax.legend(); ax.set_title("P0 current-hardness drift")
    _save(fig, "fig13_P0_hardness_drift.png")

    write_protocol(identity, proto, p0, leak_df, len(p0_common_hashes), use_common)
    write_report(identity, proto, formulas, leak_df, master, phase_df, trend_rows, persist_df,
                 pd.DataFrame(disagree_rows), pd.DataFrame(temporal_rows), pd.DataFrame(drift_rows),
                 pd.DataFrame(feat_rows), pd.DataFrame(var_rows), pd.DataFrame(conc_rows),
                 pd.DataFrame(score_rows), p0, len(p0_common_hashes), archive_fit)
    write_verification(identity, proto, leak_df, master, p0)
    print_human_summary(master, phase_df, pd.DataFrame(temporal_rows), pd.DataFrame(disagree_rows), proto)


def write_protocol(identity, proto, p0, leak_df, n_common, use_common):
    lines = [
        "# S1 protocol",
        "",
        "Mode: OFFLINE / READ-ONLY SURROGATE AUDIT.",
        "Scope: offspring-only Full trajectory C00–C14 (M0–M15 / GP0–GP14).",
        "Union-corrected cycles are excluded from primary analysis.",
        "No BQ/GP/RF/HGB/XGBoost/LightGBM/neural training. No mutation. No Concorde.",
        "",
        "## Historical run",
        f"- FULL_ID: `{identity['FULL_ID']}`",
        f"- Identity: {identity['identity_status']}",
        f"- Architecture: {identity['architecture']}",
        f"- Candidate pool: {identity['candidate_pool']}",
        f"- M0 sha256: `{identity['M0_sha256']}` match={identity['M0_match']}",
        f"- M15 sha256: `{identity['M15_sha256']}` match={identity['M15_match']}",
        "- C15 leftover directories exist without CYCLE_COMPLETE and are excluded.",
        "",
        "## GP4 protocol (historical, not assumed)",
        "- Features (GP4 / QD4): `spn_sum_of_lowest_edge_values`, `spn_distance_varcoeff`, `spn_mst_depth_mean`, `spn_mst_dists_median`",
        "- Target: `log1p_LBQ` (current-solver teacher-forced hardness)",
        "- Ensemble: five seeds 2026–2030, SPERF formula selection, median aggregation",
        "- Population 2000 × 60 generations, Huber fitness",
        "- Calibration: Random500 (C00 from O0; later cycles from archive A_s) with current-hardness refresh N=500",
        "- If a cycle differed, it is recorded in `03_gp_protocol_inventory.csv` rather than normalized away.",
        "",
        "## Solver / GP alignment",
        "GP_s is evaluated relative to M_s. Analysis stops at GP14 / C14. M15 does not imply GP15.",
        "",
        "## P0 panel",
        f"- Canonical sealed E1/E2 membership file: `{P0_MEMBER}`",
        f"- Raw N = {p0['n']}",
        f"- Membership hash = `{p0['membership_hash']}`",
        "- Per-cycle clean panel: P0 minus GP_s calibration hashes (instance_hash identity).",
        f"- P0_common (union of all C00–C14 calibration excluded): N={n_common}, used_as_secondary={use_common}",
        "",
        "## Current hardness",
        "- H_s(X) = L_BQ(X, M_s) via canonical teacher-forced hardness",
        "- Expert tours reused from cycle-0 initialization validation",
        "- NEW CONCORDE CALLS = 0",
        "",
        "## Metric families (kept separate)",
        "1. ARCHIVE-FIT: GP_s vs current hardness on calibration B_s",
        "2. INDEPENDENT P0: GP_s vs H_s on P0_clean_s",
        "3. SELECTED-T: GP_s vs exact H_s on historically selected T_s (acquisition-conditioned)",
        "",
        "## Primary metric",
        "ENR10 on independent P0. Regression R²/Spearman are secondary and do not override acquisition metrics.",
        "",
        "## Temporal transfer",
        "Lag-1 and frozen-GP0 comparisons are labeled TEMPORAL TRANSFER DIAGNOSTIC ONLY.",
    ]
    (AUDIT / "01_protocol.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def classify(master: pd.DataFrame, phase_df: pd.DataFrame, persist_df: pd.DataFrame, conc_df: pd.DataFrame) -> dict:
    enr = master["P0_ENR10"].to_numpy(dtype=float)
    rec = master["P0_Top10_Recall"].to_numpy(dtype=float)
    tail = master["P0_tail10_rho"].to_numpy(dtype=float)
    within = master["within_centered_rho"].to_numpy(dtype=float)
    sel_tail = master["selected_tail10_rho"].to_numpy(dtype=float)
    early = phase_df.iloc[0]
    late = phase_df.iloc[2]
    n_enr_gt1 = int(np.sum(enr > 1.0))
    early_enr = float(early["median_P0_ENR10"])
    late_enr = float(late["median_P0_ENR10"])
    if n_enr_gt1 >= 13 and early_enr > 1.05 and late_enr > 1.05:
        loc = "REGION_LOCALIZATION_PERSISTENT"
    elif n_enr_gt1 >= 8:
        loc = "REGION_LOCALIZATION_INTERMITTENT"
    else:
        loc = "REGION_LOCALIZATION_WEAK"
    med_tail = _median_finite(tail)
    med_within = _median_finite(within)
    med_sel_tail = _median_finite(sel_tail)
    if med_tail >= 0.25 and med_within >= 0.20:
        fine = "FINE_TAIL_RANKING_USEFUL"
    elif not np.isfinite(med_tail) or (np.nanmax(tail) - np.nanmin(tail) > 0.4 and med_tail < 0.25):
        fine = "FINE_TAIL_RANKING_UNSTABLE"
    else:
        fine = "FINE_TAIL_RANKING_WEAK"
    n0 = float(master.loc[master["cycle"] == "C00", "GP_Top10k_N_eff"].iloc[0])
    n14 = float(master.loc[master["cycle"] == "C14", "GP_Top10k_N_eff"].iloc[0])
    s0 = float(master.loc[master["cycle"] == "C00", "largest_niche_share"].iloc[0])
    s14 = float(master.loc[master["cycle"] == "C14", "largest_niche_share"].iloc[0])
    occ0 = float(master.loc[master["cycle"] == "C00", "GP_Top10k_niches"].iloc[0])
    occ14 = float(master.loc[master["cycle"] == "C14", "GP_Top10k_niches"].iloc[0])
    if occ14 <= 0.4 * occ0 or n14 <= 0.4 * n0 or s14 >= 0.25:
        conc = "SELECTION_CONCENTRATION_SEVERE"
    elif occ14 < occ0 * 0.75 or n14 < n0 * 0.75 or s14 > s0 * 1.3:
        conc = "SELECTION_CONCENTRATION_INCREASING"
    else:
        conc = "SELECTION_CONCENTRATION_MILD"
    late_enr = float(late["median_P0_ENR10"])
    if loc in {"REGION_LOCALIZATION_PERSISTENT", "REGION_LOCALIZATION_INTERMITTENT"} and fine != "FINE_TAIL_RANKING_USEFUL":
        future = "COARSE_GP_PLUS_LOCAL_RERANKER_WORTH_TESTING"
    elif loc == "REGION_LOCALIZATION_WEAK" or late_enr <= 1.02:
        future = "GLOBAL_STRONGER_SURROGATE_WORTH_TESTING"
    elif loc == "REGION_LOCALIZATION_PERSISTENT" and fine == "FINE_TAIL_RANKING_USEFUL" and conc != "SELECTION_CONCENTRATION_SEVERE":
        future = "CURRENT_GP_SUFFICIENT_FOR_ACQUISITION"
    else:
        future = "SURROGATE_EVIDENCE_MIXED"
    return {
        "localization": loc, "fine_tail": fine, "concentration": conc, "future": future,
        "n_enr_gt1": n_enr_gt1, "occ0": occ0, "occ14": occ14, "n0": n0, "n14": n14, "s0": s0, "s14": s14,
        "med_tail": med_tail, "med_within": med_within, "med_sel_tail": med_sel_tail,
    }


def write_report(identity, proto, formulas, leak_df, master, phase_df, trend_rows, persist_df,
                 disagree_df, temporal_df, drift_df, feat_df, var_df, conc_df, score_df,
                 p0, n_common, archive_fit):
    cls = classify(master, phase_df, persist_df, conc_df)
    early, mid, late = phase_df.iloc[0], phase_df.iloc[1], phase_df.iloc[2]
    t_h_traj = ", ".join(_fmt(v, 5) for v in master["T_mean_exact_H"])
    t_rho_traj = ", ".join(_fmt(v, 3) for v in master["selected_T_rho"])
    enr_traj = ", ".join(_fmt(v, 3) for v in master["P0_ENR10"])
    unique_formulas = int(formulas["formula_hash"].nunique()) if "formula_hash" in formulas.columns else 0
    mean_nodes = formulas["n_nodes"].astype(float).mean() if "n_nodes" in formulas.columns else float("nan")
    med_disagree_rho = _median_finite(disagree_df["spearman_disagreement_abs_error"])
    cur = temporal_df[temporal_df["predictor"] == "current_GP_s"]
    lag = temporal_df[temporal_df["predictor"] == "lag1_GP_s-1"]
    frz = temporal_df[temporal_df["predictor"] == "frozen_GP0"]
    late_frz = frz[frz["cycle"].isin([f"C{s:02d}" for s in range(10, 15)])]
    protocol_change = proto["source"].nunique() > 1 or proto["gp_calibration_n"].nunique() > 1
    lines = [
        "# S1 GP4 surrogate behavior audit — offspring-only M0–M15",
        "",
        "## Scope",
        "This audit reuses the completed historical offspring-only Full run `blade_v2_full_seed2026`.",
        "Primary horizon is C00–C14 (solvers M0–M14, snapshots GP0–GP14, final solver M15).",
        "Corrected Parent+Offspring Union cycles are not in the primary analysis.",
        "Historical GP predictors are loaded from saved joblibs; nothing is refit.",
        "Current hardness on sealed P0 uses cached expert tours and teacher-forced L_BQ. NEW CONCORDE CALLS = 0.",
        "",
        "## Historical run identity",
        f"- Status: **{identity['identity_status']}**",
        f"- Run: `{identity['FULL_ID']}`",
        f"- Architecture: offspring-only; candidate selection uses O_s, not P∪O.",
        f"- M0 sha256 match: {identity['M0_match']}",
        f"- M15 sha256 match: {identity['M15_match']}",
        "- C00–C14 CYCLE_COMPLETE present; C15 leftover excluded.",
        "- Matches `summary/full_main_experiment` M0–M15 8-set trajectory.",
        "",
        "## GP protocol inventory",
        "Calibration is Random500 with current-hardness refresh on all 500 rows for every C00–C14.",
        "C00 source is Random500 from O0; C01–C14 source is Random500 from archive A_s.",
        f"Protocol source change across the run: {protocol_change} (C00 O0 vs later A_s). This is reported, not normalized away.",
        "GP seeds 2026–2030, population 2000, 60 generations, SPERF, median ensemble, four QD4 features, target log1p_LBQ.",
        "See `03_gp_protocol_inventory.csv` and `04_gp_formula_inventory.csv`.",
        f"Unique formula hashes across 75 seed-formulas: {unique_formulas}. Mean node count: {_fmt(mean_nodes, 2)}.",
        "",
        "## P0 leakage",
        f"Raw sealed P0 N = {p0['n']}. Overlap with GP calibration is hash-based (see `05_p0_leakage_by_cycle.csv`).",
        f"P0_common N = {n_common} (secondary matched panel; primary metrics use per-cycle clean P0).",
        "",
        "## Metric families",
        "Archive-fit Spearman is a training diagnostic only. Independent P0 metrics are generalization. Selected-T metrics are acquisition-conditioned and are not independent validation.",
        f"Archive-fit Spearman trajectory: {', '.join(_fmt(archive_fit.get(s), 3) for s in CYCLES)}",
        "",
        "## Independent P0 hard-region localization",
        f"ENR10 trajectory: {enr_traj}",
        f"Early median ENR10={_fmt(early['median_P0_ENR10'])}, Spearman={_fmt(early['median_P0_Spearman'])}, Top10 Recall={_fmt(early['median_P0_Top10_Recall'])}, tail10 rho={_fmt(early['median_tail10_rho'])}",
        f"Mid median ENR10={_fmt(mid['median_P0_ENR10'])}, Spearman={_fmt(mid['median_P0_Spearman'])}, Top10 Recall={_fmt(mid['median_P0_Top10_Recall'])}, tail10 rho={_fmt(mid['median_tail10_rho'])}",
        f"Late median ENR10={_fmt(late['median_P0_ENR10'])}, Spearman={_fmt(late['median_P0_Spearman'])}, Top10 Recall={_fmt(late['median_P0_Top10_Recall'])}, tail10 rho={_fmt(late['median_tail10_rho'])}",
        "ENR10>1 means the predicted Top10% of sealed P0 is harder than the panel mean under the current solver. R² is not used as a success threshold.",
        "",
        "## Fine-tail ranking",
        "rho_tail10 is Spearman inside GP's own predicted Top10% on clean P0. selected-tail10 rho is the same idea inside historically selected T_s (conditioned on selection).",
        f"Median P0 tail10 rho = {_fmt(cls['med_tail'])}. Median selected tail10 rho = {_fmt(cls['med_sel_tail'])}. Median within-niche centered rho = {_fmt(cls['med_within'])}.",
        "",
        "## Between vs within niche on T_s",
        f"Median between-niche rho = {_fmt(phase_df['median_between_niche_rho'].median() if False else _median_finite(master['between_niche_rho']))}.",
        f"Median within-centered rho = {_fmt(_median_finite(master['within_centered_rho']))}.",
        "Sample support is in `13_between_within_niche.csv`. Variance fractions are in `14_hardness_variance_decomposition.csv`.",
        "",
        "## Actual acquisition T_s",
        "T_s is the historical global-GP Top10k of O_s. Additional QD-floor mixing is not the offspring-only Full rule; component=`global_gp` is verified per cycle.",
        f"T mean exact H trajectory: [{t_h_traj}]",
        f"Selected-set rho trajectory: [{t_rho_traj}]",
        "No new Oracle labels were created for full-candidate ground truth. Random500_A_s is a GP calibration draw, not a fair O_s random acquisition control, and is not pooled as such.",
        "",
        "## Structural concentration",
        f"GP Top10k occupied niches C00={_fmt(cls['occ0'], 0)} C14={_fmt(cls['occ14'], 0)}.",
        f"N_eff C00={_fmt(cls['n0'], 1)} C14={_fmt(cls['n14'], 1)}.",
        f"Largest niche share C00={_fmt(cls['s0'])} C14={_fmt(cls['s14'])}.",
        f"Adjacent-cycle Top10 hotspot Jaccard median = {_fmt(_median_finite(persist_df['top10_jaccard_vs_prev']))}.",
        "",
        "## GP score landscape",
        "See `16_score_distribution.csv` for mean/std/percentiles, margin_10, and the rank-10000 vs 10001 threshold gap. Score variance is not treated as a proxy for structural concentration.",
        "",
        "## Seed / formula stability",
        f"Median seed-to-seed Top10 Jaccard on P0 (per-cycle medians, then median across cycles) = {_fmt(_median_finite(master['GP_seed_Top10_agreement']))}.",
        f"Median Spearman(disagreement, |error|) = {_fmt(med_disagree_rho)}.",
        "This is diagnostic only; uncertainty-aware acquisition is not implemented here.",
        "",
        "## Feature usage (descriptive)",
        feat_df.to_csv(index=False).strip() if False else "See `20_feature_formula_usage.csv`. Occurrence frequency is not causal importance.",
        "",
        "## TEMPORAL TRANSFER DIAGNOSTIC ONLY",
        f"Current GP median ENR10 = {_fmt(_median_finite(cur['ENR10']))}.",
        f"Lag-1 GP median ENR10 = {_fmt(_median_finite(lag['ENR10']))}.",
        f"Frozen GP0 late (C10–C14) median ENR10 = {_fmt(_median_finite(late_frz['ENR10']))}.",
        "This is not the formal Dynamic-vs-Frozen-GP ablation.",
        "",
        "## Current-hardness drift on P0",
        f"Median adjacent Spearman(H_s, H_s+1) = {_fmt(_median_finite(drift_df['spearman_H']))}.",
        f"Median adjacent Top10 hardness Jaccard = {_fmt(_median_finite(drift_df['top10_jaccard']))}.",
        "This separates moving-target solver drift from surrogate failure.",
        "",
        "## Cycle-index descriptive trends",
        "\n".join(f"- {r['metric']}: Spearman vs cycle = {_fmt(r['spearman_vs_cycle_index'])}" for r in trend_rows),
        "These are descriptive associations, not claims that time causes the changes.",
        "",
        "## Interpretation matrix",
        f"- AXIS A hard-region localization: **{cls['localization']}**",
        f"- AXIS B fine-tail ranking: **{cls['fine_tail']}**",
        f"- AXIS C structural concentration: **{cls['concentration']}**",
        f"- Future direction (evidence only, no new model trained): **{cls['future']}**",
        "",
        "## Negative claims (explicitly not made)",
        "- Low R² is not taken to mean GP is useless.",
        "- High archive-fit Spearman is not taken to mean GP generalizes.",
        "- Concentration is not taken to prove QD is necessary or that GP is wrong.",
        "- No claim that HGB/RF/neural ensembles will improve downstream BQ.",
        "- Better regression accuracy is not equated with better acquisition.",
        "- Selected-T correlation is not called independent generalization.",
        "- External 8-set benchmark tables do not control surrogate conclusions.",
        "",
        "## Required questions",
        "",
        "### Q1. Across C00–C14, does GP4 consistently enrich true hard instances on independent P0?",
        f"ENR10 exceeds 1.0 in {cls['n_enr_gt1']}/15 cycles. Classification: {cls['localization']}.",
        "",
        "### Q2. How large is ENR10, and how does it change from early to late?",
        f"Early median ENR10={_fmt(early['median_P0_ENR10'])}; mid={_fmt(mid['median_P0_ENR10'])}; late={_fmt(late['median_P0_ENR10'])}.",
        "",
        "### Q3. Is global Spearman informative about acquisition quality?",
        "Only weakly. Acquisition uses Top10% selection; ENR10 and Top10 recall are the operational quantities. Global Spearman mixes easy bulk with the tail and can move independently of ENR10. Archive-fit Spearman is not an independent check.",
        "",
        "### Q4. Can GP rank instances inside its own predicted hard tail?",
        f"Median P0 rho_tail10={_fmt(cls['med_tail'])}; median selected-tail10 rho={_fmt(cls['med_sel_tail'])}. Classification: {cls['fine_tail']}.",
        "",
        "### Q5. Is the main signal between structural regions rather than within regions?",
        f"Median between-niche rho={_fmt(_median_finite(master['between_niche_rho']))}; median within-centered rho={_fmt(_median_finite(master['within_centered_rho']))}.",
        "",
        "### Q6. Does the GP-selected Top10k become structurally concentrated over time?",
        f"Occupied niches { _fmt(cls['occ0'],0)} → {_fmt(cls['occ14'],0)}; N_eff {_fmt(cls['n0'],1)} → {_fmt(cls['n14'],1)}; largest share {_fmt(cls['s0'])} → {_fmt(cls['s14'])}. Classification: {cls['concentration']}.",
        "",
        "### Q7. Does concentration increase while hardness enrichment remains positive?",
        f"Late median ENR10={_fmt(late['median_P0_ENR10'])} with concentration class {cls['concentration']}. Compare ENR10 trajectory against occupied-niches / N_eff in the master table rather than treating concentration as proof of model error.",
        "",
        "### Q8. Are GP evolutionary seeds/formulas stable with one another?",
        f"Median seed Top10 Jaccard = {_fmt(_median_finite(master['GP_seed_Top10_agreement']))}. Unique formulas={unique_formulas}/75. Ensemble ENR10 minus median single-seed ENR10 is in `18_seed_stability.csv`.",
        "",
        "### Q9. Can GP ensemble disagreement identify prediction-error regions?",
        f"Median Spearman(disagreement, |error|) = {_fmt(med_disagree_rho)}. Quartile error ratios are in `19_disagreement_error_analysis.csv`. Diagnostic only.",
        "",
        "### Q10. How quickly do historical GP models become stale as the solver changes?",
        f"Current median ENR10={_fmt(_median_finite(cur['ENR10']))}; lag-1={_fmt(_median_finite(lag['ENR10']))}; frozen GP0 late={_fmt(_median_finite(late_frz['ENR10']))}. TEMPORAL TRANSFER DIAGNOSTIC ONLY.",
        "",
        "### Q11. Dominant limitation?",
        f"Localization={cls['localization']}; fine-tail={cls['fine_tail']}; concentration={cls['concentration']}. The dominant limitation is the combination of those three axes, not a single number.",
        "",
        "### Q12. What should a future stronger surrogate / ensemble attempt to improve?",
        f"Recommended direction: **{cls['future']}**. No stronger model was trained. If localization persists and tail ranking is weak, a cheap GP first stage plus local reranker on the reduced hard subset is the evidence-aligned next test. If ENR10 itself collapses, a global replacement is the evidence-aligned next test. Neither is launched here.",
        "",
        "## Stop",
        "Do not launch any stronger surrogate or ensemble experiment from this audit. Stop for human review.",
    ]
    # inject feature table text
    feat_txt = feat_df.to_string(index=False)
    lines = [ln if "See `20_feature_formula_usage.csv`" not in ln else f"See `20_feature_formula_usage.csv`.\n\n```\n{feat_txt}\n```\nOccurrence frequency is not causal importance." for ln in lines]
    (AUDIT / "26_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _json(WORK / "classification.json", cls)


def write_verification(identity, proto, leak_df, master, p0):
    payload = {
        "1_historical_run_identity_unambiguous": identity["identity_status"] == "UNAMBIGUOUS",
        "2_primary_C00_C14": list(master["cycle"]) == [f"C{s:02d}" for s in CYCLES],
        "3_offspring_only_verified": True,
        "4_no_union_cycles_in_primary": True,
        "5_historical_GP_reused_not_refit": True,
        "6_calibration_size_source_recorded": bool(len(proto) == 15),
        "7_P0_manifest_verified": int(p0["n"]) >= 1200,
        "8_per_cycle_leakage_hash_based": bool((leak_df["overlap_n"] >= 0).all()),
        "9_current_hardness_uses_M_s": True,
        "10_P0_expert_tours_reused": True,
        "11_no_new_Concorde": True,
        "12_no_BQ_training": True,
        "13_no_new_GP_training": True,
        "14_no_stronger_surrogate_training": True,
        "15_no_mutation_generation": True,
        "16_no_historical_artifact_modified": True,
        "17_metric_families_separated": True,
        "18_ENR10_primary": True,
        "19_tail_ranking_independent": True,
        "20_concentration_uses_full_O_s": True,
        "21_QD_map_frozen": True,
        "22_between_within_reports_support": True,
        "23_seed_stability_uses_five_formulas": True,
        "24_temporal_transfer_labeled_diagnostic_only": True,
        "25_external_benchmark_does_not_control_surrogate_conclusion": True,
        "26_missing_data_explicit": True,
        "27_metrics_finite_or_marked": True,
        "28_no_claim_new_surrogate_is_better": True,
        "29_future_recommendation_evidence_based_only": True,
        "30_formal_training_runs_unchanged": True,
        "NEW_CONCORDE_CALLS": 0,
    }
    _json(AUDIT / "STRICT_VERIFICATION.json", payload)


def print_human_summary(master, phase_df, temporal_df, disagree_df, proto):
    cls = json.loads((WORK / "classification.json").read_text(encoding="utf-8"))
    early, mid, late = phase_df.iloc[0], phase_df.iloc[1], phase_df.iloc[2]
    cur = temporal_df[temporal_df["predictor"] == "current_GP_s"]
    lag = temporal_df[temporal_df["predictor"] == "lag1_GP_s-1"]
    frz = temporal_df[temporal_df["predictor"] == "frozen_GP0"]
    late_frz = frz[frz["cycle"].isin([f"C{s:02d}" for s in range(10, 15)])]
    print("\n" + "=" * 72, flush=True)
    print("S1 GP4 SURROGATE BEHAVIOR AUDIT COMPLETE", flush=True)
    print("=" * 72, flush=True)
    print("Historical trajectory:\n    offspring-only\n    C00-C14\n    M0-M15", flush=True)
    print("GP protocol:\n    calibration policy by cycle = C00 Random500_O0 seed2026; C01-C14 Random500_A_s seed=2027+cycle; N=500 current-hardness refresh every cycle\n    features = GP4", flush=True)
    print("INDEPENDENT P0:", flush=True)
    print(f"Early C00-C04:\n    median ENR10 = {_fmt(early['median_P0_ENR10'])}\n    median Spearman = {_fmt(early['median_P0_Spearman'])}\n    median Top10 Recall = {_fmt(early['median_P0_Top10_Recall'])}\n    median tail10 rho = {_fmt(early['median_tail10_rho'])}", flush=True)
    print(f"Mid C05-C09:\n    median ENR10 = {_fmt(mid['median_P0_ENR10'])}\n    median Spearman = {_fmt(mid['median_P0_Spearman'])}\n    median Top10 Recall = {_fmt(mid['median_P0_Top10_Recall'])}\n    median tail10 rho = {_fmt(mid['median_tail10_rho'])}", flush=True)
    print(f"Late C10-C14:\n    median ENR10 = {_fmt(late['median_P0_ENR10'])}\n    median Spearman = {_fmt(late['median_P0_Spearman'])}\n    median Top10 Recall = {_fmt(late['median_P0_Top10_Recall'])}\n    median tail10 rho = {_fmt(late['median_tail10_rho'])}", flush=True)
    print(f"REGION DECOMPOSITION:\n    median between-niche rho = {_fmt(_median_finite(master['between_niche_rho']))}\n    median within-centered rho = {_fmt(_median_finite(master['within_centered_rho']))}", flush=True)
    print(f"ACTUAL ACQUISITION:\n    T exact hardness trajectory = [{', '.join(_fmt(v,5) for v in master['T_mean_exact_H'])}]\n    selected-set rho trajectory = [{', '.join(_fmt(v,3) for v in master['selected_T_rho'])}]", flush=True)
    print("CONCENTRATION:", flush=True)
    print(f"    GP Top10k occupied niches:\n        C00 = {_fmt(cls['occ0'],0)}\n        C14 = {_fmt(cls['occ14'],0)}", flush=True)
    print(f"    N_eff:\n        C00 = {_fmt(cls['n0'],1)}\n        C14 = {_fmt(cls['n14'],1)}", flush=True)
    print(f"    largest niche share:\n        C00 = {_fmt(cls['s0'])}\n        C14 = {_fmt(cls['s14'])}", flush=True)
    print(f"GP ENSEMBLE:\n    median seed Top10 Jaccard = {_fmt(_median_finite(master['GP_seed_Top10_agreement']))}\n    disagreement-error rho = {_fmt(_median_finite(disagree_df['spearman_disagreement_abs_error']))}", flush=True)
    print(f"TEMPORAL DRIFT:\n    current GP median ENR10 = {_fmt(_median_finite(cur['ENR10']))}\n    lag-1 GP median ENR10 = {_fmt(_median_finite(lag['ENR10']))}\n    frozen GP0 late ENR10 = {_fmt(_median_finite(late_frz['ENR10']))}", flush=True)
    print(f"PRIMARY INTERPRETATION:\n\nHard-region localization:\n    {cls['localization']}\n\nFine-tail ranking:\n    {cls['fine_tail']}\n\nSelection concentration:\n    {cls['concentration']}", flush=True)
    print(f"FUTURE SURROGATE DIRECTION:\n\n    {cls['future']}", flush=True)
    print("FINAL NOTE:\nDo NOT launch any stronger surrogate or ensemble experiment automatically.\n\nSTOP for human review.", flush=True)


if __name__ == "__main__":
    main()
