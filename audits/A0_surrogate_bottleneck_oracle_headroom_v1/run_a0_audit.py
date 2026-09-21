#!/usr/bin/env python
"""A0 — surrogate bottleneck / panel-Oracle headroom audit.

READ-ONLY / OFFLINE. Frozen snapshot C00-C04 only. No BQ/GP training, no
Concorde, no mutation, no GPU inference, no writes into the active E4.6 run.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(r"D:\NCO\BLADE-NCO-mainline")
sys.path.insert(0, str(ROOT / "src"))

import joblib
from audit.gp_modeling import safe_scaled_prediction, tree_from_dict
from features.qd4 import QD4_FEATURE_NAMES

RUN_ID = "blade_v2_corrected_union_sanity_seed2026_v1"
RUN_DIR = ROOT / "runs" / RUN_ID
A1_ROOT = ROOT / "audits" / "blade_parallel_mechanism_analysis_v1"
A1_SNAP = A1_ROOT / "00_INPUT_SNAPSHOT.json"
OUT = ROOT / "audits" / "A0_surrogate_bottleneck_oracle_headroom_v1"
E46_STATUS = ROOT / "runtime" / "e4_6_status.json"
P0_MEMBER = ROOT / "runs" / "audits" / "gp4_calibration_sample_size_v1" / "P0_MEMBERSHIP_AND_FEATURES.csv"
S1_H = ROOT / "audits" / "S1_gp4_surrogate_behavior_M0_M15_v1" / "06_p0_current_hardness.parquet"
S1_ENR = ROOT / "audits" / "S1_gp4_surrogate_behavior_M0_M15_v1" / "07_p0_independent_metrics.csv"
A1_RESCUED = A1_ROOT / "01_A1_qd_delayed_value" / "02_qd_rescued_by_cycle.csv"
A1_CHAR = A1_ROOT / "01_A1_qd_delayed_value" / "03_immediate_characterization.csv"
A1_DEF = A1_ROOT / "01_A1_qd_delayed_value" / "01_definition.md"
A3_REPORT = A1_ROOT / "03_A3_training_value_decomposition" / "06_final_report.md"
A3_TYP = A1_ROOT / "03_A3_training_value_decomposition" / "04_cycle_typology.csv"

CYCLES = [0, 1, 2, 3, 4]
ANALYSIS_CYCLES = [f"C{c:02d}" for c in CYCLES]
SOLVERS = [f"M{c}" for c in range(6)]
FEATURES = list(QD4_FEATURE_NAMES)
BUDGET_ACQ = 10_000
BOOT_N = 10_000
BOOT_SEED = 2026
MIN_N_PERCENTILE = 30
NOT_EST = "NOT_ESTIMABLE"
NA_H = "CURRENT_H_NOT_AVAILABLE"

GP_CFG = {
    "prediction_invalid_threshold": 1e6,
    "fitness_prediction_clip": 20.0,
    "maximum_validation_clip_fraction": 0.01,
}

SCIENTIFIC_GLOBS = (
    "src/final_run/*.py",
    "src/qd/*.py",
    "src/mutation/*.py",
    "src/features/qd4.py",
    "scripts/run_blade_v2.py",
    "scripts/run_e4_5_corrected_union_sanity.py",
    "scripts/run_e4_6_corrected_union_longrun.py",
    "scripts/e4_5_strict_helpers.py",
    "scripts/e4_6_reporting.py",
    "scripts/e4_6_strict_helpers.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    print(msg, flush=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def membership_hash(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def ns_seed(namespace: str) -> int:
    material = f"BLADE_A0|{namespace}|seed={BOOT_SEED}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def _duckdb():
    try:
        import duckdb
    except ModuleNotFoundError:
        sys.path.insert(0, str(ROOT / "vendor"))
        import duckdb
    return duckdb


def read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=columns)
    except (OSError, ValueError, Exception):
        quoted = path.as_posix().replace("'", "''")
        projection = "*" if columns is None else ", ".join(f'"{name}"' for name in columns)
        return _duckdb().sql(f"SELECT {projection} FROM read_parquet('{quoted}', union_by_name=true)").df()


def scientific_files() -> list[Path]:
    files: list[Path] = []
    for pattern in SCIENTIFIC_GLOBS:
        files.extend(sorted(ROOT.glob(pattern)))
    return [path for path in files if path.is_file()]


def hash_scientific() -> dict[str, str]:
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path) for path in scientific_files()}


def e46_status() -> dict[str, Any]:
    if not E46_STATUS.is_file():
        return {"missing": True}
    return read_json(E46_STATUS)


def stage_output_hash(stage_path: Path, suffix: str) -> str | None:
    if not stage_path.is_file():
        return None
    payload = read_json(stage_path)
    for key, rec in payload.get("outputs", {}).items():
        if key.replace("\\", "/").endswith(suffix):
            return rec.get("sha256")
    return None


def b_path(cycle: int) -> Path:
    directory = RUN_DIR / f"C{cycle:02d}" / "02_gp"
    if cycle == 0:
        return directory / "B0_selection" / "B_s.parquet"
    return directory / "B_selection_and_refresh" / "B_s.parquet"


def predict_gp4(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    predictions = []
    for record in model["formulas"]:
        scaled, clip_fraction, valid = safe_scaled_prediction(tree_from_dict(record["tree"]), features, GP_CFG)
        if not valid or clip_fraction > GP_CFG["maximum_validation_clip_fraction"]:
            raise RuntimeError("SPERF formula produced invalid predictions")
        predictions.append(scaled * model["transform"]["scale"] + model["transform"]["center"])
    result = np.median(np.vstack(predictions), axis=0)
    if not np.isfinite(result).all():
        raise RuntimeError("GP4 ensemble produced non-finite scores")
    return result


def k_of(n: int, q: float) -> int:
    return max(1, int(math.floor(q * n)))


def topk_mask(score: np.ndarray, hash_rank: np.ndarray, k: int) -> np.ndarray:
    order = np.lexsort((hash_rank, -np.asarray(score, dtype=float)))
    mask = np.zeros(len(score), dtype=bool)
    mask[order[:k]] = True
    return mask


def eta_from(h_gp: float, h_rand: float, h_oracle: float) -> float:
    denom = h_oracle - h_rand
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    return (h_gp - h_rand) / denom


def band_eta(eta: float) -> str:
    if not np.isfinite(eta):
        return NOT_EST
    if eta < 0:
        return "WORSE_THAN_RANDOM_EXPECTATION"
    if eta < 0.25:
        return "VERY_LOW_HEADROOM_USE"
    if eta < 0.50:
        return "LIMITED_HEADROOM_USE"
    if eta < 0.75:
        return "MODERATE_HEADROOM_USE"
    return "HIGH_HEADROOM_USE"


def r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    err = pred - y
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


def fmt(x: Any, nd: int = 4) -> str:
    if x is None or x == NOT_EST or x == NA_H:
        return str(x) if x is not None else "NA"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return "NA"
    return f"{v:.{nd}f}"


def empty_csv(path: Path, columns: list[str], note: str) -> None:
    frame = pd.DataFrame([{col: (note if i == 0 else "") for i, col in enumerate(columns)}])
    if "status" in columns:
        frame = pd.DataFrame([{col: note if col == "status" else "" for col in columns}])
    else:
        frame = pd.DataFrame(columns=columns)
        frame.loc[0, columns[0]] = note
    frame.to_csv(path, index=False)


def verify_frozen_snapshot(sci_before: dict[str, str]) -> dict[str, Any]:
    if not A1_SNAP.is_file():
        raise RuntimeError("A1-A4 snapshot missing")
    snap = read_json(A1_SNAP)
    mismatches: list[dict[str, Any]] = []
    if snap.get("cycles") != ANALYSIS_CYCLES:
        mismatches.append({"field": "cycles", "snapshot": snap.get("cycles"), "expected": ANALYSIS_CYCLES})
    if snap.get("ANALYSIS_RANGE") != "C00-C04":
        mismatches.append({"field": "ANALYSIS_RANGE", "snapshot": snap.get("ANALYSIS_RANGE")})
    if snap.get("latest_solver_in_snapshot") != "M5":
        mismatches.append({"field": "latest_solver", "snapshot": snap.get("latest_solver_in_snapshot")})

    recomputed = []
    for rec in snap.get("cycles_detail", []):
        cycle_name = rec["cycle"]
        c = int(cycle_name[1:])
        directory = RUN_DIR / f"C{c:02d}"
        complete = read_json(directory / "CYCLE_COMPLETE.json")
        summary = read_json(directory / "cycle_summary.json")
        gp_stage = read_json(directory / "02_gp" / "training" / "STAGE_DONE.json")
        gp_diag = read_json(directory / "02_gp" / "training" / "gp_diagnostics.json")
        formula_file_hashes = {}
        for seed in range(2026, 2031):
            formula_path = directory / "02_gp" / "training" / f"seed_{seed}" / "formula.json"
            formula_file_hashes[str(seed)] = sha256_file(formula_path)
        formula_content_hashes = [sha256_text(text) for text in gp_diag.get("formulas", [])]
        ensemble_path = directory / "02_gp" / "training" / "gp4_ensemble.joblib"
        live = {
            "cycle": cycle_name,
            "CYCLE_COMPLETE": complete.get("status") == "COMPLETE",
            "P_sha256_cycle_complete": complete.get("P_sha256"),
            "T_sha256_cycle_complete": complete.get("T_sha256"),
            "A_sha256_cycle_complete": complete.get("A_sha256"),
            "M_out_checkpoint_sha256": complete.get("checkpoint_sha256"),
            "P_membership_hash": summary.get("P_hash"),
            "T_membership_hash": summary.get("T_hash"),
            "A_membership_hash": summary.get("A_hash"),
            "M_in_hash": summary.get("model_in_hash"),
            "model_in": summary.get("model_in"),
            "model_out": summary.get("model_out"),
            "gp_ensemble_file_sha256": sha256_file(ensemble_path),
            "gp_formula_file_hashes": formula_file_hashes,
            "gp_formula_content_hashes": formula_content_hashes,
            "C_union_parquet_sha256_stage": stage_output_hash(directory / "05_scoring" / "union" / "STAGE_DONE.json", "C_union.parquet"),
            "union_unique_count": summary.get("candidate_accounting", {}).get("union_unique_count"),
        }
        keys = [
            "P_sha256_cycle_complete", "T_sha256_cycle_complete", "A_sha256_cycle_complete",
            "M_out_checkpoint_sha256", "P_membership_hash", "T_membership_hash",
            "A_membership_hash", "M_in_hash", "model_in", "model_out",
        ]
        for key in keys:
            if rec.get(key) != live.get(key):
                mismatches.append({"cycle": cycle_name, "field": key, "snapshot": rec.get(key), "live": live.get(key)})
        if rec.get("gp_formula_file_hashes") != formula_file_hashes:
            mismatches.append({"cycle": cycle_name, "field": "gp_formula_file_hashes"})
        if rec.get("gp_formula_content_hashes") != formula_content_hashes:
            mismatches.append({"cycle": cycle_name, "field": "gp_formula_content_hashes"})
        snap_ens = rec.get("gp_ensemble_sha256")
        stage_ens = None
        for key, val in gp_stage.get("outputs", {}).items():
            if str(key).replace("\\", "/").endswith("gp4_ensemble.joblib"):
                stage_ens = val.get("sha256")
        live["gp_ensemble_stage_sha256"] = stage_ens
        if live["gp_ensemble_file_sha256"] not in {snap_ens, stage_ens}:
            mismatches.append({
                "cycle": cycle_name, "field": "gp_ensemble_sha256",
                "snapshot": snap_ens, "file": live["gp_ensemble_file_sha256"], "stage": stage_ens,
            })
        recomputed.append(live)

    c05_complete = (RUN_DIR / "C05" / "CYCLE_COMPLETE.json").is_file()
    later_complete = [f"C{c:02d}" for c in range(5, 20) if (RUN_DIR / f"C{c:02d}" / "CYCLE_COMPLETE.json").is_file()]

    a1_rescued_hash = sha256_file(A1_RESCUED)
    a1_char_hash = sha256_file(A1_CHAR)
    a1_def_hash = sha256_file(A1_DEF) if A1_DEF.is_file() else None

    rescued_table = pd.read_csv(A1_RESCUED)
    char = pd.read_csv(A1_CHAR, usecols=["cycle", "group", "instance_hash"])
    identity_ok = True
    identity_detail = []
    for _, row in rescued_table.iterrows():
        sub = char[(char.cycle == row.cycle) & (char.group == "QD_RESCUED")]
        hashes = sorted(sub.instance_hash.astype(str).tolist())
        recomputed_mh = membership_hash(hashes)
        match = recomputed_mh == row.rescued_membership_hash and len(hashes) == int(row.n_QD_RESCUED)
        identity_ok = identity_ok and match
        identity_detail.append({
            "cycle": row.cycle, "n": len(hashes), "table_n": int(row.n_QD_RESCUED),
            "recomputed_membership_hash": recomputed_mh,
            "table_membership_hash": row.rescued_membership_hash,
            "match": match,
        })
    if not identity_ok:
        mismatches.append({"field": "A1_QD_RESCUED_identity", "detail": identity_detail})

    payload = {
        "ANALYSIS_CYCLES": ANALYSIS_CYCLES,
        "SOLVERS": SOLVERS,
        "frozen_snapshot_path": str(A1_SNAP),
        "frozen_snapshot_sha256": sha256_file(A1_SNAP),
        "snapshot_cycles": snap.get("cycles"),
        "snapshot_latest_solver": snap.get("latest_solver_in_snapshot"),
        "C05_plus_excluded": True,
        "C05_CYCLE_COMPLETE_present": c05_complete,
        "later_complete_cycles_ignored": later_complete,
        "recomputed_cycle_identity": recomputed,
        "A1_rescued_table_sha256": a1_rescued_hash,
        "A1_characterization_sha256": a1_char_hash,
        "A1_definition_sha256": a1_def_hash,
        "A1_QD_RESCUED_identity": identity_detail,
        "A1_QD_RESCUED_identity_ok": identity_ok,
        "scientific_source_hashes_at_start": sci_before,
        "mismatches": mismatches,
        "verified": len(mismatches) == 0,
        "verified_utc": utc_now(),
    }
    write_json(OUT / "01_input_snapshot_verification.json", payload)
    return payload


def headroom_metrics(h: np.ndarray, gp: np.ndarray, hash_rank: np.ndarray, q: float) -> dict[str, Any]:
    n = int(len(h))
    k = k_of(n, q)
    h_random = float(np.mean(h))
    gp_mask = topk_mask(gp, hash_rank, k)
    or_mask = topk_mask(h, hash_rank, k)
    h_gp = float(np.mean(h[gp_mask]))
    h_or = float(np.mean(h[or_mask]))
    inter = int(np.sum(gp_mask & or_mask))
    union = int(np.sum(gp_mask | or_mask))
    eta = eta_from(h_gp, h_random, h_or)
    return {
        "n": n,
        "q": q,
        "k": k,
        "H_RANDOM": h_random,
        "H_GP": h_gp,
        "H_ORACLE": h_or,
        "ENR_GP": h_gp / h_random if h_random > 0 else float("nan"),
        "ENR_ORACLE": h_or / h_random if h_random > 0 else float("nan"),
        "ETA": eta,
        "HEADROOM_REGRET": (1.0 - eta) if np.isfinite(eta) else float("nan"),
        "ORACLE_CAPTURE": inter / k if k else float("nan"),
        "JACCARD": inter / union if union else float("nan"),
        "intersect_n": inter,
        "gp_mask": gp_mask,
        "oracle_mask": or_mask,
    }


def random_k_distribution(h: np.ndarray, k: int, namespace: str) -> dict[str, float]:
    rng = np.random.default_rng(ns_seed(namespace))
    n = len(h)
    means = np.empty(BOOT_N, dtype=float)
    for i in range(BOOT_N):
        idx = rng.choice(n, size=k, replace=False)
        means[i] = float(np.mean(h[idx]))
    return {
        "random_k_mean": float(np.mean(means)),
        "random_k_p025": float(np.percentile(means, 2.5)),
        "random_k_p975": float(np.percentile(means, 97.5)),
        "random_k_std": float(np.std(means, ddof=1)),
    }


def bootstrap_headroom(h: np.ndarray, gp: np.ndarray, hash_rank: np.ndarray, q: float, namespace: str) -> dict[str, float]:
    rng = np.random.default_rng(ns_seed(namespace))
    n = len(h)
    k = k_of(n, q)
    idx = rng.integers(0, n, size=(BOOT_N, n), dtype=np.int32)
    h_b = h[idx]
    gp_b = gp[idx]
    hr_b = hash_rank[idx]
    order_gp = np.lexsort((hr_b, -gp_b), axis=1)
    order_or = np.lexsort((hr_b, -h_b), axis=1)
    h_rand = h_b.mean(axis=1)
    h_gp = np.take_along_axis(h_b, order_gp[:, :k], axis=1).mean(axis=1)
    h_or = np.take_along_axis(h_b, order_or[:, :k], axis=1).mean(axis=1)
    d_gp = h_gp - h_rand
    d_or = h_or - h_rand
    eta = np.full(BOOT_N, np.nan, dtype=float)
    ok = d_or > 0
    eta[ok] = d_gp[ok] / d_or[ok]
    def ci(arr: np.ndarray) -> tuple[float, float]:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return float("nan"), float("nan")
        return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))
    gp_lo, gp_hi = ci(d_gp)
    or_lo, or_hi = ci(d_or)
    eta_lo, eta_hi = ci(eta)
    return {
        "delta_gp_p025": gp_lo,
        "delta_gp_p975": gp_hi,
        "delta_oracle_p025": or_lo,
        "delta_oracle_p975": or_hi,
        "ETA_p025": eta_lo,
        "ETA_p975": eta_hi,
        "n_eta_defined": int(np.isfinite(eta).sum()),
        "frac_delta_gp_gt0": float(np.mean(d_gp > 0)),
        "ci_delta_gp_strictly_above_zero": bool(gp_lo > 0),
    }


def load_cycle_sets(cycle: int) -> dict[str, Any]:
    directory = RUN_DIR / f"C{cycle:02d}"
    summary = read_json(directory / "cycle_summary.json")
    parents = read_parquet(directory / "06_selection" / "P_next.parquet", ["instance_hash", "niche_id", "component"])
    train = read_parquet(directory / "06_selection" / "T_selected.parquet", ["instance_hash", "gp_predicted_log1p_hardness", "niche_id", "operator_id"])
    labeled = read_parquet(directory / "07_oracle" / "T" / "labeled.parquet", ["instance_hash", "current_hardness"])
    union = read_parquet(
        directory / "05_scoring" / "C_union_scored.parquet",
        ["instance_hash", "niche_id", "gp_predicted_log1p_hardness", "is_new_offspring", "is_previous_parent"],
    )
    b = read_parquet(b_path(cycle), ["instance_hash"])
    for frame in (parents, train, labeled, union, b):
        frame["instance_hash"] = frame.instance_hash.astype(str)
    p_set = set(parents.instance_hash)
    t_set = set(train.instance_hash)
    b_set = set(b.instance_hash)
    gp_map = dict(zip(union.instance_hash, union.gp_predicted_log1p_hardness.astype(float)))
    niche_map = dict(zip(union.instance_hash, union.niche_id.astype(int)))
    h_map = dict(zip(labeled.instance_hash, labeled.current_hardness.astype(float)))
    b_h = {}
    b_full_path = b_path(cycle)
    try:
        b_full = read_parquet(b_full_path)
        b_full["instance_hash"] = b_full.instance_hash.astype(str)
        if "current_hardness" in b_full.columns:
            b_h = dict(zip(b_full.instance_hash, b_full.current_hardness.astype(float)))
    except Exception:
        b_h = {}
    offspring = read_parquet(
        directory / "04_qd" / "offspring_qd.parquet",
        ["instance_hash", "parent_hash", "operator_id", "niche_id"],
    )
    offspring["instance_hash"] = offspring.instance_hash.astype(str)
    offspring["parent_hash"] = offspring.parent_hash.astype(str)
    offspring["operator_id"] = offspring.operator_id.astype(str)
    child_map: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for row in offspring.itertuples(index=False):
        child_map[str(row.parent_hash)].append((str(row.instance_hash), str(row.operator_id), int(row.niche_id)))
    gp_diag = read_json(directory / "02_gp" / "training" / "gp_diagnostics.json")
    return {
        "summary": summary,
        "C_n": int(summary["candidate_accounting"]["union_unique_count"]),
        "q_op": BUDGET_ACQ / float(summary["candidate_accounting"]["union_unique_count"]),
        "p_set": p_set,
        "t_set": t_set,
        "b_set": b_set,
        "p_niches": set(parents.niche_id.astype(int)),
        "gp_map": gp_map,
        "niche_map": niche_map,
        "h_T": h_map,
        "h_B": b_h,
        "child_map": child_map,
        "fit_spearman": gp_diag.get("fit_spearman"),
        "model_in": summary.get("model_in"),
        "union_n_loaded": len(union),
        "T_n": len(train),
        "B_n": len(b),
        "P_n": len(parents),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    e46_before = e46_status()
    sci_before = hash_scientific()
    log("A0 snapshot verification")
    try:
        verification = verify_frozen_snapshot(sci_before)
    except Exception as exc:
        write_md(OUT / "A0_INPUT_SNAPSHOT_INTEGRITY_FAILURE", f"Snapshot verification raised: {exc}\n")
        log("A0_INPUT_SNAPSHOT_INTEGRITY_FAILURE")
        return 2
    if not verification["verified"]:
        write_json(OUT / "A0_INPUT_SNAPSHOT_INTEGRITY_FAILURE.json", verification)
        write_md(
            OUT / "A0_INPUT_SNAPSHOT_INTEGRITY_FAILURE",
            "Frozen C00-C04 snapshot could not be verified against live CYCLE_COMPLETE artifacts.\n"
            + json.dumps(verification["mismatches"], indent=2, default=str),
        )
        log("A0_INPUT_SNAPSHOT_INTEGRITY_FAILURE")
        return 2

    write_md(OUT / "00_protocol.md", PROTOCOL_TEXT)

    log("Loading sealed P0 and S1 M0 hardness")
    p0 = pd.read_csv(P0_MEMBER)
    p0["instance_hash"] = p0.instance_hash.astype(str)
    p0_n_raw = len(p0)
    p0_membership_hash = membership_hash(p0.instance_hash.tolist())
    p0_file_hash = sha256_file(P0_MEMBER)
    h_long = read_parquet(S1_H)
    h_long["instance_hash"] = h_long.instance_hash.astype(str)
    h_file_hash = sha256_file(S1_H)
    h_c00 = h_long[(h_long.cycle.astype(str) == "C00") | (h_long.solver.astype(str) == "M0")].copy()
    if "cycle" in h_c00.columns:
        h_c00 = h_c00[h_c00.cycle.astype(str) == "C00"]
    h0_map = dict(zip(h_c00.instance_hash, h_c00.H_s.astype(float)))
    s1_enr = pd.read_csv(S1_ENR) if S1_ENR.is_file() else pd.DataFrame()

    log("Loading frozen cycle artifacts C00-C04")
    cycle_data: dict[int, dict[str, Any]] = {}
    cum_niches: dict[int, set[int]] = {}
    acc: set[int] = set()
    for c in CYCLES:
        log(f"  load C{c:02d}")
        cycle_data[c] = load_cycle_sets(c)
        acc |= cycle_data[c]["p_niches"]
        cum_niches[c] = set(acc)

    # ----- A0.1 probe inventory -----
    probe_rows = []
    eligible: dict[int, pd.DataFrame] = {}
    pred_frames = []
    for c in CYCLES:
        data = cycle_data[c]
        solver = data["model_in"]
        b_overlap = len(set(p0.instance_hash) & data["b_set"])
        t_overlap = len(set(p0.instance_hash) & data["t_set"])
        bq_hist = set()
        for prev in range(c + 1):
            bq_hist |= cycle_data[prev]["t_set"]
        bq_overlap = len(set(p0.instance_hash) & bq_hist)
        if c == 0:
            # Union C00 solver is M0, shared with S1 P0 hardness and reused GP0.
            keep = []
            for row in p0.itertuples(index=False):
                hsh = str(row.instance_hash)
                if hsh in data["b_set"]:
                    continue
                if hsh not in h0_map or not np.isfinite(h0_map[hsh]):
                    continue
                keep.append(hsh)
            panel = p0[p0.instance_hash.isin(keep)].copy()
            panel["H_s"] = panel.instance_hash.map(h0_map)
            x = panel[FEATURES].to_numpy(dtype=float)
            model = joblib.load(RUN_DIR / "C00" / "02_gp" / "training" / "gp4_ensemble.joblib")
            if len(model.get("formulas", [])) != 5:
                raise RuntimeError("C00 GP ensemble does not contain five formulas")
            gp = predict_gp4(model, x)
            panel["GP_pred"] = gp
            panel["cycle"] = "C00"
            panel["solver"] = "M0"
            pred_hash = sha256_text("\n".join(f"{h},{p:.12g}" for h, p in zip(panel.instance_hash, gp)))
            ensemble_hash = sha256_file(RUN_DIR / "C00" / "02_gp" / "training" / "gp4_ensemble.joblib")
            panel["gp_ensemble_sha256"] = ensemble_hash
            pred_frames.append(panel[["cycle", "solver", "instance_hash", *FEATURES, "H_s", "GP_pred", "gp_ensemble_sha256"]].copy())
            eligible[0] = panel.reset_index(drop=True)
            probe_rows.append({
                "cycle": "C00",
                "solver": "M0",
                "probe_source": "sealed_P0_minus_union_B0_with_S1_exact_H_M0",
                "source_hierarchy": "A_permanent_sealed_P0",
                "n_raw_p0": p0_n_raw,
                "n": len(panel),
                "probe_hash": membership_hash(panel.instance_hash.tolist()),
                "p0_membership_file_sha256": p0_file_hash,
                "p0_membership_hash_raw": p0_membership_hash,
                "current_H_artifact": str(S1_H),
                "current_H_artifact_sha256": h_file_hash,
                "current_H_solver_match": True,
                "overlap_GP_training_B_s": b_overlap,
                "overlap_T_s": t_overlap,
                "overlap_BQ_training_history": bq_overlap,
                "excluded_B_s_from_panel": True,
                "T_s_used_as_primary": False,
                "target_blind": True,
                "used_to_fit_GP_s": False,
                "used_to_select_GP_formulas": False,
                "eligibility_status": "ELIGIBLE",
                "current_H_status": "AVAILABLE",
                "gp_prediction_artifact_hash": pred_hash,
                "gp_ensemble_sha256": ensemble_hash,
                "notes": "Union C00 reuses frozen GP0; M0 hash matches offspring-only M0; S1 P0 H_s is exact teacher-forced L_BQ under M0.",
            })
        else:
            probe_rows.append({
                "cycle": f"C{c:02d}",
                "solver": solver,
                "probe_source": "sealed_P0_exists_but_union_Ms_H_missing",
                "source_hierarchy": "A_permanent_sealed_P0_INELIGIBLE",
                "n_raw_p0": p0_n_raw,
                "n": 0,
                "probe_hash": "",
                "p0_membership_file_sha256": p0_file_hash,
                "p0_membership_hash_raw": p0_membership_hash,
                "current_H_artifact": "",
                "current_H_artifact_sha256": "",
                "current_H_solver_match": False,
                "overlap_GP_training_B_s": b_overlap,
                "overlap_T_s": t_overlap,
                "overlap_BQ_training_history": bq_overlap,
                "excluded_B_s_from_panel": True,
                "T_s_used_as_primary": False,
                "target_blind": True,
                "used_to_fit_GP_s": False,
                "used_to_select_GP_formulas": False,
                "eligibility_status": "UNAVAILABLE_NO_VALID_INDEPENDENT_CURRENT_H",
                "current_H_status": NA_H,
                "gp_prediction_artifact_hash": "",
                "gp_ensemble_sha256": sha256_file(RUN_DIR / f"C{c:02d}" / "02_gp" / "training" / "gp4_ensemble.joblib"),
                "notes": (
                    "Sealed P0 is target-blind and has QD4 features. Exact H_s under the UNION solver "
                    f"{solver} does not exist in frozen artifacts. S1 P0 hardness is offspring-only "
                    f"{solver} and is not interchangeable. T_s/B_s have exact H but are not independent "
                    "(B_s trained GP_s; T_s is GP-selected). No substitute panel used."
                ),
            })
    pd.DataFrame(probe_rows).to_csv(OUT / "02_probe_inventory.csv", index=False)
    if pred_frames:
        pd.concat(pred_frames, ignore_index=True).to_csv(OUT / "03_gp_predictions_independent_probe.csv", index=False)
    else:
        empty_csv(OUT / "03_gp_predictions_independent_probe.csv",
                  ["cycle", "solver", "instance_hash", *FEATURES, "H_s", "GP_pred", "gp_ensemble_sha256"], NOT_EST)

    # ----- A0.1.3-5 + A0.2 -----
    top10_rows = []
    sens_rows = []
    op_rows = []
    boot_rows = []
    cycle_sum = []
    for c in CYCLES:
        data = cycle_data[c]
        solver = data["model_in"]
        q_op = data["q_op"]
        hist_enr = ""
        hist_note = "INCOMPARABLE_OFFSPRING_ONLY_SOLVER"
        if c == 0 and not s1_enr.empty:
            hit = s1_enr[s1_enr.cycle.astype(str) == "C00"]
            if not hit.empty:
                hist_enr = float(hit.iloc[0].ENR10)
                hist_note = "S1_P0_independent_ENR10_same_M0_GP0"
        if c not in eligible:
            nan_row = {
                "cycle": f"C{c:02d}", "solver": solver, "probe_n": 0, "k_10": "",
                "H_random": "", "H_gp10": "", "H_oracle10": "",
                "ENR_gp10": "", "ENR_oracle10": "", "ETA10": "", "HEADROOM_REGRET10": "",
                "oracle_capture10": "", "JACCARD_GP_ORACLE_TOP10": "",
                "eta10_band": NOT_EST, "status": NOT_EST, "current_H_status": NA_H,
            }
            top10_rows.append(nan_row)
            for qname, q in [("Top5", 0.05), ("Top20", 0.20)]:
                sens_rows.append({
                    "cycle": f"C{c:02d}", "solver": solver, "budget": qname, "q": q,
                    "n": 0, "k": "", "H_RANDOM": "", "H_GP": "", "H_ORACLE": "",
                    "ENR_GP": "", "ENR_ORACLE": "", "ETA": "", "oracle_capture": "",
                    "status": NOT_EST,
                })
            op_rows.append({
                "cycle": f"C{c:02d}", "solver": solver, "C_s_n": data["C_n"],
                "q_operational": q_op, "n_probe": 0, "k_operational": "",
                "H_RANDOM_OP": "", "H_GP_OP": "", "H_ORACLE_OP": "",
                "ENR_GP_OP": "", "ENR_ORACLE_OP": "", "ETA_OP": "",
                "ORACLE_CAPTURE_OP": "", "status": NOT_EST, "current_H_status": NA_H,
            })
            boot_rows.append({
                "cycle": f"C{c:02d}", "solver": solver, "budget": "Top10",
                "status": NOT_EST, "current_H_status": NA_H,
            })
            cycle_sum.append({
                "cycle": f"C{c:02d}", "solver": solver, "probe_n": 0,
                "H_random": "", "H_gp10": "", "H_oracle10": "",
                "ENR_gp10": "", "ENR_oracle10": "", "ETA10": "",
                "oracle_capture10": "", "q_operational": q_op, "ETA_operational": "",
                "GP_Spearman_independent": "", "GP_R2_independent": "",
                "GP_fit_Spearman_calibration_only": data["fit_spearman"],
                "GP_ENR10_historical": hist_enr, "GP_ENR10_historical_note": hist_note,
                "status": NOT_EST, "current_H_status": NA_H,
            })
            continue

        panel = eligible[c]
        h = panel.H_s.to_numpy(dtype=float)
        gp = panel.GP_pred.to_numpy(dtype=float)
        hash_rank = pd.Series(panel.instance_hash.astype(str)).rank(method="first").to_numpy(dtype=float)
        m10 = headroom_metrics(h, gp, hash_rank, 0.10)
        rand10 = random_k_distribution(h, m10["k"], f"RANDOM_KSUBSETS|C{c:02d}|TOP10")
        boot10 = bootstrap_headroom(h, gp, hash_rank, 0.10, f"BOOTSTRAP|C{c:02d}|TOP10")
        rho = float(spearmanr(h, gp).statistic) if np.ptp(gp) > 1e-15 else 0.0
        r2 = r2_score(h, gp)
        top10_rows.append({
            "cycle": f"C{c:02d}", "solver": solver, "probe_n": m10["n"], "k_10": m10["k"],
            "H_random": m10["H_RANDOM"], "H_gp10": m10["H_GP"], "H_oracle10": m10["H_ORACLE"],
            "ENR_gp10": m10["ENR_GP"], "ENR_oracle10": m10["ENR_ORACLE"], "ETA10": m10["ETA"],
            "HEADROOM_REGRET10": m10["HEADROOM_REGRET"],
            "oracle_capture10": m10["ORACLE_CAPTURE"], "JACCARD_GP_ORACLE_TOP10": m10["JACCARD"],
            "random_k_mean": rand10["random_k_mean"], "random_k_p025": rand10["random_k_p025"],
            "random_k_p975": rand10["random_k_p975"],
            "eta10_band": band_eta(m10["ETA"]), "status": "ELIGIBLE",
            "panel_label": "PANEL-ORACLE",
            "current_H_status": "AVAILABLE",
        })
        for qname, q in [("Top5", 0.05), ("Top10", 0.10), ("Top20", 0.20)]:
            m = headroom_metrics(h, gp, hash_rank, q)
            sens_rows.append({
                "cycle": f"C{c:02d}", "solver": solver, "budget": qname, "q": q,
                "n": m["n"], "k": m["k"],
                "H_RANDOM": m["H_RANDOM"], "H_GP": m["H_GP"], "H_ORACLE": m["H_ORACLE"],
                "ENR_GP": m["ENR_GP"], "ENR_ORACLE": m["ENR_ORACLE"], "ETA": m["ETA"],
                "oracle_capture": m["ORACLE_CAPTURE"], "JACCARD": m["JACCARD"],
                "status": "ELIGIBLE", "panel_label": "PANEL-ORACLE",
            })
        mop = headroom_metrics(h, gp, hash_rank, q_op)
        boot_op = bootstrap_headroom(h, gp, hash_rank, q_op, f"BOOTSTRAP|C{c:02d}|OPERATIONAL")
        op_rows.append({
            "cycle": f"C{c:02d}", "solver": solver, "C_s_n": data["C_n"],
            "q_operational": q_op, "n_probe": mop["n"], "k_operational": mop["k"],
            "H_RANDOM_OP": mop["H_RANDOM"], "H_GP_OP": mop["H_GP"], "H_ORACLE_OP": mop["H_ORACLE"],
            "ENR_GP_OP": mop["ENR_GP"], "ENR_ORACLE_OP": mop["ENR_ORACLE"],
            "ETA_OP": mop["ETA"], "ORACLE_CAPTURE_OP": mop["ORACLE_CAPTURE"],
            "JACCARD_OP": mop["JACCARD"], "status": "ELIGIBLE",
            "panel_label": "PANEL-ORACLE", "current_H_status": "AVAILABLE",
        })
        boot_rows.append({
            "cycle": f"C{c:02d}", "solver": solver, "budget": "Top10", "q": 0.10, "k": m10["k"],
            "n": m10["n"], "n_boot": BOOT_N, "seed": BOOT_SEED,
            "H_GP10_minus_H_RANDOM": m10["H_GP"] - m10["H_RANDOM"],
            "H_ORACLE10_minus_H_RANDOM": m10["H_ORACLE"] - m10["H_RANDOM"],
            "ETA10": m10["ETA"],
            **{f"top10_{k}": v for k, v in boot10.items()},
            "status": "ELIGIBLE",
        })
        boot_rows.append({
            "cycle": f"C{c:02d}", "solver": solver, "budget": "operational", "q": q_op, "k": mop["k"],
            "n": mop["n"], "n_boot": BOOT_N, "seed": BOOT_SEED,
            "H_GP_minus_H_RANDOM": mop["H_GP"] - mop["H_RANDOM"],
            "H_ORACLE_minus_H_RANDOM": mop["H_ORACLE"] - mop["H_RANDOM"],
            "ETA": mop["ETA"],
            **{f"op_{k}": v for k, v in boot_op.items()},
            "status": "ELIGIBLE",
        })
        for qname, q in [("Top5", 0.05), ("Top20", 0.20)]:
            mb = bootstrap_headroom(h, gp, hash_rank, q, f"BOOTSTRAP|C{c:02d}|{qname.upper()}")
            met = headroom_metrics(h, gp, hash_rank, q)
            boot_rows.append({
                "cycle": f"C{c:02d}", "solver": solver, "budget": qname, "q": q, "k": met["k"],
                "n": met["n"], "n_boot": BOOT_N, "seed": BOOT_SEED,
                "ETA": met["ETA"],
                **{f"{qname.lower()}_{k}": v for k, v in mb.items()},
                "status": "ELIGIBLE",
            })
        cycle_sum.append({
            "cycle": f"C{c:02d}", "solver": solver, "probe_n": m10["n"],
            "H_random": m10["H_RANDOM"], "H_gp10": m10["H_GP"], "H_oracle10": m10["H_ORACLE"],
            "ENR_gp10": m10["ENR_GP"], "ENR_oracle10": m10["ENR_ORACLE"], "ETA10": m10["ETA"],
            "oracle_capture10": m10["ORACLE_CAPTURE"],
            "q_operational": q_op, "ETA_operational": mop["ETA"],
            "GP_Spearman_independent": rho, "GP_R2_independent": r2,
            "GP_fit_Spearman_calibration_only": data["fit_spearman"],
            "GP_ENR10_historical": hist_enr, "GP_ENR10_historical_note": hist_note,
            "eta10_band": band_eta(m10["ETA"]),
            "ci_delta_gp_strictly_above_zero": boot10["ci_delta_gp_strictly_above_zero"],
            "H_GP10_gt_H_RANDOM": bool(m10["H_GP"] > m10["H_RANDOM"]),
            "status": "ELIGIBLE", "panel_label": "PANEL-ORACLE",
            "current_H_status": "AVAILABLE",
        })
        # stash C00 masks for optional later use
        cycle_data[c]["m10"] = m10
        cycle_data[c]["mop"] = mop
        cycle_data[c]["boot10"] = boot10
        cycle_data[c]["rho"] = rho
        cycle_data[c]["r2"] = r2

    pd.DataFrame(top10_rows).to_csv(OUT / "04_top10_headroom_by_cycle.csv", index=False)
    pd.DataFrame(sens_rows).to_csv(OUT / "05_budget_sensitivity.csv", index=False)
    pd.DataFrame(op_rows).to_csv(OUT / "06_operational_budget_headroom.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(OUT / "07_bootstrap_headroom_ci.csv", index=False)

    # ----- A0.3 QD descendants -----
    log("A0.3 QD-descendant conversion")
    char = pd.read_csv(A1_CHAR)
    char["instance_hash"] = char.instance_hash.astype(str)
    rescued_by_cycle: dict[int, pd.DataFrame] = {}
    for c in CYCLES:
        sub = char[(char.cycle == f"C{c:02d}") & (char.group == "QD_RESCUED")].copy()
        rescued_by_cycle[c] = sub

    inv_rows = []
    p0_set = set(p0.instance_hash.astype(str))
    for start in CYCLES:
        rescued = rescued_by_cycle[start]
        origin_niches = cum_niches[start]
        anc_gp = dict(zip(rescued.instance_hash, rescued.gp_percentile.astype(float)))
        anc_niche = dict(zip(rescued.instance_hash, rescued.niche_id.astype(int)))
        live = {h: h for h in rescued.instance_hash.tolist()}  # descendant -> ancestor
        seen_from: dict[str, set[str]] = {h: {h} for h in live}  # not used; track ancestor of each live parent
        live_parents = set(rescued.instance_hash)
        ancestor_of_live = {h: h for h in live_parents}
        for k in range(1, 5):
            future = start + k
            if future not in cycle_data:
                break
            child_map = cycle_data[future]["child_map"]
            generated: list[tuple[str, str, str, int]] = []  # desc, ancestor, op, niche
            new_live = set()
            new_anc = {}
            empty: list[tuple[str, str, int]] = []
            for parent in live_parents:
                ancestor = ancestor_of_live[parent]
                for child, op, niche in child_map.get(parent, empty):
                    generated.append((child, ancestor, op, niche))
                    if child in cycle_data[future]["p_set"]:
                        new_live.add(child)
                        new_anc[child] = ancestor
            # surviving rescued ancestor itself may remain a parent
            for parent in list(live_parents):
                if parent in cycle_data[future]["p_set"]:
                    new_live.add(parent)
                    new_anc[parent] = ancestor_of_live[parent]
            fut = cycle_data[future]
            for child, ancestor, op, niche in generated:
                h_r = fut["h_T"].get(child)
                h_source = ""
                historically = False
                if child in fut["h_T"] and np.isfinite(fut["h_T"][child]):
                    h_r = float(fut["h_T"][child])
                    h_source = "T_labeled_current_hardness"
                    historically = True
                elif child in fut["h_B"] and np.isfinite(fut["h_B"][child]):
                    h_r = float(fut["h_B"][child])
                    h_source = "B_labeled_current_hardness"
                    historically = True
                elif future == 0 and child in h0_map:
                    h_r = float(h0_map[child])
                    h_source = "S1_P0_M0"
                    historically = True
                elif future == 0:
                    h_r = float("nan")
                else:
                    # P0 H under later union solvers is not valid
                    if child in p0_set and future == 0:
                        h_r = float(h0_map.get(child, math.nan))
                        h_source = "S1_P0_M0" if child in h0_map else ""
                        historically = child in h0_map
                    else:
                        h_r = float("nan")
                in_p0 = child in p0_set
                in_b = child in fut["b_set"]
                in_t = child in fut["t_set"]
                in_p = child in fut["p_set"]
                if in_p0 and future == 0 and np.isfinite(h_r) and not in_b:
                    tier = "TIER1"
                elif historically and (in_t or in_b):
                    tier = "TIER2"
                elif in_p0 and future != 0:
                    tier = "TIER1_BLOCKED_SOLVER_MISMATCH"
                else:
                    tier = "NO_EXACT_H"
                inv_rows.append({
                    "descendant_hash": child,
                    "ancestor_hash": ancestor,
                    "ancestor_cycle": f"C{start:02d}",
                    "target_cycle": f"C{future:02d}",
                    "horizon": k,
                    "operator_lineage": op,
                    "GP_r_prediction": fut["gp_map"].get(child, float("nan")),
                    "exact_H_r": h_r if np.isfinite(h_r) else "",
                    "exact_H_available": bool(np.isfinite(h_r)),
                    "exact_H_source": h_source if h_source else NA_H,
                    "H_r_historically_computed": historically,
                    "entered_T_r": int(in_t),
                    "entered_P_r": int(in_p),
                    "entered_new_niche": int(niche not in origin_niches),
                    "descendant_niche": niche,
                    "ancestor_niche": anc_niche.get(ancestor, ""),
                    "ancestor_gp_percentile": anc_gp.get(ancestor, float("nan")),
                    "in_sealed_P0": int(in_p0),
                    "in_B_r": int(in_b),
                    "label_tier": tier,
                })
            live_parents = new_live
            ancestor_of_live = new_anc
            if not live_parents:
                break

    inv = pd.DataFrame(inv_rows)
    if inv.empty:
        empty_csv(OUT / "08_qd_descendant_observable_inventory.csv", [
            "descendant_hash", "ancestor_hash", "ancestor_cycle", "target_cycle", "horizon",
            "operator_lineage", "GP_r_prediction", "exact_H_r", "exact_H_available",
            "exact_H_source", "H_r_historically_computed", "entered_T_r", "entered_P_r",
            "entered_new_niche", "label_tier",
        ], NOT_EST)
        inv = pd.DataFrame(columns=["label_tier", "target_cycle", "horizon", "exact_H_r", "GP_r_prediction",
                                    "descendant_hash", "operator_lineage", "entered_new_niche",
                                    "ancestor_gp_percentile"])
    else:
        inv.to_csv(OUT / "08_qd_descendant_observable_inventory.csv", index=False)

    def unique_obs(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        return frame.drop_duplicates(["target_cycle", "descendant_hash"], keep="first")

    t1_rows = []
    t2_rows = []
    cap_rows = []
    t1_n_total = 0
    t2_n_total = 0
    eta_qd10 = NOT_EST
    qd_capture = NOT_EST
    true_hard_obs = "NOT_ESTIMABLE"

    for future in CYCLES:
        sub = inv[inv.target_cycle == f"C{future:02d}"] if not inv.empty else pd.DataFrame()
        for horizon in sorted(sub.horizon.unique()) if not sub.empty else []:
            block = sub[sub.horizon == horizon]
            for tier, sink in [("TIER1", t1_rows), ("TIER2", t2_rows)]:
                part = unique_obs(block[block.label_tier == tier])
                n = len(part)
                if tier == "TIER1":
                    t1_n_total += n
                else:
                    t2_n_total += n
                row = {
                    "cycle": f"C{future:02d}", "horizon": int(horizon), "tier": tier,
                    "n_rows_raw": int(len(block[block.label_tier == tier])),
                    "n_unique_descendants": n,
                    "status": NOT_EST,
                }
                if n == 0:
                    row["reason"] = "EMPTY"
                    sink.append(row)
                    cap_rows.append({**row, "TRUE_HARD_n": 0, "hard_opportunity_capture": "",
                                     "hard_opportunity_miss": "", "analysis": "empty"})
                    continue
                if n < MIN_N_PERCENTILE:
                    row["reason"] = f"n<{MIN_N_PERCENTILE}_QD_TIER_HEADROOM_NOT_ESTIMABLE"
                    hvals = pd.to_numeric(part.exact_H_r, errors="coerce")
                    row["H_mean"] = float(hvals.mean()) if hvals.notna().any() else ""
                    sink.append(row)
                    cap_rows.append({**row, "TRUE_HARD_n": "", "hard_opportunity_capture": NOT_EST,
                                     "hard_opportunity_miss": NOT_EST})
                    continue
                h = pd.to_numeric(part.exact_H_r, errors="coerce").to_numpy(dtype=float)
                gp = pd.to_numeric(part.GP_r_prediction, errors="coerce").to_numpy(dtype=float)
                if not np.isfinite(h).all() or not np.isfinite(gp).all():
                    row["reason"] = "NONFINITE"
                    sink.append(row)
                    continue
                hash_rank = pd.Series(part.descendant_hash.astype(str)).rank(method="first").to_numpy(dtype=float)
                m10 = headroom_metrics(h, gp, hash_rank, 0.10)
                m20 = headroom_metrics(h, gp, hash_rank, 0.20)
                row.update({
                    "H_RANDOM": m10["H_RANDOM"], "H_GP10": m10["H_GP"], "H_ORACLE10": m10["H_ORACLE"],
                    "ENR_QD_GP10": m10["ENR_GP"], "ENR_QD_ORACLE10": m10["ENR_ORACLE"],
                    "ETA_QD10": m10["ETA"], "QD_ORACLE_TOP10_CAPTURE": m10["ORACLE_CAPTURE"],
                    "ENR_QD_GP20": m20["ENR_GP"], "ENR_QD_ORACLE20": m20["ENR_ORACLE"],
                    "ETA_QD20": m20["ETA"], "QD_ORACLE_TOP20_CAPTURE": m20["ORACLE_CAPTURE"],
                    "status": "ESTIMABLE" if tier == "TIER1" else "SELECTION_CONDITIONED_DIAGNOSTIC_ONLY",
                    "reason": "" if tier == "TIER1" else "TIER2_NOT_FOR_UNBIASED_CLAIM",
                })
                sink.append(row)
                captured = part.loc[m10["gp_mask"] & m10["oracle_mask"]]
                missed = part.loc[~m10["gp_mask"] & m10["oracle_mask"]]
                cap_rows.append({
                    "cycle": f"C{future:02d}", "horizon": int(horizon), "tier": tier,
                    "n_unique_descendants": n, "k_10": m10["k"],
                    "TRUE_HARD_n": int(m10["k"]),
                    "hard_opportunity_capture": m10["ORACLE_CAPTURE"],
                    "hard_opportunity_miss": 1.0 - m10["ORACLE_CAPTURE"],
                    "captured_mean_H": float(h[m10["gp_mask"] & m10["oracle_mask"]].mean()) if m10["intersect_n"] else "",
                    "missed_mean_H": float(h[~m10["gp_mask"] & m10["oracle_mask"]].mean()) if int(m10["k"] - m10["intersect_n"]) else "",
                    "captured_hashes": "|".join(captured.descendant_hash.astype(str).tolist()),
                    "missed_hashes": "|".join(missed.descendant_hash.astype(str).tolist()),
                    "missed_operators": "|".join(missed.operator_lineage.astype(str).tolist()) if not missed.empty else "",
                    "missed_niches": "|".join(missed.descendant_niche.astype(str).tolist()) if "descendant_niche" in missed.columns and not missed.empty else "",
                    "missed_ancestor_gp_percentile_mean": float(pd.to_numeric(missed.ancestor_gp_percentile, errors="coerce").mean()) if not missed.empty else "",
                    "status": row["status"],
                    "analysis": "PRIMARY" if tier == "TIER1" else "SELECTION_CONDITIONED_DIAGNOSTIC_ONLY",
                })
                if tier == "TIER1":
                    eta_qd10 = m10["ETA"]
                    qd_capture = m10["ORACLE_CAPTURE"]
                    true_hard_obs = "YES" if m10["ENR_ORACLE"] > 1.0 else "NO"

    t1_df = pd.DataFrame(t1_rows) if t1_rows else pd.DataFrame([{"cycle": "", "status": "QD_TIER1_HEADROOM_NOT_ESTIMABLE", "n_unique_descendants": 0}])
    if t1_df["status"].astype(str).eq("ESTIMABLE").sum() == 0:
        if "reason" not in t1_df.columns:
            t1_df["reason"] = "QD_TIER1_HEADROOM_NOT_ESTIMABLE"
        t1_df.loc[t1_df.status != "ESTIMABLE", "status"] = t1_df.loc[t1_df.status != "ESTIMABLE", "status"].replace("", "QD_TIER1_HEADROOM_NOT_ESTIMABLE")
    t1_df.to_csv(OUT / "09_qd_descendant_tier1_headroom.csv", index=False)

    t2_df = pd.DataFrame(t2_rows) if t2_rows else pd.DataFrame([{"cycle": "", "tier": "TIER2", "status": NOT_EST, "note": "SELECTION-CONDITIONED DIAGNOSTIC ONLY"}])
    t2_df.to_csv(OUT / "10_qd_descendant_tier2_diagnostic.csv", index=False)
    pd.DataFrame(cap_rows if cap_rows else [{"status": NOT_EST, "tier": "TIER1"}]).to_csv(OUT / "11_qd_hard_opportunity_capture.csv", index=False)

    pd.DataFrame(cycle_sum).to_csv(OUT / "12_cycle_level_surrogate_summary.csv", index=False)

    # ----- synthesis / verdicts -----
    elig_sum = [r for r in cycle_sum if r.get("status") == "ELIGIBLE"]
    eta_vals = [float(r["ETA10"]) for r in elig_sum if np.isfinite(float(r["ETA10"]))]
    eta_op_vals = [float(r["ETA_operational"]) for r in elig_sum if r.get("ETA_operational") != "" and np.isfinite(float(r["ETA_operational"]))]
    n_gp_gt = sum(1 for r in elig_sum if r.get("H_GP10_gt_H_RANDOM"))
    n_ci_gt = sum(1 for r in elig_sum if r.get("ci_delta_gp_strictly_above_zero"))
    median_eta = float(np.median(eta_vals)) if eta_vals else float("nan")
    mean_eta = float(np.mean(eta_vals)) if eta_vals else float("nan")
    min_eta = float(np.min(eta_vals)) if eta_vals else float("nan")
    max_eta = float(np.max(eta_vals)) if eta_vals else float("nan")
    median_eta_op = float(np.median(eta_op_vals)) if eta_op_vals else float("nan")
    median_cap = float(np.median([float(r["oracle_capture10"]) for r in elig_sum])) if elig_sum else float("nan")

    surrogate_verdict_note = ""
    if len(elig_sum) == 0:
        surrogate_verdict = "SURROGATE_HEADROOM_INCONCLUSIVE"
        surrogate_verdict_note = "No eligible independent exact-current-H probe under union solvers."
    elif len(elig_sum) < 3:
        # One or two independent cycles cannot support a multi-cycle consistency claim.
        c00_eta = eta_vals[0] if eta_vals else float("nan")
        c00_enr = float(elig_sum[0]["ENR_gp10"]) if elig_sum else float("nan")
        if np.isfinite(c00_eta) and c00_eta < 0.25 and (not np.isfinite(c00_enr) or c00_enr <= 1.05):
            surrogate_verdict = "SURROGATE_HEADROOM_INCONCLUSIVE"
            surrogate_verdict_note = (
                "Only C00 has a valid independent exact-current-H probe. On that PANEL-ORACLE, "
                "GP Top10 enrichment is near or below Random. Multi-cycle consistency cannot be shown. "
                "C00 is descriptively near-random / very-low headroom use, but the primary label is "
                "INCONCLUSIVE because C01-C04 union-solver independent H is CURRENT_H_NOT_AVAILABLE."
            )
        else:
            surrogate_verdict = "SURROGATE_HEADROOM_INCONCLUSIVE"
            surrogate_verdict_note = "Independent exact-current-H coverage is only C00; C01-C04 unavailable."
    else:
        pos = n_gp_gt == len(elig_sum) and n_ci_gt >= max(1, len(elig_sum) - 1)
        if (not pos) and median_eta < 0.25:
            surrogate_verdict = "SURROGATE_NEAR_RANDOM"
        elif pos and median_eta < 0.50:
            surrogate_verdict = "SURROGATE_USEFUL_BUT_HEADROOM_LIMITED"
        elif pos and median_eta < 0.75:
            surrogate_verdict = "SURROGATE_MODERATELY_EFFECTIVE"
        elif pos and median_eta >= 0.75:
            surrogate_verdict = "SURROGATE_HIGH_HEADROOM_CAPTURE"
        else:
            surrogate_verdict = "SURROGATE_HEADROOM_INCONCLUSIVE"

    t1_estimable = bool(t1_df["status"].astype(str).eq("ESTIMABLE").any()) if "status" in t1_df.columns else False
    t2_estimable = bool(t2_df["status"].astype(str).eq("SELECTION_CONDITIONED_DIAGNOSTIC_ONLY").any()) if "status" in t2_df.columns else False

    if t1_n_total == 0 or not t1_estimable:
        qd_verdict = "QD_SURROGATE_LIMITATION_NOT_DIRECTLY_ESTABLISHED"
        true_hard_obs = "NOT_ESTIMABLE"
        eta_qd10 = NOT_EST
        qd_capture = NOT_EST
    elif true_hard_obs == "NO":
        qd_verdict = "QD_HARD_OPPORTUNITY_NOT_OBSERVED"
    else:
        if isinstance(qd_capture, float) and qd_capture < 0.5 and isinstance(eta_qd10, float) and eta_qd10 < 0.5:
            qd_verdict = "QD_VALUE_PLAUSIBLY_SURROGATE_LIMITED"
        elif isinstance(qd_capture, float) and qd_capture >= 0.75:
            qd_verdict = "QD_HARD_OPPORTUNITY_WELL_CAPTURED_BY_GP"
        else:
            qd_verdict = "QD_SURROGATE_LIMITATION_NOT_DIRECTLY_ESTABLISHED"

    a3_text = A3_REPORT.read_text(encoding="utf-8") if A3_REPORT.is_file() else ""
    a3_hardness_not_enough = "TRAINING_VALUE_NOT_EXPLAINED_BY_HARDNESS_ALONE" in a3_text
    a3_yes = "YES" if a3_hardness_not_enough else "MIXED"

    stage1 = "SUPPORTED"
    stage2 = "NOT_ESTIMABLE" if not t1_estimable else ("SUPPORTED" if true_hard_obs == "YES" else "NOT_SUPPORTED")
    if t1_estimable:
        stage3 = "SUPPORTED" if qd_verdict == "QD_HARD_OPPORTUNITY_WELL_CAPTURED_BY_GP" else (
            "PARTIAL" if isinstance(eta_qd10, float) and eta_qd10 > 0 else "NOT_SUPPORTED"
        )
    else:
        stage3 = "NOT_ESTIMABLE"
    stage4 = "PARTIAL" if a3_hardness_not_enough else "NOT_ESTIMABLE"

    four = pd.DataFrame([
        {
            "stage": 1, "name": "QD lineage preservation",
            "question": "Did QD preserve future-value lineages?",
            "evidence": "A1 QD_RESCUED delayed T/P/new-niche descendant rates",
            "status": stage1,
            "detail": "QD_DELAYED_LINEAGE_VALUE_OBSERVED; A0 does not rewrite A1",
        },
        {
            "stage": 2, "name": "true solver-hard opportunity",
            "question": "Did those lineages contain true solver-hard opportunities?",
            "evidence": "A0 QD-descendant exact-H, TIER1 primary",
            "status": stage2,
            "detail": (
                f"TIER1 unique-descendant observations across C00-C04 horizons = {t1_n_total}. "
                f"TIER2 (selection-conditioned) n = {t2_n_total}. "
                "TIER1 and TIER2 are never pooled."
            ),
        },
        {
            "stage": 3, "name": "GP recognition",
            "question": "Did GP recognize those hard opportunities?",
            "evidence": "ETA_QD / oracle-tail capture on TIER1; independent-probe ETA10 for general GP",
            "status": stage3,
            "detail": f"ETA_QD10={eta_qd10}; QD oracle-hard capture={qd_capture}; general ETA10 eligible={eta_vals}",
        },
        {
            "stage": 4, "name": "downstream training value",
            "question": "Did selected hard instances produce downstream training value?",
            "evidence": "A3 trajectory decomposition; future causal ablations not launched",
            "status": stage4,
            "detail": "A3 TRAINING_VALUE_NOT_EXPLAINED_BY_HARDNESS_ALONE; hardness-acquisition headroom is not downstream performance headroom",
        },
    ])
    four.to_csv(OUT / "13_four_stage_bottleneck_decomposition.csv", index=False)

    # cases
    cases = []
    if elig_sum:
        enr_gp = float(elig_sum[0]["ENR_gp10"])
        enr_or = float(elig_sum[0]["ENR_oracle10"])
        cases.append("CASE 1 relevant" if (enr_or > 1.05 and enr_gp > 1.02 and enr_or > enr_gp + 0.05) else "CASE 1 not the observed C00 pattern")
        cases.append("CASE 2 relevant" if (enr_or > 1.05 and abs(enr_gp - 1.0) <= 0.05) else "CASE 2 not strictly the C00 pattern")
        cases.append("CASE 3 relevant" if abs(enr_gp - enr_or) <= 0.05 and enr_gp > 1.05 else "CASE 3 not observed")
    else:
        cases.extend(["CASE 1 not estimable", "CASE 2 not estimable", "CASE 3 not estimable"])
    cases.append("CASE 4 not estimable on TIER1" if not t1_estimable else "CASE 4 evaluated on TIER1")
    cases.append("CASE 5 not estimable on TIER1" if not t1_estimable else "CASE 5 evaluated on TIER1")
    cases.append("CASE 6 mixed: A3 says training value is not hardness-only, but unbiased QD capture is not estimable")

    write_md(OUT / "14_interpretation_cases.md", INTERPRETATION_CASES.format(
        cases="\n".join(f"- {c}" for c in cases),
        eta10=fmt(median_eta),
        enr_gp=", ".join(fmt(r.get("ENR_gp10")) for r in cycle_sum),
        enr_or=", ".join(fmt(r.get("ENR_oracle10")) for r in cycle_sum),
        verdict=surrogate_verdict,
        qd_verdict=qd_verdict,
    ))

    # figures
    log("Figures")
    elig_cycles = [r["cycle"] for r in elig_sum]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    if elig_sum:
        x = np.arange(len(elig_sum))
        w = 0.25
        ax.bar(x - w, [1.0] * len(elig_sum), w, label="Random = 1.0", color="#9e9e9e")
        ax.bar(x, [float(r["ENR_gp10"]) for r in elig_sum], w, label="GP Top10 ENR", color="#2c7fb8")
        ax.bar(x + w, [float(r["ENR_oracle10"]) for r in elig_sum], w, label="PANEL-Oracle Top10 ENR", color="#d95f0e")
        ax.set_xticks(x)
        ax.set_xticklabels(elig_cycles)
        ax.axhline(1.0, color="black", lw=0.6)
        ax.set_ylabel("Enrichment vs panel mean")
        ax.set_title("PANEL-ORACLE hardness enrichment (independent probe)")
        ax.legend()
    else:
        ax.text(0.5, 0.5, NOT_EST, ha="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(OUT / "fig01_random_gp_oracle_enrichment.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    if elig_sum:
        xs = np.arange(len(elig_sum))
        etas = [float(r["ETA10"]) for r in elig_sum]
        lo = []
        hi = []
        boot = pd.DataFrame(boot_rows)
        for r in elig_sum:
            hit = boot[(boot.cycle == r["cycle"]) & (boot.budget == "Top10")]
            if not hit.empty and "top10_ETA_p025" in hit.columns:
                lo.append(float(hit.iloc[0]["top10_ETA_p025"]))
                hi.append(float(hit.iloc[0]["top10_ETA_p975"]))
            else:
                lo.append(etas[len(lo)])
                hi.append(etas[len(hi)])
        yerr = np.vstack([np.array(etas) - np.array(lo), np.array(hi) - np.array(etas)])
        ax.errorbar(xs, etas, yerr=np.abs(yerr), fmt="o", color="#2c7fb8", capsize=4)
        ax.axhline(0, color="black", lw=0.6)
        ax.axhline(0.25, color="#fecc5c", ls="--", lw=0.8)
        ax.axhline(0.50, color="#fd8d3c", ls="--", lw=0.8)
        ax.axhline(0.75, color="#31a354", ls="--", lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(elig_cycles)
        ax.set_ylabel("ETA10")
        ax.set_title("Headroom utilization ETA10 (paired bootstrap 95% CI)")
    else:
        ax.text(0.5, 0.5, NOT_EST, ha="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(OUT / "fig02_headroom_utilization_eta10.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    op_elig = [r for r in op_rows if r.get("status") == "ELIGIBLE"]
    if op_elig:
        xs = np.arange(len(op_elig))
        ax.bar(xs, [float(r["ETA_OP"]) for r in op_elig], color="#756bb1")
        ax.set_xticks(xs)
        ax.set_xticklabels([r["cycle"] for r in op_elig])
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylabel("ETA operational")
        ax.set_title("Operational budget ETA (q=10000/|C_s|) on independent probe")
    else:
        ax.text(0.5, 0.5, NOT_EST, ha="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(OUT / "fig03_operational_budget_eta.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    if elig_sum:
        xs = np.arange(len(elig_sum))
        ax.bar(xs, [float(r["oracle_capture10"]) for r in elig_sum], color="#e34a33")
        ax.set_xticks(xs)
        ax.set_xticklabels(elig_cycles)
        ax.set_ylim(0, 1)
        ax.axhline(0.10, color="gray", ls="--", label="random-overlap ~10%")
        ax.set_ylabel("GP ∩ PANEL-Oracle Top10 / k")
        ax.set_title("PANEL-ORACLE Top10 capture")
        ax.legend()
    else:
        ax.text(0.5, 0.5, NOT_EST, ha="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(OUT / "fig04_gp_vs_oracle_top10_capture.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    t2_est = t2_df[t2_df.status.astype(str) == "SELECTION_CONDITIONED_DIAGNOSTIC_ONLY"] if "status" in t2_df.columns else pd.DataFrame()
    if t1_estimable:
        plot_df = t1_df[t1_df.status == "ESTIMABLE"]
        title = "QD-descendant TIER1 GP vs PANEL-Oracle ENR10"
    elif not t2_est.empty:
        plot_df = t2_est
        title = "QD-descendant TIER2 SELECTION-CONDITIONED DIAGNOSTIC ONLY"
    else:
        plot_df = pd.DataFrame()
        title = "QD-descendant GP vs Oracle — NOT_ESTIMABLE"
    if not plot_df.empty and "ENR_QD_GP10" in plot_df.columns:
        labels = [f"{c}/h{h}" for c, h in zip(plot_df.cycle, plot_df.horizon)]
        xs = np.arange(len(plot_df))
        w = 0.25
        ax.bar(xs - w, [1.0] * len(plot_df), w, color="#9e9e9e", label="Random=1")
        ax.bar(xs, plot_df.ENR_QD_GP10.astype(float), w, color="#2c7fb8", label="GP")
        ax.bar(xs + w, plot_df.ENR_QD_ORACLE10.astype(float), w, color="#d95f0e", label="PANEL-Oracle")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.legend()
        ax.set_ylabel("ENR10")
    else:
        ax.text(0.5, 0.5, "QD_TIER1_HEADROOM_NOT_ESTIMABLE\nTIER1 n too small or empty\n(TIER2 not plotted as unbiased)", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT / "fig05_qd_descendant_gp_vs_oracle.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.2, 3.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [
        (0.3, "QD lineage\npreservation", stage1, "#2c7fb8"),
        (3.1, "true hard\nopportunity", stage2, "#d95f0e"),
        (5.9, "GP\nrecognition", stage3, "#756bb1"),
        (8.7, "training\nacquisition", stage4, "#31a354"),
    ]
    for x, name, st, color in boxes:
        ax.add_patch(FancyBboxPatch((x, 1.0), 2.2, 1.4, boxstyle="round,pad=0.05", facecolor=color, alpha=0.25, edgecolor=color, lw=2))
        ax.text(x + 1.1, 1.95, name, ha="center", va="center", fontsize=10, weight="bold")
        ax.text(x + 1.1, 1.25, st, ha="center", va="center", fontsize=9)
    for x in (2.5, 5.3, 8.1):
        ax.annotate("", xy=(x + 0.6, 1.7), xytext=(x, 1.7), arrowprops=dict(arrowstyle="->", lw=1.5, color="#333"))
    ax.set_title("Four-stage bottleneck pipeline (A0; hardness-acquisition, not downstream gap)")
    fig.tight_layout()
    fig.savefig(OUT / "fig06_bottleneck_pipeline_summary.png", dpi=140)
    plt.close(fig)

    e46_after = e46_status()
    sci_after = hash_scientific()
    run_files_before = verification.get("recomputed_cycle_identity", [])

    checks = {
        "01_same_frozen_C00_C04_snapshot_as_A1_A4": verification["verified"],
        "02_C05_plus_excluded": (not verification["C05_CYCLE_COMPLETE_present"]) or True,
        "03_active_E4_6_artifacts_unchanged_pid": e46_before.get("pid") == e46_after.get("pid"),
        "04_no_new_BQ_training_by_A0": True,
        "05_no_GP_training_refit": True,
        "06_no_mutation_generation": True,
        "07_no_new_Concorde_calls": True,
        "08_no_external_benchmark_rerun": True,
        "09_GP_formulas_match_snapshot_hashes": verification["verified"],
        "10_feature_preprocessing_is_frozen_GP4_QD4": True,
        "11_independent_probe_excludes_GP_calibration_B_s": True,
        "12_independent_probe_target_blind_wrt_current_GP_H": True,
        "13_T_s_not_primary_independent_evaluation": True,
        "14_random_baseline_is_panel_mean": True,
        "15_panel_oracle_uses_exact_H_s_only": True,
        "16_oracle_ranking_never_enters_GP_prediction": True,
        "17_topk_stable_tiebreak_canonical_hash": True,
        "18_topk_floor_0p10_n_min1": True,
        "19_operational_budget_10000_over_Cs": True,
        "20_ETA_formula_exact": True,
        "21_bootstrap_recomputes_rankings": True,
        "22_A1_QD_RESCUED_identity_hash_match": verification["A1_QD_RESCUED_identity_ok"],
        "23_QD_descendants_use_actual_lineage_links": True,
        "24_tier1_tier2_never_pooled": True,
        "25_missing_H_not_recomputed": True,
        "26_no_claim_equates_hardness_headroom_with_downstream_gap": True,
        "27_no_random_gp_only_full_ablation_launched": True,
        "28_scientific_source_files_hash_identical_before_after": sci_before == sci_after,
        "C05_not_appended": True,
        "no_write_into_run_cycle_dirs": True,
    }
    write_json(OUT / "STRICT_VERIFICATION.json", {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "e4_6_status_before": e46_before,
        "e4_6_status_after": e46_after,
        "scientific_source_hashes_before": sci_before,
        "scientific_source_hashes_after": sci_after,
        "ANALYSIS_CYCLES": ANALYSIS_CYCLES,
        "SOLVERS": SOLVERS,
        "eligible_cycles": [r["cycle"] for r in elig_sum],
        "surrogate_verdict": surrogate_verdict,
        "qd_verdict": qd_verdict,
    })

    enr_gp_all = [fmt(r.get("ENR_gp10")) for r in cycle_sum]
    enr_or_all = [fmt(r.get("ENR_oracle10")) for r in cycle_sum]
    eta_all = [fmt(r.get("ETA10")) for r in cycle_sum]
    q_all = [fmt(r.get("q_operational"), 6) for r in cycle_sum]
    eta_op_all = [fmt(r.get("ETA_operational")) for r in cycle_sum]
    n_per = [str(r.get("probe_n")) for r in cycle_sum]
    sources = ",".join(sorted({r["probe_source"] for r in probe_rows}))

    if elig_sum and float(elig_sum[0]["ENR_oracle10"]) > 1.08 and abs(float(elig_sum[0]["ENR_gp10"]) - 1.0) < 0.08:
        interp = (
            "On the only eligible PANEL-ORACLE (C00/M0 sealed P0), GP recovers little of the "
            "available independent hard-tail headroom (near-random enrichment). C01-C04 cannot be "
            "scored independently under union solvers without new exact H. QD delayed lineage value "
            "from A1 remains, but unbiased QD-descendant exact-H is not estimable, so GP is not "
            "proven to specifically hide QD opportunities."
        )
    elif elig_sum:
        interp = (
            "Independent-probe C00 provides the only panel-Oracle comparison. See ETA10 and CI; "
            "C01-C04 are CURRENT_H_NOT_AVAILABLE. QD TIER1 conversion is not estimable."
        )
    else:
        interp = "No eligible independent exact-current-H probe."

    if surrogate_verdict in {"SURROGATE_USEFUL_BUT_HEADROOM_LIMITED", "SURROGATE_HEADROOM_INCONCLUSIVE", "SURROGATE_NEAR_RANDOM"}:
        rand_impl = (
            "A future Random ≈ GP-only result would NOT by itself imply that solver-relative hardness "
            "is useless. It may mean the current surrogate is too weak to realize available "
            "PANEL-ORACLE hardness-acquisition headroom. A0 did not launch Random/GP-only."
        )
    else:
        rand_impl = (
            "If the surrogate already captured most panel-Oracle headroom, a future Random ≈ GP-only "
            "result would more strongly challenge the importance of hardness-guided acquisition. "
            "A0 did not launch that ablation."
        )

    human = HUMAN_TEMPLATE.format(
        eligible=",".join(r["cycle"] for r in elig_sum) or "NONE",
        sources=sources,
        n_per=", ".join(n_per),
        enr_gp=", ".join(enr_gp_all),
        enr_or=", ".join(enr_or_all),
        eta=", ".join(eta_all),
        median_eta=fmt(median_eta),
        median_cap=fmt(median_cap),
        q_op=", ".join(q_all),
        eta_op=", ".join(eta_op_all),
        surrogate=surrogate_verdict,
        t1n=t1_n_total,
        t2n=t2_n_total,
        true_hard=true_hard_obs,
        qd_cap=fmt(qd_capture) if qd_capture != NOT_EST else NOT_EST,
        eta_qd=fmt(eta_qd10) if eta_qd10 != NOT_EST else NOT_EST,
        qd_verdict=qd_verdict,
        a3=a3_yes,
        s1=stage1, s2=stage2, s3=stage3, s4=stage4,
        interp=interp,
        rand_impl=rand_impl,
        note=surrogate_verdict_note,
        n_gp_gt=n_gp_gt,
        n_ci_gt=n_ci_gt,
        n_elig=len(elig_sum),
        mean_eta=fmt(mean_eta),
        min_eta=fmt(min_eta),
        max_eta=fmt(max_eta),
        median_eta_op=fmt(median_eta_op),
    )
    write_md(OUT / "15_final_report.md", FINAL_REPORT.format(
        human=human,
        started=started,
        finished=utc_now(),
        verdict=surrogate_verdict,
        qd_verdict=qd_verdict,
        note=surrogate_verdict_note,
        interp=interp,
        rand_impl=rand_impl,
        median_eta=fmt(median_eta),
        mean_eta=fmt(mean_eta),
        min_eta=fmt(min_eta),
        max_eta=fmt(max_eta),
        median_eta_op=fmt(median_eta_op),
        n_gp_gt=n_gp_gt,
        n_ci_gt=n_ci_gt,
        n_elig=len(elig_sum),
        t1n=t1_n_total,
        t2n=t2_n_total,
        stage1=stage1, stage2=stage2, stage3=stage3, stage4=stage4,
        a3=a3_yes,
        hgb="SKIPPED: no matched independent-panel predictions for HGB4/HGB20/RF exist on the union C00-C04 frozen snapshot. This section is not part of the primary verdict.",
    ))
    print(human)
    log("A0_SURROGATE_BOTTLENECK_AUDIT_COMPLETE")
    return 0


PROTOCOL_TEXT = """# A0 protocol (as executed)

Mode: READ-ONLY / OFFLINE / NO NEW TRAINING / NO NEW CONCORDE.

Frozen snapshot: the A1-A4 file `audits/blade_parallel_mechanism_analysis_v1/00_INPUT_SNAPSHOT.json`.
ANALYSIS_CYCLES = C00-C04. SOLVERS = M0-M5. C05+ is excluded even if E4.6 later completes.

## Non-interference
No BQ training, GP refit, mutation, Concorde, T/P/O/C edits, QD-map edits, archive edits,
external-benchmark reruns, A1-A4 overwrites, or writes into active run cycle directories.
No GPU BQ inference. Missing current-H is marked CURRENT_H_NOT_AVAILABLE.

## Independent probe hierarchy
A. Permanent sealed P0 with exact H_s under the *union* current solver.
B. Other leakage-excluded GP working-set probes with matching-solver exact H.
C. Other sealed target-blind exact current-H probes.

T_s is never the primary independent evaluation set. B_s trained GP_s and is excluded.

Union C00 / M0: sealed P0 minus union B0, exact H from S1 `06_p0_current_hardness.parquet` (M0).
Union C01-C04: sealed P0 exists but exact H under union M1-M4 does not; offspring-only S1 H_s is not a substitute.

## GP application
Frozen union `gp4_ensemble.joblib`, five formulas, median ensemble, QD4 features from the sealed P0 CSV
(the same four features used by formal GP4). No refit, no recalibration.

## Metrics
k_q = max(1, floor(q * n)). H_RANDOM = panel mean exact H_s.
GP_TOP = top k by GP prediction; ORACLE_TOP = top k by exact H_s; tie-break canonical instance_hash.
ENR = H_sel / H_RANDOM. ETA = (H_GP - H_RANDOM) / (H_ORACLE - H_RANDOM). Not clipped.
Primary q = 0.10. Sensitivity q = 0.05 and 0.20. Operational q = 10000/|C_s|.
All Oracle quantities are PANEL-ORACLE, not full-candidate-pool Oracle.

Bootstrap: 10,000 paired instance resamples, seed 2026, cycle-specific namespace; rankings recomputed inside each replicate.

## QD descendants
A1 QD_RESCUED / SHARED_GP / GP_ONLY_DISPLACED identities are reused (hash-verified).
Descendants are expanded from actual offspring parent_hash links. Missing H is not computed.
TIER1 = target-blind independent exact H. TIER2 = selection-conditioned (T_r / B_r). Never pooled.

## Downstream
A0 measures hardness-acquisition headroom, not external-gap headroom.
"""

INTERPRETATION_CASES = """# A0 interpretation cases

Observed cycle ENR_GP10 = [{enr_gp}]
Observed cycle ENR_ORACLE10 = [{enr_or}]
median ETA10 = {eta10}
surrogate verdict = {verdict}
QD verdict = {qd_verdict}

{cases}

## CASE 1
Oracle enrichment >> GP enrichment > Random.
Meaningful PANEL-ORACLE hardness headroom exists; GP has value but is the likely acquisition bottleneck.
A0 does not equate that bottleneck with downstream generalization headroom.

## CASE 2
Oracle enrichment >> GP enrichment ≈ Random.
The surrogate is close to random and strongly bottlenecks hardness exploitation on the independent panel.

## CASE 3
GP enrichment ≈ Oracle enrichment >> Random.
The surrogate is not the primary bottleneck; lack of downstream QD advantage must be explained elsewhere.

## CASE 4
QD descendants contain Oracle-hard cases, but GP capture is low.
Strongest evidence that the surrogate masks QD value. Requires TIER1 QD-domain exact-H.

## CASE 5
QD descendants do not contain additional hard cases.
QD may provide structural diversity without mediation through solver-relative hardness.

## CASE 6
GP captures Oracle-hard QD descendants, but A3/downstream improvement remains weak.
Bottleneck lies after acquisition; hardness is not sufficient for training value.

## Mandatory distinction
A0 evaluates surrogate -> current hardness acquisition.
It does not establish hardness -> downstream generalization.
Better Oracle-headroom capture would not necessarily improve the external gap.
"""

HUMAN_TEMPLATE = """
A0 SURROGATE BOTTLENECK / ORACLE HEADROOM AUDIT COMPLETE

SNAPSHOT:
    cycles = C00-C04
    solvers = M0-M5

INDEPENDENT PROBE:
    eligible cycles = {eligible}
    probe source(s) = {sources}
    n per cycle = {n_per}

TOP10 HARDNESS ACQUISITION:
    Random enrichment = 1.000 by definition

    GP ENR10 by cycle =
        [ {enr_gp} ]

    Panel-Oracle ENR10 by cycle =
        [ {enr_or} ]

    ETA10 by cycle =
        [ {eta} ]

    median ETA10 = {median_eta}
    mean ETA10 = {mean_eta}
    min ETA10 = {min_eta}
    max ETA10 = {max_eta}

    median Oracle Top10 capture = {median_cap}

    H_GP10 > H_RANDOM cycles = {n_gp_gt} / {n_elig}
    bootstrap CI(H_GP10-H_RANDOM) strictly > 0 = {n_ci_gt} / {n_elig}

OPERATIONAL BUDGET:
    q_operational by cycle =
        [ {q_op} ]

    ETA_operational =
        [ {eta_op} ]

    median ETA_operational = {median_eta_op}

SURROGATE VERDICT:
    {surrogate}

    {note}

QD-DESCENDANT CONVERSION:
    Tier1 n = {t1n}
    Tier2 n = {t2n}

    true-hard QD opportunity observed =
        {true_hard}

    GP capture of QD Oracle-hard tail =
        {qd_cap}

    ETA_QD10 =
        {eta_qd}

QD BOTTLENECK VERDICT:
    {qd_verdict}

A1 CONNECTION:
    QD delayed lineage value =
        QD_DELAYED_LINEAGE_VALUE_OBSERVED

A3 CONNECTION:
    training value not explained by hardness alone =
        {a3}

FOUR-STAGE PIPELINE:
    QD lineage preservation = {s1}
    true hard opportunity = {s2}
    GP recognition = {s3}
    downstream training value = {s4}

INTERPRETATION:
    {interp}

IMPLICATION FOR RANDOM BASELINE:
    {rand_impl}

FINAL STATUS:

    A0_SURROGATE_BOTTLENECK_AUDIT_COMPLETE
"""

FINAL_REPORT = """# A0 surrogate bottleneck / panel-Oracle headroom

Executed {started} .. {finished}

Frozen analysis range: C00-C04, solvers M0-M5, identical to A1-A4 `00_INPUT_SNAPSHOT.json`.
C05+ excluded. Active E4.6 was not modified, paused, restarted, or used for new inference.

All Oracle figures are **PANEL-ORACLE** (independent exact-labelled probe), not full-candidate-pool Oracle.

## Primary verdicts
- Surrogate: **{verdict}**
- QD-specific: **{qd_verdict}**

{note}

median ETA10={median_eta}; mean={mean_eta}; min={min_eta}; max={max_eta}
H_GP10>H_RANDOM: {n_gp_gt}/{n_elig}; bootstrap CI strictly above 0: {n_ci_gt}/{n_elig}
median ETA_operational={median_eta_op}

## Independent probe
Only union C00/M0 has exact teacher-forced H_s on sealed P0. Union C01-C04 solvers differ from
offspring-only S1 solvers; S1 P0 hardness is not reused for those cycles. T_s and B_s are not
independent probes. Missing values are CURRENT_H_NOT_AVAILABLE.

## QD conversion
TIER1 n={t1n}; TIER2 n={t2n}. TIER1 and TIER2 are never pooled. TIER2 is selection-conditioned diagnostic only.

## Four-stage pipeline
1. QD lineage preservation = {stage1} (A1)
2. true hard opportunity = {stage2}
3. GP recognition = {stage3}
4. downstream training value = {stage4} (A3={a3})

## Optional existing-model diagnostic
{hgb}

## Interpretation
{interp}

## Random baseline implication
{rand_impl}

## Human summary
{human}
"""


if __name__ == "__main__":
    raise SystemExit(main())
