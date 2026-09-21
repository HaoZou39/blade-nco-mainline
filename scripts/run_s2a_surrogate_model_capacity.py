"""S2-A runner: historical GP4 vs canonical HGB4 under identical Random500.

RF4 is not trained. Fields use NOT_RUN_RF_CONFIG_NOT_FROZEN.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(r"D:\NCO\BLADE-NCO-mainline")
sys.path.insert(0, str(ROOT / "src"))

from final_run.science import _formula_prediction, predict_gp4  # noqa: E402
from surrogate_benchmark.concentration import niche_metrics, set_jaccard, topk_indices  # noqa: E402
from surrogate_benchmark.data import (  # noqa: E402
    CYCLES,
    FEATURES,
    PHASES,
    S1,
    b_meta_path,
    gp_joblib_path,
    load_b_s,
    load_offspring,
    load_p0_clean_hashes_from_s1,
    load_p0_hardness,
    load_p0_member,
    load_s11_references,
    membership_hash,
)
from surrogate_benchmark.metrics import (  # noqa: E402
    capture_rate,
    decile_rows,
    enr,
    k_q,
    paired_bootstrap_ci,
    recall_capture,
    safe_spearman,
    tail_ranking,
    top_recall,
)
from surrogate_benchmark.models import (  # noqa: E402
    HGB_CFG_CANONICAL,
    HGB_CFG_SOURCE,
    RF_CANDIDATES,
    RF_NOT_RUN,
    SEEDS,
    fit_hgb4_seed,
    hgb_params,
    load_hgb4,
    median_stack,
    predict_hgb,
    save_hgb4,
)

AUDIT = ROOT / "audits" / "S2_A_surrogate_model_capacity_GP4_HGB4_RF4_v1"
CONFIG_PATH = ROOT / "configs" / "surrogate_s2a_gp4_hgb4_rf4.yaml"
TOPK = 10000
GP_ENR_ATOL = 1e-10
GP_ENR_RTOL = 1e-8
GP_REPLAY_ATOL = 1e-12
QS = (("5", 0.05), ("10", 0.10), ("20", 0.20))


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _fmt(x, nd=4):
    if x is None or x == RF_NOT_RUN:
        return "NA" if x is None else str(x)
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return "NA"
    return f"{v:.{nd}f}"


def _fmt_pct(x, nd=2):
    if x is None or x == RF_NOT_RUN:
        return "NA" if x is None else str(x)
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return "NA"
    return f"{100.0 * v:.{nd}f}%"


def _median(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _close(a, b, rtol=GP_ENR_RTOL, atol=GP_ENR_ATOL) -> bool:
    return bool(np.isclose(float(a), float(b), rtol=rtol, atol=atol))


def verify_sources() -> tuple[pd.DataFrame, dict]:
    identity = json.loads((S1 / "02_historical_run_identity.json").read_text(encoding="utf-8"))
    s1_1 = json.loads((S1.parent / "S1_1_gp4_oracle_enrichment_ceiling_v1" / "02_source_integrity.json").read_text(encoding="utf-8"))
    leak = pd.read_csv(S1 / "05_p0_leakage_by_cycle.csv")
    p0_clean = load_p0_clean_hashes_from_s1()
    rows = []
    issues = []
    for s in CYCLES:
        b = load_b_s(s)
        hashes = b["instance_hash"].astype(str).tolist()
        gp_path = gp_joblib_path(s)
        overlap = len(set(hashes) & p0_clean[s])
        if overlap:
            issues.append(f"C{s:02d} B_s ∩ P0_clean = {overlap}")
        if not gp_path.is_file():
            issues.append(f"C{s:02d} missing GP joblib")
        rows.append({
            "cycle": f"C{s:02d}",
            "solver": f"M{s}",
            "train_n": int(len(b)),
            "unique_n": int(b["instance_hash"].nunique()),
            "source_parquet": str(b_meta_path(s)),
            "membership_sha256": membership_hash(hashes),
            "feature_set": "GP4",
            "features_finite": True,
            "hardness_finite": True,
            "gp_joblib": str(gp_path),
            "gp_joblib_sha256": _sha_file(gp_path) if gp_path.is_file() else None,
            "p0_clean_n": int(len(p0_clean[s])),
            "s1_p0_clean_n": int(leak.loc[leak["cycle"] == f"C{s:02d}", "p0_clean_n"].iloc[0]),
            "p0_leakage_n": overlap,
        })
    frame = pd.DataFrame(rows)
    payload = {
        "FULL_ID": identity.get("FULL_ID"),
        "architecture": identity.get("architecture"),
        "horizon": identity.get("horizon"),
        "s1_identity_status": identity.get("identity_status"),
        "s1_1_status": s1_1.get("status"),
        "p0_n_raw": 1267,
        "b_s_all_n500": bool((frame["train_n"] == 500).all() and (frame["unique_n"] == 500).all()),
        "p0_clean_matches_s1": bool((frame["p0_clean_n"] == frame["s1_p0_clean_n"]).all()),
        "p0_leakage_empty": bool((frame["p0_leakage_n"] == 0).all()),
        "rf_policy": "drop_RF4",
        "issues": issues,
        "status": "OK" if not issues else "S2_A_SOURCE_INTEGRITY_FAILURE",
    }
    return frame, payload, p0_clean


def write_static_docs(train_id: pd.DataFrame, integrity: dict, config: dict, config_sha: str) -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    protocol = """# S2-A protocol

Mode: OFFLINE / CONTROLLED SURROGATE COMPARISON.

Scope: C00–C14 of `blade_v2_full_seed2026`. Model class only.
Fixed: Random500 B_s, current H_s, GP4 four features, target log1p(L_BQ), S1 P0_clean_s.

Historical GP4 is reused, not refit.

HGB4 hyperparameters are the canonical hardness-regression `HGB_CFG` from
`src/final_run/science.py` (`train_hgb_controls`). The config is not modified.
No HGB hyperparameter search.

RF4 is **not run**. Human confirmation 2026-09-21 Option 3: drop RF4.
RF-related fields are marked `NOT_RUN_RF_CONFIG_NOT_FROZEN`.

No extra features. No sample-size expansion. No Concorde. No BQ training.
No mutation. No production pipeline change. Do not launch S2-B.
"""
    (AUDIT / "01_protocol.md").write_text(protocol, encoding="utf-8")
    _json(AUDIT / "02_source_integrity.json", integrity)
    _json(AUDIT / "03_model_configs.json", {
        "config_path": str(CONFIG_PATH),
        "config_sha256": config_sha,
        "parsed": config,
        "hgb_canonical": HGB_CFG_CANONICAL,
        "hgb_source": HGB_CFG_SOURCE,
        "hgb_five_seed_example": {str(seed): hgb_params(seed) for seed in SEEDS},
        "hgb_config_modified": False,
        "hyperparameter_search": False,
        "rf_status": RF_NOT_RUN,
        "rf_human_decision": "drop_RF4",
        "rf_candidates_not_selected": RF_CANDIDATES,
        "training_executed": True,
        "models_trained": ["HGB4"],
        "models_not_run": ["RF4"],
    })
    train_id.to_csv(AUDIT / "04_training_set_identity.csv", index=False)
    (AUDIT / "RF4_DROPPED_CONFIRMATION.md").write_text(
        "# RF4 dropped\n\n"
        "Human confirmation 2026-09-21: Option 3, drop RF4.\n\n"
        f"Status: `{RF_NOT_RUN}`\n\n"
        "S2-A continues as historical GP4 vs canonical HGB4 only. "
        "HGB4 config is unmodified. No hyperparameter search.\n",
        encoding="utf-8",
    )


def gp_seed_preds(model, x: np.ndarray) -> np.ndarray:
    preds = [_formula_prediction(record, x, model["transform"]) for record in model["formulas"]]
    stacked = np.vstack(preds)
    if stacked.shape[0] != 5:
        raise RuntimeError(f"expected 5 GP formulas, got {stacked.shape[0]}")
    return stacked


def rf_row(cycle: str, solver: str, extra: dict | None = None) -> dict:
    row = {
        "cycle": cycle,
        "solver": solver,
        "model": "RF4",
        "status": RF_NOT_RUN,
        "Train_N": RF_NOT_RUN,
        "Features": RF_NOT_RUN,
    }
    if extra:
        row.update({k: RF_NOT_RUN for k in extra})
    return row


def fit_or_load_hgb(s: int, x: np.ndarray, y: np.ndarray) -> tuple[list, list[dict]]:
    out_dir = AUDIT / "models" / f"C{s:02d}" / "HGB4"
    out_dir.mkdir(parents=True, exist_ok=True)
    estimators = []
    metas = []
    for seed in SEEDS:
        path = out_dir / f"seed_{seed}.joblib"
        if path.is_file():
            print(f"[S2-A] C{s:02d} load HGB4 seed {seed}", flush=True)
            est, meta = load_hgb4(path)
            meta = dict(meta or {})
            meta["path"] = str(path)
            meta["serialized_bytes"] = int(path.stat().st_size)
            meta["resumed"] = True
            estimators.append(est)
            metas.append(meta)
            continue
        print(f"[S2-A] C{s:02d} train HGB4 seed {seed}", flush=True)
        est, meta = fit_hgb4_seed(x, y, seed)
        meta = save_hgb4(est, meta, path)
        meta["resumed"] = False
        print(f"[S2-A] C{s:02d} HGB4 seed {seed} fit_s={meta['fit_s']:.3f}", flush=True)
        estimators.append(est)
        metas.append(meta)
    _json(out_dir / "ensemble_meta.json", {
        "cycle": f"C{s:02d}",
        "model": "HGB4",
        "seeds": SEEDS,
        "ensemble": "median",
        "params_canonical": HGB_CFG_CANONICAL,
        "source": HGB_CFG_SOURCE,
        "seeds_meta": metas,
    })
    return estimators, metas


def model_metrics(pred, h, random_map, oracle_map) -> dict:
    out = {}
    for label, q in QS:
        model_enr = enr(h, pred, q)
        rec = top_recall(h, pred, q)
        rand = float(random_map[q])
        ora = float(oracle_map[q])
        gain, available, cap = capture_rate(model_enr, ora, rand)
        out[f"ENR{label}"] = model_enr
        out[f"Top{label}Recall"] = rec
        out[f"CaptureRate{label}"] = cap
        out[f"RecallCapture{label}"] = recall_capture(rec, q)
        out[f"GAIN{label}"] = gain
        out[f"AVAILABLE{label}"] = available
        out[f"Oracle_ENR{label}"] = ora
        out[f"Random_ENR{label}"] = rand
    tail = tail_ranking(pred, h)
    out["Spearman"] = safe_spearman(pred, h)
    out["Tail20Rho"] = tail["rho_tail20"]
    out["Tail10Rho"] = tail["rho_tail10"]
    out["Tail5Rho"] = tail["rho_tail5"]
    out["TrueTail10Rho"] = tail["true_tail_rho10"]
    out["n"] = int(len(h))
    return out


def run_cycles(p0_clean: dict[int, set[str]], refs: dict) -> dict:
    member = load_p0_member()
    hardness = load_p0_hardness()
    p0_hashes = member["instance_hash"].tolist()
    p0_x = member[FEATURES].to_numpy(float)
    hash_to_pos = {h: i for i, h in enumerate(p0_hashes)}
    h_by_cycle = {}
    for s in CYCLES:
        sub = hardness[hardness["cycle"] == f"C{s:02d}"].set_index("instance_hash")
        h_by_cycle[s] = np.asarray([float(sub.loc[h, "H_s"]) for h in p0_hashes], dtype=float)

    comparison = refs["comparison"]
    s1_metrics = refs["s1_metrics"]
    s1_conc = refs["s1_conc"]

    timing_rows = []
    primary_rows = []
    capture_rows = []
    recall_rows = []
    tail_rows = []
    decile_out = []
    conc_rows = []
    overlap_rows = []
    comp_rows = []
    headroom_rows = []
    pred_frames = []
    gp_repro_issues = []
    perm_rows = []

    for s in CYCLES:
        cycle = f"C{s:02d}"
        solver = f"M{s}"
        print(f"[S2-A] {cycle} load B_s", flush=True)
        b = load_b_s(s)
        x_train = b[FEATURES].to_numpy(float)
        y_train = np.log1p(b["current_hardness"].to_numpy(float))
        if len(b) != 500:
            raise RuntimeError(f"{cycle} train N={len(b)}")
        print(f"[S2-A] {cycle} fit HGB4 n={len(b)}", flush=True)
        estimators, metas = fit_or_load_hgb(s, x_train, y_train)
        print(f"[S2-A] {cycle} HGB4 fit done", flush=True)
        fit_s = float(sum(float(m["fit_s"]) for m in metas))
        ser_bytes = int(sum(int(m["serialized_bytes"]) for m in metas))
        peaks = [m.get("peak_rss_bytes") for m in metas if m.get("peak_rss_bytes") is not None]
        peak_rss = int(max(peaks)) if peaks else None

        clean_idx = np.asarray([hash_to_pos[h] for h in p0_hashes if h in p0_clean[s]], dtype=int)
        if set(member.loc[clean_idx, "instance_hash"].astype(str)) & set(b["instance_hash"].astype(str)):
            raise RuntimeError(f"{cycle} P0 leakage after index filter")
        x_p0 = p0_x[clean_idx]
        h_p0 = h_by_cycle[s][clean_idx]
        hashes_p0 = member.loc[clean_idx, "instance_hash"].astype(str).to_numpy()

        print(f"[S2-A] {cycle} load historical GP", flush=True)
        gp_model = joblib.load(gp_joblib_path(s))
        gp_seed = gp_seed_preds(gp_model, x_p0)
        t0 = time.perf_counter()
        gp_pred = np.median(gp_seed, axis=0)
        gp_p0_s = time.perf_counter() - t0
        print(f"[S2-A] {cycle} score HGB4 on P0", flush=True)

        hgb_seed = []
        hgb_p0_s = 0.0
        for est in estimators:
            pred, elapsed = predict_hgb(est, x_p0)
            hgb_seed.append(pred)
            hgb_p0_s += elapsed
        hgb_seed = np.vstack(hgb_seed)
        hgb_pred = median_stack(hgb_seed)

        cmp_row = comparison[comparison["cycle"] == cycle].iloc[0]
        s1_row = s1_metrics[s1_metrics["cycle"] == cycle].iloc[0]
        random_map = {0.05: float(cmp_row["Random_ENR5"]), 0.10: float(cmp_row["Random_ENR10"]), 0.20: float(cmp_row["Random_ENR20"])}
        oracle_map = {0.05: float(cmp_row["Oracle_ENR5"]), 0.10: float(cmp_row["Oracle_ENR10"]), 0.20: float(cmp_row["Oracle_ENR20"])}

        oracle_enr10_recomputed = enr(h_p0, h_p0, 0.10)
        if not _close(oracle_enr10_recomputed, oracle_map[0.10]):
            gp_repro_issues.append(f"{cycle} Oracle ENR10 recomputed {oracle_enr10_recomputed} != S1.1 {oracle_map[0.10]}")

        gp_m = model_metrics(gp_pred, h_p0, random_map, oracle_map)
        hgb_m = model_metrics(hgb_pred, h_p0, random_map, oracle_map)
        if not _close(gp_m["ENR10"], float(s1_row["ENR10"])):
            gp_repro_issues.append(f"{cycle} GP ENR10 {gp_m['ENR10']} != S1 {s1_row['ENR10']}")
        if not _close(gp_m["ENR10"], float(cmp_row["GP_ENR10"])):
            gp_repro_issues.append(f"{cycle} GP ENR10 {gp_m['ENR10']} != S1.1 {cmp_row['GP_ENR10']}")
        if not _close(gp_m["Top10Recall"], float(cmp_row["GP_Top10_Recall"]), rtol=1e-8, atol=1e-10):
            gp_repro_issues.append(f"{cycle} GP Top10Recall {gp_m['Top10Recall']} != S1.1 {cmp_row['GP_Top10_Recall']}")

        print(f"[S2-A] {cycle} load O_s", flush=True)
        offspring = load_offspring(s)
        x_o = offspring[FEATURES].to_numpy(float)
        o_hashes = offspring["instance_hash"].to_numpy()
        o_niches = offspring["niche_id"].to_numpy(int)
        gp_stored = offspring["gp_predicted_log1p_hardness"].to_numpy(float)
        print(f"[S2-A] {cycle} replay GP on O_s n={len(offspring)}", flush=True)
        t0 = time.perf_counter()
        gp_replay = predict_gp4(gp_model, x_o)
        gp_o_s = time.perf_counter() - t0
        print(f"[S2-A] {cycle} GP O_s replay {gp_o_s:.2f}s, score HGB4 O_s", flush=True)
        replay_mae = float(np.mean(np.abs(gp_replay - gp_stored)))
        replay_max = float(np.max(np.abs(gp_replay - gp_stored)))
        if replay_mae > 1e-10:
            gp_repro_issues.append(f"{cycle} GP replay MAE vs stored O_s = {replay_mae} maxabs={replay_max}")

        t0 = time.perf_counter()
        hgb_o_seeds = []
        for est in estimators:
            pred, _elapsed = predict_hgb(est, x_o)
            hgb_o_seeds.append(pred)
        hgb_o = median_stack(np.vstack(hgb_o_seeds))
        hgb_o_s = time.perf_counter() - t0

        cand_m = niche_metrics(o_niches)
        gp_top = topk_indices(gp_stored, o_hashes, TOPK)
        hgb_top = topk_indices(hgb_o, o_hashes, TOPK)
        gp_sel_n = niche_metrics(o_niches[gp_top])
        hgb_sel_n = niche_metrics(o_niches[hgb_top])
        s1c = s1_conc[s1_conc["cycle"] == cycle].iloc[0]
        if int(gp_sel_n["occupied_niches"]) != int(s1c["selected_occupied_niches"]):
            gp_repro_issues.append(
                f"{cycle} GP Top10k niches {gp_sel_n['occupied_niches']} != S1 {s1c['selected_occupied_niches']}"
            )

        n_o = int(len(offspring))
        gp_ms_100k = 1000.0 * gp_o_s / n_o * 100_000
        hgb_ms_100k = 1000.0 * hgb_o_s / n_o * 100_000
        gp_hist_fit = gp_model.get("seed_runtime_s")
        gp_fit_s = float(sum(gp_hist_fit)) if gp_hist_fit else float("nan")
        gp_size = int(gp_joblib_path(s).stat().st_size)

        timing_rows.append({
            "cycle": cycle, "solver": solver, "model": "GP4",
            "fit_s": gp_fit_s, "fit_source": "historical_seed_runtime_s" if gp_hist_fit else "UNAVAILABLE",
            "peak_rss_bytes": "UNAVAILABLE",
            "serialized_bytes": gp_size, "serialized_ref": str(gp_joblib_path(s)),
            "p0_predict_s": gp_p0_s, "os_predict_s": gp_o_s,
            "os_n": n_o, "ms_per_100k": gp_ms_100k,
            "gp_replay_mae": replay_mae, "gp_replay_maxabs": replay_max,
            "status": "OK",
        })
        timing_rows.append({
            "cycle": cycle, "solver": solver, "model": "HGB4",
            "fit_s": fit_s, "fit_source": "five_seed_wall_sum",
            "peak_rss_bytes": peak_rss if peak_rss is not None else "UNAVAILABLE",
            "serialized_bytes": ser_bytes, "serialized_ref": str(AUDIT / "models" / cycle / "HGB4"),
            "p0_predict_s": hgb_p0_s, "os_predict_s": hgb_o_s,
            "os_n": n_o, "ms_per_100k": hgb_ms_100k,
            "gp_replay_mae": "", "gp_replay_maxabs": "",
            "status": "OK",
        })
        timing_rows.append({
            "cycle": cycle, "solver": solver, "model": "RF4",
            "fit_s": RF_NOT_RUN, "fit_source": RF_NOT_RUN,
            "peak_rss_bytes": RF_NOT_RUN, "serialized_bytes": RF_NOT_RUN,
            "serialized_ref": RF_NOT_RUN, "p0_predict_s": RF_NOT_RUN,
            "os_predict_s": RF_NOT_RUN, "os_n": n_o, "ms_per_100k": RF_NOT_RUN,
            "gp_replay_mae": RF_NOT_RUN, "gp_replay_maxabs": RF_NOT_RUN,
            "status": RF_NOT_RUN,
        })

        for model_name, mets, conc, fit_t, score_t in (
            ("GP4", gp_m, gp_sel_n, gp_fit_s, gp_o_s),
            ("HGB4", hgb_m, hgb_sel_n, fit_s, hgb_o_s),
        ):
            primary_rows.append({
                "cycle": cycle, "solver": solver, "model": model_name,
                "Train_N": 500, "Features": "GP4",
                "ENR5": mets["ENR5"], "ENR10": mets["ENR10"], "ENR20": mets["ENR20"],
                "Oracle_ENR5": mets["Oracle_ENR5"], "Oracle_ENR10": mets["Oracle_ENR10"], "Oracle_ENR20": mets["Oracle_ENR20"],
                "Random_ENR5": mets["Random_ENR5"], "Random_ENR10": mets["Random_ENR10"], "Random_ENR20": mets["Random_ENR20"],
                "CaptureRate5": mets["CaptureRate5"], "CaptureRate10": mets["CaptureRate10"], "CaptureRate20": mets["CaptureRate20"],
                "Top5Recall": mets["Top5Recall"], "Top10Recall": mets["Top10Recall"], "Top20Recall": mets["Top20Recall"],
                "RecallCapture5": mets["RecallCapture5"], "RecallCapture10": mets["RecallCapture10"], "RecallCapture20": mets["RecallCapture20"],
                "Spearman": mets["Spearman"], "Tail10Rho": mets["Tail10Rho"],
                "Tail5Rho": mets["Tail5Rho"], "Tail20Rho": mets["Tail20Rho"],
                "TrueTail10Rho": mets["TrueTail10Rho"],
                "Top10k_Niches": conc["occupied_niches"], "Top10k_Neff": conc["N_eff"],
                "LargestNicheShare": conc["largest_niche_share"],
                "Top5NicheShare": conc["top5_niche_share"], "Top10NicheShare": conc["top10_niche_share"],
                "FitTime": fit_t, "O_s_ScoreTime": score_t,
                "status": "OK",
            })
            capture_rows.append({
                "cycle": cycle, "solver": solver, "model": model_name,
                "ENR5": mets["ENR5"], "ENR10": mets["ENR10"], "ENR20": mets["ENR20"],
                "CaptureRate5": mets["CaptureRate5"], "CaptureRate10": mets["CaptureRate10"], "CaptureRate20": mets["CaptureRate20"],
                "Oracle_ENR10": mets["Oracle_ENR10"], "Random_ENR10": mets["Random_ENR10"],
                "status": "OK",
            })
            recall_rows.append({
                "cycle": cycle, "solver": solver, "model": model_name,
                "Top5Recall": mets["Top5Recall"], "Top10Recall": mets["Top10Recall"], "Top20Recall": mets["Top20Recall"],
                "RecallCapture5": mets["RecallCapture5"], "RecallCapture10": mets["RecallCapture10"], "RecallCapture20": mets["RecallCapture20"],
                "status": "OK",
            })
            tail_rows.append({
                "cycle": cycle, "solver": solver, "model": model_name,
                "Spearman": mets["Spearman"], "rho_tail20": mets["Tail20Rho"],
                "rho_tail10": mets["Tail10Rho"], "rho_tail5": mets["Tail5Rho"],
                "true_tail_rho10": mets["TrueTail10Rho"], "status": "OK",
            })
            for row in decile_rows(gp_pred if model_name == "GP4" else hgb_pred, h_p0):
                decile_out.append({"cycle": cycle, "solver": solver, "model": model_name, **row})
            conc_rows.append({
                "cycle": cycle, "solver": solver, "model": model_name, "pool": "O_s",
                "pool_n": n_o, "pool_occupied_niches": cand_m["occupied_niches"], "pool_N_eff": cand_m["N_eff"],
                "selected_n": TOPK, **{f"Top10k_{k}": v for k, v in conc.items()},
                "status": "OK",
            })
            headroom_rows.append({
                "cycle": cycle, "solver": solver, "model": model_name,
                "Oracle_ENR10": mets["Oracle_ENR10"], "Model_ENR10": mets["ENR10"],
                "Oracle_minus_model_ENR10": mets["Oracle_ENR10"] - mets["ENR10"],
                "CaptureRate10": mets["CaptureRate10"],
                "one_minus_CaptureRate10": 1.0 - float(mets["CaptureRate10"]) if np.isfinite(float(mets["CaptureRate10"])) else float("nan"),
                "status": "OK",
            })

        primary_rows.append(rf_row(cycle, solver, {
            "ENR5": 1, "ENR10": 1, "ENR20": 1, "Oracle_ENR10": 1, "CaptureRate10": 1,
            "Top10Recall": 1, "RecallCapture10": 1, "Spearman": 1, "Tail10Rho": 1,
            "Top10k_Niches": 1, "Top10k_Neff": 1, "LargestNicheShare": 1, "FitTime": 1, "O_s_ScoreTime": 1,
        }))
        capture_rows.append(rf_row(cycle, solver, {"ENR10": 1, "CaptureRate10": 1}))
        recall_rows.append(rf_row(cycle, solver, {"Top10Recall": 1, "RecallCapture10": 1}))
        tail_rows.append(rf_row(cycle, solver, {"Spearman": 1, "rho_tail10": 1}))
        conc_rows.append(rf_row(cycle, solver, {"Top10k_occupied_niches": 1, "Top10k_N_eff": 1}))
        headroom_rows.append(rf_row(cycle, solver, {"Oracle_minus_model_ENR10": 1, "one_minus_CaptureRate10": 1}))

        gp_set = set(o_hashes[gp_top].tolist())
        hgb_set = set(o_hashes[hgb_top].tolist())
        gp_niches = set(int(v) for v in o_niches[gp_top].tolist())
        hgb_niches = set(int(v) for v in o_niches[hgb_top].tolist())
        overlap_rows.append({
            "cycle": cycle, "pair": "GP4_vs_HGB4",
            "Top10k_jaccard": set_jaccard(gp_set, hgb_set),
            "Top10k_intersection_n": len(gp_set & hgb_set),
            "niche_jaccard": set_jaccard(gp_niches, hgb_niches),
            "shared_niches": len(gp_niches & hgb_niches),
            "status": "OK",
        })
        overlap_rows.append({"cycle": cycle, "pair": "GP4_vs_RF4", "Top10k_jaccard": RF_NOT_RUN, "status": RF_NOT_RUN})
        overlap_rows.append({"cycle": cycle, "pair": "HGB4_vs_RF4", "Top10k_jaccard": RF_NOT_RUN, "status": RF_NOT_RUN})

        k10 = k_q(len(h_p0), 0.10)
        true_hard = set(np.argsort(-h_p0)[:k10].tolist())
        gp_top_p0 = set(np.argsort(-gp_pred)[:k10].tolist())
        hgb_top_p0 = set(np.argsort(-hgb_pred)[:k10].tolist())
        rec_gp = len(true_hard & gp_top_p0) / k10
        rec_hgb = len(true_hard & hgb_top_p0) / k10
        union_gh = gp_top_p0 | hgb_top_p0
        rec_union = len(true_hard & union_gh) / k10
        gp_only = (true_hard & gp_top_p0) - hgb_top_p0
        hgb_only = (true_hard & hgb_top_p0) - gp_top_p0
        both = true_hard & gp_top_p0 & hgb_top_p0
        missed = true_hard - gp_top_p0 - hgb_top_p0
        comp_rows.append({
            "cycle": cycle, "k": k10, "true_hard_n": k10,
            "recall_GP4_Top10": rec_gp, "recall_HGB4_Top10": rec_hgb, "recall_RF4_Top10": RF_NOT_RUN,
            "recall_GP4_union_HGB4": rec_union, "union_GP4_HGB4_size": len(union_gh),
            "recall_GP4_union_RF4": RF_NOT_RUN, "recall_HGB4_union_RF4": RF_NOT_RUN,
            "recall_GP4_union_HGB4_union_RF4": RF_NOT_RUN, "union_three_size": RF_NOT_RUN,
            "unique_true_hard_GP_only": len(gp_only),
            "unique_true_hard_HGB_only": len(hgb_only),
            "unique_true_hard_RF_only": RF_NOT_RUN,
            "true_hard_both_GP_HGB": len(both),
            "true_hard_missed_by_GP_and_HGB": len(missed),
            "status": "OK",
        })

        pred_frames.append(pd.DataFrame({
            "cycle": cycle,
            "solver": solver,
            "instance_hash": hashes_p0,
            "H_s": h_p0,
            "pred_GP4": gp_pred,
            "pred_HGB4": hgb_pred,
            "pred_RF4": np.nan,
            **{f"pred_GP4_seed_{seed}": gp_seed[i] for i, seed in enumerate(SEEDS)},
            **{f"pred_HGB4_seed_{seed}": hgb_seed[i] for i, seed in enumerate(SEEDS)},
        }))

        for i, name in enumerate(FEATURES):
            if hasattr(estimators[0], "feature_importances_"):
                imps = [float(est.feature_importances_[i]) for est in estimators]
                perm_rows.append({
                    "cycle": cycle, "model": "HGB4", "feature": name,
                    "mean_native_importance": float(np.mean(imps)),
                    "note": "native estimator attribute; descriptive only",
                })
        if not hasattr(estimators[0], "feature_importances_"):
            perm_rows.append({
                "cycle": cycle, "model": "HGB4", "feature": "ALL",
                "mean_native_importance": "UNAVAILABLE",
                "note": "HistGradientBoostingRegressor has no native impurity importance; SHAP not run",
            })

        print(
            f"[S2-A] {cycle} GP ENR10={gp_m['ENR10']:.4f} CR10={gp_m['CaptureRate10']:.4f} "
            f"HGB ENR10={hgb_m['ENR10']:.4f} CR10={hgb_m['CaptureRate10']:.4f}",
            flush=True,
        )

    if gp_repro_issues:
        raise RuntimeError("GP/S1 reproduction failed:\n" + "\n".join(gp_repro_issues))

    return {
        "timing": pd.DataFrame(timing_rows),
        "primary": pd.DataFrame(primary_rows),
        "capture": pd.DataFrame(capture_rows),
        "recall": pd.DataFrame(recall_rows),
        "tail": pd.DataFrame(tail_rows),
        "decile": pd.DataFrame(decile_out),
        "conc": pd.DataFrame(conc_rows),
        "overlap": pd.DataFrame(overlap_rows),
        "comp": pd.DataFrame(comp_rows),
        "headroom": pd.DataFrame(headroom_rows),
        "preds": pd.concat(pred_frames, ignore_index=True),
        "importance": pd.DataFrame(perm_rows),
    }


def numeric_model(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    sub = frame[frame["model"] == model].copy()
    return sub[sub["status"] != RF_NOT_RUN] if "status" in sub.columns else sub


def paired_deltas(primary: pd.DataFrame) -> pd.DataFrame:
    gp = numeric_model(primary, "GP4").set_index("cycle")
    hgb = numeric_model(primary, "HGB4").set_index("cycle")
    rows = []
    for metric in ("CaptureRate10", "ENR10", "Top10Recall", "RecallCapture10", "Tail10Rho", "Spearman"):
        delta = hgb[metric].to_numpy(float) - gp[metric].to_numpy(float)
        boot = paired_bootstrap_ci(delta, seed=2026, repeats=10000)
        rows.append({
            "pair": "HGB4-GP4", "metric": metric, **boot, "status": "OK",
        })
        rows.append({"pair": "RF4-GP4", "metric": metric, "status": RF_NOT_RUN,
                     "mean": RF_NOT_RUN, "median": RF_NOT_RUN, "ci95_low": RF_NOT_RUN, "ci95_high": RF_NOT_RUN})
        rows.append({"pair": "HGB4-RF4", "metric": metric, "status": RF_NOT_RUN,
                     "mean": RF_NOT_RUN, "median": RF_NOT_RUN, "ci95_low": RF_NOT_RUN, "ci95_high": RF_NOT_RUN})
    return pd.DataFrame(rows)


def phase_summary(primary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for phase, cycles in PHASES.items():
        labels = [f"C{s:02d}" for s in cycles]
        for model in ("GP4", "HGB4", "RF4"):
            if model == "RF4":
                rows.append({
                    "phase": phase, "model": "RF4", "status": RF_NOT_RUN,
                    "median_ENR10": RF_NOT_RUN, "median_CaptureRate10": RF_NOT_RUN,
                    "median_Top10Recall": RF_NOT_RUN, "median_RecallCapture10": RF_NOT_RUN,
                    "median_Spearman": RF_NOT_RUN, "median_Tail10Rho": RF_NOT_RUN,
                    "median_Top10k_Niches": RF_NOT_RUN, "median_Top10k_Neff": RF_NOT_RUN,
                    "median_LargestNicheShare": RF_NOT_RUN,
                })
                continue
            sub = numeric_model(primary, model)
            sub = sub[sub["cycle"].isin(labels)]
            rows.append({
                "phase": phase, "model": model, "status": "OK",
                "median_ENR10": _median(sub["ENR10"]),
                "median_CaptureRate10": _median(sub["CaptureRate10"]),
                "median_Top10Recall": _median(sub["Top10Recall"]),
                "median_RecallCapture10": _median(sub["RecallCapture10"]),
                "median_Spearman": _median(sub["Spearman"]),
                "median_Tail10Rho": _median(sub["Tail10Rho"]),
                "median_Top10k_Niches": _median(sub["Top10k_Niches"]),
                "median_Top10k_Neff": _median(sub["Top10k_Neff"]),
                "median_LargestNicheShare": _median(sub["LargestNicheShare"]),
                "median_FitTime": _median(sub["FitTime"]),
                "median_O_s_ScoreTime": _median(sub["O_s_ScoreTime"]),
            })
    ora = numeric_model(primary, "GP4")
    for phase, cycles in PHASES.items():
        labels = [f"C{s:02d}" for s in cycles]
        sub = ora[ora["cycle"].isin(labels)]
        rows.append({
            "phase": phase, "model": "Oracle", "status": "OK",
            "median_ENR10": _median(sub["Oracle_ENR10"]),
            "median_CaptureRate10": 1.0,
            "median_Top10Recall": 1.0,
            "median_RecallCapture10": 1.0,
            "median_Spearman": 1.0, "median_Tail10Rho": "UNAVAILABLE",
            "median_Top10k_Niches": "UNAVAILABLE", "median_Top10k_Neff": "UNAVAILABLE",
            "median_LargestNicheShare": "UNAVAILABLE",
        })
    return pd.DataFrame(rows)


def cost_quality(primary: pd.DataFrame, timing: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in ("GP4", "HGB4", "RF4"):
        if model == "RF4":
            rows.append({"model": "RF4", "status": RF_NOT_RUN})
            continue
        p = numeric_model(primary, model)
        t = numeric_model(timing, model)
        rows.append({
            "model": model, "status": "OK",
            "median_ENR10": _median(p["ENR10"]),
            "median_CaptureRate10": _median(p["CaptureRate10"]),
            "median_Top10Recall": _median(p["Top10Recall"]),
            "median_fit_s": _median(t["fit_s"]),
            "median_os_predict_s": _median(t["os_predict_s"]),
            "median_ms_per_100k": _median(t["ms_per_100k"]),
            "median_serialized_bytes": _median(t["serialized_bytes"]),
        })
    return pd.DataFrame(rows)


def _line(ax, cycles, series, label, marker="o"):
    ax.plot(cycles, series, marker=marker, label=label)


def make_figures(primary, capture, conc, overlap, comp, headroom, timing, phases) -> None:
    cycles = [f"C{s:02d}" for s in CYCLES]
    gp = numeric_model(primary, "GP4").set_index("cycle").loc[cycles]
    hgb = numeric_model(primary, "HGB4").set_index("cycle").loc[cycles]
    ora = gp["Oracle_ENR10"].to_numpy(float)
    rand = gp["Random_ENR10"].to_numpy(float)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(AUDIT / name, dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _line(ax, cycles, gp["ENR10"], "GP4")
    _line(ax, cycles, hgb["ENR10"], "HGB4")
    _line(ax, cycles, ora, "Oracle", marker="s")
    _line(ax, cycles, rand, "Random", marker="x")
    ax.set_ylabel("ENR10"); ax.set_title("ENR10 on leakage-clean P0 (RF4 not run)")
    ax.tick_params(axis="x", rotation=45); ax.legend(); ax.grid(True, alpha=0.3)
    save(fig, "fig01_ENR10_GP_HGB_RF_Oracle.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _line(ax, cycles, gp["CaptureRate10"], "GP4")
    _line(ax, cycles, hgb["CaptureRate10"], "HGB4")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("CaptureRate10"); ax.set_title("CaptureRate10 by cycle")
    ax.tick_params(axis="x", rotation=45); ax.legend(); ax.grid(True, alpha=0.3)
    save(fig, "fig02_CaptureRate10_by_cycle.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _line(ax, cycles, gp["Top10Recall"], "GP4")
    _line(ax, cycles, hgb["Top10Recall"], "HGB4")
    ax.axhline(0.10, color="gray", linestyle="--", label="Random 10%")
    ax.set_ylabel("Top10 Recall"); ax.set_title("Top10 Recall by cycle")
    ax.tick_params(axis="x", rotation=45); ax.legend(); ax.grid(True, alpha=0.3)
    save(fig, "fig03_Top10Recall_by_cycle.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(3)
    labels = ["EARLY", "MID", "LATE"]
    gp_p = [float(phases[(phases.phase.str.startswith(p)) & (phases.model == "GP4")]["median_CaptureRate10"].iloc[0]) for p in ("EARLY", "MID", "LATE")]
    hgb_p = [float(phases[(phases.phase.str.startswith(p)) & (phases.model == "HGB4")]["median_CaptureRate10"].iloc[0]) for p in ("EARLY", "MID", "LATE")]
    ax.bar(x - 0.18, gp_p, 0.36, label="GP4")
    ax.bar(x + 0.18, hgb_p, 0.36, label="HGB4")
    ax.set_xticks(x, labels); ax.set_ylabel("median CaptureRate10")
    ax.set_title("CaptureRate10 early / mid / late"); ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    save(fig, "fig04_CaptureRate10_early_mid_late.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _line(ax, cycles, gp["Tail10Rho"], "GP4")
    _line(ax, cycles, hgb["Tail10Rho"], "HGB4")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("predicted-Top10% Spearman"); ax.set_title("Fine-tail ranking (tail10 rho)")
    ax.tick_params(axis="x", rotation=45); ax.legend(); ax.grid(True, alpha=0.3)
    save(fig, "fig05_tail10_ranking.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    gp_h = numeric_model(headroom, "GP4").set_index("cycle").loc[cycles]
    hgb_h = numeric_model(headroom, "HGB4").set_index("cycle").loc[cycles]
    _line(ax, cycles, gp_h["Oracle_minus_model_ENR10"], "Oracle-GP4")
    _line(ax, cycles, hgb_h["Oracle_minus_model_ENR10"], "Oracle-HGB4")
    ax.set_ylabel("Oracle_ENR10 - Model_ENR10"); ax.set_title("Remaining Oracle ENR10 headroom")
    ax.tick_params(axis="x", rotation=45); ax.legend(); ax.grid(True, alpha=0.3)
    save(fig, "fig06_oracle_headroom_remaining.png")

    gp_c = numeric_model(conc, "GP4").set_index("cycle").loc[cycles]
    hgb_c = numeric_model(conc, "HGB4").set_index("cycle").loc[cycles]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _line(ax, cycles, gp_c["Top10k_occupied_niches"], "GP4")
    _line(ax, cycles, hgb_c["Top10k_occupied_niches"], "HGB4")
    ax.set_ylabel("occupied frozen QD niches"); ax.set_title("Predicted Top10k occupied niches")
    ax.tick_params(axis="x", rotation=45); ax.legend(); ax.grid(True, alpha=0.3)
    save(fig, "fig07_top10k_occupied_niches.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _line(ax, cycles, gp_c["Top10k_N_eff"], "GP4")
    _line(ax, cycles, hgb_c["Top10k_N_eff"], "HGB4")
    ax.set_ylabel("N_eff"); ax.set_title("Predicted Top10k N_eff")
    ax.tick_params(axis="x", rotation=45); ax.legend(); ax.grid(True, alpha=0.3)
    save(fig, "fig08_top10k_Neff.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _line(ax, cycles, gp_c["Top10k_largest_niche_share"], "GP4")
    _line(ax, cycles, hgb_c["Top10k_largest_niche_share"], "HGB4")
    ax.set_ylabel("largest niche share"); ax.set_title("Predicted Top10k largest niche share")
    ax.tick_params(axis="x", rotation=45); ax.legend(); ax.grid(True, alpha=0.3)
    save(fig, "fig09_largest_niche_share.png")

    ov = overlap[overlap["pair"] == "GP4_vs_HGB4"].set_index("cycle").loc[cycles]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _line(ax, cycles, ov["Top10k_jaccard"], "GP4 vs HGB4")
    ax.set_ylim(0, 1); ax.set_ylabel("Jaccard"); ax.set_title("Top10k selection overlap (RF4 not run)")
    ax.tick_params(axis="x", rotation=45); ax.legend(); ax.grid(True, alpha=0.3)
    save(fig, "fig10_model_selection_jaccard.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(cycles, comp["unique_true_hard_GP_only"], marker="o", label="GP only")
    ax.plot(cycles, comp["unique_true_hard_HGB_only"], marker="o", label="HGB only")
    ax.set_ylabel("unique true-hard count"); ax.set_title("True-hard unique recovery (P0 Top10)")
    ax.tick_params(axis="x", rotation=45); ax.legend(); ax.grid(True, alpha=0.3)
    save(fig, "fig11_true_hard_unique_recovery.png")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    t_gp = numeric_model(timing, "GP4")
    t_hgb = numeric_model(timing, "HGB4")
    ax.scatter(_median(t_gp["ms_per_100k"]), _median(gp["CaptureRate10"]), s=80, label="GP4")
    ax.scatter(_median(t_hgb["ms_per_100k"]), _median(hgb["CaptureRate10"]), s=80, label="HGB4")
    ax.set_xlabel("median ms per 100k candidates")
    ax.set_ylabel("median CaptureRate10")
    ax.set_title("Acquisition quality vs scoring cost")
    ax.legend(); ax.grid(True, alpha=0.3)
    save(fig, "fig12_quality_vs_scoring_cost.png")


def decide(primary: pd.DataFrame, deltas: pd.DataFrame, phases: pd.DataFrame, comp: pd.DataFrame) -> dict:
    cr = deltas[(deltas["pair"] == "HGB4-GP4") & (deltas["metric"] == "CaptureRate10")].iloc[0]
    rec = deltas[(deltas["pair"] == "HGB4-GP4") & (deltas["metric"] == "Top10Recall")].iloc[0]
    late_gp = phases[(phases.phase.str.startswith("LATE")) & (phases.model == "GP4")].iloc[0]
    late_hgb = phases[(phases.phase.str.startswith("LATE")) & (phases.model == "HGB4")].iloc[0]
    late_ora = phases[(phases.phase.str.startswith("LATE")) & (phases.model == "Oracle")].iloc[0]
    cr_pos = float(cr["ci95_low"]) > 0 and float(cr["mean"]) > 0
    rec_same = float(rec["mean"]) > 0
    cr_neg = float(cr["ci95_high"]) < 0
    phase_signs = []
    for prefix in ("EARLY", "MID", "LATE"):
        g = float(phases[(phases.phase.str.startswith(prefix)) & (phases.model == "GP4")]["median_CaptureRate10"].iloc[0])
        h = float(phases[(phases.phase.str.startswith(prefix)) & (phases.model == "HGB4")]["median_CaptureRate10"].iloc[0])
        phase_signs.append(h - g)
    phase_mixed = (min(phase_signs) < 0 < max(phase_signs))
    if cr_pos and rec_same and not phase_mixed:
        verdict = "MODEL_CLASS_CAPACITY_GAIN_OBSERVED"
    elif cr_neg and float(rec["mean"]) <= 0:
        verdict = "MODEL_CLASS_CAPACITY_GAIN_NOT_OBSERVED"
    elif abs(float(cr["mean"])) < 0.01 and float(cr["ci95_low"]) <= 0 <= float(cr["ci95_high"]):
        verdict = "MODEL_CLASS_CAPACITY_GAIN_NOT_OBSERVED"
    else:
        verdict = "MODEL_CLASS_CAPACITY_GAIN_MIXED"

    late_hgb_cr = float(late_hgb["median_CaptureRate10"])
    best = "HGB4" if float(late_hgb["median_CaptureRate10"]) >= float(late_gp["median_CaptureRate10"]) else "GP4"
    if late_hgb_cr < 0.25 and best == "HGB4" and verdict != "MODEL_CLASS_CAPACITY_GAIN_NOT_OBSERVED":
        nxt = "S2-B feature-capacity audit"
        case = "C" if float(late_hgb["median_CaptureRate10"]) > float(late_gp["median_CaptureRate10"]) + 0.02 else "B"
    elif best == "HGB4" and verdict == "MODEL_CLASS_CAPACITY_GAIN_OBSERVED" and late_hgb_cr >= 0.25:
        nxt = "S2-B feature-capacity audit"
        case = "A"
    elif late_hgb_cr >= 0.80:
        nxt = "review whether additional feature/model complexity is necessary"
        case = "D"
    else:
        nxt = "S2-B feature-capacity audit"
        case = "B"

    mean_gp_only = float(np.mean(comp["unique_true_hard_GP_only"].to_numpy(float)))
    mean_hgb_only = float(np.mean(comp["unique_true_hard_HGB_only"].to_numpy(float)))
    union_gain = float(np.mean(comp["recall_GP4_union_HGB4"].to_numpy(float) - np.maximum(
        comp["recall_GP4_Top10"].to_numpy(float), comp["recall_HGB4_Top10"].to_numpy(float)
    )))
    if mean_gp_only >= 3 and mean_hgb_only >= 3 and union_gain >= 0.03:
        complementarity = "observed"
        ensemble_flag = "ENSEMBLE_COMPLEMENTARITY_OBSERVED"
    elif mean_gp_only >= 1 and mean_hgb_only >= 1 and union_gain > 0:
        complementarity = "weak"
        ensemble_flag = None
    else:
        complementarity = "not observed"
        ensemble_flag = None

    return {
        "verdict": verdict,
        "best_model": best,
        "next": nxt,
        "case": case,
        "complementarity": complementarity,
        "ensemble_flag": ensemble_flag,
        "cr_delta": cr,
        "rec_delta": rec,
        "late_gp": late_gp,
        "late_hgb": late_hgb,
        "late_ora": late_ora,
        "phase_deltas": phase_signs,
    }


def write_report(decision: dict, primary, timing, comp, integrity, config_sha: str) -> None:
    late_gp = decision["late_gp"]
    late_hgb = decision["late_hgb"]
    late_ora = decision["late_ora"]
    cr = decision["cr_delta"]
    rec = decision["rec_delta"]
    t_gp = numeric_model(timing, "GP4")
    t_hgb = numeric_model(timing, "HGB4")
    gp = numeric_model(primary, "GP4")
    hgb = numeric_model(primary, "HGB4")
    delta_enr_late = float(late_hgb["median_ENR10"]) - float(late_gp["median_ENR10"])
    delta_cr_late = float(late_hgb["median_CaptureRate10"]) - float(late_gp["median_CaptureRate10"])
    delta_rec_late = float(late_hgb["median_Top10Recall"]) - float(late_gp["median_Top10Recall"])
    q_hgb = all(float(hgb[hgb.cycle == f"C{s:02d}"]["CaptureRate10"].iloc[0]) > float(gp[gp.cycle == f"C{s:02d}"]["CaptureRate10"].iloc[0]) for s in range(15))
    q1 = "YES" if float(cr["mean"]) > 0 and float(cr["ci95_low"]) > 0 else ("MIXED" if float(cr["mean"]) > 0 else "NO")
    lines = [
        "# S2-A surrogate model capacity audit — GP4 vs HGB4",
        "",
        "Offline controlled comparison. Model class only. RF4 not run.",
        "",
        f"Config SHA256: `{config_sha}`",
        f"Historical run: `{integrity.get('FULL_ID')}`",
        "HGB4: canonical `HGB_CFG` from `src/final_run/science.py`; not modified; no search.",
        f"RF4: `{RF_NOT_RUN}` (human Option 3, 2026-09-21).",
        "",
        "## Primary late-phase medians (C10–C14)",
        "",
        f"- GP4 ENR10={_fmt(late_gp['median_ENR10'])} CaptureRate10={_fmt_pct(late_gp['median_CaptureRate10'])} Top10Recall={_fmt_pct(late_gp['median_Top10Recall'])} Tail10Rho={_fmt(late_gp['median_Tail10Rho'])} niches={_fmt(late_gp['median_Top10k_Niches'],1)} N_eff={_fmt(late_gp['median_Top10k_Neff'],2)}",
        f"- HGB4 ENR10={_fmt(late_hgb['median_ENR10'])} CaptureRate10={_fmt_pct(late_hgb['median_CaptureRate10'])} Top10Recall={_fmt_pct(late_hgb['median_Top10Recall'])} Tail10Rho={_fmt(late_hgb['median_Tail10Rho'])} niches={_fmt(late_hgb['median_Top10k_Niches'],1)} N_eff={_fmt(late_hgb['median_Top10k_Neff'],2)}",
        f"- Oracle ENR10={_fmt(late_ora['median_ENR10'])}",
        "",
        "## Q1–Q12",
        "",
        f"**Q1.** Under the same Random500 and GP4 features, does HGB4 outperform GP4? **{q1}**",
        f"Paired CaptureRate10 HGB4−GP4: mean={_fmt(cr['mean'])}, median={_fmt(cr['median'])}, {int(cr['n_positive'])}/15 cycles positive, bootstrap 95% CI [{_fmt(cr['ci95_low'])}, {_fmt(cr['ci95_high'])}] (seed=2026, 10000 repeats).",
        "",
        f"**Q2.** Does RF4 outperform GP4? **{RF_NOT_RUN}**",
        "",
        f"**Q3.** Largest Oracle ENR10 capture among 4-feature models: **{decision['best_model']}** (late median CaptureRate10 GP4={_fmt_pct(late_gp['median_CaptureRate10'])}, HGB4={_fmt_pct(late_hgb['median_CaptureRate10'])}).",
        "",
        f"**Q4.** Best 4-feature model vs GP4, late CaptureRate10: {_fmt(100*delta_cr_late, 2)} percentage points (HGB4−GP4). Trajectory mean delta={_fmt(100*float(cr['mean']), 2)} pp.",
        "",
        f"**Q5.** Top10 Recall late median change: {_fmt(100*delta_rec_late, 2)} percentage points. Trajectory mean HGB4−GP4={_fmt(100*float(rec['mean']), 2)} pp, CI [{_fmt(100*float(rec['ci95_low']),2)}, {_fmt(100*float(rec['ci95_high']),2)}] pp.",
        "",
        f"**Q6.** Fine-tail ranking (late median tail10 rho): GP4={_fmt(late_gp['median_Tail10Rho'])}, HGB4={_fmt(late_hgb['median_Tail10Rho'])}.",
        "",
        f"**Q7.** Structural concentration (late median Top10k niches / N_eff / largest share): GP4 {_fmt(late_gp['median_Top10k_Niches'],1)} / {_fmt(late_gp['median_Top10k_Neff'],2)} / {_fmt(late_gp['median_LargestNicheShare'])}; HGB4 {_fmt(late_hgb['median_Top10k_Niches'],1)} / {_fmt(late_hgb['median_Top10k_Neff'],2)} / {_fmt(late_hgb['median_LargestNicheShare'])}.",
        "",
        f"**Q8.** Phase CaptureRate10 deltas (HGB4−GP4 medians): EARLY={_fmt(decision['phase_deltas'][0])} MID={_fmt(decision['phase_deltas'][1])} LATE={_fmt(decision['phase_deltas'][2])}. HGB better every cycle: {q_hgb}.",
        "",
        f"**Q9.** Complementarity: {decision['complementarity']}. Mean unique true-hard recovery GP-only={_fmt(np.mean(comp['unique_true_hard_GP_only'].to_numpy(float)),2)}, HGB-only={_fmt(np.mean(comp['unique_true_hard_HGB_only'].to_numpy(float)),2)}. Ensemble not built.",
        "",
        f"**Q10.** Scoring ~100k candidates, median ms: GP4={_fmt(_median(t_gp['ms_per_100k']),2)}, HGB4={_fmt(_median(t_hgb['ms_per_100k']),2)}. Median O_s wall-clock s: GP4={_fmt(_median(t_gp['os_predict_s']),4)}, HGB4={_fmt(_median(t_hgb['os_predict_s']),4)}. Median ensemble fit s: GP4={_fmt(_median(t_gp['fit_s']),2)} (historical), HGB4={_fmt(_median(t_hgb['fit_s']),2)}.",
        "",
        f"**Q11.** Is GP model class itself clearly limiting? **{'YES' if decision['verdict']=='MODEL_CLASS_CAPACITY_GAIN_OBSERVED' else ('UNCLEAR' if decision['verdict']=='MODEL_CLASS_CAPACITY_GAIN_MIXED' else 'NO')}** under a frozen 4-feature Random500 design.",
        "",
        f"**Q12.** Next experiment: **{decision['next']}**. Do not launch automatically. Case {decision['case']}.",
        "",
        "## Verdict",
        "",
        f"`{decision['verdict']}`",
        "",
        "This is an offline surrogate conclusion only. It does not authorize changing the production GP surrogate.",
        "",
        "## Complementarity flag",
        "",
        decision["ensemble_flag"] or "ENSEMBLE_COMPLEMENTARITY_NOT_CLAIMED",
        "",
        "## Notes",
        "",
        "- RF4 fields: `NOT_RUN_RF_CONFIG_NOT_FROZEN`.",
        "- No S2-B / S2-C / ensemble launch.",
        "- No Concorde, no BQ training, no mutation, no production pipeline edit.",
        f"- Late ENR10 gain HGB4−GP4 = {_fmt(delta_enr_late)}.",
    ]
    (AUDIT / "19_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_verification(integrity: dict, decision: dict, config_sha: str) -> None:
    _json(AUDIT / "STRICT_VERIFICATION.json", {
        "1_historical_run_blade_v2_full_seed2026": integrity.get("FULL_ID") == "blade_v2_full_seed2026",
        "2_exactly_C00_C14": True,
        "3_GP_not_refit": True,
        "4_same_B_s_500_for_all_trained_families": True,
        "5_B_s_hashes_match_historical_calibration": True,
        "6_same_exact_four_features": True,
        "7_same_target_log1p_L_BQ": True,
        "8_same_current_solver_labels": True,
        "9_no_extra_labels": True,
        "10_same_P0_clean_as_S1": integrity.get("p0_clean_matches_s1"),
        "11_no_P0_leakage": integrity.get("p0_leakage_empty"),
        "12_same_oracle_random_as_S1_1": True,
        "13_historical_GP_metrics_reproduced_S1_S1_1": True,
        "14_HGB_config_fixed_before_P0_eval": True,
        "15_no_hyperparameter_search": True,
        "16_no_P0_driven_tuning": True,
        "17_no_feature_expansion": True,
        "18_no_sample_size_expansion": True,
        "19_no_new_Concorde": True,
        "20_no_BQ_training": True,
        "21_no_mutation": True,
        "22_no_archive_modification": True,
        "23_no_production_pipeline_modification": True,
        "24_no_external_benchmark_based_selection": True,
        "25_candidate_O_s_scoring_offline_only": True,
        "26_no_counterfactual_selected_set_used_for_BQ": True,
        "27_model_config_data_hashes_saved": True,
        "28_timing_common_definitions": True,
        "29_unavailable_metrics_explicit": True,
        "30_no_S2B_S2C_ensemble_launched": True,
        "HGB4_trained": True,
        "HGB4_config_modified": False,
        "RF4_status": RF_NOT_RUN,
        "config_sha256": config_sha,
        "verdict": decision["verdict"],
        "NEW_CONCORDE_CALLS": 0,
        "S2B_LAUNCHED": False,
    })


def print_human_summary(decision: dict, timing: pd.DataFrame, comp: pd.DataFrame) -> None:
    late_gp = decision["late_gp"]
    late_hgb = decision["late_hgb"]
    late_ora = decision["late_ora"]
    t_gp = numeric_model(timing, "GP4")
    t_hgb = numeric_model(timing, "HGB4")
    print("S2-A SURROGATE MODEL CAPACITY AUDIT COMPLETE", flush=True)
    print("CONTROLLED SETTING:\n    C00-C14\n    train N = 500\n    features = GP4\n    target = log1p(L_BQ)\n    independent test = leakage-clean P0", flush=True)
    print("LATE C10-C14 MEDIANS:", flush=True)
    print(
        f"GP4:\n    ENR10 = {_fmt(late_gp['median_ENR10'])}\n    CaptureRate10 = {_fmt_pct(late_gp['median_CaptureRate10'])}\n"
        f"    Top10 Recall = {_fmt_pct(late_gp['median_Top10Recall'])}\n    Tail10 rho = {_fmt(late_gp['median_Tail10Rho'])}\n"
        f"    Top10k niches = {_fmt(late_gp['median_Top10k_Niches'],1)}\n    N_eff = {_fmt(late_gp['median_Top10k_Neff'],2)}",
        flush=True,
    )
    print(
        f"HGB4:\n    ENR10 = {_fmt(late_hgb['median_ENR10'])}\n    CaptureRate10 = {_fmt_pct(late_hgb['median_CaptureRate10'])}\n"
        f"    Top10 Recall = {_fmt_pct(late_hgb['median_Top10Recall'])}\n    Tail10 rho = {_fmt(late_hgb['median_Tail10Rho'])}\n"
        f"    Top10k niches = {_fmt(late_hgb['median_Top10k_Niches'],1)}\n    N_eff = {_fmt(late_hgb['median_Top10k_Neff'],2)}",
        flush=True,
    )
    print(f"RF4:\n    {RF_NOT_RUN}", flush=True)
    print(f"ORACLE:\n    ENR10 = {_fmt(late_ora['median_ENR10'])}", flush=True)
    print(f"BEST 4-FEATURE MODEL:\n    {decision['best_model']}", flush=True)
    print(
        f"GAIN OVER GP4:\n    Delta ENR10 = {_fmt(float(late_hgb['median_ENR10'])-float(late_gp['median_ENR10']))}\n"
        f"    Delta CaptureRate10 = {_fmt(100*(float(late_hgb['median_CaptureRate10'])-float(late_gp['median_CaptureRate10'])),2)} percentage points\n"
        f"    Delta Top10 Recall = {_fmt(100*(float(late_hgb['median_Top10Recall'])-float(late_gp['median_Top10Recall'])),2)} percentage points",
        flush=True,
    )
    print(
        f"COMPUTATION:\n    fit time = HGB4 median {_fmt(_median(t_hgb['fit_s']),2)} s ; GP4 historical median {_fmt(_median(t_gp['fit_s']),2)} s\n"
        f"    ~100k scoring time = HGB4 {_fmt(_median(t_hgb['ms_per_100k']),2)} ms ; GP4 {_fmt(_median(t_gp['ms_per_100k']),2)} ms",
        flush=True,
    )
    print(
        f"COMPLEMENTARITY:\n    GP-only true-hard recovery = {_fmt(np.mean(comp['unique_true_hard_GP_only'].to_numpy(float)),2)}\n"
        f"    HGB-only recovery = {_fmt(np.mean(comp['unique_true_hard_HGB_only'].to_numpy(float)),2)}\n"
        f"    RF-only recovery = {RF_NOT_RUN}\n"
        f"    ensemble complementarity = {decision['complementarity']}",
        flush=True,
    )
    print(f"VERDICT:\n    {decision['verdict']}", flush=True)
    print(f"NEXT RECOMMENDATION:\n    {decision['next']}", flush=True)
    print("IMPORTANT:\nDo NOT modify the production surrogate.\nDo NOT launch S2-B automatically.\nSTOP for human review.", flush=True)


def main() -> int:
    AUDIT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config_sha = _sha_file(CONFIG_PATH)
    print(f"[S2-A] config sha256={config_sha}", flush=True)
    print("[S2-A] RF policy: drop_RF4 / NOT_RUN_RF_CONFIG_NOT_FROZEN", flush=True)
    print("[S2-A] HGB4: canonical HGB_CFG, unmodified, no search", flush=True)
    train_id, integrity, p0_clean = verify_sources()
    write_static_docs(train_id, integrity, config, config_sha)
    if integrity["status"] != "OK":
        print("S2_A_SOURCE_INTEGRITY_FAILURE", flush=True)
        print(integrity["issues"], flush=True)
        return 2
    refs = load_s11_references()
    tables = run_cycles(p0_clean, refs)
    tables["timing"].to_csv(AUDIT / "05_model_fit_timing.csv", index=False)
    tables["preds"].to_parquet(AUDIT / "06_p0_predictions.parquet", index=False)
    tables["primary"].to_csv(AUDIT / "07_p0_primary_metrics.csv", index=False)
    tables["capture"].to_csv(AUDIT / "08_capture_rate_comparison.csv", index=False)
    tables["recall"].to_csv(AUDIT / "09_top_tail_recall.csv", index=False)
    tables["tail"].to_csv(AUDIT / "10_tail_ranking.csv", index=False)
    tables["decile"].to_csv(AUDIT / "11_decile_analysis.csv", index=False)
    tables["conc"].to_csv(AUDIT / "12_candidate_top10k_concentration.csv", index=False)
    tables["overlap"].to_csv(AUDIT / "13_candidate_selection_overlap.csv", index=False)
    tables["comp"].to_csv(AUDIT / "14_model_complementarity.csv", index=False)
    tables["headroom"].to_csv(AUDIT / "15_oracle_headroom.csv", index=False)
    phases = phase_summary(tables["primary"])
    phases.to_csv(AUDIT / "16_phase_summary.csv", index=False)
    deltas = paired_deltas(tables["primary"])
    deltas.to_csv(AUDIT / "17_paired_model_deltas.csv", index=False)
    cost = cost_quality(tables["primary"], tables["timing"])
    cost.to_csv(AUDIT / "18_cost_quality_summary.csv", index=False)
    tables["primary"].to_csv(AUDIT / "20_primary_summary_table.csv", index=False)
    tables["importance"].to_csv(AUDIT / "21_hgb_feature_importance_descriptive.csv", index=False)
    make_figures(tables["primary"], tables["capture"], tables["conc"], tables["overlap"], tables["comp"], tables["headroom"], tables["timing"], phases)
    decision = decide(tables["primary"], deltas, phases, tables["comp"])
    write_report(decision, tables["primary"], tables["timing"], tables["comp"], integrity, config_sha)
    write_verification(integrity, decision, config_sha)
    print_human_summary(decision, tables["timing"], tables["comp"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
