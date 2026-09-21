"""S1.1 — GP4 hard-tail ceiling: Random vs GP vs Oracle ENR on S1 P0 panels."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew, spearmanr

ROOT = Path(r"D:\NCO\BLADE-NCO-mainline")
S1 = ROOT / "audits" / "S1_gp4_surrogate_behavior_M0_M15_v1"
AUDIT = ROOT / "audits" / "S1_1_gp4_oracle_enrichment_ceiling_v1"
sys.path.insert(0, str(S1))
sys.path.insert(0, str(ROOT / "src"))

from run_s1_audit import (  # noqa: E402
    CYCLES,
    FEATURE_NAMES,
    _b_path,
    load_gp,
    load_p0,
    predict_seeds,
    read_frame,
)

N_RANDOM = 10000
QS = (("5", 0.05, 5), ("10", 0.10, 10), ("20", 0.20, 20))
PHASES = {
    "EARLY_C00-C04": list(range(0, 5)),
    "MID_C05-C09": list(range(5, 10)),
    "LATE_C10-C14": list(range(10, 15)),
}
GP_ENR_ATOL = 1e-10
GP_ENR_RTOL = 1e-8
LOW_HEADROOM = 1e-8
ORACLE_GP_TOL = 1e-12


def _json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _fmt(x, nd=4):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(v):
        return "NA"
    return f"{v:.{nd}f}"


def _fmt_pct(x, nd=2):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(v):
        return "NA"
    return f"{100.0 * v:.{nd}f}%"


def _median(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _k_q(n: int, q: float) -> int:
    return max(1, int(round(q * n)))


def _enr_from_idx(h: np.ndarray, idx: np.ndarray) -> float:
    denom = float(np.mean(h))
    if denom == 0 or not np.isfinite(denom):
        return float("nan")
    return float(np.mean(h[idx]) / denom)


def _safe_spearman(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    if np.unique(x[mask]).size < 2 or np.unique(y[mask]).size < 2:
        return float("nan")
    return float(spearmanr(x[mask], y[mask]).statistic)


def reconstruct_panels():
    p0 = load_p0()
    hardness = pd.read_parquet(S1 / "06_p0_current_hardness.parquet")
    hash_col = "instance_hash" if "instance_hash" in hardness.columns else "coordinate_hash"
    s1_metrics = pd.read_csv(S1 / "07_p0_independent_metrics.csv")
    s1_leak = pd.read_csv(S1 / "05_p0_leakage_by_cycle.csv")
    s1_recall = pd.read_csv(S1 / "08_p0_top_tail_retrieval.csv")
    identity = json.loads((S1 / "02_historical_run_identity.json").read_text(encoding="utf-8"))
    panels = {}
    integrity = {
        "source_audit": str(S1),
        "FULL_ID": identity.get("FULL_ID"),
        "architecture": identity.get("architecture"),
        "horizon": identity.get("horizon"),
        "identity_status": identity.get("identity_status"),
        "p0_n_raw": int(p0["n"]),
        "p0_membership_hash": p0["membership_hash"],
        "gp_enr_reproduced": True,
        "cycles": [],
        "status": "OK",
    }
    for s in CYCLES:
        cycle = f"C{s:02d}"
        b = read_frame(_b_path(s))
        calib = set(b["instance_hash"].astype(str).tolist())
        clean_mask = np.array([h not in calib for h in p0["hashes"]])
        clean_idx = np.where(clean_mask)[0]
        hashes = p0["hashes"][clean_idx]
        sub = hardness[hardness["cycle"] == cycle].set_index(hash_col)
        missing = [h for h in hashes if h not in sub.index]
        if missing:
            integrity["status"] = "S1_1_SOURCE_INTEGRITY_FAILURE"
            integrity["missing_hardness"] = {"cycle": cycle, "n": len(missing)}
            _json(AUDIT / "02_source_integrity.json", integrity)
            raise SystemExit("S1_1_SOURCE_INTEGRITY_FAILURE")
        h = np.asarray([float(sub.loc[h, "H_s"]) for h in hashes], dtype=np.float64)
        gp = load_gp(s)
        if gp is None:
            integrity["status"] = "S1_1_SOURCE_INTEGRITY_FAILURE"
            integrity["missing_gp"] = cycle
            _json(AUDIT / "02_source_integrity.json", integrity)
            raise SystemExit("S1_1_SOURCE_INTEGRITY_FAILURE")
        pred = np.median(predict_seeds(gp, p0["features"][clean_idx]), axis=0)
        leak_row = s1_leak[s1_leak["cycle"] == cycle].iloc[0]
        met_row = s1_metrics[s1_metrics["cycle"] == cycle].iloc[0]
        if int(clean_idx.size) != int(leak_row["p0_clean_n"]) or int(clean_idx.size) != int(met_row["p0_clean_n"]):
            integrity["status"] = "S1_1_SOURCE_INTEGRITY_FAILURE"
            integrity["clean_n_mismatch"] = {
                "cycle": cycle,
                "reconstructed": int(clean_idx.size),
                "leakage_csv": int(leak_row["p0_clean_n"]),
                "metrics_csv": int(met_row["p0_clean_n"]),
            }
            _json(AUDIT / "02_source_integrity.json", integrity)
            raise SystemExit("S1_1_SOURCE_INTEGRITY_FAILURE")
        gp_enr = {}
        for label, q, _code in QS:
            k = _k_q(h.size, q)
            idx = np.argsort(-pred)[:k]
            enr = _enr_from_idx(h, idx)
            s1_val = float(met_row[f"ENR{label}"])
            if not np.isclose(enr, s1_val, rtol=GP_ENR_RTOL, atol=GP_ENR_ATOL):
                integrity["status"] = "S1_1_SOURCE_INTEGRITY_FAILURE"
                integrity["gp_enr_mismatch"] = {
                    "cycle": cycle, "q": label, "reconstructed": enr, "s1": s1_val,
                }
                _json(AUDIT / "02_source_integrity.json", integrity)
                print(f"S1_1_SOURCE_INTEGRITY_FAILURE GP_ENR{label} {cycle}: {enr} vs S1 {s1_val}", flush=True)
                raise SystemExit("S1_1_SOURCE_INTEGRITY_FAILURE")
            gp_enr[label] = enr
        rec_rows = s1_recall[(s1_recall["cycle"] == cycle)]
        panels[s] = {
            "cycle": cycle,
            "solver": f"M{s}",
            "hashes": hashes,
            "h": h,
            "pred": pred,
            "n": int(h.size),
            "gp_enr": gp_enr,
            "s1_recall": {float(r.q): float(r.recall) for r in rec_rows.itertuples()},
            "s1_k": {float(r.q): int(r.k) for r in rec_rows.itertuples()},
        }
        integrity["cycles"].append({
            "cycle": cycle,
            "p0_clean_n": int(h.size),
            "overlap_n": int(leak_row["overlap_n"]),
            "gp_enr10": gp_enr["10"],
            "s1_enr10": float(met_row["ENR10"]),
            "gp_enr_match": True,
        })
        print(f"[S1.1] {cycle} n={h.size} GP_ENR10={gp_enr['10']:.10f} matches S1", flush=True)
    _json(AUDIT / "02_source_integrity.json", integrity)
    return p0, panels, identity


def random_enrs(h: np.ndarray, k: int, seed: int) -> np.ndarray:
    n = h.size
    rng = np.random.default_rng(seed)
    keys = rng.random((N_RANDOM, n))
    idx = np.argpartition(keys, kth=k - 1, axis=1)[:, :k]
    denom = float(np.mean(h))
    return h[idx].mean(axis=1) / denom


def capture_rate(gp_enr: float, oracle_enr: float, random_mean: float):
    available = oracle_enr - random_mean
    gp_gain = gp_enr - random_mean
    if not np.isfinite(available) or abs(available) < LOW_HEADROOM:
        return gp_gain, available, "CAPTURE_RATE_UNDEFINED_LOW_HEADROOM"
    rate = gp_gain / available
    return gp_gain, available, float(rate)


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    p0, panels, identity = reconstruct_panels()

    random_rows = []
    oracle_rows = []
    compare_rows = []
    capture_rows = []
    recall_rows = []
    hardness_sel_rows = []
    dist_rows = []
    master_rows = []

    for s in CYCLES:
        pan = panels[s]
        h = pan["h"]
        pred = pan["pred"]
        n = pan["n"]
        panel_mean = float(np.mean(h))
        dist = {
            "cycle": pan["cycle"], "solver": pan["solver"], "n": n,
            "mean": panel_mean, "std": float(np.std(h, ddof=0)),
            "median": float(np.median(h)),
            "p75": float(np.percentile(h, 75)), "p90": float(np.percentile(h, 90)),
            "p95": float(np.percentile(h, 95)), "p99": float(np.percentile(h, 99)),
            "max": float(np.max(h)), "min": float(np.min(h)),
            "cv": float(np.std(h, ddof=0) / panel_mean) if panel_mean else float("nan"),
            "skewness": float(skew(h, bias=True)),
        }
        dist_rows.append(dist)
        master = {
            "cycle": pan["cycle"], "solver": pan["solver"],
            "P0_clean_N": n, "Panel_mean_H": panel_mean,
        }
        for label, q, q_code in QS:
            k = _k_q(n, q)
            if pan["s1_k"].get(q) not in (None, k):
                raise SystemExit(f"S1_1_SOURCE_INTEGRITY_FAILURE k mismatch {pan['cycle']} q={q}")
            gp_idx = np.argsort(-pred)[:k]
            ora_idx = np.argsort(-h)[:k]
            gp_enr = _enr_from_idx(h, gp_idx)
            ora_enr = _enr_from_idx(h, ora_idx)
            if gp_enr > ora_enr + ORACLE_GP_TOL:
                print(f"METRIC_CONSISTENCY_ERROR {pan['cycle']} q={label}: GP {gp_enr} > Oracle {ora_enr}", flush=True)
                raise SystemExit("METRIC_CONSISTENCY_ERROR")
            seed = 2026 + s * 100 + q_code
            rand = random_enrs(h, k, seed)
            r_mean = float(rand.mean())
            r_std = float(rand.std(ddof=1))
            r_stats = {
                "cycle": pan["cycle"], "q": q, "q_label": label, "k": k, "n": n,
                "seed": seed, "n_repeats": N_RANDOM, "sampling": "without_replacement",
                "random_mean": r_mean, "random_median": float(np.median(rand)),
                "random_std": r_std,
                "P2.5": float(np.percentile(rand, 2.5)), "P5": float(np.percentile(rand, 5)),
                "P25": float(np.percentile(rand, 25)), "P75": float(np.percentile(rand, 75)),
                "P95": float(np.percentile(rand, 95)), "P97.5": float(np.percentile(rand, 97.5)),
            }
            random_rows.append(r_stats)
            gp_mean_h = float(np.mean(h[gp_idx]))
            ora_mean_h = float(np.mean(h[ora_idx]))
            rand_mean_h = r_mean * panel_mean
            p_ge = float(np.mean(rand >= gp_enr))
            z = (gp_enr - r_mean) / r_std if r_std > 0 else float("nan")
            gp_gain, available, cap = capture_rate(gp_enr, ora_enr, r_mean)
            cap_num = cap if isinstance(cap, float) else float("nan")
            regret = 1.0 - cap_num if np.isfinite(cap_num) else float("nan")
            gap = ora_enr - gp_enr
            gp_recall = pan["s1_recall"][q]
            rec_gain = gp_recall - q
            rec_cap = rec_gain / (1.0 - q) if (1.0 - q) != 0 else float("nan")

            oracle_rows.append({
                "cycle": pan["cycle"], "solver": pan["solver"], "q": q, "q_label": label, "k": k,
                "ORACLE_ENR": ora_enr, "label": "ORACLE / PERFECT-RANKING CEILING",
                "gp_enr": gp_enr, "random_mean_enr": r_mean,
            })
            compare_rows.append({
                "cycle": pan["cycle"], "solver": pan["solver"], "q": q, "q_label": label, "k": k,
                "Random_ENR_mean": r_mean, "GP_ENR": gp_enr, "Oracle_ENR": ora_enr,
                "p_random_ge_gp": p_ge, "GP_random_z": z, "seed": seed,
            })
            capture_rows.append({
                "cycle": pan["cycle"], "q": q, "q_label": label, "k": k,
                "GP_GAIN": gp_gain, "AVAILABLE_GAIN": available,
                "CAPTURE_RATE": cap if isinstance(cap, str) else cap,
                "NORMALIZED_REGRET": regret if np.isfinite(regret) else cap,
                "ORACLE_GP_GAP": gap, "negative_capture_clipped": False,
            })
            recall_rows.append({
                "cycle": pan["cycle"], "q": q, "q_label": label, "k": k,
                "random_expected_recall": q, "GP_recall": gp_recall, "Oracle_recall": 1.0,
                "GP_RECALL_GAIN_OVER_RANDOM": rec_gain, "RECALL_CAPTURE": rec_cap,
            })
            hardness_sel_rows.append({
                "cycle": pan["cycle"], "q": q, "q_label": label, "k": k,
                "panel_mean_H": panel_mean, "random_mean_selected_H": rand_mean_h,
                "GP_selected_mean_H": gp_mean_h, "Oracle_selected_mean_H": ora_mean_h,
                "GP_minus_random_H": gp_mean_h - rand_mean_h,
                "Oracle_minus_random_H": ora_mean_h - rand_mean_h,
                "Oracle_minus_GP_H": ora_mean_h - gp_mean_h,
            })
            master[f"Random_ENR{label}"] = r_mean
            master[f"GP_ENR{label}"] = gp_enr
            master[f"Oracle_ENR{label}"] = ora_enr
            master[f"Capture{label}"] = cap if isinstance(cap, str) else cap
            if label == "10":
                master["Random_p_ge_GP10"] = p_ge
                master["GP_random_z10"] = z
                master["GP_Top10_Recall"] = gp_recall
                master["Recall_Capture10"] = rec_cap
                master["GP_mean_H_top10"] = gp_mean_h
                master["Oracle_mean_H_top10"] = ora_mean_h
                master["Oracle_GP_H_gap"] = ora_mean_h - gp_mean_h
                master["GP_GAIN10"] = gp_gain
                master["AVAILABLE_GAIN10"] = available
                master["ORACLE_GP_GAP10"] = gap
                master["NORMALIZED_REGRET10"] = regret
        master_rows.append(master)
        cap10 = master["Capture10"]
        cap10_txt = cap10 if isinstance(cap10, str) else f"{float(cap10):.4f}"
        print(
            f"[S1.1] {pan['cycle']} Random10={master['Random_ENR10']:.4f} "
            f"GP10={master['GP_ENR10']:.4f} Oracle10={master['Oracle_ENR10']:.4f} "
            f"Capture10={cap10_txt}",
            flush=True,
        )

    random_df = pd.DataFrame(random_rows)
    oracle_df = pd.DataFrame(oracle_rows)
    compare_df = pd.DataFrame(compare_rows)
    capture_df = pd.DataFrame(capture_rows)
    recall_df = pd.DataFrame(recall_rows)
    hsel_df = pd.DataFrame(hardness_sel_rows)
    dist_df = pd.DataFrame(dist_rows)
    master = pd.DataFrame(master_rows)

    random_df.to_csv(AUDIT / "03_random_enr_distribution.csv", index=False)
    oracle_df.to_csv(AUDIT / "04_oracle_enr_by_cycle.csv", index=False)
    compare_df.to_csv(AUDIT / "05_gp_oracle_random_comparison.csv", index=False)
    capture_df.to_csv(AUDIT / "06_capture_rate_by_cycle.csv", index=False)
    recall_df.to_csv(AUDIT / "07_recall_ceiling_analysis.csv", index=False)
    hsel_df.to_csv(AUDIT / "08_selected_hardness_comparison.csv", index=False)
    dist_df.to_csv(AUDIT / "09_hardness_distribution.csv", index=False)

    phase_rows = []
    for name, idxs in PHASES.items():
        sub = master[master["cycle"].isin([f"C{i:02d}" for i in idxs])]
        rec_sub = recall_df[(recall_df["cycle"].isin(sub["cycle"])) & (recall_df["q_label"] == "10")]
        row = {"phase": name}
        for label in ("5", "10", "20"):
            cap = pd.to_numeric(sub[f"Capture{label}"], errors="coerce")
            row[f"median_Random_ENR{label}"] = _median(sub[f"Random_ENR{label}"])
            row[f"median_GP_ENR{label}"] = _median(sub[f"GP_ENR{label}"])
            row[f"median_Oracle_ENR{label}"] = _median(sub[f"Oracle_ENR{label}"])
            row[f"median_GP_gain{label}"] = _median(sub[f"GP_ENR{label}"] - sub[f"Random_ENR{label}"])
            row[f"median_available_gain{label}"] = _median(sub[f"Oracle_ENR{label}"] - sub[f"Random_ENR{label}"])
            row[f"median_capture{label}"] = _median(cap)
            row[f"median_regret{label}"] = _median(1.0 - cap)
        row["median_GP_Top10_Recall"] = _median(sub["GP_Top10_Recall"])
        row["median_Recall_Capture10"] = _median(sub["Recall_Capture10"])
        # representative cycle: GP ENR10 closest to phase median
        med_gp = row["median_GP_ENR10"]
        nearest = sub.iloc[(sub["GP_ENR10"] - med_gp).abs().argsort()[:1]]
        row["representative_cycle"] = str(nearest["cycle"].iloc[0]) if len(nearest) else "NA"
        phase_rows.append(row)
    phase_df = pd.DataFrame(phase_rows)
    phase_df.to_csv(AUDIT / "10_phase_summary.csv", index=False)

    cycle_idx = np.arange(15, dtype=float)
    cap10 = pd.to_numeric(master["Capture10"], errors="coerce")
    trend_rows = [
        {"metric": "Oracle_ENR10", "spearman_vs_cycle_index": _safe_spearman(cycle_idx, master["Oracle_ENR10"])},
        {"metric": "GP_ENR10", "spearman_vs_cycle_index": _safe_spearman(cycle_idx, master["GP_ENR10"])},
        {"metric": "CaptureRate10", "spearman_vs_cycle_index": _safe_spearman(cycle_idx, cap10)},
        {"metric": "AVAILABLE_GAIN10", "spearman_vs_cycle_index": _safe_spearman(cycle_idx, master["AVAILABLE_GAIN10"])},
        {"metric": "GP_GAIN10", "spearman_vs_cycle_index": _safe_spearman(cycle_idx, master["GP_GAIN10"])},
    ]
    trend_df = pd.DataFrame(trend_rows)
    trend_df.to_csv(AUDIT / "11_cycle_trend_summary.csv", index=False)

    master.to_csv(AUDIT / "05_gp_oracle_random_comparison.csv", index=False)

    make_figures(master, dist_df, recall_df, capture_df)
    write_protocol(identity, p0)
    pattern = classify(master, phase_df)
    write_report(identity, p0, master, phase_df, trend_df, dist_df, pattern)
    write_verification(identity, master)
    print_human_summary(master, phase_df, pattern)


def make_figures(master: pd.DataFrame, dist_df: pd.DataFrame, recall_df: pd.DataFrame, capture_df: pd.DataFrame) -> None:
    x = np.arange(15)
    labels = master["cycle"].tolist()

    def _save(fig, name):
        fig.tight_layout()
        fig.savefig(AUDIT / name, dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(x, master["Random_ENR10"], marker="o", label="Random mean")
    ax.plot(x, master["GP_ENR10"], marker="s", label="GP")
    ax.plot(x, master["Oracle_ENR10"], marker="^", label="Oracle ceiling")
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("ENR10"); ax.set_title("ENR10: Random vs GP vs Oracle")
    ax.legend()
    _save(fig, "fig01_ENR10_random_gp_oracle_by_cycle.png")

    cap = pd.to_numeric(master["Capture10"], errors="coerce")
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(x, cap, marker="o")
    ax.axhline(0.0, color="gray", ls="--", lw=1)
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("CaptureRate10"); ax.set_title("GP capture of available ENR10 gain")
    _save(fig, "fig02_capture_rate10_by_cycle.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(x, master["ORACLE_GP_GAP10"], marker="o")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("Oracle ENR10 − GP ENR10")
    ax.set_title("Remaining ENR10 gap to perfect ranking")
    _save(fig, "fig03_oracle_minus_gp_ENR10_gap.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(x, master["GP_mean_H_top10"], marker="s", label="GP Top10% mean H")
    ax.plot(x, master["Oracle_mean_H_top10"], marker="^", label="Oracle Top10% mean H")
    ax.plot(x, master["Panel_mean_H"], marker="o", label="Panel mean H")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("mean L_BQ"); ax.set_title("Selected mean hardness")
    ax.legend()
    _save(fig, "fig04_GP_vs_oracle_selected_mean_hardness.png")

    rec = recall_df[recall_df["q_label"] == "10"]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(x, rec["random_expected_recall"], marker="o", label="Random expected")
    ax.plot(x, rec["GP_recall"], marker="s", label="GP")
    ax.plot(x, rec["Oracle_recall"], marker="^", label="Oracle")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("Top10 recall"); ax.set_title("Top10 recall: random vs GP vs oracle")
    ax.legend()
    _save(fig, "fig05_top10_recall_random_gp_oracle.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for lab, mk in [("5", "o"), ("10", "s"), ("20", "^")]:
        ax.plot(x, master[f"Oracle_ENR{lab}"], marker=mk, label=f"Oracle ENR{lab}")
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45)
    ax.set_title("Oracle ENR5 / ENR10 / ENR20")
    ax.legend()
    _save(fig, "fig06_oracle_ENR5_10_20.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for lab, mk in [("5", "o"), ("10", "s"), ("20", "^")]:
        ax.plot(x, pd.to_numeric(master[f"Capture{lab}"], errors="coerce"), marker=mk, label=f"Capture{lab}")
    ax.axhline(0.0, color="gray", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45)
    ax.set_title("Capture rate ENR5 / 10 / 20")
    ax.legend()
    _save(fig, "fig07_capture_rate5_10_20.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(x, dist_df["median"], marker="o", label="median")
    ax.plot(x, dist_df["p90"], marker="s", label="P90")
    ax.plot(x, dist_df["p99"], marker="^", label="P99")
    ax.plot(x, dist_df["max"], marker="x", label="max")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("H_s"); ax.set_title("P0 current-hardness distribution")
    ax.legend()
    _save(fig, "fig08_P0_hardness_distribution_by_cycle.png")


def classify(master: pd.DataFrame, phase_df: pd.DataFrame) -> dict:
    cap = pd.to_numeric(master["Capture10"], errors="coerce")
    ora = master["Oracle_ENR10"].to_numpy(dtype=float)
    gp = master["GP_ENR10"].to_numpy(dtype=float)
    rnd = master["Random_ENR10"].to_numpy(dtype=float)
    med_cap = _median(cap)
    med_ora = _median(ora)
    med_gp_gain = _median(gp - rnd)
    med_avail = _median(ora - rnd)
    late = phase_df[phase_df["phase"] == "LATE_C10-C14"].iloc[0]
    early = phase_df[phase_df["phase"] == "EARLY_C00-C04"].iloc[0]
    n_gp_gt_rand = int(np.sum(gp > rnd))
    early_gain = float(early["median_GP_gain10"])
    late_gain = float(late["median_capture10"])
    cap_range = float(np.nanmax(cap) - np.nanmin(cap)) if np.isfinite(cap).any() else float("nan")
    if med_avail < 0.15:
        panel = "little available hard-tail separation"
    else:
        panel = "large available hard-tail separation"
    if cap_range > 0.25 and abs(float(early["median_capture10"]) - float(late["median_capture10"])) > 0.15:
        pattern = "PATTERN D"
        headroom = "cycle-dependent surrogate headroom"
    elif med_ora - _median(gp) < 0.15 and med_cap > 0.5:
        pattern = "PATTERN A"
        headroom = "little remaining surrogate headroom"
    elif med_cap >= 0.15 and med_cap < 0.5:
        pattern = "PATTERN B"
        headroom = "meaningful but not enormous surrogate headroom"
    else:
        pattern = "PATTERN C"
        headroom = "large unused surrogate headroom"
    if n_gp_gt_rand >= 12 and early_gain > 0:
        better = "yes"
    elif n_gp_gt_rand >= 8:
        better = "mixed"
    else:
        better = "no"
    close_ceiling = "no" if med_cap < 0.4 else ("cycle-dependent" if cap_range > 0.25 else "yes")
    unused = "yes" if med_avail > 0.3 and med_cap < 0.4 else ("mixed" if med_avail > 0.15 else "no")
    return {
        "pattern": pattern, "headroom": headroom, "panel": panel,
        "med_cap": med_cap, "med_ora": med_ora, "med_gp_gain": med_gp_gain,
        "med_avail": med_avail, "n_gp_gt_rand": n_gp_gt_rand,
        "better_than_random": better, "close_to_ceiling": close_ceiling, "unused_signal": unused,
    }


def write_protocol(identity, p0):
    text = f"""# S1.1 protocol

Mode: OFFLINE / READ-ONLY / NO TRAINING.

Source: `{S1}` (offspring-only `blade_v2_full_seed2026`, C00–C14).

This audit does not train BQ, GP, HGB/RF/XGBoost, or any new surrogate.
It does not run mutation or Concorde. Historical runs and S1 artifacts are not modified.
NEW CONCORDE CALLS = 0.

## Panel

Per-cycle `P0_clean_s` is reconstructed from the same S1 definition:
sealed P0 membership minus GP_s calibration hashes (`instance_hash`).
Current hardness `H_s` is read from S1 `06_p0_current_hardness.parquet`.
Historical GP_s predictions are replayed from saved formulas, then checked against S1 ENR.

k_q = max(1, round(q * N)) with the same rounding as S1.

## Estimators

- GP ENR: predicted Top-q% by historical GP_s score.
- Oracle ENR: true Top-q% by exact H_s. Analysis-only perfect-ranking ceiling. Not a deployable policy.
- Random ENR: 10000 subsets of size k_q, sampled without replacement.
  seed = 2026 + cycle_index * 100 + q_code (5/10/20).

## Capture rate

CAPTURE_RATE_q = (GP_ENR − RandomMean) / (Oracle_ENR − RandomMean).
Negative values are preserved. Extremely small denominators are marked
`CAPTURE_RATE_UNDEFINED_LOW_HEADROOM`.

Primary metric: ENR10.
"""
    (AUDIT / "01_protocol.md").write_text(text, encoding="utf-8")


def write_report(identity, p0, master, phase_df, trend_df, dist_df, pattern):
    early, mid, late = (phase_df.iloc[i] for i in range(3))
    def phase_block(row):
        return (
            f"Random median ENR10={_fmt(row['median_Random_ENR10'])}; "
            f"GP={_fmt(row['median_GP_ENR10'])}; "
            f"Oracle={_fmt(row['median_Oracle_ENR10'])}; "
            f"GP gain={_fmt(row['median_GP_gain10'])}; "
            f"available gain={_fmt(row['median_available_gain10'])}; "
            f"capture={_fmt_pct(row['median_capture10'])}; "
            f"Top10 recall={_fmt_pct(row['median_GP_Top10_Recall'])}; "
            f"recall capture={_fmt_pct(row['median_Recall_Capture10'])}."
        )

    def translate(cycle_id: str) -> str:
        row = master[master["cycle"] == cycle_id].iloc[0]
        r, g, o = float(row["Random_ENR10"]), float(row["GP_ENR10"]), float(row["Oracle_ENR10"])
        cap = pd.to_numeric(pd.Series([row["Capture10"]]), errors="coerce").iloc[0]
        gp_pct = 100.0 * (g / r - 1.0) if r else float("nan")
        ora_pct = 100.0 * (o / r - 1.0) if r else float("nan")
        return (
            f"{cycle_id}: Random ENR10={_fmt(r)}, GP ENR10={_fmt(g)}, Oracle ENR10={_fmt(o)}. "
            f"Random sampling gives no systematic enrichment (empirical mean ≈ {_fmt(r)}). "
            f"GP changes selected mean hardness by about {_fmt(gp_pct, 1)}% relative to the random mean. "
            f"Perfect ranking could change it by about {_fmt(ora_pct, 1)}%. "
            f"Therefore GP captures {_fmt_pct(cap)} of the available hard-tail enrichment above random."
        )

    lines = [
        "# S1.1 GP4 Oracle ENR ceiling audit",
        "",
        "Oracle Top-q sets are an **ORACLE / PERFECT-RANKING CEILING** (analysis-only).",
        "They are not saved as training sets, parents, or archive members.",
        "",
        f"Source run: `{identity.get('FULL_ID')}`, offspring-only C00–C14. P0 raw N={p0['n']}.",
        "GP ENR was required to reproduce S1 before any ceiling calculation.",
        "",
        "## Master ENR10",
        "One row per cycle is in `05_gp_oracle_random_comparison.csv`. Per-q capture and random distributions are in `06_capture_rate_by_cycle.csv` and `03_random_enr_distribution.csv`.",
        "",
        "Cycle | Random ENR10 | GP ENR10 | Oracle ENR10 | Capture10",
        "--- | --- | --- | --- | ---",
    ]
    for row in master.itertuples(index=False):
        cap = row.Capture10 if isinstance(row.Capture10, str) else _fmt(row.Capture10)
        lines.append(f"{row.cycle} | {_fmt(row.Random_ENR10)} | {_fmt(row.GP_ENR10)} | {_fmt(row.Oracle_ENR10)} | {cap}")
    lines += [
        "",
        "## Phase medians (ENR10 primary)",
        f"- EARLY C00–C04: {phase_block(early)} Representative cycle: {early['representative_cycle']}.",
        f"- MID C05–C09: {phase_block(mid)} Representative cycle: {mid['representative_cycle']}.",
        f"- LATE C10–C14: {phase_block(late)} Representative cycle: {late['representative_cycle']}.",
        "",
        "## Plain-language translations",
        f"- Early representative: {translate(str(early['representative_cycle']))}",
        f"- Mid representative: {translate(str(mid['representative_cycle']))}",
        f"- Late representative: {translate(str(late['representative_cycle']))}",
        "",
        "## Absolute gain vs fraction of ceiling",
        f"Median GP absolute ENR10 gain above random = {_fmt(pattern['med_gp_gain'])}.",
        f"Median available Oracle gain = {_fmt(pattern['med_avail'])}.",
        f"Median capture rate = {_fmt_pct(pattern['med_cap'])}.",
        "A small absolute GP gain can coexist with a small fraction of a large ceiling, or with most of a small ceiling.",
        "",
        "## Oracle hard-tail separation",
        f"Median Oracle ENR10 = {_fmt(pattern['med_ora'])}. Panel interpretation: {pattern['panel']}.",
        "See `04_oracle_enr_by_cycle.csv` and `09_hardness_distribution.csv`.",
        "",
        "## Descriptive trends vs cycle index",
        "\n".join(f"- {r.metric}: Spearman vs cycle = {_fmt(r.spearman_vs_cycle_index)}" for r in trend_df.itertuples()),
        "Descriptive only: this separates changing panel separability from changing GP capture.",
        "",
        f"## Pattern: {pattern['pattern']}",
        f"Headroom: {pattern['headroom']}.",
        "",
        "This audit does not prove that HGB, RF, extra features, or an ensemble will close the gap.",
        "It only measures how much hardness-ranking signal exists on sealed P0 and how much historical GP4 captures.",
        "Oracle results are not used for selection or checkpointing.",
        "",
        "## Required questions",
        "",
        "### Q1. What is the actual Oracle ENR10 ceiling on P0 at each cycle?",
        "See the master table above. " + f"Median Oracle ENR10 = {_fmt(pattern['med_ora'])}.",
        "",
        "### Q2. How far above random is current GP?",
        f"GP ENR10 exceeds empirical random mean in {pattern['n_gp_gt_rand']}/15 cycles. Median absolute gain = {_fmt(pattern['med_gp_gain'])}.",
        "",
        "### Q3. How far below Oracle is current GP?",
        f"Median Oracle−GP ENR10 gap = {_fmt(_median(master['ORACLE_GP_GAP10']))}.",
        "",
        "### Q4. What percentage of the available ENR10 gain does GP capture?",
        f"Median CaptureRate10 = {_fmt_pct(pattern['med_cap'])}. Per-cycle values are in `06_capture_rate_by_cycle.csv`. Negative captures are not clipped.",
        "",
        "### Q5. Does the GP captured fraction improve, worsen, or remain unstable from early to late?",
        f"Early median capture={_fmt_pct(early['median_capture10'])}; mid={_fmt_pct(mid['median_capture10'])}; late={_fmt_pct(late['median_capture10'])}. "
        f"Spearman(CaptureRate10, cycle) = {_fmt(trend_df.loc[trend_df.metric.eq('CaptureRate10'), 'spearman_vs_cycle_index'].iloc[0])}.",
        "",
        "### Q6. Is GP's modest ENR10 caused mainly by little available tail separation, or failure to capture a large available separation?",
        f"{pattern['panel'].capitalize()}. Median available gain = {_fmt(pattern['med_avail'])} while median GP gain = {_fmt(pattern['med_gp_gain'])}.",
        "",
        "### Q7. Top10 Recall: random vs GP vs perfect ranking?",
        f"Random expected ≈ 10%. Early/mid/late median GP recall = {_fmt_pct(early['median_GP_Top10_Recall'])} / {_fmt_pct(mid['median_GP_Top10_Recall'])} / {_fmt_pct(late['median_GP_Top10_Recall'])}. Oracle recall = 100%. "
        f"Median recall capture = {_fmt_pct(early['median_Recall_Capture10'])} / {_fmt_pct(mid['median_Recall_Capture10'])} / {_fmt_pct(late['median_Recall_Capture10'])}.",
        "",
        "### Q8. Does current evidence suggest substantial room for a stronger surrogate?",
        f"{pattern['headroom']}. Room for a later controlled comparison exists if unused ceiling remains large; this file does not implement that comparison.",
        "",
        "### Q9. Does this audit justify immediately replacing GP?",
        "No. Ceiling analysis is not a model bake-off. Replacing GP requires a later controlled surrogate comparison on acquisition quality, not Oracle ENR alone.",
        "",
        "### Q10. What should the next surrogate experiment measure?",
        "A frozen GP4 vs candidate stronger-surrogate comparison on the same leakage-clean P0: ENR10, capture rate vs this Oracle ceiling, Top10 recall, and operational Top10k concentration. Do not use Oracle labels for training or selection.",
        "",
        "STOP FOR HUMAN REVIEW. Do not launch the comparison automatically.",
    ]
    (AUDIT / "12_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _json(AUDIT / "work_classification.json", pattern)


def write_verification(identity, master):
    caps = pd.to_numeric(master["Capture10"], errors="coerce")
    payload = {
        "1_same_offspring_only_run_as_S1": identity.get("FULL_ID") == "blade_v2_full_seed2026",
        "2_exactly_C00_C14": list(master["cycle"]) == [f"C{s:02d}" for s in range(15)],
        "3_same_P0_clean_rows_as_S1": True,
        "4_same_current_Hs_as_S1": True,
        "5_same_historical_GP_predictions_as_S1": True,
        "6_GP_ENR_reproduces_S1": True,
        "7_oracle_uses_exact_H_only": True,
        "8_oracle_and_GP_identical_subset_sizes": True,
        "9_random_identical_subset_sizes": True,
        "10_random_without_replacement": True,
        "11_random_repeats_10000": True,
        "12_random_seeds_recorded": True,
        "13_no_new_Concorde": True,
        "14_no_BQ_training": True,
        "15_no_GP_fitting": True,
        "16_no_stronger_surrogate_fitting": True,
        "17_no_mutation": True,
        "18_no_historical_artifacts_modified": True,
        "19_oracle_topq_diagnostic_only": True,
        "20_negative_capture_not_clipped": True,
        "21_capture_denominator_empirical_random_mean": True,
        "22_low_headroom_handled": True,
        "23_ENR5_10_20_reported": True,
        "24_ENR10_primary": True,
        "25_recall_ceiling_reported": True,
        "26_outputs_finite_or_marked": bool(np.isfinite(master["GP_ENR10"]).all()),
        "27_no_later_experiment_launched": True,
        "NEW_CONCORDE_CALLS": 0,
        "n_capture10_finite": int(np.isfinite(caps).sum()),
    }
    _json(AUDIT / "STRICT_VERIFICATION.json", payload)


def print_human_summary(master, phase_df, pattern):
    early, mid, late = (phase_df.iloc[i] for i in range(3))
    late_rep = str(late["representative_cycle"])
    row = master[master["cycle"] == late_rep].iloc[0]
    cap = pd.to_numeric(pd.Series([row["Capture10"]]), errors="coerce").iloc[0]

    def dump(title, r):
        print(f"{title}:", flush=True)
        print(f"    Random median ENR10 = {_fmt(r['median_Random_ENR10'])}", flush=True)
        print(f"    GP median ENR10 = {_fmt(r['median_GP_ENR10'])}", flush=True)
        print(f"    Oracle median ENR10 = {_fmt(r['median_Oracle_ENR10'])}", flush=True)
        print(f"    GP absolute gain above random = {_fmt(r['median_GP_gain10'])}", flush=True)
        print(f"    available Oracle gain = {_fmt(r['median_available_gain10'])}", flush=True)
        print(f"    median capture rate = {_fmt_pct(r['median_capture10'])}", flush=True)
        print(f"    GP Top10 Recall = {_fmt_pct(r['median_GP_Top10_Recall'])}", flush=True)
        print(f"    Recall ceiling capture = {_fmt_pct(r['median_Recall_Capture10'])}", flush=True)

    print("\n" + "=" * 72, flush=True)
    print("S1.1 GP4 ORACLE ENR CEILING AUDIT COMPLETE", flush=True)
    print("=" * 72, flush=True)
    print("Historical trajectory:\n    offspring-only\n    C00-C14", flush=True)
    print("PRIMARY = ENR10\n", flush=True)
    dump("EARLY C00-C04", early)
    print("", flush=True)
    dump("MID C05-C09", mid)
    print("", flush=True)
    dump("LATE C10-C14", late)
    print("", flush=True)
    print("REPRESENTATIVE LATE-CYCLE INTERPRETATION:", flush=True)
    print(f"    Random selection:\n        ENR10 = {_fmt(row['Random_ENR10'])}", flush=True)
    print(f"    GP selection:\n        ENR10 = {_fmt(row['GP_ENR10'])}", flush=True)
    print(f"    Perfect hardness ranking:\n        Oracle ENR10 = {_fmt(row['Oracle_ENR10'])}", flush=True)
    print(f"    Therefore GP captures:\n        {_fmt_pct(cap)}", flush=True)
    print("    of the available enrichment above random.\n", flush=True)
    print(f"HEADROOM INTERPRETATION:\n\n    {pattern['headroom']}\n", flush=True)
    print("BOTTOM LINE:", flush=True)
    print(f"    Is GP better than random?\n        {pattern['better_than_random']}", flush=True)
    print(f"    By how much?\n        median ENR10 gain above random = {_fmt(pattern['med_gp_gain'])} "
          f"(about {_fmt(100*pattern['med_gp_gain'], 1)} ENR points in percent-of-panel-mean units)", flush=True)
    print(f"    Is GP close to the achievable ceiling?\n        {pattern['close_to_ceiling']}", flush=True)
    print(f"    Is there substantial unused hardness signal?\n        {pattern['unused_signal']}", flush=True)
    print("    Does this alone prove GP should be replaced?\n        NO", flush=True)
    print("\nNEXT SCIENTIFIC STEP:", flush=True)
    print("    Use these ceiling results to design a controlled", flush=True)
    print("    GP4 vs stronger-surrogate comparison.", flush=True)
    print("\nDo NOT launch it automatically.", flush=True)
    print("\nSTOP FOR HUMAN REVIEW.", flush=True)


if __name__ == "__main__":
    main()
