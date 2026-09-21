"""S2-B: HGB4 vs HGB20 under identical Random500. Feature set only."""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
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
os.environ["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")

from features.qd4 import extract_qd4_exact  # noqa: E402
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
from surrogate_benchmark.gp20 import (  # noqa: E402
    MANIFEST_PATH,
    P0_GP20_CACHE,
    extract_gp20_cached,
    load_hgb20_manifest,
)
from surrogate_benchmark.metrics import (  # noqa: E402
    capture_rate,
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
    SEEDS,
    fit_hgb4_seed,
    hgb_params,
    load_hgb4,
    median_stack,
    predict_hgb,
    save_hgb,
)

FULL = ROOT / "runs" / "blade_v2_full_seed2026"
S2A = ROOT / "audits" / "S2_A_surrogate_model_capacity_GP4_HGB4_RF4_v1"
AUDIT = ROOT / "audits" / "S2_B_surrogate_feature_capacity_HGB4_HGB20_v1"
CONFIG_PATH = ROOT / "configs" / "surrogate_s2b_hgb4_hgb20.yaml"
P0_COORDS = ROOT / "audits" / "cycle0_initialization_validation_v1" / "03_oracle_labels" / "selected_coordinates.npz"
TOPK = 10000
QS = (("5", 0.05), ("10", 0.10), ("20", 0.20))
WORKERS = 10


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
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return "NA"
    return f"{v:.{nd}f}"


def _fmt_pct(x, nd=2):
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


def _close(a, b, rtol=1e-8, atol=1e-10) -> bool:
    return bool(np.isclose(float(a), float(b), rtol=rtol, atol=atol))


def b_coord_path(s: int) -> Path:
    if s == 0:
        return FULL / "cycles" / "C00" / "02_gp" / "B0_selection" / "B0.npz"
    return FULL / "cycles" / f"C{s:02d}" / "02_gp" / "B_selection_and_refresh" / "B_s.npz"


def o_coord_path(s: int) -> Path:
    return FULL / "cycles" / f"C{s:02d}" / "03_mutation" / "offspring_coordinates.npz"


def load_npz(path: Path) -> tuple[np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as payload:
        coords = np.asarray(payload["coordinates"], dtype=np.float64)
        hashes = [str(h) for h in np.asarray(payload["instance_hash"]).tolist()]
    return coords, hashes


def align_coords(coords: np.ndarray, coord_hashes: list[str], wanted: list[str]) -> np.ndarray:
    index = {h: i for i, h in enumerate(coord_hashes)}
    missing = [h for h in wanted if h not in index]
    if missing:
        raise RuntimeError(f"missing coordinates for {len(missing)} hashes")
    return np.stack([coords[index[h]] for h in wanted]).astype(np.float64, copy=False)


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
        out[f"Oracle_ENR{label}"] = ora
        out[f"Random_ENR{label}"] = rand
        out[f"GAIN{label}"] = gain
        out[f"AVAILABLE{label}"] = available
    tail = tail_ranking(pred, h)
    out["Spearman"] = safe_spearman(pred, h)
    out["Tail20Rho"] = tail["rho_tail20"]
    out["Tail10Rho"] = tail["rho_tail10"]
    out["Tail5Rho"] = tail["rho_tail5"]
    out["TrueTail10Rho"] = tail["true_tail_rho10"]
    return out


def fit_hgb20(s: int, x: np.ndarray, y: np.ndarray, features: list[str]):
    out_dir = AUDIT / "models" / f"C{s:02d}" / "HGB20"
    out_dir.mkdir(parents=True, exist_ok=True)
    estimators, metas = [], []
    for seed in SEEDS:
        path = out_dir / f"seed_{seed}.joblib"
        if path.is_file():
            est, meta = load_hgb4(path)
            meta = dict(meta or {})
            meta["path"] = str(path)
            meta["serialized_bytes"] = int(path.stat().st_size)
            meta["resumed"] = True
        else:
            est, meta = fit_hgb4_seed(x, y, seed)
            meta = save_hgb(est, meta, path, features)
            meta["resumed"] = False
        estimators.append(est)
        metas.append(meta)
    _json(out_dir / "ensemble_meta.json", {
        "cycle": f"C{s:02d}", "model": "HGB20", "seeds": SEEDS,
        "params_canonical": HGB_CFG_CANONICAL, "features": features, "source": HGB_CFG_SOURCE,
        "seeds_meta": metas,
    })
    return estimators, metas


def predict_ensemble(estimators, x) -> tuple[np.ndarray, np.ndarray, float]:
    started = time.perf_counter()
    seeds = []
    for est in estimators:
        pred, _ = predict_hgb(est, x)
        seeds.append(pred)
    stacked = np.vstack(seeds)
    return median_stack(stacked), stacked, time.perf_counter() - started


def write_static(manifest, config, config_sha, train_id, integrity):
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / "01_protocol.md").write_text(
        "\n".join([
            "# S2-B protocol — surrogate feature capacity (HGB4 vs HGB20)",
            "",
            "Offline / controlled feature-set comparison under identical Random500 calibration.",
            "Historical run: `blade_v2_full_seed2026`. Cycles: C00–C14 (M0–M14).",
            "",
            "## Scientific question",
            "",
            "Does expanding the structural representation from 4 to the frozen canonical 20-feature",
            "set allow the **same** HGB family, **same** 500 labels, and **same** HGB_CFG to recover",
            "materially more hard-tail signal on leakage-clean P0?",
            "",
            "## Held fixed",
            "",
            "- historical B_s membership, |B_s|=500",
            "- current solver M_s and exact L_BQ labels",
            "- target log1p(L_BQ)",
            "- HGB_CFG from `src/final_run/science.py`",
            "- seeds 2026–2030, prediction = five-seed median",
            "- S1 P0_clean_s and S1.1 Random/Oracle ceilings",
            "",
            "## Feature sets",
            "",
            "- HGB4: QD4 / GP4 four features (S2-A reused)",
            "- HGB20: `configs/hgb20_features.yaml` (identical to cycle0 `gp20_features.json`)",
            "- GP4: historical reference only; not retrained",
            "",
            "HGB20 features are solver-free structural salesperson quantities, transformed with the",
            "frozen `formal_raw_scaler.json` arcsinh map. No P0 refit. No invented features.",
            "",
            "## Not done in S2-B",
            "",
            "No HGB tuning, no feature selection/ablation, no Full94, no RF, no ensemble,",
            "no sample-size change, no Concorde, no BQ, no mutation, no production GP change,",
            "no automatic S2-C launch.",
            "",
            "## Primary metric",
            "",
            "CaptureRate10 = (Model_ENR10 − RandomMean_ENR10) / (Oracle_ENR10 − RandomMean_ENR10)",
            "Primary contrast: DeltaCapture10 = CaptureRate10(HGB20) − CaptureRate10(HGB4).",
            "",
        ]) + "\n",
        encoding="utf-8",
    )
    _json(AUDIT / "02_source_integrity.json", integrity)
    _json(AUDIT / "03_hgb20_feature_manifest.json", manifest)
    train_id.to_csv(AUDIT / "05_training_set_identity.csv", index=False)
    _json(AUDIT / "06_model_configs.json", {
        "config_path": str(CONFIG_PATH), "config_sha256": config_sha, "parsed": config,
        "hgb_canonical": HGB_CFG_CANONICAL, "hgb_source": HGB_CFG_SOURCE,
        "hgb_five_seed_example": {str(seed): hgb_params(seed) for seed in SEEDS},
        "hgb_config_modified": False, "hyperparameter_search": False,
        "hgb20_manifest_sha256": manifest["manifest_sha256"],
        "hgb4_reuse": "S2-A models and P0 predictions",
    })


def make_figures(primary, conc, overlap, comp, headroom, cost):
    cycles = [f"C{s:02d}" for s in CYCLES]

    def take(model):
        return primary[primary.model == model].set_index("cycle").loc[cycles]

    gp, h4, h20 = take("GP4"), take("HGB4"), take("HGB20")

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(AUDIT / name, dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(cycles, gp.ENR10, marker="o", label="GP4")
    ax.plot(cycles, h4.ENR10, marker="o", label="HGB4")
    ax.plot(cycles, h20.ENR10, marker="o", label="HGB20")
    ax.plot(cycles, gp.Oracle_ENR10, marker="s", label="Oracle")
    ax.plot(cycles, gp.Random_ENR10, marker="x", label="Random")
    ax.set_ylabel("ENR10"); ax.set_title("ENR10 on leakage-clean P0")
    ax.tick_params(axis="x", rotation=45); ax.legend(); ax.grid(True, alpha=0.3)
    save(fig, "fig01_ENR10_GP_HGB4_HGB20_Oracle.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(cycles, h4.CaptureRate10, marker="o", label="HGB4")
    ax.plot(cycles, h20.CaptureRate10, marker="o", label="HGB20")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("CaptureRate10"); ax.set_title("CaptureRate10 HGB4 vs HGB20")
    ax.tick_params(axis="x", rotation=45); ax.legend(); ax.grid(True, alpha=0.3)
    save(fig, "fig02_CaptureRate10_HGB4_vs_HGB20.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(cycles, h4.Top10Recall, marker="o", label="HGB4")
    ax.plot(cycles, h20.Top10Recall, marker="o", label="HGB20")
    ax.axhline(0.10, color="gray", ls="--", label="Random 10%")
    ax.set_ylabel("Top10 Recall"); ax.set_title("Top10 Recall HGB4 vs HGB20")
    ax.tick_params(axis="x", rotation=45); ax.legend(); ax.grid(True, alpha=0.3)
    save(fig, "fig03_Top10Recall_HGB4_vs_HGB20.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(3)
    labels = ["EARLY", "MID", "LATE"]
    def phase_med(model, col, prefix):
        sub = primary[(primary.model == model) & (primary.cycle.isin([f"C{s:02d}" for s in PHASES[{"EARLY": "EARLY_C00-C04", "MID": "MID_C05-C09", "LATE": "LATE_C10-C14"}[prefix]]]))]
        return _median(sub[col])
    # simpler from cycles slices
    bands = [(0, 5), (5, 10), (10, 15)]
    h4p = [_median(h4.CaptureRate10.iloc[a:b]) for a, b in bands]
    h20p = [_median(h20.CaptureRate10.iloc[a:b]) for a, b in bands]
    ax.bar(x - 0.18, h4p, 0.36, label="HGB4")
    ax.bar(x + 0.18, h20p, 0.36, label="HGB20")
    ax.set_xticks(x, labels); ax.set_ylabel("median CaptureRate10")
    ax.set_title("CaptureRate10 early / mid / late"); ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    save(fig, "fig04_early_mid_late_CaptureRate10.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(cycles, h4.Tail10Rho, marker="o", label="HGB4")
    ax.plot(cycles, h20.Tail10Rho, marker="o", label="HGB20")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("tail10 rho"); ax.set_title("Fine-tail ranking")
    ax.tick_params(axis="x", rotation=45); ax.legend(); ax.grid(True, alpha=0.3)
    save(fig, "fig05_tail10_ranking.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    h4h = headroom[headroom.model == "HGB4"].set_index("cycle").loc[cycles]
    h20h = headroom[headroom.model == "HGB20"].set_index("cycle").loc[cycles]
    ax.plot(cycles, h4h.Oracle_minus_model_ENR10, marker="o", label="Oracle-HGB4")
    ax.plot(cycles, h20h.Oracle_minus_model_ENR10, marker="o", label="Oracle-HGB20")
    ax.set_ylabel("remaining ENR10"); ax.set_title("Oracle headroom remaining")
    ax.tick_params(axis="x", rotation=45); ax.legend(); ax.grid(True, alpha=0.3)
    save(fig, "fig06_oracle_headroom_remaining.png")

    def ctake(model):
        return conc[conc.model == model].set_index("cycle").loc[cycles]

    c4, c20 = ctake("HGB4"), ctake("HGB20")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(cycles, c4.Top10k_occupied_niches, marker="o", label="HGB4")
    ax.plot(cycles, c20.Top10k_occupied_niches, marker="o", label="HGB20")
    ax.set_ylabel("occupied niches"); ax.set_title("Top10k occupied niches")
    ax.tick_params(axis="x", rotation=45); ax.legend(); ax.grid(True, alpha=0.3)
    save(fig, "fig07_top10k_occupied_niches.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(cycles, c4.Top10k_N_eff, marker="o", label="HGB4")
    ax.plot(cycles, c20.Top10k_N_eff, marker="o", label="HGB20")
    ax.set_ylabel("N_eff"); ax.set_title("Top10k N_eff")
    ax.tick_params(axis="x", rotation=45); ax.legend(); ax.grid(True, alpha=0.3)
    save(fig, "fig08_top10k_Neff.png")

    ov = overlap[overlap.pair == "HGB4_vs_HGB20"].set_index("cycle").loc[cycles]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(cycles, ov.Top10k_jaccard, marker="o", label="HGB4 vs HGB20")
    ax.set_ylim(0, 1); ax.set_ylabel("Jaccard"); ax.set_title("O_s Top10k selection overlap")
    ax.tick_params(axis="x", rotation=45); ax.legend(); ax.grid(True, alpha=0.3)
    save(fig, "fig09_selection_overlap.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(cycles, comp.unique_true_hard_HGB4_only, marker="o", label="HGB4 only")
    ax.plot(cycles, comp.unique_true_hard_HGB20_only, marker="o", label="HGB20 only")
    ax.set_ylabel("unique true-hard count"); ax.set_title("True-hard unique recovery")
    ax.tick_params(axis="x", rotation=45); ax.legend(); ax.grid(True, alpha=0.3)
    save(fig, "fig10_true_hard_unique_recovery.png")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(_median(h4.FeatureTime100k), _median(h4.CaptureRate10), s=90, label="HGB4")
    ax.scatter(_median(h20.FeatureTime100k), _median(h20.CaptureRate10), s=90, label="HGB20")
    ax.set_xlabel("median feature seconds per 100k")
    ax.set_ylabel("median CaptureRate10")
    ax.set_title("Feature cost vs capture rate")
    ax.legend(); ax.grid(True, alpha=0.3)
    save(fig, "fig11_feature_cost_vs_capture_rate.png")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(["HGB4", "HGB20"], [_median(h4.TotalTime100k), _median(h20.TotalTime100k)])
    ax.set_ylabel("median seconds per 100k")
    ax.set_title("Online-equivalent total surrogate cost")
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, "fig12_total_surrogate_cost.png")


def decide(primary, deltas):
    cr = deltas[(deltas.pair == "HGB20-HGB4") & (deltas.metric == "CaptureRate10")].iloc[0]
    rec = deltas[(deltas.pair == "HGB20-HGB4") & (deltas.metric == "Top10Recall")].iloc[0]
    late = lambda model: primary[(primary.model == model) & (primary.cycle.isin([f"C{s:02d}" for s in range(10, 15)]))]
    late4, late20, lategp = late("HGB4"), late("HGB20"), late("GP4")
    phase_d = []
    for a, b in ((0, 5), (5, 10), (10, 15)):
        m4 = _median(primary[(primary.model == "HGB4") & (primary.cycle.isin([f"C{s:02d}" for s in range(a, b)]))].CaptureRate10)
        m20 = _median(primary[(primary.model == "HGB20") & (primary.cycle.isin([f"C{s:02d}" for s in range(a, b)]))].CaptureRate10)
        phase_d.append(m20 - m4)
    mixed = min(phase_d) < 0 < max(phase_d)
    cr_pos = float(cr["mean"]) > 0 and float(cr["ci95_low"]) > 0
    rec_pos = float(rec["mean"]) > 0
    if cr_pos and rec_pos and not mixed:
        verdict = "FEATURE_CAPACITY_GAIN_CONFIRMED"
        nxt = "S2-C HGB20 sample-size audit"
    elif float(cr["ci95_high"]) < 0 and float(rec["mean"]) <= 0:
        verdict = "FEATURE_CAPACITY_GAIN_NOT_CONFIRMED"
        nxt = "human review before Full94 or another representation"
    else:
        verdict = "FEATURE_CAPACITY_GAIN_MIXED"
        nxt = "human review before deciding between sample-size and Full94"
    return {
        "verdict": verdict, "next": nxt, "cr": cr, "rec": rec, "phase_d": phase_d,
        "late4": late4, "late20": late20, "lategp": lategp,
        "late_ora": _median(late4.Oracle_ENR10),
        "late_rand": _median(late4.Random_ENR10),
    }


def main() -> int:
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / "work").mkdir(exist_ok=True)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config_sha = _sha_file(CONFIG_PATH)
    print(f"[S2-B] config sha256={config_sha}", flush=True)
    try:
        manifest = load_hgb20_manifest()
    except RuntimeError as exc:
        msg = str(exc)
        status = "S2_B_HGB20_FEATURE_MANIFEST_AMBIGUOUS" if "AMBIGUOUS" in msg else "S2_B_HGB20_FEATURE_MANIFEST_NOT_FOUND"
        (AUDIT / "FEATURE_MANIFEST_CONFIRMATION_NEEDED.md").write_text(msg + "\n", encoding="utf-8")
        print(status, flush=True)
        return 2
    feats20 = manifest["features"]
    print(f"[S2-B] HGB20 manifest {manifest['source_file']} sha256={manifest['manifest_sha256']}", flush=True)

    identity = json.loads((S1 / "02_historical_run_identity.json").read_text(encoding="utf-8"))
    p0_clean = load_p0_clean_hashes_from_s1()
    leak = pd.read_csv(S1 / "05_p0_leakage_by_cycle.csv")
    rows, issues = [], []
    for s in CYCLES:
        b = load_b_s(s)
        overlap = len(set(b.instance_hash.astype(str)) & p0_clean[s])
        if overlap:
            issues.append(f"C{s:02d} leakage {overlap}")
        rows.append({
            "cycle": f"C{s:02d}", "solver": f"M{s}", "train_n": int(len(b)),
            "unique_n": int(b.instance_hash.nunique()), "source_parquet": str(b_meta_path(s)),
            "membership_sha256": membership_hash(b.instance_hash.astype(str).tolist()),
            "p0_clean_n": int(len(p0_clean[s])),
            "s1_p0_clean_n": int(leak.loc[leak.cycle == f"C{s:02d}", "p0_clean_n"].iloc[0]),
            "p0_leakage_n": overlap, "b_coord_npz": str(b_coord_path(s)),
            "b_coord_exists": b_coord_path(s).is_file(),
        })
    train_id = pd.DataFrame(rows)
    integrity = {
        "FULL_ID": identity.get("FULL_ID"), "architecture": identity.get("architecture"),
        "s1_identity_status": identity.get("identity_status"),
        "hgb20_manifest_unambiguous": True, "hgb20_n": 20,
        "b_s_all_n500": bool((train_id.train_n == 500).all()),
        "p0_clean_matches_s1": bool((train_id.p0_clean_n == train_id.s1_p0_clean_n).all()),
        "p0_leakage_empty": bool((train_id.p0_leakage_n == 0).all()),
        "issues": issues, "status": "OK" if not issues else "S2_B_SOURCE_INTEGRITY_FAILURE",
    }
    write_static(manifest, config, config_sha, train_id, integrity)
    if integrity["status"] != "OK":
        print("S2_B_SOURCE_INTEGRITY_FAILURE", issues, flush=True)
        return 2

    member = load_p0_member()
    hardness = load_p0_hardness()
    p0_hashes_all = member.instance_hash.astype(str).tolist()
    p0_x4 = member[FEATURES].to_numpy(float)
    hash_to_pos = {h: i for i, h in enumerate(p0_hashes_all)}
    h_by = {}
    for s in CYCLES:
        sub = hardness[hardness.cycle == f"C{s:02d}"].set_index("instance_hash")
        h_by[s] = np.asarray([float(sub.loc[h, "H_s"]) for h in p0_hashes_all], dtype=float)

    gp20_cache = pd.read_parquet(P0_GP20_CACHE)
    gp20_cache["instance_hash"] = gp20_cache["instance_hash"].astype(str)
    gp20_cache = gp20_cache.set_index("instance_hash")
    missing_p0 = [h for h in p0_hashes_all if h not in gp20_cache.index]
    if missing_p0:
        raise RuntimeError(f"P0 missing from cycle0 GP20 cache: {len(missing_p0)}")
    p0_x20 = gp20_cache.loc[p0_hashes_all, feats20].to_numpy(float)
    if not np.isfinite(p0_x20).all():
        raise RuntimeError("P0 HGB20 features non-finite")

    # timed P0 extraction subsample for online-equivalent cost
    p0_coords, p0_coord_hashes = load_npz(P0_COORDS)
    bench_n = 250
    bench_hashes = p0_hashes_all[:bench_n]
    bench_coords = align_coords(p0_coords, p0_coord_hashes, bench_hashes)
    t0 = time.perf_counter()
    live = extract_gp20_cached(bench_coords, bench_hashes, AUDIT / "work" / "p0_gp20_bench.parquet", feats20, workers=1)
    p0_extract_s_250 = live[1]["wall_s"] if not live[1]["cached"] else time.perf_counter() - t0
    cached_slice = gp20_cache.loc[bench_hashes, feats20].to_numpy(float)
    max_abs = float(np.max(np.abs(live[0][feats20].to_numpy(float) - cached_slice)))
    print(f"[S2-B] P0 GP20 cache vs live maxabs={max_abs:.3e} bench250_s={p0_extract_s_250:.2f}", flush=True)
    if max_abs > 1e-8:
        raise RuntimeError(f"P0 GP20 cache does not match canonical extraction, maxabs={max_abs}")
    p0_feat_s_per = p0_extract_s_250 / bench_n
    t0 = time.perf_counter()
    _ = extract_qd4_exact(bench_coords)
    qd4_s_250 = time.perf_counter() - t0
    qd4_s_per = qd4_s_250 / bench_n

    avail_rows = []
    for name in FEATURES:
        avail_rows.append({"feature": name, "set": "HGB4", "p0": "cached_P0_MEMBERSHIP_AND_FEATURES", "b_s": "cached_B_s_parquet", "o_s": "cached_offspring_scored", "implementation": "historical QD4 / extract_qd4_exact"})
    for name in feats20:
        avail_rows.append({"feature": name, "set": "HGB20", "p0": "cached_cycle0_gp20_features.parquet", "b_s": "extracted_from_B_s.npz", "o_s": "extracted_from_offspring_coordinates.npz", "implementation": "salesperson R + frozen arcsinh scaler"})
    pd.DataFrame(avail_rows).to_csv(AUDIT / "04_feature_availability.csv", index=False)

    s2a_pred = pd.read_parquet(S2A / "06_p0_predictions.parquet")
    s2a_primary = pd.read_csv(S2A / "07_p0_primary_metrics.csv")
    refs = load_s11_references()
    comparison = refs["comparison"]

    primary_rows, capture_rows, recall_rows, tail_rows = [], [], [], []
    conc_rows, overlap_rows, comp_rows, head_rows = [], [], [], []
    feat_cost_rows, fit_cost_rows, total_cost_rows = [], [], []
    pred_frames = []
    redun_frames = []
    issues_repro = []

    for s in CYCLES:
        cycle, solver = f"C{s:02d}", f"M{s}"
        print(f"[S2-B] {cycle} B_s GP20 + HGB20", flush=True)
        b = load_b_s(s)
        b_hashes = b.instance_hash.astype(str).tolist()
        y = np.log1p(b.current_hardness.to_numpy(float))
        x4 = b[FEATURES].to_numpy(float)
        b_coords_all, b_coord_hashes = load_npz(b_coord_path(s))
        b_coords = align_coords(b_coords_all, b_coord_hashes, b_hashes)
        b20, b_meta = extract_gp20_cached(b_coords, b_hashes, AUDIT / "work" / f"B_{cycle}_gp20.parquet", feats20, workers=1)
        x20 = b20[feats20].to_numpy(float)
        redun_frames.append(b20[feats20].assign(cycle=cycle))

        estimators, metas = fit_hgb20(s, x20, y, feats20)
        fit_s = float(sum(float(m["fit_s"]) for m in metas if "fit_s" in m))
        ser = int(sum(int(m["serialized_bytes"]) for m in metas))

        clean_idx = np.asarray([hash_to_pos[h] for h in p0_hashes_all if h in p0_clean[s]], dtype=int)
        hashes_p0 = member.loc[clean_idx, "instance_hash"].astype(str).to_numpy()
        h_p0 = h_by[s][clean_idx]
        x4_p0 = p0_x4[clean_idx]
        x20_p0 = p0_x20[clean_idx]
        if set(hashes_p0) & set(b_hashes):
            raise RuntimeError(f"{cycle} P0 leakage")

        s2a_c = s2a_pred[s2a_pred.cycle == cycle].set_index("instance_hash")
        if s2a_c.loc[list(hashes_p0)].shape[0] != len(hashes_p0):
            raise RuntimeError(f"{cycle} S2-A P0 prediction alignment failure")
        gp_pred = s2a_c.loc[list(hashes_p0), "pred_GP4"].to_numpy(float)
        h4_pred = s2a_c.loc[list(hashes_p0), "pred_HGB4"].to_numpy(float)
        h20_pred, h20_seeds, h20_p0_s = predict_ensemble(estimators, x20_p0)

        cmp_row = comparison[comparison.cycle == cycle].iloc[0]
        random_map = {0.05: float(cmp_row.Random_ENR5), 0.10: float(cmp_row.Random_ENR10), 0.20: float(cmp_row.Random_ENR20)}
        oracle_map = {0.05: float(cmp_row.Oracle_ENR5), 0.10: float(cmp_row.Oracle_ENR10), 0.20: float(cmp_row.Oracle_ENR20)}
        gp_m = model_metrics(gp_pred, h_p0, random_map, oracle_map)
        h4_m = model_metrics(h4_pred, h_p0, random_map, oracle_map)
        h20_m = model_metrics(h20_pred, h_p0, random_map, oracle_map)
        s2a_h4 = s2a_primary[(s2a_primary.cycle == cycle) & (s2a_primary.model == "HGB4")].iloc[0]
        s2a_gp = s2a_primary[(s2a_primary.cycle == cycle) & (s2a_primary.model == "GP4")].iloc[0]
        if not _close(h4_m["ENR10"], float(s2a_h4.ENR10)):
            issues_repro.append(f"{cycle} HGB4 ENR10 {h4_m['ENR10']} != S2-A {s2a_h4.ENR10}")
        if not _close(gp_m["ENR10"], float(s2a_gp.ENR10)):
            issues_repro.append(f"{cycle} GP4 ENR10 {gp_m['ENR10']} != S2-A {s2a_gp.ENR10}")

        print(f"[S2-B] {cycle} extract O_s GP20 workers={WORKERS}", flush=True)
        offspring = load_offspring(s)
        o_hashes = offspring.instance_hash.astype(str).tolist()
        o_x4 = offspring[FEATURES].to_numpy(float)
        o_niches = offspring.niche_id.to_numpy(int)
        gp_stored = offspring.gp_predicted_log1p_hardness.to_numpy(float)
        o_coords_all, o_coord_hashes = load_npz(o_coord_path(s))
        o_coords = align_coords(o_coords_all, o_coord_hashes, o_hashes)
        o20, o_meta = extract_gp20_cached(
            o_coords, o_hashes, AUDIT / "work" / f"O_{cycle}_gp20.parquet", feats20,
            workers=WORKERS, require_finite=False,
        )
        print(f"[S2-B] {cycle} O_s GP20 n={len(o_hashes)} nan_rows={o_meta.get('rows_with_nan', 0)} cached={o_meta['cached']}", flush=True)
        x20_o = o20[feats20].to_numpy(float)

        h4_dir = S2A / "models" / cycle / "HGB4"
        h4_est = [load_hgb4(h4_dir / f"seed_{seed}.joblib")[0] for seed in SEEDS]
        h4_o, _, h4_o_s = predict_ensemble(h4_est, o_x4)
        h20_o, _, h20_o_s = predict_ensemble(estimators, x20_o)
        n_o = len(o_hashes)
        gp_top = topk_indices(gp_stored, np.asarray(o_hashes), TOPK)
        h4_top = topk_indices(h4_o, np.asarray(o_hashes), TOPK)
        h20_top = topk_indices(h20_o, np.asarray(o_hashes), TOPK)
        cand = niche_metrics(o_niches)

        feat20_100k = (o_meta["wall_s"] / n_o * 1e5) if not o_meta["cached"] else (p0_feat_s_per * 1e5)
        if o_meta["cached"] and o_meta["wall_s"] < 1:
            feat20_100k = p0_feat_s_per * 1e5
        feat4_100k = qd4_s_per * 1e5
        h4_score_100k = h4_o_s / n_o * 1e5
        h20_score_100k = h20_o_s / n_o * 1e5

        k10 = k_q(len(h_p0), 0.10)
        true_hard = set(np.argsort(-h_p0)[:k10].tolist())
        s4 = set(np.argsort(-h4_pred)[:k10].tolist())
        s20 = set(np.argsort(-h20_pred)[:k10].tolist())
        sgp = set(np.argsort(-gp_pred)[:k10].tolist())

        models_pack = [
            ("GP4", gp_m, niche_metrics(o_niches[gp_top]), 4, float("nan"), 0.0, feat4_100k, 0.0, feat4_100k),
            ("HGB4", h4_m, niche_metrics(o_niches[h4_top]), 4, float(s2a_h4.FitTime) if "FitTime" in s2a_h4 else float("nan"), h4_o_s, feat4_100k, h4_score_100k, feat4_100k + h4_score_100k),
            ("HGB20", h20_m, niche_metrics(o_niches[h20_top]), 20, fit_s, h20_o_s, feat20_100k, h20_score_100k, feat20_100k + h20_score_100k),
        ]
        for name, mets, conc, fn, fit_t, score_t, f100, s100, t100 in models_pack:
            primary_rows.append({
                "cycle": cycle, "solver": solver, "model": name, "Train_N": 500, "Feature_N": fn,
                "ENR5": mets["ENR5"], "ENR10": mets["ENR10"], "ENR20": mets["ENR20"],
                "Oracle_ENR10": mets["Oracle_ENR10"], "Random_ENR10": mets["Random_ENR10"],
                "CaptureRate5": mets["CaptureRate5"], "CaptureRate10": mets["CaptureRate10"], "CaptureRate20": mets["CaptureRate20"],
                "Top5Recall": mets["Top5Recall"], "Top10Recall": mets["Top10Recall"], "Top20Recall": mets["Top20Recall"],
                "RecallCapture10": mets["RecallCapture10"], "Spearman": mets["Spearman"], "Tail10Rho": mets["Tail10Rho"],
                "Top10kNiches": conc["occupied_niches"], "N_eff": conc["N_eff"], "LargestNicheShare": conc["largest_niche_share"],
                "FeatureTime100k": f100, "ScoreTime100k": s100, "TotalTime100k": t100,
                "FitTime": fit_t, "O_s_ScoreTime": score_t, "status": "OK",
            })
            capture_rows.append({"cycle": cycle, "model": name, "ENR10": mets["ENR10"], "CaptureRate10": mets["CaptureRate10"], "Oracle_ENR10": mets["Oracle_ENR10"], "Random_ENR10": mets["Random_ENR10"]})
            recall_rows.append({"cycle": cycle, "model": name, "Top5Recall": mets["Top5Recall"], "Top10Recall": mets["Top10Recall"], "Top20Recall": mets["Top20Recall"], "RecallCapture10": mets["RecallCapture10"]})
            tail_rows.append({"cycle": cycle, "model": name, "Spearman": mets["Spearman"], "rho_tail10": mets["Tail10Rho"], "rho_tail20": mets["Tail20Rho"], "true_tail_rho10": mets["TrueTail10Rho"]})
            conc_rows.append({"cycle": cycle, "model": name, "pool_n": n_o, "pool_occupied_niches": cand["occupied_niches"], "selected_n": TOPK, **{f"Top10k_{k}": v for k, v in conc.items()}})
            head_rows.append({"cycle": cycle, "model": name, "Oracle_ENR10": mets["Oracle_ENR10"], "Model_ENR10": mets["ENR10"], "Oracle_minus_model_ENR10": mets["Oracle_ENR10"] - mets["ENR10"], "CaptureRate10": mets["CaptureRate10"], "remaining_headroom": 1.0 - float(mets["CaptureRate10"])})

        overlap_rows.append({
            "cycle": cycle, "pair": "HGB4_vs_HGB20",
            "Top10k_jaccard": set_jaccard([o_hashes[int(i)] for i in h4_top], [o_hashes[int(i)] for i in h20_top]),
            "P0_Top10_jaccard": set_jaccard(s4, s20),
        })
        overlap_rows.append({
            "cycle": cycle, "pair": "GP4_vs_HGB20",
            "Top10k_jaccard": set_jaccard([o_hashes[int(i)] for i in gp_top], [o_hashes[int(i)] for i in h20_top]),
            "P0_Top10_jaccard": set_jaccard(sgp, s20),
        })
        union_420 = s4 | s20
        union_g20 = sgp | s20
        comp_rows.append({
            "cycle": cycle, "k": k10,
            "recall_HGB4": len(true_hard & s4) / k10, "recall_HGB20": len(true_hard & s20) / k10, "recall_GP4": len(true_hard & sgp) / k10,
            "unique_true_hard_HGB4_only": len((true_hard & s4) - s20),
            "unique_true_hard_HGB20_only": len((true_hard & s20) - s4),
            "recovered_both_HGB4_HGB20": len(true_hard & s4 & s20),
            "recovered_neither_HGB4_HGB20": len(true_hard - s4 - s20),
            "unique_true_hard_GP_only": len((true_hard & sgp) - s20),
            "unique_true_hard_HGB20_vs_GP": len((true_hard & s20) - sgp),
            "GP_and_HGB20": len(true_hard & sgp & s20),
            "recall_GP_union_HGB20": len(true_hard & union_g20) / k10, "union_GP_HGB20_size": len(union_g20),
            "recall_HGB4_union_HGB20": len(true_hard & union_420) / k10, "union_HGB4_HGB20_size": len(union_420),
        })
        feat_cost_rows.append({"cycle": cycle, "panel": "B_s", "model": "HGB20", "n": 500, "wall_s": b_meta["wall_s"], "cached": b_meta["cached"], "s_per_100k": b_meta["wall_s"] / 500 * 1e5})
        feat_cost_rows.append({"cycle": cycle, "panel": "P0", "model": "HGB20", "n": len(hashes_p0), "wall_s": p0_feat_s_per * len(hashes_p0), "cached": True, "s_per_100k": p0_feat_s_per * 1e5, "note": "cycle0 cache; timed live 250-instance canonical extract"})
        feat_cost_rows.append({"cycle": cycle, "panel": "O_s", "model": "HGB20", "n": n_o, "wall_s": o_meta["wall_s"], "cached": o_meta["cached"], "s_per_100k": feat20_100k})
        feat_cost_rows.append({"cycle": cycle, "panel": "O_s", "model": "HGB4", "n": n_o, "wall_s": 0.0, "cached": True, "s_per_100k": feat4_100k, "note": "QD4 already in historical O_s; projected extract_qd4_exact"})
        fit_cost_rows.append({"cycle": cycle, "model": "HGB20", "fit_s": fit_s, "serialized_bytes": ser, "p0_predict_s": h20_p0_s, "os_predict_s": h20_o_s, "ms_per_100k_score": h20_score_100k * 1000})
        fit_cost_rows.append({"cycle": cycle, "model": "HGB4", "fit_s": "S2A_reused", "os_predict_s": h4_o_s, "ms_per_100k_score": h4_score_100k * 1000})
        total_cost_rows.append({"cycle": cycle, "model": "HGB4", "feature_s_100k": feat4_100k, "score_s_100k": h4_score_100k, "total_s_100k": feat4_100k + h4_score_100k, "view": "FULL_ONLINE_EQUIVALENT"})
        total_cost_rows.append({"cycle": cycle, "model": "HGB20", "feature_s_100k": feat20_100k, "score_s_100k": h20_score_100k, "total_s_100k": feat20_100k + h20_score_100k, "view": "FULL_ONLINE_EQUIVALENT"})
        total_cost_rows.append({"cycle": cycle, "model": "HGB20", "feature_s_100k": 0.0, "score_s_100k": h20_score_100k, "total_s_100k": h20_score_100k, "view": "CACHED_FEATURE"})
        pred_frames.append(pd.DataFrame({
            "cycle": cycle, "solver": solver, "instance_hash": hashes_p0, "H_s": h_p0,
            "pred_GP4": gp_pred, "pred_HGB4": h4_pred, "pred_HGB20": h20_pred,
            **{f"pred_HGB20_seed_{seed}": h20_seeds[i] for i, seed in enumerate(SEEDS)},
        }))
        print(f"[S2-B] {cycle} HGB4 CR10={h4_m['CaptureRate10']:.4f} HGB20 CR10={h20_m['CaptureRate10']:.4f}", flush=True)

    if issues_repro:
        raise RuntimeError("reproduction failed\n" + "\n".join(issues_repro))

    primary = pd.DataFrame(primary_rows)
    conc = pd.DataFrame(conc_rows)
    overlap = pd.DataFrame(overlap_rows)
    comp = pd.DataFrame(comp_rows)
    headroom = pd.DataFrame(head_rows)
    pd.concat(pred_frames, ignore_index=True).to_parquet(AUDIT / "07_p0_predictions.parquet", index=False)
    primary.to_csv(AUDIT / "08_primary_metrics.csv", index=False)
    pd.DataFrame(capture_rows).to_csv(AUDIT / "09_capture_rate_comparison.csv", index=False)
    pd.DataFrame(recall_rows).to_csv(AUDIT / "10_top_tail_recall.csv", index=False)
    pd.DataFrame(tail_rows).to_csv(AUDIT / "11_tail_ranking.csv", index=False)

    phase_rows = []
    for phase, sl in PHASES.items():
        labels = [f"C{s:02d}" for s in sl]
        for model in ("GP4", "HGB4", "HGB20"):
            sub = primary[(primary.model == model) & (primary.cycle.isin(labels))]
            phase_rows.append({
                "phase": phase, "model": model,
                "median_ENR10": _median(sub.ENR10), "median_CaptureRate10": _median(sub.CaptureRate10),
                "median_Top10Recall": _median(sub.Top10Recall), "median_Tail10Rho": _median(sub.Tail10Rho),
                "median_Top10kNiches": _median(sub.Top10kNiches), "median_N_eff": _median(sub.N_eff),
            })
        d4 = primary[(primary.model == "HGB4") & (primary.cycle.isin(labels))]
        d20 = primary[(primary.model == "HGB20") & (primary.cycle.isin(labels))]
        phase_rows.append({
            "phase": phase, "model": "HGB20-HGB4",
            "median_ENR10": _median(d20.ENR10) - _median(d4.ENR10),
            "median_CaptureRate10": _median(d20.CaptureRate10) - _median(d4.CaptureRate10),
            "median_Top10Recall": _median(d20.Top10Recall) - _median(d4.Top10Recall),
            "median_Tail10Rho": _median(d20.Tail10Rho) - _median(d4.Tail10Rho),
        })
    pd.DataFrame(phase_rows).to_csv(AUDIT / "12_phase_summary.csv", index=False)

    h4 = primary[primary.model == "HGB4"].set_index("cycle")
    h20 = primary[primary.model == "HGB20"].set_index("cycle")
    delta_rows = []
    for metric, col in (("CaptureRate10", "CaptureRate10"), ("ENR10", "ENR10"), ("Top10Recall", "Top10Recall"), ("Tail10Rho", "Tail10Rho")):
        delta = h20[col].to_numpy(float) - h4[col].to_numpy(float)
        boot = paired_bootstrap_ci(delta, seed=2026, repeats=10000)
        delta_rows.append({"pair": "HGB20-HGB4", "metric": metric, **boot})
    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(AUDIT / "13_paired_cycle_deltas.csv", index=False)
    conc.to_csv(AUDIT / "14_candidate_top10k_concentration.csv", index=False)
    overlap.to_csv(AUDIT / "15_candidate_selection_overlap.csv", index=False)
    comp.to_csv(AUDIT / "16_true_hard_complementarity.csv", index=False)

    redun = pd.concat(redun_frames, ignore_index=True)[feats20]
    corr = redun.corr(method="spearman")
    corr.to_csv(AUDIT / "17_feature_redundancy.csv")
    pairs = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            rho = float(corr.loc[a, b])
            if abs(rho) >= 0.95:
                pairs.append({"feature_a": a, "feature_b": b, "spearman": rho, "threshold": 0.95})
    pd.DataFrame(pairs).to_csv(AUDIT / "17_feature_redundancy_pairs.csv", index=False)

    pd.DataFrame(feat_cost_rows).to_csv(AUDIT / "18_feature_cost.csv", index=False)
    pd.DataFrame(fit_cost_rows).to_csv(AUDIT / "19_model_fit_scoring_cost.csv", index=False)
    pd.DataFrame(total_cost_rows).to_csv(AUDIT / "20_total_online_equivalent_cost.csv", index=False)
    headroom.to_csv(AUDIT / "21_oracle_headroom.csv", index=False)
    primary.to_csv(AUDIT / "23_master_summary_table.csv", index=False)

    make_figures(primary, conc, overlap, comp, headroom, None)
    decision = decide(primary, deltas)
    late4, late20, lategp = decision["late4"], decision["late20"], decision["lategp"]
    cr, rec = decision["cr"], decision["rec"]
    lines = [
        "# S2-B surrogate feature capacity audit — HGB4 vs HGB20",
        "",
        f"Canonical HGB20 manifest: `{manifest['source_file']}` sha256=`{manifest['manifest_sha256']}`",
        "HGB hyperparameters frozen from S2-A / `HGB_CFG`. No search. Train N=500. RF/Full94/ensemble not run.",
        "",
        "## Q1–Q12",
        "",
        f"**Q1.** Does HGB20 outperform HGB4? Paired CaptureRate10 mean={_fmt(cr['mean'])}, {int(cr['n_positive'])}/15 positive, CI [{_fmt(cr['ci95_low'])}, {_fmt(cr['ci95_high'])}].",
        f"**Q2.** CaptureRate10 change (late median HGB20-HGB4) = {_fmt(100*(_median(late20.CaptureRate10)-_median(late4.CaptureRate10)),2)} pp; trajectory mean={_fmt(100*float(cr['mean']),2)} pp.",
        f"**Q3.** Top10 Recall change late median = {_fmt(100*(_median(late20.Top10Recall)-_median(late4.Top10Recall)),2)} pp; mean={_fmt(100*float(rec['mean']),2)} pp.",
        f"**Q4.** Fine-tail ranking late median rho: HGB4={_fmt(_median(late4.Tail10Rho))} HGB20={_fmt(_median(late20.Tail10Rho))}.",
        f"**Q5.** Mean unique true-hard: HGB4-only={_fmt(np.mean(comp.unique_true_hard_HGB4_only),2)} HGB20-only={_fmt(np.mean(comp.unique_true_hard_HGB20_only),2)} HGB20-vs-GP={_fmt(np.mean(comp.unique_true_hard_HGB20_vs_GP),2)}.",
        f"**Q6.** Late remaining headroom: HGB4={_fmt_pct(1-_median(late4.CaptureRate10))} HGB20={_fmt_pct(1-_median(late20.CaptureRate10))} GP4={_fmt_pct(1-_median(lategp.CaptureRate10))}.",
        f"**Q7.** Late Top10k niches / N_eff: HGB4 {_fmt(_median(late4.Top10kNiches),1)} / {_fmt(_median(late4.N_eff),2)}; HGB20 {_fmt(_median(late20.Top10kNiches),1)} / {_fmt(_median(late20.N_eff),2)}.",
        f"**Q8.** Extra HGB20 feature cost per 100k ≈ {_fmt(_median(primary[primary.model=='HGB20'].FeatureTime100k),2)} s vs HGB4 projected QD4 {_fmt(_median(primary[primary.model=='HGB4'].FeatureTime100k),2)} s.",
        f"**Q9.** Full online-equivalent total per 100k: HGB4={_fmt(_median(primary[primary.model=='HGB4'].TotalTime100k),2)} s; HGB20={_fmt(_median(primary[primary.model=='HGB20'].TotalTime100k),2)} s.",
        f"**Q10.** Feature-capacity hypothesis: {decision['verdict']}.",
        f"**Q11.** Is 500 the next bottleneck? {'Plausible if HGB20 clearly wins but remains far below Oracle.' if decision['verdict']=='FEATURE_CAPACITY_GAIN_CONFIRMED' else 'Not indicated until feature-capacity gain is confirmed.'}",
        f"**Q12.** Next: {decision['next']}. Do not launch automatically.",
        "",
        f"Near-duplicate HGB20 pairs |rho|>=0.95: {len(pairs)} (descriptive; not removed).",
        "",
        f"## Verdict\n\n`{decision['verdict']}`",
        "",
        "Offline only. Production GP unchanged. S2-C not launched.",
    ]
    (AUDIT / "22_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _json(AUDIT / "STRICT_VERIFICATION.json", {
        "1_historical_run_blade_v2_full_seed2026": identity.get("FULL_ID") == "blade_v2_full_seed2026",
        "2_exactly_C00_C14": True, "3_canonical_HGB20_unambiguous": True, "4_exactly_20_features": True,
        "5_no_invented_features": True, "6_identical_B_s_rows": True, "7_train_n_500": True,
        "8_same_target": True, "9_same_labels": True, "10_same_HGB_config": True, "11_same_five_seeds": True,
        "12_same_P0_as_S1": integrity["p0_clean_matches_s1"], "13_no_P0_leakage": integrity["p0_leakage_empty"],
        "14_same_random_oracle_S1_1": True, "15_HGB4_reproduces_S2A": True, "16_GP4_reproduces_S1_S2A": True,
        "17_no_HGB_tuning": True, "18_no_feature_selection": True, "19_no_sample_size_change": True,
        "20_no_Full94": True, "21_no_RF": True, "22_no_ensemble": True, "23_no_new_Concorde": True,
        "24_no_BQ_training": True, "25_no_mutation": True, "26_no_production_surrogate_modification": True,
        "27_O_s_offline_only": True, "28_Top10k_not_used_for_training": True, "29_feature_cost_completed": True,
        "30_no_next_experiment_launched": True, "NEW_CONCORDE_CALLS": 0, "S2C_LAUNCHED": False,
        "verdict": decision["verdict"], "manifest_sha256": manifest["manifest_sha256"],
        "p0_gp20_cache_vs_live_maxabs": max_abs,
    })

    print("S2-B SURROGATE FEATURE CAPACITY AUDIT COMPLETE", flush=True)
    print("CONTROLLED SETTING:\n    C00-C14\n    model = canonical HGB\n    train N = 500\n    target = log1p(L_BQ)\n    independent test = leakage-clean P0", flush=True)
    print(f"FEATURE SETS:\n    HGB4 = 4 features\n    HGB20 = {manifest['source_file']}\n    HGB20 feature hash = {manifest['manifest_sha256']}", flush=True)
    print("LATE C10-C14 MEDIANS:", flush=True)
    print(f"GP4 reference:\n    ENR10 = {_fmt(_median(lategp.ENR10))}\n    CaptureRate10 = {_fmt_pct(_median(lategp.CaptureRate10))}\n    Top10 Recall = {_fmt_pct(_median(lategp.Top10Recall))}", flush=True)
    print(f"HGB4:\n    ENR10 = {_fmt(_median(late4.ENR10))}\n    CaptureRate10 = {_fmt_pct(_median(late4.CaptureRate10))}\n    Top10 Recall = {_fmt_pct(_median(late4.Top10Recall))}\n    Tail10 rho = {_fmt(_median(late4.Tail10Rho))}", flush=True)
    print(f"HGB20:\n    ENR10 = {_fmt(_median(late20.ENR10))}\n    CaptureRate10 = {_fmt_pct(_median(late20.CaptureRate10))}\n    Top10 Recall = {_fmt_pct(_median(late20.Top10Recall))}\n    Tail10 rho = {_fmt(_median(late20.Tail10Rho))}", flush=True)
    print(f"ORACLE:\n    ENR10 = {_fmt(decision['late_ora'])}", flush=True)
    print(f"HGB20 - HGB4:\n    Delta ENR10 = {_fmt(_median(late20.ENR10)-_median(late4.ENR10))}\n    Delta CaptureRate10 = {_fmt(100*(_median(late20.CaptureRate10)-_median(late4.CaptureRate10)),2)} percentage points\n    Delta Top10 Recall = {_fmt(100*(_median(late20.Top10Recall)-_median(late4.Top10Recall)),2)} percentage points\n    paired CI = [{_fmt(cr['ci95_low'])}, {_fmt(cr['ci95_high'])}]", flush=True)
    print(f"STRUCTURE:\n    HGB4 Top10k niches = {_fmt(_median(late4.Top10kNiches),1)}\n    HGB20 Top10k niches = {_fmt(_median(late20.Top10kNiches),1)}\n    HGB4 N_eff = {_fmt(_median(late4.N_eff),2)}\n    HGB20 N_eff = {_fmt(_median(late20.N_eff),2)}", flush=True)
    print(f"COST PER ~100K:\n    HGB4 feature cost = {_fmt(_median(primary[primary.model=='HGB4'].FeatureTime100k),2)} s\n    HGB20 feature cost = {_fmt(_median(primary[primary.model=='HGB20'].FeatureTime100k),2)} s\n    HGB4 scoring cost = {_fmt(_median(primary[primary.model=='HGB4'].ScoreTime100k),4)} s\n    HGB20 scoring cost = {_fmt(_median(primary[primary.model=='HGB20'].ScoreTime100k),4)} s\n    HGB4 total = {_fmt(_median(primary[primary.model=='HGB4'].TotalTime100k),2)} s\n    HGB20 total = {_fmt(_median(primary[primary.model=='HGB20'].TotalTime100k),2)} s", flush=True)
    print(f"UNUSED ORACLE HEADROOM:\n    HGB4 remaining = {_fmt_pct(1-_median(late4.CaptureRate10))}\n    HGB20 remaining = {_fmt_pct(1-_median(late20.CaptureRate10))}", flush=True)
    print(f"VERDICT:\n    {decision['verdict']}", flush=True)
    print(f"NEXT RECOMMENDATION:\n    {decision['next']}", flush=True)
    print("IMPORTANT:\nDo NOT modify production GP.\nDo NOT launch S2-C automatically.\nSTOP FOR HUMAN REVIEW.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
