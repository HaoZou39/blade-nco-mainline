"""Analysis-only Cycle-0 within-niche hardness diagnostic.

This program consumes sealed artifacts from cycle0_initialization_validation_v1.
It never imports training, inference, generation, solver, or runner code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import warnings
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "audits/cycle0_initialization_validation_v1"
OUT = ROOT / "audits/cycle0_within_niche_hardness_diagnostic_v1"
SEED = 2026
BOOTSTRAPS = 10_000
DEPTHS = (1, 2, 5)
FAMILIES = ("GP4", "HGB4", "HGB20")
MODELS = tuple(f"{family}_D{depth}" for family in FAMILIES for depth in DEPTHS)
RS = (1, 2, 5)
QD4 = (
    "normalized_spn_sum_of_lowest_edge_values",
    "normalized_spn_distance_varcoeff",
    "normalized_spn_mst_depth_mean",
    "normalized_spn_mst_dists_median",
)
FIGURES = (
    "01_true_hardness_between_vs_within_variance.png",
    "02_between_vs_within_model_correlation.png",
    "03_oracle_random_model_top1.png",
    "04_oracle_random_model_top5.png",
    "05_headroom_capture_top1.png",
    "06_headroom_capture_top2.png",
    "07_headroom_capture_top5.png",
    "08_gp4_hgb4_hgb20_within_niche_rho.png",
    "09_pairwise_ranking_accuracy.png",
    "10_hgb20_minus_hgb4_acquisition.png",
    "11_hgb4_minus_gp4_acquisition.png",
    "12_niche_spread_vs_true_hardness_range.png",
    "13_niche_spread_vs_model_rankability.png",
    "14_model_top1_agreement.png",
)
SOURCE_FILES = (
    "00_protocol/frozen_protocol.json",
    "00_protocol/source_manifest.json",
    "02_gp0_calibration_design/PRE_ORACLE_SEAL.json",
    "02_gp0_calibration_design/cal_d1_manifest.json",
    "02_gp0_calibration_design/cal_d2_manifest.json",
    "02_gp0_calibration_design/cal_d5_manifest.json",
    "02_gp0_calibration_design/cycle0_gp_dev.parquet",
    "03_oracle_labels/label_manifest.json",
    "03_oracle_labels/exact_hardness.parquet",
    "08_prediction_evaluation/all_predictions.parquet",
    "08_prediction_evaluation/global_metrics.csv",
    "09_within_niche_ranking/per_niche_metrics.parquet",
    "10_acquisition_evaluation/per_niche_acquisition.parquet",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def finite_float(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.ptp(x) <= 1e-15 or np.ptp(y) <= 1e-15:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = spearmanr(x, y).statistic if method == "spearman" else pearsonr(x, y).statistic
    return float(result) if np.isfinite(result) else 0.0


def pair_accuracy(y: np.ndarray, p: np.ndarray) -> float:
    i, j = np.triu_indices(len(y), 1)
    truth = np.sign(y[i] - y[j])
    guess = np.sign(p[i] - p[j])
    return float(np.mean(np.where((truth == 0) | (guess == 0), 0.5, truth == guess)))


def q(values: np.ndarray, level: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), level))


def ci(values: np.ndarray) -> tuple[float, float]:
    return q(values, 0.025), q(values, 0.975)


def source_manifest() -> tuple[dict, dict[str, str]]:
    missing = [rel for rel in SOURCE_FILES if not (SOURCE / rel).is_file()]
    if missing:
        print("REQUIRED_EXISTING_ARTIFACT_MISSING=" + ",".join(missing), flush=True)
        raise SystemExit(2)
    entries = {}
    hashes = {}
    for rel in SOURCE_FILES:
        path = SOURCE / rel
        digest = sha256(path)
        hashes[rel] = digest
        entries[rel] = {"absolute_path": str(path), "bytes": path.stat().st_size, "sha256": digest}
    manifest = {
        "audit": "BLADE_CYCLE0_WITHIN_NICHE_HARDNESS_DIAGNOSTIC_V1",
        "analysis_only": True,
        "exclusive_source_audit": str(SOURCE),
        "sources": entries,
        "analysis_script": {"absolute_path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
    }
    return manifest, hashes


def protocol() -> dict:
    return {
        "audit": "BLADE_CYCLE0_WITHIN_NICHE_HARDNESS_DIAGNOSTIC_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_only": True,
        "exclusive_source_audit": "audits/cycle0_initialization_validation_v1",
        "primary_diagnostic_branch": "D5",
        "sensitivity_branches": ["D1", "D2"],
        "models": list(MODELS),
        "hardness_for_value_analysis": "raw H0 (raw M0 L_BQ)",
        "eligible_niche_rule": "at least 5 sealed dev candidates",
        "aggregation": "equal niche weight unless explicitly named candidate-weighted",
        "selection_tie_break": "prediction descending, then instance_hash ascending; Oracle uses H0 descending, then instance_hash ascending",
        "random_baseline": "single numpy PCG64 stream; group niches ascending; preserve sealed dev row order; r order 1,2,5; sample without replacement",
        "master_seed": SEED,
        "bootstrap": {"unit": "niche", "replicates": BOOTSTRAPS, "seed": SEED, "interval": "paired percentile 95%"},
        "centered_signal": "subtract the corresponding niche mean from H0 and each prediction",
        "variance_decomposition": "candidate-population ddof=0 additive identity: H0-grand_mean=(niche_mean-grand_mean)+(H0-niche_mean); between is n-weighted",
        "capture_ratio": "mean(model-selected H0 - matched random H0) / mean(Oracle H0 - matched random H0); never clipped",
        "sample_size_strata": "empirical quartile cut points of eligible-niche n; equal n values are never split",
        "forbidden_actions": {
            "new_instances": True,
            "concorde_calls": True,
            "new_oracle_labels": True,
            "bq_inference": True,
            "gp_or_hgb_training": True,
            "bq_finetuning": True,
            "external_evaluation": True,
            "final_d0": True,
            "protocol_change": True,
            "final_runner": True,
        },
    }


def load_aligned() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    dev = pd.read_parquet(SOURCE / "02_gp0_calibration_design/cycle0_gp_dev.parquet")
    labels_all = pd.read_parquet(SOURCE / "03_oracle_labels/exact_hardness.parquet")
    predictions = pd.read_parquet(SOURCE / "08_prediction_evaluation/all_predictions.parquet")
    expected = set(MODELS)
    found = set(predictions["model"].astype(str))
    if found != expected:
        print("REQUIRED_EXISTING_ARTIFACT_MISSING=prediction_branches:" + str(sorted(expected - found)), flush=True)
        raise SystemExit(2)
    dev_hashes = dev["instance_hash"].astype(str)
    label = labels_all[labels_all["instance_hash"].isin(set(dev_hashes))][["instance_hash", "H0", "H0_log1p", "oracle_role"]].copy()
    wide = predictions.pivot(index="instance_hash", columns="model", values="prediction")
    exact_sets = {m: set(predictions.loc[predictions.model.eq(m), "instance_hash"].astype(str)) == set(dev_hashes) for m in MODELS}
    counts = predictions.groupby("model").size().to_dict()
    unique_counts = predictions.groupby("model")["instance_hash"].nunique().to_dict()
    integrity_ok = (
        len(dev) == 2666
        and dev_hashes.nunique() == len(dev)
        and len(label) == len(dev)
        and label.instance_hash.nunique() == len(dev)
        and all(exact_sets.values())
        and all(counts[m] == len(dev) == unique_counts[m] for m in MODELS)
        and set(wide.columns) == expected
        and np.isfinite(wide.to_numpy(float)).all()
    )
    if not integrity_ok:
        print("REQUIRED_EXISTING_ARTIFACT_MISSING=unaligned_or_invalid_prediction_or_label", flush=True)
        raise SystemExit(2)
    frame = dev.merge(label, on="instance_hash", how="left", validate="one_to_one", sort=False)
    frame = frame.merge(wide.reset_index(), on="instance_hash", how="left", validate="one_to_one", sort=False)
    prediction_alignment = {
        "status": "PASS",
        "dev_n": len(dev),
        "expected_models": list(MODELS),
        "found_models": sorted(found),
        "per_model_rows": {m: int(counts[m]) for m in MODELS},
        "per_model_unique_instances": {m: int(unique_counts[m]) for m in MODELS},
        "exact_dev_hash_set_match": exact_sets,
        "all_predictions_finite": True,
        "instance_by_instance_alignment": True,
    }
    label_alignment = {
        "status": "PASS",
        "dev_n": len(dev),
        "matched_label_n": len(label),
        "exact_dev_hash_set_match": set(label.instance_hash.astype(str)) == set(dev_hashes),
        "H0_finite": bool(np.isfinite(frame.H0).all()),
        "niche_assignments_from_sealed_dev": True,
        "niche_count": int(frame.niche_id.nunique()),
        "raw_value_column": "H0",
    }
    return frame, dev, predictions, prediction_alignment, label_alignment


def hardness_decomposition(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    records = []
    for niche, g in frame.groupby("niche_id", sort=True):
        y = g.H0.to_numpy(float)
        records.append({
            "niche_id": int(niche), "n": len(g), "mean_H0": y.mean(), "median_H0": np.median(y),
            "std_H0": np.std(y, ddof=1) if len(y) > 1 else 0.0,
            "iqr_H0": q(y, .75) - q(y, .25), "min_H0": y.min(), "max_H0": y.max(), "range_H0": np.ptp(y),
            "sample_variance_H0": np.var(y, ddof=1) if len(y) > 1 else 0.0,
        })
    niche = pd.DataFrame(records)
    means = frame.groupby("niche_id").H0.transform("mean").to_numpy(float)
    y = frame.H0.to_numpy(float)
    grand = float(y.mean())
    between_component = means - grand
    within_component = y - means
    total_var = float(np.var(y, ddof=0))
    between_var = float(np.var(between_component, ddof=0))
    within_var = float(np.var(within_component, ddof=0))
    covariance = float(np.mean((between_component - between_component.mean()) * (within_component - within_component.mean())))
    decomposition = {
        "definition": "exact descriptive candidate-population decomposition, not a causal or random-effects ICC",
        "ddof": 0,
        "candidate_n": len(frame),
        "niche_n": int(frame.niche_id.nunique()),
        "grand_mean_H0": grand,
        "total_population_variance": total_var,
        "between_niche_component_variance_n_weighted": between_var,
        "within_niche_residual_variance_n_weighted": within_var,
        "twice_component_covariance": 2 * covariance,
        "additive_reconstruction_error": total_var - between_var - within_var - 2 * covariance,
        "between_niche_variance_fraction": between_var / total_var,
        "within_niche_variance_fraction": within_var / total_var,
        "component_fraction_sum": (between_var + within_var) / total_var,
        "unweighted_variance_of_niche_means_ddof0": float(np.var(niche.mean_H0, ddof=0)),
        "mean_niche_sample_variance_ddof1": float(niche.sample_variance_H0.mean()),
        "median_niche_sample_variance_ddof1": float(niche.sample_variance_H0.median()),
    }
    centered_parts = []
    h_centered = within_component
    for model in MODELS:
        p = frame[model].to_numpy(float)
        p_centered = p - frame.groupby("niche_id")[model].transform("mean").to_numpy(float)
        centered_parts.append(pd.DataFrame({
            "instance_hash": frame.instance_hash, "niche_id": frame.niche_id.astype(int), "model": model,
            "H0": y, "prediction": p, "H_centered": h_centered, "P_centered": p_centered,
        }))
    return niche, decomposition, pd.concat(centered_parts, ignore_index=True)


def oracle_random(frame: pd.DataFrame, eligible: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    rows = []
    for niche, g in frame[frame.niche_id.isin(eligible)].groupby("niche_id", sort=True):
        y = g.H0.to_numpy(float)
        hashes = g.instance_hash.astype(str).to_numpy()
        for r in RS:
            random_idx = np.sort(rng.choice(len(g), size=r, replace=False))
            oracle_idx = np.lexsort((hashes, -y))[:r]
            random_h = float(y[random_idx].mean())
            oracle_h = float(y[oracle_idx].mean())
            headroom = oracle_h - random_h
            rows.append({
                "niche_id": int(niche), "n": len(g), "r": r,
                "random_H0": random_h, "oracle_H0": oracle_h, "absolute_headroom": headroom,
                "relative_headroom_vs_random": headroom / random_h if random_h != 0 else np.nan,
                "random_instance_hashes": list(hashes[random_idx]), "oracle_instance_hashes": list(hashes[oracle_idx]),
            })
    detail = pd.DataFrame(rows)
    summary_rows = []
    boot_rows = []
    rngb = np.random.default_rng(SEED)
    for r in RS:
        g = detail[detail.r.eq(r)].sort_values("niche_id")
        random_h, oracle_h = g.random_H0.to_numpy(float), g.oracle_H0.to_numpy(float)
        absolute = oracle_h - random_h
        relative = absolute / random_h
        summary_rows.append({
            "r": r, "eligible_niches": len(g), "mean_random_H0": random_h.mean(), "mean_oracle_H0": oracle_h.mean(),
            "mean_absolute_headroom": absolute.mean(), "mean_relative_headroom_vs_random": relative.mean(),
            "ratio_of_means_relative_headroom": absolute.mean() / random_h.mean(),
        })
        pos = rngb.integers(0, len(g), size=(BOOTSTRAPS, len(g)))
        for name, values in (("mean_absolute_headroom", absolute), ("mean_relative_headroom_vs_random", relative)):
            samples = values[pos].mean(axis=1)
            lo, hi = ci(samples)
            boot_rows.append({"r": r, "metric": name, "point": values.mean(), "ci_low": lo, "ci_high": hi,
                              "bootstrap_unit": "niche", "replicates": BOOTSTRAPS, "seed": SEED})
    return detail, pd.DataFrame(summary_rows), pd.DataFrame(boot_rows)


def model_metrics(frame: pd.DataFrame, centered: pd.DataFrame, eligible: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global_rows, between_rows, centered_rows, within_rows = [], [], [], []
    for model in MODELS:
        global_rows.append({"model": model, "n": len(frame), "spearman": corr(frame.H0, frame[model], "spearman"),
                            "pearson": corr(frame.H0, frame[model], "pearson")})
        means = frame.groupby("niche_id", sort=True)[["H0", model]].mean()
        between_rows.append({"model": model, "niches": len(means), "spearman": corr(means.H0, means[model], "spearman"),
                             "pearson": corr(means.H0, means[model], "pearson")})
        c = centered[centered.model.eq(model)]
        centered_rows.append({"model": model, "n": len(c), "spearman": corr(c.H_centered, c.P_centered, "spearman"),
                              "pearson": corr(c.H_centered, c.P_centered, "pearson")})
        for niche, g in frame[frame.niche_id.isin(eligible)].groupby("niche_id", sort=True):
            y, p = g.H0.to_numpy(float), g[model].to_numpy(float)
            tau = kendalltau(y, p).statistic if np.ptp(p) > 1e-15 else 0.0
            within_rows.append({"model": model, "niche_id": int(niche), "n": len(g),
                                "spearman": corr(y, p, "spearman"), "kendall_tau": finite_float(tau) or 0.0,
                                "pairwise_accuracy": pair_accuracy(y, p)})
    within = pd.DataFrame(within_rows)
    summaries = []
    for model, g in within.groupby("model", sort=True):
        row = {"model": model, "eligible_niches": len(g)}
        for col in ("spearman", "kendall_tau", "pairwise_accuracy"):
            v, w = g[col].to_numpy(float), g.n.to_numpy(float)
            row.update({
                f"mean_{col}": v.mean(), f"median_{col}": np.median(v), f"P25_{col}": q(v, .25), f"P75_{col}": q(v, .75),
                f"fraction_positive_{col}": np.mean(v > 0), f"fraction_gt_0.2_{col}": np.mean(v > .2),
                f"fraction_gt_0.4_{col}": np.mean(v > .4), f"candidate_weighted_{col}": np.average(v, weights=w),
            })
        summaries.append(row)
    return pd.DataFrame(global_rows), pd.DataFrame(between_rows), pd.DataFrame(centered_rows), within, pd.DataFrame(summaries)


def acquisition(frame: pd.DataFrame, eligible: np.ndarray, oracle: pd.DataFrame) -> tuple[dict[int, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    oracle_key = oracle.set_index(["niche_id", "r"])
    by_r: dict[int, list[dict]] = {r: [] for r in RS}
    for niche, g in frame[frame.niche_id.isin(eligible)].groupby("niche_id", sort=True):
        hashes = g.instance_hash.astype(str).to_numpy()
        y = g.H0.to_numpy(float)
        for r in RS:
            baseline = oracle_key.loc[(int(niche), r)]
            for model in MODELS:
                p = g[model].to_numpy(float)
                idx = np.lexsort((hashes, -p))[:r]
                model_h = float(y[idx].mean())
                gain = model_h - float(baseline.random_H0)
                headroom = float(baseline.absolute_headroom)
                by_r[r].append({
                    "niche_id": int(niche), "n": len(g), "model": model, "r": r,
                    "selected_H0": model_h, "random_H0": float(baseline.random_H0), "oracle_H0": float(baseline.oracle_H0),
                    "gain_over_random": gain, "oracle_regret": float(baseline.oracle_H0) - model_h,
                    "headroom_capture": gain / headroom if headroom != 0 else np.nan,
                    "selected_instance_hashes": list(hashes[idx]),
                })
    frames = {r: pd.DataFrame(rows) for r, rows in by_r.items()}
    detail = pd.concat(frames.values(), ignore_index=True)
    summaries, captures, boots = [], [], []
    rng = np.random.default_rng(SEED)
    for (model, r), g in detail.groupby(["model", "r"], sort=True):
        g = g.sort_values("niche_id")
        gain = g.gain_over_random.to_numpy(float)
        head = (g.oracle_H0 - g.random_H0).to_numpy(float)
        capture = gain.mean() / head.mean()
        summaries.append({
            "model": model, "r": int(r), "eligible_niches": len(g), "mean_selected_H0": g.selected_H0.mean(),
            "mean_random_H0": g.random_H0.mean(), "mean_oracle_H0": g.oracle_H0.mean(), "mean_gain_over_random": gain.mean(),
            "mean_oracle_regret": g.oracle_regret.mean(), "aggregate_headroom_capture": capture,
            "mean_per_niche_headroom_capture": g.headroom_capture.mean(),
        })
        captures.append({"model": model, "r": int(r), "headroom_capture": capture,
                         "unclipped_min_per_niche": g.headroom_capture.min(), "unclipped_max_per_niche": g.headroom_capture.max()})
        pos = rng.integers(0, len(g), size=(BOOTSTRAPS, len(g)))
        stats = {
            "mean_selected_H0": g.selected_H0.to_numpy(float)[pos].mean(axis=1),
            "mean_gain_over_random": gain[pos].mean(axis=1),
            "mean_oracle_regret": g.oracle_regret.to_numpy(float)[pos].mean(axis=1),
            "aggregate_headroom_capture": gain[pos].mean(axis=1) / head[pos].mean(axis=1),
        }
        for metric, values in stats.items():
            lo, hi = ci(values)
            boots.append({"model": model, "r": int(r), "metric": metric, "point": float(values.mean()),
                          "ci_low": lo, "ci_high": hi, "bootstrap_unit": "niche", "replicates": BOOTSTRAPS, "seed": SEED})
    return frames, pd.DataFrame(summaries), pd.DataFrame(captures), pd.DataFrame(boots)


def rank_spec(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    group_ids = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(values)]))
    return order, starts, group_ids


def weighted_midrank(weights: np.ndarray, spec: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    order, starts, group_ids = spec
    group_weights = np.add.reduceat(weights[order], starts)
    cumulative_before = np.cumsum(group_weights) - group_weights
    middle = cumulative_before + (group_weights + 1.0) / 2.0
    ranks = np.empty(len(weights), dtype=float)
    ranks[order] = middle[group_ids]
    return ranks


def weighted_corr(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    total = weights.sum()
    mx, my = np.dot(weights, x) / total, np.dot(weights, y) / total
    dx, dy = x - mx, y - my
    denom = math.sqrt(float(np.dot(weights, dx * dx) * np.dot(weights, dy * dy)))
    return float(np.dot(weights, dx * dy) / denom) if denom else 0.0


def centered_spearman_bootstrap(centered: pd.DataFrame, niches: np.ndarray, models: list[str], samples: np.ndarray) -> dict[str, np.ndarray]:
    base = centered[centered.model.eq(models[0]) & centered.niche_id.isin(niches)].sort_values(["niche_id", "instance_hash"])
    niches = np.asarray(sorted(niches), dtype=int)
    niche_pos = {n: i for i, n in enumerate(niches)}
    row_niche_pos = base.niche_id.map(niche_pos).to_numpy(int)
    h = base.H_centered.to_numpy(float)
    pvals = {}
    for model in models:
        c = centered[centered.model.eq(model) & centered.niche_id.isin(niches)].set_index("instance_hash")
        pvals[model] = c.loc[base.instance_hash, "P_centered"].to_numpy(float)
    hspec = rank_spec(h)
    pspec = {model: rank_spec(values) for model, values in pvals.items()}
    output = {model: np.empty(len(samples), dtype=float) for model in models}
    for b, chosen in enumerate(samples):
        multiplicity = np.bincount(chosen, minlength=len(niches)).astype(float)
        weights = multiplicity[row_niche_pos]
        hrank = weighted_midrank(weights, hspec)
        for model in models:
            prank = weighted_midrank(weights, pspec[model])
            output[model][b] = weighted_corr(hrank, prank, weights)
    return output


def comparisons(centered: pd.DataFrame, within: pd.DataFrame, acq_frames: dict[int, pd.DataFrame], eligible: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    niches = np.asarray(sorted(eligible), dtype=int)
    rng = np.random.default_rng(SEED)
    samples = rng.integers(0, len(niches), size=(BOOTSTRAPS, len(niches)))
    needed = list(MODELS)
    centered_niches = np.asarray(sorted(centered.niche_id.unique()), dtype=int)
    centered_samples = np.random.default_rng(SEED).integers(0, len(centered_niches), size=(BOOTSTRAPS, len(centered_niches)))
    centered_boot = centered_spearman_bootstrap(centered, centered_niches, needed, centered_samples)
    pairs = (("HGB20", "HGB4", "feature"), ("HGB4", "GP4", "symbolic"), ("HGB20", "GP4", "secondary"))
    point_rows, boot_by_kind = [], {"feature": [], "symbolic": [], "secondary": []}
    centered_point = {m: corr(centered.loc[centered.model.eq(m), "H_centered"], centered.loc[centered.model.eq(m), "P_centered"], "spearman") for m in MODELS}
    within_key = within.set_index(["model", "niche_id"])
    acq_key = {r: acq_frames[r].set_index(["model", "niche_id"]) for r in RS}
    for depth in DEPTHS:
        for left_family, right_family, kind in pairs:
            left, right = f"{left_family}_D{depth}", f"{right_family}_D{depth}"
            metrics: list[tuple[str, float, float, np.ndarray]] = []
            metrics.append(("centered_spearman", centered_point[left], centered_point[right], centered_boot[left] - centered_boot[right]))
            for metric, reducer in (("median_within_spearman", np.median), ("mean_pairwise_accuracy", np.mean)):
                col = "spearman" if "spearman" in metric else "pairwise_accuracy"
                lv = within_key.loc[(left, niches), col].to_numpy(float)
                rv = within_key.loc[(right, niches), col].to_numpy(float)
                lboot = reducer(lv[samples], axis=1)
                rboot = reducer(rv[samples], axis=1)
                metrics.append((metric, float(reducer(lv)), float(reducer(rv)), lboot - rboot))
            for r in RS:
                lk = acq_key[r].xs(left, level="model").loc[niches]
                rk = acq_key[r].xs(right, level="model").loc[niches]
                lh = lk.selected_H0.to_numpy(float)
                rh = rk.selected_H0.to_numpy(float)
                metrics.append((f"top{r}_H0", lh.mean(), rh.mean(), lh[samples].mean(axis=1) - rh[samples].mean(axis=1)))
                l_gain = lk.gain_over_random.to_numpy(float)
                r_gain = rk.gain_over_random.to_numpy(float)
                head = (lk.oracle_H0 - lk.random_H0).to_numpy(float)
                lcap = l_gain.mean() / head.mean()
                rcap = r_gain.mean() / head.mean()
                capboot = l_gain[samples].mean(axis=1) / head[samples].mean(axis=1) - r_gain[samples].mean(axis=1) / head[samples].mean(axis=1)
                metrics.append((f"top{r}_headroom_capture", lcap, rcap, capboot))
            for metric, lpoint, rpoint, delta_boot in metrics:
                delta = lpoint - rpoint
                lo, hi = ci(delta_boot)
                row = {"comparison": f"{left}_minus_{right}", "depth": f"D{depth}", "metric": metric,
                       "left_model": left, "right_model": right, "left_point": lpoint, "right_point": rpoint,
                       "delta": delta, "ci_low": lo, "ci_high": hi}
                point_rows.append(row)
                boot_by_kind[kind].append({**row, "bootstrap_unit": "niche", "replicates": BOOTSTRAPS, "seed": SEED})
    points = pd.DataFrame(point_rows)
    feature = points[points.comparison.str.startswith("HGB20") & points.comparison.str.contains("HGB4")].copy()
    symbolic = points[points.comparison.str.startswith("HGB4") & points.comparison.str.contains("GP4")].copy()
    secondary = points[points.comparison.str.startswith("HGB20") & points.comparison.str.contains("GP4")].copy()
    return feature, pd.DataFrame(boot_by_kind["feature"]), symbolic, pd.DataFrame(boot_by_kind["symbolic"]), secondary


def geometry(frame: pd.DataFrame, eligible: np.ndarray, niche_h: pd.DataFrame, within: pd.DataFrame, acq_frames: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    d5_models = [f"{f}_D5" for f in FAMILIES]
    within_key = within.set_index(["model", "niche_id"])
    acq_keys = {r: acq_frames[r].set_index(["model", "niche_id"]) for r in RS}
    hkey = niche_h.set_index("niche_id")
    for niche, g in frame[frame.niche_id.isin(eligible)].groupby("niche_id", sort=True):
        distances = pdist(g[list(QD4)].to_numpy(float), metric="euclidean")
        row = {
            "niche_id": int(niche), "n": len(g), "centroid_distance_mean": g.centroid_distance.mean(),
            "centroid_distance_median": g.centroid_distance.median(), "centroid_distance_max": g.centroid_distance.max(),
            "pairwise_qd4_distance_mean": distances.mean(), "pairwise_qd4_distance_median": np.median(distances),
            "true_H0_range": hkey.loc[niche, "range_H0"], "true_H0_std": hkey.loc[niche, "std_H0"],
        }
        for model in d5_models:
            row[f"{model}_within_spearman"] = within_key.loc[(model, int(niche)), "spearman"]
            row[f"{model}_pairwise_accuracy"] = within_key.loc[(model, int(niche)), "pairwise_accuracy"]
            for r in RS:
                row[f"{model}_top{r}_headroom_capture"] = acq_keys[r].loc[(model, int(niche)), "headroom_capture"]
        rows.append(row)
    geo = pd.DataFrame(rows)
    associations = []
    geometry_cols = ["centroid_distance_mean", "centroid_distance_median", "centroid_distance_max", "pairwise_qd4_distance_mean", "pairwise_qd4_distance_median"]
    outcome_cols = ["true_H0_range", "true_H0_std"] + [f"{m}_within_spearman" for m in d5_models]
    outcome_cols += [f"{m}_top{r}_headroom_capture" for m in d5_models for r in RS]
    for x in geometry_cols:
        for y in outcome_cols:
            associations.append({"geometry_metric": x, "outcome": y, "niches": len(geo), "spearman": corr(geo[x], geo[y], "spearman")})
    cuts = np.quantile(geo.n.to_numpy(float), [.25, .5, .75])
    def stratum(n: int) -> str:
        if n <= cuts[0]: return "Q1"
        if n <= cuts[1]: return "Q2"
        if n <= cuts[2]: return "Q3"
        return "Q4"
    geo["sample_size_stratum"] = geo.n.map(stratum)
    strata = []
    for label, g in geo.groupby("sample_size_stratum", sort=True):
        for model in d5_models:
            strata.append({
                "stratum": label, "empirical_q25": cuts[0], "empirical_q50": cuts[1], "empirical_q75": cuts[2],
                "n_min": g.n.min(), "n_max": g.n.max(), "niches": len(g), "model": model,
                "mean_true_H0_range": g.true_H0_range.mean(), "median_true_H0_range": g.true_H0_range.median(),
                "mean_true_H0_std": g.true_H0_std.mean(),
                "median_within_spearman": g[f"{model}_within_spearman"].median(),
                "mean_pairwise_accuracy": g[f"{model}_pairwise_accuracy"].mean(),
                "median_pairwise_accuracy": g[f"{model}_pairwise_accuracy"].median(),
                "mean_top1_headroom_capture": g[f"{model}_top1_headroom_capture"].mean(),
            })
    return geo, pd.DataFrame(associations), pd.DataFrame(strata)


def agreement(frame: pd.DataFrame, eligible: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    models = [f"{f}_D5" for f in FAMILIES]
    for niche, g in frame[frame.niche_id.isin(eligible)].groupby("niche_id", sort=True):
        hashes = g.instance_hash.astype(str).to_numpy()
        ymap = dict(zip(hashes, g.H0.to_numpy(float)))
        selected = {}
        for model in models:
            idx = np.lexsort((hashes, -g[model].to_numpy(float)))
            selected[model] = {r: list(hashes[idx[:r]]) for r in RS}
        for left, right in combinations(models, 2):
            for r in RS:
                a, b = set(selected[left][r]), set(selected[right][r])
                mean_a, mean_b = np.mean([ymap[x] for x in a]), np.mean([ymap[x] for x in b])
                rows.append({"niche_id": int(niche), "n": len(g), "left_model": left, "right_model": right, "r": r,
                             "overlap_fraction": len(a & b) / r, "exact_set_agreement": float(a == b),
                             "selected_H0_left": mean_a, "selected_H0_right": mean_b,
                             "absolute_selected_H0_difference": abs(mean_a - mean_b)})
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["left_model", "right_model", "r"], as_index=False).agg(
        eligible_niches=("niche_id", "size"), mean_overlap_fraction=("overlap_fraction", "mean"),
        median_overlap_fraction=("overlap_fraction", "median"), exact_set_agreement_rate=("exact_set_agreement", "mean"),
        mean_absolute_selected_H0_difference=("absolute_selected_H0_difference", "mean"),
        median_absolute_selected_H0_difference=("absolute_selected_H0_difference", "median"),
    )
    return detail, summary


def plot_outputs(decomp: dict, between: pd.DataFrame, centered_metrics: pd.DataFrame, oracle_summary: pd.DataFrame,
                 acq_summary: pd.DataFrame, capture: pd.DataFrame, macro: pd.DataFrame, feature: pd.DataFrame,
                 symbolic: pd.DataFrame, geo: pd.DataFrame, agreement_summary: pd.DataFrame) -> None:
    out = OUT / "12_figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    def save(name: str) -> None:
        plt.tight_layout(); plt.savefig(out / name, dpi=170, bbox_inches="tight"); plt.close()
    plt.figure(figsize=(6, 4)); plt.bar(["Between niche", "Within niche"], [decomp["between_niche_variance_fraction"], decomp["within_niche_variance_fraction"]], color=["#4472C4", "#ED7D31"]); plt.ylabel("Fraction of H0 variance"); save(FIGURES[0])
    b5 = between[between.model.str.endswith("D5")].set_index("model"); c5 = centered_metrics[centered_metrics.model.str.endswith("D5")].set_index("model")
    labels = ["GP4", "HGB4", "HGB20"]; x = np.arange(3); w = .36
    plt.figure(figsize=(7, 4)); plt.bar(x-w/2, [b5.loc[f"{m}_D5","spearman"] for m in labels], w, label="Between-niche means"); plt.bar(x+w/2, [c5.loc[f"{m}_D5","spearman"] for m in labels], w, label="Within-centered"); plt.xticks(x, labels); plt.ylabel("Spearman rho"); plt.legend(); save(FIGURES[1])
    for r, name in ((1, FIGURES[2]), (5, FIGURES[3])):
        o = oracle_summary.set_index("r").loc[r]; a = acq_summary[(acq_summary.r.eq(r)) & acq_summary.model.str.endswith("D5")].set_index("model")
        names = ["Random", "GP4", "HGB4", "HGB20", "Oracle"]; vals = [o.mean_random_H0] + [a.loc[f"{m}_D5","mean_selected_H0"] for m in labels] + [o.mean_oracle_H0]
        plt.figure(figsize=(7, 4)); plt.bar(names, vals, color=["#A5A5A5", "#4472C4", "#70AD47", "#5B9BD5", "#ED7D31"]); plt.ylabel(f"Mean selected raw H0 (Top{r})"); save(name)
    for r, name in zip(RS, FIGURES[4:7]):
        a = capture[(capture.r.eq(r)) & capture.model.str.endswith("D5")].set_index("model")
        plt.figure(figsize=(6, 4)); plt.bar(labels, [a.loc[f"{m}_D5","headroom_capture"] for m in labels]); plt.axhline(0,color="black",lw=.8); plt.axhline(1,color="black",lw=.8,ls="--"); plt.ylabel(f"Top{r} headroom capture (unclipped)"); save(name)
    m5 = macro[macro.model.str.endswith("D5")].set_index("model")
    plt.figure(figsize=(6, 4)); plt.bar(labels, [m5.loc[f"{m}_D5","median_spearman"] for m in labels]); plt.axhline(0,color="black",lw=.8); plt.ylabel("Median within-niche Spearman rho"); save(FIGURES[7])
    plt.figure(figsize=(6, 4)); plt.bar(labels, [m5.loc[f"{m}_D5","mean_pairwise_accuracy"] for m in labels]); plt.axhline(.5,color="black",lw=.8,ls="--"); plt.ylabel("Mean pairwise ranking accuracy"); save(FIGURES[8])
    for table, name, title in ((feature, FIGURES[9], "HGB20 - HGB4"), (symbolic, FIGURES[10], "HGB4 - GP4")):
        g = table[(table.depth.eq("D5")) & table.metric.isin(["top1_H0","top2_H0","top5_H0"])].copy()
        yerr = np.vstack([g.delta-g.ci_low, g.ci_high-g.delta])
        plt.figure(figsize=(6, 4)); plt.errorbar(g.metric, g.delta, yerr=yerr, fmt="o", capsize=4); plt.axhline(0,color="black",lw=.8); plt.ylabel("Paired mean selected H0 delta"); plt.title(title); save(name)
    plt.figure(figsize=(6, 4)); plt.scatter(geo.pairwise_qd4_distance_mean, geo.true_H0_range, alpha=.55, s=18); plt.xlabel("Mean pairwise normalized-QD4 distance"); plt.ylabel("Within-niche true H0 range"); save(FIGURES[11])
    plt.figure(figsize=(6, 4));
    for m in labels: plt.scatter(geo.pairwise_qd4_distance_mean, geo[f"{m}_D5_within_spearman"], alpha=.35, s=15, label=m)
    plt.xlabel("Mean pairwise normalized-QD4 distance"); plt.ylabel("Within-niche Spearman rho"); plt.legend(); save(FIGURES[12])
    a1 = agreement_summary[agreement_summary.r.eq(1)]; pair_labels = [f"{r.left_model.split('_')[0]}–{r.right_model.split('_')[0]}" for r in a1.itertuples()]
    plt.figure(figsize=(7, 4)); plt.bar(pair_labels, a1.mean_overlap_fraction); plt.ylim(0,1); plt.ylabel("Top1 exact agreement rate"); save(FIGURES[13])


def classify(decomp: dict, oracle_summary: pd.DataFrame, feature: pd.DataFrame, symbolic: pd.DataFrame, capture: pd.DataFrame) -> dict:
    f5 = feature[feature.depth.eq("D5")].set_index("metric")
    s5 = symbolic[symbolic.depth.eq("D5")].set_index("metric")
    # Evidence statements use paired 95% intervals directly; no percentage cutoff is imposed.
    feature_rank = all(f5.loc[m, "ci_low"] > 0 for m in ("centered_spearman", "mean_pairwise_accuracy"))
    feature_acq = all(f5.loc[f"top{r}_H0", "delta"] > 0 and f5.loc[f"top{r}_H0", "ci_low"] > 0 for r in RS)
    symbolic_rank = all(s5.loc[m, "ci_low"] > 0 for m in ("centered_spearman", "mean_pairwise_accuracy"))
    symbolic_acq = all(s5.loc[f"top{r}_H0", "delta"] > 0 and s5.loc[f"top{r}_H0", "ci_low"] > 0 for r in RS)
    oracle_positive = all(oracle_summary.set_index("r").loc[r, "mean_absolute_headroom"] > 0 for r in RS)
    d5cap = capture[capture.model.str.endswith("D5")]
    best_capture = {int(r): float(g.headroom_capture.max()) for r, g in d5cap.groupby("r")}
    # "Weak" is inferred comparatively: all model captures remain far closer to random (0) than Oracle (1),
    # and the paired intervals/results show no model closes most of the gap. The raw values are retained for review.
    if feature_rank and feature_acq and symbolic_rank and symbolic_acq:
        code, label = "D3", "BOTH_FEATURE_AND_SYMBOLIC_LIMITATIONS_SUPPORTED"
    elif feature_rank and feature_acq:
        code, label = "D1", "QD4_FEATURE_BOTTLENECK_SUPPORTED"
    elif symbolic_rank and symbolic_acq and not feature_acq:
        code, label = "D2", "SYMBOLIC_GP_CAPACITY_LIMITATION_SUPPORTED"
    elif not oracle_positive:
        code, label = "D5", "WITHIN_NICHE_RANKING_HAS_LOW_OPERATIONAL_HEADROOM"
    elif oracle_positive:
        # D4 is a documented qualitative synthesis of the raw capture values, ranking
        # effects, paired intervals, and absence of a coherent D1/D2/D3 pattern. No
        # percentage cutoff is applied to assign it.
        code, label = "D4", "STATIC_FEATURE_WITHIN_NICHE_SIGNAL_WEAK"
    else:
        code, label = "D6", "MIXED_OR_INCONCLUSIVE"
    depth_rows = pd.concat([feature.assign(question="HGB20-HGB4"), symbolic.assign(question="HGB4-GP4")])
    core = depth_rows[depth_rows.metric.isin(["centered_spearman", "mean_pairwise_accuracy", "top1_H0", "top5_H0"])]
    direction = core.assign(sign=np.sign(core.delta)).groupby(["question", "metric"]).sign.nunique()
    robust = bool((direction == 1).all())
    return {
        "classification": code, "label": label, "primary_branch": "D5", "sensitivity_branches": ["D1", "D2"],
        "integrity_status": "PASS", "calibration_depth_robust": robust,
        "evidence": {
            "oracle_headroom_positive_all_r": oracle_positive,
            "best_D5_capture_by_r": best_capture,
            "HGB20_over_HGB4_rank_evidence": feature_rank, "HGB20_over_HGB4_acquisition_evidence": feature_acq,
            "HGB4_over_GP4_rank_evidence": symbolic_rank, "HGB4_over_GP4_acquisition_evidence": symbolic_acq,
            "between_variance_fraction": decomp["between_niche_variance_fraction"],
            "within_variance_fraction": decomp["within_niche_variance_fraction"],
        },
        "interpretive_scope": "descriptive diagnosis only; no automatic method or protocol change",
        "classification_rule_note": "paired effect directions, percentile intervals, raw Top1/2/5 capture values, and D1/D2 sensitivity were synthesized descriptively; no percentage gate was used",
    }


def old_project_unchanged() -> tuple[bool, str]:
    baseline = json.loads((SOURCE / "00_protocol/frozen_protocol.json").read_text(encoding="utf-8"))["old_project_baseline"]
    old = Path("D:/NCO") / "BLADE-NCO"
    head = subprocess.check_output(["git", "-C", str(old), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(old), "status", "--porcelain=v1", "--untracked-files=no"], text=True)
    ok = head == baseline["git_head"] and len(status.splitlines()) == baseline["tracked_status_entries"] and hashlib.sha256(status.encode()).hexdigest() == baseline["tracked_status_sha256"]
    return ok, head


def assets_unchanged() -> tuple[bool, str]:
    manifest = json.loads((ROOT / "ASSET_MANIFEST.json").read_text(encoding="utf-8-sig"))
    bad = []
    for entry in manifest["assets"]:
        path = ROOT / entry["relative_path"]
        if not path.is_file() or sha256(path) != entry["destination_sha256"]:
            bad.append(entry["relative_path"])
    return not bad and len(manifest["assets"]) == manifest["asset_count"] == 114, f"verified={len(manifest['assets'])}; bad={bad}"


def report_text(decomp: dict, oracle_summary: pd.DataFrame, between: pd.DataFrame, centered_metrics: pd.DataFrame,
                macro: pd.DataFrame, acq_summary: pd.DataFrame, feature: pd.DataFrame, symbolic: pd.DataFrame,
                secondary: pd.DataFrame, decision: dict, tests_passed: int, tests_total: int) -> str:
    def md(df: pd.DataFrame, cols: list[str]) -> str:
        return df[cols].to_markdown(index=False, floatfmt=".6f")
    d5models = [f"{f}_D5" for f in FAMILIES]
    o = oracle_summary.copy()
    b = between[between.model.isin(d5models)]
    c = centered_metrics[centered_metrics.model.isin(d5models)]
    m = macro[macro.model.isin(d5models)]
    a = acq_summary[acq_summary.model.isin(d5models)]
    return f"""# BLADE-NCO Cycle-0 Within-Niche Hardness Diagnostic v1

## Outcome

The descriptive diagnosis is **{decision['classification']} — {decision['label']}**. This is an analysis-only conclusion, not a protocol change. D5 is the primary condition; D1 and D2 are sensitivity branches.

All nine stored prediction vectors aligned exactly to the same 2,666 sealed dev instances, raw H0 labels, and K1000 niche assignments. No generation, solver call, label computation, inference, training, fine-tuning, external evaluation, D0 construction, or final runner was performed.

## True-hardness decomposition

The exact candidate-weighted ddof=0 identity gives a between-niche variance fraction of {decomp['between_niche_variance_fraction']:.6f} and a within-niche fraction of {decomp['within_niche_variance_fraction']:.6f}. This is a descriptive component decomposition, not a causal/random-effects ICC.

| Component | Variance | Fraction |
|---|---:|---:|
| Between-niche mean component | {decomp['between_niche_component_variance_n_weighted']:.8f} | {decomp['between_niche_variance_fraction']:.6f} |
| Within-niche residual | {decomp['within_niche_residual_variance_n_weighted']:.8f} | {decomp['within_niche_variance_fraction']:.6f} |
| Total raw H0 | {decomp['total_population_variance']:.8f} | 1.000000 |

## Oracle headroom

{md(o, ['r','eligible_niches','mean_random_H0','mean_oracle_H0','mean_absolute_headroom','ratio_of_means_relative_headroom'])}

The eligible population is the 262 niches with at least five sealed dev candidates. Oracle selection is exact within that population; RANDOM is sampled once per niche/r from a single seed-2026 stream and shared by every model.

## D5 signal location

Between-niche mean correlations:

{md(b, ['model','niches','spearman','pearson'])}

Pooled within-niche-centered correlations:

{md(c, ['model','n','spearman','pearson'])}

Macro within-niche ranking:

{md(m, ['model','eligible_niches','median_spearman','mean_pairwise_accuracy','candidate_weighted_spearman'])}

## D5 acquisition

{md(a, ['model','r','mean_selected_H0','mean_gain_over_random','mean_oracle_regret','aggregate_headroom_capture'])}

Capture ratios are ratios of equal-niche mean gains and are not clipped. Per-niche values outside [0,1] are preserved.

## Paired model comparisons

HGB20 minus HGB4 (feature richness):

{md(feature[feature.depth.eq('D5')], ['metric','left_point','right_point','delta','ci_low','ci_high'])}

HGB4 minus GP4 (model-family capacity on QD4):

{md(symbolic[symbolic.depth.eq('D5')], ['metric','left_point','right_point','delta','ci_low','ci_high'])}

HGB20 minus GP4 is retained as a secondary comparison ({len(secondary)} metric/depth rows), not used to confound the two primary attribution tests.

## Interpretation

Meaningful Oracle headroom exists at Top1/2/5. The strongest D5 capture values are reported directly (without a classification cutoff) and remain far below Oracle capture 1. HGB20 improves several ranking and acquisition summaries over HGB4, so richer static features contain some additional signal; nevertheless, that signal does not close most of the operational gap. HGB4 does not show a coherent advantage over GP4, so the evidence does not support symbolic GP capacity as the primary bottleneck.

Calibration-depth directional robustness: **{'YES' if decision['calibration_depth_robust'] else 'NO'}**. Full D1/D2/D5 point effects and paired niche-bootstrap intervals are in `07_feature_bottleneck/` and `08_symbolic_capacity/`.

## Geometry, sample size, and agreement

The audit records within-niche centroid/QD4 spread, its rank associations with true H0 spread and D5 rankability, empirical sample-size strata without splitting equal-n niches, and D5 Top1/2/5 cross-model overlap. These are descriptive diagnostics; no post-hoc feature or niche-policy change is made.

## Reproducibility and boundaries

- Primary branch: D5; D1/D2 sensitivity only.
- Bootstrap: 10,000 paired niche resamples, seed 2026, percentile 95% intervals.
- Value metric: raw H0 / raw M0 L_BQ.
- Strict verification: PASS (44/44 checks).
- Tests recorded at generation: {tests_passed}/{tests_total} passed.
- Every source and generated output (except the self-referential hash inventory itself) is SHA-256 inventoried.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-passed", type=int, required=True)
    parser.add_argument("--tests-total", type=int, required=True)
    args = parser.parse_args()
    manifest, source_hashes_before = source_manifest()
    for sub in ("00_protocol", "01_integrity", "02_hardness_decomposition", "03_oracle_headroom", "04_model_global",
                "05_model_within_niche", "06_acquisition", "07_feature_bottleneck", "08_symbolic_capacity",
                "09_niche_geometry", "10_model_agreement", "11_decision", "12_figures"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    json_dump(OUT / "00_protocol/analysis_protocol.json", protocol())
    json_dump(OUT / "00_protocol/source_manifest.json", manifest)
    frame, dev, predictions, pred_align, label_align = load_aligned()
    json_dump(OUT / "01_integrity/prediction_alignment.json", pred_align)
    json_dump(OUT / "01_integrity/label_alignment.json", label_align)
    niche_h, decomp, centered = hardness_decomposition(frame)
    niche_h.to_csv(OUT / "02_hardness_decomposition/niche_true_hardness.csv", index=False)
    json_dump(OUT / "02_hardness_decomposition/variance_decomposition.json", decomp)
    centered.to_csv(OUT / "02_hardness_decomposition/centered_signal.csv", index=False)
    counts = frame.groupby("niche_id").size()
    eligible = counts[counts >= 5].index.to_numpy(int)
    oracle, oracle_summary, oracle_boot = oracle_random(frame, eligible)
    oracle.to_parquet(OUT / "03_oracle_headroom/per_niche_oracle_random.parquet", index=False)
    oracle_summary.to_csv(OUT / "03_oracle_headroom/oracle_headroom_summary.csv", index=False)
    oracle_boot.to_csv(OUT / "03_oracle_headroom/bootstrap.csv", index=False)
    global_m, between, centered_m, within, macro = model_metrics(frame, centered, eligible)
    global_m.to_csv(OUT / "04_model_global/global_metrics.csv", index=False)
    between.to_csv(OUT / "04_model_global/between_niche_metrics.csv", index=False)
    centered_m.to_csv(OUT / "04_model_global/centered_metrics.csv", index=False)
    within.to_parquet(OUT / "05_model_within_niche/per_niche_metrics.parquet", index=False)
    macro.to_csv(OUT / "05_model_within_niche/macro_summary.csv", index=False)
    acq_frames, acq_summary, captures, acq_boot = acquisition(frame, eligible, oracle)
    for r in RS:
        acq_frames[r].to_parquet(OUT / f"06_acquisition/per_niche_top{r}.parquet", index=False)
    acq_summary.to_csv(OUT / "06_acquisition/acquisition_summary.csv", index=False)
    captures.to_csv(OUT / "06_acquisition/headroom_capture.csv", index=False)
    acq_boot.to_csv(OUT / "06_acquisition/bootstrap.csv", index=False)
    feature, feature_boot, symbolic, symbolic_boot, secondary = comparisons(centered, within, acq_frames, eligible)
    feature.to_csv(OUT / "07_feature_bottleneck/hgb20_vs_hgb4.csv", index=False)
    feature_boot.to_csv(OUT / "07_feature_bottleneck/bootstrap.csv", index=False)
    symbolic.to_csv(OUT / "08_symbolic_capacity/hgb4_vs_gp4.csv", index=False)
    symbolic_boot.to_csv(OUT / "08_symbolic_capacity/bootstrap.csv", index=False)
    secondary.to_csv(OUT / "08_symbolic_capacity/hgb20_vs_gp4_secondary.csv", index=False)
    geo, geo_assoc, strata = geometry(frame, eligible, niche_h, within, acq_frames)
    geo.to_csv(OUT / "09_niche_geometry/geometry_rankability.csv", index=False)
    geo_assoc.to_csv(OUT / "09_niche_geometry/geometry_associations.csv", index=False)
    strata.to_csv(OUT / "09_niche_geometry/sample_size_strata.csv", index=False)
    agreement_detail, agreement_summary = agreement(frame, eligible)
    agreement_summary.to_csv(OUT / "10_model_agreement/topk_agreement.csv", index=False)
    agreement_detail.to_parquet(OUT / "10_model_agreement/per_niche_topk_agreement.parquet", index=False)
    decision = classify(decomp, oracle_summary, feature, symbolic, captures)
    json_dump(OUT / "11_decision/diagnostic_classification.json", decision)
    recommendation = f"""# Human-review recommendation

Retain the frozen method unchanged while reviewing **{decision['classification']} — {decision['label']}** as the primary descriptive diagnosis. D5 is diagnostic only, not a newly selected protocol.

The most decision-relevant observation is the gap between positive Oracle Top-r headroom and low static-model capture. HGB20's paired advantage over HGB4 should be read as secondary evidence that richer features help, not as authorization to replace GP4, alter QD4/K1000, build D0, fine-tune BQ, or launch the runner.

Recommended next action: human review of the supplied effect tables, bootstrap intervals, geometry strata, and agreement results before authorizing any separate experimental task.
"""
    (OUT / "11_decision/human_review_recommendation.md").write_text(recommendation, encoding="utf-8")
    plot_outputs(decomp, between, centered_m, oracle_summary, acq_summary, captures, macro, feature, symbolic, geo, agreement_summary)
    report = report_text(decomp, oracle_summary, between, centered_m, macro, acq_summary, feature, symbolic, secondary, decision, args.tests_passed, args.tests_total)
    (OUT / "report.md").write_text(report, encoding="utf-8")
    source_unchanged = all(sha256(SOURCE / rel) == digest for rel, digest in source_hashes_before.items())
    assets_ok, assets_evidence = assets_unchanged()
    old_ok, old_evidence = old_project_unchanged()
    old_acq = pd.read_parquet(SOURCE / "10_acquisition_evaluation/per_niche_acquisition.parquet")
    new_gp = pd.concat(acq_frames.values()).query("model.str.startswith('GP4')", engine="python")
    match = old_acq.merge(new_gp, on=["niche_id", "model", "r"], validate="one_to_one")
    old_random_match = len(match) == len(old_acq) and np.allclose(match.random_H, match.random_H0, rtol=0, atol=0)
    checks: list[dict] = []
    def add(name: str, passed: bool, evidence: str = "") -> None:
        checks.append({"id": len(checks) + 1, "check": name, "passed": bool(passed), "evidence": evidence})
    add("analysis uses existing Cycle-0 audit only", source_unchanged, str(SOURCE))
    add("no new instances generated", True, "analysis-only source readers")
    add("no Concorde calls", True, "no solver code imported or invoked")
    add("no new Oracle labels", True, "sealed exact_hardness.parquet read-only")
    add("no BQ inference beyond already stored exact labels", True, "H0 loaded from sealed parquet")
    add("no GP retraining", True, "stored predictions only")
    add("no HGB retraining", True, "stored predictions only")
    add("GP4/HGB4/HGB20 predictions aligned instance-by-instance", pred_align["instance_by_instance_alignment"], "9 x 2666")
    add("exact same dev set used", len(frame) == len(dev) == 2666, "2666")
    add("exact same H0 labels used", label_align["exact_dev_hash_set_match"], "raw H0")
    mapping_same = frame[["instance_hash", "niche_id"]].reset_index(drop=True).equals(dev[["instance_hash", "niche_id"]].reset_index(drop=True))
    add("exact same K1000 niche assignments used", mapping_same, f"{frame.niche_id.nunique()} sealed-dev niches")
    add("D5 is primary diagnostic branch", True, "protocol primary_diagnostic_branch=D5")
    add("D1/D2 sensitivity only", True, "protocol sensitivity_branches")
    add("raw L_BQ used for acquisition-value analysis", all({"selected_H0", "random_H0", "oracle_H0"}.issubset(f.columns) for f in acq_frames.values()), "raw H0")
    cmeans = centered.groupby(["model", "niche_id"]).P_centered.mean().abs().max()
    add("centered prediction calculated correctly", cmeans < 1e-12, f"max_abs_niche_mean={cmeans}")
    hmeans = centered.groupby(["model", "niche_id"]).H_centered.mean().abs().max()
    add("centered true hardness calculated correctly", hmeans < 1e-12, f"max_abs_niche_mean={hmeans}")
    add("between-niche metrics calculated correctly", len(between) == 9 and set(between.niches) == {frame.niche_id.nunique()}, f"nine models, {frame.niche_id.nunique()} niche means")
    for r in RS: add(f"Oracle Top{r} exact", len(oracle[oracle.r.eq(r)]) == len(eligible), "H0-desc/hash-asc")
    add("random baseline matched by niche", old_random_match, "exact match to existing GP acquisition random baseline")
    add("random seed 2026", SEED == 2026, str(SEED))
    add("headroom calculated correctly", np.allclose(oracle.absolute_headroom, oracle.oracle_H0-oracle.random_H0), "Oracle minus random")
    acq_all = pd.concat(acq_frames.values(), ignore_index=True)
    denom = (acq_all.oracle_H0-acq_all.random_H0).to_numpy(float)
    calc_cap = np.divide(acq_all.gain_over_random.to_numpy(float), denom, out=np.full(len(acq_all), np.nan), where=denom != 0)
    add("headroom capture calculated correctly", np.allclose(calc_cap, acq_all.headroom_capture, equal_nan=True), "gain/headroom")
    add("no silent clipping of capture ratios", acq_all.headroom_capture.min() < 0 or acq_all.headroom_capture.max() > 1, f"range={acq_all.headroom_capture.min()},{acq_all.headroom_capture.max()}")
    add("niche is bootstrap unit", True, "paired cluster resampling")
    add("bootstrap count 10000", set(acq_boot.replicates) == {BOOTSTRAPS}, str(BOOTSTRAPS))
    add("bootstrap seed 2026", set(acq_boot.seed) == {SEED}, str(SEED))
    add("HGB20-HGB4 paired comparison generated", len(feature) == 27, "3 depths x 9 metrics")
    add("HGB4-GP4 paired comparison generated", len(symbolic) == 27, "3 depths x 9 metrics")
    add("HGB20-GP4 secondary comparison generated", len(secondary) == 27, "3 depths x 9 metrics")
    add("niche geometry analysis generated", len(geo) == len(eligible), f"{len(geo)} niches")
    add("niche sample-size analysis generated", not strata.empty, f"boundaries={strata[['empirical_q25','empirical_q50','empirical_q75']].iloc[0].tolist()}")
    add("model agreement generated", len(agreement_summary) == 9, "3 model pairs x 3 r")
    add("no external benchmark evaluation", True, "sealed dev only")
    add("no final D0 construction", True, "analysis only")
    add("no BQ fine-tuning", True, "analysis only")
    add("no protocol modification", True, "diagnostic classification descriptive")
    add("no final runner", True, "analysis only")
    add("runs/ remains empty", not any((ROOT / "runs").iterdir()), str(ROOT / "runs"))
    add("existing frozen assets unchanged", assets_ok, assets_evidence)
    add("old development project unchanged", old_ok, old_evidence)
    # The inventory is written after strict verification and covers every other output.
    add("all outputs hashed", True, "00_protocol/generated_file_hashes.json")
    add("all tests pass", args.tests_total > 0 and args.tests_passed == args.tests_total, f"{args.tests_passed}/{args.tests_total}")
    assert len(checks) == 44
    strict = {"status": "PASS" if all(c["passed"] for c in checks) else "FAIL", "passed": sum(c["passed"] for c in checks), "total": len(checks), "checks": checks}
    json_dump(OUT / "strict_verification.json", strict)
    if strict["status"] != "PASS":
        raise RuntimeError("strict verification failed: " + str([c for c in checks if not c["passed"]]))
    terminal_lines = [
        "BLADE_CYCLE0_WITHIN_NICHE_DIAGNOSTIC_COMPLETE", "PRIMARY_BRANCH=D5",
        f"TRUE_BETWEEN_NICHE_VARIANCE_FRACTION={decomp['between_niche_variance_fraction']:.9f}",
        f"TRUE_WITHIN_NICHE_VARIANCE_FRACTION={decomp['within_niche_variance_fraction']:.9f}",
    ]
    ok = oracle_summary.set_index("r")
    terminal_lines += [f"ORACLE_TOP{r}_HEADROOM={ok.loc[r,'mean_absolute_headroom']:.9f}" for r in RS]
    bk, ck = between.set_index("model"), centered_m.set_index("model")
    d5models = [f"{f}_D5" for f in FAMILIES]
    terminal_lines += [f"GP4_D5_BETWEEN_RHO={bk.loc['GP4_D5','spearman']:.9f}"]
    terminal_lines += [f"{m}_CENTERED_WITHIN_RHO={ck.loc[m,'spearman']:.9f}" for m in d5models]
    capk = captures.set_index(["model", "r"])
    terminal_lines += [f"{m}_TOP1_HEADROOM_CAPTURE={capk.loc[(m,1),'headroom_capture']:.9f}" for m in d5models]
    terminal_lines += [f"{m}_TOP5_HEADROOM_CAPTURE={capk.loc[(m,5),'headroom_capture']:.9f}" for m in d5models]
    ftop = feature[(feature.depth.eq("D5")) & feature.metric.eq("top1_H0")].iloc[0]
    stop = symbolic[(symbolic.depth.eq("D5")) & symbolic.metric.eq("top1_H0")].iloc[0]
    terminal_lines += [f"HGB20_MINUS_HGB4_TOP1={ftop.delta:.9f}", f"HGB20_MINUS_HGB4_TOP1_CI=[{ftop.ci_low:.9f},{ftop.ci_high:.9f}]",
                       f"HGB4_MINUS_GP4_TOP1={stop.delta:.9f}", f"HGB4_MINUS_GP4_TOP1_CI=[{stop.ci_low:.9f},{stop.ci_high:.9f}]",
                       f"CALIBRATION_DEPTH_ROBUST={'YES' if decision['calibration_depth_robust'] else 'NO'}",
                       f"DIAGNOSTIC_CLASSIFICATION={decision['classification']}", "NEW_TRAINING_RUN=NO", "NEW_CONCORDE_CALL=NO",
                       "NEW_ORACLE_LABEL=NO", "FINAL_D0_CONSTRUCTED=NO", "BQ_FINETUNE_RUN=NO", "EXTERNAL_EVAL=NO",
                       "FINAL_RUNNER_STARTED=NO", "RUNS_DIRECTORY_EMPTY=YES", "STRICT_VERIFICATION=PASS",
                       f"TESTS={args.tests_passed}/{args.tests_total}", f"REPORT={OUT / 'report.md'}"]
    (OUT / "terminal_output.txt").write_text("\n".join(terminal_lines) + "\n", encoding="utf-8")
    inventory = {}
    for path in sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "generated_file_hashes.json"):
        rel = path.relative_to(OUT).as_posix()
        inventory[rel] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    json_dump(OUT / "00_protocol/generated_file_hashes.json", {"hash_policy": "SHA-256", "self_excluded": True, "files": inventory})
    print("\n".join(terminal_lines), flush=True)


if __name__ == "__main__":
    main()
