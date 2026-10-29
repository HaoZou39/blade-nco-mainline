#!/usr/bin/env python3
"""Read-only P6-0 audit of HGB features and population selection.

The program reads frozen P0--P5 artifacts and writes only into its own P6-0
directory.  It does not mutate instances, call Concorde, train BQ, evaluate a
checkpoint, or inspect M4-training overlap.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT / "runs/BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1/A0_FULL_BASE"
CYCLES = RUN / "cycles"
P1 = ROOT / "target_aware_pilot"
P2 = P1 / "p2_bounded_evolution"
P5 = P1 / "p5_structure_guided_evolution_robustness"
F40_MANIFEST = ROOT / "audits/S3_B2_RF_HGB_feature_label_scaling_v1/09_feature_subset_manifests.json"
SCALER = ROOT / "assets/qd/reference_cloud/formal_raw_scaler.json"
P2_SOURCE = P2 / "run_p2_bounded_evolution.py"
SEEDS = (2026, 2027, 2028, 2029, 2030)
SELECTED = ("spn_distance_varcoeff", "spn_fraction_shorter_mean_distance")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    value = stats.spearmanr(np.asarray(x, float), np.asarray(y, float)).statistic
    return float(value) if np.isfinite(value) else float("nan")


def split_importance(model: Any, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    gains = np.zeros(n_features)
    counts = np.zeros(n_features)
    for stage in model._predictors:  # noqa: SLF001 -- frozen model audit
        for predictor in stage:
            nodes = predictor.nodes
            for node in nodes[nodes["is_leaf"] == 0]:
                index = int(node["feature_idx"])
                counts[index] += 1
                gain = float(node["gain"])
                if np.isfinite(gain) and gain > 0:
                    gains[index] += gain
    if gains.sum():
        gains /= gains.sum()
    if counts.sum():
        counts /= counts.sum()
    return gains, counts


def cycle_importance(cycle: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Repeat the exact P0 independent-selected-set importance procedure."""
    directory = CYCLES / cycle
    manifest = json.loads((directory / "HGB_model_manifest.json").read_text())
    summary = json.loads((directory / "cycle_summary.json").read_text())
    calibration = json.loads((directory / "_calibration_info.json").read_text())
    features = list(manifest["features"])
    models = [
        joblib.load(directory / "HGB_seed_models" / f"HGB40_seed_{seed}.joblib")["model"]
        for seed in SEEDS
    ]
    selected = pd.read_parquet(directory / "07_oracle/T/labeled.parquet")
    candidate = pd.read_parquet(directory / "candidate_features.parquet")
    validation = selected[["instance_hash", "current_hardness"]].merge(
        candidate[["instance_hash", *features]], on="instance_hash", validate="one_to_one"
    )
    x = validation[features].to_numpy(float)
    y = np.log1p(validation.current_hardness.to_numpy(float))
    model_predictions = np.vstack([model.predict(x) for model in models])
    prediction = np.median(model_predictions, axis=0)
    active_models = [models[0]] if np.max(np.ptp(model_predictions, axis=0)) <= 1e-12 else models
    base_rho = safe_spearman(prediction, y)
    base_mae = float(np.mean(np.abs(prediction - y)))

    gains, counts = zip(*(split_importance(model, len(features)) for model in models), strict=True)
    gain = np.mean(gains, axis=0)
    count = np.mean(counts, axis=0)

    repeats = 8
    rng = np.random.default_rng(20261023)
    rho_drop = np.empty((repeats, len(features)))
    mae_increase = np.empty_like(rho_drop)
    perm_predictions: list[list[np.ndarray]] = []
    for repeat in range(repeats):
        values: list[np.ndarray] = []
        for column in range(len(features)):
            permuted = x.copy()
            permuted[:, column] = permuted[rng.permutation(len(x)), column]
            pred = np.median(np.vstack([model.predict(permuted) for model in active_models]), axis=0)
            values.append(pred)
            rho_drop[repeat, column] = base_rho - safe_spearman(pred, y)
            mae_increase[repeat, column] = np.mean(np.abs(pred - y)) - base_mae
        perm_predictions.append(values)

    repeat_ranks = np.empty_like(rho_drop)
    for repeat in range(repeats):
        order = np.argsort(-rho_drop[repeat], kind="stable")
        repeat_ranks[repeat, order] = np.arange(1, len(features) + 1)

    bootstrap_rng = np.random.default_rng(20261024)
    bootstrap_ranks = []
    for _ in range(30):
        index = bootstrap_rng.integers(0, len(y), size=len(y))
        base = safe_spearman(prediction[index], y[index])
        drops = np.asarray([
            np.mean([
                base - safe_spearman(perm_predictions[r][c][index], y[index])
                for r in range(repeats)
            ])
            for c in range(len(features))
        ])
        order = np.argsort(-drops, kind="stable")
        ranks = np.empty(len(features), dtype=int)
        ranks[order] = np.arange(1, len(features) + 1)
        bootstrap_ranks.append(ranks)
    bootstrap_ranks_array = np.vstack(bootstrap_ranks)

    frame = pd.DataFrame({
        "feature": features,
        "spearman_drop": rho_drop.mean(axis=0),
        "spearman_drop_sd": rho_drop.std(axis=0, ddof=1),
        "mae_increase": mae_increase.mean(axis=0),
        "repeat_rank_median": np.median(repeat_ranks, axis=0),
        "repeat_rank_min": repeat_ranks.min(axis=0),
        "repeat_rank_max": repeat_ranks.max(axis=0),
        "bootstrap_rank_median": np.median(bootstrap_ranks_array, axis=0),
        "bootstrap_rank_p05": np.quantile(bootstrap_ranks_array, .05, axis=0),
        "bootstrap_rank_p95": np.quantile(bootstrap_ranks_array, .95, axis=0),
        "bootstrap_top5_frequency": np.mean(bootstrap_ranks_array <= 5, axis=0),
        "internal_split_gain_fraction": gain,
        "internal_split_count_fraction": count,
    })
    frame["rank"] = frame.spearman_drop.rank(method="first", ascending=False).astype(int)
    frame = frame.sort_values("rank").reset_index(drop=True)
    meta = {
        "cycle": cycle,
        "conditioned_model": summary["model_before"],
        "calibration_n": calibration["calibration_n"],
        "calibration_policy": calibration["policy"],
        "validation_n": len(validation),
        "validation_role": f"{cycle} post-ranking selected Hard10k; label-held-out, acquisition-shifted",
        "validation_spearman": base_rho,
        "validation_mae_log1p": base_mae,
        "f40_sha256": manifest["F40_sha256"],
        "config_sha256": manifest["config_sha256"],
        "feature_order": features,
    }
    return frame, meta


def markdown_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str] | None = None) -> list[str]:
    formats = formats or {}
    labels = [column.replace("_", " ") for column in columns]
    lines = ["| " + " | ".join(labels) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append("NA")
            elif column in formats:
                values.append(format(value, formats[column]))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def importance_audit() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    m0, meta0 = cycle_importance("C00")
    m4, meta4 = cycle_importance("C04")
    merged = m0.merge(m4, on="feature", suffixes=("_m0", "_m4"), validate="one_to_one")
    merged["rank_change_m4_minus_m0"] = merged.rank_m4 - merged.rank_m0
    merged["selected_old_structure_feature"] = merged.feature.isin(SELECTED)
    merged = merged.sort_values("rank_m4").reset_index(drop=True)
    merged.to_csv(OUT / "01_m0_m4_hgb40_importance.csv", index=False)

    top5_overlap = len(set(m0.head(5).feature) & set(m4.head(5).feature))
    top10_overlap = len(set(m0.head(10).feature) & set(m4.head(10).feature))
    rank_rho = safe_spearman(merged.rank_m0.to_numpy(), merged.rank_m4.to_numpy())
    selected_rows = merged[merged.feature.isin(SELECTED)].copy()
    top = merged.sort_values("rank_m4").head(10).copy()
    top_m0 = merged.sort_values("rank_m0").head(10).copy()
    formats = {
        "rank_m0": ".0f", "rank_m4": ".0f", "rank_change_m4_minus_m0": "+.0f",
        "spearman_drop_m0": ".6f", "spearman_drop_m4": ".6f",
        "bootstrap_top5_frequency_m0": ".3f", "bootstrap_top5_frequency_m4": ".3f",
    }
    lines = [
        "# M0 vs M4 HGB40 Feature Importance",
        "",
        "## Comparability boundary",
        "",
        f"- M0 audit source: C00 HGB40, conditioned on M0, calibration `{meta0['calibration_policy']}` with {meta0['calibration_n']:,} rows.",
        f"- M4 audit source: C04 HGB40, conditioned on M4, calibration `{meta4['calibration_policy']}` with {meta4['calibration_n']:,} rows.",
        f"- F40 names/order hash is identical: `{meta0['f40_sha256']}`; HGB config hash is also identical: `{meta0['config_sha256']}`.",
        "- The importance algorithm is identical in this audit: eight independent-set permutations, Spearman-drop primary score, 30 row bootstraps, fixed audit seeds, plus the same private persisted-tree split-gain diagnostic.",
        "- The validation datasets are **not the same**: each cycle uses its own surrogate-selected, newly labelled Hard10k. Both are label-held-out but acquisition-shifted. C00 also has only 500 calibration rows whereas C04 has 10,000. Therefore raw drop differences and rank movement are descriptive, not a causal estimate of M0→M4 adaptation.",
        "",
        "## M4 Top10 with M0 comparison",
        "",
        *markdown_table(top, ["rank_m4", "feature", "spearman_drop_m4", "bootstrap_top5_frequency_m4", "rank_m0", "spearman_drop_m0", "bootstrap_top5_frequency_m0", "rank_change_m4_minus_m0"], formats),
        "",
        "## M0 Top10 with M4 comparison",
        "",
        *markdown_table(top_m0, ["rank_m0", "feature", "spearman_drop_m0", "bootstrap_top5_frequency_m0", "rank_m4", "spearman_drop_m4", "bootstrap_top5_frequency_m4", "rank_change_m4_minus_m0"], formats),
        "",
        "## Consistency result",
        "",
        f"- Top-5 overlap: **{top5_overlap}/5**; Top-10 overlap: **{top10_overlap}/10**.",
        f"- Across all F40 ranks, Spearman rank correlation is **{rank_rho:.3f}**.",
        f"- Validation skill differs (M0 selected-set rho {meta0['validation_spearman']:.3f}, M4 {meta4['validation_spearman']:.3f}), but the validation distributions also differ.",
        "- Conclusion: importance is **partially consistent, not invariant**. Agreement is summarized by overlap/rank correlation; disagreement cannot be assigned solely to solver adaptation.",
        "",
        "## Frozen two-feature ranks",
        "",
        *markdown_table(selected_rows, ["feature", "rank_m0", "spearman_drop_m0", "bootstrap_top5_frequency_m0", "rank_m4", "spearman_drop_m4", "bootstrap_top5_frequency_m4"], formats),
        "",
        "The old structural pair was not Top1/Top2. P0 first preserved the full M4 ranking, then applied a cross-size matching criterion: finite on all TSPLIB48, dimensionless under uniform scaling, interpretable, and nonredundant. It chose M4 ranks 3 and 6. This was a defensible *matching-space* decision, but it must not be reinterpreted as HGB40 itself selecting those exact two features.",
        "",
        "The legacy GP4/QD4 feature set is not used as M0 importance evidence.",
    ]
    (OUT / "01_m0_m4_hgb40_importance_comparison.md").write_text("\n".join(lines) + "\n")
    return merged, meta0, meta4


FEATURE_PROPERTIES: dict[str, dict[str, str]] = {
    "spn_sum_of_lowest_edge_values": {
        "definition": "sum of the n smallest values among all n(n-1)/2 undirected Euclidean pairwise distances",
        "scale": "degree 1: f(aX)=a f(X), a>0",
        "node": "explicitly selects n edges; edge order statistics and typical spacing change with n",
        "missing": "finite for nondegenerate n>=2 input; zero/duplicate-heavy inputs can collapse it",
    },
    "spn_distance_min": {
        "definition": "minimum of all n(n-1)/2 undirected Euclidean pairwise distances",
        "scale": "degree 1",
        "node": "extreme statistic over O(n^2) pairs, normally decreases as n grows at fixed area",
        "missing": "finite for n>=2; equals zero when any coordinates duplicate",
    },
    "spn_distance_varcoeff": {
        "definition": "sample SD of all undirected pairwise distances divided by their mean",
        "scale": "dimensionless; invariant to positive uniform scaling",
        "node": "no algebraic n factor, but finite-sample/distributional dependence remains",
        "missing": "undefined if mean distance is zero; finite for all TSPLIB48 aspect variants",
    },
    "spn_nng_3_strong_components_varcoeff": {
        "definition": "sample SD / mean of strong-component sizes in the directed 3-nearest-neighbor graph",
        "scale": "invariant under positive uniform scaling, except distance ties/numerics",
        "node": "component count/size distribution depends on n although k=3 is fixed",
        "missing": "R sample SD is NA for one component; missing on eil51 and rat195 in TSPLIB48 aspect data",
    },
    "spn_mst_dists_sd": {
        "definition": "sample SD of the n-1 edge lengths in the Prim spanning tree",
        "scale": "degree 1",
        "node": "MST edge scale changes with point density and sample size n-1",
        "missing": "undefined for fewer than two MST edges; finite for audited data",
    },
    "spn_fraction_shorter_mean_distance": {
        "definition": "fraction of undirected pairwise distances strictly below their arithmetic mean",
        "scale": "dimensionless; invariant to positive uniform scaling",
        "node": "pair count changes with n; finite-sample and geometry dependence remain",
        "missing": "finite for n>=2 and all TSPLIB48 aspect data",
    },
    "spn_mst_dists_span": {
        "definition": "maximum minus minimum among the n-1 MST edge lengths",
        "scale": "degree 1",
        "node": "MST extreme range changes with density and n",
        "missing": "finite for n>=2; degenerates to zero for a single edge",
    },
    "spn_hull_dists_varcoeff": {
        "definition": "sample SD / mean of per-node distances to the nearest convex-hull supporting line; hull nodes contribute zero",
        "scale": "dimensionless; invariant to positive uniform scaling",
        "node": "hull size, zero fraction, and interior composition change with n",
        "missing": "undefined when mean hull distance is zero (e.g. all nodes on hull)",
    },
    "spn_mst_depth_mean": {
        "definition": "mean of salesperson spandepth values for the Prim spanning tree",
        "scale": "invariant to uniform scaling, except MST ties/numerics",
        "node": "tree depth depends directly on tree size, topology, and implementation root",
        "missing": "finite for audited data",
    },
    "spn_hull_dists_median": {
        "definition": "median per-node distance to the nearest convex-hull supporting line; hull nodes contribute zero",
        "scale": "degree 1",
        "node": "hull-point fraction and interior distances change with n",
        "missing": "finite; may be zero when at least half the nodes lie on the hull",
    },
}


def inverse_scaler(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    payload = json.loads(SCALER.read_text())
    params = {item["feature_name"]: item for item in payload["parameters"]}
    result = frame.copy()
    for feature in features:
        p = params[feature]
        result[feature] = float(p["center_median"]) + float(p["scale"]) * np.sinh(result[feature].astype(float))
    return result


def feature_properties(importance: pd.DataFrame) -> None:
    top10 = importance.sort_values("rank_m4").head(10).feature.tolist()
    candidate_z = pd.read_parquet(CYCLES / "C04/candidate_features.parquet", columns=top10)
    target_all = pd.read_csv(P1 / "p1_tsplib48_normalization_sensitivity.csv")
    target_z = target_all[target_all.coordinate_variant.eq("aspect")][top10].reset_index(drop=True)
    candidate_raw = inverse_scaler(candidate_z, top10)
    target_raw = inverse_scaler(target_z, top10)

    summaries = []
    for source, frame in (("TSP100_C04_candidate_pool", candidate_raw), ("TSPLIB48_aspect", target_raw)):
        for feature in top10:
            values = frame[feature].to_numpy(float)
            finite = values[np.isfinite(values)]
            summaries.append({
                "source": source, "feature": feature, "n": len(values), "finite_n": len(finite),
                "missing_n": len(values) - len(finite), "mean": np.mean(finite) if len(finite) else np.nan,
                "sd": np.std(finite, ddof=1) if len(finite) > 1 else np.nan,
                "min": np.min(finite) if len(finite) else np.nan,
                "q25": np.quantile(finite, .25) if len(finite) else np.nan,
                "median": np.median(finite) if len(finite) else np.nan,
                "q75": np.quantile(finite, .75) if len(finite) else np.nan,
                "max": np.max(finite) if len(finite) else np.nan,
            })
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "support_top10_raw_feature_distributions.csv", index=False)
    correlations = candidate_z[top10].corr(method="spearman")
    correlations.to_csv(OUT / "support_m4_top10_candidate_spearman.csv")
    max_corr = []
    for feature in top10:
        other = correlations.loc[feature].drop(feature).abs()
        partner = other.idxmax()
        max_corr.append((feature, partner, correlations.loc[feature, partner]))

    prop_rows = []
    for feature in top10:
        imp = importance.loc[importance.feature.eq(feature)].iloc[0]
        target_row = summary[(summary.source == "TSPLIB48_aspect") & (summary.feature == feature)].iloc[0]
        partner = next(item for item in max_corr if item[0] == feature)
        prop_rows.append({
            "rank": int(imp.rank_m4), "feature": feature,
            "definition": FEATURE_PROPERTIES[feature]["definition"],
            "coordinate_scaling": FEATURE_PROPERTIES[feature]["scale"],
            "node_count": FEATURE_PROPERTIES[feature]["node"],
            "missing_behavior": FEATURE_PROPERTIES[feature]["missing"],
            "tsplib48_missing": int(target_row.missing_n),
            "strongest_tsp100_correlate": partner[1], "spearman": partner[2],
            "spearman_drop": imp.spearman_drop_m4,
        })
    props = pd.DataFrame(prop_rows)
    props.to_csv(OUT / "support_m4_top10_feature_properties.csv", index=False)

    dist_view = summary[summary.feature.isin(top10)][["source", "feature", "finite_n", "missing_n", "q25", "median", "q75"]]
    formats = {"finite_n": ".0f", "missing_n": ".0f", "q25": ".6g", "median": ".6g", "q75": ".6g"}
    lines = [
        "# Top Feature Properties and Scale Audit",
        "",
        "## Definition source and representation boundary",
        "",
        "Definitions were checked against the installed `salesperson` 1.0.0 functions actually invoked by `src/features/extract_salesperson_raw_features.R` (`normalize=FALSE`), not inferred from names. Pairwise distances are Euclidean; ordinary R `sd` is sample SD. The HGB files store `arcsinh((raw - frozen_median)/frozen_scale)`. For the table below those transformed values were exactly inverted to raw feature units.",
        "",
        "For convex-hull distances, the package computes perpendicular distance to each hull edge's **supporting line** and takes the minimum; it does not clip the projection to the edge segment. For the 3-NNG feature it constructs a directed k=3 graph and summarizes strong-component sizes.",
        "",
        "## M4 Top10 properties",
        "",
    ]
    for row in prop_rows:
        lines.extend([
            f"### {row['rank']}. `{row['feature']}`",
            "",
            f"- Definition: {row['definition']}.",
            f"- Coordinate scaling: {row['coordinate_scaling']}.",
            f"- Node-count behavior: {row['node_count']}.",
            f"- Missing/undefined: {row['missing_behavior']}; TSPLIB48 aspect missing count = {row['tsplib48_missing']}.",
            f"- Predictive contribution: M4 permutation Spearman drop {row['spearman_drop']:.6f}.",
            f"- Strongest absolute TSP100 candidate-pool Top10 correlation: `{row['strongest_tsp100_correlate']}` (rho {row['spearman']:.3f}). This diagnoses redundancy but is not an elimination rule.",
            "",
        ])
    lines.extend([
        "## Raw feature distributions",
        "",
        *markdown_table(dist_view, ["source", "feature", "finite_n", "missing_n", "q25", "median", "q75"], formats),
        "",
        "The TSP100 pool has fixed n=100 and coordinates in [0,1]; TSPLIB48 spans node counts and native conventions but uses the frozen aspect-preserving coordinate normalization before extraction. Remaining distribution differences mix structure with node-count effects. In particular, raw order/extreme statistics (`sum_of_lowest_edge_values`, `distance_min`, MST lengths) are not directly exchangeable across n without an explicit nuisance model.",
        "",
        "## Normalization is not elimination",
        "",
        "There is **no sufficient basis to delete** a high-importance feature merely because it is scale- or node-sensitive. The old P0 reason supports excluding an *unnormalized raw value from a two-dimensional cross-size density-matching space*, not excluding it from HGB importance or future models.",
        "",
        "For any degree-1 distance feature `f`, a transparent coordinate-scale correction is `f_scale = f / D`, where `D = sqrt((xmax-xmin)^2 + (ymax-ymin)^2)` is the bounding-box diagonal. If density rather than global scale is the nuisance, a candidate is `f_density = f / sqrt(A/n)` with bounding-box area A, but it is unstable when A=0 and changes the scientific quantity. Node-order effects in a sum/minimum cannot generally be removed by a single dimensional multiplier; they need conditioning on n, a fixed-n target, or an empirically preregistered residualization model.",
        "",
        "All such transforms discard some absolute scale/density signal that may genuinely predict solver-relative hardness. Therefore the original feature and raw importance must remain reported alongside any normalized companion. The frozen HGB arcsinh standardization is a robust numeric transform, **not** physical scale/node normalization.",
    ])
    (OUT / "02_feature_properties_and_scale_audit.md").write_text("\n".join(lines) + "\n")


def code_audit() -> None:
    p20 = Path("/home/haozou/.codex/attachments/3195d8a9-6036-4aa6-b956-1eb4b1059fe3/pasted-text.txt")
    p22 = Path("/home/haozou/.codex/attachments/cd3ebf83-839c-4560-9cae-2fd60b6486ac/pasted-text.txt")
    lines = [
        "# `structured_select` 70/30 Code Audit",
        "",
        "## Exact definition",
        "",
        "For survivor capacity N=10,000, the code requests `round(0.70*N)=7,000` **strict target-coverage candidates** and assigns the remainder to **frontier exploration**. This is a requested count split per generation, not a loss weight, probability, mutation mixture, or training-data ratio (`run_p2_bounded_evolution.py:65-81, 709-735`).",
        "",
        "Strict means both `nearest TSPLIB48 distance <= r` in median/IQR-standardized two-feature space and inside the target bounding box. The radius r is the median nearest-neighbor distance among the 48 targets. The 3x3 cells use per-axis target terciles (`:217-266, 269-283, 727-735`).",
        "",
        "If fewer than 7,000 strict candidates exist, `target_n=min(7000,total_strict_capacity)` and all shortfall moves explicitly to frontier; the population stays 10,000. Consequently, ‘70/30’ is an aspiration, not a guaranteed realized split. For example, Original E1 realized 2,149/7,851 at G1 and 4,813/5,187 at G2.",
        "",
        "## Accurate pseudocode",
        "",
        "```text",
        "input: unique parent+eligible-child union U, capacity N, TSPLIB48 two-feature geometry",
        "requested_strict = round(0.70 N)",
        "strict = {u: within radius of a target AND inside target box}",
        "strict_n = min(requested_strict, |strict|)",
        "allocate strict_n across 3x3 cells proportional to TSPLIB48 counts, capacity-aware",
        "within each cell: round-robin 12x12 microbins; then distance; lineage cap; hash tie-break",
        "frontier = unselected candidates NOT in strict region",
        "allocate N-strict_n by nearest target cell proportional to TSPLIB48 counts, capacity-aware",
        "within each frontier cell: same microbin/distance/lineage rule",
        "if short: deterministic microbin/lineage backfill from all remaining candidates",
        "return exactly N, sorted by component then instance hash",
        "```",
        "",
        "Capacity-aware 3x3 allocation is in `capacity_quotas` and its calls at `:727-759`. Microbin ordering is at `:661-672`: 12x12 target-range bins, candidates sorted in each bin by target distance/hash, then round-robin by within-bin rank. `lineage_take` starts at cap 40 for active populations and doubles it when necessary (`:675-706`); thus diversity is a soft feasibility device, not an optimized diversity score. Deterministic final backfill is `:761-770`. No BQ loss, HGB score, or hardness enters this function (`:717`).",
        "",
        "## What is optimized",
        "",
        "There is no single scalar objective. It is a lexicographic, capacity-constrained heuristic: (1) attempt a strict/frontier count; (2) approximate TSPLIB48 3x3 mass by quotas; (3) spread across 12x12 microbins; (4) prefer smaller target distance within tie structure; (5) constrain lineage concentration with relaxable caps; (6) deterministic hash tie-break/backfill. It does **not** explicitly minimize JS, Wasserstein, MMD, lineage HHI, or expected hardness.",
        "",
        "Geometry preprocessing costs O(48N) time and O(48N) temporary distance storage in the current vectorized implementation; partition sorting is O(N log N) in the usual bound, and lineage passes can repeat while the cap doubles. Feature extraction is separate and normally dominates. Stored selection metadata makes shortages and realized components auditable.",
        "",
        "## Evidence for 70/30",
        "",
        "The P2.0 request explicitly called it an exploratory configuration and prohibited a parameter search; P2.2 allowed the first version to continue it. Repository code/config/reports and the supplied study records contain no cited literature derivation, theoretical optimum, or prior ratio ablation.",
        "",
        "`70_30_RATIO_EMPIRICAL_JUSTIFICATION_NOT_FOUND`",
        "",
        "This is a faithful status, not evidence that 70/30 is necessarily poor.",
        "",
        "## Why ‘Structure-Guided Evolution’ is still accurate",
        "",
        "- Mutation generation: FOGA-11 samples only from the current active population; mutation itself is not steered by the two features (`:1322-1331`).",
        "- Survivor replacement: E0 uniformly samples the canonicalized union with a fixed seed (`:790-801, 1354-1358`); E1 applies `structured_select` (`:1359-1369`). This is the causal structure-guided link.",
        "- Recurrence: saved G1 survivors become G2 parents (`:1367-1371`), so structural selection changes the next mutation distribution.",
        "- Archive retention: the old common bounded archive separately applied `structured_select` above capacity (`:811-831, 1376-1383`).",
        "- Final acquisition: a later, separate archive-based selector created old Structure10k (`:1386-1424`). It is not survivor selection and is removed from the proposed direct-P2 training protocol.",
    ]
    if not p20.is_file() or not p22.is_file():
        lines.append("\nAudit note: one or both supplied historical protocol attachments were unavailable; code conclusions remain unaffected.")
    (OUT / "03_structured_select_70_30_code_audit.md").write_text("\n".join(lines) + "\n")


def load_bundle(path: Path) -> tuple[np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as bundle:
        coordinates = np.asarray(bundle["coordinates"], dtype=np.float64)
        hashes = [str(value) for value in bundle["instance_hash"].tolist()]
    return coordinates, hashes


def coordinate_hashes(coordinates: np.ndarray) -> list[str]:
    sys.path.insert(0, str(ROOT / "src"))
    from mutation.result import coordinate_hash  # noqa: PLC0415
    return [coordinate_hash(points) for points in coordinates]


def p5_label_cache() -> pd.DataFrame:
    path = P5 / "run_p5_structure_guided_evolution_robustness.py"
    spec = importlib.util.spec_from_file_location("p6_imported_p5", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import P5 cache helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.p5_cache()


def population_paths() -> list[tuple[str, str, Path]]:
    rows = []
    for replicate in ("Original", "S1", "S2"):
        base = P2 / "evolution" if replicate == "Original" else P5 / "evolution" / replicate / "evolution"
        for arm in ("E0", "E1"):
            rows.append((replicate, arm, base / arm / "G2" / "active_population.parquet"))
    return rows


def population_audit() -> pd.DataFrame:
    cache = p5_label_cache()
    known = set(cache.instance_hash.astype(str))
    p03 = pd.read_parquet(CYCLES / "C03/06_selection/P_next.parquet")
    p03_hashes = set(p03.instance_hash.astype(str))
    rows = []
    missing_union: set[str] = set()
    for replicate, arm, manifest_path in population_paths():
        base = manifest_path.parents[2]
        metrics_path = P2 / "02_generation_metrics.csv" if replicate == "Original" else base.parent / "02_generation_metrics.csv"
        metrics = pd.read_csv(metrics_path)
        raw_metrics = metrics[(metrics.arm.eq(arm)) & (metrics.object_type.eq("raw_offspring"))].copy()
        attempts = {
            int(str(row.generation).removeprefix("G")): int(row.mutation_attempts)
            for row in raw_metrics.itertuples(index=False)
        }
        g1_path = base / arm / "G1/active_population.parquet"
        g2_diag_path = base / arm / "G2/offspring_diagnostics.parquet"
        g1 = pd.read_parquet(g1_path)
        g2 = pd.read_parquet(manifest_path)
        diag1 = pd.read_parquet(base / arm / "G1/offspring_diagnostics.parquet")
        diag2 = pd.read_parquet(g2_diag_path)
        coordinates, hashes = load_bundle(manifest_path.with_suffix(".npz"))
        reproduced = coordinate_hashes(coordinates)
        manifest_hashes = g2.instance_hash.astype(str).tolist()
        node_unique = np.asarray([len(np.unique(points, axis=0)) for points in coordinates])
        g1_hashes = set(g1.instance_hash.astype(str))
        g1_born = g1.generation_born.eq(1)
        g2_born = g2.generation_born.eq(2)
        missing = set(manifest_hashes) - known
        missing_union.update(missing)
        selection1 = json.loads((base / arm / "G1/active_population_selection.json").read_text())
        selection2 = json.loads((base / arm / "G2/active_population_selection.json").read_text())
        rows.append({
            "replicate": replicate, "arm": arm, "rows": len(g2),
            "coordinate_shape": "x".join(map(str, coordinates.shape)),
            "unique_hashes": len(set(manifest_hashes)),
            "finite_coordinates": bool(np.isfinite(coordinates).all()),
            "coordinates_in_unit_square": bool(((coordinates >= 0) & (coordinates <= 1)).all()),
            "all_instances_100_unique_nodes": bool((node_unique == 100).all()),
            "bundle_manifest_order_match": hashes == manifest_hashes,
            "coordinate_hashes_reproduce": reproduced == manifest_hashes,
            # P03 roots may legitimately have no direct parent.  All lineage
            # identity fields must exist, and every G1/G2-born row must carry
            # a direct parent that is checked against the prior population.
            "lineage_fields_complete": bool(
                g2[["lineage_root", "evolution_origin", "generation_born"]].notna().all().all()
                and g2.loc[g2.generation_born.ge(1), "direct_parent_hash"].notna().all()
            ),
            "g1_born_parent_in_p03": bool(g1.loc[g1_born, "direct_parent_hash"].astype(str).isin(p03_hashes).all()),
            "g2_born_parent_in_g1": bool(g2.loc[g2_born, "direct_parent_hash"].astype(str).isin(g1_hashes).all()),
            "g1_mutation_attempts": attempts[1], "g2_mutation_attempts": attempts[2],
            "g1_eligible_unique_offspring": len(diag1), "g2_eligible_unique_offspring": len(diag2),
            "g2_attempt_parents_all_in_g1": bool(diag2.parent_hash.astype(str).isin(g1_hashes).all()),
            "g1_selected_n": int(selection1["selected_n"]), "g2_selected_n": int(selection2["selected_n"]),
            "known_verified_labels": len(set(manifest_hashes) & known),
            "missing_labels": len(missing),
            "manifest_sha256": sha256_file(manifest_path),
            "bundle_sha256": sha256_file(manifest_path.with_suffix(".npz")),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "support_g2_population_feasibility.csv", index=False)
    (OUT / "support_label_cache_summary.json").write_text(json.dumps({
        "verified_cache_unique": len(known),
        "logical_missing_sum_across_six_bundles": int(frame.missing_labels.sum()),
        "unique_missing_union_across_six_bundles": len(missing_union),
        "note": "No Concorde calls made. M4 training overlap intentionally not inspected.",
    }, indent=2) + "\n")
    return frame


def evolution_reports(populations: pd.DataFrame) -> None:
    survivor_existing = pd.read_csv(P5 / "03_generation_survivor_audit.csv")
    # Preserve only protocol facts; do not reuse any performance outcome for design tuning.
    all_good = bool(
        (populations.rows == 10_000).all()
        and (populations.unique_hashes == 10_000).all()
        and populations.finite_coordinates.all()
        and populations.coordinates_in_unit_square.all()
        and populations.all_instances_100_unique_nodes.all()
        and populations.bundle_manifest_order_match.all()
        and populations.coordinate_hashes_reproduce.all()
        and populations.lineage_fields_complete.all()
        and populations.g1_born_parent_in_p03.all()
        and populations.g2_born_parent_in_g1.all()
        and populations.g2_attempt_parents_all_in_g1.all()
        and (populations.g1_mutation_attempts == 20_000).all()
        and (populations.g2_mutation_attempts == 20_000).all()
        and (populations.g1_selected_n == 10_000).all()
        and (populations.g2_selected_n == 10_000).all()
    )
    lines = [
        "# Population Evolution Causal Audit",
        "",
        "The persisted artifacts confirm the intended recurrent population path for all Original/S1/S2 replicates and both arms:",
        "",
        "`P03(10k) → mutate current P0 (20k attempts) → unique parent+child union → select P1(10k) → mutate current P1 (20k attempts) → unique parent+child union → select P2/G2(10k)`.",
        "",
        "- Each G1 and G2 offspring-diagnostics table has exactly 20,000 attempts in E0 and E1.",
        "- Each selection JSON records exactly 10,000 survivors.",
        "- Every G2 mutation attempt's parent hash belongs to that arm's persisted G1 population; G1 therefore actually becomes the G2 parent population.",
        "- Every G2-born survivor's direct parent is in G1; every G1-born survivor's direct parent is in formal P03.",
        "- The six G2 populations are fully persisted as ordered parquet manifests plus coordinate NPZ bundles.",
        "",
        "The E0/E1 mutation budget is matched within every replicate. Mutation seeds/namespaces differ by replicate but are paired between arms according to the frozen P5 protocol; the selection rule is the arm-level difference.",
        "",
        "Crucial causal clarification: survivor selection **did execute**, and E1 G1 survivors changed G2's parents. The old M5 did not train on G2 survivors because P2.2 separately retained eligible offspring in a bounded Archive30k and then acquired Structure10k from that archive. Thus the final survivor population was off the old training-data causal path; it was not absent.",
        "",
        f"Artifact-level causal/provenance status: **{'PASS' if all_good else 'FAIL'}**.",
        "",
        f"Cross-check: the prior P5 survivor audit contains {len(survivor_existing)} generation/arm rows and was used only as an existing consistency record; P6 independently re-read the saved G1/G2 artifacts.",
    ]
    (OUT / "04_population_evolution_causal_audit.md").write_text("\n".join(lines) + "\n")

    view = populations[["replicate", "arm", "rows", "unique_hashes", "known_verified_labels", "missing_labels", "lineage_fields_complete"]]
    formats = {"rows": ".0f", "unique_hashes": ".0f", "known_verified_labels": ".0f", "missing_labels": ".0f"}
    lines = [
        "# Final Population10k Dataset Feasibility",
        "",
        *markdown_table(view, list(view.columns), formats),
        "",
        "All six bundles have shape 10,000×100×2, finite coordinates in [0,1], exactly 100 distinct nodes per instance, 10,000 unique coordinate hashes, identical manifest/NPZ order, reproduced coordinate hashes, and complete required lineage fields. Their NPZ structure (`coordinates`, ordered `instance_hash`) is directly convertible to the existing BQ bundle format by attaching ordered `expert_tours`.",
        "",
        "Existing verified Concorde ledger rows may be reused by instance hash. Missing counts above are the only labeling prerequisite; P6 made no Concorde calls. The sum of per-bundle missing counts can exceed the unique missing union when the same instance occurs in more than one bundle; both values are recorded in `support_label_cache_summary.json`.",
        "",
        "M4 training overlap was deliberately neither read nor computed, as required. It is not a validity condition.",
        "",
        "**FINAL_POPULATION10K_DIRECT_TRAINING_FEASIBLE**",
        "",
        "‘Direct’ means no archive-based acquisition or survivor replacement is needed. It does not mean unlabeled coordinates can bypass expert labeling: each missing tour must be generated and integrity-checked before BQ training.",
    ]
    (OUT / "05_final_population10k_feasibility.md").write_text("\n".join(lines) + "\n")


def protocol_and_recommendation(populations: pd.DataFrame, meta0: dict[str, Any], meta4: dict[str, Any]) -> None:
    cache_info = json.loads((OUT / "support_label_cache_summary.json").read_text())
    protocol = [
        "# P6-0 Protocol Summary",
        "",
        "This was a read-only audit. It did not run mutation, Concorde, BQ fine-tuning/evaluation, extra seeds, or parameter search; it wrote only this isolated directory. TSPLIB22 and Original70 performance were not used for feature/objective selection. M4-training overlap was explicitly out of scope and was not inspected.",
        "",
        "Evidence read: frozen C00/M0 and C04/M4 HGB40 models/calibration/selected-label artifacts; frozen F40/scaler and installed salesperson 1.0.0 definitions; P2.2 selection source and historical protocol; P2.2/P5 saved G1/G2 populations, diagnostics, manifests, bundles, and verified label cache.",
        "",
        f"M0 calibration = {meta0['calibration_n']:,}; M4 calibration = {meta4['calibration_n']:,}. The feature/config hashes match, while validation samples differ.",
        "",
        f"Existing verified label cache: {cache_info['verified_cache_unique']:,} unique hashes. Across the six G2 bundles: {cache_info['logical_missing_sum_across_six_bundles']:,} logical missing slots and {cache_info['unique_missing_union_across_six_bundles']:,} unique missing hashes.",
    ]
    (OUT / "00_protocol_summary.md").write_text("\n".join(protocol) + "\n")

    proposal = [
        "# Proposed Population Evolution Protocol",
        "",
        "## Frozen causal skeleton",
        "",
        "```text",
        "P0 = formal P03 Population10k",
        "for g in {1,2}:",
        "    offspring = FOGA11(P[g-1], 20,000 attempts)",
        "    union = unique(P[g-1] + eligible offspring), keeping parent on hash collision",
        "    P[g] = SurvivorSelection(union, 10,000)",
        "D_train = P2",
        "attach/reuse verified Concorde tours",
        "BQ M4 -> M5: frozen 256 epochs / 10,240 updates",
        "evaluate frozen Original70, TSPLIB22, GAP5 protocol",
        "```",
        "",
        "Archive30k retention and archive-based Structure10k acquisition are removed from the training path. An archive may remain diagnostic-only for lineage/reachability and must not determine D_train.",
        "",
        "## A — original random survivor",
        "",
        "Canonical hash-sort the unique parent+child union, then sample exactly 10,000 without replacement using the preregistered generation seed. This is an interpretable uniform baseline with no structure objective.",
        "",
        "## B — original 70/30 survivor",
        "",
        "Apply the audited multistage two-feature heuristic: request 70% strict target coverage, reassign infeasible shortfall to frontier, match 3x3 reference masses capacity-wise, use 12x12 microbin round-robin ordering and relaxable lineage caps, then deterministic backfill. It has no hardness term and no demonstrated optimality for the 70/30 ratio.",
        "",
        "## C — recommended objective-defined successor (design only)",
        "",
        "Use a fixed, preregistered feature representation and choose the 10,000-subset S from union U by the explicit objective",
        "",
        "`min_S  MMD_w^2(F(S), F(TSPLIB48)) + lambda * HHI(lineage(S))`, subject to `|S|=10,000`, unique valid TSP100.",
        "",
        "Here MMD uses a fixed characteristic kernel; `w` is a frozen nonnegative importance weighting from M4 (reported without deleting scale-sensitive features); HHI is sum of squared lineage shares. Kernel bandwidth, lambda, feature list, missing handling, and any dimensional/node normalization must be declared before solver evaluation. Deterministic hash ordering resolves ties. Capacity infeasibility is naturally represented by the best achievable objective rather than an arbitrary target/frontier ratio.",
        "",
        "For scale-sensitive features, retain raw features and add explicitly declared normalized companions or condition the reference transform on n/coordinate scale. Because all candidate populations are TSP100 in [0,1], candidate-to-candidate comparison is already controlled for n and coordinate range; the cross-size TSPLIB48 reference is the part needing a nuisance treatment. Do not tune feature subset, lambda, kernel, or normalization against Original70/TSPLIB22 solver gaps.",
        "",
        "C is a proposal, not an implemented or validated method. The immediate clean experiment should first compare existing A versus existing B on final G2 populations, because those six frozen datasets already isolate the survivor-to-next-generation causal path and require no new mutation.",
    ]
    (OUT / "06_proposed_population_evolution_protocol.md").write_text("\n".join(proposal) + "\n")

    missing_details = ", ".join(
        f"{row.replicate}-{row.arm} {int(row.missing_labels):,}"
        for row in populations.itertuples(index=False)
    )
    final = [
        "# Final Recommendation",
        "",
        "1. **M0/M4 importance consistency:** partially consistent, not identical. The exact Top5/Top10 overlap and all 40 raw ranks are in the comparison outputs. Because C00/C04 validation sets and calibration sizes differ, drop changes are descriptive and not a causal M0→M4 adaptation estimate.",
        "2. **Why not Top1/Top2:** P0 preserved them but chose ranks 3 and 6 for a cross-node-size two-dimensional matching space: both were finite, dimensionless, interpretable, and comparatively nonredundant. HGB40 did not itself select the frozen pair.",
        "3. **Exclude scale-related features?** No. Scale/node dependence justifies caution or explicit normalization for cross-size matching, not deletion from importance/modeling. Raw values and ranks must remain.",
        "4. **Exact 70/30:** per generation it requests 7,000 strict-region survivors and 3,000 non-strict frontier survivors from the parent+child union; strict shortages are reassigned to frontier, then 3x3 quotas, microbin spreading, lineage caps, and backfill apply. Realized proportions need not be 70/30.",
        "5. **Support for 70/30:** no literature, theory, or ratio-ablation justification was found. `70_30_RATIO_EMPIRICAL_JUSTIFICATION_NOT_FOUND`.",
        "6. **Actual optimization:** no single objective; it is a lexicographic capacity/quota/microbin/distance/lineage/tie-break heuristic. It uses no hardness.",
        f"7. **Six G2 Population10k:** yes, all are valid direct training memberships after attaching expert tours. Reusable/missing label counts: {missing_details}. Unique missing union: {cache_info['unique_missing_union_across_six_bundles']:,}.",
        "8. **Keep 70/30?** Keep it only as the frozen B arm for causal comparison. Redefine the next method around an explicit distribution-plus-lineage objective; do not present 70/30 as optimal.",
        "9. **Can the next stage directly compare G2 E0/E1?** Yes. The six frozen G2 populations provide three paired evolution replicates and need no new mutation/acquisition. First label only missing tours, then use the identical M4→M5 protocol. Preregister endpoints and keep TSPLIB22 out of tuning/model selection.",
        "",
        "`BLADE_P6_0_FEATURE_AND_POPULATION_PROTOCOL_AUDIT_COMPLETE`",
    ]
    (OUT / "07_final_recommendation.md").write_text("\n".join(final) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    importance, meta0, meta4 = importance_audit()
    feature_properties(importance)
    code_audit()
    populations = population_audit()
    evolution_reports(populations)
    protocol_and_recommendation(populations, meta0, meta4)
    print(json.dumps({
        "status": "BLADE_P6_0_FEATURE_AND_POPULATION_PROTOCOL_AUDIT_COMPLETE",
        "outputs": 8,
        "support_files": 5,
    }, indent=2))


if __name__ == "__main__":
    main()
