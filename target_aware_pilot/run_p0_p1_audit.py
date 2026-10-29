#!/usr/bin/env python3
"""Read-only P0/P1 audit for M4-conditioned target-aware acquisition.

This script only reads frozen experiment artifacts and writes below
``target_aware_pilot``.  It never trains BQ, runs Concorde, mutates candidates,
or changes the formal A0 implementation.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import math
import os
import subprocess
import tempfile
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
from scipy import stats
from scipy.spatial.distance import jensenshannon


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "target_aware_pilot"
FIG = OUT / "figures"
RUN = ROOT / "runs" / "BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1"
C04 = RUN / "A0_FULL_BASE" / "cycles" / "C04"
M4 = RUN / "A0_FULL_BASE" / "checkpoints" / "M4_E256.tsp.best"
SPLIT_ROOT = (
    ROOT
    / "audits/external_benchmark_suite_v2/evoreal_structure_audit/corrected_v2/_calibration_src"
)
ORIGINAL70 = ROOT / "assets/evaluation/original_tsplib70"
SCALER = ROOT / "assets/qd/reference_cloud/formal_raw_scaler.json"
R_EXTRACTOR = ROOT / "src/features/extract_salesperson_raw_features.R"
F40_MANIFEST = ROOT / "audits/S3_B2_RF_HGB_feature_label_scaling_v1/09_feature_subset_manifests.json"
SEEDS = (2026, 2027, 2028, 2029, 2030)
SELECTED_FEATURES = (
    "spn_distance_varcoeff",
    "spn_fraction_shorter_mean_distance",
)
F40_FEATURES = tuple(json.loads(F40_MANIFEST.read_text())["F40"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def q(values: np.ndarray, probability: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, probability)) if len(values) else float("nan")


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    result = stats.spearmanr(np.asarray(left, float), np.asarray(right, float)).statistic
    return float(result) if np.isfinite(result) else float("nan")


def model_payloads() -> list[dict]:
    return [
        joblib.load(C04 / "HGB_seed_models" / f"HGB40_seed_{seed}.joblib")
        for seed in SEEDS
    ]


def private_split_importance(model, feature_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized split gain/count from sklearn's persisted tree nodes.

    HistGradientBoostingRegressor intentionally has no public
    ``feature_importances_``.  These values are an explicitly labelled audit
    proxy over the private persisted node representation, not a public sklearn
    feature importance API.
    """

    gains = np.zeros(feature_count, dtype=float)
    counts = np.zeros(feature_count, dtype=float)
    for stage in model._predictors:  # noqa: SLF001 - frozen-artifact audit
        for predictor in stage:
            nodes = predictor.nodes
            split = nodes[nodes["is_leaf"] == 0]
            for node in split:
                index = int(node["feature_idx"])
                counts[index] += 1.0
                gain = float(node["gain"])
                if np.isfinite(gain) and gain > 0:
                    gains[index] += gain
    if gains.sum() > 0:
        gains /= gains.sum()
    if counts.sum() > 0:
        counts /= counts.sum()
    return gains, counts


def audit_p0() -> dict:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    manifest = json.loads((C04 / "HGB_model_manifest.json").read_text())
    summary = json.loads((C04 / "cycle_summary.json").read_text())
    calibration_info = json.loads((C04 / "_calibration_info.json").read_text())
    features = list(manifest["features"])
    expected_f40 = json.loads(F40_MANIFEST.read_text())["F40"]
    if features != expected_f40 or len(features) != 40:
        raise RuntimeError("C04 HGB feature order does not match frozen F40")
    if summary["model_before"] != "M4" or summary["input_checkpoint_sha256"] != sha256_file(M4):
        raise RuntimeError("C04 is not conditioned on the frozen A0 Full M4 checkpoint")
    if calibration_info["calibration_n"] != 10_000 or manifest["n_train"] != 10_000:
        raise RuntimeError("C04 calibration is not exactly 10,000")

    train_x_frame = pd.read_parquet(C04 / "_state_B_features.parquet")
    train_y_frame = pd.read_parquet(C04 / "B_current_hardness.parquet")
    if train_x_frame.instance_hash.astype(str).tolist() != train_y_frame.instance_hash.astype(str).tolist():
        raise RuntimeError("calibration feature/target alignment failure")
    train_x = train_x_frame[features].to_numpy(float)
    train_y = train_y_frame.y_log1p.to_numpy(float)

    # Independent labels were obtained only after the deployed HGB ranked the
    # candidate union.  They are label-held-out, but acquisition-shifted.
    selected = pd.read_parquet(C04 / "07_oracle/T/labeled.parquet")
    candidate_x = pd.read_parquet(C04 / "candidate_features.parquet")
    validation = selected[["instance_hash", "current_hardness"]].merge(
        candidate_x, on="instance_hash", how="left", validate="one_to_one"
    )
    if len(validation) != 10_000 or validation[features].isna().any().any():
        raise RuntimeError("independent C04 selected-set feature join failure")
    valid_x = validation[features].to_numpy(float)
    valid_y = np.log1p(validation.current_hardness.to_numpy(float))

    payloads = model_payloads()
    models = [item["model"] for item in payloads]
    model_predictions = np.vstack([model.predict(valid_x) for model in models])
    prediction = np.median(model_predictions, axis=0)
    seed_spread = float(np.max(np.ptp(model_predictions, axis=0)))
    permutation_models = [models[0]] if seed_spread <= 1e-12 else models
    base_rho = safe_spearman(prediction, valid_y)
    base_mae = float(np.mean(np.abs(prediction - valid_y)))

    tail_n = len(valid_y) // 10
    pred_top = np.argsort(-prediction, kind="stable")[:tail_n]
    true_top = set(np.argsort(-valid_y, kind="stable")[:tail_n].tolist())
    tail_recall = float(len(set(pred_top.tolist()) & true_top) / tail_n)
    tail_enrichment = float(np.mean(valid_y[pred_top]) / np.mean(valid_y))
    tail_lift = float(np.mean(valid_y[pred_top]) - np.mean(valid_y))

    raw_gains, raw_counts = [], []
    for model in models:
        gain, count = private_split_importance(model, len(features))
        raw_gains.append(gain)
        raw_counts.append(count)
    raw_gain = np.mean(raw_gains, axis=0)
    raw_count = np.mean(raw_counts, axis=0)

    # Repeated independent-set permutation importance.  Predictions are kept
    # so resampling stability can be assessed without additional model fits.
    permutation_repeats = 8
    rng = np.random.default_rng(20261023)
    rho_drop = np.empty((permutation_repeats, len(features)), dtype=float)
    mae_increase = np.empty_like(rho_drop)
    permuted_predictions: list[list[np.ndarray]] = []
    for repeat in range(permutation_repeats):
        repeat_predictions = []
        for column in range(len(features)):
            permuted = valid_x.copy()
            permuted[:, column] = permuted[rng.permutation(len(permuted)), column]
            value = np.median(
                np.vstack([model.predict(permuted) for model in permutation_models]), axis=0
            )
            repeat_predictions.append(value)
            rho_drop[repeat, column] = base_rho - safe_spearman(value, valid_y)
            mae_increase[repeat, column] = float(np.mean(np.abs(value - valid_y)) - base_mae)
        permuted_predictions.append(repeat_predictions)

    mean_drop = rho_drop.mean(axis=0)
    mean_mae_increase = mae_increase.mean(axis=0)
    # Rank 1 is most important. Stable mergesort plus feature-name order gives
    # deterministic handling of rare exact ties.
    repeat_ranks = np.empty_like(rho_drop)
    for repeat in range(permutation_repeats):
        order = np.argsort(-rho_drop[repeat], kind="stable")
        repeat_ranks[repeat, order] = np.arange(1, len(features) + 1)

    # Bootstrap rows of the already computed independent predictions.
    bootstrap_ranks = []
    bootstrap_rng = np.random.default_rng(20261024)
    for _ in range(30):
        index = bootstrap_rng.integers(0, len(valid_y), size=len(valid_y))
        yb = valid_y[index]
        base_b = safe_spearman(prediction[index], yb)
        drops = np.asarray(
            [
                np.mean(
                    [
                        base_b - safe_spearman(permuted_predictions[r][c][index], yb)
                        for r in range(permutation_repeats)
                    ]
                )
                for c in range(len(features))
            ]
        )
        order = np.argsort(-drops, kind="stable")
        ranks = np.empty(len(features), dtype=int)
        ranks[order] = np.arange(1, len(features) + 1)
        bootstrap_ranks.append(ranks)
    bootstrap_ranks = np.vstack(bootstrap_ranks)

    frame = pd.DataFrame(
        {
            "feature_order": np.arange(1, len(features) + 1),
            "feature": features,
            "sklearn_public_feature_importance_available": False,
            "internal_split_gain_fraction_mean": raw_gain,
            "internal_split_count_fraction_mean": raw_count,
            "permutation_spearman_drop_mean": mean_drop,
            "permutation_spearman_drop_sd": rho_drop.std(axis=0, ddof=1),
            "permutation_mae_increase_mean": mean_mae_increase,
            "permutation_rank_median": np.median(repeat_ranks, axis=0),
            "permutation_rank_min": repeat_ranks.min(axis=0),
            "permutation_rank_max": repeat_ranks.max(axis=0),
            "bootstrap_rank_median": np.median(bootstrap_ranks, axis=0),
            "bootstrap_rank_p05": np.quantile(bootstrap_ranks, 0.05, axis=0),
            "bootstrap_rank_p95": np.quantile(bootstrap_ranks, 0.95, axis=0),
            "bootstrap_top5_frequency": np.mean(bootstrap_ranks <= 5, axis=0),
        }
    )
    frame["permutation_rank"] = frame.permutation_spearman_drop_mean.rank(
        method="first", ascending=False
    ).astype(int)
    frame["internal_gain_rank"] = frame.internal_split_gain_fraction_mean.rank(
        method="first", ascending=False
    ).astype(int)
    frame = frame.sort_values(["permutation_rank", "feature"]).reset_index(drop=True)
    frame.to_csv(OUT / "01_m4_hgb40_feature_importance.csv", index=False)

    top5 = frame.head(5).feature.tolist()
    correlations = validation[top5].corr(method="spearman")
    correlations.to_csv(OUT / "p0_top5_spearman_correlation.csv")
    long_corr = correlations.stack().rename("spearman").reset_index()
    long_corr.columns = ["feature_a", "feature_b", "spearman"]
    long_corr.to_csv(OUT / "p0_top5_spearman_correlation_long.csv", index=False)

    validation_metrics = {
        "validation_role": "C04 post-ranking selected Hard10k; labels excluded from HGB fit; acquisition-shifted",
        "n": len(valid_y),
        "spearman": base_rho,
        "mae_log1p_hardness": base_mae,
        "hard_tail_definition": "top decile within the independent selected-set validation sample",
        "hard_tail_recall_at_10pct": tail_recall,
        "hard_tail_mean_ratio_enrichment": tail_enrichment,
        "hard_tail_mean_log1p_lift": tail_lift,
        "seed_prediction_max_range": seed_spread,
        "train_spearman_manifest": manifest["median_train_spearman"],
        "train_mae_manifest": manifest["median_train_mae"],
    }
    (OUT / "p0_validation_metrics.json").write_text(json.dumps(validation_metrics, indent=2) + "\n")

    plot = frame.sort_values("permutation_spearman_drop_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 10))
    ax.barh(plot.feature, plot.permutation_spearman_drop_mean, color="#3366aa")
    ax.set_xlabel("Independent-set permutation importance (Spearman drop)")
    ax.set_title("M4-conditioned HGB40 feature importance")
    fig.tight_layout()
    fig.savefig(FIG / "01_feature_importance_permutation.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(correlations.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(top5)), [name.replace("spn_", "") for name in top5], rotation=45, ha="right")
    ax.set_yticks(range(len(top5)), [name.replace("spn_", "") for name in top5])
    for i in range(len(top5)):
        for j in range(len(top5)):
            ax.text(j, i, f"{correlations.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Spearman rho")
    ax.set_title("Top-5 feature correlation on independent labels")
    fig.tight_layout()
    fig.savefig(FIG / "02_top5_spearman_correlation.png", dpi=180)
    plt.close(fig)

    return {
        "manifest": manifest,
        "summary": summary,
        "calibration_info": calibration_info,
        "importance": frame,
        "validation_metrics": validation_metrics,
        "top5": top5,
        "correlations": correlations,
        "train_x": train_x_frame,
        "validation": validation,
    }


def load_original70() -> list[dict]:
    manifest = pd.read_csv(ORIGINAL70 / "source_manifest_bq83.csv")
    manifest = manifest[manifest.edge_weight_type.eq("EUC_2D")].copy()
    if len(manifest) != 70 or manifest.source_instance.nunique() != 70:
        raise RuntimeError("canonical OriginalTSPLIB70 manifest is not 70 unique EUC_2D instances")
    records = []
    for row in manifest.itertuples(index=False):
        path = ORIGINAL70 / "instances" / row.size_bin / f"{row.source_instance}.tsp.gz"
        dimension = None
        coordinates = []
        in_coordinates = False
        with gzip.open(path, "rt", encoding="ascii") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                upper = line.upper()
                if upper.startswith("DIMENSION"):
                    dimension = int(line.replace(":", " ").split()[-1])
                elif upper == "NODE_COORD_SECTION":
                    in_coordinates = True
                elif in_coordinates and upper in {"EOF", "DISPLAY_DATA_SECTION", "TOUR_SECTION"}:
                    break
                elif in_coordinates and line:
                    parts = line.split()
                    coordinates.append((int(parts[0]), float(parts[1]), float(parts[2])))
        coordinates.sort(key=lambda item: item[0])
        xy = np.asarray([(x, y) for _, x, y in coordinates], dtype=float)
        if dimension is None or xy.shape != (dimension, 2):
            raise RuntimeError(f"invalid TSPLIB coordinate asset {path}")
        records.append(
            {
                "instance_id": str(row.source_instance),
                "n_nodes": int(dimension),
                "known_optimum": int(row.known_optimum),
                "edge_weight_type": str(row.edge_weight_type),
                "source_file": str(path.relative_to(ROOT)),
                "source_file_sha256": sha256_file(path),
                "coordinates": xy,
            }
        )
    return records


def load_author_split() -> tuple[dict[str, str], dict[str, np.ndarray]]:
    split, author_coordinates = {}, {}
    for label in ("S1", "S2", "S3"):
        path = SPLIT_ROOT / f"tsplib_type{label}.txt"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = ast.literal_eval(line.strip())
            name = str(record[0])
            xy = np.asarray(record[2:], dtype=float).reshape(-1, 2)
            if name in split:
                raise RuntimeError(f"duplicate EvoReal development name {name}")
            split[name] = label
            author_coordinates[name] = xy
    if len(split) != 48:
        raise RuntimeError(f"EvoReal development split has {len(split)} rather than 48 instances")
    return split, author_coordinates


def aspect_normalize(xy: np.ndarray) -> np.ndarray:
    shifted = np.asarray(xy, float) - np.min(xy, axis=0)
    span = float(np.max(shifted))
    return shifted / span if span > 0 else shifted


def per_axis_normalize(xy: np.ndarray) -> np.ndarray:
    shifted = np.asarray(xy, float) - np.min(xy, axis=0)
    span = np.max(shifted, axis=0)
    span[span <= 0] = 1.0
    return shifted / span


def extract_selected_salesperson(variants: list[tuple[str, np.ndarray]]) -> pd.DataFrame:
    params_payload = json.loads(SCALER.read_text())
    params = {
        item["feature_name"]: (float(item["center_median"]), float(item["scale"]))
        for item in params_payload["parameters"]
    }
    with tempfile.TemporaryDirectory(prefix="target_aware_tsplib48_") as temporary:
        work = Path(temporary)
        input_csv, raw_csv, schema_csv = (
            work / "coordinates.csv",
            work / "raw_features.csv",
            work / "schema.csv",
        )
        with input_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["state_id", "node_idx", "x", "y"])
            for state_id, xy in variants:
                for index, (x, y) in enumerate(xy):
                    writer.writerow([state_id, index, repr(float(x)), repr(float(y))])
        process = subprocess.run(
            ["Rscript", str(R_EXTRACTOR), str(input_csv), str(raw_csv), str(schema_csv)],
            text=True,
            capture_output=True,
            check=False,
        )
        if process.returncode or not raw_csv.is_file():
            raise RuntimeError(f"salesperson extraction failed: {process.stderr[-3000:]}")
        raw = pd.read_csv(raw_csv)
    expected = [item[0] for item in variants]
    if raw.state_id.astype(str).tolist() != expected:
        raise RuntimeError("salesperson output order differs from requested TSPLIB variant order")
    if "feature_error" in raw and raw.feature_error.notna().any():
        errors = raw.loc[raw.feature_error.notna(), ["state_id", "feature_error"]]
        raise RuntimeError(f"salesperson feature errors:\n{errors.to_string(index=False)}")
    output = raw[["state_id", *F40_FEATURES]].copy()
    for feature in F40_FEATURES:
        center, scale = params[feature]
        output[feature] = np.arcsinh((output[feature].to_numpy(float) - center) / scale)
    if np.isinf(output[list(F40_FEATURES)].to_numpy(float)).any():
        raise RuntimeError("TSPLIB48 F40 extraction includes infinite values")
    return output


def nearest_distances(source: np.ndarray, reference: np.ndarray, block: int = 4096) -> np.ndarray:
    result = np.empty(len(source), dtype=float)
    for start in range(0, len(source), block):
        values = source[start : start + block]
        squared = ((values[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
        result[start : start + len(values)] = np.sqrt(np.min(squared, axis=1))
    return result


def target_region_mask(values: np.ndarray, target: np.ndarray, radius: float) -> np.ndarray:
    return nearest_distances(values, target) <= radius


def coarse_cells(values: np.ndarray, edges: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    inside = np.ones(len(values), dtype=bool)
    indices = []
    for column, edge in enumerate(edges):
        inside &= (values[:, column] >= edge[0]) & (values[:, column] <= edge[-1])
        # Tercile bin, with the maximum included in the final cell.
        indices.append(np.clip(np.digitize(values[:, column], edge[1:-1]), 0, 2))
    return 3 * indices[0] + indices[1], inside


def distribution_metrics(counts: np.ndarray) -> dict:
    counts = np.asarray(counts, float)
    total = counts.sum()
    probability = counts / total if total else np.zeros_like(counts)
    positive = probability[probability > 0]
    entropy = float(-(positive * np.log(positive)).sum()) if len(positive) else 0.0
    return {
        "n_in_coarse_target_box": int(total),
        "occupied_cells": int(np.sum(counts > 0)),
        "effective_cells": float(np.exp(entropy)),
        "largest_cell_share": float(probability.max()) if total else float("nan"),
    }


def audit_p1(p0: dict) -> dict:
    audited_importance = pd.read_csv(OUT / "01_m4_hgb40_feature_importance.csv")
    selected_ranks = audited_importance.set_index("feature").loc[list(SELECTED_FEATURES), "permutation_rank"]
    if int(selected_ranks.max()) > 6:
        raise RuntimeError("P1 selected features must come from the audited P0 Top 6")
    records = load_original70()
    record_by_name = {record["instance_id"].casefold(): record for record in records}
    author_split, author_xy = load_author_split()
    dev_names = list(author_split)
    missing = [name for name in dev_names if name.casefold() not in record_by_name]
    if missing:
        raise RuntimeError(f"EvoReal48 names missing from Original70: {missing}")
    heldout_names = sorted(set(record_by_name) - {name.casefold() for name in dev_names})
    if len(heldout_names) != 22:
        raise RuntimeError(f"held-out complement has {len(heldout_names)} rather than 22 instances")

    manifest_rows = []
    variants = []
    for name in dev_names:
        record = record_by_name[name.casefold()]
        if len(author_xy[name]) != record["n_nodes"] or not np.allclose(author_xy[name], record["coordinates"]):
            raise RuntimeError(f"author split coordinates differ from frozen Original70 asset for {name}")
        manifest_rows.append(
            {
                key: value
                for key, value in record.items()
                if key != "coordinates"
            }
            | {"evoreal_split": author_split[name], "role": "development_reference"}
        )
        xy = record["coordinates"]
        variants.extend(
            [
                (f"{name}__native", xy),
                (f"{name}__aspect", aspect_normalize(xy)),
                (f"{name}__per_axis", per_axis_normalize(xy)),
            ]
        )
    reference_manifest = pd.DataFrame(manifest_rows)
    reference_manifest.to_csv(OUT / "03_tsplib48_reference_manifest.csv", index=False)
    (OUT / "p1_heldout22_names.json").write_text(
        json.dumps(
            {
                "n": 22,
                "names": heldout_names,
                "policy": "identity audit only; no held-out coordinates/features used in P0/P1 design",
            },
            indent=2,
        )
        + "\n"
    )

    cached_extract = OUT / "p1_tsplib48_normalization_sensitivity.csv"
    if cached_extract.is_file():
        extracted = pd.read_csv(cached_extract)
        expected_states = {item[0] for item in variants}
        if not set(F40_FEATURES).issubset(extracted.columns) or set(extracted.state_id) != expected_states:
            extracted = extract_selected_salesperson(variants)
    else:
        extracted = extract_selected_salesperson(variants)
    parsed = extracted.state_id.str.rsplit("__", n=1, expand=True)
    if "instance_id" not in extracted:
        extracted["instance_id"] = parsed[0]
    if "coordinate_variant" not in extracted:
        extracted["coordinate_variant"] = parsed[1]
    if "n_nodes" not in extracted or "evoreal_split" not in extracted:
        extracted = extracted.merge(
            reference_manifest[["instance_id", "n_nodes", "evoreal_split"]],
            on="instance_id",
            validate="many_to_one",
        )
    extracted.to_csv(OUT / "p1_tsplib48_normalization_sensitivity.csv", index=False)
    missing = pd.DataFrame(
        {
            "feature": F40_FEATURES,
            "missing_native": [
                int(extracted.loc[extracted.coordinate_variant.eq("native"), feature].isna().sum())
                for feature in F40_FEATURES
            ],
            "missing_aspect": [
                int(extracted.loc[extracted.coordinate_variant.eq("aspect"), feature].isna().sum())
                for feature in F40_FEATURES
            ],
            "missing_per_axis": [
                int(extracted.loc[extracted.coordinate_variant.eq("per_axis"), feature].isna().sum())
                for feature in F40_FEATURES
            ],
        }
    )
    missing.to_csv(OUT / "p1_tsplib48_f40_missingness.csv", index=False)
    target_frame = extracted[extracted.coordinate_variant.eq("aspect")].copy()
    target_frame = target_frame.set_index("instance_id").loc[dev_names].reset_index()
    if target_frame[list(SELECTED_FEATURES)].isna().any().any():
        bad = {
            feature: target_frame.loc[target_frame[feature].isna(), "instance_id"].tolist()
            for feature in SELECTED_FEATURES
            if target_frame[feature].isna().any()
        }
        raise RuntimeError(f"selected P1 features are undefined on TSPLIB48: {bad}")

    foga_features = pd.read_parquet(C04 / "candidate_features.parquet")
    union = pd.read_parquet(C04 / "05_scoring/union/C_union.parquet")
    predictions = pd.read_parquet(C04 / "candidate_predictions.parquet")
    hard = pd.read_parquet(C04 / "06_selection/T_selected.parquet")
    foga = (
        foga_features[["instance_hash", *SELECTED_FEATURES]]
        .merge(
            union[["instance_hash", "source", "coordinate_origin", "operator_id", "is_previous_parent", "is_new_offspring"]],
            on="instance_hash",
            validate="one_to_one",
        )
        .merge(
            predictions[["instance_hash", "hgb_predicted_log1p_hardness"]],
            on="instance_hash",
            validate="one_to_one",
        )
    )
    hard_frame = hard[["instance_hash"]].merge(
        foga, on="instance_hash", validate="one_to_one"
    )
    if foga.instance_hash.nunique() != len(foga) or hard_frame.instance_hash.nunique() != 10_000:
        raise RuntimeError("FOGA/Hard10k identity is not unique")

    target = target_frame[list(SELECTED_FEATURES)].to_numpy(float)
    scale = np.subtract(*np.quantile(target, [0.75, 0.25], axis=0))
    if np.any(scale <= 0):
        raise RuntimeError("TSPLIB48 selected-feature IQR is degenerate")
    center = np.median(target, axis=0)
    target_z = (target - center) / scale
    foga_z = (foga[list(SELECTED_FEATURES)].to_numpy(float) - center) / scale
    hard_z = (hard_frame[list(SELECTED_FEATURES)].to_numpy(float) - center) / scale
    target_pair = np.sqrt(((target_z[:, None, :] - target_z[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(target_pair, np.inf)
    target_loo_nearest = np.min(target_pair, axis=1)
    radius = float(np.median(target_loo_nearest))
    foga_to_target = nearest_distances(foga_z, target_z)
    hard_to_target = nearest_distances(hard_z, target_z)
    target_to_foga = nearest_distances(target_z, foga_z)
    target_to_hard = nearest_distances(target_z, hard_z)
    foga_region = foga_to_target <= radius
    hard_region = hard_to_target <= radius

    edges = [
        np.asarray([np.min(target[:, i]), q(target[:, i], 1 / 3), q(target[:, i], 2 / 3), np.max(target[:, i])])
        for i in range(2)
    ]
    target_cell, target_inside = coarse_cells(target, edges)
    foga_cell, foga_inside = coarse_cells(foga[list(SELECTED_FEATURES)].to_numpy(float), edges)
    hard_cell, hard_inside = coarse_cells(hard_frame[list(SELECTED_FEATURES)].to_numpy(float), edges)
    counts = {
        "TSPLIB48": np.bincount(target_cell[target_inside], minlength=9),
        "FOGA_candidates": np.bincount(foga_cell[foga_inside], minlength=9),
        "BLADE_Hard10k": np.bincount(hard_cell[hard_inside], minlength=9),
    }
    target_probability = counts["TSPLIB48"] / counts["TSPLIB48"].sum()
    js = {}
    for group in ("FOGA_candidates", "BLADE_Hard10k"):
        probability = counts[group] / counts[group].sum() if counts[group].sum() else np.zeros(9)
        js[group] = float(jensenshannon(target_probability, probability, base=2.0) ** 2)

    coverage = {
        "target_iqr_scaled_radius": radius,
        "radius_definition": "median TSPLIB48 leave-one-out nearest-neighbor distance",
        "target_loo_nearest_p25": q(target_loo_nearest, 0.25),
        "target_loo_nearest_median": q(target_loo_nearest, 0.50),
        "target_loo_nearest_p75": q(target_loo_nearest, 0.75),
        "foga_target_points_covered_fraction": float(np.mean(target_to_foga <= radius)),
        "hard10k_target_points_covered_fraction": float(np.mean(target_to_hard <= radius)),
        "foga_unique_candidates_in_smoothed_target_region": int(np.sum(foga_region)),
        "hard10k_unique_candidates_in_smoothed_target_region": int(np.sum(hard_region)),
        "foga_region_fraction": float(np.mean(foga_region)),
        "hard10k_region_fraction": float(np.mean(hard_region)),
        "foga_nearest_target_distance_median": q(foga_to_target, 0.5),
        "hard10k_nearest_target_distance_median": q(hard_to_target, 0.5),
        "foga_nearest_candidate_for_target_p90": q(target_to_foga, 0.9),
        "hard10k_nearest_candidate_for_target_p90": q(target_to_hard, 0.9),
        "foga_js_divergence_coarse_vs_target": js["FOGA_candidates"],
        "hard10k_js_divergence_coarse_vs_target": js["BLADE_Hard10k"],
        "foga_region_predicted_hardness_mean": float(foga.loc[foga_region, "hgb_predicted_log1p_hardness"].mean()),
        "foga_region_predicted_hardness_p90": q(foga.loc[foga_region, "hgb_predicted_log1p_hardness"], 0.9),
        "foga_all_predicted_hardness_mean": float(foga.hgb_predicted_log1p_hardness.mean()),
        "hard_region_predicted_hardness_mean": float(hard_frame.loc[hard_region, "hgb_predicted_log1p_hardness"].mean()),
        "foga_region_previous_parent_n": int(np.sum(foga_region & foga.is_previous_parent.fillna(False).to_numpy(bool))),
        "foga_region_new_offspring_n": int(np.sum(foga_region & foga.is_new_offspring.fillna(False).to_numpy(bool))),
    }
    for group, value in counts.items():
        for key, item in distribution_metrics(value).items():
            coverage[f"{group}_{key}"] = item
    (OUT / "p1_coverage_metrics.json").write_text(json.dumps(coverage, indent=2) + "\n")

    # Normalization and finite-node-size diagnostics use development references only.
    normalization_rows = []
    for feature in SELECTED_FEATURES:
        pivot = extracted.pivot(index="instance_id", columns="coordinate_variant", values=feature)
        normalization_rows.append(
            {
                "feature": feature,
                "native_vs_aspect_max_abs": float(np.max(np.abs(pivot.native - pivot.aspect))),
                "native_vs_per_axis_median_abs": float(np.median(np.abs(pivot.native - pivot.per_axis))),
                "native_vs_per_axis_max_abs": float(np.max(np.abs(pivot.native - pivot.per_axis))),
                "spearman_vs_log_n": safe_spearman(target_frame[feature], np.log(target_frame.n_nodes)),
                "pearson_vs_log_n": float(stats.pearsonr(target_frame[feature], np.log(target_frame.n_nodes)).statistic),
            }
        )
    normalization = pd.DataFrame(normalization_rows)
    normalization.to_csv(OUT / "p1_scale_dependence.csv", index=False)

    # Corrected EvoReal diagnostics are reused because they were calibrated on
    # these exact author-provided 48 instances; no held-out instance is involved.
    evoreal_path = SPLIT_ROOT.parent / "calibration_author_tsplib_features.csv"
    evoreal = pd.read_csv(evoreal_path)
    if set(evoreal.instance_id.astype(str)) != set(dev_names) or len(evoreal) != 48:
        raise RuntimeError("corrected EvoReal diagnostic table is not the exact TSPLIB48 reference")
    evoreal.to_csv(OUT / "p1_evoreal_feature_diagnostic.csv", index=False)
    evoreal_summary = (
        evoreal.groupby("author_split")[["fft_energy", "nn_ratio"]]
        .agg(["count", "mean", "median", "std", "min", "max"])
    )
    evoreal_summary.to_csv(OUT / "p1_evoreal_feature_summary.csv")

    # Required audit CSV: compact summaries plus all nine coarse cells.
    audit_rows = []
    for group, frame in (
        ("TSPLIB48", target_frame),
        ("FOGA_candidates", foga),
        ("BLADE_Hard10k", hard_frame),
    ):
        for feature in SELECTED_FEATURES:
            values = frame[feature].to_numpy(float)
            audit_rows.append(
                {
                    "record_type": "feature_summary",
                    "group": group,
                    "feature": feature,
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)),
                    "min": float(np.min(values)),
                    "p10": q(values, 0.1),
                    "p25": q(values, 0.25),
                    "median": q(values, 0.5),
                    "p75": q(values, 0.75),
                    "p90": q(values, 0.9),
                    "max": float(np.max(values)),
                }
            )
    for group, group_counts in counts.items():
        for cell in range(9):
            audit_rows.append(
                {
                    "record_type": "coarse_3x3_cell",
                    "group": group,
                    "cell": cell,
                    "x_tercile": cell // 3,
                    "y_tercile": cell % 3,
                    "count": int(group_counts[cell]),
                    "share_within_target_box": float(group_counts[cell] / max(1, group_counts.sum())),
                }
            )
    pd.DataFrame(audit_rows).to_csv(OUT / "04_joint_distribution_audit.csv", index=False)

    detail_target = target_frame[
        ["instance_id", "n_nodes", "evoreal_split", *SELECTED_FEATURES]
    ].copy()
    detail_target["group"] = "TSPLIB48"
    detail_foga = foga[["instance_hash", *SELECTED_FEATURES, "hgb_predicted_log1p_hardness", "source", "operator_id"]].copy()
    detail_foga["group"] = "FOGA_candidates"
    detail_foga["in_smoothed_target_region"] = foga_region
    detail_hard = hard_frame[["instance_hash", *SELECTED_FEATURES, "hgb_predicted_log1p_hardness", "source", "operator_id"]].copy()
    detail_hard["group"] = "BLADE_Hard10k"
    detail_hard["in_smoothed_target_region"] = hard_region
    pd.concat([detail_target, detail_foga, detail_hard], ignore_index=True, sort=False).to_parquet(
        OUT / "p1_joint_distribution_instances.parquet", index=False
    )

    # Figures.
    rng = np.random.default_rng(20261023)
    sample_index = rng.choice(len(foga), size=min(15_000, len(foga)), replace=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(foga.iloc[sample_index][SELECTED_FEATURES[0]], foga.iloc[sample_index][SELECTED_FEATURES[1]], s=4, alpha=0.10, label="FOGA sample")
    ax.scatter(hard_frame[SELECTED_FEATURES[0]], hard_frame[SELECTED_FEATURES[1]], s=5, alpha=0.20, label="Hard10k")
    for label, marker in (("S1", "o"), ("S2", "s"), ("S3", "^")):
        sub = target_frame[target_frame.evoreal_split.eq(label)]
        ax.scatter(sub[SELECTED_FEATURES[0]], sub[SELECTED_FEATURES[1]], s=55, marker=marker, edgecolor="black", linewidth=0.5, label=f"TSPLIB48 {label}")
    ax.set_xlabel(SELECTED_FEATURES[0].replace("spn_", ""))
    ax.set_ylabel(SELECTED_FEATURES[1].replace("spn_", ""))
    ax.set_title("Target-aware two-feature overlay")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "03_joint_scatter_overlay.png", dpi=180)
    plt.close(fig)

    all_values = np.vstack([target, foga[list(SELECTED_FEATURES)].to_numpy(float), hard_frame[list(SELECTED_FEATURES)].to_numpy(float)])
    xlim = np.quantile(all_values[:, 0], [0.001, 0.999])
    ylim = np.quantile(all_values[:, 1], [0.001, 0.999])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    panels = (
        ("TSPLIB48", target),
        ("FOGA candidates", foga[list(SELECTED_FEATURES)].to_numpy(float)),
        ("BLADE Hard10k", hard_frame[list(SELECTED_FEATURES)].to_numpy(float)),
    )
    for ax, (title, values) in zip(axes, panels):
        if len(values) <= 100:
            ax.scatter(values[:, 0], values[:, 1], s=28, alpha=0.75)
        else:
            ax.hexbin(values[:, 0], values[:, 1], gridsize=28, mincnt=1, bins="log", cmap="viridis")
        ax.set_title(title)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(SELECTED_FEATURES[0].replace("spn_", ""))
    axes[0].set_ylabel(SELECTED_FEATURES[1].replace("spn_", ""))
    fig.tight_layout()
    fig.savefig(FIG / "04_joint_distribution_panels.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(foga[SELECTED_FEATURES[0]], foga[SELECTED_FEATURES[1]], s=3, alpha=0.025, color="gray", label="FOGA")
    ax.scatter(foga.loc[foga_region, SELECTED_FEATURES[0]], foga.loc[foga_region, SELECTED_FEATURES[1]], s=3, alpha=0.15, color="#2a9d8f", label="FOGA in target neighborhoods")
    ax.scatter(target[:, 0], target[:, 1], s=50, color="#d62728", edgecolor="black", linewidth=0.4, label="TSPLIB48")
    ax.set_xlabel(SELECTED_FEATURES[0].replace("spn_", ""))
    ax.set_ylabel(SELECTED_FEATURES[1].replace("spn_", ""))
    ax.set_title("Smoothed target-region coverage")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "05_foga_target_region_coverage.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(foga.hgb_predicted_log1p_hardness, bins=50, density=True, alpha=0.35, label="all FOGA")
    ax.hist(foga.loc[foga_region, "hgb_predicted_log1p_hardness"], bins=50, density=True, alpha=0.45, label="FOGA target region")
    ax.hist(hard_frame.hgb_predicted_log1p_hardness, bins=50, density=True, histtype="step", linewidth=1.8, label="Hard10k")
    ax.set_xlabel("M4-conditioned predicted log1p hardness")
    ax.set_ylabel("density")
    ax.set_title("Predicted hardness in the target structural region")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "06_target_region_predicted_hardness.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, feature in zip(axes, SELECTED_FEATURES):
        for label, marker in (("S1", "o"), ("S2", "s"), ("S3", "^")):
            sub = target_frame[target_frame.evoreal_split.eq(label)]
            ax.scatter(sub.n_nodes, sub[feature], marker=marker, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("node count (log scale)")
        ax.set_ylabel(feature.replace("spn_", ""))
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle("TSPLIB48 node-count dependence")
    fig.tight_layout()
    fig.savefig(FIG / "07_node_count_dependence.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for label, marker in (("S1", "o"), ("S2", "s"), ("S3", "^")):
        sub = evoreal[evoreal.author_split.eq(label)]
        ax.scatter(sub.fft_energy, sub.nn_ratio, s=45, marker=marker, label=label)
    ax.axvline(35.0, linestyle="--", color="gray", linewidth=1)
    ax.axhline(0.5, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("EvoReal FFT Energy")
    ax.set_ylabel("EvoReal nearest-neighbor CV")
    ax.set_title("EvoReal diagnostics on TSPLIB48")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "08_evoreal_fft_nncv.png", dpi=180)
    plt.close(fig)

    return {
        "reference_manifest": reference_manifest,
        "heldout_names": heldout_names,
        "target": target_frame,
        "foga": foga,
        "hard": hard_frame,
        "coverage": coverage,
        "normalization": normalization,
        "counts": counts,
        "evoreal": evoreal,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("p0", "p1", "all"), default="p0", nargs="?")
    args = parser.parse_args()
    if args.phase == "p1":
        cached = pd.read_csv(OUT / "01_m4_hgb40_feature_importance.csv")
        result = {"top5": cached.sort_values("permutation_rank").head(5).feature.tolist()}
    else:
        result = audit_p0()
        print(json.dumps({"phase": args.phase, "top5": result["top5"], **result["validation_metrics"]}, indent=2))
    if args.phase in {"p1", "all"}:
        p1 = audit_p1(result)
        print(json.dumps({"selected_features": SELECTED_FEATURES, **p1["coverage"]}, indent=2))


if __name__ == "__main__":
    main()
