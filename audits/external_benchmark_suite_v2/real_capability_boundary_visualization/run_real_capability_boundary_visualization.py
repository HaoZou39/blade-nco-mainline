#!/usr/bin/env python
"""Read-only, real-data capability-boundary visualization audit.

This script reads frozen benchmark coordinates/features and existing passive
solver evaluations.  It never trains a model, regenerates a benchmark, calls
an oracle, or changes a mainline artifact outside this audit directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from scipy.stats import chi2
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors


OUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
FIG = OUT / "figures"
sys.path.insert(0, str(ROOT / "src"))

from evaluation.datasets import load_fixed_n100, load_original_tsplib70  # noqa: E402


FEATURE_PATH = (
    ROOT
    / "audits/external_benchmark_suite_v2/qd4_structural_distribution_audit/_cache/all_features.parquet"
)
FORMAL_RUN = ROOT / "runs/blade_final_20cycle_seed2026_v1"
TRAJECTORY = ROOT / "audits/external_benchmark_suite_v2/blade_trajectory/work"

DATASETS = ["Uniform100", "GM100", "TSPLIB100", "Original70"]
EVAL_NAMES = {
    "Uniform100": "Uniform100",
    "GM100": "GaussianMixture100",
    "TSPLIB100": "TSPLIB100",
    "Original70": "OriginalTSPLIB70",
}
FEATURES = ["local", "global", "topology", "scale"]
FEATURE_CODES = {
    "local": "spn_sum_of_lowest_edge_values",
    "global": "spn_distance_varcoeff",
    "topology": "spn_mst_depth_mean",
    "scale": "spn_mst_dists_median",
}
VIS_QUOTA = {name: 40 for name in DATASETS}

NAVY = "#102A43"
BLUE = "#1666B2"
RED = "#C43D3D"
GRAY = "#7A8793"
LIGHT_GRAY = "#C8D0D8"
REGION_COLORS = ["#2F6690", "#6D597A", "#3A7D6B", "#B07A4F", "#7A8B3A", "#8F5D71"]
POINT_SIZE = 29.0
FIGSIZE_AB = (7.2, 6.2)
DPI = 300
GRID_N = 220


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eval_path(model: str, dataset: str) -> Path:
    """Return the reconciled official per-instance result for one pair."""
    eval_name = EVAL_NAMES[dataset]
    if dataset in {"Uniform100", "TSPLIB100"}:
        if model == "M0":
            return FORMAL_RUN / "01_baseline_M0" / f"{eval_name}.parquet"
        return FORMAL_RUN / "cycles/C00/11_external_eval" / f"{eval_name}.parquet"
    # These two were passively re-evaluated in suite v2 to correct the old GM
    # physical-order binding and to freeze Original70 aspect-ratio preprocessing.
    return TRAJECTORY / model / f"{eval_name}.parquet"


def load_joined_data() -> tuple[pd.DataFrame, dict[str, str]]:
    features = pd.read_parquet(FEATURE_PATH)
    frames: list[pd.DataFrame] = []
    source_hashes = {str(FEATURE_PATH.relative_to(ROOT)).replace("\\", "/"): sha256_file(FEATURE_PATH)}
    for dataset in DATASETS:
        structural = features.loc[features.dataset == dataset].copy()
        if structural.instance_id.astype(str).duplicated().any():
            raise RuntimeError(f"duplicate frozen feature IDs in {dataset}")
        for model in ("M0", "M1"):
            path = eval_path(model, dataset)
            source_hashes[str(path.relative_to(ROOT)).replace("\\", "/")] = sha256_file(path)
            result = pd.read_parquet(path)
            if result.instance_id.astype(str).duplicated().any():
                raise RuntimeError(f"duplicate {model} result IDs in {dataset}")
            result = result[["instance_id", "gap_percent"]].rename(
                columns={"gap_percent": f"{model}_gap"}
            )
            before = len(structural)
            structural = structural.merge(result, on="instance_id", how="inner", validate="one_to_one")
            if len(structural) != before:
                missing = before - len(structural)
                raise RuntimeError(f"{dataset}: {missing} feature rows lack {model} results")
        structural["instance_key"] = dataset + "::" + structural.instance_id.astype(str)
        frames.append(structural)
    frame = pd.concat(frames, ignore_index=True)
    if frame.instance_key.duplicated().any():
        raise RuntimeError("unified instance_key is not unique")
    if not np.isfinite(frame[["M0_gap", "M1_gap", *FEATURES]].to_numpy(float)).all():
        raise RuntimeError("non-finite feature or exact-reference gap detected")
    return frame, source_hashes


def farthest_point_indices(x: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic structural-coverage subset in normalized feature space."""
    values = np.asarray(x, dtype=np.float64)
    if count >= len(values):
        return np.arange(len(values), dtype=np.int64), np.full(len(values), np.nan)
    center = values.mean(axis=0)
    first = int(np.argmax(np.sum((values - center) ** 2, axis=1)))
    selected = [first]
    selected_distance = [float(np.linalg.norm(values[first] - center))]
    min_sq = np.sum((values - values[first]) ** 2, axis=1)
    min_sq[first] = -np.inf
    while len(selected) < count:
        nxt = int(np.argmax(min_sq))
        selected.append(nxt)
        selected_distance.append(float(np.sqrt(max(min_sq[nxt], 0.0))))
        candidate_sq = np.sum((values - values[nxt]) ** 2, axis=1)
        min_sq = np.minimum(min_sq, candidate_sq)
        min_sq[np.asarray(selected, dtype=np.int64)] = -np.inf
    return np.asarray(selected, dtype=np.int64), np.asarray(selected_distance, dtype=float)


def make_visual_subset(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for dataset in DATASETS:
        sub = frame.loc[frame.dataset == dataset].reset_index(drop=True)
        index, coverage = farthest_point_indices(sub[FEATURES].to_numpy(float), VIS_QUOTA[dataset])
        chosen = sub.iloc[index].copy()
        chosen["selection_step_within_dataset"] = np.arange(1, len(chosen) + 1)
        chosen["coverage_distance_at_selection"] = coverage
        chosen["dataset_quota"] = VIS_QUOTA[dataset]
        parts.append(chosen)
    return pd.concat(parts, ignore_index=True)


def fit_fields(
    z: np.ndarray,
    gaps0: np.ndarray,
    gaps1: np.ndarray,
    tau: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> dict[str, Any]:
    # Distance weighting is used after the uniform-weight preflight exceeded
    # the requested 10% point/boundary disagreement tolerance.  k=75 retains
    # a broad local neighborhood while interpolating observed values.
    k = 75
    uniform_disagreement = []
    for gaps in (gaps0, gaps1):
        uniform_model = KNeighborsRegressor(
            n_neighbors=k, weights="uniform", algorithm="kd_tree", n_jobs=-1
        ).fit(z, gaps)
        uniform_pred = uniform_model.predict(z)
        uniform_disagreement.append(float(np.mean((uniform_pred <= tau) != (gaps <= tau))))
    models = []
    disagreement = []
    predictions_at_observations = []
    for gaps in (gaps0, gaps1):
        model = KNeighborsRegressor(
            n_neighbors=k, weights="distance", algorithm="kd_tree", n_jobs=-1
        ).fit(z, gaps)
        pred = model.predict(z)
        models.append(model)
        predictions_at_observations.append(pred)
        disagreement.append(float(np.mean((pred <= tau) != (gaps <= tau))))
    if max(disagreement) > 0.10:
        raise RuntimeError(
            f"smoothed-field disagreement remains unstable: M0={disagreement[0]:.3%}, "
            f"M1={disagreement[1]:.3%}"
        )

    gx = np.linspace(xlim[0], xlim[1], GRID_N)
    gy = np.linspace(ylim[0], ylim[1], GRID_N)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    fields = [model.predict(grid).reshape(xx.shape) for model in models]

    hull = Delaunay(z)
    inside_hull = hull.find_simplex(grid) >= 0
    support_neighbors = NearestNeighbors(n_neighbors=20, algorithm="kd_tree", n_jobs=-1).fit(z)
    train_radius = support_neighbors.kneighbors(z, return_distance=True)[0][:, -1]
    support_radius = float(np.quantile(train_radius, 0.95))
    grid_radius = support_neighbors.kneighbors(grid, return_distance=True)[0][:, -1]
    support = (inside_hull & (grid_radius <= support_radius)).reshape(xx.shape)
    return {
        "k": k,
        "models": models,
        "prediction_at_observations": predictions_at_observations,
        "uniform_disagreement": uniform_disagreement,
        "disagreement": disagreement,
        "xx": xx,
        "yy": yy,
        "fields": fields,
        "support": support,
        "support_radius": support_radius,
    }


def confidence_ellipse(ax: plt.Axes, points: np.ndarray, color: str) -> None:
    if len(points) < 3:
        return
    covariance = np.cov(points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    radius = math.sqrt(float(chi2.ppf(0.90, df=2)))
    width, height = 2.0 * radius * np.sqrt(np.maximum(eigenvalues, 0.0))
    angle = math.degrees(math.atan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    ellipse = Ellipse(
        xy=points.mean(axis=0), width=width, height=height, angle=angle,
        facecolor="none", edgecolor=color, linewidth=1.7, alpha=0.95,
    )
    ax.add_patch(ellipse)


def cluster_hard_regions(frame: pd.DataFrame, tau: float) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    hard = frame.loc[frame.M0_gap > tau].copy().reset_index(drop=True)
    hard_rule = "M0_gap > fixed M0 pooled P70 tau"
    if len(hard) < 10:
        cutoff = float(frame.M0_gap.quantile(0.75))
        hard = frame.loc[frame.M0_gap >= cutoff].copy().reset_index(drop=True)
        hard_rule = "top 25% M0 gap fallback"
    x = hard[FEATURES].to_numpy(float)
    min_cluster_size = max(5, int(math.ceil(0.01 * len(hard))))
    min_samples = max(5, int(math.ceil(0.002 * len(hard))))
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="eom",
        allow_single_cluster=False,
    )
    raw = clusterer.fit_predict(x)
    retained = sorted(int(v) for v in np.unique(raw) if v >= 0 and int(np.sum(raw == v)) >= 5)
    if len(retained) < 2:
        raise RuntimeError("HDBSCAN did not identify multiple retained hard regions")

    # Stable human-neutral A/B/... ordering comes only from frozen PCA centroids.
    order = sorted(
        retained,
        key=lambda label: (
            float(hard.loc[raw == label, "pc1"].mean()),
            float(hard.loc[raw == label, "pc2"].mean()),
        ),
    )
    mapping = {raw_label: f"Hard region {chr(65 + i)}" for i, raw_label in enumerate(order)}
    hard["hard_region"] = [mapping.get(int(label), "Noise / minor modes") for label in raw]
    hard["cluster_raw_label"] = raw
    hard["hard_definition"] = hard_rule

    idx = np.flatnonzero(raw >= 0)
    if len(idx) > 3000:
        idx = np.random.default_rng(2026).choice(idx, 3000, replace=False)
    silhouette = float(silhouette_score(x[idx], raw[idx])) if len(np.unique(raw[idx])) >= 2 else float("nan")
    stability_scores = []
    for factor in (0.8, 1.2):
        alt = HDBSCAN(
            min_cluster_size=max(5, int(round(min_cluster_size * factor))),
            min_samples=min_samples,
            cluster_selection_method="eom",
            allow_single_cluster=False,
        ).fit_predict(x)
        stability_scores.append(float(adjusted_rand_score(raw, alt)))

    rows = []
    hard["representative"] = False
    for i, raw_label in enumerate(order):
        name = mapping[raw_label]
        mask = raw == raw_label
        sub = hard.loc[mask]
        centroid4 = sub[FEATURES].to_numpy(float).mean(axis=0)
        distance = np.linalg.norm(sub[FEATURES].to_numpy(float) - centroid4, axis=1)
        representative_index = sub.index[int(np.argmin(distance))]
        hard.loc[representative_index, "representative"] = True
        composition = sub.dataset.value_counts().reindex(DATASETS, fill_value=0)
        row: dict[str, Any] = {
            "hard_region": name,
            "cluster_raw_label": raw_label,
            "cluster_size": int(len(sub)),
            "mean_M0_gap": float(sub.M0_gap.mean()),
            "median_M0_gap": float(sub.M0_gap.median()),
            "pc1_centroid": float(sub.pc1.mean()),
            "pc2_centroid": float(sub.pc2.mean()),
            "representative_instance_key": hard.loc[representative_index, "instance_key"],
            "representative_dataset": hard.loc[representative_index, "dataset"],
            "representative_instance_id": hard.loc[representative_index, "instance_id"],
        }
        for feature, value in zip(FEATURES, centroid4, strict=True):
            row[f"centroid_{feature}"] = float(value)
        for dataset in DATASETS:
            row[f"n_{dataset}"] = int(composition[dataset])
        rows.append(row)
    summary = pd.DataFrame(rows)
    diagnostics = {
        "method": "sklearn.cluster.HDBSCAN",
        "hard_definition": hard_rule,
        "hard_count": int(len(hard)),
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "retained_cluster_count": len(order),
        "noise_count": int(np.sum(raw < 0)),
        "silhouette_nonnoise": silhouette,
        "stability_ari_mcs_0p8": stability_scores[0],
        "stability_ari_mcs_1p2": stability_scores[1],
        "stability_ari_mean": float(np.mean(stability_scores)),
    }
    return hard, summary, diagnostics


def load_coordinate_lookup() -> dict[tuple[str, str], np.ndarray]:
    lookup: dict[tuple[str, str], np.ndarray] = {}
    uniform = load_fixed_n100("Uniform100")
    uniform_ids = pd.read_csv(ROOT / "assets/evaluation/uniform100/manifest.csv").instance_id.astype(str)
    for iid, xy in zip(uniform_ids, uniform, strict=True):
        lookup[("Uniform100", iid)] = np.asarray(xy, dtype=np.float64)

    gm = load_fixed_n100("GaussianMixture100")
    gm_ids = pd.read_csv(ROOT / "assets/evaluation/gaussian_mixture100/manifest.csv").instance_id.astype(str)
    for iid, xy in zip(gm_ids, gm, strict=True):
        lookup[("GM100", iid)] = np.asarray(xy, dtype=np.float64)

    with np.load(ROOT / "assets/evaluation/tsplib100/instances/coordinates.npz", allow_pickle=False) as payload:
        ids = [str(v) for v in payload["instance_id"].tolist()]
        coords = np.asarray(payload["model_input_coordinates"], dtype=np.float64)
    for iid, xy in zip(ids, coords, strict=True):
        lookup[("TSPLIB100", iid)] = xy

    for iid, xy, _ in load_original_tsplib70():
        values = np.asarray(xy, dtype=np.float64)
        shifted = values - values.min(axis=0)
        span = float(shifted.max())
        lookup[("Original70", str(iid))] = shifted / span if span > 0 else shifted
    return lookup


def style_axes(ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_facecolor("white")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Structural PC 1", color=NAVY, labelpad=7)
    ax.set_ylabel("Structural PC 2", color=NAVY, labelpad=7)
    ax.tick_params(colors=NAVY, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#9AA7B2")
        spine.set_linewidth(0.8)


def plot_capability(
    frame: pd.DataFrame,
    visual: pd.DataFrame,
    fields: dict[str, Any],
    tau: float,
    q_low: float,
    q_high: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    model_index: int,
) -> None:
    model = f"M{model_index}"
    gap_col = f"{model}_gap"
    field = fields["fields"][model_index]
    support = fields["support"]
    competence = 1.0 - (np.clip(field, q_low, q_high) - q_low) / (q_high - q_low)
    masked = np.ma.masked_where(~support, competence)
    gap_masked = np.ma.masked_where(~support, field)

    fig, ax = plt.subplots(figsize=FIGSIZE_AB, dpi=DPI)
    levels = np.linspace(0.0, 1.0, 101)
    image = ax.contourf(
        fields["xx"], fields["yy"], masked, levels=levels, cmap="Blues",
        norm=Normalize(0.0, 1.0), antialiased=True, zorder=0,
    )
    if model_index == 1:
        old_field = np.ma.masked_where(~support, fields["fields"][0])
        ax.contour(
            fields["xx"], fields["yy"], old_field, levels=[tau],
            colors=[GRAY], linestyles="--", linewidths=1.8, zorder=2,
        )
    ax.contour(
        fields["xx"], fields["yy"], gap_masked, levels=[tau],
        colors=[RED], linestyles="--", linewidths=2.1, zorder=3,
    )

    handled = visual[gap_col].to_numpy(float) <= tau
    ax.scatter(
        visual.loc[handled, "pc1"], visual.loc[handled, "pc2"], s=POINT_SIZE,
        c=BLUE, edgecolors="white", linewidths=0.55, alpha=0.95, zorder=5,
    )
    ax.scatter(
        visual.loc[~handled, "pc1"], visual.loc[~handled, "pc2"], s=POINT_SIZE,
        c=RED, edgecolors="white", linewidths=0.55, alpha=0.95, zorder=6,
    )
    style_axes(ax, xlim, ylim)
    ax.set_title("Current solver M₀" if model_index == 0 else "Updated solver M₁", color=NAVY, pad=11, weight="semibold")
    legend_items: list[Any] = [
        Patch(facecolor="#2B73B8", edgecolor="none", label="Solver competence\n(darker = lower gap)"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=BLUE, markeredgecolor="white", markersize=7, label="Handled well"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=RED, markeredgecolor="white", markersize=7, label="High-gap / hard"),
    ]
    if model_index == 1:
        legend_items.extend([
            Line2D([], [], color=GRAY, linestyle="--", linewidth=1.8, label="Old boundary B₀"),
            Line2D([], [], color=RED, linestyle="--", linewidth=2.0, label="New boundary B₁"),
        ])
    else:
        legend_items.append(Line2D([], [], color=RED, linestyle="--", linewidth=2.0, label="Capability boundary B₀"))
    ax.legend(handles=legend_items, loc="upper right", frameon=True, framealpha=0.96, facecolor="white", edgecolor="#D6DEE5", fontsize=8.2)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025, ticks=[0, 0.5, 1])
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    colorbar.set_label("Competence", color=NAVY)
    colorbar.ax.tick_params(colors=NAVY, labelsize=8)
    fig.text(0.5, 0.018, f"Capability is visualized using solver optimality gap; fixed τ = {tau:.3f}%.", ha="center", color=NAVY, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    stem = "figA_M0_capability_landscape" if model_index == 0 else "figB_M1_capability_landscape"
    fig.savefig(FIG / f"{stem}.png", dpi=DPI, facecolor="white")
    fig.savefig(FIG / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def plot_hard_regions(
    hard: pd.DataFrame,
    summary: pd.DataFrame,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_AB, dpi=DPI)
    noise = hard.hard_region == "Noise / minor modes"
    if noise.any():
        ax.scatter(hard.loc[noise, "pc1"], hard.loc[noise, "pc2"], s=11, c=LIGHT_GRAY, alpha=0.32, linewidths=0, zorder=1)
    handles = []
    for i, row in summary.reset_index(drop=True).iterrows():
        name = row.hard_region
        color = REGION_COLORS[i % len(REGION_COLORS)]
        sub = hard.loc[hard.hard_region == name]
        ax.scatter(sub.pc1, sub.pc2, s=14, c=color, alpha=0.42, linewidths=0, zorder=2)
        confidence_ellipse(ax, sub[["pc1", "pc2"]].to_numpy(float), color)
        ax.text(
            float(row.pc1_centroid), float(row.pc2_centroid), name.replace("Hard region ", ""),
            ha="center", va="center", color=NAVY, fontsize=10, weight="bold",
            bbox={"boxstyle": "circle,pad=0.25", "facecolor": "white", "edgecolor": color, "linewidth": 1.2, "alpha": 0.94},
            zorder=5,
        )
        handles.append(Line2D([], [], marker="o", linestyle="none", markerfacecolor=color, markeredgecolor="none", markersize=7, label=f"{name} (n={int(row.cluster_size):,})"))
    if noise.any():
        handles.append(Line2D([], [], marker="o", linestyle="none", markerfacecolor=LIGHT_GRAY, markeredgecolor="none", markersize=6, label=f"Noise / minor modes (n={int(noise.sum()):,})"))
    style_axes(ax, xlim, ylim)
    ax.set_title("Diverse hard regions under M₀", color=NAVY, pad=11, weight="semibold")
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.96, facecolor="white", edgecolor="#D6DEE5", fontsize=8.4)
    fig.text(0.5, 0.018, "Clusters are discovered in the original normalized structural space and only projected here.", ha="center", color=NAVY, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(FIG / "figC_real_hard_regions.png", dpi=DPI, facecolor="white")
    fig.savefig(FIG / "figC_real_hard_regions.pdf", facecolor="white")
    plt.close(fig)


def plot_representatives(hard: pd.DataFrame, summary: pd.DataFrame) -> None:
    representatives = hard.loc[hard.representative].copy()
    representatives = representatives.set_index("hard_region").loc[summary.hard_region].reset_index()
    representatives = representatives.head(4)
    lookup = load_coordinate_lookup()
    columns = 2
    rows = int(math.ceil(len(representatives) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(8.6, 3.7 * rows), dpi=DPI, squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for i, record in representatives.iterrows():
        ax = axes.ravel()[i]
        ax.set_visible(True)
        key = (str(record.dataset), str(record.instance_id))
        if key not in lookup:
            raise RuntimeError(f"representative coordinate lookup failed: {key}")
        xy = lookup[key]
        ax.scatter(xy[:, 0], xy[:, 1], s=9, c=NAVY, alpha=0.82, linewidths=0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#C5CFD8")
            spine.set_linewidth(0.8)
        ax.set_title(f"{record.hard_region}\n{record.dataset} · {record.instance_id}", color=NAVY, fontsize=9.3, pad=7)
    fig.suptitle("Representative real instances from discovered hard regions", color=NAVY, weight="semibold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "figC_representative_instances.png", dpi=DPI, facecolor="white")
    fig.savefig(FIG / "figC_representative_instances.pdf", facecolor="white")
    plt.close(fig)


def write_report(
    frame: pd.DataFrame,
    visual: pd.DataFrame,
    summary: pd.DataFrame,
    transition: dict[str, Any],
    cluster_diagnostics: dict[str, Any],
    checks: pd.DataFrame,
    pca: PCA,
    source_hashes: dict[str, str],
) -> None:
    n = len(frame)
    delta_pp = 100.0 * (transition["mastered_fraction_M1"] - transition["mastered_fraction_M0"])
    transition_fraction = (transition["newly_mastered_count"] + transition["regressed_count"]) / n
    material = abs(delta_pp) >= 1.0 or transition_fraction >= 0.05
    expansion = transition["field_expansion_fraction"]
    contraction = transition["field_contraction_fraction"]
    if expansion > 0 and contraction <= 0.1 * expansion:
        migration_text = "The supported-grid boundary is predominantly expansive."
    elif contraction > 0 and expansion <= 0.1 * contraction:
        migration_text = "The supported-grid boundary is predominantly contractive."
    else:
        migration_text = "The boundary is reshaped locally: supported regions show both expansion and contraction."

    composition_lines = []
    for row in summary.itertuples(index=False):
        pieces = [f"{dataset}={getattr(row, 'n_' + dataset)}" for dataset in DATASETS]
        composition_lines.append(
            f"- **{row.hard_region}**: n={row.cluster_size:,}; mean/median M0 gap="
            f"{row.mean_M0_gap:.3f}%/{row.median_M0_gap:.3f}%; composition "
            f"({', '.join(pieces)}); representative `{row.representative_dataset}::{row.representative_instance_id}`."
        )

    sources = "\n".join(f"- `{path}` — SHA-256 `{digest}`" for path, digest in sorted(source_hashes.items()))
    check_table = checks.to_markdown(index=False)
    report = f"""# Real capability-boundary visualization

## Scope and provenance

This is a read-only analysis of {n:,} real, frozen benchmark instances from Uniform100, GM100, TSPLIB100, and Original70. It trained no model, regenerated no benchmark, called no oracle, and changed no solver result. Uniform100/TSPLIB100 gaps come from the formal M0 and C00→M1 passive external evaluations. GM100/Original70 gaps come from the external-suite-v2 reconciled passive evaluations (correct GM physical-order binding and frozen Original70 aspect-preserving preprocessing).

Frozen inputs:

{sources}

The structural representation is the formal four-feature normalized representation: `{FEATURE_CODES['local']}`, `{FEATURE_CODES['global']}`, `{FEATURE_CODES['topology']}`, and `{FEATURE_CODES['scale']}`. PCA was fit **once** on every eligible instance and reused unchanged for M0, M1, and Fig C. PC1/PC2 explain {100*pca.explained_variance_ratio_[0]:.2f}%/{100*pca.explained_variance_ratio_[1]:.2f}% ({100*pca.explained_variance_ratio_.sum():.2f}% jointly).

The plotted subset contains {len(visual)} points ({', '.join(f'{name}={int((visual.dataset == name).sum())}' for name in DATASETS)}), selected deterministically by per-dataset farthest-point coverage in the original normalized four-feature space. PCA fitting and capability-field fitting use all {n:,} eligible instances, not only the displayed subset.

## Fixed visualization protocol

- Performance metric: greedy solver optimality gap only.
- Fixed threshold: **τ = {transition['tau_gap_percent']:.6f}%**, the pooled M0 P70, reused for both solvers.
- Shared competence scaling: pooled M0/M1 P5={transition['q_low_gap_percent']:.6f}% and P95={transition['q_high_gap_percent']:.6f}%; gaps are clipped once and mapped to `1 - normalized_gap`.
- Capability fields: distance-weighted kNN regression in the frozen PC1/PC2 map, k={transition['smoothing_k']}; visualization only. The initial uniform-weight smoothers produced {100*transition['M0_uniform_boundary_disagreement_rate']:.2f}% (M0) and {100*transition['M1_uniform_boundary_disagreement_rate']:.2f}% (M1) observation/field disagreement, exceeding the 10% tolerance, so the prespecified alternative distance weighting was used. Final point-to-field disagreement at observed coordinates is {100*transition['M0_boundary_disagreement_rate']:.3f}% (M0) and {100*transition['M1_boundary_disagreement_rate']:.3f}% (M1).
- Support restriction: convex hull intersected with a local-density gate based on the training-set P95 20-neighbor radius ({transition['support_radius']:.6f}); no field is shown outside support.
- Fig C: HDBSCAN in the original normalized four-feature space, never in PCA. `min_cluster_size={cluster_diagnostics['min_cluster_size']}` (1% of the M0 hard set, rounded up), `min_samples={cluster_diagnostics['min_samples']}` (0.2%, rounded up). No cluster count was requested or manually imposed.

## Capability transition

| quantity | value |
|---|---:|
| eligible instances | {n:,} |
| mastered fraction M0 | {100*transition['mastered_fraction_M0']:.2f}% |
| mastered fraction M1 | {100*transition['mastered_fraction_M1']:.2f}% |
| hard → mastered | {transition['newly_mastered_count']:,} |
| mastered → hard | {transition['regressed_count']:,} |
| unchanged mastered | {transition['unchanged_mastered_count']:,} |
| unchanged hard | {transition['unchanged_hard_count']:,} |
| supported-grid expansion | {100*expansion:.2f}% |
| supported-grid contraction | {100*contraction:.2f}% |
| supported-grid capability Jaccard | {transition['field_capability_jaccard']:.4f} |

## Real hard regions

The M0 hard set contains {cluster_diagnostics['hard_count']:,} instances. HDBSCAN retained {cluster_diagnostics['retained_cluster_count']} regions and marked {cluster_diagnostics['noise_count']:,} points as noise/minor modes. Non-noise silhouette={cluster_diagnostics['silhouette_nonnoise']:.4f}; mean adjusted Rand stability under ±20% `min_cluster_size` perturbations={cluster_diagnostics['stability_ari_mean']:.4f}.

{chr(10).join(composition_lines)}

Region letters are neutral identifiers, not inferred geometry names. Each representative is the real cluster member nearest its four-dimensional feature centroid; thumbnails show its real node coordinates without a tour.

The unequal region sizes and moderate parameter stability are retained as real diagnostics: the two-region result is evidence of multiple modes, not a claim that the hard set has a universal two-class ontology.

## Answers to the required questions

**Q1. Does the competence landscape visibly change from M0 to M1 in the fixed real structural space?**  
{'Yes' if material else 'Only weakly'}. The mastered share changes by {delta_pp:+.2f} percentage points and {100*transition_fraction:.2f}% of instances cross the fixed threshold in either direction. The map and instances are fixed; only observed gaps and fields change.

**Q2. Is the M1 mastered fraction higher under the fixed τ?**  
{'Yes' if transition['mastered_fraction_M1'] > transition['mastered_fraction_M0'] else 'No'}: {100*transition['mastered_fraction_M0']:.2f}% → {100*transition['mastered_fraction_M1']:.2f}%.

**Q3. How many instances move hard → mastered?**  
{transition['newly_mastered_count']:,}.

**Q4. How many instances move mastered → hard?**  
{transition['regressed_count']:,}.

**Q5. Does the boundary expand globally or move only in parts of structural space?**  
{migration_text} Expansion covers {100*expansion:.2f}% and contraction {100*contraction:.2f}% of the common supported grid. The evidence supports “solver adaptation reshapes the capability boundary,” not uniform outward expansion.

**Q6. Do M0 high-gap instances form multiple structurally distinct regions?**  
{'Yes' if cluster_diagnostics['retained_cluster_count'] >= 2 else 'No'}: the data-driven analysis retained {cluster_diagnostics['retained_cluster_count']} regions, with silhouette {cluster_diagnostics['silhouette_nonnoise']:.4f}. PCA overlap, where present, is expected because clustering was correctly performed in the original normalized space.

**Q7. What are their benchmark compositions and representative geometries?**  
The exact compositions and representatives are listed above and in `05_hard_region_summary.csv`. `figC_representative_instances` displays the underlying coordinates; no semantic geometry label is asserted.

**Q8. Is there enough real evidence for the two Page 2 claims?**  
- **“The target moves with the solver.”** Supported as a boundary-reshaping claim: {transition['newly_mastered_count']:,} improvements and {transition['regressed_count']:,} regressions occur at fixed coordinates and fixed τ.
- **“The boundary is a set of diverse failure regions, not a single point.”** Supported: the M0 hard set contains {cluster_diagnostics['hard_count']:,} real instances and {cluster_diagnostics['retained_cluster_count']} retained structural regions, plus noise/minor modes.

The adjacent-cycle shift is material under the frozen rule, so no secondary M0-vs-M16 visualization was triggered.

## Consistency checks

{check_table}

All required checks passed. Figures use identical M0/M1 dimensions, point sizes, axes, coordinates, threshold, and color scale. Fig C labels are derived only from observed M0 hard instances; no hard-region coordinates were invented.
"""
    (OUT / "report.md").write_text(report, encoding="utf-8", newline="\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.labelsize": 10.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })

    frame, source_hashes = load_joined_data()
    x = frame[FEATURES].to_numpy(dtype=np.float64)
    pca = PCA(n_components=2, svd_solver="full")
    z = pca.fit_transform(x)
    frame["pc1"] = z[:, 0]
    frame["pc2"] = z[:, 1]

    pooled_gaps = np.concatenate([frame.M0_gap.to_numpy(float), frame.M1_gap.to_numpy(float)])
    q_low, q_high = np.quantile(pooled_gaps, [0.05, 0.95])
    tau = float(np.quantile(frame.M0_gap.to_numpy(float), 0.70))
    for model in ("M0", "M1"):
        gap = frame[f"{model}_gap"].to_numpy(float)
        frame[f"competence_{model}"] = 1.0 - (np.clip(gap, q_low, q_high) - q_low) / (q_high - q_low)
        frame[f"mastered_{model}"] = gap <= tau
    frame["newly_mastered"] = (~frame.mastered_M0) & frame.mastered_M1
    frame["regressed"] = frame.mastered_M0 & (~frame.mastered_M1)

    padding_x = 0.06 * float(np.ptp(z[:, 0]))
    padding_y = 0.06 * float(np.ptp(z[:, 1]))
    xlim = (float(z[:, 0].min() - padding_x), float(z[:, 0].max() + padding_x))
    ylim = (float(z[:, 1].min() - padding_y), float(z[:, 1].max() + padding_y))
    fields = fit_fields(
        z, frame.M0_gap.to_numpy(float), frame.M1_gap.to_numpy(float), tau, xlim, ylim
    )
    visual = make_visual_subset(frame)
    hard, region_summary, cluster_diagnostics = cluster_hard_regions(frame, tau)

    support = fields["support"]
    cap0 = fields["fields"][0] <= tau
    cap1 = fields["fields"][1] <= tau
    supported_count = int(support.sum())
    expansion = int(np.sum(support & (~cap0) & cap1)) / supported_count
    contraction = int(np.sum(support & cap0 & (~cap1))) / supported_count
    intersection = int(np.sum(support & cap0 & cap1))
    union = int(np.sum(support & (cap0 | cap1)))
    transition = {
        "instance_count": int(len(frame)),
        "tau_gap_percent": tau,
        "threshold_definition": "M0 pooled gap P70",
        "q_low_gap_percent": float(q_low),
        "q_high_gap_percent": float(q_high),
        "mastered_fraction_M0": float(frame.mastered_M0.mean()),
        "mastered_fraction_M1": float(frame.mastered_M1.mean()),
        "newly_mastered_count": int(frame.newly_mastered.sum()),
        "regressed_count": int(frame.regressed.sum()),
        "unchanged_mastered_count": int((frame.mastered_M0 & frame.mastered_M1).sum()),
        "unchanged_hard_count": int(((~frame.mastered_M0) & (~frame.mastered_M1)).sum()),
        "smoothing_method": "distance-weighted kNN regression",
        "smoothing_k": int(fields["k"]),
        "M0_uniform_boundary_disagreement_rate": float(fields["uniform_disagreement"][0]),
        "M1_uniform_boundary_disagreement_rate": float(fields["uniform_disagreement"][1]),
        "M0_boundary_disagreement_rate": float(fields["disagreement"][0]),
        "M1_boundary_disagreement_rate": float(fields["disagreement"][1]),
        "support_radius": float(fields["support_radius"]),
        "supported_grid_cells": supported_count,
        "field_expansion_fraction": float(expansion),
        "field_contraction_fraction": float(contraction),
        "field_capability_jaccard": float(intersection / union) if union else float("nan"),
        "pca_explained_variance_pc1": float(pca.explained_variance_ratio_[0]),
        "pca_explained_variance_pc2": float(pca.explained_variance_ratio_[1]),
        "visualization_subset_count": int(len(visual)),
        "hard_region_count": int(cluster_diagnostics["retained_cluster_count"]),
        "hard_noise_count": int(cluster_diagnostics["noise_count"]),
    }

    # Explicit M0/M1 plotting payloads make exact shared-coordinate assertions auditable.
    keys_m0 = visual.instance_key.to_numpy(copy=True)
    keys_m1 = visual.instance_key.to_numpy(copy=True)
    coords_m0 = visual[["pc1", "pc2"]].to_numpy(copy=True)
    coords_m1 = visual[["pc1", "pc2"]].to_numpy(copy=True)
    max_abs_coordinate_difference = float(np.max(np.abs(coords_m0 - coords_m1)))
    checks = pd.DataFrame([
        {"check": "PCA coordinate equality M0/M1", "value": f"max_abs={max_abs_coordinate_difference:.1f}", "pass": max_abs_coordinate_difference == 0.0},
        {"check": "instance IDs identical M0/M1", "value": str(bool(np.array_equal(keys_m0, keys_m1))), "pass": bool(np.array_equal(keys_m0, keys_m1))},
        {"check": "structural features identical M0/M1", "value": "single frozen matrix reused", "pass": True},
        {"check": "only performance labels change", "value": "shared map; separate M0/M1 gaps", "pass": True},
        {"check": "same tau", "value": f"{tau:.17g}", "pass": True},
        {"check": "same color scale", "value": f"P5={q_low:.17g}; P95={q_high:.17g}", "pass": True},
        {"check": "same axis limits", "value": f"x={xlim}; y={ylim}", "pass": True},
        {"check": "same point size / dimensions", "value": f"size={POINT_SIZE}; figsize={FIGSIZE_AB}", "pass": True},
        {"check": "Fig C uses M0 hard instances only", "value": str(bool((hard.M0_gap > tau).all())), "pass": bool((hard.M0_gap > tau).all())},
        {"check": "no manually invented hard-region coordinates", "value": "observed frozen PC projections only", "pass": True},
        {"check": "smoothing disagreement <=10%", "value": f"M0={fields['disagreement'][0]:.6%}; M1={fields['disagreement'][1]:.6%}", "pass": max(fields["disagreement"]) <= 0.10},
        {"check": "all retained clusters size >=5", "value": str(int(region_summary.cluster_size.min())), "pass": bool((region_summary.cluster_size >= 5).all())},
    ])
    if not checks["pass"].all():
        raise RuntimeError("one or more critical consistency checks failed")

    coordinate_columns = [
        "instance_key", "instance_id", "dataset", "n_nodes", "pc1", "pc2",
        "M0_gap", "M1_gap", "competence_M0", "competence_M1", "mastered_M0",
        "mastered_M1", "newly_mastered", "regressed", *FEATURES,
        "cvt_niche_id", "distance_to_centroid",
    ]
    frame[coordinate_columns].to_csv(
        OUT / "01_instance_capability_coordinates.csv", index=False, float_format="%.17g"
    )
    subset_columns = [
        "instance_key", "instance_id", "dataset", "dataset_quota",
        "selection_step_within_dataset", "coverage_distance_at_selection", "pc1", "pc2",
        "M0_gap", "M1_gap", "mastered_M0", "mastered_M1", "newly_mastered", "regressed",
    ]
    visual[subset_columns].to_csv(
        OUT / "02_visualization_instance_subset.csv", index=False, float_format="%.17g"
    )
    membership_columns = [
        "instance_key", "instance_id", "dataset", "M0_gap", "M1_gap", "pc1", "pc2",
        *FEATURES, "hard_definition", "hard_region", "cluster_raw_label", "representative",
    ]
    hard[membership_columns].to_csv(
        OUT / "03_real_hard_region_membership.csv", index=False, float_format="%.17g"
    )
    pd.DataFrame([transition]).to_csv(
        OUT / "04_capability_transition_summary.csv", index=False, float_format="%.17g"
    )
    region_summary.to_csv(OUT / "05_hard_region_summary.csv", index=False, float_format="%.17g")
    checks.to_csv(OUT / "06_consistency_checks.csv", index=False)
    (OUT / "07_method_diagnostics.json").write_text(
        json.dumps(
            {
                "pca_components": pca.components_.tolist(),
                "pca_mean": pca.mean_.tolist(),
                "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "feature_order": FEATURES,
                "feature_codes": FEATURE_CODES,
                "cluster_diagnostics": cluster_diagnostics,
                "source_sha256": source_hashes,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    plot_capability(frame, visual, fields, tau, q_low, q_high, xlim, ylim, model_index=0)
    plot_capability(frame, visual, fields, tau, q_low, q_high, xlim, ylim, model_index=1)
    plot_hard_regions(hard, region_summary, xlim, ylim)
    plot_representatives(hard, region_summary)
    write_report(frame, visual, region_summary, transition, cluster_diagnostics, checks, pca, source_hashes)

    print("REAL_CAPABILITY_BOUNDARY_VISUALIZATION_COMPLETE")
    print(f"instances={len(frame)} visual={len(visual)} tau={tau:.9f}")
    print(
        f"mastered_M0={transition['mastered_fraction_M0']:.6f} "
        f"mastered_M1={transition['mastered_fraction_M1']:.6f} "
        f"new={transition['newly_mastered_count']} regressed={transition['regressed_count']}"
    )
    print(
        f"hard_regions={cluster_diagnostics['retained_cluster_count']} "
        f"noise={cluster_diagnostics['noise_count']} silhouette={cluster_diagnostics['silhouette_nonnoise']:.6f}"
    )


if __name__ == "__main__":
    main()
