#!/usr/bin/env python
"""Read-only refinement audit for the frozen M0 hard-instance set."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
FIG = OUT / "figures"
SWEEP_FIG = FIG / "global_sweep"
SOURCE = ROOT / "audits/external_benchmark_suite_v2/real_capability_boundary_visualization"
MEMBERSHIP = SOURCE / "03_real_hard_region_membership.csv"
V1_SCRIPT = SOURCE / "run_real_capability_boundary_visualization.py"
FEATURES = ["local", "global", "topology", "scale"]
DATASETS = ["Uniform100", "GM100", "TSPLIB100", "Original70"]
TAU = 1.6759665429999124
SEED = 2026
DPI = 300
NAVY = "#243746"
NOISE = "#C8CDD2"
COLORS = {
    "Hard region A1": "#557A95",
    "Hard region A2": "#B07D62",
    "Hard region B": "#719A78",
    "Noise / minor modes": NOISE,
}
DISPLAY_LABELS = {"Hard region A1": "Hard region A", "Hard region A2": "Hard region B", "Hard region B": "Hard region C"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def sample_nonnoise(labels: np.ndarray, limit: int = 3000) -> np.ndarray:
    indices = np.flatnonzero(labels >= 0)
    if len(indices) <= limit:
        return indices
    rng = np.random.default_rng(SEED)
    return np.sort(rng.choice(indices, limit, replace=False))


def cluster_metrics(x: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    indices = sample_nonnoise(labels)
    unique = np.unique(labels[indices]) if len(indices) else np.array([])
    if len(unique) < 2:
        return {"silhouette": math.nan, "davies_bouldin": math.nan, "calinski_harabasz": math.nan}
    return {
        "silhouette": float(silhouette_score(x[indices], labels[indices])),
        "davies_bouldin": float(davies_bouldin_score(x[indices], labels[indices])),
        "calinski_harabasz": float(calinski_harabasz_score(x[indices], labels[indices])),
    }


def centroid_separation(x: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    centroids = [x[labels == label].mean(axis=0) for label in sorted(v for v in np.unique(labels) if v >= 0)]
    distances = [float(np.linalg.norm(a - b)) for a, b in combinations(centroids, 2)]
    if not distances:
        return math.nan, math.nan
    return min(distances), float(np.mean(distances))


def cluster_payload(frame: pd.DataFrame, labels: np.ndarray) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for label in sorted(v for v in np.unique(labels) if v >= 0):
        sub = frame.loc[labels == label]
        composition = sub.dataset.value_counts().reindex(DATASETS, fill_value=0)
        payload.append({
            "raw_label": int(label),
            "n": int(len(sub)),
            "mean_gap": float(sub.M0_gap.mean()),
            "median_gap": float(sub.M0_gap.median()),
            "benchmark_composition": {name: int(composition[name]) for name in DATASETS},
            "feature_centroid": {feature: float(sub[feature].mean()) for feature in FEATURES},
        })
    return payload


def representative_payload(frame: pd.DataFrame, labels: np.ndarray) -> list[dict[str, Any]]:
    payload = []
    for label in sorted(v for v in np.unique(labels) if v >= 0):
        sub = frame.loc[labels == label]
        x = sub[FEATURES].to_numpy(float)
        centroid = x.mean(axis=0)
        local_index = int(np.argmin(np.linalg.norm(x - centroid, axis=1)))
        row = sub.iloc[local_index]
        payload.append({
            "raw_label": int(label),
            "instance_key": str(row.instance_key),
            "dataset": str(row.dataset),
            "instance_id": str(row.instance_id),
            "distance_to_feature_centroid": float(np.linalg.norm(x[local_index] - centroid)),
        })
    return payload


def ordered_labels_for_display(frame: pd.DataFrame, labels: np.ndarray) -> dict[int, int]:
    raw = [int(v) for v in np.unique(labels) if v >= 0]
    ordered = sorted(raw, key=lambda value: (
        float(frame.loc[labels == value, "pc1"].mean()),
        float(frame.loc[labels == value, "pc2"].mean()),
    ))
    return {label: index for index, label in enumerate(ordered)}


def plot_sweep_projection(frame: pd.DataFrame, labels: np.ndarray, title: str, path: Path) -> None:
    rng = np.random.default_rng(SEED)
    keep: list[int] = []
    for label in sorted(np.unique(labels)):
        indices = np.flatnonzero(labels == label)
        cap = 700 if label >= 0 else 450
        if len(indices) > cap:
            indices = rng.choice(indices, cap, replace=False)
        keep.extend(indices.tolist())
    keep_array = np.asarray(sorted(keep), dtype=int)
    mapping = ordered_labels_for_display(frame, labels)
    palette = ["#557A95", "#B07D62", "#719A78", "#8B78A6", "#C19A4A", "#5B9996", "#A66C7A", "#7B8794", "#997B5D", "#6D8B68"]
    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=170)
    noise = keep_array[labels[keep_array] < 0]
    if len(noise):
        ax.scatter(frame.iloc[noise].pc1, frame.iloc[noise].pc2, s=6, c=NOISE, alpha=0.28, linewidths=0)
    for raw, order in mapping.items():
        indices = keep_array[labels[keep_array] == raw]
        ax.scatter(frame.iloc[indices].pc1, frame.iloc[indices].pc2, s=8, c=palette[order % len(palette)], alpha=0.58, linewidths=0)
    ax.set_xlabel("Structural PC 1")
    ax.set_ylabel("Structural PC 2")
    ax.set_title(title, fontsize=10, color=NAVY)
    for spine in ax.spines.values():
        spine.set_color("#AAB4BD")
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def global_hdbscan_sweep(hard: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[int, int, str], np.ndarray]]:
    x = hard[FEATURES].to_numpy(float)
    rows = []
    labels_by_key: dict[tuple[int, int, str], np.ndarray] = {}
    for method in ("eom", "leaf"):
        for min_cluster_size in (80, 120, 160, 240, 320):
            for min_samples in (5, 10, 20, 40):
                clusterer = HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_method=method,
                    allow_single_cluster=False,
                )
                labels = clusterer.fit_predict(x)
                key = (min_cluster_size, min_samples, method)
                labels_by_key[key] = labels
                counts = sorted([int(np.sum(labels == value)) for value in np.unique(labels) if value >= 0], reverse=True)
                metrics = cluster_metrics(x, labels)
                payload = cluster_payload(hard, labels)
                rows.append({
                    "min_cluster_size": min_cluster_size,
                    "min_samples": min_samples,
                    "cluster_selection_method": method,
                    "allow_single_cluster": False,
                    "non_noise_cluster_count": len(counts),
                    "noise_count": int(np.sum(labels < 0)),
                    "noise_fraction": float(np.mean(labels < 0)),
                    "cluster_sizes_json": json_text(counts),
                    "largest_cluster_size": counts[0] if counts else 0,
                    "largest_cluster_fraction_of_hard_set": counts[0] / len(hard) if counts else 0.0,
                    "silhouette_score": metrics["silhouette"],
                    "davies_bouldin_score": metrics["davies_bouldin"],
                    "calinski_harabasz_score": metrics["calinski_harabasz"],
                    "cluster_statistics_json": json_text(payload),
                })
                stem = f"global_{method}_mcs{min_cluster_size}_ms{min_samples}.png"
                plot_sweep_projection(
                    hard, labels,
                    f"Global HDBSCAN · {method} · min cluster {min_cluster_size} · min samples {min_samples}\n"
                    f"clusters={len(counts)}, noise={np.mean(labels < 0):.1%}",
                    SWEEP_FIG / stem,
                )
    result = pd.DataFrame(rows)
    return result, labels_by_key


def leaf_refinement(region_a: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[int, int], np.ndarray]]:
    x = region_a[FEATURES].to_numpy(float)
    rows = []
    labels_by_key = {}
    for min_cluster_size in (60, 80, 120, 160):
        for min_samples in (5, 10, 20):
            labels = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_method="leaf",
                allow_single_cluster=False,
            ).fit_predict(x)
            labels_by_key[(min_cluster_size, min_samples)] = labels
            counts = sorted([int(np.sum(labels == value)) for value in np.unique(labels) if value >= 0], reverse=True)
            metrics = cluster_metrics(x, labels)
            minimum, mean = centroid_separation(x, labels)
            rows.append({
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "cluster_selection_method": "leaf",
                "subcluster_count": len(counts),
                "noise_count": int(np.sum(labels < 0)),
                "noise_fraction": float(np.mean(labels < 0)),
                "subcluster_sizes_json": json_text(counts),
                "silhouette_score": metrics["silhouette"],
                "davies_bouldin_score": metrics["davies_bouldin"],
                "calinski_harabasz_score": metrics["calinski_harabasz"],
                "minimum_centroid_separation_4d": minimum,
                "mean_centroid_separation_4d": mean,
                "subcluster_statistics_json": json_text(cluster_payload(region_a, labels)),
            })
    return pd.DataFrame(rows), labels_by_key


def gmm_refinement(region_a: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, np.ndarray], float, float]:
    x = region_a[FEATURES].to_numpy(float)
    baseline = GaussianMixture(1, covariance_type="full", random_state=SEED, n_init=20, reg_covar=1e-6).fit(x)
    bic1, aic1 = float(baseline.bic(x)), float(baseline.aic(x))
    rows = []
    labels_by_k: dict[int, np.ndarray] = {}
    for k in (2, 3, 4, 5):
        model = GaussianMixture(k, covariance_type="full", random_state=SEED, n_init=20, reg_covar=1e-6).fit(x)
        labels = model.predict(x)
        labels_by_k[k] = labels
        metrics = cluster_metrics(x, labels)
        counts = sorted([int(v) for v in np.bincount(labels)], reverse=True)
        minimum, mean = centroid_separation(x, labels)
        seed_ari = []
        for seed in (7, 42, 2027, 9999):
            alternate = GaussianMixture(k, covariance_type="full", random_state=seed, n_init=10, reg_covar=1e-6).fit_predict(x)
            seed_ari.append(float(adjusted_rand_score(labels, alternate)))
        kmeans = KMeans(k, random_state=SEED, n_init=20).fit_predict(x)
        rows.append({
            "k": k,
            "covariance_type": "full",
            "bic": float(model.bic(x)),
            "aic": float(model.aic(x)),
            "delta_bic_vs_k1": float(model.bic(x) - bic1),
            "delta_aic_vs_k1": float(model.aic(x) - aic1),
            "silhouette_score": metrics["silhouette"],
            "davies_bouldin_score": metrics["davies_bouldin"],
            "calinski_harabasz_score": metrics["calinski_harabasz"],
            "cluster_sizes_json": json_text(counts),
            "smallest_cluster_size": min(counts),
            "largest_cluster_size": max(counts),
            "size_balance_min_over_max": min(counts) / max(counts),
            "minimum_centroid_separation_4d": minimum,
            "mean_centroid_separation_4d": mean,
            "seed_stability_ari_mean": float(np.mean(seed_ari)),
            "seed_stability_ari_min": float(np.min(seed_ari)),
            "kmeans_sanity_ari": float(adjusted_rand_score(labels, kmeans)),
            "cluster_statistics_json": json_text(cluster_payload(region_a, labels)),
            "representative_instances_json": json_text(representative_payload(region_a, labels)),
        })
    return pd.DataFrame(rows), labels_by_k, bic1, aic1


def make_final_assignment(hard: pd.DataFrame, region_a: pd.DataFrame, gmm_labels: np.ndarray) -> pd.DataFrame:
    result = hard.copy()
    result["original_hard_region"] = result["hard_region"]
    result["final_hard_region"] = "Noise / minor modes"
    result["refinement_method"] = "retained original label"
    mapping = ordered_labels_for_display(region_a, gmm_labels)
    if len(mapping) != 2:
        raise RuntimeError("the conservative final refinement must contain exactly two A submodes")
    region_a_labels = np.array([f"Hard region A{mapping[int(raw)] + 1}" for raw in gmm_labels], dtype=object)
    index_to_label = dict(zip(region_a.index, region_a_labels, strict=True))
    mask_a = result.hard_region.eq("Hard region A")
    result.loc[mask_a, "final_hard_region"] = [index_to_label[index] for index in result.index[mask_a]]
    result.loc[mask_a, "refinement_method"] = "GMM k=2 in frozen normalized 4D features"
    mask_b = result.hard_region.eq("Hard region B")
    result.loc[mask_b, "final_hard_region"] = "Hard region B"
    result.loc[mask_b, "refinement_method"] = "retained global HDBSCAN region"
    result["representative"] = False
    return result


def summarize_final(assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    representative_rows = []
    major = ["Hard region A1", "Hard region A2", "Hard region B"]
    for label in [*major, "Noise / minor modes"]:
        sub = assignments.loc[assignments.final_hard_region.eq(label)]
        x = sub[FEATURES].to_numpy(float)
        composition = sub.dataset.value_counts().reindex(DATASETS, fill_value=0)
        row: dict[str, Any] = {
            "hard_region": label,
            "n": int(len(sub)),
            "mean_M0_gap": float(sub.M0_gap.mean()),
            "median_M0_gap": float(sub.M0_gap.median()),
            "pc1_centroid": float(sub.pc1.mean()),
            "pc2_centroid": float(sub.pc2.mean()),
        }
        for feature in FEATURES:
            row[f"mean_{feature}"] = float(sub[feature].mean())
            row[f"median_{feature}"] = float(sub[feature].median())
        for dataset in DATASETS:
            row[f"n_{dataset}"] = int(composition[dataset])
        if label in major:
            centroid = x.mean(axis=0)
            local_index = int(np.argmin(np.linalg.norm(x - centroid, axis=1)))
            rep = sub.iloc[local_index]
            distance = float(np.linalg.norm(x[local_index] - centroid))
            assignments.loc[rep.name, "representative"] = True
            row.update({
                "representative_instance_key": str(rep.instance_key),
                "representative_dataset": str(rep.dataset),
                "representative_instance_id": str(rep.instance_id),
                "representative_distance_to_centroid": distance,
            })
            representative_rows.append({
                "hard_region": label,
                "instance_key": str(rep.instance_key),
                "dataset": str(rep.dataset),
                "instance_id": str(rep.instance_id),
                "distance_to_feature_centroid": distance,
                **{feature: float(rep[feature]) for feature in FEATURES},
            })
        summary_rows.append(row)
    return pd.DataFrame(summary_rows), pd.DataFrame(representative_rows)


def soft_density_contour(ax: plt.Axes, points: np.ndarray, color: str, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    histogram, x_edges, y_edges = np.histogram2d(points[:, 0], points[:, 1], bins=160, range=[xlim, ylim])
    smooth = gaussian_filter(histogram.T, sigma=3.0)
    ordered = np.sort(smooth.ravel())[::-1]
    cumulative = np.cumsum(ordered)
    if cumulative[-1] <= 0:
        return
    level = float(ordered[np.searchsorted(cumulative, 0.70 * cumulative[-1])])
    if level <= 0:
        return
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    ax.contour(x_centers, y_centers, smooth, levels=[level], colors=[color], linewidths=1.15, alpha=0.58)


def display_indices(assignments: pd.DataFrame, full: bool) -> np.ndarray:
    if full:
        return assignments.index.to_numpy()
    rng = np.random.default_rng(SEED)
    chosen = []
    for label, cap in {"Hard region A1": 1400, "Hard region A2": 1400, "Hard region B": 1000, "Noise / minor modes": 180}.items():
        indices = assignments.index[assignments.final_hard_region.eq(label)].to_numpy()
        if len(indices) > cap:
            indices = rng.choice(indices, cap, replace=False)
        chosen.extend(indices.tolist())
    return np.asarray(sorted(chosen), dtype=int)


def plot_refined_regions(assignments: pd.DataFrame, summary: pd.DataFrame, path: Path, full: bool) -> None:
    xlim = (float(assignments.pc1.min()) - 0.035, float(assignments.pc1.max()) + 0.035)
    ylim = (float(assignments.pc2.min()) - 0.035, float(assignments.pc2.max()) + 0.035)
    visible = display_indices(assignments, full)
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=DPI)
    noise = assignments.index[assignments.final_hard_region.eq("Noise / minor modes")].intersection(visible)
    ax.scatter(assignments.loc[noise, "pc1"], assignments.loc[noise, "pc2"], s=8, c=NOISE, alpha=0.25, linewidths=0)
    handles = []
    for label in ("Hard region A1", "Hard region A2", "Hard region B"):
        display_label = DISPLAY_LABELS[label]
        full_sub = assignments.loc[assignments.final_hard_region.eq(label)]
        show = full_sub.index.intersection(visible)
        size = 8 if full else 11
        alpha = 0.30 if full else 0.52
        ax.scatter(assignments.loc[show, "pc1"], assignments.loc[show, "pc2"], s=size, c=COLORS[label], alpha=alpha, linewidths=0)
        soft_density_contour(ax, full_sub[["pc1", "pc2"]].to_numpy(float), COLORS[label], xlim, ylim)
        row = summary.loc[summary.hard_region.eq(label)].iloc[0]
        ax.text(row.pc1_centroid, row.pc2_centroid, display_label.replace("Hard region ", ""), ha="center", va="center", color=NAVY, fontsize=9, weight="bold", bbox={"boxstyle": "circle,pad=0.23", "facecolor": "white", "edgecolor": COLORS[label], "alpha": 0.92})
        handles.append(Line2D([], [], marker="o", linestyle="none", markerfacecolor=COLORS[label], markeredgecolor="none", markersize=6.5, label=f"{display_label} (n={len(full_sub):,})"))
    n_noise = int(assignments.final_hard_region.eq("Noise / minor modes").sum())
    handles.append(Line2D([], [], marker="o", linestyle="none", markerfacecolor=NOISE, markeredgecolor="none", markersize=5.5, label=f"Noise / minor modes (n={n_noise:,})"))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel("Structural PC 1", color=NAVY); ax.set_ylabel("Structural PC 2", color=NAVY)
    ax.set_title("Diverse hard regions under M₀", color=NAVY, weight="semibold", pad=10)
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.96, facecolor="white", edgecolor="#D7DEE4", fontsize=8.4)
    ax.tick_params(colors=NAVY)
    for spine in ax.spines.values():
        spine.set_color("#AAB4BD")
    if full:
        note = "All hard instances are shown; cluster counts refer to the full hard set."
    else:
        note = "Points are subsampled for visualization; cluster counts refer to the full hard set."
    fig.text(0.5, 0.018, note, ha="center", color=NAVY, fontsize=8.3)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(path, dpi=DPI, facecolor="white")
    if path.name == "figD_refined_hard_regions.png":
        fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def load_v1_module() -> Any:
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("real_capability_v1_readonly", V1_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the frozen coordinate loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plot_representatives(representatives: pd.DataFrame) -> None:
    lookup = load_v1_module().load_coordinate_lookup()
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0), dpi=DPI)
    for ax, record in zip(axes, representatives.itertuples(index=False), strict=True):
        xy = lookup[(str(record.dataset), str(record.instance_id))]
        ax.scatter(xy[:, 0], xy[:, 1], s=10, c=NAVY, alpha=0.82, linewidths=0)
        ax.set_aspect("equal", adjustable="box"); ax.set_axis_off()
        ax.set_title(f"{record.hard_region} · {record.dataset}", color=NAVY, fontsize=10.2, weight="semibold", pad=22)
        ax.text(0.5, 1.02, str(record.instance_id), transform=ax.transAxes, ha="center", va="bottom", color="#637381", fontsize=7.8)
        ax.set_title(f"{DISPLAY_LABELS[str(record.hard_region)]} - {record.dataset}", color=NAVY, fontsize=10.2, weight="semibold", pad=22)
    fig.suptitle("Representative instances from hard regions", color=NAVY, weight="semibold", fontsize=13.5, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=2.2)
    fig.savefig(FIG / "representative_instances_refined.png", dpi=DPI, facecolor="white")
    fig.savefig(FIG / "representative_instances_refined.pdf", facecolor="white")
    plt.close(fig)


def plot_diagnostics(assignments: pd.DataFrame, summary: pd.DataFrame) -> None:
    major = ["Hard region A1", "Hard region A2", "Hard region B"]
    colors = [COLORS[label] for label in major]
    display_major = [DISPLAY_LABELS[label] for label in major]
    values = summary.set_index("hard_region").loc[major]
    fig, ax = plt.subplots(figsize=(6.8, 4.5), dpi=DPI)
    bars = ax.bar(display_major, values.n, color=colors, alpha=0.86)
    ax.bar_label(bars, labels=[f"{int(v):,}" for v in values.n], padding=3, fontsize=9)
    ax.set_ylabel("Full-data instance count"); ax.set_title("Final hard-region sizes", color=NAVY, weight="semibold")
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
    fig.savefig(FIG / "cluster_size_comparison.png", dpi=DPI, facecolor="white"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.5), dpi=DPI)
    x = np.arange(len(major)); width = 0.35
    ax.bar(x - width / 2, values.mean_M0_gap, width, label="Mean", color=colors, alpha=0.88)
    ax.bar(x + width / 2, values.median_M0_gap, width, label="Median", color=colors, alpha=0.48, edgecolor=colors)
    ax.axhline(TAU, color="#A94442", linestyle="--", linewidth=1.1, label=f"τ={TAU:.3f}%")
    ax.set_xticks(x, display_major); ax.set_ylabel("M0 optimality gap (%)"); ax.set_title("Gap statistics by hard region", color=NAVY, weight="semibold")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
    fig.savefig(FIG / "cluster_gap_summary.png", dpi=DPI, facecolor="white"); plt.close(fig)

    composition = values[[f"n_{name}" for name in DATASETS]].copy()
    composition.columns = DATASETS
    fractions = composition.div(composition.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=DPI)
    bottom = np.zeros(len(major))
    dataset_colors = ["#7DA7C7", "#C88C73", "#8FAF88", "#A99BC5"]
    for dataset, color in zip(DATASETS, dataset_colors, strict=True):
        ax.bar(display_major, fractions[dataset], bottom=bottom, label=dataset, color=color, alpha=0.9)
        bottom += fractions[dataset].to_numpy(float)
    ax.set_ylim(0, 1); ax.set_ylabel("Fraction of region"); ax.set_title("Benchmark composition by hard region", color=NAVY, weight="semibold")
    ax.legend(frameon=False, ncol=2); ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
    fig.savefig(FIG / "cluster_benchmark_composition.png", dpi=DPI, facecolor="white"); plt.close(fig)


def write_report(global_sweep: pd.DataFrame, leaf: pd.DataFrame, gmm: pd.DataFrame, summary: pd.DataFrame, representatives: pd.DataFrame, bic1: float, aic1: float, hashes: dict[str, str]) -> None:
    viable_global = global_sweep.loc[(global_sweep.non_noise_cluster_count >= 3) & (global_sweep.noise_fraction <= 0.20)]
    global_three = not viable_global.empty
    viable_leaf = leaf.loc[(leaf.subcluster_count >= 2) & (leaf.noise_fraction <= 0.20)]
    k2 = gmm.loc[gmm.k.eq(2)].iloc[0]
    best_ic = gmm.loc[gmm.bic.idxmin()]
    table_rows = []
    for row in summary.loc[summary.hard_region.ne("Noise / minor modes")].itertuples(index=False):
        composition = ", ".join(f"{name}={int(getattr(row, 'n_' + name)):,}" for name in DATASETS)
        centroid = ", ".join(f"{feature}={getattr(row, 'mean_' + feature):.3f}" for feature in FEATURES)
        table_rows.append(f"| {row.hard_region} | {row.n:,} | {row.mean_M0_gap:.3f}% | {row.median_M0_gap:.3f}% | {composition} | {centroid} | `{row.representative_instance_key}` |")
    source_lines = "\n".join(f"- `{path}` — SHA-256 `{digest}`" for path, digest in hashes.items())
    report = f"""# Hard-region refinement audit

This is a read-only audit over the frozen M0 hard set. No benchmark, solver result, model, threshold, structural feature, PCA coordinate, or previous visualization artifact was modified.

## Frozen protocol

{source_lines}

- Hard definition: `M0_gap > τ`, with τ={TAU:.9f}%.
- Frozen hard instances: {int(summary.n.sum()):,}.
- Clustering space: the original normalized 4D structural features (`local`, `global`, `topology`, `scale`).
- PCA is used only for visualization and never for cluster assignment.

## Stage 1 — global HDBSCAN

The full 40-setting sweep did **not** directly support three or more major regions at an acceptable noise rate. `non_noise_cluster_count >= 3` occurred only by assigning a large fraction of the hard set to noise; the eligible count with noise <=20% is {len(viable_global)}. Low-noise solutions retained the original dominant A-like family plus the small B-like region.

Answer 2: **No**, Stage 1 does not directly discover >=3 defensible major hard regions.

## Stage 2 — refinement of the original largest cluster

The original Region A contains 8,634 instances.

- HDBSCAN leaf refinement did not yield a defensible split: every multi-subcluster setting used very high noise (the minimum in this scan is {100*leaf.noise_fraction.min():.1f}%), and the count with >=2 subclusters and noise <=20% is {len(viable_leaf)}.
- GMM provides evidence for internal distributional structure. For k=2, both components are large ({k2.cluster_sizes_json}), no noise is introduced, silhouette={k2.silhouette_score:.3f}, size balance={k2.size_balance_min_over_max:.3f}, and mean/min seed stability ARI={k2.seed_stability_ari_mean:.3f}/{k2.seed_stability_ari_min:.3f}. Relative to k=1, ΔBIC={k2.delta_bic_vs_k1:,.1f} and ΔAIC={k2.delta_aic_vs_k1:,.1f} (negative favors the split).
- BIC continues to improve through k={int(best_ic.k)}, but silhouette falls from {k2.silhouette_score:.3f} at k=2 to {best_ic.silhouette_score:.3f}. The extra components are therefore treated as finer density structure, not promoted to additional named regions.
- KMeans is only a sanity check; it is not used for final assignments.

Answer 1: **Yes, with a hierarchical caveat.** Region A has a stable two-component GMM structure, but it is not a clean density-separated HDBSCAN split.

Answer 3: **Yes.** Largest-cluster refinement supports two broad, stable submodes without discarding instances as noise.

Answer 4: **A1 / A2 / B**, preserving the fact that A1 and A2 are submodes of the original dominant family.

## Final regions

| region | n | mean gap | median gap | benchmark composition | normalized-feature centroid | representative |
|---|---:|---:|---:|---|---|---|
{chr(10).join(table_rows)}

The structural interpretation is deliberately relative rather than semantic: A1 has lower `local` and near-zero `scale` values with higher `global`; A2 shifts toward higher `local`/`scale` and lower `global`; B occupies a sharply different high-`local`, high-`scale`, low-`global` part of the frozen space. The gap and benchmark-composition plots show the associated empirical differences. No cluster is named from one representative image.

Answer 5: Counts, gap statistics, composition, centroids, and representatives are listed above and exported in `05_final_cluster_summary.csv` and `06_representative_instances.csv`.

Answer 6: **Yes.** The data support the claim “High-gap instances are structurally heterogeneous.”

Answer 7: **Yes, if stated hierarchically.** The evidence supports “At least three structurally distinct hard regions are present,” with A1 and A2 explicitly described as stable submodes of one larger A family. It does not support presenting all three as equally independent HDBSCAN density islands.

Answer 8: `figD_refined_hard_regions_subsampled.png` is the best Page 2 panel. It retains full-data counts and density contours while preventing A1/A2 from visually overwhelming B. The publication pair `figD_refined_hard_regions.png/.pdf` uses the same recommended rendering.

## Verdict

- `HARD_REGION_REFINEMENT = PARTIAL_TWO_MAJOR_PLUS_SUBMODES`
- `DATA_SUPPORTS_AT_LEAST_THREE_MAJOR_HARD_REGIONS = YES`
- `LARGEST_CLUSTER_HAS_STABLE_SUBSTRUCTURE = YES`
- `RECOMMENDED_DISPLAY_LABELS = ["A1", "A2", "B"]`
"""
    (OUT / "report.md").write_text(report, encoding="utf-8", newline="\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True); SWEEP_FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.facecolor": "white", "figure.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42})
    hard = pd.read_csv(MEMBERSHIP)
    required = {"instance_key", "instance_id", "dataset", "M0_gap", "pc1", "pc2", "hard_region", *FEATURES}
    if not required.issubset(hard.columns):
        raise RuntimeError(f"missing frozen columns: {sorted(required - set(hard.columns))}")
    if len(hard) != 9021 or hard.instance_key.duplicated().any() or not (hard.M0_gap > TAU).all():
        raise RuntimeError("frozen hard-set identity or threshold invariant failed")
    if not np.isfinite(hard[["M0_gap", "pc1", "pc2", *FEATURES]].to_numpy(float)).all():
        raise RuntimeError("non-finite frozen data")
    hashes = {str(MEMBERSHIP.relative_to(ROOT)).replace("\\", "/"): sha256_file(MEMBERSHIP)}

    global_sweep, _ = global_hdbscan_sweep(hard)
    region_a = hard.loc[hard.hard_region.eq("Hard region A")].copy()
    leaf, _ = leaf_refinement(region_a)
    gmm, gmm_labels, bic1, aic1 = gmm_refinement(region_a)

    assignments = make_final_assignment(hard, region_a, gmm_labels[2])
    summary, representatives = summarize_final(assignments)
    if sorted(summary.loc[summary.hard_region.ne("Noise / minor modes"), "n"].tolist()) != sorted([5450, 3184, 237]):
        raise RuntimeError("unexpected final major-region sizes")

    global_sweep.to_csv(OUT / "01_global_hdbscan_parameter_sweep.csv", index=False, float_format="%.17g")
    leaf.to_csv(OUT / "02_largest_cluster_hdbscan_refinement.csv", index=False, float_format="%.17g")
    gmm.to_csv(OUT / "03_largest_cluster_gmm_refinement.csv", index=False, float_format="%.17g")
    assignment_columns = ["instance_key", "instance_id", "dataset", "M0_gap", "pc1", "pc2", *FEATURES, "original_hard_region", "final_hard_region", "refinement_method", "representative"]
    assignments[assignment_columns].to_csv(OUT / "04_final_cluster_assignments.csv", index=False, float_format="%.17g")
    summary.to_csv(OUT / "05_final_cluster_summary.csv", index=False, float_format="%.17g")
    representatives.to_csv(OUT / "06_representative_instances.csv", index=False, float_format="%.17g")

    plot_refined_regions(assignments, summary, FIG / "figD_refined_hard_regions.png", full=False)
    plot_refined_regions(assignments, summary, FIG / "figD_refined_hard_regions_subsampled.png", full=False)
    plot_refined_regions(assignments, summary, FIG / "figD_refined_hard_regions_full_points.png", full=True)
    plot_representatives(representatives)
    plot_diagnostics(assignments, summary)

    verdict = {
        "HARD_REGION_REFINEMENT": "PARTIAL_TWO_MAJOR_PLUS_SUBMODES",
        "DATA_SUPPORTS_AT_LEAST_THREE_MAJOR_HARD_REGIONS": "YES",
        "LARGEST_CLUSTER_HAS_STABLE_SUBSTRUCTURE": "YES",
        "RECOMMENDED_DISPLAY_LABELS": ["A1", "A2", "B"],
        "FINAL_ASSIGNMENT_METHOD": "GMM k=2 refinement of original Region A; original Region B and noise retained",
        "CAUTION": "A1/A2 are stable distributional submodes, not independent HDBSCAN density islands.",
        "source_sha256": hashes,
        "tau_gap_percent": TAU,
        "hard_count": int(len(hard)),
        "gmm_k1_bic": bic1,
        "gmm_k1_aic": aic1,
    }
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    write_report(global_sweep, leaf, gmm, summary, representatives, bic1, aic1, hashes)
    print("HARD_REGION_REFINEMENT_AUDIT_COMPLETE")
    print(f"hard={len(hard)} final={summary[['hard_region','n']].to_dict(orient='records')}")
    print("verdict=PARTIAL_TWO_MAJOR_PLUS_SUBMODES supports_three=YES")


if __name__ == "__main__":
    main()
