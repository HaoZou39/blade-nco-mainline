#!/usr/bin/env python3
"""QD4 STRUCTURAL DISTRIBUTION AUDIT — read-only, frozen official assets only.

Does not train BQ/GP, refit CVT, regenerate benchmarks, call Concorde,
or modify solver evaluation results.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.datasets import load_fixed_n100, load_original_tsplib70  # noqa: E402
from evaluation.external_v2 import load_new_dataset  # noqa: E402
from features.qd4 import QD4_FEATURE_NAMES, extract_qd4, extract_qd4_exact  # noqa: E402
from qd.coordinates import assign_nearest, load_centroids, normalize_qd4  # noqa: E402

OUT = Path(__file__).resolve().parent
CACHE = OUT / "_cache"
FIG = OUT / "figures"
TABLE = OUT / "tables"

DISPLAY = {
    "local": "local (spn_sum_of_lowest_edge_values)",
    "global": "global (spn_distance_varcoeff)",
    "topology": "topology (spn_mst_depth_mean)",
    "scale": "scale (spn_mst_dists_median)",
}
SHORT = {
    "spn_sum_of_lowest_edge_values": "local",
    "spn_distance_varcoeff": "global",
    "spn_mst_depth_mean": "topology",
    "spn_mst_dists_median": "scale",
}
DATASETS = [
    "Uniform100",
    "GM100",
    "TSPLIB100",
    "Original70",
    "EvoReal-S1 Gen.",
    "EvoReal-S2 Gen.",
    "EvoReal-S3 Gen.",
]
COLORS = {
    "Uniform100": "#1f77b4",
    "GM100": "#ff7f0e",
    "TSPLIB100": "#d62728",
    "Original70": "#9467bd",
    "EvoReal-S1 Gen.": "#2ca02c",
    "EvoReal-S2 Gen.": "#17becf",
    "EvoReal-S3 Gen.": "#8c564b",
}
CUMULATIVE_ORDER = [
    "Uniform100",
    "GM100",
    "EvoReal-S1 Gen.",
    "EvoReal-S2 Gen.",
    "EvoReal-S3 Gen.",
    "TSPLIB100",
    "Original70",
]


def lehd_aspect_ratio_preserving_normalize(xy: np.ndarray) -> np.ndarray:
    """Frozen Original70 preprocess: global min/max over all xy (preserves aspect ratio)."""
    values = np.asarray(xy, dtype=np.float64)
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax <= vmin:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def load_primary_datasets() -> dict[str, dict[str, Any]]:
    """Load frozen official benchmark coordinates (no regeneration)."""
    payloads: dict[str, dict[str, Any]] = {}

    uniform = load_fixed_n100("Uniform100")
    manifest_u = pd.read_csv(ROOT / "assets/evaluation/uniform100/manifest.csv")
    payloads["Uniform100"] = {
        "coordinates": uniform,
        "instance_ids": manifest_u.instance_id.astype(str).tolist(),
        "source": "assets/evaluation/uniform100/instances/coordinates.npz",
        "preprocess": "identity (frozen unit-square coordinates)",
        "extractor": "extract_qd4_exact",
    }

    gm = load_fixed_n100("GaussianMixture100")
    manifest_g = pd.read_csv(ROOT / "assets/evaluation/gaussian_mixture100/manifest.csv")
    payloads["GM100"] = {
        "coordinates": gm,
        "instance_ids": manifest_g.instance_id.astype(str).tolist(),
        "source": "assets/evaluation/gaussian_mixture100/instances/coordinates.npz",
        "preprocess": "identity (frozen unit-square coordinates)",
        "extractor": "extract_qd4_exact",
    }

    with np.load(
        ROOT / "assets/evaluation/tsplib100/instances/coordinates.npz", allow_pickle=False
    ) as payload:
        tsplib = np.asarray(payload["model_input_coordinates"], dtype=np.float64)
        tsplib_ids = [str(x) for x in payload["instance_id"].tolist()]
    payloads["TSPLIB100"] = {
        "coordinates": tsplib,
        "instance_ids": tsplib_ids,
        "source": "assets/evaluation/tsplib100/instances/coordinates.npz::model_input_coordinates",
        "preprocess": "frozen model_input_coordinates (unit square)",
        "extractor": "extract_qd4_exact",
    }

    records = list(load_original_tsplib70())
    o70_coords = [lehd_aspect_ratio_preserving_normalize(item[1]) for item in records]
    payloads["Original70"] = {
        "coordinates": o70_coords,
        "instance_ids": [item[0] for item in records],
        "source": "assets/evaluation/original_tsplib70 (EUC_2D gzip + source_manifest_bq83.csv)",
        "preprocess": "lehd_aspect_ratio_preserving_normalize (global minmax over all xy)",
        "extractor": "extract_qd4 (variable N)",
        "variable_n": True,
        "n_nodes": [int(item[1].shape[0]) for item in records],
    }

    for code, label in [
        ("EvoReal_S1_100", "EvoReal-S1 Gen."),
        ("EvoReal_S2_100", "EvoReal-S2 Gen."),
        ("EvoReal_S3_100", "EvoReal-S3 Gen."),
    ]:
        coords, rows = load_new_dataset(code)
        payloads[label] = {
            "coordinates": np.asarray(coords, dtype=np.float64),
            "instance_ids": [row["instance_id"] for row in rows],
            "source": f"datasets/external_v2/{code}/coordinates.npy",
            "preprocess": "identity (frozen [0,1] generator output)",
            "extractor": "extract_qd4_exact",
        }
    return payloads


def extract_features_for_dataset(name: str, payload: dict[str, Any]) -> pd.DataFrame:
    cache_path = CACHE / f"features_{name.replace(' ', '_').replace('.', '')}.parquet"
    if cache_path.is_file():
        return pd.read_parquet(cache_path)

    centroids = load_centroids()
    ids = payload["instance_ids"]
    if payload.get("variable_n"):
        rows = []
        for iid, xy in zip(ids, payload["coordinates"], strict=True):
            raw = extract_qd4(np.asarray(xy, dtype=np.float64)[None, ...], state_ids=[iid])
            norm = normalize_qd4(raw)
            niche = int(assign_nearest(norm, centroids)[0])
            dist = float(np.linalg.norm(norm[0] - centroids[niche]))
            rows.append(
                {
                    "dataset": name,
                    "instance_id": iid,
                    "local": float(norm[0, 0]),
                    "global": float(norm[0, 1]),
                    "topology": float(norm[0, 2]),
                    "scale": float(norm[0, 3]),
                    "local_transformed": float(raw[0, 0]),
                    "global_transformed": float(raw[0, 1]),
                    "topology_transformed": float(raw[0, 2]),
                    "scale_transformed": float(raw[0, 3]),
                    "cvt_niche_id": niche,
                    "distance_to_centroid": dist,
                    "n_nodes": int(xy.shape[0]),
                }
            )
        frame = pd.DataFrame(rows)
    else:
        xy = np.asarray(payload["coordinates"], dtype=np.float64)
        raw_parts = []
        batch = 512
        for start in range(0, len(xy), batch):
            raw_parts.append(extract_qd4_exact(xy[start : start + batch]))
            print(f"  [{name}] extracted {min(start + batch, len(xy))}/{len(xy)}", flush=True)
        raw = np.concatenate(raw_parts, axis=0)
        norm = normalize_qd4(raw)
        niches = assign_nearest(norm, centroids)
        dists = np.linalg.norm(norm - centroids[niches], axis=1)
        frame = pd.DataFrame(
            {
                "dataset": name,
                "instance_id": ids,
                "local": norm[:, 0],
                "global": norm[:, 1],
                "topology": norm[:, 2],
                "scale": norm[:, 3],
                "local_transformed": raw[:, 0],
                "global_transformed": raw[:, 1],
                "topology_transformed": raw[:, 2],
                "scale_transformed": raw[:, 3],
                "cvt_niche_id": niches.astype(np.int64),
                "distance_to_centroid": dists,
                "n_nodes": 100,
            }
        )
    frame.to_parquet(cache_path, index=False)
    return frame


def feature_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        sub = frame[frame.dataset == dataset]
        for feat in ("local", "global", "topology", "scale"):
            values = sub[feat].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "dataset": dataset,
                    "feature_display": feat,
                    "feature_code": QD4_FEATURE_NAMES[["local", "global", "topology", "scale"].index(feat)],
                    "N": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "median": float(np.median(values)),
                    "Q25": float(np.percentile(values, 25)),
                    "Q75": float(np.percentile(values, 75)),
                    "P10": float(np.percentile(values, 10)),
                    "P90": float(np.percentile(values, 90)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "IQR": float(np.percentile(values, 75) - np.percentile(values, 25)),
                    "CV": float(np.std(values, ddof=1) / abs(np.mean(values)))
                    if len(values) > 1 and abs(np.mean(values)) > 1e-12
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def niche_occupancy(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        sub = frame[frame.dataset == dataset]
        niches = sub.cvt_niche_id.to_numpy(dtype=np.int64)
        n = len(niches)
        counts = pd.Series(niches).value_counts()
        occupied = int(len(counts))
        p = counts.to_numpy(dtype=np.float64) / n
        h = float(-(p * np.log(p)).sum())
        denom = float(np.log(min(n, 1000)))
        rows.append(
            {
                "dataset": dataset,
                "N": n,
                "occupied_niches": occupied,
                "occupancy_rate": occupied / 1000.0,
                "mean_instances_per_occupied_niche": float(n / occupied) if occupied else float("nan"),
                "max_instances_in_one_niche": int(counts.max()) if occupied else 0,
                "normalized_niche_entropy": h / denom if denom > 0 else float("nan"),
                "effective_niches": float(np.exp(h)),
                "raw_entropy_nats": h,
            }
        )
    return pd.DataFrame(rows)


def occupied_sets(frame: pd.DataFrame) -> dict[str, set[int]]:
    return {
        dataset: set(frame.loc[frame.dataset == dataset, "cvt_niche_id"].astype(int).tolist())
        for dataset in DATASETS
    }


def pairwise_niche_overlap(sets: dict[str, set[int]]) -> pd.DataFrame:
    rows = []
    for a in DATASETS:
        for b in DATASETS:
            inter = len(sets[a] & sets[b])
            union = len(sets[a] | sets[b])
            rows.append(
                {
                    "dataset_a": a,
                    "dataset_b": b,
                    "intersection": inter,
                    "union": union,
                    "jaccard": inter / union if union else float("nan"),
                    "coverage_a_by_b": inter / len(sets[a]) if sets[a] else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _mean_pairwise_l2(x: np.ndarray, y: np.ndarray | None = None, chunk: int = 512) -> float:
    """Mean Euclidean distance; if y is None, upper triangle of X (exclude diagonal)."""
    x = np.asarray(x, dtype=np.float64)
    if y is None:
        n = len(x)
        if n < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(0, n, chunk):
            xi = x[i : i + chunk]
            # distances within chunk (upper triangle)
            d = pairwise_distances(xi, xi, metric="euclidean")
            iu = np.triu_indices(len(xi), k=1)
            total += float(d[iu].sum())
            count += int(len(iu[0]))
            # distances to later chunks
            for j in range(i + chunk, n, chunk):
                xj = x[j : j + chunk]
                d2 = pairwise_distances(xi, xj, metric="euclidean")
                total += float(d2.sum())
                count += int(d2.size)
        return total / max(count, 1)
    y = np.asarray(y, dtype=np.float64)
    total = 0.0
    count = 0
    for i in range(0, len(x), chunk):
        xi = x[i : i + chunk]
        for j in range(0, len(y), chunk):
            yj = y[j : j + chunk]
            d = pairwise_distances(xi, yj, metric="euclidean")
            total += float(d.sum())
            count += int(d.size)
    return total / max(count, 1)


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Energy distance (not squared): sqrt(2 A - B - C) with Euclidean metric."""
    a = _mean_pairwise_l2(x, y)
    b = _mean_pairwise_l2(x, None)
    c = _mean_pairwise_l2(y, None)
    val = max(0.0, 2.0 * a - b - c)
    return float(np.sqrt(val))


def mmd_rbf(x: np.ndarray, y: np.ndarray, bandwidth: float, chunk: int = 512) -> float:
    """Biased MMD^2 with RBF kernel, return sqrt(MMD^2) as distance-like score."""
    gamma = 1.0 / (2.0 * bandwidth * bandwidth)

    def mean_k(a: np.ndarray, b: np.ndarray | None = None) -> float:
        if b is None:
            n = len(a)
            if n == 0:
                return 0.0
            total = 0.0
            count = 0
            for i in range(0, n, chunk):
                ai = a[i : i + chunk]
                d = pairwise_distances(ai, ai, metric="euclidean")
                k = np.exp(-gamma * d * d)
                total += float(k.sum())
                count += int(k.size)
                for j in range(i + chunk, n, chunk):
                    bj = a[j : j + chunk]
                    d2 = pairwise_distances(ai, bj, metric="euclidean")
                    k2 = np.exp(-gamma * d2 * d2)
                    total += float(k2.sum())
                    count += int(k2.size)
            return total / max(count, 1)
        total = 0.0
        count = 0
        for i in range(0, len(a), chunk):
            ai = a[i : i + chunk]
            for j in range(0, len(b), chunk):
                bj = b[j : j + chunk]
                d = pairwise_distances(ai, bj, metric="euclidean")
                k = np.exp(-gamma * d * d)
                total += float(k.sum())
                count += int(k.size)
        return total / max(count, 1)

    mmd2 = mean_k(x, None) + mean_k(y, None) - 2.0 * mean_k(x, y)
    return float(np.sqrt(max(0.0, mmd2)))


def median_bandwidth(samples: dict[str, np.ndarray], max_points: int = 2000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    parts = []
    for arr in samples.values():
        if len(arr) <= max_points // len(samples):
            parts.append(arr)
        else:
            idx = rng.choice(len(arr), size=max_points // len(samples), replace=False)
            parts.append(arr[idx])
    pooled = np.concatenate(parts, axis=0)
    # subsample pairwise for median heuristic
    if len(pooled) > 1500:
        idx = rng.choice(len(pooled), size=1500, replace=False)
        pooled = pooled[idx]
    d = pairwise_distances(pooled, metric="euclidean")
    iu = np.triu_indices(len(pooled), k=1)
    med = float(np.median(d[iu]))
    return max(med, 1e-8)


def unique_coverage(sets: dict[str, set[int]]) -> pd.DataFrame:
    rows = []
    all_union = set().union(*sets.values()) if sets else set()
    for d in DATASETS:
        others = set().union(*(sets[o] for o in DATASETS if o != d)) if len(DATASETS) > 1 else set()
        unique = sets[d] - others
        marginal = len(sets[d] - others)
        rows.append(
            {
                "dataset": d,
                "occupied_niches": len(sets[d]),
                "unique_contribution": len(unique),
                "unique_fraction_of_own": len(unique) / len(sets[d]) if sets[d] else float("nan"),
                "marginal_coverage_given_others": marginal,
                "union_all": len(all_union),
            }
        )
    return pd.DataFrame(rows)


def cumulative_coverage(sets: dict[str, set[int]]) -> pd.DataFrame:
    rows = []
    running: set[int] = set()
    for d in CUMULATIVE_ORDER:
        before = len(running)
        added = sets[d] - running
        running |= sets[d]
        rows.append(
            {
                "step": d,
                "added_niches": len(added),
                "cumulative_occupied_niches": len(running),
                "cumulative_occupancy_rate": len(running) / 1000.0,
                "delta_from_previous": len(running) - before,
            }
        )
    return pd.DataFrame(rows)


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(8)


def save_fig(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig1_marginals(frame: pd.DataFrame) -> None:
    feats = ("local", "global", "topology", "scale")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), facecolor="white")
    for ax, feat in zip(axes.ravel(), feats):
        data = [frame.loc[frame.dataset == d, feat].to_numpy() for d in DATASETS]
        parts = ax.violinplot(data, positions=np.arange(len(DATASETS)), showmeans=False, showextrema=False, widths=0.85)
        for body, d in zip(parts["bodies"], DATASETS):
            body.set_facecolor(COLORS[d])
            body.set_alpha(0.35)
            body.set_edgecolor(COLORS[d])
        bp = ax.boxplot(
            data,
            positions=np.arange(len(DATASETS)),
            widths=0.28,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.0},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
        )
        for patch, d in zip(bp["boxes"], DATASETS):
            patch.set_facecolor(COLORS[d])
            patch.set_alpha(0.85)
        # light jitter of a subsample
        rng = np.random.default_rng(0)
        for i, (d, vals) in enumerate(zip(DATASETS, data)):
            if len(vals) > 400:
                idx = rng.choice(len(vals), size=400, replace=False)
                sample = vals[idx]
            else:
                sample = vals
            ax.scatter(
                i + rng.normal(0, 0.04, size=len(sample)),
                sample,
                s=3,
                alpha=0.15,
                color=COLORS[d],
                linewidths=0,
                zorder=2,
            )
        ax.set_xticks(np.arange(len(DATASETS)))
        ax.set_xticklabels([d.replace(" Gen.", "") for d in DATASETS], rotation=35, ha="right", fontsize=7)
        ax.set_title(DISPLAY[feat], fontsize=9)
        ax.set_ylabel("normalized QD4", fontsize=8)
        style_axes(ax)
    fig.suptitle("Fig 1. QD4 marginal distributions (normalized [0,1])", fontsize=11)
    fig.tight_layout()
    save_fig(fig, "fig1_qd4_marginal_distributions")


def fig2_pca(frame: pd.DataFrame) -> dict[str, float]:
    feats = frame[["local", "global", "topology", "scale"]].to_numpy(dtype=np.float64)
    pca = PCA(n_components=2, random_state=0)
    xy = pca.fit_transform(feats)
    ev = pca.explained_variance_ratio_
    frame = frame.copy()
    frame["pc1"] = xy[:, 0]
    frame["pc2"] = xy[:, 1]

    fig, ax = plt.subplots(figsize=(8.5, 6.5), facecolor="white")
    for d in DATASETS:
        sub = frame[frame.dataset == d]
        ax.scatter(sub.pc1, sub.pc2, s=6, alpha=0.35, color=COLORS[d], label=d, linewidths=0)
    ax.set_xlabel(f"PC1 ({100 * ev[0]:.1f}% var.)", fontsize=9)
    ax.set_ylabel(f"PC2 ({100 * ev[1]:.1f}% var.)", fontsize=9)
    note = ""
    if ev[0] + ev[1] < 0.70:
        note = "\n2D PCA is illustrative only."
        ax.text(0.02, 0.98, "2D PCA is illustrative only.", transform=ax.transAxes, va="top", fontsize=8, color="#444")
    ax.set_title("Fig 2. QD4 PCA structural map (all PRIMARY benchmarks)" + note, fontsize=10)
    ax.legend(fontsize=7, markerscale=2, frameon=False, loc="best")
    style_axes(ax)
    fig.tight_layout()
    save_fig(fig, "fig2_qd4_pca_all")

    fig, ax = plt.subplots(figsize=(8.5, 6.5), facecolor="white")
    for d in DATASETS:
        sub = frame[frame.dataset == d]
        if d in {"TSPLIB100", "Original70"}:
            ax.scatter(sub.pc1, sub.pc2, s=10, alpha=0.55, color=COLORS[d], label=d, linewidths=0, zorder=3)
        else:
            ax.scatter(sub.pc1, sub.pc2, s=4, alpha=0.08, color="#bbbbbb", label=None, linewidths=0, zorder=1)
    ax.scatter([], [], s=4, color="#bbbbbb", label="other benchmarks (faded)")
    ax.set_xlabel(f"PC1 ({100 * ev[0]:.1f}% var.)", fontsize=9)
    ax.set_ylabel(f"PC2 ({100 * ev[1]:.1f}% var.)", fontsize=9)
    if ev[0] + ev[1] < 0.70:
        ax.text(0.02, 0.98, "2D PCA is illustrative only.", transform=ax.transAxes, va="top", fontsize=8, color="#444")
    ax.set_title("Fig 2b. TSPLIB100 / Original70 highlighted", fontsize=10)
    ax.legend(fontsize=7, markerscale=2, frameon=False)
    style_axes(ax)
    fig.tight_layout()
    save_fig(fig, "fig2b_qd4_pca_tsplib_highlight")
    return {"pc1_explained_variance": float(ev[0]), "pc2_explained_variance": float(ev[1]), "pc1_pc2_sum": float(ev.sum())}


def fig3_occupancy(occ: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), facecolor="white")
    x = np.arange(len(DATASETS))
    axes[0].bar(x, occ.set_index("dataset").loc[DATASETS, "occupied_niches"], color=[COLORS[d] for d in DATASETS])
    axes[0].set_ylabel("occupied CVT niches", fontsize=8)
    axes[0].set_title("Occupied niches / 1000", fontsize=9)
    axes[1].bar(x, occ.set_index("dataset").loc[DATASETS, "effective_niches"], color=[COLORS[d] for d in DATASETS])
    axes[1].set_ylabel("effective niches = exp(H)", fontsize=8)
    axes[1].set_title("Effective niche diversity", fontsize=9)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([d.replace(" Gen.", "") for d in DATASETS], rotation=35, ha="right", fontsize=7)
        style_axes(ax)
    fig.suptitle("Fig 3. CVT-1000 niche occupancy / effective niches", fontsize=11)
    fig.tight_layout()
    save_fig(fig, "fig3_cvt_niche_occupancy")


def fig4_overlap(overlap: pd.DataFrame, value_col: str, stem: str, title: str) -> None:
    mat = overlap.pivot(index="dataset_a", columns="dataset_b", values=value_col).loc[DATASETS, DATASETS]
    fig, ax = plt.subplots(figsize=(7.2, 6.0), facecolor="white")
    im = ax.imshow(mat.to_numpy(dtype=np.float64), cmap="viridis", vmin=0, vmax=1 if value_col != "jaccard" else None)
    ax.set_xticks(np.arange(len(DATASETS)))
    ax.set_yticks(np.arange(len(DATASETS)))
    ax.set_xticklabels([d.replace(" Gen.", "") for d in DATASETS], rotation=40, ha="right", fontsize=7)
    ax.set_yticklabels([d.replace(" Gen.", "") for d in DATASETS], fontsize=7)
    for i in range(len(DATASETS)):
        for j in range(len(DATASETS)):
            val = mat.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color="white" if val > 0.45 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=10)
    style_axes(ax)
    fig.tight_layout()
    save_fig(fig, stem)


def fig5_distance(mat: pd.DataFrame, stem: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.0), facecolor="white")
    arr = mat.loc[DATASETS, DATASETS].to_numpy(dtype=np.float64)
    im = ax.imshow(arr, cmap="magma")
    ax.set_xticks(np.arange(len(DATASETS)))
    ax.set_yticks(np.arange(len(DATASETS)))
    ax.set_xticklabels([d.replace(" Gen.", "") for d in DATASETS], rotation=40, ha="right", fontsize=7)
    ax.set_yticklabels([d.replace(" Gen.", "") for d in DATASETS], fontsize=7)
    for i in range(len(DATASETS)):
        for j in range(len(DATASETS)):
            ax.text(j, i, f"{arr[i, j]:.3f}", ha="center", va="center", fontsize=6, color="white" if arr[i, j] > arr.max() * 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=10)
    style_axes(ax)
    fig.tight_layout()
    save_fig(fig, stem)


def fig6_cumulative(cum: pd.DataFrame, uniq: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), facecolor="white")
    x = np.arange(len(cum))
    axes[0].plot(x, cum.cumulative_occupied_niches, marker="o", color="#222", linewidth=1.5)
    axes[0].bar(x, cum.added_niches, alpha=0.45, color=[COLORS[d] for d in CUMULATIVE_ORDER])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"+{d.replace(' Gen.', '')}" if i else d for i, d in enumerate(CUMULATIVE_ORDER)], rotation=35, ha="right", fontsize=7)
    axes[0].set_ylabel("CVT niches", fontsize=8)
    axes[0].set_title("Cumulative niche coverage", fontsize=9)
    style_axes(axes[0])

    u = uniq.set_index("dataset").loc[DATASETS]
    axes[1].bar(np.arange(len(DATASETS)), u.unique_contribution, color=[COLORS[d] for d in DATASETS])
    axes[1].set_xticks(np.arange(len(DATASETS)))
    axes[1].set_xticklabels([d.replace(" Gen.", "") for d in DATASETS], rotation=35, ha="right", fontsize=7)
    axes[1].set_ylabel("unique niches", fontsize=8)
    axes[1].set_title("Order-independent unique contribution", fontsize=9)
    style_axes(axes[1])
    fig.suptitle("Fig 6. Benchmark complementarity on CVT-1000 niches", fontsize=11)
    fig.tight_layout()
    save_fig(fig, "fig6_benchmark_niche_coverage")


def internal_heterogeneity(frame: pd.DataFrame, dataset: str) -> dict[str, Any]:
    sub = frame[frame.dataset == dataset]
    x = sub[["local", "global", "topology", "scale"]].to_numpy(dtype=np.float64)
    # pairwise distance sample
    rng = np.random.default_rng(0)
    if len(x) > 1500:
        idx = rng.choice(len(x), size=1500, replace=False)
        xs = x[idx]
    else:
        xs = x
    d = pairwise_distances(xs, metric="euclidean")
    iu = np.triu_indices(len(xs), k=1)
    dists = d[iu]
    niche_n = sub.cvt_niche_id.nunique()
    counts = sub.cvt_niche_id.value_counts().to_numpy(dtype=np.float64)
    p = counts / counts.sum()
    h = float(-(p * np.log(p)).sum())
    feat_stats = {}
    for feat in ("local", "global", "topology", "scale"):
        v = sub[feat].to_numpy(dtype=np.float64)
        feat_stats[feat] = {
            "CV": float(np.std(v, ddof=1) / abs(np.mean(v))) if abs(np.mean(v)) > 1e-12 else float("nan"),
            "IQR": float(np.percentile(v, 75) - np.percentile(v, 25)),
            "range": float(v.max() - v.min()),
        }
    kmeans_rows = []
    for k in (2, 3, 4, 5):
        if len(x) <= k:
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = km.fit_predict(x)
        sil = float(silhouette_score(x, labels)) if len(set(labels)) > 1 else float("nan")
        kmeans_rows.append({"k": k, "silhouette": sil, "inertia": float(km.inertia_)})
    return {
        "dataset": dataset,
        "N": int(len(sub)),
        "occupied_niches": int(niche_n),
        "effective_niches": float(np.exp(h)),
        "pairwise_qd4_distance_mean": float(dists.mean()),
        "pairwise_qd4_distance_std": float(dists.std(ddof=1)) if len(dists) > 1 else 0.0,
        "pairwise_qd4_distance_median": float(np.median(dists)),
        "pairwise_qd4_distance_p90": float(np.percentile(dists, 90)),
        "feature_stats": feat_stats,
        "kmeans_exploratory": kmeans_rows,
    }


def decide_verdicts(
    frame: pd.DataFrame,
    occ: pd.DataFrame,
    energy: pd.DataFrame,
    uniq: pd.DataFrame,
    het_tsplib: dict[str, Any],
    het_o70: dict[str, Any],
) -> dict[str, str]:
    occ_i = occ.set_index("dataset")
    # Compare TSPLIB spread vs Uniform (tight synthetic) and GM
    tsp_occ = float(occ_i.loc["TSPLIB100", "occupied_niches"])
    uni_occ = float(occ_i.loc["Uniform100", "occupied_niches"])
    gm_occ = float(occ_i.loc["GM100", "occupied_niches"])
    tsp_eff = float(occ_i.loc["TSPLIB100", "effective_niches"])
    uni_eff = float(occ_i.loc["Uniform100", "effective_niches"])

    # feature ranges
    def rng(ds: str, feat: str) -> float:
        v = frame.loc[frame.dataset == ds, feat]
        return float(v.max() - v.min())

    tsp_ranges = {f: rng("TSPLIB100", f) for f in ("local", "global", "topology", "scale")}
    uni_ranges = {f: rng("Uniform100", f) for f in ("local", "global", "topology", "scale")}
    wider = sum(tsp_ranges[f] > 1.5 * uni_ranges[f] for f in tsp_ranges)

    # multivariate distances: TSPLIB far from Uniform and GM?
    e_tu = float(energy.loc["TSPLIB100", "Uniform100"])
    e_tg = float(energy.loc["TSPLIB100", "GM100"])
    e_ug = float(energy.loc["Uniform100", "GM100"])

    evidence = []
    evidence.append(tsp_occ >= max(uni_occ, gm_occ) * 1.2 or tsp_eff >= uni_eff * 1.5)
    evidence.append(wider >= 2)
    evidence.append(het_tsplib["pairwise_qd4_distance_p90"] > 0.25)
    evidence.append(e_tu > 0.5 * e_ug or e_tg > 0.5 * e_ug)

    if all(evidence):
        het_verdict = "TSPLIB_STRUCTURAL_HETEROGENEITY_CONFIRMED"
    elif sum(evidence) >= 2:
        het_verdict = "TSPLIB_STRUCTURAL_HETEROGENEITY_PARTIAL"
    else:
        het_verdict = "TSPLIB_STRUCTURAL_HETEROGENEITY_NOT_SUPPORTED"

    unique_vals = uniq.set_index("dataset").unique_contribution
    total_unique = int(unique_vals.sum())
    top_share = float(unique_vals.max() / total_unique) if total_unique else 0.0
    n_with_unique = int((unique_vals > 0).sum())
    if n_with_unique >= 5 and top_share < 0.55:
        comp = "STRONG"
    elif n_with_unique >= 3:
        comp = "MODERATE"
    else:
        comp = "WEAK"
    return {
        "TSPLIB_STRUCTURAL_HETEROGENEITY": het_verdict,
        "BENCHMARK_COMPLEMENTARITY": comp,
        "evidence_flags": str(evidence),
        "tsp_occupied": str(tsp_occ),
        "uni_occupied": str(uni_occ),
        "gm_occupied": str(gm_occ),
        "wider_than_uniform_count": str(wider),
        "o70_occupied": str(float(occ_i.loc["Original70", "occupied_niches"])),
        "o70_effective": str(float(occ_i.loc["Original70", "effective_niches"])),
        "het_o70_p90": str(het_o70["pairwise_qd4_distance_p90"]),
    }


def write_report(
    payloads: dict[str, Any],
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    occ: pd.DataFrame,
    overlap: pd.DataFrame,
    energy: pd.DataFrame,
    mmd: pd.DataFrame,
    wass: pd.DataFrame,
    uniq: pd.DataFrame,
    cum: pd.DataFrame,
    pca_info: dict[str, float],
    het_tsplib: dict[str, Any],
    het_o70: dict[str, Any],
    verdicts: dict[str, str],
    bandwidth: float,
) -> None:
    occ_i = occ.set_index("dataset")
    sets = occupied_sets(frame)

    def cov(a: str, b: str) -> float:
        sa, sb = sets[a], sets[b]
        return len(sa & sb) / len(sa) if sa else float("nan")

    def jac(a: str, b: str) -> float:
        sa, sb = sets[a], sets[b]
        u = len(sa | sb)
        return len(sa & sb) / u if u else float("nan")

    # EvoReal overlap among themselves
    evo_pairs = [
        ("EvoReal-S1 Gen.", "EvoReal-S2 Gen."),
        ("EvoReal-S1 Gen.", "EvoReal-S3 Gen."),
        ("EvoReal-S2 Gen.", "EvoReal-S3 Gen."),
    ]
    evo_lines = "\n".join(
        f"- {a} vs {b}: Jaccard={jac(a,b):.3f}, Coverage({a} by {b})={cov(a,b):.3f}, Coverage({b} by {a})={cov(b,a):.3f}"
        for a, b in evo_pairs
    )

    focus_lines = []
    for a in ("TSPLIB100", "Original70"):
        for b in ("Uniform100", "GM100", "EvoReal-S1 Gen.", "EvoReal-S2 Gen.", "EvoReal-S3 Gen."):
            focus_lines.append(
                f"- {a} vs {b}: Jaccard={jac(a,b):.3f}, Coverage({a} by {b})={cov(a,b):.3f}, Coverage({b} by {a})={cov(b,a):.3f}, Energy={energy.loc[a,b]:.4f}"
            )

    # Regime coverage heuristics: compare niche overlap with Uniform (uniform-like) and GM (cluster-like)
    def regime_note(ds: str) -> str:
        c_u = cov(ds, "Uniform100")
        c_g = cov(ds, "GM100")
        own = occ_i.loc[ds, "occupied_niches"]
        return (
            f"{ds}: occupied={int(own)}, effective={occ_i.loc[ds,'effective_niches']:.1f}; "
            f"Coverage({ds} niches by Uniform)={c_u:.3f}, by GM100={c_g:.3f}; "
            f"unique niches={int(uniq.set_index('dataset').loc[ds,'unique_contribution'])}"
        )

    redundant = uniq.set_index("dataset")
    redundant_list = redundant[redundant.unique_contribution <= max(1, 0.02 * redundant.unique_contribution.max())].index.tolist()

    q1 = (
        "No — TSPLIB100 is not a single tight structural distribution."
        if verdicts["TSPLIB_STRUCTURAL_HETEROGENEITY"] != "TSPLIB_STRUCTURAL_HETEROGENEITY_NOT_SUPPORTED"
        else "Yes — evidence is consistent with a relatively compact structural region."
    )
    q2 = (
        f"Original70 occupies {int(occ_i.loc['Original70','occupied_niches'])} niches "
        f"({occ_i.loc['Original70','occupancy_rate']:.3f} rate) with effective_niches="
        f"{occ_i.loc['Original70','effective_niches']:.1f} and pairwise QD4 distance P90="
        f"{het_o70['pairwise_qd4_distance_p90']:.3f}."
    )

    lines = [
        "# QD4 Structural Distribution Audit",
        "",
        "Independent, read-only analysis of frozen N≈100 PRIMARY benchmarks in the formal BLADE QD4 + CVT-1000 space.",
        "",
        "> These labels denote EvoReal generator families rather than strictly pure structural classes.",
        "",
        "## Feature definitions (canonical code ↔ display)",
        "",
        "| display | canonical code field |",
        "|---|---|",
        "| local | `spn_sum_of_lowest_edge_values` |",
        "| global | `spn_distance_varcoeff` |",
        "| topology | `spn_mst_depth_mean` |",
        "| scale | `spn_mst_dists_median` |",
        "",
        "Pipeline: formal `extract_qd4_exact` (N=100) / `extract_qd4` (Original70 variable N) → arcsinh robust transform → Normalization A P1/P99 clip to [0,1] → nearest of frozen CVT-1000 centroids.",
        "",
        "## Datasets (frozen official files)",
        "",
    ]
    for d in DATASETS:
        p = payloads[d]
        n = int((frame.dataset == d).sum())
        lines.append(f"- **{d}**: N={n}; source=`{p['source']}`; preprocess=`{p['preprocess']}`; extractor=`{p['extractor']}`")
    lines += [
        "",
        "GM500 excluded from PRIMARY to avoid confounding size with structure.",
        "",
        "## Analysis 1 — Feature marginal summaries",
        "",
        "See `tables/02_feature_distribution_summary.csv` and `figures/fig1_qd4_marginal_distributions.(pdf|png)`.",
        "",
        "## Analysis 2 — PCA visualization (illustrative only for quantitative claims)",
        "",
        f"- PC1 explained variance: **{100*pca_info['pc1_explained_variance']:.2f}%**",
        f"- PC2 explained variance: **{100*pca_info['pc2_explained_variance']:.2f}%**",
        f"- PC1+PC2: **{100*pca_info['pc1_pc2_sum']:.2f}%**"
        + (" — **2D PCA is illustrative only.**" if pca_info["pc1_pc2_sum"] < 0.70 else ""),
        "",
        "All CVT / distance metrics remain in original normalized 4D QD space.",
        "",
        "## Analysis 3 — CVT-1000 niche occupancy",
        "",
        occ.to_markdown(index=False),
        "",
        "## Analysis 4 — Niche overlap (Jaccard + directional Coverage)",
        "",
        "Absolute Jaccard is expected to be modest when |instances| << 1000 niches; directional Coverage is primary.",
        "",
        *focus_lines,
        "",
        "### EvoReal generator-family overlap",
        "",
        evo_lines,
        "",
        "## Analysis 5 — Structural distances (normalized QD4 4D)",
        "",
        f"MMD RBF bandwidth (pooled median heuristic): **{bandwidth:.6f}**",
        "",
        "Matrices: `tables/05_pairwise_energy_distance.csv`, `tables/06_pairwise_mmd.csv`, `tables/07_feature_wasserstein.csv`.",
        "",
        "## Analysis 6 — TSPLIB / Original70 internal heterogeneity",
        "",
        f"### TSPLIB100",
        f"- occupied niches: {het_tsplib['occupied_niches']}",
        f"- effective niches: {het_tsplib['effective_niches']:.2f}",
        f"- pairwise QD4 distance mean/median/P90: {het_tsplib['pairwise_qd4_distance_mean']:.4f} / {het_tsplib['pairwise_qd4_distance_median']:.4f} / {het_tsplib['pairwise_qd4_distance_p90']:.4f}",
        f"- feature CV/IQR: {json.dumps(het_tsplib['feature_stats'])}",
        f"- exploratory k-means silhouettes (not claimed as true classes): {json.dumps(het_tsplib['kmeans_exploratory'])}",
        "",
        f"### Original70",
        f"- occupied niches: {het_o70['occupied_niches']}",
        f"- effective niches: {het_o70['effective_niches']:.2f}",
        f"- pairwise QD4 distance mean/median/P90: {het_o70['pairwise_qd4_distance_mean']:.4f} / {het_o70['pairwise_qd4_distance_median']:.4f} / {het_o70['pairwise_qd4_distance_p90']:.4f}",
        f"- feature CV/IQR: {json.dumps(het_o70['feature_stats'])}",
        f"- exploratory k-means silhouettes (structural regimes only): {json.dumps(het_o70['kmeans_exploratory'])}",
        "",
        "## Analysis 7 — Benchmark complementarity",
        "",
        cum.to_markdown(index=False),
        "",
        uniq.to_markdown(index=False),
        "",
        "## Answers to required questions",
        "",
        f"**Q1. TSPLIB100 single tight distribution?** {q1}",
        "",
        f"**Q2. Original70 internal heterogeneity?** {q2}",
        "",
        "**Q3. Do TSPLIB / Original70 span Uniform-like, cluster-like, and other regimes?**",
        "",
        f"- {regime_note('TSPLIB100')}",
        f"- {regime_note('Original70')}",
        "",
        "Interpretation uses niche overlap with Uniform100 (uniform-like) and GM100 (cluster-like) plus residual unique niches as other structural regimes — not discrete S1/S2/S3 class labels.",
        "",
        "**Q4. EvoReal S1/S2/S3 Gen. separation vs overlap?**",
        "",
        evo_lines,
        "",
        "**Q5. Complementary structural coverage across 7 families?** See cumulative coverage and unique contributions; complementarity verdict below.",
        "",
        f"**Q6. Most unique QD4 niches?** Top: {', '.join(redundant.sort_values('unique_contribution', ascending=False).index[:3].tolist())} "
        f"({', '.join(str(int(x)) for x in redundant.sort_values('unique_contribution', ascending=False).unique_contribution.head(3).tolist())}).",
        "",
        f"**Q7. Near-redundant families?** Candidates with near-zero unique niches relative to the suite: {redundant_list if redundant_list else 'none with near-zero unique contribution'}.",
        "",
        "## Verdict",
        "",
        f"- **TSPLIB_STRUCTURAL_HETEROGENEITY** = `{verdicts['TSPLIB_STRUCTURAL_HETEROGENEITY']}`",
        f"- **BENCHMARK_COMPLEMENTARITY** = `{verdicts['BENCHMARK_COMPLEMENTARITY']}`",
        "",
        f"Evidence flags (occupancy, feature-range width, internal pairwise spread, multivariate distance): {verdicts['evidence_flags']}",
        "",
        "## Integrity checklist",
        "",
        "- [x] 所有 benchmark 都是 frozen official files",
        "- [x] 没有重新生成 benchmark",
        "- [x] 没有重新 fit CVT",
        "- [x] 使用正式 CVT-1000 centroids",
        "- [x] 使用正式 QD4 feature extractor",
        "- [x] 使用正式 feature normalization",
        "- [x] Original70 使用冻结的 aspect-ratio-preserving preprocessing",
        "- [x] 没有训练 BQ",
        "- [x] 没有训练 GP",
        "- [x] 没有调用 Concorde",
        "- [x] 没有改变任何 solver evaluation result",
        "- [x] PCA 只用于 visualization",
        "- [x] 所有 distance metrics 均在原始 normalized QD4 4D space 中计算",
        "",
        "Audit complete. No BLADE mainline pipeline changes were made.",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    CACHE.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    TABLE.mkdir(parents=True, exist_ok=True)

    print("Loading frozen datasets...", flush=True)
    payloads = load_primary_datasets()
    provenance = {
        name: {
            "source": payloads[name]["source"],
            "preprocess": payloads[name]["preprocess"],
            "extractor": payloads[name]["extractor"],
            "n_instances": len(payloads[name]["instance_ids"]),
        }
        for name in DATASETS
    }
    (OUT / "dataset_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    frames = []
    for name in DATASETS:
        print(f"Extracting QD4 + CVT niches: {name}", flush=True)
        frames.append(extract_features_for_dataset(name, payloads[name]))
    frame = pd.concat(frames, ignore_index=True)
    # public table with required columns
    public = frame[
        ["dataset", "instance_id", "local", "global", "topology", "scale", "cvt_niche_id", "distance_to_centroid"]
    ].copy()
    public.to_csv(TABLE / "01_qd4_features_per_instance.csv", index=False)
    frame.to_parquet(CACHE / "all_features.parquet", index=False)

    print("Summaries / occupancy / overlap...", flush=True)
    summary = feature_summary(frame)
    summary.to_csv(TABLE / "02_feature_distribution_summary.csv", index=False)
    occ = niche_occupancy(frame)
    occ.to_csv(TABLE / "03_niche_occupancy_summary.csv", index=False)
    sets = occupied_sets(frame)
    overlap = pairwise_niche_overlap(sets)
    overlap.to_csv(TABLE / "04_pairwise_niche_overlap.csv", index=False)

    print("Structural distances (energy / MMD / Wasserstein)...", flush=True)
    samples = {
        d: frame.loc[frame.dataset == d, ["local", "global", "topology", "scale"]].to_numpy(dtype=np.float64)
        for d in DATASETS
    }
    # For very large sets, subsample for pairwise distance matrices to keep runtime practical,
    # but keep full data for occupancy / unique coverage / feature summaries.
    rng = np.random.default_rng(0)
    dist_samples = {}
    for d, arr in samples.items():
        if len(arr) > 2000:
            idx = rng.choice(len(arr), size=2000, replace=False)
            dist_samples[d] = arr[idx]
            print(f"  distance subsample {d}: {len(arr)} -> 2000", flush=True)
        else:
            dist_samples[d] = arr
    bandwidth = median_bandwidth(dist_samples)
    energy = pd.DataFrame(index=DATASETS, columns=DATASETS, dtype=float)
    mmd = pd.DataFrame(index=DATASETS, columns=DATASETS, dtype=float)
    for i, a in enumerate(DATASETS):
        for j, b in enumerate(DATASETS):
            if j < i:
                energy.loc[a, b] = energy.loc[b, a]
                mmd.loc[a, b] = mmd.loc[b, a]
            elif i == j:
                energy.loc[a, b] = 0.0
                mmd.loc[a, b] = 0.0
            else:
                print(f"  energy/mmd {a} vs {b}", flush=True)
                energy.loc[a, b] = energy_distance(dist_samples[a], dist_samples[b])
                mmd.loc[a, b] = mmd_rbf(dist_samples[a], dist_samples[b], bandwidth)
    energy.to_csv(TABLE / "05_pairwise_energy_distance.csv")
    mmd.to_csv(TABLE / "06_pairwise_mmd.csv")

    wass_rows = []
    for feat in ("local", "global", "topology", "scale"):
        for a in DATASETS:
            for b in DATASETS:
                va = frame.loc[frame.dataset == a, feat].to_numpy()
                vb = frame.loc[frame.dataset == b, feat].to_numpy()
                wass_rows.append(
                    {
                        "feature": feat,
                        "feature_code": QD4_FEATURE_NAMES[["local", "global", "topology", "scale"].index(feat)],
                        "dataset_a": a,
                        "dataset_b": b,
                        "wasserstein_1d": float(wasserstein_distance(va, vb)),
                    }
                )
    wass = pd.DataFrame(wass_rows)
    wass.to_csv(TABLE / "07_feature_wasserstein.csv", index=False)

    uniq = unique_coverage(sets)
    uniq.to_csv(TABLE / "08_benchmark_unique_coverage.csv", index=False)
    cum = cumulative_coverage(sets)
    cum.to_csv(TABLE / "09_cumulative_niche_coverage.csv", index=False)

    print("Internal heterogeneity...", flush=True)
    het_tsplib = internal_heterogeneity(frame, "TSPLIB100")
    het_o70 = internal_heterogeneity(frame, "Original70")
    (OUT / "internal_heterogeneity.json").write_text(
        json.dumps({"TSPLIB100": het_tsplib, "Original70": het_o70}, indent=2), encoding="utf-8"
    )

    print("Figures...", flush=True)
    fig1_marginals(frame)
    pca_info = fig2_pca(frame)
    fig3_occupancy(occ)
    fig4_overlap(overlap, "jaccard", "fig4a_pairwise_jaccard", "Fig 4a. Pairwise niche Jaccard")
    fig4_overlap(overlap, "coverage_a_by_b", "fig4b_pairwise_coverage", "Fig 4b. Coverage(A by B) = |A∩B|/|A|")
    fig5_distance(energy, "fig5a_energy_distance", "Fig 5a. Pairwise Energy Distance (QD4 4D)")
    fig5_distance(mmd, "fig5b_mmd", "Fig 5b. Pairwise MMD (RBF, QD4 4D)")
    fig6_cumulative(cum, uniq)

    verdicts = decide_verdicts(frame, occ, energy, uniq, het_tsplib, het_o70)
    (OUT / "verdict.json").write_text(json.dumps(verdicts, indent=2), encoding="utf-8")
    write_report(
        payloads,
        frame,
        summary,
        occ,
        overlap,
        energy,
        mmd,
        wass,
        uniq,
        cum,
        pca_info,
        het_tsplib,
        het_o70,
        verdicts,
        bandwidth,
    )
    print(f"Done in {time.perf_counter()-started:.1f}s", flush=True)
    print(json.dumps(verdicts, indent=2), flush=True)
    print(f"Outputs under {OUT}", flush=True)


if __name__ == "__main__":
    main()
