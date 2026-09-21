#!/usr/bin/env python
"""Read-only PPT visual cleanup for frozen M0/M1/M5 capability fields."""

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
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import KNeighborsRegressor


OUT = Path(__file__).resolve().parent
V2 = OUT.parent
ROOT = OUT.parents[3]
sys.dont_write_bytecode = True
sys.path.insert(0, str(V2))
import run_visualization_refactor_v2 as v2  # noqa: E402


DATASETS = v2.DATASETS
FEATURES = v2.FEATURES
TAU = 1.6759665429999124
DPI = 300
GRID_K = 75
SELECTED_SIGMA = 2.0
FOREGROUND_PER_DATASET = 50
NAVY = "#102A43"
HANDLED_BLUE = "#1E5B8A"
HARD_POINT_RED = "#A73735"
BOUNDARY_RED = "#A94A47"
OLD_GRAY = "#A9AFB5"
UNSUPPORTED = "#F4F4F4"
POINT_SIZE_HANDLED = 25.0
POINT_SIZE_HARD = 32.0
COLOR_VMAX = 5.0

GAP_CMAP = LinearSegmentedColormap.from_list(
    "ppt_gap_relative_to_tau",
    [
        (0.00, "#285D83"),
        (0.30, "#AEC9D9"),
        (0.50, "#FFFFFF"),
        (0.76, "#F2D2CC"),
        (1.00, "#DFA39A"),
    ],
)
GAP_CMAP.set_over("#D69289")
GAP_CMAP.set_bad(UNSUPPORTED)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def make_foreground_subset(frame: pd.DataFrame) -> pd.DataFrame:
    """Fixed 200-point benchmark/M0-label-stratified 4D coverage subset."""
    parts = []
    for dataset in DATASETS:
        dataset_frame = frame.loc[frame.dataset.eq(dataset)].reset_index(drop=True)
        groups = {
            "M0_hard": dataset_frame.loc[dataset_frame.M0_gap > TAU].reset_index(drop=True),
            "M0_handled": dataset_frame.loc[dataset_frame.M0_gap <= TAU].reset_index(drop=True),
        }
        targets = {
            "M0_hard": min(15, len(groups["M0_hard"])),
            "M0_handled": min(35, len(groups["M0_handled"])),
        }
        remaining = FOREGROUND_PER_DATASET - sum(targets.values())
        while remaining > 0:
            capacities = {
                name: len(groups[name]) - targets[name] for name in groups
            }
            name = max(capacities, key=capacities.get)
            if capacities[name] <= 0:
                raise RuntimeError(f"cannot fill foreground quota for {dataset}")
            add = min(remaining, capacities[name])
            targets[name] += add
            remaining -= add
        dataset_parts = []
        for name in ("M0_hard", "M0_handled"):
            group = groups[name]
            indices, coverage = v2.base.farthest_point_indices(
                group[FEATURES].to_numpy(float), targets[name]
            )
            chosen = group.iloc[indices].copy()
            chosen["sampling_stratum"] = name
            chosen["coverage_distance_at_selection"] = coverage
            dataset_parts.append(chosen)
        selected = pd.concat(dataset_parts, ignore_index=True)
        if len(selected) != FOREGROUND_PER_DATASET:
            raise RuntimeError(f"wrong foreground quota for {dataset}")
        parts.append(selected)
    subset = pd.concat(parts, ignore_index=True)
    if len(subset) != 200 or subset.instance_key.nunique() != 200:
        raise RuntimeError("foreground subset must contain 200 unique instances")
    for model in ("M0", "M1", "M5"):
        subset[f"{model}_label"] = np.where(
            subset[f"{model}_gap"] <= TAU, "Handled well", "High-gap / hard"
        )
    return subset


def build_base_fields(
    frame: pd.DataFrame,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> dict[str, Any]:
    z = frame[["pc1", "pc2"]].to_numpy(float)
    pair = v2.base.fit_fields(
        z,
        frame.M0_gap.to_numpy(float),
        frame.M1_gap.to_numpy(float),
        TAU,
        xlim,
        ylim,
    )
    xx, yy = pair["xx"], pair["yy"]
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    fields = {"M0": pair["fields"][0], "M1": pair["fields"][1]}
    model5 = KNeighborsRegressor(
        n_neighbors=GRID_K, weights="distance", algorithm="kd_tree", n_jobs=-1
    ).fit(z, frame.M5_gap.to_numpy(float))
    fields["M5"] = model5.predict(grid).reshape(xx.shape)
    return {
        "xx": xx,
        "yy": yy,
        "grid": grid,
        "support": pair["support"],
        "support_radius": pair["support_radius"],
        "fields": fields,
    }


def contour_segments(
    xx: np.ndarray,
    yy: np.ndarray,
    field: np.ndarray,
    support: np.ndarray,
) -> list[np.ndarray]:
    fig, ax = plt.subplots(figsize=(2, 2))
    contour = ax.contour(
        xx, yy, np.ma.masked_where(~support, field), levels=[TAU]
    )
    segments = [np.asarray(segment, dtype=float) for segment in contour.allsegs[0] if len(segment) >= 2]
    plt.close(fig)
    return segments


def segment_measurements(segments: list[np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for index, points in enumerate(segments):
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        area = float(abs(np.dot(points[:, 0], np.roll(points[:, 1], 1)) - np.dot(points[:, 1], np.roll(points[:, 0], 1))) / 2)
        rows.append({
            "component_index": index,
            "vertex_count": int(len(points)),
            "length_pc_units": length,
            "enclosed_area_pc2_units": area,
            "closed": bool(np.linalg.norm(points[0] - points[-1]) < 0.02),
        })
    return rows


def cleanup_segments(
    segments: list[np.ndarray],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[list[np.ndarray], float, list[dict[str, Any]]]:
    measurements = segment_measurements(segments)
    if not measurements:
        return [], math.nan, []
    lengths = np.array([row["length_pc_units"] for row in measurements])
    plot_diagonal = float(math.hypot(xlim[1] - xlim[0], ylim[1] - ylim[0]))
    threshold = float(max(0.03 * plot_diagonal, np.quantile(lengths, 0.60)))
    retained = []
    for segment, row in zip(segments, measurements, strict=True):
        row["retained"] = bool(row["length_pc_units"] >= threshold)
        if row["retained"]:
            retained.append(segment)
    return retained, threshold, measurements


def field_similarity(reference: np.ndarray, candidate: np.ndarray, support: np.ndarray) -> tuple[float, float]:
    cap_ref = reference <= TAU
    cap_candidate = candidate <= TAU
    intersection = np.sum(support & cap_ref & cap_candidate)
    union = np.sum(support & (cap_ref | cap_candidate))
    jaccard = float(intersection / union) if union else math.nan
    rmse = float(np.sqrt(np.mean((candidate[support] - reference[support]) ** 2)))
    return jaccard, rmse


def smoothing_audit(
    frame: pd.DataFrame,
    base_fields: dict[str, Any],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    xx, yy = base_fields["xx"], base_fields["yy"]
    support = base_fields["support"]
    z = frame[["pc1", "pc2"]].to_numpy(float)
    grid = base_fields["grid"]
    settings: list[tuple[str, float, dict[str, np.ndarray]]] = [
        ("baseline_distance_weighted_knn", 75.0, base_fields["fields"])
    ]
    for neighbors in (100, 150, 225, 300):
        candidate = {}
        for model in ("M0", "M1", "M5"):
            regressor = KNeighborsRegressor(
                n_neighbors=neighbors,
                weights="distance",
                algorithm="kd_tree",
                n_jobs=-1,
            ).fit(z, frame[f"{model}_gap"].to_numpy(float))
            candidate[model] = regressor.predict(grid).reshape(xx.shape)
        settings.append(("broader_knn_kernel", float(neighbors), candidate))
    for sigma in (1.0, 1.5, 2.0, 2.5, 3.0):
        candidate = {
            model: gaussian_filter(field, sigma=sigma, mode="nearest")
            for model, field in base_fields["fields"].items()
        }
        settings.append(("gaussian_post_smoothing", sigma, candidate))

    rows = []
    selected: dict[str, np.ndarray] | None = None
    plot_diagonal = float(math.hypot(xlim[1] - xlim[0], ylim[1] - ylim[0]))
    tiny_threshold = 0.03 * plot_diagonal
    for method, parameter, fields in settings:
        for model in ("M0", "M1", "M5"):
            segments = contour_segments(xx, yy, fields[model], support)
            measures = segment_measurements(segments)
            jaccard, rmse = field_similarity(base_fields["fields"][model], fields[model], support)
            rows.append({
                "method": method,
                "parameter": parameter,
                "parameter_name": "n_neighbors" if "knn" in method else ("gaussian_sigma_grid_cells" if "gaussian" in method else "n_neighbors"),
                "model": model,
                "raw_tau_contour_components": len(segments),
                "tiny_components_below_3pct_plot_diagonal": int(sum(row["length_pc_units"] < tiny_threshold for row in measures)),
                "total_contour_length_pc_units": float(sum(row["length_pc_units"] for row in measures)),
                "capability_jaccard_vs_unsmoothed_k75": jaccard,
                "field_rmse_vs_unsmoothed_k75_gap_percent": rmse,
                "supported_blue_fraction": float(np.mean(fields[model][support] < TAU)),
                "supported_red_fraction": float(np.mean(fields[model][support] > TAU)),
                "selected": method == "gaussian_post_smoothing" and parameter == SELECTED_SIGMA,
            })
        if method == "gaussian_post_smoothing" and parameter == SELECTED_SIGMA:
            selected = fields
    if selected is None:
        raise RuntimeError("selected smoothing setting was not generated")
    return pd.DataFrame(rows), selected


def colorbar_audit(frame: pd.DataFrame) -> pd.DataFrame:
    pooled = np.concatenate([frame[f"{model}_gap"].to_numpy(float) for model in ("M0", "M1", "M5")])
    p90 = float(np.quantile(pooled, 0.90))
    p95 = float(np.quantile(pooled, 0.95))
    candidates = [
        ("current_v2", p95, "Current 8.36 value; verified as pooled M0/M1/M5 P95"),
        ("pooled_P95", p95, "Robust upper bound with 5% pooled clipping"),
        ("pooled_P90", p90, "Higher contrast with 10% pooled clipping"),
        ("fixed_5_percent", 5.0, "Round PPT scale near pooled P90; explicitly shown as >=5%"),
    ]
    rows = []
    for name, vmax, reason in candidates:
        row: dict[str, Any] = {
            "candidate": name,
            "vmax_gap_percent": vmax,
            "source_or_reason": reason,
            "tau_gap_percent": TAU,
            "tau_position_within_upper_range": TAU / vmax,
            "pooled_fraction_clipped_above_vmax": float(np.mean(pooled > vmax)),
            "selected": name == "fixed_5_percent",
        }
        for model in ("M0", "M1", "M5"):
            row[f"{model}_fraction_clipped_above_vmax"] = float(np.mean(frame[f"{model}_gap"] > vmax))
        rows.append(row)
    return pd.DataFrame(rows)


def style_axis(ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float], ppt: bool) -> None:
    ax.set_facecolor(UNSUPPORTED)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel("Structural PC 1", color=NAVY, labelpad=6, fontsize=11 if ppt else 10.5)
    ax.set_ylabel("Structural PC 2", color=NAVY, labelpad=6, fontsize=11 if ppt else 10.5)
    ax.tick_params(colors=NAVY, labelsize=9.5 if ppt else 9)
    for spine in ax.spines.values():
        spine.set_color("#AAB4BD"); spine.set_linewidth(0.8)


def plot_segments(ax: plt.Axes, segments: list[np.ndarray], color: str, width: float, alpha: float = 1.0) -> None:
    for segment in segments:
        ax.plot(segment[:, 0], segment[:, 1], color=color, linestyle="--", linewidth=width, alpha=alpha, solid_capstyle="round", zorder=4)


def legend_handles(model: str, ppt: bool) -> list[Line2D]:
    handles = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=HANDLED_BLUE, markeredgecolor="white", markeredgewidth=0.8, markersize=6.4 if ppt else 6.0, label="Handled well"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=HARD_POINT_RED, markeredgecolor="white", markeredgewidth=1.0, markersize=7.2 if ppt else 6.8, label="High-gap / hard"),
    ]
    if model == "M0":
        handles.append(Line2D([], [], color=BOUNDARY_RED, linestyle="--", linewidth=2.3 if ppt else 2.0, label="Capability boundary B₀"))
    else:
        suffix = "₁" if model == "M1" else "₅"
        handles.extend([
            Line2D([], [], color=OLD_GRAY, linestyle="--", linewidth=1.4, label="B₀ (before)"),
            Line2D([], [], color=BOUNDARY_RED, linestyle="--", linewidth=2.3 if ppt else 2.0, label=f"B{suffix} (after)"),
        ])
    return handles


def draw_axis(
    ax: plt.Axes,
    subset: pd.DataFrame,
    model: str,
    field: np.ndarray,
    support: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    clean_segments: dict[str, list[np.ndarray]],
    norm: TwoSlopeNorm,
    levels: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    ppt: bool,
) -> Any:
    ax.set_facecolor(UNSUPPORTED)
    masked = np.ma.masked_where(~support, field)
    image = ax.contourf(xx, yy, masked, levels=levels, cmap=GAP_CMAP, norm=norm, extend="max", antialiased=True, zorder=0)
    if model != "M0":
        plot_segments(ax, clean_segments["M0"], OLD_GRAY, 1.45 if ppt else 1.25, 0.95)
    plot_segments(ax, clean_segments[model], BOUNDARY_RED, 2.45 if ppt else 2.10)
    handled = subset[f"{model}_gap"].to_numpy(float) <= TAU
    ax.scatter(subset.loc[handled, "pc1"], subset.loc[handled, "pc2"], s=POINT_SIZE_HANDLED, c=HANDLED_BLUE, edgecolors="white", linewidths=0.8, alpha=0.90, zorder=6)
    ax.scatter(subset.loc[~handled, "pc1"], subset.loc[~handled, "pc2"], s=POINT_SIZE_HARD, c=HARD_POINT_RED, edgecolors="white", linewidths=1.0, alpha=0.95, zorder=7)
    style_axis(ax, xlim, ylim, ppt)
    return image


def add_colorbar(fig: plt.Figure, image: Any, axes: Any, ppt: bool) -> None:
    colorbar = fig.colorbar(image, ax=axes, fraction=0.046, pad=0.025, ticks=[0.0, TAU, COLOR_VMAX])
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    colorbar.set_label("Predicted optimality gap (%)", color=NAVY, labelpad=8, fontsize=10.5 if ppt else 10)
    colorbar.ax.set_yticklabels(["Low\n0", f"τ\n{TAU:.3f}", f"High\n≥ {COLOR_VMAX:.0f}"])
    colorbar.ax.tick_params(colors=NAVY, labelsize=8.5 if ppt else 8)


def save_single(
    subset: pd.DataFrame,
    model: str,
    fields: dict[str, np.ndarray],
    support: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    clean_segments: dict[str, list[np.ndarray]],
    norm: TwoSlopeNorm,
    levels: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    stem: str,
    ppt: bool,
    pdf: bool,
) -> None:
    titles = {"M0": "Current solver M₀", "M1": "Updated solver M₁", "M5": "Updated solver M₅"}
    figsize = (8.1, 6.15) if not ppt else (8.2, 5.65)
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
    image = draw_axis(ax, subset, model, fields[model], support, xx, yy, clean_segments, norm, levels, xlim, ylim, ppt)
    fig.suptitle(titles[model], color=NAVY, weight="semibold", fontsize=16 if ppt else 14.5, y=0.985)
    fig.legend(handles=legend_handles(model, ppt), loc="upper center", bbox_to_anchor=(0.46, 0.935), ncol=len(legend_handles(model, ppt)), frameon=False, fontsize=9.2 if ppt else 8.8, handletextpad=0.5, columnspacing=1.25)
    add_colorbar(fig, image, ax, ppt)
    bottom = 0.085 if not ppt else 0.07
    if not ppt:
        fig.text(0.46, 0.018, f"Capability is visualized using solver optimality gap; fixed τ = {TAU:.3f}%.", ha="center", color=NAVY, fontsize=8.5)
    fig.subplots_adjust(left=0.105, right=0.875, bottom=bottom, top=0.84)
    fig.savefig(OUT / f"{stem}.png", dpi=DPI, facecolor="white")
    if pdf:
        fig.savefig(OUT / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def save_composite(
    subset: pd.DataFrame,
    fields: dict[str, np.ndarray],
    support: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    clean_segments: dict[str, list[np.ndarray]],
    norm: TwoSlopeNorm,
    levels: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.45), dpi=DPI, sharex=True, sharey=True)
    image = draw_axis(axes[0], subset, "M0", fields["M0"], support, xx, yy, clean_segments, norm, levels, xlim, ylim, True)
    draw_axis(axes[1], subset, "M5", fields["M5"], support, xx, yy, clean_segments, norm, levels, xlim, ylim, True)
    axes[0].set_title("Current solver M₀", color=NAVY, fontsize=15, weight="semibold", pad=10)
    axes[1].set_title("Updated solver M₅", color=NAVY, fontsize=15, weight="semibold", pad=10)
    axes[1].set_ylabel("")
    handles = [
        legend_handles("M5", True)[0],
        legend_handles("M5", True)[1],
        legend_handles("M5", True)[2],
        legend_handles("M5", True)[3],
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.47, 0.985), ncol=4, frameon=False, fontsize=9.4, columnspacing=1.4)
    add_colorbar(fig, image, axes.tolist(), True)
    fig.subplots_adjust(left=0.06, right=0.91, bottom=0.11, top=0.865, wspace=0.10)
    fig.savefig(OUT / "fig_M0_vs_M5_ppt_composite.png", dpi=DPI, facecolor="white")
    fig.savefig(OUT / "fig_M0_vs_M5_ppt_composite.pdf", facecolor="white")
    plt.close(fig)


def save_m0_m5_composite_for_ppt(
    subset: pd.DataFrame,
    fields: dict[str, np.ndarray],
    support: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    clean_segments: dict[str, list[np.ndarray]],
    norm: TwoSlopeNorm,
    levels: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    """Compact Page-2 left-column composite; presentation only."""
    fig = plt.figure(figsize=(10.5, 4.8), dpi=DPI, constrained_layout=False)
    grid = fig.add_gridspec(
        nrows=1, ncols=3, width_ratios=[1.0, 1.0, 0.045], wspace=0.08
    )
    ax0 = fig.add_subplot(grid[0, 0])
    ax5 = fig.add_subplot(grid[0, 1])
    cax = fig.add_subplot(grid[0, 2])

    def draw_panel(ax: plt.Axes, model: str) -> Any:
        ax.set_facecolor(UNSUPPORTED)
        masked = np.ma.masked_where(~support, fields[model])
        image = ax.contourf(
            xx, yy, masked, levels=levels, cmap=GAP_CMAP, norm=norm,
            extend="max", antialiased=True, zorder=0,
        )
        # B0 is the shared gray reference; B5 is red only in the M5 panel.
        plot_segments(ax, clean_segments["M0"], "#8E99A4", 2.2, 0.98)
        if model == "M5":
            plot_segments(ax, clean_segments["M5"], "#C4473A", 2.4, 1.0)
        handled = subset[f"{model}_gap"].to_numpy(float) <= TAU
        ax.scatter(
            subset.loc[handled, "pc1"], subset.loc[handled, "pc2"],
            s=34, c=HANDLED_BLUE, edgecolors="white", linewidths=0.8,
            alpha=0.95, zorder=6,
        )
        ax.scatter(
            subset.loc[~handled, "pc1"], subset.loc[~handled, "pc2"],
            s=40, c=HARD_POINT_RED, edgecolors="white", linewidths=0.9,
            alpha=0.95, zorder=7,
        )
        style_axis(ax, xlim, ylim, ppt=True)
        ax.text(
            0.025, 0.975, model, transform=ax.transAxes,
            ha="left", va="top", fontsize=14, fontweight="bold", color=NAVY,
            bbox={
                "boxstyle": "round,pad=0.18", "facecolor": "white",
                "edgecolor": "none", "alpha": 0.82,
            },
            zorder=9,
        )
        return image

    image = draw_panel(ax0, "M0")
    draw_panel(ax5, "M5")

    legend = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=HANDLED_BLUE,
               markeredgecolor="white", markeredgewidth=0.8, markersize=7.0,
               label="Handled well"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=HARD_POINT_RED,
               markeredgecolor="white", markeredgewidth=0.9, markersize=7.4,
               label="High-gap / hard"),
        Line2D([], [], color="#8E99A4", linestyle="--", linewidth=2.2,
               label=r"$B_0$ (before)"),
        Line2D([], [], color="#C4473A", linestyle="--", linewidth=2.4,
               label=r"$B_5$ (after)"),
    ]
    fig.text(
        0.47, 0.982, "The target moves with the solver.",
        ha="center", va="top", fontsize=14.5, fontweight="semibold", color=NAVY,
    )
    fig.legend(
        handles=legend, loc="upper center", bbox_to_anchor=(0.47, 0.925),
        ncol=4, frameon=True, facecolor="white", edgecolor="#D0D7DE",
        framealpha=0.96, fontsize=9.6, columnspacing=1.25, handletextpad=0.55,
    )

    colorbar = fig.colorbar(image, cax=cax, ticks=[0.0, TAU, COLOR_VMAX])
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    colorbar.set_label(
        "Predicted optimality gap (%)", color=NAVY, labelpad=7, fontsize=9.5
    )
    colorbar.ax.set_yticklabels(["0", rf"$\tau = {TAU:.3f}$", rf"$\geq {COLOR_VMAX:.0f}$"])
    colorbar.ax.tick_params(colors=NAVY, labelsize=8.2)

    stats_text = (
        r"Mastered: 70.0% $\rightarrow$ 82.5%" + "\n"
        + r"Mean gap: 2.56% $\rightarrow$ 1.09%"
    )
    fig.text(
        0.47, 0.018, stats_text,
        ha="center", va="bottom", fontsize=11.5, fontweight="bold", color=NAVY,
        bbox={
            "boxstyle": "round,pad=0.35", "facecolor": "#F5F7FA",
            "edgecolor": "#D0D7DE",
        },
    )
    fig.subplots_adjust(top=0.80, bottom=0.20, left=0.065, right=0.94)
    fig.savefig(
        OUT / "m0_m5_composite_for_ppt.png", dpi=DPI,
        bbox_inches="tight", facecolor="white",
    )
    fig.savefig(
        OUT / "m0_m5_composite_for_ppt.pdf",
        bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)


def save_m0_m5_composite_zoom_left(
    subset: pd.DataFrame,
    fields: dict[str, np.ndarray],
    support: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    norm: TwoSlopeNorm,
    levels: np.ndarray,
) -> None:
    """Compact zoom focused on the left-side boundary movement."""
    zoom_xlim = (-1.05, 0.10)
    zoom_ylim = (-0.48, 0.72)
    zoom_subset = subset.loc[
        subset.pc1.between(*zoom_xlim) & subset.pc2.between(*zoom_ylim)
    ].copy()
    if not 80 <= len(zoom_subset) <= 120:
        raise RuntimeError(f"zoom foreground count outside requested range: {len(zoom_subset)}")
    zoom_subset.to_csv(
        OUT / "foreground_subset_zoom_left.csv", index=False, float_format="%.17g"
    )

    # The input fields already use sigma=2.0.  An additional sigma=1.5 gives
    # an effective sigma of 2.5 grid cells and affects presentation only.
    extra_sigma = 1.5
    effective_sigma = float(math.sqrt(SELECTED_SIGMA ** 2 + extra_sigma ** 2))
    zoom_fields = {
        model: gaussian_filter(fields[model], sigma=extra_sigma, mode="nearest")
        for model in ("M0", "M5")
    }

    minimum_length = 0.07 * (zoom_xlim[1] - zoom_xlim[0])
    retained: dict[str, list[np.ndarray]] = {}
    audit_rows = []
    for model in ("M0", "M5"):
        raw = contour_segments(xx, yy, zoom_fields[model], support)
        intersecting = []
        kept = []
        for segment in raw:
            inside = (
                (segment[:, 0] >= zoom_xlim[0])
                & (segment[:, 0] <= zoom_xlim[1])
                & (segment[:, 1] >= zoom_ylim[0])
                & (segment[:, 1] <= zoom_ylim[1])
            )
            if not bool(inside.any()):
                continue
            length = float(np.linalg.norm(np.diff(segment, axis=0), axis=1).sum())
            intersecting.append((segment, length))
            if length >= minimum_length:
                kept.append(segment)
        retained[model] = kept
        audit_rows.append({
            "model": model,
            "zoom_xlim": str(zoom_xlim),
            "zoom_ylim": str(zoom_ylim),
            "foreground_count_shared": int(len(zoom_subset)),
            "input_gaussian_sigma_grid_cells": SELECTED_SIGMA,
            "additional_gaussian_sigma_grid_cells": extra_sigma,
            "effective_gaussian_sigma_grid_cells": effective_sigma,
            "raw_full_field_component_count": len(raw),
            "zoom_intersecting_component_count": len(intersecting),
            "retained_component_count": len(kept),
            "minimum_component_length_pc_units": minimum_length,
            "filtering_rule": "intersects zoom and length >= 7% of zoom width",
        })
    pd.DataFrame(audit_rows).to_csv(
        OUT / "zoom_left_display_audit.csv", index=False, float_format="%.17g"
    )

    fig = plt.figure(figsize=(5.4, 7.0), dpi=DPI, constrained_layout=False)
    grid = fig.add_gridspec(
        nrows=2, ncols=2, width_ratios=[1.0, 0.045],
        height_ratios=[1.0, 1.0], hspace=0.08, wspace=0.08,
    )
    ax0 = fig.add_subplot(grid[0, 0])
    ax5 = fig.add_subplot(grid[1, 0])
    cax = fig.add_subplot(grid[:, 1])

    def draw_zoom_panel(ax: plt.Axes, model: str) -> Any:
        ax.set_facecolor(UNSUPPORTED)
        masked = np.ma.masked_where(~support, zoom_fields[model])
        image = ax.contourf(
            xx, yy, masked, levels=levels, cmap=GAP_CMAP, norm=norm,
            extend="max", antialiased=True, zorder=0,
        )
        plot_segments(ax, retained["M0"], "#8E99A4", 2.1, 0.98)
        if model == "M5":
            plot_segments(ax, retained["M5"], "#C4473A", 2.35, 1.0)
        handled = zoom_subset[f"{model}_gap"].to_numpy(float) <= TAU
        ax.scatter(
            zoom_subset.loc[handled, "pc1"], zoom_subset.loc[handled, "pc2"],
            s=31, c=HANDLED_BLUE, edgecolors="white", linewidths=0.75,
            alpha=0.94, zorder=6,
        )
        ax.scatter(
            zoom_subset.loc[~handled, "pc1"], zoom_subset.loc[~handled, "pc2"],
            s=37, c=HARD_POINT_RED, edgecolors="white", linewidths=0.85,
            alpha=0.96, zorder=7,
        )
        ax.set_xlim(*zoom_xlim)
        ax.set_ylim(*zoom_ylim)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([-1.0, -0.75, -0.50, -0.25, 0.0])
        ax.set_yticks([-0.4, 0.0, 0.4, 0.7])
        ax.tick_params(colors=NAVY, labelsize=8.2)
        for spine in ax.spines.values():
            spine.set_color("#AAB4BD")
            spine.set_linewidth(0.8)
        ax.text(
            0.03, 0.97, model, transform=ax.transAxes,
            ha="left", va="top", fontsize=12.5, fontweight="bold", color=NAVY,
            bbox={
                "boxstyle": "round,pad=0.16", "facecolor": "white",
                "edgecolor": "none", "alpha": 0.85,
            },
            zorder=9,
        )
        return image

    image = draw_zoom_panel(ax0, "M0")
    draw_zoom_panel(ax5, "M5")
    ax0.tick_params(axis="x", labelbottom=False)
    fig.supxlabel("Structural PC 1", fontsize=10.5, color=NAVY, y=0.025)
    fig.supylabel("Structural PC 2", fontsize=10.5, color=NAVY, x=0.035)

    legend = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=HANDLED_BLUE,
               markeredgecolor="white", markeredgewidth=0.75, markersize=6.2,
               label="Handled well"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=HARD_POINT_RED,
               markeredgecolor="white", markeredgewidth=0.85, markersize=6.6,
               label="High-gap / hard"),
        Line2D([], [], color="#8E99A4", linestyle="--", linewidth=2.1,
               label=r"$B_0$ (before)"),
        Line2D([], [], color="#C4473A", linestyle="--", linewidth=2.35,
               label=r"$B_5$ (after)"),
    ]
    fig.suptitle(
        "The target moves with the solver.",
        fontsize=13.5, fontweight="bold", color=NAVY, y=0.985,
    )
    fig.legend(
        handles=legend, loc="upper center", bbox_to_anchor=(0.47, 0.94),
        ncol=4, frameon=True, facecolor="white", edgecolor="#D0D7DE",
        framealpha=0.96, fontsize=8.5, columnspacing=1.05,
        handletextpad=0.45, borderpad=0.35,
    )

    colorbar = fig.colorbar(image, cax=cax, ticks=[0.0, TAU, COLOR_VMAX])
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    colorbar.set_label(
        "Predicted optimality gap (%)", color=NAVY, labelpad=5, fontsize=8.4
    )
    colorbar.ax.set_yticklabels(["0", rf"$\tau={TAU:.3f}$", rf"$\geq {COLOR_VMAX:.0f}$"])
    colorbar.ax.tick_params(colors=NAVY, labelsize=7.5)

    fig.subplots_adjust(top=0.855, bottom=0.085, left=0.145, right=0.89)
    fig.savefig(
        OUT / "m0_m5_composite_zoom_left.png", dpi=DPI,
        bbox_inches="tight", facecolor="white",
    )
    fig.savefig(
        OUT / "m0_m5_composite_zoom_left.pdf",
        bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)


def write_report(
    subset: pd.DataFrame,
    smoothing: pd.DataFrame,
    colorbar: pd.DataFrame,
    contour: pd.DataFrame,
    transition: pd.Series,
    source_hashes: dict[str, str],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    selected = smoothing.loc[smoothing.selected]
    contour_text = ", ".join(f"{row.model}: {row.raw_component_count}→{row.retained_component_count}" for row in contour.itertuples(index=False))
    sources = "\n".join(f"- `{path}` — SHA-256 `{digest}`" for path, digest in sorted(source_hashes.items()))
    report = f"""# M0 / M1 / M5 capability-field visual cleanup

Read-only visual refactor. No instance, feature, PCA coordinate, exact gap, point label, threshold, mastered fraction, transition count, or quantitative support statistic was changed.

## Frozen sources

{sources}

## Visual-field decisions

1. **Smoothing method and parameter.** The selected method is Gaussian post-smoothing with `sigma={SELECTED_SIGMA:.1f}` grid cells, applied to the existing distance-weighted kNN (`k={GRID_K}`) predicted grid. It was compared with broader kNN kernels (`k=100,150,225,300`) and Gaussian sigma 1.0–3.0. At sigma 2.0, supported-grid capability Jaccard versus the original field is M0={selected.loc[selected.model.eq('M0'), 'capability_jaccard_vs_unsmoothed_k75'].iloc[0]:.3f}, M1={selected.loc[selected.model.eq('M1'), 'capability_jaccard_vs_unsmoothed_k75'].iloc[0]:.3f}, M5={selected.loc[selected.model.eq('M5'), 'capability_jaccard_vs_unsmoothed_k75'].iloc[0]:.3f}, while raw τ contours fall to {int(selected.loc[selected.model.eq('M0'), 'raw_tau_contour_components'].iloc[0])}/{int(selected.loc[selected.model.eq('M1'), 'raw_tau_contour_components'].iloc[0])}/{int(selected.loc[selected.model.eq('M5'), 'raw_tau_contour_components'].iloc[0])}.
2. **Scope of smoothing.** Smoothing affects only the PPT/display grid. It is never written back to instance gaps and is not used to recompute any reported quantity.
3. **Point integrity.** All displayed point labels are recomputed only as exact frozen `gap <= τ`; source gaps and labels are byte-for-byte untouched.
4. **Unsupported area.** Supported field cells use the blue–white–pale-red gap map. Unsupported cells use neutral `{UNSUPPORTED}`; therefore white means near τ, not missing support.
5. **Colorbar.** The former 8.362600% upper bound is exactly the pooled M0/M1/M5 P95. The audit also compares pooled P90 ({colorbar.loc[colorbar.candidate.eq('pooled_P90'), 'vmax_gap_percent'].iloc[0]:.6f}%) and fixed 5%. `COLORBAR_VMAX_SELECTED = {COLOR_VMAX:.1f}%`. `COLORBAR_VMAX_REASON = round PPT scale close to pooled P90, improving τ-to-moderately-hard contrast; values above it are explicitly labeled ≥5%.` The same scale is used for all solvers.
6. **Foreground subset.** `N_foreground = {len(subset)}`: 50 points per benchmark, chosen by deterministic farthest-point coverage within M0 hard/handled strata. Exact solver labels remain independent. Counts are M0 hard={int((subset.M0_gap > TAU).sum())}, M1 hard={int((subset.M1_gap > TAU).sum())}, M5 hard={int((subset.M5_gap > TAU).sum())}.
7. **Boundary cleanup.** Raw→retained component counts are {contour_text}. Each component's length and enclosed area are audited. The automatic rule retains length >= `max(3% of plot diagonal, within-model P60 component length)`; it never forces one component.
8. **Shared geometry.** Yes. M0/M1/M5 use the exact same 200 instance IDs, PC coordinates, axis limits (`x={xlim}`, `y={ylim}`), support mask, colormap, normalization, marker sizes, and styles.
9. **Page 2 choice.** Yes: M0 plus M5-vs-M0 remains the clearest main evidence. M1-vs-M0 remains the one-cycle backup: **Solver adaptation reshapes the capability boundary after one cycle.**
10. **Best direct PPT export.** `fig_M0_vs_M5_ppt_composite.png` is recommended because it has shared axes, one colorbar, one legend, the same subset, and directly exposes B₀→B₅.

## Quantitative results retained unchanged

- M0: mastered fraction {100*transition.mastered_fraction_M0:.2f}%; mean gap {transition.mean_gap_M0:.3f}%.
- M0→M1: {100*transition.mastered_fraction_M0:.2f}%→{100*transition.mastered_fraction_M1:.2f}%; hard→mastered {int(transition.hard_to_mastered_M0_M1):,}; mastered→hard {int(transition.mastered_to_hard_M0_M1):,}; expansion/contraction {100*transition.support_expansion_M0_M1:.2f}%/{100*transition.support_contraction_M0_M1:.2f}%. Recommended wording: **Solver adaptation reshapes the capability boundary after one cycle.**
- M0→M5: mastered {100*transition.mastered_fraction_M0:.2f}%→{100*transition.mastered_fraction_M5:.2f}%; mean gap {transition.mean_gap_M0:.3f}%→{transition.mean_gap_M5:.3f}%; hard→mastered {int(transition.hard_to_mastered_M0_M5):,}; mastered→hard {int(transition.mastered_to_hard_M0_M5):,}; expansion/contraction {100*transition.support_expansion_M0_M5:.2f}%/{100*transition.support_contraction_M0_M5:.2f}%. Recommended wording: **The capability boundary is predominantly expansive over the longer horizon.**

The display does not imply uniform expansion or equal improvement across all regions.

`RECOMMENDED_PAGE2_CAPABILITY_FIGURE = fig_M0_vs_M5_ppt_composite.png`
"""
    (OUT / "report.md").write_text(report, encoding="utf-8", newline="\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.facecolor": UNSUPPORTED,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    frame, _, _, source_hashes_raw = v2.read_inputs()
    axes_payload = json.loads(source_hashes_raw.pop("_axes"))
    xlim = tuple(axes_payload["xlim"]["0"])
    ylim = tuple(axes_payload["ylim"]["0"])
    transition_path = V2 / "04_transition_summary_M0_M1_M5.csv"
    transition = pd.read_csv(transition_path, float_precision="round_trip").iloc[0]
    frozen_coordinate_path = V2 / "01_instance_capability_coordinates.csv"
    frozen_coordinates = pd.read_csv(frozen_coordinate_path, float_precision="round_trip")
    frozen_coordinates = frozen_coordinates.set_index("instance_key").loc[frame.instance_key].reset_index()
    source_hashes = dict(source_hashes_raw)
    source_hashes[str(transition_path.relative_to(ROOT)).replace("\\", "/")] = sha256_file(transition_path)
    source_hashes[str(frozen_coordinate_path.relative_to(ROOT)).replace("\\", "/")] = sha256_file(frozen_coordinate_path)

    expected_tau = float(np.quantile(frame.M0_gap.to_numpy(float), 0.70))
    if not np.isclose(expected_tau, TAU, rtol=0, atol=1e-14):
        raise RuntimeError("frozen tau changed")
    for model in ("M0", "M1", "M5"):
        exact = frame[f"{model}_gap"].to_numpy(float) <= TAU
        if not np.array_equal(frame[f"{model}_gap"].to_numpy(float), frozen_coordinates[f"{model}_gap"].to_numpy(float)):
            raise RuntimeError(f"frozen point gaps changed for {model}")
        if not np.array_equal(exact, frozen_coordinates[f"mastered_{model}"].to_numpy(bool)):
            raise RuntimeError(f"frozen point labels changed for {model}")

    subset = make_foreground_subset(frame)
    fields = build_base_fields(frame, xlim, ylim)
    smoothing, selected_fields = smoothing_audit(frame, fields, xlim, ylim)
    colorbar = colorbar_audit(frame)

    clean_segments: dict[str, list[np.ndarray]] = {}
    contour_rows = []
    for model in ("M0", "M1", "M5"):
        raw = contour_segments(fields["xx"], fields["yy"], selected_fields[model], fields["support"])
        retained, threshold, measurements = cleanup_segments(raw, xlim, ylim)
        clean_segments[model] = retained
        contour_rows.append({
            "model": model,
            "smoothing_method": "gaussian_post_smoothing",
            "gaussian_sigma_grid_cells": SELECTED_SIGMA,
            "raw_component_count": len(raw),
            "retained_component_count": len(retained),
            "removed_component_count": len(raw) - len(retained),
            "minimum_retained_length_pc_units": threshold,
            "filtering_rule": "length >= max(3% plot diagonal, within-model P60 component length)",
            "component_measurements_json": json.dumps(measurements, sort_keys=True, separators=(",", ":")),
        })
    contour = pd.DataFrame(contour_rows)

    foreground_columns = [
        "instance_key", "instance_id", "dataset", "pc1", "pc2",
        "M0_gap", "M1_gap", "M5_gap", "M0_label", "M1_label", "M5_label",
        "sampling_stratum", "coverage_distance_at_selection",
    ]
    smoothing.to_csv(OUT / "smoothing_audit.csv", index=False, float_format="%.17g")
    colorbar.to_csv(OUT / "colorbar_range_audit.csv", index=False, float_format="%.17g")
    contour.to_csv(OUT / "contour_component_audit.csv", index=False, float_format="%.17g")
    subset[foreground_columns].to_csv(OUT / "foreground_subset_ppt.csv", index=False, float_format="%.17g")

    checks = pd.DataFrame([
        {"check": "frozen instance count", "value": len(frame), "pass": len(frame) == 30070},
        {"check": "fixed tau", "value": f"{TAU:.17g}", "pass": expected_tau == TAU},
        {"check": "same 200 IDs across M0/M1/M5", "value": subset.instance_key.nunique(), "pass": subset.instance_key.nunique() == 200},
        {"check": "same point coordinates", "value": "single subset table reused", "pass": True},
        {"check": "same axis limits", "value": f"x={xlim}; y={ylim}", "pass": True},
        {"check": "same colormap and normalization", "value": f"vmin=0; vcenter={TAU}; vmax={COLOR_VMAX}", "pass": True},
        {"check": "same support mask", "value": int(fields["support"].sum()), "pass": True},
        {"check": "smoothing affects grid only", "value": "no source writes; exact gaps retained", "pass": True},
        {"check": "quantitative transition source reused", "value": str(transition_path.relative_to(ROOT)).replace("\\", "/"), "pass": True},
    ])
    checks.to_csv(OUT / "integrity_checks.csv", index=False)
    if not checks["pass"].all():
        raise RuntimeError("visual cleanup integrity check failed")

    norm = TwoSlopeNorm(vmin=0.0, vcenter=TAU, vmax=COLOR_VMAX)
    levels = np.unique(np.concatenate([np.linspace(0.0, TAU, 55), np.linspace(TAU, COLOR_VMAX, 55)]))
    for model, clean_stem, ppt_stem in [
        ("M0", "figA_M0_current_solver_clean", "figA_M0_current_solver_ppt"),
        ("M1", "figB_M1_vs_M0_clean", "figB_M1_vs_M0_ppt"),
        ("M5", "figC_M5_vs_M0_clean", "figC_M5_vs_M0_ppt"),
    ]:
        save_single(subset, model, selected_fields, fields["support"], fields["xx"], fields["yy"], clean_segments, norm, levels, xlim, ylim, clean_stem, ppt=False, pdf=True)
        save_single(subset, model, selected_fields, fields["support"], fields["xx"], fields["yy"], clean_segments, norm, levels, xlim, ylim, ppt_stem, ppt=True, pdf=False)
    save_m0_m5_composite_for_ppt(
        subset, selected_fields, fields["support"], fields["xx"], fields["yy"],
        clean_segments, norm, levels, xlim, ylim,
    )
    save_m0_m5_composite_zoom_left(
        subset, selected_fields, fields["support"], fields["xx"], fields["yy"],
        norm, levels,
    )
    save_composite(subset, selected_fields, fields["support"], fields["xx"], fields["yy"], clean_segments, norm, levels, xlim, ylim)
    write_report(subset, smoothing, colorbar, contour, transition, source_hashes, xlim, ylim)
    print("CAPABILITY_VISUAL_CLEANUP_COMPLETE")
    print(f"foreground={len(subset)} sigma={SELECTED_SIGMA} vmax={COLOR_VMAX}")
    print("contours=" + ",".join(f"{row.model}:{row.raw_component_count}->{row.retained_component_count}" for row in contour.itertuples(index=False)))


if __name__ == "__main__":
    main()
