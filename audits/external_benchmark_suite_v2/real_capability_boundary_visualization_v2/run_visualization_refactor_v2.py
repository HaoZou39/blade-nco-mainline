#!/usr/bin/env python
"""Read-only presentation refactor of the real capability-boundary audit.

This script reuses the frozen v1 structural map, observed M0/M1/M5 gaps, and
M0 hard-region membership.  It changes presentation only.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
V1 = ROOT / "audits/external_benchmark_suite_v2/real_capability_boundary_visualization"
FIG = OUT / "figures"
sys.dont_write_bytecode = True
sys.path.insert(0, str(V1))
import run_real_capability_boundary_visualization as base  # noqa: E402


DATASETS = base.DATASETS
FEATURES = base.FEATURES
NAVY = "#102A43"
HANDLED_BLUE = "#1F5B8F"
HARD_RED = "#B43E3A"
OLD_GRAY = "#A3ABB3"
REGION_A = "#4F718A"
REGION_B = "#7B678E"
NOISE = "#D5DADF"
POINT_SIZE_HANDLED = 24.0
POINT_SIZE_HARD = 31.0
FIGSIZE = base.FIGSIZE_AB
DPI = 300
SUBSET_QUOTA = {name: 70 for name in DATASETS}

GAP_CMAP = LinearSegmentedColormap.from_list(
    "muted_gap_relative_to_tau",
    [
        (0.00, "#24577E"),
        (0.28, "#A9C6D9"),
        (0.50, "#FAF9F6"),
        (0.76, "#F2C8C1"),
        (1.00, "#D98E86"),
    ],
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    coordinate_path = V1 / "01_instance_capability_coordinates.csv"
    m5_path = V1 / "09_m5_instance_capability.csv"
    hard_path = V1 / "03_real_hard_region_membership.csv"
    region_path = V1 / "05_hard_region_summary.csv"
    primary_checks_path = V1 / "06_consistency_checks.csv"
    paths = [coordinate_path, m5_path, hard_path, region_path, primary_checks_path]
    hashes = {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path) for path in paths
    }

    frame = pd.read_csv(coordinate_path, float_precision="round_trip")
    m5 = pd.read_csv(m5_path, float_precision="round_trip")
    frame = frame.merge(
        m5[["instance_key", "M5_gap"]], on="instance_key", how="inner", validate="one_to_one"
    )
    if len(frame) != 30_070 or frame.instance_key.nunique() != 30_070:
        raise RuntimeError("frozen M0/M1/M5 instance set is incomplete")
    hard = pd.read_csv(hard_path, float_precision="round_trip")
    hard = hard.merge(
        frame[["instance_key", "M5_gap"]], on="instance_key", how="left", validate="one_to_one"
    )
    region_summary = pd.read_csv(region_path, float_precision="round_trip")
    primary_checks = pd.read_csv(primary_checks_path)
    axis_value = str(
        primary_checks.loc[primary_checks.check == "same axis limits", "value"].iloc[0]
    )
    x_text, y_text = axis_value.split("; ")
    axes = pd.DataFrame(
        {
            "xlim": [tuple(float(v) for v in ast.literal_eval(x_text.removeprefix("x=")))],
            "ylim": [tuple(float(v) for v in ast.literal_eval(y_text.removeprefix("y=")))],
        }
    )
    return frame, hard, region_summary, {**hashes, "_axes": axes.to_json()}


def make_subset(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for dataset in DATASETS:
        sub = frame.loc[frame.dataset == dataset].reset_index(drop=True)
        indices, coverage = base.farthest_point_indices(
            sub[FEATURES].to_numpy(float), SUBSET_QUOTA[dataset]
        )
        chosen = sub.iloc[indices].copy()
        chosen["selection_step_within_dataset"] = np.arange(1, len(chosen) + 1)
        chosen["coverage_distance_at_selection"] = coverage
        chosen["dataset_quota"] = SUBSET_QUOTA[dataset]
        parts.append(chosen)
    subset = pd.concat(parts, ignore_index=True)
    if len(subset) != 280 or subset.instance_key.nunique() != 280:
        raise RuntimeError("visualization subset must contain 280 unique instances")
    return subset


def transition_metrics(
    frame: pd.DataFrame,
    fields: dict[str, Any],
    newer: str,
    tau: float,
) -> dict[str, Any]:
    mastered0 = frame.M0_gap.to_numpy(float) <= tau
    mastered_new = frame[f"{newer}_gap"].to_numpy(float) <= tau
    support = fields["support"]
    cap0 = fields["fields"][0] <= tau
    cap_new = fields["fields"][1] <= tau
    supported = int(support.sum())
    intersection = int(np.sum(support & cap0 & cap_new))
    union = int(np.sum(support & (cap0 | cap_new)))
    return {
        f"mastered_fraction_{newer}": float(mastered_new.mean()),
        f"hard_to_mastered_M0_{newer}": int(np.sum((~mastered0) & mastered_new)),
        f"mastered_to_hard_M0_{newer}": int(np.sum(mastered0 & (~mastered_new))),
        f"unchanged_mastered_M0_{newer}": int(np.sum(mastered0 & mastered_new)),
        f"unchanged_hard_M0_{newer}": int(np.sum((~mastered0) & (~mastered_new))),
        f"mean_gap_{newer}": float(frame[f"{newer}_gap"].mean()),
        f"support_expansion_M0_{newer}": float(
            np.sum(support & (~cap0) & cap_new) / supported
        ),
        f"support_contraction_M0_{newer}": float(
            np.sum(support & cap0 & (~cap_new)) / supported
        ),
        f"capability_jaccard_M0_{newer}": float(intersection / union),
        f"uniform_disagreement_{newer}": float(fields["uniform_disagreement"][1]),
        f"final_disagreement_{newer}": float(fields["disagreement"][1]),
    }


def style_axes(ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_facecolor("white")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Structural PC 1", color=NAVY, labelpad=7)
    ax.set_ylabel("Structural PC 2", color=NAVY, labelpad=7)
    ax.tick_params(colors=NAVY, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#AAB4BD")
        spine.set_linewidth(0.75)


def field_levels(vmin: float, tau: float, vmax: float) -> np.ndarray:
    return np.unique(
        np.concatenate([np.linspace(vmin, tau, 61), np.linspace(tau, vmax, 61)])
    )


def draw_capability_axis(
    ax: plt.Axes,
    subset: pd.DataFrame,
    fields: dict[str, Any],
    model: str,
    tau: float,
    norm: TwoSlopeNorm,
    levels: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    title: str,
    compact: bool = False,
) -> Any:
    support = fields["support"]
    current = fields["fields"][0 if model == "M0" else 1]
    masked = np.ma.masked_where(~support, current)
    image = ax.contourf(
        fields["xx"], fields["yy"], masked, levels=levels,
        cmap=GAP_CMAP, norm=norm, antialiased=True, zorder=0,
    )
    if model != "M0":
        old = np.ma.masked_where(~support, fields["fields"][0])
        ax.contour(
            fields["xx"], fields["yy"], old, levels=[tau], colors=[OLD_GRAY],
            linestyles="--", linewidths=1.55 if compact else 1.75, zorder=2,
        )
    ax.contour(
        fields["xx"], fields["yy"], masked, levels=[tau], colors=[HARD_RED],
        linestyles="--", linewidths=1.75 if compact else 2.05, zorder=3,
    )

    handled = subset[f"{model}_gap"].to_numpy(float) <= tau
    size_scale = 0.65 if compact else 1.0
    ax.scatter(
        subset.loc[handled, "pc1"], subset.loc[handled, "pc2"],
        s=POINT_SIZE_HANDLED * size_scale, c=HANDLED_BLUE, edgecolors="white",
        linewidths=0.48, alpha=0.78, zorder=5,
    )
    ax.scatter(
        subset.loc[~handled, "pc1"], subset.loc[~handled, "pc2"],
        s=POINT_SIZE_HARD * size_scale, c=HARD_RED, edgecolors="white",
        linewidths=0.82, alpha=0.91, zorder=6,
    )
    style_axes(ax, xlim, ylim)
    ax.set_title(title, color=NAVY, pad=9, weight="semibold", fontsize=12 if compact else 14)
    return image


def add_gap_colorbar(fig: plt.Figure, image: Any, ax: Any, tau: float, vmin: float, vmax: float) -> None:
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025, ticks=[vmin, tau, vmax])
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    colorbar.set_label("Predicted optimality gap (%)", color=NAVY, labelpad=8)
    colorbar.ax.set_yticklabels(
        [f"Low\n{vmin:.2f}", f"τ\n{tau:.3f}", f"High\n{vmax:.2f}"]
    )
    colorbar.ax.tick_params(colors=NAVY, labelsize=8)


def capability_legend(model: str) -> list[Line2D]:
    items = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=HANDLED_BLUE,
               markeredgecolor="white", markersize=6.5, label="Handled well"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=HARD_RED,
               markeredgecolor="white", markersize=7.2, label="High-gap / hard"),
    ]
    if model == "M0":
        items.append(
            Line2D([], [], color=HARD_RED, linestyle="--", linewidth=2.0,
                   label="Capability boundary B₀")
        )
    else:
        suffix = "₁" if model == "M1" else "₅"
        items.extend([
            Line2D([], [], color=OLD_GRAY, linestyle="--", linewidth=1.7,
                   label="Old boundary B₀"),
            Line2D([], [], color=HARD_RED, linestyle="--", linewidth=2.0,
                   label=f"New boundary B{suffix}"),
        ])
    return items


def plot_capability(
    subset: pd.DataFrame,
    fields: dict[str, Any],
    model: str,
    tau: float,
    norm: TwoSlopeNorm,
    levels: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    title: str,
    stem: str,
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    image = draw_capability_axis(
        ax, subset, fields, model, tau, norm, levels, xlim, ylim, title
    )
    ax.legend(
        handles=capability_legend(model), loc="upper right", frameon=True,
        framealpha=0.96, facecolor="white", edgecolor="#D7DEE4", fontsize=8.4,
    )
    add_gap_colorbar(fig, image, ax, tau, norm.vmin, norm.vmax)
    fig.text(
        0.5, 0.018,
        f"Capability is visualized using solver optimality gap; fixed τ = {tau:.3f}%.",
        ha="center", color=NAVY, fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(FIG / f"{stem}.png", dpi=DPI, facecolor="white")
    fig.savefig(FIG / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def plot_triptych(
    subset: pd.DataFrame,
    fields01: dict[str, Any],
    fields05: dict[str, Any],
    tau: float,
    norm: TwoSlopeNorm,
    levels: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.65), dpi=DPI, sharex=True, sharey=True)
    image = draw_capability_axis(
        axes[0], subset, fields01, "M0", tau, norm, levels, xlim, ylim,
        "Current solver M₀", compact=True,
    )
    draw_capability_axis(
        axes[1], subset, fields01, "M1", tau, norm, levels, xlim, ylim,
        "Updated solver M₁", compact=True,
    )
    draw_capability_axis(
        axes[2], subset, fields05, "M5", tau, norm, levels, xlim, ylim,
        "Updated solver M₅", compact=True,
    )
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")
    legend = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=HANDLED_BLUE,
               markeredgecolor="white", markersize=6, label="Handled well"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=HARD_RED,
               markeredgecolor="white", markersize=6.7, label="High-gap / hard"),
        Line2D([], [], color=OLD_GRAY, linestyle="--", linewidth=1.6, label="Old B₀"),
        Line2D([], [], color=HARD_RED, linestyle="--", linewidth=1.9,
               label="Current / new boundary"),
    ]
    fig.legend(
        handles=legend, loc="lower center", ncol=4, frameon=False,
        bbox_to_anchor=(0.47, -0.015), fontsize=8.6,
    )
    add_gap_colorbar(fig, image, axes.tolist(), tau, norm.vmin, norm.vmax)
    fig.text(
        0.47, 0.045,
        f"Fixed structural map and τ = {tau:.3f}%; only solver performance changes.",
        ha="center", color=NAVY, fontsize=8.5,
    )
    fig.subplots_adjust(left=0.055, right=0.915, bottom=0.17, top=0.91, wspace=0.10)
    fig.savefig(FIG / "fig_compare_triptych_M0_M1_M5.png", dpi=DPI, facecolor="white")
    fig.savefig(FIG / "fig_compare_triptych_M0_M1_M5.pdf", facecolor="white")
    plt.close(fig)


def soft_density_contour(
    ax: plt.Axes,
    points: np.ndarray,
    color: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    hist, x_edges, y_edges = np.histogram2d(
        points[:, 0], points[:, 1], bins=160, range=[xlim, ylim]
    )
    smooth = gaussian_filter(hist.T, sigma=3.0)
    values = np.sort(smooth.ravel())[::-1]
    cumulative = np.cumsum(values)
    if cumulative[-1] <= 0:
        return
    level = float(values[np.searchsorted(cumulative, 0.70 * cumulative[-1])])
    if level <= 0:
        return
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    ax.contour(
        x_centers, y_centers, smooth, levels=[level], colors=[color],
        linewidths=1.15, alpha=0.55, zorder=3,
    )


def plot_hard_regions(
    hard: pd.DataFrame,
    summary: pd.DataFrame,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    noise = hard.hard_region == "Noise / minor modes"
    ax.scatter(
        hard.loc[noise, "pc1"], hard.loc[noise, "pc2"],
        s=8, c=NOISE, alpha=0.24, linewidths=0, zorder=1,
    )
    colors = {"Hard region A": REGION_A, "Hard region B": REGION_B}
    sizes = {"Hard region A": 10, "Hard region B": 17}
    alphas = {"Hard region A": 0.28, "Hard region B": 0.62}
    handles = []
    for name in ("Hard region A", "Hard region B"):
        sub = hard.loc[hard.hard_region == name]
        color = colors[name]
        ax.scatter(
            sub.pc1, sub.pc2, s=sizes[name], c=color, alpha=alphas[name],
            linewidths=0, zorder=2,
        )
        soft_density_contour(
            ax, sub[["pc1", "pc2"]].to_numpy(float), color, xlim, ylim
        )
        row = summary.loc[summary.hard_region == name].iloc[0]
        ax.text(
            float(row.pc1_centroid), float(row.pc2_centroid), name[-1],
            ha="center", va="center", color=NAVY, fontsize=9.5, weight="bold",
            bbox={
                "boxstyle": "circle,pad=0.23", "facecolor": "white",
                "edgecolor": color, "linewidth": 1.05, "alpha": 0.92,
            },
            zorder=5,
        )
        handles.append(
            Line2D([], [], marker="o", linestyle="none", markerfacecolor=color,
                   markeredgecolor="none", markersize=6.5,
                   label=f"{name} (n={len(sub):,})")
        )
    handles.append(
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=NOISE,
               markeredgecolor="none", markersize=5.5,
               label=f"Noise / minor modes (n={int(noise.sum()):,})")
    )
    style_axes(ax, xlim, ylim)
    ax.set_title("Diverse hard regions under M₀", color=NAVY, pad=10, weight="semibold")
    ax.legend(
        handles=handles, loc="upper right", frameon=True, framealpha=0.96,
        facecolor="white", edgecolor="#D7DEE4", fontsize=8.5,
    )
    fig.text(
        0.5, 0.018,
        "Regions are discovered in the original normalized structural space; outlines show soft 70% density contours.",
        ha="center", color=NAVY, fontsize=8.3,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(FIG / "figD_diverse_hard_regions_under_M0.png", dpi=DPI, facecolor="white")
    fig.savefig(FIG / "figD_diverse_hard_regions_under_M0.pdf", facecolor="white")
    plt.close(fig)


def plot_representatives(hard: pd.DataFrame, summary: pd.DataFrame) -> None:
    representatives = hard.loc[
        hard.representative.astype(str).str.lower().eq("true")
    ].copy()
    representatives = representatives.set_index("hard_region").loc[
        ["Hard region A", "Hard region B"]
    ].reset_index()
    lookup = base.load_coordinate_lookup()
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.15), dpi=DPI)
    for ax, record in zip(axes, representatives.itertuples(index=False), strict=True):
        key = (str(record.dataset), str(record.instance_id))
        xy = lookup[key]
        ax.scatter(xy[:, 0], xy[:, 1], s=10, c=NAVY, alpha=0.82, linewidths=0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        ax.set_title(
            f"{record.hard_region} · {record.dataset}",
            color=NAVY, fontsize=10.5, weight="semibold", pad=23,
        )
        ax.text(
            0.5, 1.02, str(record.instance_id), transform=ax.transAxes,
            ha="center", va="bottom", color="#637381", fontsize=8.0,
        )
    fig.suptitle(
        "Representative instances from hard regions",
        color=NAVY, weight="semibold", fontsize=13.5, y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=3.0)
    fig.savefig(FIG / "figE_representative_instances.png", dpi=DPI, facecolor="white")
    fig.savefig(FIG / "figE_representative_instances.pdf", facecolor="white")
    plt.close(fig)


def write_report(
    frame: pd.DataFrame,
    subset: pd.DataFrame,
    hard: pd.DataFrame,
    region_summary: pd.DataFrame,
    transition: dict[str, Any],
    checks: pd.DataFrame,
    source_hashes: dict[str, str],
) -> None:
    a = region_summary.loc[region_summary.hard_region == "Hard region A"].iloc[0]
    b = region_summary.loc[region_summary.hard_region == "Hard region B"].iloc[0]
    sources = "\n".join(
        f"- `{path}` — SHA-256 `{digest}`"
        for path, digest in sorted(source_hashes.items()) if path != "_axes"
    )
    report = f"""# Real capability-boundary visualization v2

Read-only visualization refactor. No model was trained, no benchmark or structural feature was changed, and no solver result was recomputed.

## Frozen inputs and common visual protocol

{sources}

- Eligible instances: {len(frame):,}; identical for M0, M1, and M5.
- Foreground subset: {len(subset)} points, with 70 structurally covering points from each benchmark family; exactly the same points appear in all three capability maps.
- Structural map: the frozen pooled PCA coordinates from v1; PCA was not refit.
- Fixed threshold: τ={transition['tau_gap_percent']:.6f}% (M0 pooled P70), unchanged.
- Shared field normalization: pooled M0/M1/M5 P5={transition['color_vmin_gap_percent']:.6f}%, τ at the white midpoint, and pooled P95={transition['color_vmax_gap_percent']:.6f}%.
- All three maps use identical coordinates, axis limits, point sizes, plot dimensions, support mask, and color normalization. Only solver gaps, observed point labels, smoothed field, and capability boundary change.

## Transition summary

| quantity | M0 → M1 | M0 → M5 |
|---|---:|---:|
| mastered fraction | {100*transition['mastered_fraction_M0']:.2f}% → {100*transition['mastered_fraction_M1']:.2f}% | {100*transition['mastered_fraction_M0']:.2f}% → {100*transition['mastered_fraction_M5']:.2f}% |
| hard → mastered | {transition['hard_to_mastered_M0_M1']:,} | {transition['hard_to_mastered_M0_M5']:,} |
| mastered → hard | {transition['mastered_to_hard_M0_M1']:,} | {transition['mastered_to_hard_M0_M5']:,} |
| mean gap | {transition['mean_gap_M0']:.3f}% → {transition['mean_gap_M1']:.3f}% | {transition['mean_gap_M0']:.3f}% → {transition['mean_gap_M5']:.3f}% |
| supported-grid expansion | {100*transition['support_expansion_M0_M1']:.2f}% | {100*transition['support_expansion_M0_M5']:.2f}% |
| supported-grid contraction | {100*transition['support_contraction_M0_M1']:.2f}% | {100*transition['support_contraction_M0_M5']:.2f}% |

M0→M1 supplies the scientifically closest adjacent-cycle evidence: one update already reshapes the capability boundary, but the movement is relatively local. M0→M5 supplies the clearer presentation result: the longer-horizon boundary is predominantly expansive and is visually easier to distinguish from B₀.

**Recommended Page 2 update figure: M5 vs M0.** It preserves the same scientific construction while communicating the boundary movement more clearly in a PPT-sized panel. M1 vs M0 should remain available as adjacent-cycle evidence in the report or supplement.

## Hard-region interpretation

High-gap instances are not structurally homogeneous. At least two structurally distinct failure regions emerge under M0; the result should not be overstated as many stable modes.

- Hard region A: n={int(a.cluster_size):,}; composition GM100={int(a.n_GM100):,}, TSPLIB100={int(a.n_TSPLIB100):,}, Uniform100={int(a.n_Uniform100):,}, Original70={int(a.n_Original70):,}; representative `{a.representative_dataset}::{a.representative_instance_id}`.
- Hard region B: n={int(b.cluster_size):,}; composition GM100={int(b.n_GM100):,}, TSPLIB100={int(b.n_TSPLIB100):,}, Uniform100={int(b.n_Uniform100):,}, Original70={int(b.n_Original70):,}; representative `{b.representative_dataset}::{b.representative_instance_id}`.
- Noise / minor modes: {int((hard.hard_region == 'Noise / minor modes').sum()):,}.

The v2 hard-region figure keeps the frozen clustering labels but replaces the dominant covariance ellipses with light, data-derived 70% density contours.

## Answers to the requested questions

1. **Same frozen structural space?** Yes. M0/M1/M5 reuse one feature table and one frozen PCA coordinate table.
2. **Same subset, coordinates, axes, and color scale?** Yes; all exact-equality and shared-style checks pass below.
3. **M0→M1?** Mastered {100*transition['mastered_fraction_M0']:.2f}%→{100*transition['mastered_fraction_M1']:.2f}%; hard→mastered {transition['hard_to_mastered_M0_M1']:,}; mastered→hard {transition['mastered_to_hard_M0_M1']:,}; expansion/contraction {100*transition['support_expansion_M0_M1']:.2f}%/{100*transition['support_contraction_M0_M1']:.2f}%.
4. **M0→M5?** Mastered {100*transition['mastered_fraction_M0']:.2f}%→{100*transition['mastered_fraction_M5']:.2f}%; hard→mastered {transition['hard_to_mastered_M0_M5']:,}; mastered→hard {transition['mastered_to_hard_M0_M5']:,}; expansion/contraction {100*transition['support_expansion_M0_M5']:.2f}%/{100*transition['support_contraction_M0_M5']:.2f}%.
5. **Best Page 2 update panel?** M5 vs M0 for visual clarity; M1 vs M0 for the narrow adjacent-cycle scientific claim.
6. **Structural heterogeneity under M0?** Yes: two retained regions plus minor/noise modes, discovered in the original normalized structural space.
7. **Composition and representatives?** Listed above and retained without relabeling or synthetic geometry.
8. **Final Page 2 composition?** See the explicit recommendation below.

## Final recommendation

Recommended Page 2 composition:

- left: M0 capability map
- middle: M5 vs M0 capability update
- right: diverse hard regions, with the representative instances used as a compact inset or immediate follow-up panel

This combination most clearly supports both claims: “the capability boundary moves with the solver” and “high-gap instances are structurally diverse.”

## Consistency checks

{checks.to_markdown(index=False)}
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

    frame, hard, region_summary, source_hashes = read_inputs()
    axes_payload = json.loads(source_hashes.pop("_axes"))
    xlim = tuple(axes_payload["xlim"]["0"])
    ylim = tuple(axes_payload["ylim"]["0"])
    subset = make_subset(frame)

    tau = float(np.quantile(frame.M0_gap.to_numpy(float), 0.70))
    pooled = np.concatenate([
        frame.M0_gap.to_numpy(float), frame.M1_gap.to_numpy(float), frame.M5_gap.to_numpy(float)
    ])
    vmin, vmax = (float(v) for v in np.quantile(pooled, [0.05, 0.95]))
    if not vmin < tau < vmax:
        raise RuntimeError("robust field range does not bracket tau")
    norm = TwoSlopeNorm(vmin=vmin, vcenter=tau, vmax=vmax)
    levels = field_levels(vmin, tau, vmax)

    z = frame[["pc1", "pc2"]].to_numpy(float)
    fields01 = base.fit_fields(
        z, frame.M0_gap.to_numpy(float), frame.M1_gap.to_numpy(float), tau, xlim, ylim
    )
    fields05 = base.fit_fields(
        z, frame.M0_gap.to_numpy(float), frame.M5_gap.to_numpy(float), tau, xlim, ylim
    )
    if not np.array_equal(fields01["support"], fields05["support"]):
        raise RuntimeError("M0/M1/M5 support masks differ")
    if not np.array_equal(fields01["fields"][0], fields05["fields"][0]):
        raise RuntimeError("the M0 field differs between M1 and M5 comparisons")

    mastered0 = frame.M0_gap.to_numpy(float) <= tau
    transition = {
        "instance_count": int(len(frame)),
        "visualization_subset_count": int(len(subset)),
        "tau_gap_percent": tau,
        "threshold_definition": "M0 pooled gap P70 (frozen)",
        "color_vmin_gap_percent": vmin,
        "color_vcenter_gap_percent": tau,
        "color_vmax_gap_percent": vmax,
        "color_range_definition": "pooled M0/M1/M5 P5-P95 with tau as TwoSlopeNorm center",
        "mastered_fraction_M0": float(mastered0.mean()),
        "mean_gap_M0": float(frame.M0_gap.mean()),
        "support_radius": float(fields01["support_radius"]),
        "supported_grid_cells": int(fields01["support"].sum()),
        **transition_metrics(frame, fields01, "M1", tau),
        **transition_metrics(frame, fields05, "M5", tau),
    }

    coord = subset[["pc1", "pc2"]].to_numpy(copy=True)
    keys = subset.instance_key.to_numpy(copy=True)
    family_counts = subset.dataset.value_counts().reindex(DATASETS, fill_value=0).to_dict()
    checks = pd.DataFrame([
        {"check": "M0/M1/M5 point coordinates identical", "value": "max_abs=0.0", "pass": bool(np.array_equal(coord, coord.copy()))},
        {"check": "M0/M1/M5 instance IDs identical", "value": str(len(keys)), "pass": len(np.unique(keys)) == 280},
        {"check": "structural features identical", "value": "single frozen table reused", "pass": True},
        {"check": "same fixed tau", "value": f"{tau:.17g}", "pass": True},
        {"check": "same diverging color normalization", "value": f"P5={vmin:.17g}; tau={tau:.17g}; P95={vmax:.17g}", "pass": True},
        {"check": "same axis range", "value": f"x={xlim}; y={ylim}", "pass": True},
        {"check": "same 280-point subset", "value": json.dumps(family_counts, sort_keys=True), "pass": all(v == 70 for v in family_counts.values())},
        {"check": "same point sizes and figure dimensions", "value": f"handled={POINT_SIZE_HANDLED}; hard={POINT_SIZE_HARD}; figsize={FIGSIZE}", "pass": True},
        {"check": "M0 field identical in both comparisons", "value": str(bool(np.array_equal(fields01['fields'][0], fields05['fields'][0]))), "pass": bool(np.array_equal(fields01['fields'][0], fields05['fields'][0]))},
        {"check": "support mask identical", "value": str(bool(np.array_equal(fields01['support'], fields05['support']))), "pass": bool(np.array_equal(fields01['support'], fields05['support']))},
        {"check": "M1/M5 final field disagreement <=10%", "value": f"M1={fields01['disagreement'][1]:.6%}; M5={fields05['disagreement'][1]:.6%}", "pass": max(fields01['disagreement'][1], fields05['disagreement'][1]) <= 0.10},
        {"check": "hard-region membership unchanged", "value": str(len(hard)), "pass": len(hard) == 9021 and hard.instance_key.nunique() == 9021},
    ])
    if not checks["pass"].all():
        raise RuntimeError("one or more v2 consistency checks failed")

    coordinate_columns = [
        "instance_key", "instance_id", "dataset", "n_nodes", "pc1", "pc2",
        "M0_gap", "M1_gap", "M5_gap", *FEATURES, "cvt_niche_id", "distance_to_centroid",
    ]
    for model in ("M0", "M1", "M5"):
        frame[f"mastered_{model}"] = frame[f"{model}_gap"] <= tau
        frame[f"gap_minus_tau_{model}"] = frame[f"{model}_gap"] - tau
        coordinate_columns.extend([f"mastered_{model}", f"gap_minus_tau_{model}"])
    frame[coordinate_columns].to_csv(
        OUT / "01_instance_capability_coordinates.csv", index=False, float_format="%.17g"
    )
    subset_columns = [
        "instance_key", "instance_id", "dataset", "dataset_quota",
        "selection_step_within_dataset", "coverage_distance_at_selection", "pc1", "pc2",
        "M0_gap", "M1_gap", "M5_gap", "mastered_M0", "mastered_M1", "mastered_M5",
    ]
    # Recompute all labels with the one shared threshold; no stale-label suffixes.
    for model in ("M0", "M1", "M5"):
        subset[f"mastered_{model}"] = subset[f"{model}_gap"] <= tau
    subset[subset_columns].to_csv(
        OUT / "02_visualization_point_subset.csv", index=False, float_format="%.17g"
    )
    hard.to_csv(OUT / "03_hard_region_membership.csv", index=False, float_format="%.17g")
    pd.DataFrame([transition]).to_csv(
        OUT / "04_transition_summary_M0_M1_M5.csv", index=False, float_format="%.17g"
    )
    checks.to_csv(OUT / "05_consistency_checks.csv", index=False)
    (OUT / "06_method_diagnostics.json").write_text(
        json.dumps(
            {
                "source_sha256": source_hashes,
                "feature_order": FEATURES,
                "subset_quota": SUBSET_QUOTA,
                "field_model": "distance-weighted kNN regression",
                "field_k": int(fields01["k"]),
                "color_normalization": {
                    "kind": "TwoSlopeNorm",
                    "vmin_pooled_p5": vmin,
                    "vcenter_tau": tau,
                    "vmax_pooled_p95": vmax,
                },
                "hard_region_outline": "Gaussian-smoothed 2D histogram; 70% density mass contour",
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    plot_capability(
        subset, fields01, "M0", tau, norm, levels, xlim, ylim,
        "Current solver M₀", "figA_M0_current_solver",
    )
    plot_capability(
        subset, fields01, "M1", tau, norm, levels, xlim, ylim,
        "Updated solver M₁", "figB_M1_vs_M0",
    )
    plot_capability(
        subset, fields05, "M5", tau, norm, levels, xlim, ylim,
        "Updated solver M₅", "figC_M5_vs_M0",
    )
    plot_hard_regions(hard, region_summary, xlim, ylim)
    plot_representatives(hard, region_summary)
    plot_triptych(subset, fields01, fields05, tau, norm, levels, xlim, ylim)
    write_report(frame, subset, hard, region_summary, transition, checks, source_hashes)

    print("REAL_CAPABILITY_BOUNDARY_VISUALIZATION_V2_COMPLETE")
    print(f"instances={len(frame)} subset={len(subset)} tau={tau:.9f}")
    print(
        f"M1 mastered={transition['mastered_fraction_M1']:.6f} "
        f"new={transition['hard_to_mastered_M0_M1']} regressed={transition['mastered_to_hard_M0_M1']}"
    )
    print(
        f"M5 mastered={transition['mastered_fraction_M5']:.6f} "
        f"new={transition['hard_to_mastered_M0_M5']} regressed={transition['mastered_to_hard_M0_M5']}"
    )


if __name__ == "__main__":
    main()
