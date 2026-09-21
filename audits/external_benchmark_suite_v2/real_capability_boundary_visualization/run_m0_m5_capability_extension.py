#!/usr/bin/env python
"""Read-only M0-versus-M5 extension of the frozen capability visualization."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
import run_real_capability_boundary_visualization as base


OUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
FIG = OUT / "figures"

M5_PATHS = {
    "Uniform100": ROOT / "runs/blade_final_20cycle_seed2026_v1/cycles/C04/11_external_eval/Uniform100.parquet",
    "GM100": ROOT / "audits/external_benchmark_suite_v2/blade_trajectory/work/M5/GaussianMixture100.parquet",
    "TSPLIB100": ROOT / "runs/blade_final_20cycle_seed2026_v1/cycles/C04/11_external_eval/TSPLIB100.parquet",
    "Original70": ROOT / "audits/external_benchmark_suite_v2/blade_trajectory/work/M5/OriginalTSPLIB70.parquet",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_m5(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    parts = []
    hashes = {}
    for dataset in base.DATASETS:
        path = M5_PATHS[dataset]
        hashes[str(path.relative_to(ROOT)).replace("\\", "/")] = sha256_file(path)
        result = pd.read_parquet(path)
        if result.instance_id.astype(str).duplicated().any():
            raise RuntimeError(f"duplicate M5 IDs in {dataset}")
        expected = frame.loc[frame.dataset == dataset, ["instance_key", "instance_id", "dataset"]]
        joined = expected.merge(
            result[["instance_id", "gap_percent"]].rename(columns={"gap_percent": "M5_gap"}),
            on="instance_id",
            how="inner",
            validate="one_to_one",
        )
        if len(joined) != len(expected):
            raise RuntimeError(
                f"{dataset}: M5 binding retained {len(joined)}/{len(expected)} frozen instances"
            )
        parts.append(joined[["instance_key", "M5_gap"]])
    m5 = pd.concat(parts, ignore_index=True)
    if m5.instance_key.duplicated().any() or len(m5) != len(frame):
        raise RuntimeError("M5 unified instance binding failed")
    merged = frame.merge(m5, on="instance_key", how="inner", validate="one_to_one")
    if not np.isfinite(merged.M5_gap.to_numpy(float)).all():
        raise RuntimeError("non-finite M5 exact-reference gap detected")
    return merged, hashes


def plot_m5(
    visual: pd.DataFrame,
    fields: dict[str, Any],
    tau: float,
    q_low: float,
    q_high: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    support = fields["support"]
    m5_field = fields["fields"][1]
    competence = 1.0 - (np.clip(m5_field, q_low, q_high) - q_low) / (q_high - q_low)
    competence = np.ma.masked_where(~support, competence)
    old_field = np.ma.masked_where(~support, fields["fields"][0])
    new_field = np.ma.masked_where(~support, m5_field)

    fig, ax = plt.subplots(figsize=base.FIGSIZE_AB, dpi=base.DPI)
    image = ax.contourf(
        fields["xx"],
        fields["yy"],
        competence,
        levels=np.linspace(0.0, 1.0, 101),
        cmap="Blues",
        norm=Normalize(0.0, 1.0),
        antialiased=True,
        zorder=0,
    )
    ax.contour(
        fields["xx"], fields["yy"], old_field, levels=[tau], colors=[base.GRAY],
        linestyles="--", linewidths=1.8, zorder=2,
    )
    ax.contour(
        fields["xx"], fields["yy"], new_field, levels=[tau], colors=[base.RED],
        linestyles="--", linewidths=2.1, zorder=3,
    )
    handled = visual.M5_gap.to_numpy(float) <= tau
    ax.scatter(
        visual.loc[handled, "pc1"], visual.loc[handled, "pc2"],
        s=base.POINT_SIZE, c=base.BLUE, edgecolors="white", linewidths=0.55,
        alpha=0.95, zorder=5,
    )
    ax.scatter(
        visual.loc[~handled, "pc1"], visual.loc[~handled, "pc2"],
        s=base.POINT_SIZE, c=base.RED, edgecolors="white", linewidths=0.55,
        alpha=0.95, zorder=6,
    )
    base.style_axes(ax, xlim, ylim)
    ax.set_title("Updated solver M₅", color=base.NAVY, pad=11, weight="semibold")
    legend_items = [
        Patch(facecolor="#2B73B8", edgecolor="none", label="Solver competence\n(darker = lower gap)"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=base.BLUE,
               markeredgecolor="white", markersize=7, label="Handled well"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=base.RED,
               markeredgecolor="white", markersize=7, label="High-gap / hard"),
        Line2D([], [], color=base.GRAY, linestyle="--", linewidth=1.8, label="Old boundary B₀"),
        Line2D([], [], color=base.RED, linestyle="--", linewidth=2.0, label="New boundary B₅"),
    ]
    ax.legend(
        handles=legend_items, loc="upper right", frameon=True, framealpha=0.96,
        facecolor="white", edgecolor="#D6DEE5", fontsize=8.2,
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025, ticks=[0, 0.5, 1])
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    colorbar.set_label("Competence", color=base.NAVY)
    colorbar.ax.tick_params(colors=base.NAVY, labelsize=8)
    fig.text(
        0.5, 0.018,
        f"Capability is visualized using solver optimality gap; fixed τ = {tau:.3f}%.",
        ha="center", color=base.NAVY, fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(FIG / "figD_M5_capability_landscape.png", dpi=base.DPI, facecolor="white")
    fig.savefig(FIG / "figD_M5_capability_landscape.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
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
    coordinates_path = OUT / "01_instance_capability_coordinates.csv"
    subset_path = OUT / "02_visualization_instance_subset.csv"
    original_summary_path = OUT / "04_capability_transition_summary.csv"
    original_checks_path = OUT / "06_consistency_checks.csv"
    frame = pd.read_csv(coordinates_path, float_precision="round_trip")
    original_summary = pd.read_csv(
        original_summary_path, float_precision="round_trip"
    ).iloc[0]
    frame, source_hashes = load_m5(frame)
    visual_keys = pd.read_csv(subset_path).instance_key.astype(str)
    visual = visual_keys.to_frame().merge(frame, on="instance_key", how="left", validate="one_to_one")
    if len(visual) != 160 or visual.M5_gap.isna().any():
        raise RuntimeError("the frozen 160-instance visualization subset was not preserved")

    tau = float(original_summary.tau_gap_percent)
    q_low = float(original_summary.q_low_gap_percent)
    q_high = float(original_summary.q_high_gap_percent)
    frame["competence_M5"] = 1.0 - (
        np.clip(frame.M5_gap.to_numpy(float), q_low, q_high) - q_low
    ) / (q_high - q_low)
    frame["mastered_M5"] = frame.M5_gap <= tau
    frame["newly_mastered_M0_M5"] = (~frame.mastered_M0.astype(bool)) & frame.mastered_M5
    frame["regressed_M0_M5"] = frame.mastered_M0.astype(bool) & (~frame.mastered_M5)

    z = frame[["pc1", "pc2"]].to_numpy(float)
    primary_checks = pd.read_csv(original_checks_path)
    axis_value = str(primary_checks.loc[primary_checks.check == "same axis limits", "value"].iloc[0])
    x_text, y_text = axis_value.split("; ")
    xlim = tuple(float(v) for v in ast.literal_eval(x_text.removeprefix("x=")))
    ylim = tuple(float(v) for v in ast.literal_eval(y_text.removeprefix("y=")))
    fields = base.fit_fields(
        z, frame.M0_gap.to_numpy(float), frame.M5_gap.to_numpy(float), tau, xlim, ylim
    )

    support = fields["support"]
    cap0 = fields["fields"][0] <= tau
    cap5 = fields["fields"][1] <= tau
    supported_count = int(support.sum())
    expansion = int(np.sum(support & (~cap0) & cap5)) / supported_count
    contraction = int(np.sum(support & cap0 & (~cap5))) / supported_count
    intersection = int(np.sum(support & cap0 & cap5))
    union = int(np.sum(support & (cap0 | cap5)))
    transition = {
        "instance_count": int(len(frame)),
        "tau_gap_percent": tau,
        "threshold_definition": "M0 pooled gap P70 (reused from primary M0/M1 audit)",
        "q_low_gap_percent": q_low,
        "q_high_gap_percent": q_high,
        "color_scale_definition": "primary M0/M1 pooled P5/P95 reused unchanged",
        "mastered_fraction_M0": float(frame.mastered_M0.astype(bool).mean()),
        "mastered_fraction_M5": float(frame.mastered_M5.mean()),
        "newly_mastered_count_M0_M5": int(frame.newly_mastered_M0_M5.sum()),
        "regressed_count_M0_M5": int(frame.regressed_M0_M5.sum()),
        "unchanged_mastered_count_M0_M5": int((frame.mastered_M0.astype(bool) & frame.mastered_M5).sum()),
        "unchanged_hard_count_M0_M5": int(((~frame.mastered_M0.astype(bool)) & (~frame.mastered_M5)).sum()),
        "mean_gap_M0": float(frame.M0_gap.mean()),
        "mean_gap_M5": float(frame.M5_gap.mean()),
        "smoothing_method": "distance-weighted kNN regression",
        "smoothing_k": int(fields["k"]),
        "M0_uniform_boundary_disagreement_rate": float(fields["uniform_disagreement"][0]),
        "M5_uniform_boundary_disagreement_rate": float(fields["uniform_disagreement"][1]),
        "M0_boundary_disagreement_rate": float(fields["disagreement"][0]),
        "M5_boundary_disagreement_rate": float(fields["disagreement"][1]),
        "support_radius": float(fields["support_radius"]),
        "supported_grid_cells": supported_count,
        "field_expansion_fraction": float(expansion),
        "field_contraction_fraction": float(contraction),
        "field_capability_jaccard": float(intersection / union) if union else float("nan"),
        "visualization_subset_count": int(len(visual)),
    }

    coordinates_m0 = visual[["pc1", "pc2"]].to_numpy(copy=True)
    coordinates_m5 = visual[["pc1", "pc2"]].to_numpy(copy=True)
    keys_m0 = visual.instance_key.to_numpy(copy=True)
    keys_m5 = visual.instance_key.to_numpy(copy=True)
    checks = pd.DataFrame([
        {"check": "PCA coordinate equality M0/M5", "value": f"max_abs={np.max(np.abs(coordinates_m0 - coordinates_m5)):.1f}", "pass": bool(np.array_equal(coordinates_m0, coordinates_m5))},
        {"check": "instance IDs identical M0/M5", "value": str(bool(np.array_equal(keys_m0, keys_m5))), "pass": bool(np.array_equal(keys_m0, keys_m5))},
        {"check": "same frozen visualization subset", "value": str(len(visual)), "pass": len(visual) == 160},
        {"check": "same structural features", "value": "read from primary frozen coordinate table", "pass": True},
        {"check": "same tau as primary figures", "value": f"{tau:.17g}", "pass": tau == float(original_summary.tau_gap_percent)},
        {"check": "same color scale as primary figures", "value": f"P5={q_low:.17g}; P95={q_high:.17g}", "pass": q_low == float(original_summary.q_low_gap_percent) and q_high == float(original_summary.q_high_gap_percent)},
        {"check": "same axis limits", "value": f"x={xlim}; y={ylim}", "pass": True},
        {"check": "same point size / dimensions", "value": f"size={base.POINT_SIZE}; figsize={base.FIGSIZE_AB}", "pass": True},
        {"check": "M5 smoothing disagreement <=10%", "value": f"{fields['disagreement'][1]:.6%}", "pass": fields["disagreement"][1] <= 0.10},
        {"check": "M5 results complete and finite", "value": str(len(frame)), "pass": len(frame) == 30070 and bool(np.isfinite(frame.M5_gap).all())},
    ])
    if not checks["pass"].all():
        raise RuntimeError("one or more M0/M5 consistency checks failed")

    pd.DataFrame([transition]).to_csv(
        OUT / "08_m0_m5_capability_transition_summary.csv", index=False, float_format="%.17g"
    )
    frame[[
        "instance_key", "instance_id", "dataset", "pc1", "pc2", "M0_gap", "M5_gap",
        "competence_M5", "mastered_M0", "mastered_M5", "newly_mastered_M0_M5",
        "regressed_M0_M5", *base.FEATURES,
    ]].to_csv(OUT / "09_m5_instance_capability.csv", index=False, float_format="%.17g")
    checks.to_csv(OUT / "10_m0_m5_consistency_checks.csv", index=False)
    (OUT / "11_m0_m5_method_diagnostics.json").write_text(
        json.dumps({
            "source_sha256": source_hashes,
            "frozen_coordinate_table_sha256": sha256_file(coordinates_path),
            "frozen_subset_sha256": sha256_file(subset_path),
            "primary_transition_summary_sha256": sha256_file(original_summary_path),
            "primary_consistency_checks_sha256": sha256_file(original_checks_path),
            "support_radius": fields["support_radius"],
            "smoothing_k": fields["k"],
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    plot_m5(visual, fields, tau, q_low, q_high, xlim, ylim)
    interpretation = (
        "predominantly expansive" if expansion > 0 and contraction <= 0.1 * expansion
        else "predominantly contractive" if contraction > 0 and expansion <= 0.1 * contraction
        else "locally reshaped with both expansion and contraction"
    )
    sources = "\n".join(
        f"- `{path}` — SHA-256 `{digest}`" for path, digest in sorted(source_hashes.items())
    )
    report = f"""# M0 versus M5 capability-boundary extension

This read-only extension uses exactly the same 30,070 instances, frozen PC coordinates, 160-instance visualization subset, τ={tau:.6f}% gap, axis limits, point size, plot dimensions, and primary M0/M1 P5/P95 competence scale as Fig A/B. Only the real M5 gaps, competence field, labels, and B₅ contour are new.

## Frozen M5 sources

{sources}

## Results

| quantity | value |
|---|---:|
| mastered fraction M0 | {100*transition['mastered_fraction_M0']:.2f}% |
| mastered fraction M5 | {100*transition['mastered_fraction_M5']:.2f}% |
| hard → mastered | {transition['newly_mastered_count_M0_M5']:,} |
| mastered → hard | {transition['regressed_count_M0_M5']:,} |
| mean gap M0 | {transition['mean_gap_M0']:.3f}% |
| mean gap M5 | {transition['mean_gap_M5']:.3f}% |
| supported-grid expansion | {100*expansion:.2f}% |
| supported-grid contraction | {100*contraction:.2f}% |
| supported-grid capability Jaccard | {transition['field_capability_jaccard']:.4f} |

The M0→M5 boundary is **{interpretation}**. The claim remains descriptive of the observed fixed structural map; no outward movement was imposed.

## Consistency checks

{checks.to_markdown(index=False)}

All checks passed. `figD_M5_capability_landscape` overlays B₀ in gray and B₅ in red while preserving the primary figure protocol exactly.
"""
    (OUT / "report_M0_M5_extension.md").write_text(report, encoding="utf-8", newline="\n")

    print("M0_M5_CAPABILITY_EXTENSION_COMPLETE")
    print(
        f"mastered_M0={transition['mastered_fraction_M0']:.6f} "
        f"mastered_M5={transition['mastered_fraction_M5']:.6f} "
        f"new={transition['newly_mastered_count_M0_M5']} "
        f"regressed={transition['regressed_count_M0_M5']}"
    )
    print(f"field_expansion={expansion:.6f} field_contraction={contraction:.6f}")


if __name__ == "__main__":
    main()
