#!/usr/bin/env python3
"""Build the frozen three-seed, per-cycle, per-dataset analysis artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_SEEDS = (2026, 1107, 2203)
EXPECTED_CYCLES = tuple(range(6))
EXPECTED_METHODS = (
    "A0_FULL_BLADE",
    "A1_WITHOUT_QD",
    "A2_WITHOUT_SURROGATE",
    "A3_RANDOM_FOGA",
    "A4_EVOREAL50K",
)
EXPECTED_BENCHMARKS = (
    "Uniform100",
    "GaussianMixture100",
    "TSPLIB100",
    "EvoReal_S1_100",
    "EvoReal_S2_100",
    "EvoReal_S3_100",
    "GaussianMixture500",
    "OriginalTSPLIB70",
)

COLORS = {
    "A0_FULL_BLADE": "#1f77b4",
    "A1_WITHOUT_QD": "#ff7f0e",
    "A2_WITHOUT_SURROGATE": "#2ca02c",
    "A3_RANDOM_FOGA": "#d62728",
    "A4_EVOREAL50K": "#9467bd",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    run_root = root / "runs" / "BLADE_FULL_PIPELINE_MULTISEED_V1"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=run_root / "analysis" / "all_raw_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=run_root / "analysis" / "three_seed_cycle_dataset_analysis",
    )
    return parser.parse_args()


def validate(frame: pd.DataFrame) -> None:
    required = {
        "seed",
        "method",
        "cycle",
        "benchmark",
        "n_instances",
        "mean_gap_percent",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"input is missing columns: {sorted(missing)}")

    if tuple(sorted(frame["seed"].unique())) != tuple(sorted(EXPECTED_SEEDS)):
        raise RuntimeError(f"unexpected seeds: {sorted(frame['seed'].unique())}")
    if set(frame["method"].unique()) != set(EXPECTED_METHODS):
        raise RuntimeError(f"unexpected methods: {sorted(frame['method'].unique())}")
    if tuple(sorted(frame["cycle"].unique())) != EXPECTED_CYCLES:
        raise RuntimeError(f"unexpected cycles: {sorted(frame['cycle'].unique())}")
    if set(frame["benchmark"].unique()) != set(EXPECTED_BENCHMARKS):
        raise RuntimeError(f"unexpected benchmarks: {sorted(frame['benchmark'].unique())}")

    keys = ["seed", "method", "cycle", "benchmark"]
    if frame.duplicated(keys).any():
        raise RuntimeError("duplicate seed/method/cycle/benchmark rows found")
    expected_rows = (
        len(EXPECTED_SEEDS)
        * len(EXPECTED_METHODS)
        * len(EXPECTED_CYCLES)
        * len(EXPECTED_BENCHMARKS)
    )
    if len(frame) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} rows, found {len(frame)}")
    if frame["mean_gap_percent"].isna().any():
        raise RuntimeError("mean_gap_percent contains missing values")
    seed_counts = frame.groupby(["method", "cycle", "benchmark"])["seed"].nunique()
    if not seed_counts.eq(len(EXPECTED_SEEDS)).all():
        raise RuntimeError("not every method/cycle/benchmark cell has three seeds")


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def make_summary(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["benchmark", "method", "cycle"], sort=False)
    summary = grouped.agg(
        seed_count=("seed", "nunique"),
        n_instances_per_seed=("n_instances", "first"),
        mean_gap_percent=("mean_gap_percent", "mean"),
        std_gap_percent=("mean_gap_percent", "std"),
        min_gap_percent=("mean_gap_percent", "min"),
        max_gap_percent=("mean_gap_percent", "max"),
    ).reset_index()
    summary["sem_gap_percent"] = summary["std_gap_percent"] / np.sqrt(summary["seed_count"])
    summary["model"] = "M" + summary["cycle"].astype(str)
    benchmark_rank = {name: index for index, name in enumerate(EXPECTED_BENCHMARKS)}
    method_rank = {name: index for index, name in enumerate(EXPECTED_METHODS)}
    summary["_benchmark_rank"] = summary["benchmark"].map(benchmark_rank)
    summary["_method_rank"] = summary["method"].map(method_rank)
    summary = summary.sort_values(["_benchmark_rank", "_method_rank", "cycle"])
    return summary.drop(columns=["_benchmark_rank", "_method_rank"])[
        [
            "benchmark",
            "method",
            "cycle",
            "model",
            "seed_count",
            "n_instances_per_seed",
            "mean_gap_percent",
            "std_gap_percent",
            "sem_gap_percent",
            "min_gap_percent",
            "max_gap_percent",
        ]
    ]


def draw_one(ax: plt.Axes, summary: pd.DataFrame, benchmark: str, legend: bool) -> None:
    local = summary[summary["benchmark"].eq(benchmark)]
    for method in EXPECTED_METHODS:
        line = local[local["method"].eq(method)].sort_values("cycle")
        x = line["cycle"].to_numpy(dtype=float)
        mean = line["mean_gap_percent"].to_numpy(dtype=float)
        std = line["std_gap_percent"].to_numpy(dtype=float)
        color = COLORS[method]
        ax.plot(x, mean, marker="o", linewidth=2.0, markersize=4.5, color=color, label=method)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
    ax.set_title(benchmark)
    ax.set_xticks(EXPECTED_CYCLES, [f"M{i}" for i in EXPECTED_CYCLES])
    ax.set_xlabel("Cycle / model")
    ax.set_ylabel("Mean optimality gap (%) - lower is better")
    ax.grid(True, alpha=0.25)
    if legend:
        ax.legend(fontsize=8, frameon=False)


def make_figures(summary: pd.DataFrame, figure_dir: Path) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for index, benchmark in enumerate(EXPECTED_BENCHMARKS, start=1):
        fig, ax = plt.subplots(figsize=(9.5, 6.0))
        draw_one(ax, summary, benchmark, legend=True)
        fig.suptitle("Three-seed mean across M0-M5 (shading: +/- 1 SD)", fontsize=12)
        fig.tight_layout()
        path = figure_dir / f"{index:02d}_{slug(benchmark)}_three_seed_mean.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs.append(path)

    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    for ax, benchmark in zip(axes.flat, EXPECTED_BENCHMARKS):
        draw_one(ax, summary, benchmark, legend=False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Three-seed mean performance by dataset and cycle (shading: +/- 1 SD)", fontsize=16)
    fig.tight_layout(rect=(0, 0.045, 1, 0.975))
    combined = figure_dir / "00_all_datasets_three_seed_mean.png"
    fig.savefig(combined, dpi=180, bbox_inches="tight")
    plt.close(fig)
    outputs.insert(0, combined)
    return outputs


def main() -> None:
    args = parse_args()
    source = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(source)
    frame = frame[frame["seed"].isin(EXPECTED_SEEDS)].copy()
    validate(frame)
    frame["model"] = "M" + frame["cycle"].astype(str)
    frame = frame.sort_values(["benchmark", "method", "cycle", "seed"])

    long_path = output_dir / "01_three_seed_per_cycle_dataset_results.csv"
    leading = ["seed", "method", "cycle", "model", "benchmark", "n_instances", "mean_gap_percent"]
    remaining = [column for column in frame.columns if column not in leading]
    frame[leading + remaining].to_csv(long_path, index=False, float_format="%.10f")

    summary = make_summary(frame)
    mean_path = output_dir / "02_three_seed_mean_by_method_cycle_dataset.csv"
    summary.to_csv(mean_path, index=False, float_format="%.10f")

    wide = frame.pivot(
        index=["benchmark", "method", "cycle", "model"],
        columns="seed",
        values="mean_gap_percent",
    ).reset_index()
    wide = wide.rename(columns={seed: f"seed_{seed}_gap_percent" for seed in EXPECTED_SEEDS})
    seed_columns = [f"seed_{seed}_gap_percent" for seed in EXPECTED_SEEDS]
    wide["three_seed_mean_gap_percent"] = wide[seed_columns].mean(axis=1)
    wide["three_seed_std_gap_percent"] = wide[seed_columns].std(axis=1, ddof=1)
    wide_path = output_dir / "03_three_seed_comparison_wide.csv"
    wide.to_csv(wide_path, index=False, float_format="%.10f")

    figures = make_figures(summary, output_dir / "figures")
    manifest = {
        "source": str(source),
        "seeds": list(EXPECTED_SEEDS),
        "methods": list(EXPECTED_METHODS),
        "cycles": list(EXPECTED_CYCLES),
        "benchmarks": list(EXPECTED_BENCHMARKS),
        "raw_rows": len(frame),
        "mean_rows": len(summary),
        "mean_definition": "unweighted arithmetic mean of per-seed mean_gap_percent",
        "std_definition": "sample standard deviation across the three seeds (ddof=1)",
        "files": [str(long_path), str(mean_path), str(wide_path), *map(str, figures)],
    }
    manifest_path = output_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"long_csv={long_path}")
    print(f"mean_csv={mean_path}")
    print(f"wide_csv={wide_path}")
    print(f"combined_figure={figures[0]}")
    print(f"individual_figures={len(figures) - 1}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
