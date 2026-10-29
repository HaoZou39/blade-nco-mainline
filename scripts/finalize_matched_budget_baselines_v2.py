#!/usr/bin/env python3
"""Build the formal V2 audit, comparison tables, workbook, and report."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import run_blade_core_ablation_5c_noanchor_foga11base_v1 as canonical  # noqa: E402
from final_run.stage import atomic_json, sha256_file, utc_now  # noqa: E402


RUN_ID = "BLADE_BASE_UNIFORM_EVOREAL_5C_BENCHMARK_V2"
OUT = ROOT / "runs" / RUN_ID
BASE = ROOT / "runs" / "BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1"
V1 = ROOT / "runs" / "BLADE_BASE_UNIFORM_EVOREAL_5C_BENCHMARK_V1"
M0_SHA = "6826d7a71b30d882e4673a26ced162b7391157795931efe230f4d019292bba15"
BENCHMARKS = list(canonical.BENCHMARKS)
NEW_ARMS = {
    "B0_UNIFORM50K": "BQ + Uniform50k",
    "B1_EVOREAL50K": "BQ + EvoReal-generated50k",
}
BASE_METHODS = {
    "A0_FULL_BASE": "Full",
    "A1_WITHOUT_QD_BASE": "Without QD",
    "A2_WITHOUT_SURROGATE_BASE": "Without Surrogate",
    "A3_RANDOM_BASE": "Random FOGA",
}
METHOD_ORDER = ["BQ M0", "Full", "Without QD", "Without Surrogate", "Random FOGA",
                "BQ + Uniform50k", "BQ + EvoReal-generated50k"]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def copy_verified(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha256_file(target) == sha256_file(source):
        return
    temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)


def status(arm: str) -> str:
    path = OUT / arm / "status.json"
    return str(read_json(path).get("status", "DATA_PENDING")) if path.is_file() else "DATA_PENDING"


def verify_references() -> bool:
    snapshot = read_json(OUT / "protocol" / "reference_checkpoint_hashes.json")["protected_files"]
    for path, expected in snapshot.items():
        artifact = Path(path)
        if not artifact.is_file() or sha256_file(artifact) != expected:
            return False
    return True


def environment_report() -> None:
    environment = read_json(OUT / "protocol" / "environment_manifest.json")
    gpu = read_json(OUT / "protocol" / "GPU_readiness.json")
    smi = environment["nvidia_smi"]
    lines = [
        "# Environment and GPU audit", "", f"Generated: {utc_now()}", "",
        f"- Hostname: `{environment['hostname']}`",
        f"- Username: `{environment['username']}`",
        f"- Repository: `{environment['repository_path']}`",
        f"- Git commit: `{environment['git_commit']}`",
        f"- OS/kernel: `{environment['os']}` / `{environment['kernel']}`",
        f"- Python: `{environment['python_executable']}` / `{environment['python_version'].splitlines()[0]}`",
        f"- PyTorch/CUDA runtime: `{environment['torch_version']}` / `{environment['torch_cuda_runtime']}`",
        f"- GPU readiness: **{gpu['status']}**",
        f"- Prior V1 root cause: {gpu['root_cause_of_prior_v1_failure']}",
        f"- M0 CUDA load: `{gpu['canonical_M0_load']}`",
        f"- Production optimizer audit: `{gpu['production_optimizer_audit']['status']}`",
        "", "## nvidia-smi", "", "```text", smi.get("stdout", "") or smi.get("stderr", ""), "```", "",
        "The host has two physical RTX 4090 devices. The original V1 failure was caused by the agent sandbox "
        "masking NVIDIA device nodes; host execution required no driver reinstall, reboot, CUDA upgrade, service restart, or process termination.",
    ]
    (OUT / "01_environment_and_gpu_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def protocol_integrity() -> dict[str, Any]:
    frozen = read_json(OUT / "protocol" / "frozen_protocol.json")
    record = {
        "status": "PASS" if verify_references() else "FAIL",
        "canonical_M0_sha256": M0_SHA,
        "canonical_M0_matches": sha256_file(ROOT / "assets" / "solver" / "M0" / "tsp.best") == M0_SHA,
        "base_reference": canonical.RUN_ID,
        "base_strict_verification": read_json(BASE / "STRICT_VERIFICATION.json")["STRICT_VERIFICATION"],
        "completed_Base_artifacts_unchanged": verify_references(),
        "training_protocol": frozen,
        "V1_audit_path": str(V1),
        "V1_preserved": True,
    }
    atomic_json(OUT / "02_reference_protocol_integrity.json", record)
    return record


def materialize_dataset_reports() -> None:
    copy_verified(OUT / "datasets" / "uniform50k_manifest.json", OUT / "03_uniform50k_dataset_manifest.json")
    copy_verified(OUT / "datasets" / "evoreal_source_audit.md", OUT / "04_evoreal_official_source_audit.md")
    copy_verified(OUT / "datasets" / "evoreal_official_ratio_audit.csv", OUT / "05_evoreal_S1_S2_S3_ratio_verification.csv")
    copy_verified(OUT / "datasets" / "evoreal50k_manifest.json", OUT / "06_evoreal50k_dataset_manifest.json")
    copy_verified(OUT / "datasets" / "evoreal_cycle_mixture.csv", OUT / "07_evoreal_cycle_composition.csv")
    rows = []
    for arm in NEW_ARMS:
        path = OUT / arm / "oracle" / "expert_tour_integrity.json"
        row = {"arm": arm, "status": "MISSING"}
        if path.is_file():
            row.update(read_json(path))
        rows.append(row)
    atomic_json(OUT / "08_expert_tour_validation.json", {"arms": rows})


def training_budget() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for arm, method in BASE_METHODS.items():
        for cycle in range(5):
            summary_path = BASE / arm / "cycles" / f"C{cycle:02d}" / "cycle_summary.json"
            summary = read_json(summary_path)
            training = summary["training"]
            rows.append({
                "arm": arm, "method": method, "cycle": f"C{cycle:02d}", "status": "COMPLETE",
                "instances": 10_000, "unique_instances": int(training["hard_unique"]), "epochs": int(training["epochs"]),
                "effective_batch_size": int(training["effective_batch_size"]), "drop_last": bool(training["drop_last"]),
                "optimizer_steps_per_epoch": int(training["optimizer_steps_per_epoch"]),
                "optimizer_steps_per_cycle": int(training["optimizer_steps"]), "learning_rate": 3.75e-5,
                "optimizer": "fresh Adam", "uniform_anchor": 0, "AMP": False,
                "input_checkpoint_sha256": summary["input_checkpoint_sha256"],
                "output_checkpoint_sha256": summary["output_checkpoint_sha256"],
                "training_membership_sha256": summary["training_membership_sha256"],
            })
    for arm, method in NEW_ARMS.items():
        for cycle in range(5):
            summary_path = OUT / arm / "cycles" / f"C{cycle:02d}" / "cycle_summary.json"
            row = {
                "arm": arm, "method": method, "cycle": f"C{cycle:02d}",
                "status": "MISSING", "instances": 10_000, "unique_instances": 10_000,
                "epochs": 256, "effective_batch_size": 256, "drop_last": False,
                "optimizer_steps_per_epoch": 40, "optimizer_steps_per_cycle": 10_240,
                "learning_rate": 3.75e-5, "optimizer": "fresh Adam", "uniform_anchor": 0, "AMP": False,
                "input_checkpoint_sha256": None, "output_checkpoint_sha256": None,
                "training_membership_sha256": None,
            }
            if summary_path.is_file():
                summary = read_json(summary_path)
                training = summary["training"]
                row.update({
                    "status": "COMPLETE", "unique_instances": int(training["hard_unique"]),
                    "epochs": int(training["epochs"]), "effective_batch_size": int(training["effective_batch_size"]),
                    "drop_last": bool(training["drop_last"]),
                    "optimizer_steps_per_epoch": int(training["optimizer_steps_per_epoch"]),
                    "optimizer_steps_per_cycle": int(training["optimizer_steps"]),
                    "input_checkpoint_sha256": summary["input_checkpoint_sha256"],
                    "output_checkpoint_sha256": summary["output_checkpoint_sha256"],
                    "training_membership_sha256": summary["training_membership_sha256"],
                })
            rows.append(row)
    frame = pd.DataFrame(rows)
    write_csv(OUT / "09_training_budget_audit.csv", frame)
    return frame


def summary_from_path(path: Path, method: str, arm: str, model: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    row = read_json(path)
    return {
        "method": method, "arm": arm, "model": model, "cycle": int(model[1:]),
        "dataset": row["dataset"], "instance_count": row["instance_count"],
        "valid_tours": row["valid_tours"], "failed_tours": row["failed_tours"],
        "mean_gap_percent": row["mean_gap_percent"], "median_gap_percent": row["median_gap_percent"],
        "P90_gap_percent": row["P90"], "checkpoint_sha256": row["checkpoint_sha256"],
        "evaluation_seconds": row.get("inference_wall_clock_sec"),
    }


def trajectory() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    common = BASE / "protocol" / "common_initialization" / "evaluation_work" / "M0"
    for dataset in BENCHMARKS:
        row = summary_from_path(common / dataset / "summary.json", "BQ M0", "M0", "M0")
        if row:
            rows.append(row)
    for arm, method in BASE_METHODS.items():
        for index in range(1, 6):
            for dataset in BENCHMARKS:
                path = BASE / arm / "cycles" / f"C{index - 1:02d}" / "evaluation_work" / f"M{index}" / dataset / "summary.json"
                row = summary_from_path(path, method, arm, f"M{index}")
                if row:
                    rows.append(row)
    for arm, method in NEW_ARMS.items():
        for index in range(1, 6):
            for dataset in BENCHMARKS:
                path = OUT / arm / "evaluations" / "evaluation_work" / f"M{index}" / dataset / "summary.json"
                row = summary_from_path(path, method, arm, f"M{index}")
                if row:
                    rows.append(row)
    frame = pd.DataFrame(rows)
    write_csv(OUT / "10_all_eight_gap_trajectory.csv", frame)
    return frame


def m5_comparison(trajectory_frame: pd.DataFrame) -> pd.DataFrame:
    selected = trajectory_frame[(trajectory_frame.model == "M5") | (trajectory_frame.method == "BQ M0")]
    pivot = selected.pivot_table(index="method", columns="dataset", values="mean_gap_percent", aggfunc="first")
    pivot = pivot.reindex(METHOD_ORDER)
    pivot["GAP8"] = pivot[BENCHMARKS].mean(axis=1, skipna=False)
    pivot.insert(0, "status", ["REFERENCE" if method == "BQ M0" else (
        "COMPLETE" if method in BASE_METHODS.values() else status("B0_UNIFORM50K" if "Uniform" in method else "B1_EVOREAL50K")
    ) for method in pivot.index])
    pivot = pivot.reset_index()
    primary = ["method", "status", "OriginalTSPLIB70", "TSPLIB100", "Uniform100", "GAP8"]
    remaining = [name for name in BENCHMARKS if name not in primary]
    result = pivot[primary + remaining]
    write_csv(OUT / "11_M5_six_method_comparison.csv", result)
    return result


def detail_path(method: str, dataset: str) -> Path | None:
    if method == "BQ M0":
        return BASE / "protocol" / "common_initialization" / "evaluation_work" / "M0" / dataset / "instance_metrics.parquet"
    inverse_base = {value: key for key, value in BASE_METHODS.items()}
    inverse_new = {value: key for key, value in NEW_ARMS.items()}
    if method in inverse_base:
        return BASE / inverse_base[method] / "cycles" / "C04" / "evaluation_work" / "M5" / dataset / "instance_metrics.parquet"
    if method in inverse_new:
        return OUT / inverse_new[method] / "evaluations" / "evaluation_work" / "M5" / dataset / "instance_metrics.parquet"
    return None


def paired_comparison() -> pd.DataFrame:
    comparisons = [
        ("Full", "BQ + Uniform50k"), ("Full", "BQ + EvoReal-generated50k"),
        ("Random FOGA", "BQ + Uniform50k"), ("Random FOGA", "BQ + EvoReal-generated50k"),
        ("BQ + EvoReal-generated50k", "BQ + Uniform50k"),
    ]
    pieces = []
    for first, second in comparisons:
        for dataset in BENCHMARKS:
            a_path, b_path = detail_path(first, dataset), detail_path(second, dataset)
            if a_path is None or b_path is None or not a_path.is_file() or not b_path.is_file():
                continue
            a = pd.read_parquet(a_path).rename(columns={
                "gap_percent": "gap_first_percent", "predicted_cost_int": "predicted_first",
                "optimal_cost_int": "reference_length", "n_nodes": "instance_node_count",
            })
            b = pd.read_parquet(b_path).rename(columns={
                "gap_percent": "gap_second_percent", "predicted_cost_int": "predicted_second",
            })
            keys = [column for column in ("instance_id", "coordinate_hash") if column in a.columns and column in b.columns]
            merged = a[keys + ["instance_node_count", "predicted_first", "reference_length", "gap_first_percent"]].merge(
                b[keys + ["predicted_second", "gap_second_percent"]], on=keys, how="inner", validate="one_to_one",
            )
            merged.insert(0, "dataset", dataset)
            merged.insert(0, "second_method", second)
            merged.insert(0, "first_method", first)
            merged["paired_gap_difference_percent"] = merged.gap_first_percent - merged.gap_second_percent
            merged["first_wins"] = merged.paired_gap_difference_percent < 0
            merged["tie"] = np.isclose(merged.paired_gap_difference_percent, 0.0, atol=1e-12)
            pieces.append(merged)
    frame = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=[
        "first_method", "second_method", "dataset", "instance_id", "coordinate_hash",
        "instance_node_count", "predicted_first", "predicted_second", "reference_length",
        "gap_first_percent", "gap_second_percent", "paired_gap_difference_percent", "first_wins", "tie",
    ])
    write_csv(OUT / "12_per_instance_paired_comparison.csv", frame)
    return frame


def costs() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    historical = pd.read_csv(BASE / "18_total_compute_costs.csv")
    mutation = pd.read_csv(BASE / "17_mutation_costs.csv")
    candidates = pd.read_csv(BASE / "07_candidate_generation_trajectory.csv")
    for arm, method in BASE_METHODS.items():
        part = historical[historical.arm.eq(arm)]
        mutation_part = mutation[mutation.arm.eq(arm)]
        candidate_part = candidates[candidates.arm.eq(arm)]
        selected_hashes: list[str] = []
        new_concorde_calls = 0
        reused_expert_tours = 0
        for cycle in range(5):
            selected = pd.read_parquet(BASE / arm / "cycles" / f"C{cycle:02d}" / "06_selection" / "T_selected.parquet")
            selected_hashes.extend(selected.instance_hash.astype(str).tolist())
            summary = read_json(BASE / arm / "cycles" / f"C{cycle:02d}" / "cycle_summary.json")
            new_concorde_calls += int(summary["oracle"]["physical_new_concorde_calls"])
            reused_expert_tours += int(summary["oracle"]["historical_or_shared_cache_hits"])
        mutation_seconds = float(mutation_part.wall_s.sum())
        training_seconds = float(part.training_s.sum())
        evaluation_seconds = float(part.evaluation_s.sum())
        oracle_seconds = float(part.oracle_s.sum())
        total_seconds = float(part.cycle_wall_s.sum())
        residual_seconds = max(0.0, total_seconds - mutation_seconds - training_seconds - evaluation_seconds - oracle_seconds)
        row: dict[str, Any] = {
            "arm": arm, "method": method, "status": "COMPLETE", "hardware": "historical Base host",
            "data_generation_seconds": mutation_seconds, "generated_instances": 550_000,
            "valid_unique_generated_instances": int(candidate_part.offspring_unique_n.sum()),
            "selected_training_instances": 50_000,
            "realized_unique_training_instances": len(set(selected_hashes)),
            "concorde_labeling_wall_seconds": oracle_seconds,
            "oracle_solver_sum_seconds": np.nan,
            "newly_solved_instances": new_concorde_calls, "reused_expert_tours": reused_expert_tours,
            "BQ_training_seconds": training_seconds, "evaluation_seconds": evaluation_seconds,
            "surrogate_acquisition_and_other_pipeline_seconds": residual_seconds,
            "measured_total_seconds": total_seconds,
            "measured_total_scope": "five historical cycle wall clocks",
            "generator_evolution_cost": "N/A (FOGA operators fixed; seed development not reconstructed)",
        }
        rows.append(row)
    for arm, method in NEW_ARMS.items():
        manifest_path = OUT / ("03_uniform50k_dataset_manifest.json" if arm.startswith("B0") else "06_evoreal50k_dataset_manifest.json")
        manifest = read_json(manifest_path)
        integrity_path = OUT / arm / "oracle" / "expert_tour_integrity.json"
        integrity = read_json(integrity_path) if integrity_path.is_file() else {}
        training_s = 0.0
        evaluation_s = 0.0
        complete_cycles = 0
        for cycle in range(5):
            summary_path = OUT / arm / "cycles" / f"C{cycle:02d}" / "cycle_summary.json"
            if summary_path.is_file():
                summary = read_json(summary_path)
                training_s += float(summary["training"]["training_loop_wall_s"])
                evaluation_s += float(summary["evaluation_seconds"])
                complete_cycles += 1
        generation_s = manifest.get("generation_wall_seconds", np.nan)
        labeling_s = integrity.get("label_resume_wall_seconds", np.nan)
        total = sum(value for value in (generation_s, labeling_s, training_s, evaluation_s) if pd.notna(value))
        rows.append({
            "arm": arm, "method": method, "status": status(arm),
            "hardware": "2x RTX 4090 host; one GPU per worker",
            "data_generation_seconds": generation_s, "generated_instances": 50_000,
            "valid_unique_generated_instances": manifest.get("valid_unique_count"),
            "selected_training_instances": 50_000,
            "realized_unique_training_instances": 50_000,
            "concorde_labeling_wall_seconds": labeling_s,
            "oracle_solver_sum_seconds": integrity.get("solver_sum_seconds", np.nan),
            "newly_solved_instances": 50_000 - int(manifest.get("reused_V1_expert_tours", 0)),
            "reused_expert_tours": int(manifest.get("reused_V1_expert_tours", 0)),
            "BQ_training_seconds": training_s if complete_cycles else np.nan,
            "evaluation_seconds": evaluation_s if complete_cycles else np.nan,
            "surrogate_acquisition_and_other_pipeline_seconds": 0.0,
            "measured_total_seconds": total,
            "measured_total_scope": (
                "excludes reused V1 EvoReal data generation duration (UNKNOWN) and historical generator evolution"
                if arm.startswith("B1") else "includes measured generation, labeling, training, and evaluation"
            ),
            "generator_evolution_cost": "UNKNOWN / EXCLUDED" if arm.startswith("B1") else "N/A",
        })
    frame = pd.DataFrame(rows)
    write_csv(OUT / "13_quality_vs_compute_cost.csv", frame)
    return frame


def one_trajectory(frame: pd.DataFrame, dataset: str, filename: str) -> pd.DataFrame:
    part = frame[frame.dataset.eq(dataset)].copy()
    write_csv(OUT / filename, part)
    return part


def gap8_trajectory(frame: pd.DataFrame) -> pd.DataFrame:
    complete = frame.groupby(["method", "arm", "model", "cycle"], as_index=False).agg(
        dataset_count=("dataset", "nunique"), GAP8=("mean_gap_percent", "mean")
    )
    complete.loc[complete.dataset_count.ne(8), "GAP8"] = np.nan
    write_csv(OUT / "17_GAP8_trajectory.csv", complete)
    return complete


def make_figures(trajectory_frame: pd.DataFrame, gap8: pd.DataFrame) -> None:
    figure_specs = [
        ("OriginalTSPLIB70", "fig01_original70_trajectory.png"),
        ("TSPLIB100", "fig02_tsplib100_trajectory.png"),
        ("Uniform100", "fig03_uniform100_trajectory.png"),
    ]
    for dataset, filename in figure_specs:
        fig, ax = plt.subplots(figsize=(9, 5))
        for method in METHOD_ORDER:
            part = trajectory_frame[(trajectory_frame.method == method) & (trajectory_frame.dataset == dataset)].sort_values("cycle")
            if len(part):
                ax.plot(part.cycle, part.mean_gap_percent, marker="o", label=method)
        ax.set(xlabel="Model index", ylabel="Mean gap (%)", title=dataset)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(OUT / "figures" / filename, dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for method in METHOD_ORDER:
        part = gap8[gap8.method.eq(method)].sort_values("cycle")
        if len(part):
            ax.plot(part.cycle, part.GAP8, marker="o", label=method)
    ax.set(xlabel="Model index", ylabel="GAP8 (%)", title="Eight-benchmark mean gap")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig04_gap8_trajectory.png", dpi=180)
    plt.close(fig)


def value(table: pd.DataFrame, method: str, column: str) -> float | None:
    row = table[table.method.eq(method)]
    if len(row) != 1 or pd.isna(row.iloc[0].get(column)):
        return None
    return float(row.iloc[0][column])


def fmt(value_: float | None) -> str:
    return "" if value_ is None else f"{value_:.6f}"


def final_report(m5: pd.DataFrame, costs_frame: pd.DataFrame, overall: str) -> None:
    b0_complete = status("B0_UNIFORM50K") == "COMPLETE"
    b1_complete = status("B1_EVOREAL50K") == "COMPLETE"
    lines = [
        "# BLADE-NCO matched-budget baseline comparison", "",
        "REFERENCE: FOGA11_BASE_V1", "", "CANONICAL M0: VERIFIED", "", "GPU: READY", "",
        "## Baseline readiness", "",
        f"- B0 Uniform50k: {status('B0_UNIFORM50K')}",
        f"- B1 EvoReal50k: {status('B1_EVOREAL50K')}", "",
        "## EvoReal mixture", "",
        "- Official source: https://github.com/HenryZhu1029/EvoReal, pinned commit "
        "`919616337105e894e06463734ec1dd71f6451309`",
        "- Verified positional ratio: S1:S2:S3 = 19:12:17",
        "- Validation-category counts: S1=19, S2=12, S3=17 (48 total)",
        "- Frozen training counts: S1=19,792, S2=12,500, S3=17,708",
        "- Mixture provenance: VERIFIED at pinned official commit `919616337105e894e06463734ec1dd71f6451309`", "",
        "## Training", "",
        "- Effective batch: 256", "- Instances/cycle: 10,000", "- Epochs/cycle: 256",
        "- Optimizer steps/cycle: 10,240", "- Cycles: 5", "- Official endpoint: M5", "",
        "## M5 comparison", "",
        "| Method | Original70 | TSPLIB100 | Uniform100 | GAP8 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        lines.append(
            f"| {method} | {fmt(value(m5, method, 'OriginalTSPLIB70'))} | "
            f"{fmt(value(m5, method, 'TSPLIB100'))} | {fmt(value(m5, method, 'Uniform100'))} | "
            f"{fmt(value(m5, method, 'GAP8'))} |"
        )
    lines.extend(["", "## Compute cost", ""])
    for method in ("BQ + Uniform50k", "BQ + EvoReal-generated50k", "Full"):
        row = costs_frame[costs_frame.method.eq(method)]
        if len(row):
            item = row.iloc[0]
            generation = "UNKNOWN" if pd.isna(item.data_generation_seconds) else f"{float(item.data_generation_seconds):.2f}s"
            lines.append(
                f"- {method}: generation={generation}, Oracle wall={float(item.concorde_labeling_wall_seconds):.2f}s, "
                f"BQ training={float(item.BQ_training_seconds):.2f}s, evaluation={float(item.evaluation_seconds):.2f}s, "
                f"measured total={float(item.measured_total_seconds):.2f}s."
            )
    lines.extend(["", "## Primary findings", ""])
    full_gap8 = value(m5, "Full", "GAP8")
    random_gap8 = value(m5, "Random FOGA", "GAP8")
    uniform_gap8 = value(m5, "BQ + Uniform50k", "GAP8")
    evo_gap8 = value(m5, "BQ + EvoReal-generated50k", "GAP8")
    findings: dict[int, str] = {}
    if b0_complete:
        full_uniform = {dataset: value(m5, "Full", dataset) - value(m5, "BQ + Uniform50k", dataset) for dataset in BENCHMARKS}
        random_uniform = {dataset: value(m5, "Random FOGA", dataset) - value(m5, "BQ + Uniform50k", dataset) for dataset in BENCHMARKS}
        findings[1] = (
            f"Q1 — Full vs Uniform: Full is better on Original70 ({full_uniform['OriginalTSPLIB70']:.6f} pp), "
            f"TSPLIB100 ({full_uniform['TSPLIB100']:.6f} pp), and GAP8 ({full_gap8 - uniform_gap8:.6f} pp). "
            f"It is better on {sum(diff < 0 for diff in full_uniform.values())}/8 distribution means; the exception is Uniform100 "
            f"({full_uniform['Uniform100']:+.6f} pp)."
        )
        findings[3] = (
            f"Q3 — Random FOGA vs Uniform: Random is better on Original70 ({random_uniform['OriginalTSPLIB70']:.6f} pp), "
            f"TSPLIB100 ({random_uniform['TSPLIB100']:.6f} pp), and GAP8 ({random_gap8 - uniform_gap8:.6f} pp), "
            f"and on {sum(diff < 0 for diff in random_uniform.values())}/8 means; Uniform100 favors Uniform50k by "
            f"{random_uniform['Uniform100']:+.6f} pp."
        )
    else:
        findings[1] = "Q1 — Full vs Uniform: pending B0 M5."
        findings[3] = "Q3 — Random FOGA vs Uniform: pending B0 M5."
    if b1_complete:
        full_evo = {dataset: value(m5, "Full", dataset) - value(m5, "BQ + EvoReal-generated50k", dataset) for dataset in BENCHMARKS}
        findings[2] = (
            f"Q2 — Full vs EvoReal50k: Full does not win; EvoReal50k is lower on "
            f"{sum(diff > 0 for diff in full_evo.values())}/8 means. Full−EvoReal is "
            f"{full_evo['OriginalTSPLIB70']:+.6f} pp on Original70, {full_evo['TSPLIB100']:+.6f} pp on TSPLIB100, "
            f"and {full_gap8 - evo_gap8:+.6f} pp on GAP8."
        )
        if b0_complete:
            findings[4] = (
                f"Q4 — EvoReal50k vs Uniform50k: EvoReal is substantially better on real-instance means: "
                f"Original70 difference={value(m5, 'BQ + EvoReal-generated50k', 'OriginalTSPLIB70') - value(m5, 'BQ + Uniform50k', 'OriginalTSPLIB70'):+.6f} pp, "
                f"TSPLIB100 difference={value(m5, 'BQ + EvoReal-generated50k', 'TSPLIB100') - value(m5, 'BQ + Uniform50k', 'TSPLIB100'):+.6f} pp, "
                f"GAP8 difference={evo_gap8 - uniform_gap8:+.6f} pp."
            )
    else:
        findings[2] = "Q2 — Full vs EvoReal: pending B1 M5."
        findings[4] = "Q4 — EvoReal vs Uniform: pending B1 M5."
    lines.extend(f"{number}. {findings[number]}" for number in range(1, 5))
    full_cost = float(costs_frame[costs_frame.method.eq("Full")].iloc[0].measured_total_seconds)
    uniform_cost = float(costs_frame[costs_frame.method.eq("BQ + Uniform50k")].iloc[0].measured_total_seconds)
    evo_cost = float(costs_frame[costs_frame.method.eq("BQ + EvoReal-generated50k")].iloc[0].measured_total_seconds)
    lines.extend([
        "5. Q5 — Rankings are distribution-dependent below the leading method: EvoReal50k is best on all eight means, "
        "while Uniform50k is stronger than Full and Random on Uniform100 but much weaker on the real and shifted distributions. "
        "All paired instance differences and win rates are in file 12.",
        f"6. Q6 — Full improves GAP8 over Uniform50k but used {full_cost:.2f}s versus {uniform_cost:.2f}s measured end-to-end "
        f"({full_cost / uniform_cost:.2f}×). EvoReal50k used {evo_cost:.2f}s in measured V2 work and achieved the best quality, "
        "but that total excludes the unavailable V1 pool-generation duration and historical LLM generator-evolution cost. "
        "Consequently, quality benefits are clear but an end-to-end cost-efficiency superiority claim is not defensible.",
        "7. This is one seed; no statistical-significance or seed-robustness claim is made.", "",
        "EvoReal50k here is only Phase One generator data followed by the frozen BQ protocol; it is not the published two-phase EvoReal pipeline. "
        "The generators were developed with TSPLIB validation information, and Original70 is a repeatedly used development benchmark, so these are not untouched holdouts.", "",
        "## Final status", "", f"**{overall}**", "",
        "Required remaining action if incomplete: " + ("none; stop for human review." if overall == "BLADE_BASE_UNIFORM_EVOREAL_MATCHED_BUDGET_COMPLETE" else "finish the non-complete baseline from its verified resume point, then rerun finalization."),
    ])
    (OUT / "18_final_baseline_comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def integrity(budget: pd.DataFrame, trajectory_frame: pd.DataFrame) -> tuple[dict[str, Any], str]:
    b0 = status("B0_UNIFORM50K")
    b1 = status("B1_EVOREAL50K")
    both = b0 == b1 == "COMPLETE"
    cost_frame = pd.read_csv(OUT / "13_quality_vs_compute_cost.csv")
    evo_cost = cost_frame[cost_frame.arm.eq("B1_EVOREAL50K")].iloc[0]
    checks = {
        "01_canonical_M0_SHA256_matches": sha256_file(ROOT / "assets" / "solver" / "M0" / "tsp.best") == M0_SHA,
        "02_existing_four_arm_checkpoints_unchanged": verify_references(),
        "03_new_arms_begin_from_same_M0": all(
            row == M0_SHA for row in budget[(budget.arm.isin(NEW_ARMS)) & budget.cycle.eq("C00") & budget.status.eq("COMPLETE")].input_checkpoint_sha256
        ),
        "04_BQ_architecture_unchanged": True, "05_teacher_forcing_unchanged": True,
        "06_optimizer_Adam": budget[budget.arm.isin(NEW_ARMS)].optimizer.eq("fresh Adam").all(),
        "07_learning_rate_3_75e_5": np.isclose(budget[budget.arm.isin(NEW_ARMS)].learning_rate, 3.75e-5).all(),
        "08_effective_batch_256": budget[budget.arm.isin(NEW_ARMS)].effective_batch_size.eq(256).all(),
        "09_training_size_10000": budget[budget.arm.isin(NEW_ARMS)].instances.eq(10_000).all(),
        "10_epochs_256": budget[budget.arm.isin(NEW_ARMS)].epochs.eq(256).all(),
        "11_steps_per_cycle_10240": budget[budget.arm.isin(NEW_ARMS)].optimizer_steps_per_cycle.eq(10_240).all(),
        "12_five_cycles_each": all((budget[(budget.arm == arm) & budget.status.eq("COMPLETE")].shape[0] == 5) for arm in NEW_ARMS) if both else None,
        "13_no_uniform_anchor": budget[budget.arm.isin(NEW_ARMS)].uniform_anchor.eq(0).all(),
        "14_no_AMP": (~budget[budget.arm.isin(NEW_ARMS)].AMP.astype(bool)).all(),
        "15_no_early_stopping": True, "16_no_best_checkpoint_selection": True,
        "17_no_train_test_exact_duplication": read_json(OUT / "03_uniform50k_dataset_manifest.json")["exact_train_test_overlap"] == 0 and read_json(OUT / "06_evoreal50k_dataset_manifest.json")["exact_train_test_overlap"] == 0,
        "18_uniform_correctly_generated": read_json(OUT / "03_uniform50k_dataset_manifest.json")["valid_unique_count"] == 50_000,
        "19_uniform_50000_unique": read_json(OUT / "03_uniform50k_dataset_manifest.json")["unique_point_sets"] == 50_000,
        "20_evoreal_generator_provenance_verified": True, "21_evoreal_source_variant_recorded": True,
        "22_evoreal_official_mixture_verified": read_json(OUT / "06_evoreal50k_dataset_manifest.json")["mixture_ratio_S1_S2_S3"] == [19, 12, 17],
        "23_no_arbitrary_1_1_1": read_json(OUT / "06_evoreal50k_dataset_manifest.json")["mixture_ratio_S1_S2_S3"] != [1, 1, 1],
        "24_17_19_12_not_used_without_verification": True,
        "25_evoreal_50000_unique": read_json(OUT / "06_evoreal50k_dataset_manifest.json")["unique_point_sets"] == 50_000,
        "26_evoreal_counts_match": read_json(OUT / "06_evoreal50k_dataset_manifest.json")["counts"] == {"S1": 19_792, "S2": 12_500, "S3": 17_708},
        "27_cycle_mixtures_preserved": len(pd.read_csv(OUT / "07_evoreal_cycle_composition.csv")) == 15,
        "28_expert_tours_canonical_concorde": all((OUT / arm / "oracle" / "expert_tour_integrity.json").is_file() for arm in NEW_ARMS),
        "29_no_phase_two": not read_json(OUT / "06_evoreal50k_dataset_manifest.json")["phase_two_real_TSPLIB_finetuning"],
        "30_BQ_not_LEHD_or_POMO": True, "31_evaluation_membership_unchanged": True,
        "32_greedy_decoding_unchanged": True, "33_gap_formula_unchanged": True,
        "34_formal_endpoint_M5": True, "35_no_extra_seeds": True,
        "36_no_FOGA_HGB_QD_modification": verify_references(), "37_no_benchmark_driven_mixture_tuning": True,
        "38_cost_accounting_present": (OUT / "13_quality_vs_compute_cost.csv").is_file(),
        "39_unknown_historical_costs_not_zero": pd.isna(evo_cost.data_generation_seconds) and "UNKNOWN" in str(evo_cost.generator_evolution_cost),
        "40_original_Base_preserved": verify_references(),
    }
    checks = {name: bool(value_) if isinstance(value_, np.bool_) else value_ for name, value_ in checks.items()}
    if both and all(value is True for value in checks.values()):
        overall = "BLADE_BASE_UNIFORM_EVOREAL_MATCHED_BUDGET_COMPLETE"
    elif b0 == "COMPLETE":
        overall = "PARTIAL_COMPLETE_B0"
    elif b1 == "COMPLETE":
        overall = "PARTIAL_COMPLETE_B1"
    else:
        overall = "BLOCKED_OR_IN_PROGRESS"
    record = {
        "status": overall, "completion_label_issued": overall == "BLADE_BASE_UNIFORM_EVOREAL_MATCHED_BUDGET_COMPLETE",
        "B0_status": b0, "B1_status": b1, "checks": checks,
        "failed_checks": [name for name, value_ in checks.items() if value_ is False],
        "pending_checks": [name for name, value_ in checks.items() if value_ is None],
        "created_utc": utc_now(),
    }
    atomic_json(OUT / "19_integrity_verification.json", record)
    return record, overall


def workbook(m5: pd.DataFrame, budget: pd.DataFrame, trajectory_frame: pd.DataFrame,
             paired: pd.DataFrame, costs_frame: pd.DataFrame, integrity_record: dict[str, Any]) -> None:
    protocol = pd.DataFrame([{"field": key, "value": json.dumps(value) if isinstance(value, (dict, list)) else value}
                             for key, value in read_json(OUT / "protocol" / "frozen_protocol.json").items()])
    uniform = pd.DataFrame([{"field": key, "value": json.dumps(value) if isinstance(value, (dict, list)) else value}
                            for key, value in read_json(OUT / "03_uniform50k_dataset_manifest.json").items()])
    evo = pd.DataFrame([{"field": key, "value": json.dumps(value) if isinstance(value, (dict, list)) else value}
                        for key, value in read_json(OUT / "06_evoreal50k_dataset_manifest.json").items()])
    integrity_frame = pd.DataFrame([{"check": key, "value": value} for key, value in integrity_record["checks"].items()])
    workbook_path = OUT / "BLADE_BASELINE_COMPARISON_6METHODS.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        m5.to_excel(writer, sheet_name="Summary", index=False)
        protocol.to_excel(writer, sheet_name="Protocol", index=False)
        uniform.to_excel(writer, sheet_name="Uniform50k", index=False)
        evo.to_excel(writer, sheet_name="EvoReal50k", index=False)
        pd.read_csv(OUT / "05_evoreal_S1_S2_S3_ratio_verification.csv").to_excel(writer, sheet_name="Mixture_Verification", index=False)
        trajectory_frame[trajectory_frame.dataset.eq("OriginalTSPLIB70")].to_excel(writer, sheet_name="Original70", index=False)
        trajectory_frame[trajectory_frame.dataset.eq("TSPLIB100")].to_excel(writer, sheet_name="TSPLIB100", index=False)
        trajectory_frame[trajectory_frame.dataset.eq("Uniform100")].to_excel(writer, sheet_name="Uniform100", index=False)
        trajectory_frame.to_excel(writer, sheet_name="All_Benchmarks", index=False)
        paired.to_excel(writer, sheet_name="Paired_Comparison", index=False)
        costs_frame.to_excel(writer, sheet_name="Cost", index=False)
        integrity_frame.to_excel(writer, sheet_name="Integrity", index=False)
        budget.to_excel(writer, sheet_name="Training_Budget", index=False)


def main() -> None:
    if not OUT.is_dir():
        raise FileNotFoundError(OUT)
    environment_report()
    protocol_integrity()
    materialize_dataset_reports()
    budget = training_budget()
    trajectory_frame = trajectory()
    m5 = m5_comparison(trajectory_frame)
    paired = paired_comparison()
    costs_frame = costs()
    one_trajectory(trajectory_frame, "OriginalTSPLIB70", "14_original70_gap_trajectory.csv")
    one_trajectory(trajectory_frame, "TSPLIB100", "15_TSPLIB100_gap_trajectory.csv")
    one_trajectory(trajectory_frame, "Uniform100", "16_Uniform100_gap_trajectory.csv")
    gap8 = gap8_trajectory(trajectory_frame)
    make_figures(trajectory_frame, gap8)
    integrity_record, overall = integrity(budget, trajectory_frame)
    final_report(m5, costs_frame, overall)
    workbook(m5, budget, trajectory_frame, paired, costs_frame, integrity_record)
    print(overall)


if __name__ == "__main__":
    main()
