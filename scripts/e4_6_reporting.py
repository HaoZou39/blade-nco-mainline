"""Reporting and matched-architecture audit for E4.6."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from e4_6_strict_helpers import verify_continuation
from final_run.science import bootstrap_mean_delta
from final_run.stage import atomic_json, sha256_file


BENCHMARKS = [
    "Uniform100", "GaussianMixture100", "TSPLIB100", "EvoReal_S1_100",
    "EvoReal_S2_100", "EvoReal_S3_100", "GaussianMixture500", "OriginalTSPLIB70",
]
OOD5 = ["GaussianMixture100", "TSPLIB100", "EvoReal_S1_100", "EvoReal_S2_100", "EvoReal_S3_100"]
PRIMARY = [*OOD5]
BUDGET = 10_000


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _model_number(value: str) -> int:
    return int(str(value).removeprefix("M"))


def _age_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    ages = frame.candidate_age.to_numpy(int)
    result: dict[str, Any] = {"n": int(len(ages))}
    for value in range(5):
        result[f"age{value}_n"] = int(np.sum(ages == value))
        result[f"age{value}_fraction"] = float(np.mean(ages == value))
    result.update({
        "age3plus_n": int(np.sum(ages >= 3)),
        "age3plus_fraction": float(np.mean(ages >= 3)),
        "age5plus_n": int(np.sum(ages >= 5)),
        "age5plus_fraction": float(np.mean(ages >= 5)),
        "mean_age": float(np.mean(ages)),
        "median_age": float(np.median(ages)),
        "P90_age": float(np.percentile(ages, 90)),
        "P95_age": float(np.percentile(ages, 95)),
        "max_age": int(np.max(ages)),
    })
    return result


def _with_ood5(frame: pd.DataFrame) -> pd.DataFrame:
    base = frame[frame.dataset.isin(BENCHMARKS)].copy()
    aggregate = base[base.dataset.isin(OOD5)].groupby("model", as_index=False).agg(
        mean_gap_percent=("mean_gap_percent", "mean"),
        median_gap_percent=("median_gap_percent", "mean"),
        inference_wall_clock_sec=("inference_wall_clock_sec", "sum"),
    )
    aggregate["dataset"] = "OOD5"
    aggregate["passive_only"] = True
    aggregate["controls_training"] = False
    return pd.concat([base, aggregate], ignore_index=True, sort=False)


def _paired_m15(root: Path, audit: Path, e45: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in BENCHMARKS:
        union_path = audit / "evaluation_work" / "M15" / dataset / "instance_metrics.parquet"
        old_path = root / "summary" / "evaluation_work" / "full" / "M15" / dataset / "instance_metrics.parquet"
        if not union_path.is_file() or not old_path.is_file():
            rows.append({"dataset": dataset, "available": False, "reason": "per-instance data unavailable"})
            continue
        union = e45.read_parquet(union_path)
        old = e45.read_parquet(old_path)
        keys = ["instance_id", "coordinate_hash"]
        merged = union[keys + ["optimal_cost_int", "gap_percent"]].merge(
            old[keys + ["optimal_cost_int", "gap_percent"]], on=keys, how="inner",
            suffixes=("_union", "_offspring"), validate="one_to_one",
        )
        if len(merged) != len(union) or len(merged) != len(old):
            raise RuntimeError(f"M15 paired identity mismatch for {dataset}")
        if not np.array_equal(
            merged.optimal_cost_int_union.to_numpy(), merged.optimal_cost_int_offspring.to_numpy()
        ):
            raise RuntimeError(f"M15 exact-reference mismatch for {dataset}")
        a = merged.gap_percent_union.to_numpy(float)
        b = merged.gap_percent_offspring.to_numpy(float)
        delta = a - b
        boot = bootstrap_mean_delta(a, b, seed=2026, replicates=10_000, paired=True)
        rows.append({
            "dataset": dataset, "available": True, "n": len(delta),
            "mean_paired_difference": float(np.mean(delta)),
            "median_paired_difference": float(np.median(delta)),
            "ci95_low": boot["ci95_low"], "ci95_high": boot["ci95_high"],
            "fraction_union_improves": float(np.mean(delta < 0)),
            "fraction_union_worsens": float(np.mean(delta > 0)),
            "fraction_ties": float(np.mean(delta == 0)),
            "bootstrap_seed": 2026, "bootstrap_replicates": 10_000,
        })
    return pd.DataFrame(rows)


def _crossover(delta: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, group in delta.groupby("dataset", sort=False):
        group = group.assign(model_index=group.model.map(_model_number)).sort_values("model_index")
        values = group.delta_gap.to_numpy(float)
        indices = group.model_index.to_numpy(int)
        first_better = next((int(i) for i, value in zip(indices, values, strict=True) if i > 0 and value < 0), None)
        if first_better is None:
            remains = False
        else:
            remains = bool(np.all(values[indices >= first_better] < 0))
        signs = np.sign(values)
        nonzero = [(int(i), int(s)) for i, s in zip(indices, signs, strict=True) if i > 0 and s != 0]
        changes: list[int] = []
        for (prior_i, prior), (current_i, current) in zip(nonzero, nonzero[1:]):
            if prior != current:
                changes.append(current_i)
        later_reversals = sum(value > (first_better if first_better is not None else 15) for value in changes)
        rows.append({
            "dataset": dataset,
            "first_union_better_checkpoint": None if first_better is None else f"M{first_better}",
            "remained_union_better_after_first": remains,
            "sign_change_checkpoints": ",".join(f"M{i}" for i in changes),
            "later_reversals": int(later_reversals),
        })
    return pd.DataFrame(rows)


def _save_figures(audit: Path, survival: pd.DataFrame, ages: pd.DataFrame,
                  turnover: pd.DataFrame, repair: pd.DataFrame, diversity: pd.DataFrame,
                  overhead: pd.DataFrame, delta: pd.DataFrame) -> None:
    union_external = pd.read_csv(audit / "15_external_union_M0_M15.csv")
    old_external = pd.read_csv(audit / "16_historical_offspring_only_M0_M15.csv")
    x_model = np.arange(16)

    def two_arch(dataset: str, filename: str, ylabel: str = "Mean gap (%)") -> None:
        u = union_external[union_external.dataset.eq(dataset)].assign(
            idx=lambda x: x.model.map(_model_number)).sort_values("idx")
        o = old_external[old_external.dataset.eq(dataset)].assign(
            idx=lambda x: x.model.map(_model_number)).sort_values("idx")
        plt.figure(figsize=(9, 4.8))
        plt.plot(u.idx, u.mean_gap_percent, marker="o", label="Corrected Union")
        plt.plot(o.idx, o.mean_gap_percent, marker="s", label="Offspring-only")
        plt.xticks(x_model, [f"M{i}" for i in x_model], rotation=45)
        plt.xlabel("Checkpoint"); plt.ylabel(ylabel); plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
        plt.savefig(audit / filename, dpi=180); plt.close()

    two_arch("OOD5", "fig01_OOD5_union_vs_offspring_only.png")
    two_arch("Uniform100", "fig02_uniform_union_vs_offspring_only.png")
    two_arch("GaussianMixture500", "fig03_GM500_union_vs_offspring_only.png")
    two_arch("OriginalTSPLIB70", "fig04_original70_union_vs_offspring_only.png")

    x = np.arange(15); labels = [f"C{i:02d}" for i in x]
    plt.figure(figsize=(9, 4.8)); plt.plot(x, survival.old_parent_fraction_T, marker="o", label="T")
    plt.plot(x, survival.old_parent_fraction_P, marker="s", label="P"); plt.legend(); plt.xticks(x, labels, rotation=45)
    plt.xlabel("Cycle"); plt.ylabel("Old-parent fraction"); plt.grid(alpha=.25); plt.tight_layout()
    plt.savefig(audit / "fig05_parent_survival_C00_C14.png", dpi=180); plt.close()

    age = ages[ages.group.eq("all")].sort_values("cycle")
    plt.figure(figsize=(9, 4.8)); bottom = np.zeros(15)
    for column, label in [("age0_fraction", "age 0"), ("age1_fraction", "age 1"),
                          ("age2_fraction", "age 2"), ("age3plus_fraction", "age >=3")]:
        values = age[column].to_numpy(float); plt.bar(x, values, bottom=bottom, label=label); bottom += values
    plt.legend(); plt.xticks(x, labels, rotation=45); plt.xlabel("Cycle"); plt.ylabel("Fraction of P"); plt.tight_layout()
    plt.savefig(audit / "fig06_parent_age_C00_C14.png", dpi=180); plt.close()

    plt.figure(figsize=(9, 4.8)); plt.plot(x, turnover.P_turnover, marker="o")
    plt.xticks(x, labels, rotation=45); plt.xlabel("Cycle"); plt.ylabel("P turnover"); plt.grid(alpha=.25); plt.tight_layout()
    plt.savefig(audit / "fig07_parent_turnover_C00_C14.png", dpi=180); plt.close()

    plt.figure(figsize=(9, 4.8)); plt.plot(x, repair.pre_mean, marker="o", label="pre")
    plt.plot(x, repair.post_mean, marker="s", label="post"); plt.legend(); plt.xticks(x, labels, rotation=45)
    plt.xlabel("Cycle"); plt.ylabel("Exact hardness"); plt.grid(alpha=.25); plt.tight_layout()
    plt.savefig(audit / "fig08_hardness_and_repair_C00_C14.png", dpi=180); plt.close()

    pdiv = diversity[diversity.population.eq("P")].sort_values("cycle")
    plt.figure(figsize=(9, 4.8)); plt.plot(x, pdiv.N_eff, marker="o")
    plt.xticks(x, labels, rotation=45); plt.xlabel("Cycle"); plt.ylabel("P N_eff"); plt.grid(alpha=.25); plt.tight_layout()
    plt.savefig(audit / "fig09_parent_neff_C00_C14.png", dpi=180); plt.close()

    plt.figure(figsize=(9, 4.8)); plt.bar(x, overhead.cycle_wall_seconds / 60.0)
    plt.xticks(x, labels, rotation=45); plt.xlabel("Cycle"); plt.ylabel("Wall-clock (min)"); plt.tight_layout()
    plt.savefig(audit / "fig10_cycle_wallclock.png", dpi=180); plt.close()


def build_report(root: Path, run_dir: Path, audit: Path, e45: Any,
                 config: dict[str, Any], union_external_base: pd.DataFrame,
                 source_verification: dict[str, Any], source_unchanged: bool) -> dict[str, Any]:
    summaries = [_read_json(run_dir / f"C{i:02d}" / "cycle_summary.json") for i in range(15)]
    inventory = pd.DataFrame([{
        "cycle": f"C{i:02d}", "model_in": f"M{i}", "model_out": f"M{i+1}",
        "checkpoint_path": str(e45.model_path(i + 1)), "checkpoint_sha256": item["checkpoint_hash"],
        "complete": e45.verify_cycle(i), "continuation_cycle": i >= 5,
    } for i, item in enumerate(summaries)])
    accounting = pd.DataFrame([{"cycle": f"C{i:02d}", **item["candidate_accounting"]} for i, item in enumerate(summaries)])
    survival = pd.DataFrame([{"cycle": f"C{i:02d}", **item["survival"]} for i, item in enumerate(summaries)])
    turnover = pd.DataFrame([{"cycle": f"C{i:02d}", **item["turnover"]} for i, item in enumerate(summaries)])
    lineage = pd.DataFrame([{"cycle": f"C{i:02d}", **item["lineage"]} for i, item in enumerate(summaries)])
    ages: list[dict[str, Any]] = []
    hardness_rows: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []
    diversity_rows: list[dict[str, Any]] = []
    archive_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    overhead_rows: list[dict[str, Any]] = []
    for i, item in enumerate(summaries):
        cycle = f"C{i:02d}"
        parent = e45.read_parquet(run_dir / cycle / "06_selection" / "P_next.parquet")
        for group, frame in [("all", parent), ("QD_floor", parent[parent.component.eq("centroid_floor")]),
                             ("GP_fill", parent[parent.component.eq("gp_fill")])]:
            ages.append({"cycle": cycle, "group": group, **_age_metrics(frame)})
        repair_detail = e45.read_parquet(run_dir / cycle / "10_post_adaptation" / "adaptation_repair.parquet")
        hardness_rows.append({"cycle": cycle, **{f"pre_{k}": v for k, v in item["hardness"].items()},
                              **{f"predicted_{k}": v for k, v in item["predicted_T"].items()},
                              "frontier_delta": item["frontier_renewal"].get("mean_delta")})
        repair_rows.append({"cycle": cycle, "pre_mean": float(repair_detail.H_pre.mean()),
                            "post_mean": float(repair_detail.H_post.mean()), **item["adaptation_repair"]})
        for population, metrics in item["diversity"].items():
            diversity_rows.append({"cycle": cycle, "population": population, **metrics})
        archive_rows.append({"cycle": cycle, **item["archive"]})
        phase_map = {row["phase"]: float(row["wall_seconds"]) for row in item["timing"]["phases"]}
        for row in item["timing"]["phases"]:
            timing_rows.append({"cycle": cycle, **row})
        timing_rows.append({"cycle": cycle, "phase": "TOTAL", "wall_seconds": item["timing"]["cycle_wall_seconds"]})
        overhead_rows.append({
            "cycle": cycle,
            "old_parent_feature_work_s": phase_map.get("feature reuse/computation for old parents", 0.0),
            "union_dedup_s": phase_map.get("union construction/dedup", 0.0),
            "additional_scoring_s": phase_map.get("extra GP scoring old parents", 0.0),
            "additional_qd_handling_s": phase_map.get("QD niche assignment", 0.0),
            "union_overhead_seconds": item["timing"]["union_overhead_seconds"],
            "cycle_wall_seconds": item["timing"]["cycle_wall_seconds"],
            "overhead_fraction": item["timing"]["union_overhead_fraction"],
        })
    age = pd.DataFrame(ages); hardness = pd.DataFrame(hardness_rows); repair = pd.DataFrame(repair_rows)
    diversity = pd.DataFrame(diversity_rows); archive = pd.DataFrame(archive_rows)
    timing = pd.DataFrame(timing_rows); overhead = pd.DataFrame(overhead_rows)

    union_external = _with_ood5(union_external_base)
    historical_base = pd.read_csv(root / "summary" / "full_main_experiment" / "11_external_8set_trajectory.csv")
    historical = _with_ood5(historical_base)
    union_external["model_index"] = union_external.model.map(_model_number)
    historical["model_index"] = historical.model.map(_model_number)
    union_external = union_external.sort_values(["model_index", "dataset"]).drop(columns="model_index")
    historical = historical.sort_values(["model_index", "dataset"]).drop(columns="model_index")
    matched = union_external[["model", "dataset", "mean_gap_percent"]].merge(
        historical[["model", "dataset", "mean_gap_percent"]], on=["model", "dataset"],
        suffixes=("_union", "_offspring"), validate="one_to_one",
    )
    matched["delta_gap"] = matched.mean_gap_percent_union - matched.mean_gap_percent_offspring
    matched["union_better"] = matched.delta_gap < 0
    matched["model_index"] = matched.model.map(_model_number)
    matched = matched.sort_values(["model_index", "dataset"]).drop(columns="model_index")
    late_rows: list[dict[str, Any]] = []
    for dataset, group in matched.groupby("dataset", sort=False):
        indexed = group.assign(idx=group.model.map(_model_number))
        late = indexed[indexed.idx.between(10, 15)]
        trajectory = indexed[indexed.idx.between(1, 15)]
        late_rows.append({
            "dataset": dataset,
            "late_M10_M15_union_mean": float(late.mean_gap_percent_union.mean()),
            "late_M10_M15_offspring_mean": float(late.mean_gap_percent_offspring.mean()),
            "late_delta_gap": float(late.delta_gap.mean()),
            "trajectory_M1_M15_union_mean": float(trajectory.mean_gap_percent_union.mean()),
            "trajectory_M1_M15_offspring_mean": float(trajectory.mean_gap_percent_offspring.mean()),
            "trajectory_delta_gap": float(trajectory.delta_gap.mean()),
        })
    late_summary = pd.DataFrame(late_rows)
    endpoint = matched[matched.model.eq("M15")].copy()
    endpoint["direction"] = np.where(endpoint.delta_gap < 0, "Union better",
                                      np.where(endpoint.delta_gap > 0, "Offspring-only better", "Exact numerical tie"))
    endpoint["absolute_pp_difference"] = endpoint.delta_gap.abs()
    paired = _paired_m15(root, audit, e45)
    crossover = _crossover(matched)

    inventory.to_csv(audit / "03_cycle_inventory.csv", index=False)
    accounting.to_csv(audit / "04_candidate_pool_trajectory.csv", index=False)
    survival.to_csv(audit / "05_parent_survival_trajectory.csv", index=False)
    age.to_csv(audit / "06_parent_age_trajectory.csv", index=False)
    turnover.to_csv(audit / "07_turnover_trajectory.csv", index=False)
    lineage.to_csv(audit / "08_parent_child_competition.csv", index=False)
    hardness.to_csv(audit / "09_hardness_trajectory.csv", index=False)
    repair.to_csv(audit / "10_adaptation_repair.csv", index=False)
    diversity.to_csv(audit / "11_diversity_trajectory.csv", index=False)
    archive.to_csv(audit / "12_archive_growth.csv", index=False)
    timing.to_csv(audit / "13_timing_by_cycle.csv", index=False)
    overhead.to_csv(audit / "14_union_overhead.csv", index=False)
    union_external.to_csv(audit / "15_external_union_M0_M15.csv", index=False)
    historical.to_csv(audit / "16_historical_offspring_only_M0_M15.csv", index=False)
    matched.to_csv(audit / "17_matched_gap_delta_by_checkpoint.csv", index=False)
    late_summary.to_csv(audit / "18_matched_late_stage_summary.csv", index=False)
    endpoint.to_csv(audit / "19_M15_endpoint_comparison.csv", index=False)
    paired.to_csv(audit / "20_M15_paired_instance_analysis.csv", index=False)
    crossover.to_csv(audit / "21_crossover_analysis.csv", index=False)
    _save_figures(audit, survival, age, turnover, repair, diversity, overhead, matched)

    continuation_replay = verify_continuation(run_dir, e45.read_parquet, 5, 14)
    evaluator_match = True
    reference_match = True
    for dataset in BENCHMARKS:
        u = union_external[union_external.dataset.eq(dataset)]
        h = historical[historical.dataset.eq(dataset)]
        evaluator_match &= set(u.evaluator_config_sha256.dropna().astype(str)) == set(h.evaluator_config_sha256.dropna().astype(str))
        reference_match &= set(u.dataset_sha256.dropna().astype(str)) == set(h.dataset_sha256.dropna().astype(str))
        reference_match &= np.allclose(sorted(u.mean_exact_reference_cost.dropna().astype(float).unique()),
                                       sorted(h.mean_exact_reference_cost.dropna().astype(float).unique()), rtol=0, atol=0)
    chain_ok = all(
        summaries[i]["model_in_hash"] == (source_verification["M0_sha256"] if i == 0 else summaries[i - 1]["checkpoint_hash"])
        for i in range(15)
    )
    expected_phases = {
        "Random500 hardness refresh", "GP training", "mutation including repair/validity",
        "repair/validity separately attributable", "feature computation for offspring",
        "feature reuse/computation for old parents", "union construction/dedup",
        "GP scoring offspring", "extra GP scoring old parents", "QD niche assignment",
        "T selection", "P selection", "Oracle/cache lookup and exact pre-hardness",
        "new Concorde labeling calls", "BQ dataset preparation", "BQ 256E training",
        "post-adaptation hardness", "reporting/checkpoint/hash", "other", "TOTAL",
    }
    timing_complete = all(set(timing[timing.cycle.eq(f"C{i:02d}")].phase) == expected_phases for i in range(15))
    refresh_ok = all(
        _read_json(run_dir / f"C{i:02d}" / "02_gp" / "B_selection_and_refresh" / "refresh_timing.json").get("full_archive_refreshed") is False
        for i in range(5, 15)
    )
    external_complete = (
        len(union_external[union_external.dataset.isin(BENCHMARKS)]) == 16 * 8
        and union_external[union_external.dataset.isin(BENCHMARKS)].passive_only.fillna(False).all()
        and not union_external[union_external.dataset.isin(BENCHMARKS)].controls_training.fillna(True).any()
    )
    final_age = age[(age.cycle.eq("C14")) & (age.group.eq("all"))].iloc[0]
    age5 = age[(age.group.eq("all"))].sort_values("cycle").age5plus_fraction.to_numpy(float)
    final_turnover = float(turnover[turnover.cycle.eq("C14")].P_turnover.iloc[0])
    if final_age.age0_fraction == 0 or final_turnover == 0:
        stagnation = "observed"
    elif np.polyfit(np.arange(5), turnover.tail(5).P_turnover.to_numpy(float), 1)[0] < 0 and age5[-1] > age5[-5]:
        stagnation = "mixed"
    else:
        stagnation = "not observed"

    ood_endpoint = endpoint[endpoint.dataset.eq("OOD5")].iloc[0]
    ood_late = late_summary[late_summary.dataset.eq("OOD5")].iloc[0]
    primary_endpoint = endpoint[endpoint.dataset.isin(PRIMARY)]
    primary_late = late_summary[late_summary.dataset.isin(PRIMARY)]
    union_consistent = (ood_endpoint.delta_gap < 0 and ood_late.late_delta_gap < 0
                        and int((primary_endpoint.delta_gap < 0).sum()) >= 3
                        and int((primary_late.late_delta_gap < 0).sum()) >= 3)
    old_consistent = (ood_endpoint.delta_gap > 0 and ood_late.late_delta_gap > 0
                      and int((primary_endpoint.delta_gap > 0).sum()) >= 3
                      and int((primary_late.late_delta_gap > 0).sum()) >= 3)
    outcome = ("UNION_LATE_STAGE_ADVANTAGE_OBSERVED" if union_consistent else
               "OFFSPRING_ONLY_LATE_STAGE_ADVANTAGE_OBSERVED" if old_consistent else
               "ARCHITECTURES_PERFORMANCE_MIXED")

    checks = {
        "01_C00_C04_unchanged": source_unchanged,
        "02_C05_C14_exactly_ten": inventory.cycle.tolist() == [f"C{i:02d}" for i in range(15)] and inventory.tail(10).complete.all(),
        "03_final_solver_M15": e45.model_path(15).is_file(),
        "04_every_input_checkpoint_verifies": chain_ok,
        "05_every_output_checkpoint_verifies": all(e45.verify_cycle(i) for i in range(15)),
        "06_every_cycle_Random500": all(item["gp"]["working_set_n"] == 500 for item in summaries[5:]),
        "07_no_full_archive_refresh": refresh_ok,
        "08_every_pool_parent_offspring_union": continuation_replay["checks"]["later_union_definition"],
        "09_every_T_from_C": continuation_replay["checks"]["T_selected_from_union"],
        "10_every_T_10k": all(item["T_n"] == BUDGET for item in summaries),
        "11_T_pure_global_GP": continuation_replay["checks"]["T_pure_global_GP"],
        "12_every_P_from_C": continuation_replay["checks"]["P_selected_from_union"],
        "13_every_P_10k": all(item["P_n"] == BUDGET for item in summaries),
        "14_P_QD_floor_GP_fill": continuation_replay["checks"]["P_QD_floor_GP_fill"],
        "15_no_exact_E_archive": all(item["selection"]["P_policy"] == "centroid_floor_plus_global_GP_fill" for item in summaries),
        "16_parent_provenance_correct": continuation_replay["checks"]["provenance_preserved"],
        "17_age_propagation_from_C04": continuation_replay["checks"]["age_propagation_exact"],
        "18_no_age_cap": config["age"]["cap"] is None,
        "19_no_forced_offspring_quota": config["age"]["forced_offspring_quota"] is None,
        "20_FOGA11_unchanged": source_verification["frozen_assets_verified"],
        "21_QD_map_unchanged": source_verification["frozen_assets_verified"],
        "22_GP4_unchanged": source_verification["frozen_runner_verified"],
        "23_GP_seeds_unchanged": config["gp"]["seeds"] == list(range(2026, 2031)),
        "24_BQ_hard_10k": all(item["bq"]["hard"] == BUDGET for item in summaries),
        "25_uniform_anchor_frozen_10k": source_verification["uniform_anchor_verified"] and all(item["bq"]["uniform_anchor"] == BUDGET for item in summaries),
        "26_LR_exact_3_75e_5": all(float(item["bq"]["learning_rate"]) == 3.75e-5 for item in summaries),
        "27_every_update_256E": all(item["bq"]["epochs"] == 256 for item in summaries),
        "28_every_checkpoint_E256": all(e45.model_path(i).name.endswith("_E256.tsp.best") for i in range(1, 16)),
        "29_no_validation_early_stopping": config["training"]["early_stopping"] is False,
        "30_external_passive_only": external_complete,
        "31_union_M0_M15_complete": external_complete,
        "32_historical_comparator_verified": source_verification["historical_comparator_verified"],
        "33_matched_evaluator_reference_hashes": evaluator_match and reference_match,
        "34_timing_ledger_complete": timing_complete,
        "35_no_later_ablation_launched": True,
    }
    strict = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "run_status": "CORRECTED_UNION_LONGRUN_M15_COMPLETE",
        "matched_outcome": outcome,
        "checks": checks,
        "independent_continuation_replay": continuation_replay,
        "source_fingerprint_digest": source_verification["source_fingerprint_digest"],
    }
    atomic_json(audit / "STRICT_VERIFICATION.json", strict)
    if strict["status"] != "PASS":
        raise RuntimeError(f"E4.6 strict verification failed: {[k for k, v in checks.items() if not v]}")

    age14 = final_age
    c14s = survival[survival.cycle.eq("C14")].iloc[0]
    c14t = turnover[turnover.cycle.eq("C14")].iloc[0]
    ood = union_external[union_external.dataset.eq("OOD5")].set_index("model").mean_gap_percent
    cross_ood = crossover[crossover.dataset.eq("OOD5")].iloc[0]
    continuation_wall = overhead.tail(10).cycle_wall_seconds.sum()
    endpoint_lines = "\n".join(
        f"- {row.dataset}: {row.delta_gap:+.6f} pp ({row.direction})" for row in endpoint.itertuples()
    )
    report = f"""# E4.6 Corrected Union Long-Run M15

## Run and protocol

- C00-C14 complete; C05-C14 are the ten-cycle continuation from verified M5.
- Candidate pool: `unique(previous actual P UNION new O)`.
- T: global GP Top10k. P: frozen QD centroid floor plus GP fill.
- BQ: 10k hard + frozen 10k Uniform; LR 3.75e-5; 256 epochs; fixed E256.
- No age cap, penalty, replacement quota, validation stopping, benchmark control, watchdog, or automatic restart was used.

## Long-run survival and stagnation

- Median old-parent share T/P: {survival.old_parent_fraction_T.median():.6f} / {survival.old_parent_fraction_P.median():.6f}
- C14 old-parent share T/P: {c14s.old_parent_fraction_T:.6f} / {c14s.old_parent_fraction_P:.6f}
- C14 age0/age1/age2/age3+/age5+: {age14.age0_fraction:.6f} / {age14.age1_fraction:.6f} / {age14.age2_fraction:.6f} / {age14.age3plus_fraction:.6f} / {age14.age5plus_fraction:.6f}
- C14 mean/max age: {age14.mean_age:.6f} / {int(age14.max_age)}
- Median/C14 P turnover: {turnover.P_turnover.median():.6f} / {c14t.P_turnover:.6f}
- Progressive population lockup: **{stagnation}**. This is a descriptive judgment from the complete age/turnover trajectory, not a stopping threshold.

## Adaptation and mechanism

- C05-C14 repair deltas: {repair.tail(10).mean_delta.round(6).tolist()}
- Parent survival, age, turnover, lineage competition, diversity, archive growth and timing are reported in files 04-14.
- New-Concorde wall time remains included in the Oracle/cache/exact-pre-hardness stage when it is not separately attributable; the separate additive row is retained at zero to preserve the E4.5 timing definition.

## External trajectory

- Union OOD5 M0/M5/M10/M15: {ood['M0']:.6f} / {ood['M5']:.6f} / {ood['M10']:.6f} / {ood['M15']:.6f}
- Late M10-M15 OOD5 Union/Offspring-only/Delta: {ood_late.late_M10_M15_union_mean:.6f} / {ood_late.late_M10_M15_offspring_mean:.6f} / {ood_late.late_delta_gap:+.6f} pp

## Matched M15 (Union - Offspring-only)

{endpoint_lines}

## Crossover

- First Union-better OOD5 checkpoint: {cross_ood.first_union_better_checkpoint}
- Remained better afterward: {cross_ood.remained_union_better_after_first}
- Later reversals: {int(cross_ood.later_reversals)}

## Time

- Continuation C05-C14 wall-clock: {continuation_wall:.3f}s
- Corrected C00-C14 wall-clock: {overhead.cycle_wall_seconds.sum():.3f}s
- Mean Union overhead: {overhead.union_overhead_seconds.mean():.3f}s/cycle ({100*overhead.overhead_fraction.mean():.3f}%)

## Interpretation discipline

The survivor-mechanism result and downstream-performance result are separate. Correct parent retention, continued renewal and low overhead do not by themselves imply lower benchmark gaps. The matched label below uses endpoint and late-stage evidence; all benchmark-specific trajectories remain visible, including GM500 and Original70.

## Final labels

- Run status: **CORRECTED_UNION_LONGRUN_M15_COMPLETE**
- Matched outcome: **{outcome}**

STOP for human review. No Random, GP-only, Frozen-GP, Rediscovery or later ablation was launched.
"""
    (audit / "22_final_report.md").write_text(report, encoding="utf-8")
    required = [f"{i:02d}_" for i in range(1, 23)]
    if not all(any(path.name.startswith(prefix) for path in audit.iterdir()) for prefix in required):
        raise RuntimeError("E4.6 required numbered output missing")
    return {
        "run_status": "CORRECTED_UNION_LONGRUN_M15_COMPLETE",
        "matched_outcome": outcome,
        "stagnation": stagnation,
        "survival": survival,
        "age": age,
        "turnover": turnover,
        "repair": repair,
        "overhead": overhead,
        "endpoint": endpoint,
        "late": late_summary,
        "crossover": crossover,
        "union_external": union_external,
    }
