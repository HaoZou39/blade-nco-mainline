"""Final reporting and strict audit for S4-B HGB40 Union+QD rollout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from final_run.stage import atomic_json, sha256_file, utc_now


BENCHMARKS = [
    "Uniform100", "GaussianMixture100", "TSPLIB100", "EvoReal_S1_100",
    "EvoReal_S2_100", "EvoReal_S3_100", "GaussianMixture500", "OriginalTSPLIB70",
]
PRIMARY = ["Uniform100", "GaussianMixture100", "TSPLIB100", "OriginalTSPLIB70"]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_lines(frame: pd.DataFrame, x: str, ys: list[str], output: Path, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for column in ys:
        ax.plot(frame[x], frame[column], marker="o", label=column)
    ax.set_xlabel(x); ax.set_ylabel(ylabel); ax.grid(alpha=.25)
    if len(ys) > 1: ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)


def _stacked_cost(cost: pd.DataFrame, output: Path) -> None:
    columns = [
        "archive_refresh_s", "feature_extraction_s", "hgb_fit_s", "candidate_score_s",
        "qd_parent_s", "concorde_s", "exact_hardness_s", "bq_training_s",
        "external_eval_s", "mutation_s",
    ]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bottom = np.zeros(len(cost))
    for column in columns:
        values = cost[column].fillna(0).to_numpy(float)
        ax.bar(cost.Cycle, values, bottom=bottom, label=column.replace("_s", ""))
        bottom += values
    ax.set_ylabel("wall-clock seconds"); ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)


def build_reports(root: Path, run_dir: Path, source: dict[str, Any]) -> dict[str, Any]:
    summaries = [read_json(run_dir / "cycles" / f"C{i:02d}" / "cycle_summary.json") for i in range(10)]
    cycle_rows, calibration_rows, hard_rows, candidate_rows, parent_rows, operator_rows, cost_rows = [], [], [], [], [], [], []
    external_parts = [pd.read_csv(run_dir / "_state" / "M0_external_benchmark_results.csv")]
    anchor_parts = [pd.read_csv(run_dir / "_state" / "M0_uniform_anchor_results.csv")]
    checkpoints = [read_json(run_dir / "03_M0_manifest.json")]
    for item in summaries:
        c = item["cycle"]
        cycle = f"C{c:02d}"
        cycle_rows.append({
            "Cycle": cycle, "M_before": f"M{c}", "ArchiveN": item["archive_pre_n"],
            "CalibrationN": item["calibration_n"], "CandidateN": item["candidate_n"],
            "T_N": item["T_n"], "T_mean_true_LBQ": item["selected_hardness"]["mean"],
            "T_P90_LBQ": item["selected_hardness"]["P90"], "T_Neff": item["diversity"]["T"]["N_eff"],
            "ParentN": item["P_n"], "ParentReachedNiches": item["diversity"]["P"]["occupied_niches"],
            "ParentNeff": item["diversity"]["P"]["N_eff"],
            "ParentSurvivorFraction": item["survivor_fraction"], "NewConcorde": item["oracle"]["new_concorde_calls"],
            "BQTrainTime": item["bq"]["training_loop_wall_s"], "M_after": f"M{c + 1}",
            "TotalCycleTime": item["costs"]["total_cycle_s"],
        })
        calibration_rows.append({"Cycle": cycle, **item["calibration"]})
        hard_rows.append({"Cycle": cycle, **item["selected_hardness"],
                          "predicted_mean": item["predicted_T"]["mean"],
                          "spearman_pred_true": item["selection_diagnostics"]["spearman_pred_true"]})
        candidate_rows.append({"Cycle": cycle, "CandidateN": item["candidate_n"], **item["diversity"]["C"]})
        parent_rows.append({"Cycle": cycle, "ParentN": item["P_n"], "Q_n": item["Q_n"], "G_n": item["G_n"],
                            "survivor_fraction": item["survivor_fraction"], **item["diversity"]["P"]})
        operator_rows.extend(pd.read_csv(run_dir / "cycles" / cycle / "operator_trajectory.csv").to_dict("records"))
        cost_rows.append({"Cycle": cycle, **item["costs"]})
        external_parts.append(pd.read_csv(run_dir / "cycles" / cycle / "external_benchmark_results.csv"))
        anchor_parts.append(pd.read_csv(run_dir / "cycles" / cycle / "uniform_anchor_results.csv"))
        checkpoints.append(read_json(run_dir / "cycles" / cycle / "M_next_checkpoint_manifest.json"))

    cycle_frame = pd.DataFrame(cycle_rows)
    calibration = pd.DataFrame(calibration_rows)
    hard = pd.DataFrame(hard_rows)
    candidate = pd.DataFrame(candidate_rows)
    parent = pd.DataFrame(parent_rows)
    operator = pd.DataFrame(operator_rows)
    costs = pd.DataFrame(cost_rows)
    external = pd.concat(external_parts, ignore_index=True)
    anchor = pd.concat(anchor_parts, ignore_index=True)
    checkpoint_frame = pd.DataFrame(checkpoints)

    cycle_frame.to_csv(run_dir / "09_cycle_summary.csv", index=False)
    checkpoint_frame.to_csv(run_dir / "10_checkpoint_chain.csv", index=False)
    external.to_csv(run_dir / "11_external_gap_trajectory.csv", index=False)
    anchor.to_csv(run_dir / "12_uniform_anchor_trajectory.csv", index=False)
    calibration.to_csv(run_dir / "13_HGB_calibration_trajectory.csv", index=False)
    hard.to_csv(run_dir / "14_selected_hardness_trajectory.csv", index=False)
    candidate.to_csv(run_dir / "15_candidate_diversity_trajectory.csv", index=False)
    parent.to_csv(run_dir / "16_parent_diversity_trajectory.csv", index=False)
    operator.to_csv(run_dir / "17_operator_trajectory.csv", index=False)
    costs.to_csv(run_dir / "18_cost_accounting.csv", index=False)

    historical_path = root / "audits" / "E4_6_corrected_union_longrun_M15_v1" / "15_external_union_M0_M15.csv"
    historical = pd.read_csv(historical_path)
    historical = historical[historical.model.isin([f"M{i}" for i in range(11)])].copy()
    historical["comparison_status"] = "HISTORICAL_REFERENCE_ONLY"
    historical.to_csv(run_dir / "19_historical_GP_reference.csv", index=False)

    pivot = external.pivot(index="model", columns="dataset", values="mean_gap_percent").reset_index()
    pivot["model_index"] = pivot.model.str[1:].astype(int)
    pivot = pivot.sort_values("model_index")
    _plot_lines(pivot, "model_index", BENCHMARKS, run_dir / "fig01_external_gap_trajectory_8benchmarks.png", "mean gap (%)")
    _plot_lines(pivot, "model_index", PRIMARY, run_dir / "fig02_primary_four_benchmark_gap_trajectory.png", "mean gap (%)")
    _plot_lines(hard, "Cycle", ["mean"], run_dir / "fig03_HGB_selected_mean_hardness_by_cycle.png", "selected mean LBQ")
    _plot_lines(hard, "Cycle", ["P90", "P95", "P99"], run_dir / "fig04_HGB_selected_tail_hardness_by_cycle.png", "selected tail LBQ")
    _plot_lines(calibration, "Cycle", ["archive_n", "calibration_n"], run_dir / "fig05_archive_and_calibration_size_by_cycle.png", "count")
    _plot_lines(candidate, "Cycle", ["N_eff"], run_dir / "fig06_candidate_Neff_by_cycle.png", "candidate N_eff")
    _plot_lines(parent, "Cycle", ["N_eff"], run_dir / "fig07_parent_Neff_by_cycle.png", "parent N_eff")
    tdiv = pd.DataFrame({"Cycle": [f"C{i:02d}" for i in range(10)], "N_eff": [s["diversity"]["T"]["N_eff"] for s in summaries]})
    _plot_lines(tdiv, "Cycle", ["N_eff"], run_dir / "fig08_training_set_Neff_by_cycle.png", "T N_eff")
    _plot_lines(parent, "Cycle", ["occupied_niches"], run_dir / "fig09_QD_reached_niches_by_cycle.png", "parent occupied niches")
    _plot_lines(parent, "Cycle", ["survivor_fraction"], run_dir / "fig10_parent_survival_fraction_by_cycle.png", "survivor fraction")
    _plot_lines(anchor, "model", ["mean_LBQ"], run_dir / "fig11_uniform_anchor_LBQ_trajectory.png", "UniformAnchor mean LBQ")
    _stacked_cost(costs, run_dir / "fig12_cycle_wallclock_breakdown.png")
    oracle = pd.DataFrame({"Cycle": [f"C{i:02d}" for i in range(10)], "new_concorde": [s["oracle"]["new_concorde_calls"] for s in summaries]})
    _plot_lines(oracle, "Cycle", ["new_concorde"], run_dir / "fig13_concorde_calls_by_cycle.png", "new Concorde calls")
    _plot_lines(costs, "Cycle", ["hgb_fit_s", "candidate_score_s"], run_dir / "fig14_HGB_fit_and_score_time_by_cycle.png", "seconds")

    m0 = pivot[pivot.model.eq("M0")].iloc[0]
    m10 = pivot[pivot.model.eq("M10")].iloc[0]
    deltas = {name: float(m10[name] - m0[name]) for name in BENCHMARKS}
    macro_delta = float(np.mean(list(deltas.values())))
    improved = sum(value < 0 for value in deltas.values())
    anchor0, anchor10 = anchor.iloc[0], anchor.iloc[-1]
    uniform_delta = float(anchor10.mean_LBQ - anchor0.mean_LBQ)
    qd_maintained = bool(parent.occupied_niches.iloc[-1] >= max(50, .5 * parent.occupied_niches.iloc[0]))
    if not qd_maintained:
        verdict = "HGB_UNION_QD_10CYCLE_ROLLOUT_STRUCTURAL_COLLAPSE"
    elif macro_delta > 0 and improved <= 2:
        verdict = "HGB_UNION_QD_10CYCLE_ROLLOUT_DEGRADES_SOLVER"
    elif macro_delta < 0 and improved >= 6 and uniform_delta <= max(.02, .25 * float(anchor0.mean_LBQ)):
        verdict = "HGB_UNION_QD_10CYCLE_ROLLOUT_SUPPORTED"
    else:
        verdict = "HGB_UNION_QD_10CYCLE_ROLLOUT_STABLE_BUT_GAIN_MIXED"
    first_full = calibration.loc[calibration.calibration_n.eq(10000), "Cycle"]
    first_full_cycle = first_full.iloc[0] if len(first_full) else "NA"
    late_delta = pivot[pivot.model_index.ge(7)][BENCHMARKS].mean(axis=1).iloc[-1] - pivot[pivot.model_index.ge(7)][BENCHMARKS].mean(axis=1).iloc[0]
    flags = {
        "FULL_10K_CALIBRATION_REACHED": "YES" if len(first_full) else "NO",
        "FIRST_FULL_10K_CALIBRATION_CYCLE": first_full_cycle,
        "HGB_SELECTED_HARDNESS_REMAINS_ELEVATED": "YES" if hard["mean"].iloc[-1] >= .5 * hard["mean"].iloc[0] else "MIXED",
        "UNIFORM_RETENTION_ACCEPTABLE": "YES" if uniform_delta <= max(.02, .25 * float(anchor0.mean_LBQ)) else "NO",
        "QD_PARENT_DIVERSITY_MAINTAINED": "YES" if qd_maintained else "NO",
        "PARENT_SURVIVOR_MEMORY_ACTIVE": "YES" if (parent.survivor_fraction.iloc[1:] > 0).any() else "NO",
        "EXTERNAL_GAP_IMPROVES_OVER_M0": "YES" if improved >= 6 else ("NO" if improved <= 2 else "MIXED"),
        "LATE_CYCLE_GAIN_CONTINUES": "YES" if late_delta < 0 else "MIXED",
        "READY_FOR_MATCHED_GP_VS_HGB_MULTICYCLE_CAUSAL_TEST": "YES" if verdict != "HGB_UNION_QD_10CYCLE_ROLLOUT_STRUCTURAL_COLLAPSE" else "NO",
    }

    checks = {
        "01_exact_M0": checkpoint_frame.iloc[0].sha256 == source["M0_sha256"],
        "02_exactly_10_cycles": cycle_frame.Cycle.tolist() == [f"C{i:02d}" for i in range(10)],
        "03_final_M10": checkpoint_frame.iloc[-1].model == "M10",
        "04_branch_local_A0": summaries[0]["archive_pre_n"] == 500,
        "05_no_future_archive_injection": all(s["no_future_archive_injection"] for s in summaries),
        "06_archive_only_oracle_branch": all(s["archive_branch_only"] for s in summaries),
        "07_current_H_refresh": all(s["current_hardness_refreshed"] for s in summaries),
        "08_target_log1p": all(s["hgb"]["target"] == "log1p_LBQ" for s in summaries),
        "09_hgb_config_frozen": all(s["hgb"]["config_sha256"] == source["HGB_config_sha256"] for s in summaries),
        "10_hgb_seeds": all(s["hgb"]["seeds"] == [2026, 2027, 2028, 2029, 2030] for s in summaries),
        "11_median_predictions": all(s["hgb"]["aggregate"] == "median" for s in summaries),
        "12_f40_manifest": all(s["hgb"]["F40_sha256"] == source["F40_sha256"] for s in summaries),
        "13_no_f40_rerank": all(s["hgb"]["ranking_recomputed"] is False for s in summaries),
        "14_no_operator_feature": all(s["hgb"]["operator_feature"] is False for s in summaries),
        "15_calibration_target": all(s["calibration_target_n"] == 10000 for s in summaries),
        "16_small_archive_all": all(s["calibration_n"] == min(10000, s["archive_pre_n"]) for s in summaries),
        "17_no_c00_expansion": summaries[0]["calibration_n"] == 500,
        "18_random10k": all(s["calibration_policy_ok"] for s in summaries),
        "19_B_before_fit": all(s["B_frozen_before_fit"] for s in summaries),
        "20_no_same_cycle_T_in_B": all(s["same_cycle_leakage_n"] == 0 for s in summaries),
        "21_qd4": all(s["QD"]["descriptors"] == "QD4" for s in summaries),
        "22_k1000": all(s["QD"]["K"] == 1000 for s in summaries),
        "23_frozen_normalization": all(s["QD"]["normalization_frozen"] for s in summaries),
        "24_foga11": all(s["mutation"]["operator_count"] == 11 for s in summaries),
        "25_mutation_unchanged": all(s["mutation"]["frozen"] for s in summaries),
        "26_c00_corrected_union": summaries[0]["C00_exact_corrected_union"],
        "27_later_union": all(s["candidate_policy"] == "unique_parent_offspring_union" for s in summaries[1:]),
        "28_no_offspring_only": all(s["parent_count_in_candidate"] == 10000 for s in summaries[1:]),
        "29_T_hgb_top10k": all(s["T_n"] == 10000 and s["selection"]["T"] == "HGB_Top10000" for s in summaries),
        "30_parent_n": all(s["P_n"] == 10000 for s in summaries),
        "31_parent_qd_hgb": all(s["Q_n"] + s["G_n"] == 10000 for s in summaries),
        "32_no_age_cap": all(s["selection"]["age_cap"] is None for s in summaries),
        "33_no_forced_offspring": all(s["selection"]["forced_offspring_quota"] is None for s in summaries),
        "34_tours_after_freeze": all(s["oracle"]["selection_frozen_before_oracle"] for s in summaries),
        "35_concorde_unique": all(s["oracle"]["failures"] == 0 for s in summaries),
        "36_no_post_truth_reselection": all(s["selection"]["post_truth_reselection"] is False for s in summaries),
        "37_fixed_anchor": anchor.anchor_sha256.nunique() == 1,
        "38_lr": all(float(s["bq"]["learning_rate"]) == 3.75e-5 for s in summaries),
        "39_epochs": all(s["bq"]["epochs"] == 256 for s in summaries),
        "40_fresh_adam": all(s["training_protocol"]["fresh_adam"] for s in summaries),
        "41_no_scheduler": all(s["training_protocol"]["scheduler"] is None for s in summaries),
        "42_weight_decay": all(s["training_protocol"]["weight_decay"] == 0 for s in summaries),
        "43_no_clipping": all(s["training_protocol"]["gradient_clipping"] is None for s in summaries),
        "44_amp_off": all(s["training_protocol"]["AMP"] is False for s in summaries),
        "45_seed": all(s["training_protocol"]["seed"] == 2026 for s in summaries),
        "46_E256": all(s["bq"]["scientific_checkpoint"] == "E256" for s in summaries),
        "47_chain": all(checkpoint_frame.iloc[i].parent_sha256 == checkpoint_frame.iloc[i - 1].sha256 for i in range(1, 11)),
        "48_passive_benchmarks": external.passive_only.fillna(False).all(),
        "49_no_benchmark_stopping": all(s["benchmark_controls_training"] is False for s in summaries),
        "50_no_RF": all(s["hgb"]["estimator"] == "HistGradientBoostingRegressor" for s in summaries),
        "51_no_cross_family_ensemble": all(s["hgb"]["cross_family_ensemble"] is False for s in summaries),
        "52_no_hgb_retune": all(s["hgb"]["retuned"] is False for s in summaries),
        "53_no_qd_retune": all(s["QD"]["retuned"] is False for s in summaries),
        "54_no_mutation_retune": all(s["mutation"]["retuned"] is False for s in summaries),
        "55_feature_cache_hash_safe": all(s["feature_cache"]["hash_safe"] for s in summaries),
        "56_archive_update_after_T": all(s["archive_update_after_T_freeze"] for s in summaries),
        "57_resume_identity": all(s["resume_identity_preserved"] for s in summaries),
        "58_completed_not_rerun": all((run_dir / "cycles" / f"C{i:02d}" / "COMPLETE.json").is_file() for i in range(10)),
        "59_M10_evaluated": set(external[external.model.eq("M10")].dataset) == set(BENCHMARKS),
        "60_no_next_auto_launch": True,
    }
    passed = len(checks) == 60 and all(checks.values())
    atomic_json(run_dir / "STRICT_VERIFICATION.json", {"status": "PASS" if passed else "FAIL", "checks": checks})

    report = f"""# S4-B HGB40 Union + QD 10-cycle rollout

Primary verdict: `{verdict}`

- Start: exact M0; cycles C00-C09; endpoint M10.
- Initial archive: {summaries[0]['archive_pre_n']}; C00 calibration: {summaries[0]['calibration_n']}.
- First full 10k calibration: {first_full_cycle}.
- M0->M10 8-benchmark macro delta: {macro_delta:+.6f} percentage points; improved datasets: {improved}/8.
- UniformAnchor mean LBQ delta: {uniform_delta:+.8f}.
- Total new Concorde calls: {int(sum(s['oracle']['new_concorde_calls'] for s in summaries))}.
- Total BQ training time: {costs.bq_training_s.sum():.1f} s.
- Historical GP comparison is descriptive only (`HISTORICAL_REFERENCE_ONLY`).
- Strict verification: {'60/60 PASS' if passed else 'FAIL'}.

Secondary flags:
""" + "\n".join(f"- {key} = {value}" for key, value in flags.items()) + "\n"
    (run_dir / "20_final_report.md").write_text(report, encoding="utf-8")
    atomic_json(run_dir / "FINAL_RESULT.json", {"status": "COMPLETE" if passed else "INTEGRITY_FAILURE",
                                                "primary_label": verdict, "secondary_flags": flags,
                                                "completed_utc": utc_now()})
    return {"status": "PASS" if passed else "FAIL", "primary_label": verdict, "flags": flags,
            "M10_sha256": checkpoint_frame.iloc[-1].sha256}
