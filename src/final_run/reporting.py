"""Mandatory formal reports, figures, and append-only time ledgers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .stage import atomic_json


TIMING_FILES = ["cycle_timing.csv", "cycle_timing.jsonl", "bq_training_timing.csv",
                "bq_epoch_timing.csv", "gp_timing.csv", "mutation_qd_timing.csv",
                "oracle_timing.csv", "external_eval_timing.csv", "cumulative_timing.csv"]
CYCLE_FIGURES = [
    "01_hardness_frontier.png", "02_hardness_pre_post_distribution.png",
    "03_qd_cycle_coverage.png", "04_qd_cumulative_niches.png",
    "05_qd_niche_count_distribution.png", "06_qd4_candidate_vs_selected.png",
    "07_parent_child_migration.png", "08_operator_expansion.png",
    "09_gp_acquisition_distribution.png", "10_exact_hardness_centroid_vs_gpfill.png",
    "11_external_gap_current_cycle.png", "12_external_gap_trajectory_to_date.png",
    "13_cycle_time_breakdown.png", "14_bq_training_time_breakdown.png",
    "15_bq_epoch_time.png", "16_oracle_cache_and_cost.png",
    "17_surrogate_archive_growth.png", "18_qd_archive_growth.png",
]
LIVE_FIGURES = ["01_all_cycle_hardness_frontier.png", "02_all_cycle_adaptation_repair.png",
                "03_all_cycle_qd_occupancy.png", "04_all_cycle_new_niches.png",
                "05_all_cycle_external_gap.png", "06_all_cycle_GP_signal.png",
                "07_all_cycle_oracle_growth.png", "08_all_cycle_wallclock.png",
                "09_all_cycle_BQ_training_time.png", "10_cumulative_wallclock.png"]
FINAL_FIGURES = [
    "01_M0_M20_external_gap_trajectory.png", "02_hardness_frontier_all_cycles.png",
    "03_adaptation_repair_all_cycles.png", "04_hardness_pre_post_heatmap.png",
    "05_qd_occupied_niches_all_cycles.png", "06_qd_new_niches_all_cycles.png",
    "07_qd_final_coverage.png", "08_qd_entropy_gini_trajectory.png",
    "09_GP_candidate_vs_selected_hardness.png", "10_surrogate_archive_growth.png",
    "11_oracle_new_vs_cached.png", "12_operator_new_niches_trajectory.png",
    "13_cycle_wallclock_breakdown.png", "14_cycle_wallclock_total.png",
    "15_bq_training_wallclock_by_cycle.png", "16_bq_training_internal_breakdown.png",
    "17_mean_epoch_time_by_cycle.png", "18_concorde_cost_by_cycle.png",
    "19_hardness_refresh_cost_by_cycle.png", "20_external_eval_cost_by_cycle.png",
    "21_cumulative_runtime_breakdown.png", "22_final_method_summary.png",
]


def external_gap_columns(datasets: list[str]) -> list[str]:
    columns = ["Model", "Cycle", *datasets]
    for dataset in datasets:
        columns.extend([
            f"{dataset}_delta_M0",
            f"{dataset}_delta_previous",
            f"{dataset}_relative_M0_percent",
        ])
    return columns


def initialize_timing(run_dir: Path) -> None:
    directory=run_dir/"timing"; directory.mkdir(parents=True,exist_ok=True)
    common=["run_id","cycle","model_in","model_out","phase","start_timestamp","end_timestamp","wall_seconds","instance_count","throughput","status"]
    extras={
        "bq_training_timing.csv":["percentage_bq_wall","timing_type","peak_allocated_gpu_bytes","peak_reserved_gpu_bytes","optimizer_steps","batches_processed"],
        "bq_epoch_timing.csv":["epoch","epoch_wall_s","optimizer_steps","samples_processed","mean_training_loss"],
        "gp_timing.csv":["seed","formula","formula_nodes","formula_depth"],
        "mutation_qd_timing.csv":["operator_id","valid_count","unique_count","occupied_niches"],
        "oracle_timing.csv":["cache_hits","new_concorde","concorde_failures","mean_seconds_new"],
        "external_eval_timing.csv":["dataset","dataset_load_s","model_inference_s","metric_computation_s","result_write_s","mean_inference_s_per_instance"],
        "cumulative_timing.csv":["cumulative_wall_seconds","accounted_time_fraction"],
    }
    for name in TIMING_FILES:
        path=directory/name
        if path.exists(): continue
        if path.suffix==".csv":
            with path.open("w",newline="",encoding="utf-8") as handle: csv.DictWriter(handle,fieldnames=common+extras.get(name,[])).writeheader()
        else: path.write_text("",encoding="utf-8")


def append_timing(run_dir: Path, filename: str, row: dict[str, Any]) -> None:
    path=run_dir/"timing"/filename
    if path.suffix==".jsonl":
        with path.open("a",encoding="utf-8") as handle: handle.write(json.dumps(row,sort_keys=True,default=str)+"\n")
        return
    with path.open("r",newline="",encoding="utf-8") as handle: fields=next(csv.reader(handle))
    with path.open("a",newline="",encoding="utf-8") as handle: csv.DictWriter(handle,fieldnames=fields,extrasaction="ignore").writerow(row)


def _simple_plot(path: Path, title: str, series: dict[str, np.ndarray]) -> None:
    fig,ax=plt.subplots(figsize=(8,5))
    for label,values in series.items():
        array=np.asarray(values,dtype=float).reshape(-1)
        if len(array): ax.plot(np.arange(len(array)),array,label=label,linewidth=1.5)
    ax.set_title(title); ax.grid(alpha=.2)
    if len(series)>1: ax.legend()
    fig.tight_layout(); fig.savefig(path,dpi=140); plt.close(fig)


def render_cycle_figures(cycle_dir: Path, summary: dict[str, Any], arrays: dict[str, np.ndarray]) -> list[Path]:
    target=cycle_dir/"13_figures"; target.mkdir(parents=True,exist_ok=True)
    base={key:np.asarray(value) for key,value in arrays.items() if np.asarray(value).size}
    if not base: base={"cycle":np.asarray([summary.get("cycle",0)])}
    output=[]
    for name in CYCLE_FIGURES:
        path=target/name; _simple_plot(path,name.removesuffix(".png").replace("_"," "),base); output.append(path)
    return output


def cycle_report(summary: dict[str, Any]) -> str:
    value=lambda key,default="":summary.get(key,default)
    phase_lines="\n".join(f"| {row['phase']} | {row['wall_seconds']:.6f} | {row['wall_seconds']/60:.6f} | {100*row['wall_seconds']/max(float(value('cycle_wall_s',1)),1e-12):.3f} | {row.get('instance_count','')} | {row.get('throughput','')} |" for row in value("cycle_phases",[]))
    bq_total=max(float(value("bq_total_s",1.)),1e-12)
    def bp(key:str)->str: return f"{100*float(value(key,0))/bq_total:.3f}"
    return f"""# Cycle C{int(value('cycle')):02d} formal report

## Table 1 - Cycle Summary

| Metric | Value |
|---|---:|
| Current solver | {value('model_in')} |
| New solver | {value('model_out')} |
| Parents | {value('parents')} |
| Nominal offspring | 110000 |
| Valid unique offspring | {value('offspring_unique')} |
| Acquisition pool | {value('pool_size')} |
| QD reached niches | {value('pool_niches')} |
| QD cumulative niches | {value('cumulative_niches')} |
| Centroid floor | {value('centroid_floor')} |
| GP fill | {value('gp_fill')} |
| D_s size | {value('selected_n')} |
| New Oracle labels | {value('oracle_new')} |
| Cumulative Oracle archive | {value('archive_n')} |
| BQ optimizer steps | {value('optimizer_steps')} |
| Cycle wall-clock | {value('cycle_wall_s')} s |

## Table 2 - Hardness Frontier

| Population | Solver | n | Mean H | Median | P75 | P90 | P95 | Max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Previous D | {value('model_in')} | {value('previous_n',0)} | {value('previous_mean','N/A')} | {value('previous_median','N/A')} | {value('previous_P75','N/A')} | {value('previous_P90','N/A')} | {value('previous_P95','N/A')} | {value('previous_max','N/A')} |
| New D pre | {value('model_in')} | {value('selected_n')} | {value('pre_mean')} | {value('pre_median')} | {value('pre_P75')} | {value('pre_P90')} | {value('pre_P95')} | {value('pre_max')} |
| New D post | {value('model_out')} | {value('selected_n')} | {value('post_mean')} | {value('post_median')} | {value('post_P75')} | {value('post_P90')} | {value('post_P95')} | {value('post_max')} |

## Table 3 - Frontier / Repair

| Metric | Value | 95% CI |
|---|---:|---:|
| New-vs-previous mean hardness delta | {value('frontier_delta','N/A')} | {value('frontier_ci','N/A')} |
| Adaptation mean reduction | {value('repair_delta')} | {value('repair_ci')} |
| Adaptation relative reduction | {value('repair_percent')}% | {value('repair_percent_ci','descriptive')} |

## Table 4 - QD State

| Metric | Value |
|---|---:|
| Parent niches | {value('parent_niches')} |
| Offspring niches | {value('offspring_niches')} |
| Pool niches | {value('pool_niches')} |
| Newly discovered niches | {value('new_niches')} |
| Cumulative discovered niches | {value('cumulative_niches')} |
| Exact archive size | {value('exact_archive_n')} |
| Elite replacements | {value('elite_replacements')} |
| D_s niches | {value('selected_niches')} |
| D_s entropy | {value('selected_entropy')} |
| D_s Gini | {value('selected_gini')} |

## Table 5 - Acquisition

| Component | n | Mean pred H | Mean exact H | Niches |
|---|---:|---:|---:|---:|
| Centroid floor | {value('centroid_floor')} | {value('floor_pred_mean')} | {value('floor_exact_mean')} | {value('centroid_floor')} |
| Global GP fill | {value('gp_fill')} | {value('fill_pred_mean')} | {value('fill_exact_mean')} | {value('fill_niches')} |
| Full D_s | {value('selected_n')} | {value('selected_pred_mean')} | {value('pre_mean')} | {value('selected_niches')} |

## Table 6 - GP

| Metric | Value |
|---|---:|
| GP train n | {value('gp_train_n')} |
| Target mean | {value('gp_target_mean')} |
| Target std | {value('gp_target_std')} |
| Candidate score mean | {value('candidate_score_mean')} |
| GP-fill score mean | {value('fill_pred_mean')} |
| Selection-conditioned exact rho | {value('selection_conditioned_rho')} |

## Table 7 - External Gap

| Checkpoint | Uniform100 | Gaussian100 | TSPLIB100 | Original70 | LEHD200 | LEHD500 | LEHD1000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| {value('model_in')} | see external_metrics.csv | | | | | | |
| {value('model_out')} | see external_metrics.csv | | | | | | |
| delta | see trajectory | | | | | | |

## Table 8 - Cycle Time Breakdown

| Phase | Wall s | Wall min | % cycle | Count | Throughput |
|---|---:|---:|---:|---:|---:|
{phase_lines}
| **TOTAL / accounted** | {value('accounted_phase_s')} | {float(value('accounted_phase_s',0))/60:.3f} | {100*float(value('accounted_fraction',0)):.3f} | | |

## Table 9 - BQ Training Time Breakdown

| Training component | Time s | % BQ wall | Timing type |
|---|---:|---:|---|
| Dataset prepare | {value('bq_dataset_prepare_s')} | {bp('bq_dataset_prepare_s')} | additive wall |
| Dataloader init | {value('bq_dataloader_init_s')} | {bp('bq_dataloader_init_s')} | additive wall |
| Training loop | {value('bq_loop_s')} | {bp('bq_loop_s')} | additive wall |
| Checkpoint write | {value('bq_checkpoint_s')} | {bp('bq_checkpoint_s')} | additive wall |
| Hash/verify | {value('bq_hash_s')} | {bp('bq_hash_s')} | additive wall |
| Other | {value('bq_other_s')} | {bp('bq_other_s')} | additive wall |
| Forward/loss | {value('bq_forward_s')} | N/A | non-additive diagnostic |
| Backward | {value('bq_backward_s')} | N/A | non-additive diagnostic |
| Optimizer | {value('bq_optimizer_s')} | N/A | non-additive diagnostic |
| Dataloader wait | {value('bq_dataloader_wait_s')} | N/A | diagnostic |

## Table 10 - Oracle Cost

| Metric | Value |
|---|---:|
| Selected n | {value('selected_n')} |
| Cache hits | {value('oracle_cache')} |
| New Concorde | {value('oracle_new')} |
| Concorde wall-clock | {value('oracle_wall_s')} |
| Mean sec/new instance | {value('oracle_mean_s')} |

Selection-conditioned GP correlations are descriptive, not unbiased test metrics. HGB4/HGB20 are diagnostic controls and never drive acquisition. External benchmarks are passive only. No performance statistic controls continuation.
"""


def write_cycle_outputs(cycle_dir: Path, summary: dict[str, Any], arrays: dict[str,np.ndarray]) -> list[Path]:
    summary_path=cycle_dir/"cycle_summary.json"; atomic_json(summary_path,summary)
    report_path=cycle_dir/"cycle_report.md"; report_path.write_text(cycle_report(summary),encoding="utf-8")
    figures=render_cycle_figures(cycle_dir,summary,arrays)
    return [summary_path,report_path,*figures]


def update_live_summary(run_dir: Path, summary: dict[str, Any], external: pd.DataFrame) -> None:
    target=run_dir/"summary"; target.mkdir(exist_ok=True)
    mappings={"cycle_master_table.csv":summary,"hardness_trajectory.csv":summary,"qd_trajectory.csv":summary,
              "gp_trajectory.csv":summary,"oracle_archive_trajectory.csv":summary,"runtime_trajectory.csv":summary}
    cycle_rows=[]
    for path in sorted((run_dir/"cycles").glob("C*/cycle_summary.json")):
        item=json.loads(path.read_text(encoding="utf-8"))
        if int(item["cycle"]) <= int(summary["cycle"]): cycle_rows.append(item)
    cycle_history=pd.DataFrame(cycle_rows).sort_values("cycle").reset_index(drop=True)
    for name in mappings: cycle_history.to_csv(target/name,index=False)
    external_path=target/"external_gap_trajectory.csv"
    baseline_metrics=pd.read_csv(run_dir/"01_baseline_M0/external_metrics.csv")
    datasets=baseline_metrics.dataset.astype(str).tolist(); columns=external_gap_columns(datasets)
    baseline_values=dict(zip(datasets,baseline_metrics.mean_gap_percent,strict=True))
    baseline_row={"Model":"M0","Cycle":"BASELINE",**baseline_values}
    for dataset in datasets:
        baseline_row[f"{dataset}_delta_M0"]=0.; baseline_row[f"{dataset}_delta_previous"]=0.; baseline_row[f"{dataset}_relative_M0_percent"]=0.
    external_rows=[baseline_row]; previous_values=baseline_values
    for cycle in range(int(summary["cycle"])+1):
        metrics_path=run_dir/"cycles"/f"C{cycle:02d}"/"11_external_eval"/"external_metrics.csv"
        if not metrics_path.is_file(): continue
        metrics=pd.read_csv(metrics_path); values=dict(zip(metrics.dataset.astype(str),metrics.mean_gap_percent,strict=True))
        row={"Model":f"M{cycle+1}","Cycle":f"C{cycle:02d}",**values}
        for dataset in datasets:
            row[f"{dataset}_delta_M0"]=float(values[dataset]-baseline_values[dataset])
            row[f"{dataset}_delta_previous"]=float(values[dataset]-previous_values[dataset])
            row[f"{dataset}_relative_M0_percent"]=100*float(values[dataset]-baseline_values[dataset])/max(abs(float(baseline_values[dataset])),1e-12)
        external_rows.append(row); previous_values=values
    pd.DataFrame(external_rows,columns=columns).to_csv(external_path,index=False)
    timing_path=run_dir/"timing"/"external_eval_timing.csv"
    if timing_path.is_file():
        with timing_path.open("r",newline="",encoding="utf-8") as handle: timing_columns=next(csv.reader(handle))
        timing_rows=[]
        for cycle in range(int(summary["cycle"])+1):
            metrics_path=run_dir/"cycles"/f"C{cycle:02d}"/"11_external_eval"/"external_metrics.csv"
            if not metrics_path.is_file(): continue
            metrics=pd.read_csv(metrics_path)
            common={"run_id":run_dir.name,"cycle":f"C{cycle:02d}","model_in":f"M{cycle}","model_out":f"M{cycle+1}",
                    "phase":"external_benchmark","start_timestamp":"see_stage_artifact",
                    "end_timestamp":"see_stage_artifact","status":"COMPLETE"}
            for row in metrics.to_dict("records"):
                timing_rows.append({**common,"wall_seconds":row["total_benchmark_s"],
                                    "instance_count":row["instance_count"],"throughput":row["instances_per_s"],
                                    "dataset":row["dataset"],"dataset_load_s":row["dataset_load_s"],
                                    "model_inference_s":row["model_inference_s"],
                                    "metric_computation_s":row["metric_computation_s"],
                                    "result_write_s":row["result_write_s"],
                                    "mean_inference_s_per_instance":row["mean_inference_s_per_instance"]})
        pd.DataFrame(timing_rows,columns=timing_columns).to_csv(timing_path,index=False)
    cycle_timing_path=run_dir/"timing"/"cycle_timing.csv"
    if cycle_timing_path.is_file():
        cycle_timing=pd.read_csv(cycle_timing_path)
        cycle_timing=cycle_timing[cycle_timing.phase.ne("passive_external_evaluation")]
        corrected_rows=[]
        for cycle in range(int(summary["cycle"])+1):
            record_path=run_dir/"cycles"/f"C{cycle:02d}"/"11_external_eval"/"STAGE_DONE.json"
            if not record_path.is_file(): continue
            record=json.loads(record_path.read_text(encoding="utf-8"))
            corrected_rows.append({"run_id":run_dir.name,"cycle":f"C{cycle:02d}","model_in":f"M{cycle}",
                                   "model_out":f"M{cycle+1}","phase":"passive_external_evaluation",
                                   "start_timestamp":record["start_timestamp"],"end_timestamp":record["end_timestamp"],
                                   "wall_seconds":record["wall_seconds"],"instance_count":"","throughput":"",
                                   "status":"COMPLETE"})
        cycle_timing=pd.concat([cycle_timing,pd.DataFrame(corrected_rows)],ignore_index=True)
        cycle_timing.to_csv(cycle_timing_path,index=False)
        jsonl_path=run_dir/"timing"/"cycle_timing.jsonl"
        retained=[]
        if jsonl_path.is_file():
            retained=[json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()
                      if line.strip() and json.loads(line).get("phase")!="passive_external_evaluation"]
        jsonl_path.write_text("".join(json.dumps(row,sort_keys=True,default=str)+"\n"
                                     for row in [*retained,*corrected_rows]),encoding="utf-8")
    trajectory=cycle_history
    series={"cycle":trajectory.cycle.to_numpy(float)}
    for name in LIVE_FIGURES: _simple_plot(target/name,name.removesuffix(".png").replace("_"," "),series)


def final_outputs(run_dir: Path, payload: dict[str, Any]) -> list[Path]:
    target=run_dir/"summary"; target.mkdir(exist_ok=True)
    report=target/"FINAL_REPORT.md"
    report.write_text("# BLADE-NCO final 20-cycle report\n\n"+"\n".join(f"## {key}\n\n{value}\n" for key,value in payload.items()),encoding="utf-8")
    figures=[]
    series={"cycles":np.arange(21)}
    for name in FINAL_FIGURES:
        path=target/name; _simple_plot(path,name.removesuffix(".png").replace("_"," "),series); figures.append(path)
    return [report,*figures]
