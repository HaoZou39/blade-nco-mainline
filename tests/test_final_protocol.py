from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from final_run.protocol import load_and_validate
from final_run import science
from final_run.reporting import external_gap_columns, update_live_summary
from final_run.science import attach_oracle_metadata
from oracle.concorde import ORACLE_LEDGER_COLUMNS, canonical_oracle_ledger


def test_final_protocol_is_frozen_and_complete():
    config=load_and_validate(ROOT/"configs/final_20cycle_protocol.yaml")
    assert config["run"]["cycle_indices"]==list(range(20))
    assert config["run"]["model_final"]=="M20"
    assert config["acquisition"]["fill"]["type"]=="global_gp_topk"
    assert config["acquisition"]["round_robin"] is False
    assert config["training"]["epochs"]==256


def test_final_runner_present():
    assert (ROOT/"scripts/run_final_20cycle.py").is_file()


def test_sperf_candidates_are_json_serializable(monkeypatch):
    monkeypatch.setattr(
        science,
        "_formula_prediction",
        lambda record, x, transform: np.asarray([1.0, 2.0]),
    )
    hall = [{"nodes": 1, "canonical_formula": "x"}]
    _, candidates = science._select_sperf(
        hall,
        np.zeros((2, 4)),
        np.asarray([1.0, 2.0]),
        {"center": 0.0, "scale": 1.0},
    )

    json.dumps(candidates)


def test_external_gap_trajectory_has_stable_schema():
    datasets = ["A", "B"]

    assert external_gap_columns(datasets) == [
        "Model", "Cycle", "A", "B",
        "A_delta_M0", "A_delta_previous", "A_relative_M0_percent",
        "B_delta_M0", "B_delta_previous", "B_relative_M0_percent",
    ]


def test_live_summary_rebuild_is_idempotent(tmp_path):
    run_dir = tmp_path / "run"
    baseline = run_dir / "01_baseline_M0"
    cycle = run_dir / "cycles" / "C00"
    external_dir = cycle / "11_external_eval"
    baseline.mkdir(parents=True)
    external_dir.mkdir(parents=True)
    metrics = pd.DataFrame({"dataset": ["A", "B"], "mean_gap_percent": [1.0, 2.0]})
    metrics.to_csv(baseline / "external_metrics.csv", index=False)
    adapted = metrics.assign(mean_gap_percent=[0.5, 2.5])
    adapted.to_csv(external_dir / "external_metrics.csv", index=False)
    summary = {"cycle": 0, "model_out": "M1", "score": 3.0}
    (cycle / "cycle_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    update_live_summary(run_dir, summary, adapted)
    update_live_summary(run_dir, summary, adapted)

    cycle_history = pd.read_csv(run_dir / "summary" / "cycle_master_table.csv")
    gap_history = pd.read_csv(run_dir / "summary" / "external_gap_trajectory.csv")
    assert len(cycle_history) == 1
    assert gap_history[["Model", "Cycle"]].values.tolist() == [["M0", "BASELINE"], ["M1", "C00"]]


def test_oracle_ledger_and_metadata_drop_accumulated_suffixes():
    ledger = pd.DataFrame({
        "instance_hash": ["a", "b"], "status": ["solved", "solved"],
        "verified": [True, True], "optimal_cost_int": [10, 20],
        "runtime_sec": [0.1, 0.2], "expert_tour": ["0 1", "1 0"],
        "tour_hash": ["ta", "tb"], "binary_hash": ["bin", "bin"],
        "niche_id": [7, 8], "status_oracle": ["stale", "stale"],
    })
    canonical = canonical_oracle_ledger(ledger)
    assert tuple(canonical.columns) == ORACLE_LEDGER_COLUMNS

    selected = pd.DataFrame({
        "instance_hash": ["a", "b"], "niche_id": [1, 2],
        "status": ["old", "old"], "status_oracle": ["older", "older"],
        "niche_id_oracle": [9, 9], "FORMAL_SELECTION_OCCURRED": ["OLD", "OLD"],
    })
    labeled = attach_oracle_metadata(selected, np.asarray([0.3, 0.4]), ledger)
    assert not labeled.columns.duplicated().any()
    assert "status_oracle" not in labeled and "niche_id_oracle" not in labeled
    assert labeled.status.tolist() == ["solved", "solved"]
    assert labeled.FORMAL_SELECTION_OCCURRED.tolist() == ["YES", "YES"]
