import json
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/"audits/cycle0_initialization_validation_v1"

def read(name): return json.loads((AUDIT/name).read_text())

def test_phase_a_replay_and_classification():
    d=read("01_repair_semantics/phase_A_decision.json")
    assert d["classification"] in ("A1","A2") and d["replayed_children"]==110000
    assert d["raw_hashes_match"] and d["final_hashes_match"]

def test_repair_rows_and_operator_table():
    assert len(pd.read_parquet(AUDIT/"01_repair_semantics/repair_instance_metrics.parquet"))==38093
    assert len(pd.read_csv(AUDIT/"01_repair_semantics/repair_operator_summary.csv"))==11

def test_calibration_nesting():
    sets=[]
    for d in (1,2,5): sets.append({x["instance_hash"] for x in read(f"02_gp0_calibration_design/cal_d{d}_manifest.json")["members"]})
    assert sets[0] <= sets[1] <= sets[2]

def test_dev_is_sealed_and_disjoint():
    dev=set(pd.read_parquet(AUDIT/"02_gp0_calibration_design/cycle0_gp_dev.parquet").instance_hash)
    d5={x["instance_hash"] for x in read("02_gp0_calibration_design/cal_d5_manifest.json")["members"]}
    assert read("02_gp0_calibration_design/PRE_ORACLE_SEAL.json")["sealed"] and not dev & d5

def test_exact_labels():
    d=read("03_oracle_labels/label_manifest.json"); frame=pd.read_parquet(AUDIT/"03_oracle_labels/exact_hardness.parquet")
    assert d["status"]=="EXACT_LABELING_COMPLETE" and len(frame)==d["deduplicated_union_n"]
    assert frame.H0.notna().all() and frame.H0_log1p.notna().all()

def test_gp_five_seed_ensembles():
    for directory in ("04_gp4_d1","05_gp4_d2","06_gp4_d5"):
        d=read(f"{directory}/ensemble.json")
        assert d["finite"] and not d["constant"] and len(d["formulas"])==5

def test_hgb_controls():
    for name in ("hgb4","hgb20"):
        for d in (1,2,5): assert (AUDIT/f"07_hgb_controls/{name}_d{d}.joblib").is_file()

def test_prediction_outputs():
    assert len(pd.read_csv(AUDIT/"08_prediction_evaluation/global_metrics.csv"))==9
    assert len(pd.read_csv(AUDIT/"08_prediction_evaluation/tail_metrics.csv"))==9

def test_within_niche_outputs():
    assert not pd.read_parquet(AUDIT/"09_within_niche_ranking/per_niche_metrics.parquet").empty
    assert len(pd.read_csv(AUDIT/"09_within_niche_ranking/summary.csv"))==9

def test_acquisition_and_bootstrap():
    for r in (1,2,5): assert not pd.read_csv(AUDIT/f"10_acquisition_evaluation/top{r}.csv").empty
    b=pd.read_csv(AUDIT/"10_acquisition_evaluation/bootstrap.csv")
    assert set(b.replicates)=={10000} and set(b.seed)=={2026}

def test_decision_and_figures():
    assert read("11_decision/phase_B_final.json")["classification"] in ("B1","B2","B3","B4")
    assert len(list((AUDIT/"12_figures").glob("*.png")))==10

def test_forbidden_boundaries():
    assert not any("initialization_validation" in path.name for path in (ROOT/"runs").rglob("*"))
    labels=pd.read_parquet(AUDIT/"03_oracle_labels/exact_hardness.parquet")
    assert "D0" not in set(labels.oracle_role)
