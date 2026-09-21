import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/"audits/gp0_calibration_depth_final_acquisition_v1"


def load(rel): return json.loads((AUDIT/rel).read_text(encoding="utf-8"))


def test_protocol_role_and_fixed_noninferiority():
    p=load("00_protocol/protocol.json")
    assert p["analysis_only"] and p["primary_dev_budget"]==602
    assert p["noninferiority"]["retention_threshold"]==.95
    assert p["noninferiority"]["margin_fraction_of_abs_D5_gain"]==.05


def test_all_existing_formula_ensembles_replay_exactly():
    d=load("01_integrity/model_manifest_alignment.json")
    assert d["status"]=="PASS" and d["all_five_seed_ensembles_exact"]
    assert [x["train_n"] for x in d["branches"]]==[435,822,1819]
    assert all(len(x["seeds"])==5 and x["full_predictions_finite"] for x in d["branches"])


def test_dev_and_full_bank_alignment():
    d=load("01_integrity/dev_alignment.json");f=load("01_integrity/full_bank_alignment.json")
    assert d["dev_n"]==2666 and d["occupied_niches"]==301 and d["shared_centroid_floor_n"]==301
    assert f["full_bank_n"]==109785 and f["occupied_niches"]==435 and f["shared_centroid_floor_n"]==435


def test_operating_proxy_is_exact():
    o=load("02_dev_operating_proxy/operating_ratio.json")
    assert o["primary_budget"]==602 and o["primary_GP_fill"]==301
    assert np.isclose(o["primary_GP_fill_fraction"],301/2365)
    assert np.isclose(o["full_GP_fill_fraction"],9565/109350)


def test_dev_selections_share_floor_and_exact_sizes():
    s=pd.read_parquet(AUDIT/"03_dev_policy_replay/selections.parquet")
    for budget in (602,1204,2408):
        for model in ("GP4_D1","GP4_D2","GP4_D5"):
            g=s[(s.budget.eq(budget))&s.model.eq(model)]
            assert len(g)==budget and g.instance_hash.nunique()==budget
            assert len(g[g.component.eq("centroid1_floor")])==301
    floors=[set(g.instance_hash) for _,g in s[s.component.eq("centroid1_floor")].groupby(["budget","model"])]
    assert all(x==floors[0] for x in floors)


def test_random_baseline_is_shared_and_has_1000_repeats():
    r=pd.read_csv(AUDIT/"04_gp_fill_value/random_baseline.csv")
    assert set(r.groupby("budget").size())=={1000} and sorted(r.budget.unique())==[602,1204,2408]
    f=pd.read_csv(AUDIT/"04_gp_fill_value/fill_summary.csv")
    assert f.groupby("budget").random_fill_H0.nunique().eq(1).all()


def test_bootstrap_is_niche_cluster_and_complete():
    b=pd.read_csv(AUDIT/"04_gp_fill_value/bootstrap.csv")
    assert set(b.unit)=={"niche cluster"} and set(b.replicates)=={10000} and set(b.seed)=={2026}
    assert len(b)==3*3*4


def test_noninferiority_outputs_are_consistent():
    n=pd.read_csv(AUDIT/"05_calibration_noninferiority/noninferiority_summary.csv")
    assert set(n.depth)=={1,2,5} and set(n.budget)=={602}
    for row in n.itertuples():
        expected=row.A_positive_gain and row.B_gain_ci_nonnegative and row.C_retention_at_least_0_95 and row.D_delta_lower_above_margin and row.numerically_stable
        assert bool(row.operationally_saturated_relative_to_D5)==bool(expected)


def test_full_bank_replays_have_exact_floor_and_fill():
    for d in (1,2,5):
        s=pd.read_parquet(AUDIT/f"06_full_bank_replay/centroid1_global_gp_d{d}.parquet")
        assert len(s)==s.instance_hash.nunique()==10000
        assert len(s[s.component.eq("centroid1_floor")])==435
        assert len(s[s.component.eq("gp_fill")])==9565


def test_full_bank_stability_outputs():
    o=pd.read_csv(AUDIT/"07_full_bank_selection_stability/overlap_summary.csv")
    r=pd.read_csv(AUDIT/"07_full_bank_selection_stability/rank_agreement.csv")
    c=pd.read_csv(AUDIT/"07_full_bank_selection_stability/cross_scoring_matrix.csv")
    assert len(o)==len(r)==len(c)==3 and c.diagnostic_only.all()
    assert r.spearman.between(-1,1).all()


def test_figures_hash_inventory_and_decision():
    assert len(list((AUDIT/"10_figures").glob("*.png")))==10
    inventory=load("00_protocol/generated_file_hashes.json")["files"]
    actual={p.relative_to(AUDIT).as_posix() for p in AUDIT.rglob("*") if p.is_file() and p.name!="generated_file_hashes.json"}
    assert set(inventory)==actual
    for rel,meta in inventory.items():assert hashlib.sha256((AUDIT/rel).read_bytes()).hexdigest()==meta["sha256"]
    d=load("09_decision/final_classification.json")
    assert d["classification"] in {f"C{i}" for i in range(1,7)} and not d["protocol_automatically_modified"]


def test_strict_and_forbidden_boundaries():
    v=load("strict_verification.json")
    assert v["status"]=="PASS" and v["passed"]==v["total"]==57
    assert not any("calibration_depth" in path.name for path in (ROOT/"runs").rglob("*"))
    assert not any("final_d0" in p.name.lower() for p in AUDIT.rglob("*"))
