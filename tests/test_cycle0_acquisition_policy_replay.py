import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audits/cycle0_acquisition_policy_replay_v1"
POLICIES = {"GLOBAL_GP", "ROUND_ROBIN_GP", "CENTROID1_GLOBAL_GP", "CENTROID1_RANDOM_FILL", "GLOBAL_RANDOM"}


def load(rel):
    return json.loads((AUDIT / rel).read_text(encoding="utf-8"))


def test_replay_protocol_has_no_scalarization_or_mutation():
    p = load("00_protocol/replay_protocol.json")
    assert p["analysis_only"] and p["scalarization"] is None and p["lambda_diversity"] is None
    assert p["full_bank_n"] == 109785 and p["full_bank_budget"] == 10000


def test_full_bank_and_prediction_alignment():
    c = load("01_integrity/candidate_alignment.json")
    p = load("01_integrity/prediction_alignment.json")
    assert c["status"] == p["status"] == "PASS"
    assert c["full_bank_n"] == 109785 and c["occupied_niches"] == 435
    assert p["formula_count"] == 5 and p["dev_predictions_exact_bitwise"]


def test_full_bank_scores_are_finite_and_unique():
    d = pd.read_parquet(AUDIT / "02_full_bank_scores/gp4_d5_scores.parquet")
    assert len(d) == d.instance_hash.nunique() == 109785
    assert np.isfinite(d.gp4_d5_score).all()
    assert len([c for c in d if c.startswith("gp4_d5_seed_")]) == 5


def test_all_full_bank_policies_have_exact_budget():
    files = ["global_gp", "round_robin_gp", "centroid1_global_gp", "centroid1_random_fill", "global_random"]
    for name in files:
        d = pd.read_parquet(AUDIT / f"03_full_bank_policy_replay/{name}.parquet")
        assert len(d) == d.instance_hash.nunique() == 10000


def test_centroid_floor_is_complete_and_fill_disjoint():
    d = pd.read_parquet(AUDIT / "03_full_bank_policy_replay/centroid1_global_gp.parquet")
    floor = d[d.selection_component.eq("centroid1_floor")]
    fill = d[d.selection_component.eq("global_fill")]
    assert len(floor) == floor.niche_id.nunique() == 435 and len(fill) == 9565
    assert set(floor.instance_hash).isdisjoint(set(fill.instance_hash))


def test_full_structural_lineage_and_operator_outputs():
    s = pd.read_csv(AUDIT / "04_full_bank_structural_metrics/policy_summary.csv")
    l = pd.read_csv(AUDIT / "09_lineage_operator_diversity/lineage_summary.csv")
    o = pd.read_csv(AUDIT / "09_lineage_operator_diversity/operator_summary.csv")
    assert set(s.policy) == set(l.policy) == POLICIES
    assert l.operator_count.between(1, 11).all() and l.all_11_operators_represented.dtype == bool
    assert set(o[o.policy.isin(POLICIES)].operator_id).issuperset(set(o[o.policy.eq("CENTROID1_FLOOR")].operator_id))


def test_dev_panel_budgets_and_oracles_are_sealed():
    a = load("01_integrity/dev_label_alignment.json")
    s = pd.read_csv(AUDIT / "05_dev_policy_replay/policy_budget_summary.csv")
    assert a["dev_n"] == 2666 and a["dev_occupied_niches"] == 301
    assert sorted(s.budget.unique()) == [301, 602, 1204, 2408]
    assert set(s.policy) == POLICIES | {"GLOBAL_ORACLE", "CENTROID1_GLOBAL_ORACLE"}


def test_random_repeats_and_bootstrap_are_exact():
    r = pd.read_parquet(AUDIT / "05_dev_policy_replay/random_repeat_metrics.parquet")
    assert set(r.groupby(["budget", "policy"]).size()) == {1000}
    b = pd.read_csv(AUDIT / "06_dev_true_hardness/bootstrap.csv")
    assert set(b.seed) == {2026}
    assert set(b[b.method.eq("paired_niche_cluster_bootstrap")].replicates) == {10000}


def test_oracle_regret_identity_is_exact():
    d = pd.read_csv(AUDIT / "07_oracle_decomposition/regret_decomposition.csv")
    assert np.max(np.abs(d.identity_error)) < 1e-14
    assert np.allclose(d.total_hybrid_regret, d.coverage_constraint_cost + d.surrogate_loss_hybrid)


def test_pareto_and_tradeoff_outputs_cover_all_budgets():
    p = pd.read_csv(AUDIT / "08_policy_tradeoff/hardness_coverage_pareto.csv")
    assert len(p) == 4 * 7 and set(p.policy) == POLICIES | {"GLOBAL_ORACLE", "CENTROID1_GLOBAL_ORACLE"}
    for name in ("hybrid_vs_global", "hybrid_vs_round_robin", "hybrid_vs_random_fill"):
        assert len(pd.read_csv(AUDIT / f"08_policy_tradeoff/{name}.csv")) == 4


def test_required_figures_and_hashes():
    figures = list((AUDIT / "12_figures").glob("*.png"))
    assert len(figures) == 13 and all(p.stat().st_size > 1000 for p in figures)
    inventory = load("00_protocol/generated_file_hashes.json")["files"]
    actual = {p.relative_to(AUDIT).as_posix() for p in AUDIT.rglob("*") if p.is_file() and p.name != "generated_file_hashes.json"}
    assert set(inventory) == actual
    for rel, meta in inventory.items():
        assert hashlib.sha256((AUDIT / rel).read_bytes()).hexdigest() == meta["sha256"]


def test_strict_verification_and_forbidden_boundaries():
    v = load("strict_verification.json")
    d = load("11_decision/diagnostic_classification.json")
    assert v["status"] == "PASS" and v["passed"] == v["total"] == 54
    assert d["classification"] in {f"A{i}" for i in range(1, 7)} and not d["protocol_automatically_modified"]
    assert not any("acquisition_policy_replay" in path.name for path in (ROOT / "runs").rglob("*"))
    assert not any("final_d0" in p.name.lower() for p in AUDIT.rglob("*"))
