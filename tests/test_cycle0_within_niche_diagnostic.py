import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audits/cycle0_within_niche_hardness_diagnostic_v1"


def load(rel):
    return json.loads((AUDIT / rel).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_protocol_is_analysis_only_and_d5_primary():
    p = load("00_protocol/analysis_protocol.json")
    assert p["analysis_only"] and p["primary_diagnostic_branch"] == "D5"
    assert p["sensitivity_branches"] == ["D1", "D2"]


def test_prediction_and_label_alignment():
    p = load("01_integrity/prediction_alignment.json")
    h = load("01_integrity/label_alignment.json")
    assert p["status"] == h["status"] == "PASS"
    assert p["dev_n"] == h["dev_n"] == 2666
    assert set(p["per_model_rows"].values()) == {2666}


def test_variance_decomposition_is_exact():
    d = load("02_hardness_decomposition/variance_decomposition.json")
    assert abs(d["component_fraction_sum"] - 1) < 1e-12
    assert abs(d["additive_reconstruction_error"]) < 1e-12
    assert 0 <= d["between_niche_variance_fraction"] <= 1


def test_centered_signal_has_zero_niche_means():
    c = pd.read_csv(AUDIT / "02_hardness_decomposition/centered_signal.csv")
    assert len(c) == 9 * 2666
    assert c.groupby(["model", "niche_id"]).H_centered.mean().abs().max() < 1e-12
    assert c.groupby(["model", "niche_id"]).P_centered.mean().abs().max() < 1e-12


def test_oracle_headroom_is_exact_and_positive():
    d = pd.read_parquet(AUDIT / "03_oracle_headroom/per_niche_oracle_random.parquet")
    assert len(d) == 262 * 3
    assert np.allclose(d.absolute_headroom, d.oracle_H0 - d.random_H0)
    assert (d.groupby("r").absolute_headroom.mean() > 0).all()


def test_all_model_metric_outputs_are_complete():
    assert len(pd.read_csv(AUDIT / "04_model_global/global_metrics.csv")) == 9
    assert len(pd.read_csv(AUDIT / "04_model_global/between_niche_metrics.csv")) == 9
    assert len(pd.read_csv(AUDIT / "04_model_global/centered_metrics.csv")) == 9
    assert len(pd.read_parquet(AUDIT / "05_model_within_niche/per_niche_metrics.parquet")) == 9 * 262


def test_acquisition_outputs_are_complete_and_unclipped():
    for r in (1, 2, 5):
        d = pd.read_parquet(AUDIT / f"06_acquisition/per_niche_top{r}.parquet")
        assert len(d) == 9 * 262 and set(d.r) == {r}
    c = pd.read_csv(AUDIT / "06_acquisition/headroom_capture.csv")
    assert len(c) == 27
    assert (c.unclipped_min_per_niche < 0).any() or (c.unclipped_max_per_niche > 1).any()


def test_paired_comparisons_and_secondary_control():
    f = pd.read_csv(AUDIT / "07_feature_bottleneck/hgb20_vs_hgb4.csv")
    s = pd.read_csv(AUDIT / "08_symbolic_capacity/hgb4_vs_gp4.csv")
    x = pd.read_csv(AUDIT / "08_symbolic_capacity/hgb20_vs_gp4_secondary.csv")
    assert len(f) == len(s) == len(x) == 27
    assert set(f.depth) == set(s.depth) == {"D1", "D2", "D5"}


def test_geometry_strata_and_agreement_exist():
    g = pd.read_csv(AUDIT / "09_niche_geometry/geometry_rankability.csv")
    s = pd.read_csv(AUDIT / "09_niche_geometry/sample_size_strata.csv")
    a = pd.read_csv(AUDIT / "10_model_agreement/topk_agreement.csv")
    assert len(g) == 262 and not s.empty and len(a) == 9


def test_required_figures_exactly_present():
    expected = {
        "01_true_hardness_between_vs_within_variance.png", "02_between_vs_within_model_correlation.png",
        "03_oracle_random_model_top1.png", "04_oracle_random_model_top5.png", "05_headroom_capture_top1.png",
        "06_headroom_capture_top2.png", "07_headroom_capture_top5.png", "08_gp4_hgb4_hgb20_within_niche_rho.png",
        "09_pairwise_ranking_accuracy.png", "10_hgb20_minus_hgb4_acquisition.png",
        "11_hgb4_minus_gp4_acquisition.png", "12_niche_spread_vs_true_hardness_range.png",
        "13_niche_spread_vs_model_rankability.png", "14_model_top1_agreement.png",
    }
    assert {p.name for p in (AUDIT / "12_figures").glob("*.png")} == expected
    assert all((AUDIT / "12_figures" / name).stat().st_size > 1000 for name in expected)


def test_hash_inventory_covers_every_other_output():
    inventory = load("00_protocol/generated_file_hashes.json")["files"]
    actual = {p.relative_to(AUDIT).as_posix() for p in AUDIT.rglob("*") if p.is_file() and p.name != "generated_file_hashes.json"}
    assert set(inventory) == actual
    assert all(digest(AUDIT / rel) == meta["sha256"] for rel, meta in inventory.items())


def test_strict_verification_and_forbidden_boundaries():
    v = load("strict_verification.json")
    assert v["status"] == "PASS" and v["passed"] == v["total"] == 44
    assert not any("within_niche_diagnostic" in path.name for path in (ROOT / "runs").rglob("*"))
    assert load("11_decision/diagnostic_classification.json")["classification"] in {f"D{i}" for i in range(1, 8)}
