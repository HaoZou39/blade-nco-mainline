#!/usr/bin/env python
"""Strict final validation and checksum manifest for DT1-A1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
WORK = OUT / "work"
A0 = ROOT / "audits" / "dt1a_operator_hardness_preflight_v1"
M5_SHA = "ef07226bb8c3e8cd3a1404eece474df44d378316fd66509b762eb5f8bc26558c"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    required = [
        "01_protocol_and_integrity.md", "02_30parent_sample_manifest.csv",
        "03_true_lbq_evaluation_manifest.csv", "04_operator_paired_results.csv",
        "05_operator_comparison.md", "06_structural_alignment.md", "07_selection_retention.md",
        "08_actual_compute_audit.md", "09_theoretical_implications.md", "10_final_decision.md",
    ]
    checks: dict[str, bool] = {"deliverables_present": all((OUT / name).is_file() for name in required)}
    preflight = json.loads((WORK / "preflight_gate.json").read_text(encoding="utf-8"))
    concorde = json.loads((WORK / "concorde_timing.json").read_text(encoding="utf-8"))
    lbq = json.loads((WORK / "lbq_timing.json").read_text(encoding="utf-8"))
    analysis = json.loads((WORK / "analysis_summary.json").read_text(encoding="utf-8"))
    sample = pd.read_csv(OUT / "02_30parent_sample_manifest.csv")
    evaluation = pd.read_csv(OUT / "03_true_lbq_evaluation_manifest.csv")
    operators = pd.read_csv(OUT / "04_operator_paired_results.csv")
    paired = pd.read_csv(WORK / "paired_attempts.csv")
    new_labels = duckdb.execute("select * from read_parquet(?)", [str(WORK / "new_M5_lbq_labels.parquet")]).df()
    new_tours = duckdb.execute("select * from read_parquet(?)", [str(WORK / "new_concorde_ledger.parquet")]).df()
    pairwise = pd.read_csv(WORK / "pairwise_operator_contrasts.csv")

    checks.update({
        "preflight_pass": preflight["status"] == "PASS",
        "checkpoint_exact": preflight["checkpoint_sha256"] == M5_SHA == lbq["checkpoint"]["sha256"],
        "sample_30_unique": len(sample) == 30 and sample.instance_hash.nunique() == 30,
        "sample_seed_frozen": sample.sampling_seed.eq(20261027).all(),
        "eleven_attempt_ids_each": sample.operator_attempt_count.eq(11).all(),
        "evaluation_360_rows": len(evaluation) == 360,
        "evaluation_roles": int((evaluation.entity_role == "parent").sum()) == 30 and int((evaluation.entity_role == "child_attempt").sum()) == 330,
        "evaluation_valid_356": int(evaluation.evaluable_candidate.sum()) == 356,
        "evaluation_invalid_4": int((~evaluation.evaluable_candidate).sum()) == 4,
        "reused_labels_20": int(evaluation.label_reused.sum()) == 20,
        "new_labels_336": int(evaluation.label_new.sum()) == 336 and len(new_labels) == 336,
        "valid_labels_complete": evaluation.loc[evaluation.evaluable_candidate, "M5_true_LBQ"].notna().all(),
        "invalid_unlabeled": evaluation.loc[~evaluation.evaluable_candidate, "M5_true_LBQ"].isna().all(),
        "raw_log_separate_exact": np.allclose(
            evaluation.loc[evaluation.evaluable_candidate, "M5_log1p_LBQ"],
            np.log1p(evaluation.loc[evaluation.evaluable_candidate, "M5_true_LBQ"]), atol=1e-12),
        "concorde_312_verified": len(new_tours) == 312 and bool(new_tours.verified.all()) and concorde["failures"] == 0,
        "concorde_identity": new_tours.binary_hash.eq("38a647d3f04196231c725e90d23c2c883c0a23899f9bab8a257ae6f777d1dec0").all(),
        "lbq_no_updates": lbq["parameter_updates"] == 0 and lbq["teacher_forced_protocol"]["parameter_updates"] == 0,
        "lbq_no_failures": lbq["failures"] == 0 and np.isfinite(new_labels.M5_true_LBQ).all(),
        "operator_rows_11": len(operators) == 11 and operators.operator_id.nunique() == 11,
        "operator_scheduled_30": operators.scheduled_attempts.eq(30).all(),
        "paired_330": len(paired) == 330 and int(paired.evaluable_child.sum()) == 326,
        "pairwise_55": len(pairwise) == 55,
        "analysis_counts": analysis["parents"] == 30 and analysis["scheduled_attempts"] == 330 and analysis["valid_unique_children"] == 326,
        "final_decision_A": "A — PILOT_VALID_EXPAND_TO_300" in (OUT / "10_final_decision.md").read_text(encoding="utf-8"),
    })

    # Reused label values must be byte-source-equivalent to DT1-A0's recorded values.
    a0_availability = pd.read_csv(A0 / "05_current_checkpoint_label_availability.csv")
    reused = evaluation[evaluation.label_reused][["entity_hash", "M5_true_LBQ"]].merge(
        a0_availability[["entity_hash", "M5_true_LBQ"]].dropna().drop_duplicates("entity_hash"),
        on="entity_hash", suffixes=("_pilot", "_a0"), validate="one_to_one")
    checks["reused_values_exact"] = len(reused) == 20 and np.array_equal(
        reused.M5_true_LBQ_pilot.to_numpy(float), reused.M5_true_LBQ_a0.to_numpy(float))
    checks = {name: bool(value) for name, value in checks.items()}

    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError(f"DT1-A1 strict validation failed: {failed}")
    verification = {
        "status": "PASS", "checks": checks,
        "prohibited_actions": {
            "mutation_rerun": False, "BQ_training": False, "HGB_retraining": False,
            "evolutionary_cycle": False, "M0_or_M6_paired_eval": False,
            "SVM_training": False, "Lyapunov_theorem": False, "expanded_to_300": False,
        },
    }
    (OUT / "STRICT_VERIFICATION.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    inventory_names = required + [
        "run_pilot.py", "analyze_pilot.py", "finalize_audit.py", "STRICT_VERIFICATION.json",
        "work/preflight_gate.json", "work/evaluation_base.csv", "work/pilot_evaluable_coordinates.npz",
        "work/new_concorde_ledger.parquet", "work/pilot_tours.parquet", "work/concorde_timing.json",
        "work/new_M5_lbq_labels.parquet", "work/lbq_timing.json", "work/paired_attempts.csv",
        "work/pairwise_operator_contrasts.csv", "work/omnibus_tests.json",
        "work/structural_correlations.csv", "work/structural_fixed_effect_model.json",
        "work/qd_region_descriptive.csv", "work/selection_retention_summary.csv", "work/analysis_summary.json",
        "work/concorde_failures.csv",
    ]
    files = []
    for name in inventory_names:
        path = OUT / name
        if path.is_file():
            files.append({"path": name, "bytes": path.stat().st_size, "sha256": sha(path)})
    manifest = {
        "schema": "DT1A1_FIXED_M5_PAIRED_PILOT_V1", "status": "PASS",
        "checkpoint_sha256": M5_SHA, "files": files,
    }
    (OUT / "AUDIT_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
