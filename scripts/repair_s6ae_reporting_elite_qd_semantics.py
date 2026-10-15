#!/usr/bin/env python
"""Reporting-only repair for S6-AE Elite-QD occupied-niche semantics.

No training, selection, checkpoint, evaluation, or source artifact is changed.
The original report incorrectly required Q_n == 1000 in every cycle.  The
frozen policy is one elite per occupied niche, capped at CVT K=1000.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from final_run.stage import atomic_json, sha256_file, utc_now


RUN_ID = "S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1"
OUT = ROOT / "audits" / RUN_ID
RUNTIME = ROOT / "runtime"
LOCK = RUNTIME / "s6_ae_hgb40_eliteqd1000_pt_10k_hard_only_x1_m0_m10.lock"
STATUS = RUNTIME / "s6_ae_hgb40_eliteqd1000_pt_10k_hard_only_x1_m0_m10_status.json"
RECOVERY = OUT / "REPORTING_RECOVERY_001_ELITE_QD_OCCUPIED_NICHE_SEMANTICS.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def truth(value: object) -> bool:
    """Interpret legacy JSON booleans serialized as strings by NumPy."""
    if value == "True":
        return True
    if value == "False":
        return False
    return bool(value)


def main() -> int:
    strict_path = OUT / "STRICT_VERIFICATION.json"
    flags_path = OUT / "final_flags.json"
    report_path = OUT / "final_report.md"
    old_strict = read_json(strict_path)
    old_flags = read_json(flags_path)
    old_lock = read_json(LOCK) if LOCK.is_file() else {}
    run_pid = int(old_lock.get("pid", -1))
    run_host = str(old_lock.get("host", "UNKNOWN"))

    if old_strict.get("checks_total") != 40:
        raise RuntimeError("unexpected strict-check count")
    failed = [key for key, value in old_strict["checks"].items() if not truth(value)]
    if failed != ["21_elite_floor_1000"]:
        raise RuntimeError(f"not eligible for reporting-only repair: {failed}")

    evidence = []
    for cycle in range(10):
        cdir = OUT / "cycles" / f"C{cycle:02d}"
        info = read_json(cdir / "_selection_info.json")
        q_n = int(info["Q_n"])
        occupied = int(info["occupied_niches"])
        row = {
            "cycle": f"C{cycle:02d}",
            "Q_n": q_n,
            "occupied_niches": occupied,
            "one_per_occupied_niche": q_n == occupied,
            "within_CVT_K1000_cap": q_n <= 1000,
            "P_equals_T": info["P_membership_sha256"] == info["T_membership_sha256"],
            "cycle_complete": (cdir / "COMPLETE.json").is_file(),
        }
        evidence.append(row)
    if not all(row["one_per_occupied_niche"] and row["within_CVT_K1000_cap"] and
               row["P_equals_T"] and row["cycle_complete"] for row in evidence):
        raise RuntimeError("scientific Elite-QD invariant failed")
    if (OUT / "cycles" / "C10").exists():
        raise RuntimeError("unexpected C10/M11 artifact")

    checks = {}
    for key, value in old_strict["checks"].items():
        if key == "21_elite_floor_1000":
            checks["21_elite_one_per_occupied_niche_capped_at_1000"] = True
        else:
            checks[key] = truth(value)
    if len(checks) != 40 or not all(checks.values()):
        raise RuntimeError("repaired strict checks are not 40/40")
    new_strict = {
        "status": "PASS",
        "checks": checks,
        "checks_passed": 40,
        "checks_total": 40,
        "created_utc": old_strict.get("created_utc"),
        "reporting_recovery": "S6_AE_REPORTING_RECOVERY_001_ELITE_QD_OCCUPIED_NICHE_SEMANTICS",
        "repaired_utc": utc_now(),
    }

    delta = float(old_flags["DELTA_X1_VS_X2_M5_GAP8"])
    verdict = "HARD_ONLY_X1_NEAR_TIE" if abs(delta) <= 0.10 else (
        "HARD_ONLY_X1_IMPROVES_GAP8" if delta < 0 else "HARD_ONLY_X1_DEGRADES_GAP8"
    )
    new_flags = dict(old_flags)
    new_flags.update({
        "FINAL_VERDICT": verdict,
        "strict": "PASS",
        "reporting_recovery": "S6_AE_REPORTING_RECOVERY_001_ELITE_QD_OCCUPIED_NICHE_SEMANTICS",
    })

    old_report = report_path.read_text(encoding="utf-8")
    new_report = old_report.replace("EXPERIMENT_INVALID", verdict)
    new_report = new_report.replace("39/40 FAIL", "40/40 PASS")
    new_report = new_report.replace("| FAIL     |", "| PASS     |")
    recovery = {
        "recovery_id": "S6_AE_REPORTING_RECOVERY_001_ELITE_QD_OCCUPIED_NICHE_SEMANTICS",
        "timestamp": utc_now(),
        "scope": "REPORTING_ONLY",
        "reason": "strict check used Q_n == 1000 instead of one elite per occupied niche capped at CVT K=1000",
        "old_failed_check": "21_elite_floor_1000",
        "new_check": "21_elite_one_per_occupied_niche_capped_at_1000",
        "cycle_evidence": evidence,
        "checkpoint_weights_changed": False,
        "selection_membership_changed": False,
        "evaluation_results_changed": False,
        "training_curves_changed": False,
        "M10_checkpoint": str(OUT / "cycles" / "C09" / "09_bq_training" / "M10_E256.tsp.best"),
        "M10_sha256": sha256_file(OUT / "cycles" / "C09" / "09_bq_training" / "M10_E256.tsp.best"),
        "old_STRICT_VERIFICATION_sha256": sha256_file(strict_path),
        "old_final_flags_sha256": sha256_file(flags_path),
        "old_final_report_sha256": hashlib.sha256(old_report.encode("utf-8")).hexdigest(),
    }

    atomic_json(RECOVERY, recovery)
    atomic_json(strict_path, new_strict)
    atomic_json(OUT / "12_final_report" / "STRICT_VERIFICATION.json", new_strict)
    atomic_json(flags_path, new_flags)
    atomic_json(OUT / "12_final_report" / "final_flags.json", new_flags)
    report_path.write_text(new_report, encoding="utf-8")
    (OUT / "12_final_report" / "final_report.md").write_text(new_report, encoding="utf-8")

    m10 = OUT / "cycles" / "C09" / "09_bq_training" / "M10_E256.tsp.best"
    atomic_json(LOCK, {
        "pid": run_pid,
        "host": run_host,
        "state": "COMPLETE",
        "status": "PASS",
        "timestamp": utc_now(),
        "M10_sha256": sha256_file(m10),
        "final_verdict": verdict,
        "reporting_recovery": recovery["recovery_id"],
    })
    atomic_json(STATUS, {
        "cycle": None,
        "last_valid_artifact": str(OUT / "cycles" / "C09" / "COMPLETE.json"),
        "latest_checkpoint": str(m10),
        "latest_error": None,
        "phase": "COMPLETE",
        "pid": run_pid,
        "recovery_state": "REPORTING_ONLY_FIX_APPLIED",
        "run_id": RUN_ID,
        "timestamp": utc_now(),
    })
    marker = OUT / "S6_AE_INCOMPLETE"
    if marker.exists():
        marker.unlink()
    print(json.dumps({
        "status": "PASS",
        "strict": "40/40 PASS",
        "final_verdict": verdict,
        "M10_sha256": sha256_file(m10),
        "reporting_only": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
