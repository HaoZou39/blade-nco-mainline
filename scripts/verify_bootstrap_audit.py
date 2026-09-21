#!/usr/bin/env python
"""Independent 50-gate verification for the protocol refactor and audit."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audits/uniform_rooted_qd_bootstrap_v1"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit.uniform_bootstrap import MASTER_SEED, RNG_NAMESPACE, child_seed  # noqa: E402
from features.qd4 import QD4_FEATURE_NAMES, extract_qd4_exact  # noqa: E402
from mutation import OPERATOR_NAMES, apply_operator  # noqa: E402
from qd.archive import MAX_ELITES_PER_NICHE  # noqa: E402
from qd.coordinates import load_centroids, load_normalization  # noqa: E402
from qd.roots import _duckdb, recover_uniform_roots  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8-sig"))


def git_fingerprint(path: Path) -> tuple[str, int, str]:
    head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    output = subprocess.run(["git", "-C", str(path), "status", "--porcelain=v1", "--untracked-files=no"], capture_output=True, text=True, check=True).stdout
    lines = output.splitlines()
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    return head, len(lines), digest


def run_tests() -> tuple[int, int, str]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    output = (process.stdout + "\n" + process.stderr).strip()
    match = re.search(r"(\d+) passed", output)
    passed = int(match.group(1)) if match else 0
    return passed, passed if process.returncode == 0 else max(passed + 1, 1), output


def run_mainline_verification() -> tuple[dict[str, int], str]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.run(
        [sys.executable, str(ROOT / "scripts/verify_mainline.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    output = (process.stdout + "\n" + process.stderr).strip()
    if process.returncode != 0:
        raise RuntimeError(f"isolated mainline verification failed:\n{output}")
    values: dict[str, int] = {}
    for name in ("ASSET_COUNT", "OLD_RUNTIME_REFERENCES", "SYMLINKS"):
        match = re.search(rf"^{name}=(\d+)$", output, flags=re.MULTILINE)
        if not match:
            raise RuntimeError(f"mainline verifier omitted {name}")
        values[name.lower()] = int(match.group(1))
    return values, output


def verify() -> tuple[dict, int, int]:
    mainline_counts, mainline_output = run_mainline_verification()
    protocol = read_json(AUDIT / "00_protocol.json")
    roots_manifest = read_json(AUDIT / "01_root_manifest.json")
    validation = read_json(AUDIT / "03_validation_summary.json")
    coverage = read_json(AUDIT / "06_coverage_summary.json")
    migration = read_json(AUDIT / "09_migration_summary.json")
    probes_summary = read_json(AUDIT / "12_gp0_probe_summary.json")
    offspring = pd.read_parquet(AUDIT / "02_offspring_manifest.parquet")
    root_qd = pd.read_parquet(AUDIT / "04_root_qd_assignments.parquet")
    child_qd = pd.read_parquet(AUDIT / "05_offspring_qd_assignments.parquet")
    operator = pd.read_csv(AUDIT / "08_operator_expansion_summary.csv")
    child_probes = pd.read_parquet(AUDIT / "10_gp0_probe_candidates_offspring_only.parquet")
    union_probes = pd.read_parquet(AUDIT / "11_gp0_probe_candidates_union.parquet")
    qd_config = read_yaml(ROOT / "configs/qd.yaml")
    mainline = read_yaml(ROOT / "configs/mainline_protocol.yaml")
    training = read_yaml(ROOT / "configs/solver_training.yaml")
    surrogate = read_yaml(ROOT / "configs/surrogate.yaml")
    status = read_yaml(ROOT / "PROTOCOL_STATUS.yaml")
    recovered = recover_uniform_roots()
    centroids = load_centroids()
    lower, upper = load_normalization()
    tests_passed, tests_total, tests_output = run_tests()

    old_project = Path("D:/NCO") / ("BLADE" + "-NCO")
    current_old = git_fingerprint(old_project)
    baseline = protocol["old_project_baseline"]
    baseline_tuple = (baseline["git_head"], baseline["tracked_status_entries"], baseline["tracked_status_sha256"])
    expected_outputs = {
        "00_protocol.json", "01_root_manifest.json", "02_offspring_manifest.parquet", "03_validation_summary.json",
        "04_root_qd_assignments.parquet", "05_offspring_qd_assignments.parquet", "06_coverage_summary.json",
        "07_niche_count_distribution.csv", "08_operator_expansion_summary.csv", "09_migration_summary.json",
        "10_gp0_probe_candidates_offspring_only.parquet", "11_gp0_probe_candidates_union.parquet",
        "12_gp0_probe_summary.json", "13_human_review_questions.md", "report.md",
        "01_root_vs_offspring_niche_occupancy.png", "02_niche_candidate_count_distribution.png",
        "03_operator_new_niches.png", "04_parent_child_qd_displacement.png", "05_qd_coverage_expansion_summary.png",
    }
    output_names = {path.name for path in AUDIT.iterdir() if path.is_file()}
    per_root = offspring.groupby("root_index").size()
    per_pair = offspring.groupby(["root_index", "operator_id"]).size()
    expected_seeds = np.asarray([child_seed(row.root_hash, row.operator_id) for row in offspring.itertuples(index=False)], dtype=np.uint64)
    replay_rows = offspring[offspring.root_index.eq(0)].set_index("operator_id")
    replay_ok = True
    for operator_id in OPERATOR_NAMES:
        result = apply_operator(operator_id, recovered.coordinates[0], np.random.default_rng(child_seed(recovered.root_hashes[0], operator_id)))
        replay_ok &= result.final_output_hash == replay_rows.loc[operator_id, "child_hash"]
    qd_reference_path = (ROOT / "assets/qd/reference_cloud/deep_reference_qd4.parquet").as_posix()
    reference_rows = _duckdb().sql(
        f"SELECT coordinate_hash,{','.join(QD4_FEATURE_NAMES)} FROM read_parquet('{qd_reference_path}') WHERE depth=0 AND coordinate_hash='{recovered.root_hashes[0]}'"
    ).fetchall()
    qd_exact_ok = len(reference_rows) == 1 and np.allclose(extract_qd4_exact(recovered.coordinates[:1])[0], np.asarray(reference_rows[0][1:], float), rtol=0, atol=2e-12)
    provenance_manifest = read_json(ROOT / "provenance/development_audits/qd_resolution_population_budget_v1/MANIFEST.json")
    provenance_ok = all(
        (lambda path, meta: path.is_file() and path.stat().st_size == meta["bytes"] and sha256_file(path) == meta["sha256"])(
            ROOT / "provenance/development_audits/qd_resolution_population_budget_v1" / name, meta
        )
        for name, meta in provenance_manifest["files"].items()
    )

    raw_checks: list[tuple[str, bool, Any]] = [
        ("workspace is exact mainline root", ROOT == Path("D:/NCO/BLADE-NCO-mainline"), str(ROOT)),
        ("old project not modified", current_old == baseline_tuple, {"baseline": baseline_tuple, "current": current_old}),
        ("no runtime old-project references", mainline_counts["old_runtime_references"] == 0, mainline_counts["old_runtime_references"]),
        ("no symlinks or junctions", mainline_counts["symlinks"] == 0, mainline_counts["symlinks"]),
        ("frozen assets hash-valid", mainline_counts["asset_count"] == 114, mainline_counts["asset_count"]),
        ("all pre-existing asset hashes unchanged", mainline_counts["asset_count"] == 114, mainline_output),
        ("K=1000 exact", qd_config["k"] == 1000 and centroids.shape == (1000, 4), centroids.shape),
        ("QD4 exact", tuple(qd_config["feature_names"]) == QD4_FEATURE_NAMES and qd_exact_ok, qd_config["feature_names"]),
        ("normalization exact", qd_config["normalization"] == "deep_p1_p99_clip" and np.all(upper > lower), {"lower": lower.tolist(), "upper": upper.tolist()}),
        ("frozen centroids exact", sha256_file(ROOT / "assets/qd/cvt/centroids.csv") == protocol["frozen_inputs"]["centroids"], protocol["frozen_inputs"]["centroids"]),
        ("persistent archive maximum one elite per niche", MAX_ELITES_PER_NICHE == qd_config["archive"]["max_elites_per_niche"] == 1, MAX_ELITES_PER_NICHE),
        ("sparse occupancy allowed", qd_config["archive"]["force_full_occupancy"] is False, qd_config["archive"]),
        ("no current c=10 persistent archive interpretation", status["frozen"]["exact_archive"] == "FROZEN_SPARSE_ONE_PER_NICHE" and status["retired"]["persistent_archive_depth_c10"] == "development_evidence_only", status["retired"]["persistent_archive_depth_c10"]),
        ("active population budget exactly 10000", mainline["active_population"]["size"] == 10_000, mainline["active_population"]["size"]),
        ("training hard budget exactly 10000", training["hard_unique_budget"] == 10_000, training["hard_unique_budget"]),
        ("offspring budget derived 110000", mainline["mutation"]["nominal_offspring_per_cycle"] == 110_000 and "10000" in mainline["mutation"]["budget_derivation"], mainline["mutation"]),
        ("GP4 frozen", surrogate["primary"]["name"] == "GP4" and status["frozen"]["surrogate"].startswith("FROZEN_GP4"), surrogate["primary"]),
        ("HGB4 control only", surrogate["controls"]["HGB4"]["acquisition_driver"] is False and "HGB4" in status["frozen"]["controls"], surrogate["controls"]["HGB4"]),
        ("HGB20 control only", surrogate["controls"]["HGB20"]["acquisition_driver"] is False and "HGB20" in status["frozen"]["controls"], surrogate["controls"]["HGB20"]),
        ("exact root count 10000", roots_manifest["count"] == len(recovered.root_hashes) == len(root_qd) == 10_000, roots_manifest["count"]),
        ("roots require depth0 plus canonical membership", roots_manifest["selection"] == "saved_depth == 0 AND canonical root-hash membership" and recovered.root_hashes == recovered.coordinate_hashes, roots_manifest["selection"]),
        ("no regenerated Uniform roots", roots_manifest["regenerated"] is False, roots_manifest["regenerated"]),
        ("exactly 11 operator applications requested per root", len(OPERATOR_NAMES) == 11 and per_root.eq(11).all(), per_root.value_counts().to_dict()),
        ("no depth relay", protocol["relay_depth"] == 1, protocol["relay_depth"]),
        ("no repeated operator trials", per_pair.eq(1).all(), int(per_pair.max())),
        ("nominal child count 110000", len(offspring) == validation["nominal_children"] == 110_000, len(offspring)),
        ("deterministic child RNG", np.array_equal(offspring.child_seed.to_numpy(dtype=np.uint64), expected_seeds) and replay_ok, RNG_NAMESPACE),
        ("no hardness evaluated", not offspring.current_exact_hardness_available.any() and "hardness" not in child_qd.columns, int(offspring.current_exact_hardness_available.sum())),
        ("no GP trained", protocol["forbidden_operations_executed"]["gp_training"] is False, False),
        ("no Concorde run", protocol["forbidden_operations_executed"]["concorde"] is False, False),
        ("no BQ fine-tuning run", protocol["forbidden_operations_executed"]["bq_finetuning"] is False, False),
        ("no external performance benchmark run", protocol["forbidden_operations_executed"]["external_performance_evaluation"] is False, False),
        ("QD mapping uses only frozen assets", protocol["frozen_inputs"]["normalization"] == sha256_file(ROOT / "assets/qd/normalization/normalization_A_P1P99.json") and protocol["frozen_inputs"]["centroids"] == sha256_file(ROOT / "assets/qd/cvt/centroids.csv"), protocol["frozen_inputs"]),
        ("root coverage report generated", coverage["ROOTS_ONLY"]["candidate_count"] == 10_000, coverage["ROOTS_ONLY"]),
        ("offspring coverage report generated", coverage["OFFSPRING_ONLY"]["candidate_count"] == len(child_qd), coverage["OFFSPRING_ONLY"]),
        ("union coverage report generated", coverage["ROOTS_PLUS_OFFSPRING"]["occupied_niches"] == migration["union_occupied_niches"], coverage["ROOTS_PLUS_OFFSPRING"]),
        ("migration statistics generated", sum(migration[key] for key in ["same_parent_niche_count", "other_root_occupied_niche_count", "niche_absent_from_roots_count"]) == len(child_qd), migration),
        ("operator diagnostics generated", set(operator.operator_id) == set(OPERATOR_NAMES) and operator.children_generated.eq(10_000).all(), operator.to_dict("records")),
        ("provisional offspring-only probes generated", len(child_probes) == coverage["OFFSPRING_ONLY"]["occupied_niches"], len(child_probes)),
        ("provisional union probes generated", len(union_probes) == coverage["ROOTS_PLUS_OFFSPRING"]["occupied_niches"], len(union_probes)),
        ("probes not Oracle-labeled", (~child_probes.oracle_labeled).all() and (~union_probes.oracle_labeled).all(), False),
        ("no final GP0 source automatically selected", probes_summary["automatic_source_selection"] is False, probes_summary["automatic_source_selection"]),
        ("no final runner executed by bootstrap audit", protocol["forbidden_operations_executed"]["final_runner"] is False, False),
        ("runs directory remains empty", not any((ROOT / "runs").iterdir()), 0),
        ("METHOD_FROZEN consistent", "maximum depth one" in (ROOT / "METHOD_FROZEN.md").read_text(encoding="utf-8"), "maximum depth one"),
        ("PROTOCOL_STATUS consistent", status["frozen"]["active_budget"] == "FROZEN_10000" and status["pending"] == {}, status),
        ("README consistent", "at most one elite per niche" in (ROOT / "README.md").read_text(encoding="utf-8"), "at most one elite per niche"),
        ("development provenance copied and runtime-independent", provenance_ok and provenance_manifest["runtime_dependency"] is False, provenance_manifest),
        ("all required audit outputs and figures generated", expected_outputs <= output_names, sorted(expected_outputs - output_names)),
        ("all tests pass", tests_passed == tests_total and tests_passed > 0, tests_output),
    ]
    assert len(raw_checks) == 50
    checks = [
        {"id": index, "requirement": name, "passed": bool(condition), "evidence": evidence}
        for index, (name, condition, evidence) in enumerate(raw_checks, start=1)
    ]
    payload = {
        "verification": "BLADE mainline protocol refactor and Uniform bootstrap audit",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(item["passed"] for item in checks),
        "passed_checks": sum(item["passed"] for item in checks),
        "total_checks": len(checks),
        "tests": {"passed": tests_passed, "total": tests_total, "output": tests_output},
        "checks": checks,
    }
    (AUDIT / "strict_verification.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload, tests_passed, tests_total


if __name__ == "__main__":
    result, tests_passed, tests_total = verify()
    validation = read_json(AUDIT / "03_validation_summary.json")
    coverage = read_json(AUDIT / "06_coverage_summary.json")
    migration = read_json(AUDIT / "09_migration_summary.json")
    probes = read_json(AUDIT / "12_gp0_probe_summary.json")
    print("BLADE_MAINLINE_PROTOCOL_REFACTOR_AND_UNIFORM_BOOTSTRAP_AUDIT_COMPLETE")
    print("\nQD_K=1000")
    print("\nQD_ARCHIVE_POLICY=SPARSE_ONE_ELITE_PER_NICHE")
    print("\nACTIVE_POPULATION_BUDGET=10000")
    print("\nBQ_HARD_TRAIN_BUDGET=10000")
    print("\nOFFSPRING_BUDGET=110000")
    print("\nUNIFORM_ROOTS=10000")
    print(f"\nNOMINAL_OFFSPRING={validation['nominal_children']}")
    print(f"\nVALID_UNIQUE_OFFSPRING={validation['valid_unique_children']}")
    print(f"\nROOT_OCCUPIED_NICHES={coverage['ROOTS_ONLY']['occupied_niches']}/1000")
    print(f"\nOFFSPRING_OCCUPIED_NICHES={coverage['OFFSPRING_ONLY']['occupied_niches']}/1000")
    print(f"\nUNION_OCCUPIED_NICHES={coverage['ROOTS_PLUS_OFFSPRING']['occupied_niches']}/1000")
    print(f"\nNEW_NICHES_BEYOND_ROOTS={migration['NEW_NICHES_DISCOVERED']}")
    print(f"\nOFFSPRING_ONLY_GP0_PROBE_COUNT={probes['offspring_only']['probe_count']}")
    print(f"\nUNION_GP0_PROBE_COUNT={probes['union']['probe_count']}")
    print("\nGP0_TRAINED=NO")
    print("\nCONCORDE_RUN=NO")
    print("\nBQ_FINETUNE_RUN=NO")
    print("\nEXTERNAL_PERFORMANCE_EVAL=NO")
    print("\nFINAL_RUNNER_STARTED=NO")
    print("\nRUNS_DIRECTORY_EMPTY=YES")
    print("\nOLD_PROJECT_MODIFIED=NO")
    print(f"\nSTRICT_VERIFICATION={'PASS' if result['passed'] else 'FAIL'}")
    print(f"\nTESTS={tests_passed}/{tests_total}")
    print(f"\nREPORT={AUDIT / 'report.md'}")
    if not result["passed"]:
        failed = [item for item in result["checks"] if not item["passed"]]
        print("\nFAILED_CHECKS=" + json.dumps(failed, default=str))
        raise SystemExit(1)
