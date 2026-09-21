#!/usr/bin/env python
"""Strict structural verification of the refactored mainline workspace."""

from __future__ import annotations

import os
from pathlib import Path
import stat
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from verify_assets import verify_all  # noqa: E402


QD4 = [
    "spn_sum_of_lowest_edge_values",
    "spn_distance_varcoeff",
    "spn_mst_depth_mean",
    "spn_mst_dists_median",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def yaml_file(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8-sig"))


def has_reparse_point(path: Path) -> bool:
    info = os.lstat(path)
    attributes = getattr(info, "st_file_attributes", 0)
    return path.is_symlink() or bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def verify_no_runtime_source_references() -> int:
    forbidden = "D:" + "\\NCO\\" + "BLADE-NCO\\"
    targets = [ROOT / "src", ROOT / "scripts", ROOT / "configs", ROOT / "README.md", ROOT / "METHOD_FROZEN.md", ROOT / "PROTOCOL_STATUS.yaml"]
    found: list[str] = []
    for target in targets:
        paths = [target] if target.is_file() else target.rglob("*")
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in {".py", ".yaml", ".md"}:
                continue
            normalized = path.read_text(encoding="utf-8-sig")
            while "\\\\" in normalized:
                normalized = normalized.replace("\\\\", "\\")
            if forbidden in normalized:
                found.append(path.relative_to(ROOT).as_posix())
    require(not found, f"old runtime references found: {found}")
    return 0


def verify_all_mainline() -> dict[str, int]:
    require(ROOT == Path("D:/NCO/BLADE-NCO-mainline"), f"wrong workspace: {ROOT}")
    require((ROOT / "configs/final_20cycle_protocol.yaml").is_file(), "final protocol config missing")
    require((ROOT / "scripts/run_final_20cycle.py").is_file(), "final runner missing")
    require((ROOT / "src/final_run").is_dir(), "final runtime package missing")

    status = yaml_file(ROOT / "PROTOCOL_STATUS.yaml")
    require(status["method_status"] == "FROZEN", "method is not frozen")
    require(status["pending"] == {}, "method-level pending fields remain")
    require(status["frozen"]["exact_archive"] == "FROZEN_SPARSE_ONE_PER_NICHE", "archive policy mismatch")
    require(status["frozen"]["active_budget"] == "FROZEN_10000", "active budget mismatch")
    require(status["frozen"]["nominal_offspring"] == "DERIVED_FROZEN_110000", "offspring mismatch")

    qd = yaml_file(ROOT / "configs/qd.yaml")
    require(qd["features"] == "QD4" and qd["feature_names"] == QD4 and qd["k"] == 1000, "QD config mismatch")
    require(qd["archive"] == {"type": "sparse_cvt_map_elites", "max_elites_per_niche": 1, "maximum_total_elites": 1000, "force_full_occupancy": False}, "sparse archive config mismatch")
    mainline = yaml_file(ROOT / "configs/mainline_protocol.yaml")
    require(mainline["active_population"]["size"] == 10_000, "active population mismatch")
    require(mainline["mutation"]["nominal_offspring_per_cycle"] == 110_000, "derived offspring mismatch")
    solver = yaml_file(ROOT / "configs/solver_training.yaml")
    require(int(solver["hard_unique_budget"]) == int(solver["uniform_anchor_budget"]) == 10_000, "training budgets mismatch")
    require(float(solver["lr"]) == 7.5e-5 and solver["epochs"] == 256 and solver["early_stopping"] is False, "training protocol mismatch")

    reparse = [path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*") if has_reparse_point(path)]
    require(not reparse, f"symlink/junction found: {reparse}")
    old_references = verify_no_runtime_source_references()
    final = yaml_file(ROOT / "configs/final_20cycle_protocol.yaml")
    require(final["run"]["cycle_indices"] == list(range(20)) and final["run"]["model_final"] == "M20", "cycle horizon mismatch")
    require(final["acquisition"]["fill"]["type"] == "global_gp_topk" and final["acquisition"]["round_robin"] is False, "final acquisition mismatch")
    counts = verify_all()
    counts["old_runtime_references"] = old_references
    counts["symlink_assets"] = len(reparse)
    return counts


if __name__ == "__main__":
    counts = verify_all_mainline()
    print("MAINLINE_PROTOCOL_STRUCTURE=PASS")
    print(f"ASSET_COUNT={counts['asset_count']}")
    print("QD_ARCHIVE_POLICY=SPARSE_ONE_ELITE_PER_NICHE")
    print("ACTIVE_POPULATION_BUDGET=10000")
    print("OFFSPRING_BUDGET=110000")
    print(f"OLD_RUNTIME_REFERENCES={counts['old_runtime_references']}")
    print(f"SYMLINKS={counts['symlink_assets']}")
    print("FINAL_RUNNER=IMPLEMENTED")
