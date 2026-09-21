"""Strict parsing and semantic validation of the frozen run configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


EXPECTED_DATASETS = [
    "Uniform100", "GaussianMixture100", "TSPLIB100", "OriginalTSPLIB70",
    "LEHD_TSP200", "LEHD_TSP500", "LEHD_TSP1000",
]


def load_and_validate(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    checks = {
        "method frozen": config["method_status"] == "FROZEN",
        "cycles C00-C19": config["run"]["cycles"] == 20 and config["run"]["cycle_indices"] == list(range(20)),
        "M0 to M20": config["run"]["model_start"] == "M0" and config["run"]["model_final"] == "M20",
        "no early stop": config["run"]["scientific_early_stop"] is False,
        "QD4 K1000": config["qd"]["features"] == "QD4" and config["qd"]["k"] == 1000 and config["qd"]["fixed_map"] is True,
        "geometry only": config["qd"]["reference_cloud_role"] == "MAP_GEOMETRY_ONLY",
        "Uniform10k": config["initialization"]["roots"]["count"] == 10_000 and config["initialization"]["roots"]["regenerate"] is False,
        "FOGA11": config["mutation"]["family"] == "FOGA11" and config["mutation"]["each_operator_once_per_parent"] is True and config["mutation"]["nominal_offspring"] == 110_000,
        "centroid plus global": config["acquisition"]["coverage_first"] is True and config["acquisition"]["fill"]["type"] == "global_gp_topk" and config["acquisition"]["round_robin"] is False,
        "GP4": config["surrogate"]["primary"] == "GP4" and config["surrogate"]["population"] == 2000 and config["surrogate"]["generations"] == 60 and config["surrogate"]["seeds"] == list(range(2026, 2031)),
        "active10k": config["active_population"]["size"] == 10_000,
        "training256": config["training"]["epochs"] == 256 and float(config["training"]["lr"]) == 7.5e-5 and config["training"]["early_stop"] is False,
        "benchmarks": config["evaluation"]["datasets"] == EXPECTED_DATASETS and config["evaluation"]["passive_only"] is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"frozen config semantic failure: {failed}")
    return config
