"""Regression checks for D3-CVT-256 working map v1.

The numeric tolerance is the Stage 2B-D gate, 1e-5. A failure stops the freeze.
It does not authorize a new fit.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[2]
PYTHON = "/home/haozou/miniconda3/envs/bq-nco/bin/python"
CLI = ROOT / "scripts/qd_map_tsp100.py"
MANIFEST = PACKAGE / "map_manifest.json"
TOLERANCE = 1e-5


def run_cli(source: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [PYTHON, str(CLI), "--input", str(source), "--map", str(MANIFEST), "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )


def max_abs(left, right) -> float:
    return float(np.max(np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))))


def main() -> None:
    coordinates = np.load(PACKAGE / "tests/golden_inputs.npy")
    expected = json.loads((PACKAGE / "tests/golden_outputs.json").read_text())
    if coordinates.shape != (len(expected["cases"]), 100, 2):
        raise SystemExit(f"golden shape {coordinates.shape}")
    with tempfile.TemporaryDirectory(prefix="qd-map-v1-") as folder:
        work = Path(folder)
        batch_out = work / "batch.json"
        batch = run_cli(PACKAGE / "tests/golden_inputs.npy", batch_out)
        if batch.returncode != 0:
            raise SystemExit(f"batch mapping failed: {batch.stderr}")
        rows = json.loads(batch_out.read_text())["assignments"]
        worst = 0.0
        for case, row in zip(expected["cases"], rows):
            raw_gap = max_abs(row["raw_d3_features"], case["raw_d3"])
            scaled_gap = max_abs(row["scaled_d3_features"], case["scaled_d3"])
            worst = max(worst, raw_gap, scaled_gap)
            if row["niche_id"] != case["niche_id"] or raw_gap > TOLERANCE or scaled_gap > TOLERANCE:
                raise SystemExit(f"FREEZE_BLOCKED {case['name']} raw {raw_gap} scaled {scaled_gap}")
            distance_gap = max_abs(row["nearest_centroid_distance"], case["nearest_centroid_distance"])
            second_gap = max_abs(row["second_nearest_centroid_distance"], case["second_nearest_centroid_distance"])
            margin_gap = max_abs(row["assignment_margin"], case["assignment_margin"])
            worst = max(worst, distance_gap, second_gap, margin_gap)
            if max(distance_gap, second_gap, margin_gap) > TOLERANCE:
                raise SystemExit(f"FREEZE_BLOCKED {case['name']} distance gap")
            if row["reference_support_status"] != case["reference_support_status"]:
                raise SystemExit(f"FREEZE_BLOCKED {case['name']} support status")
            if row["map_version"] != "d3-cvt256-v1":
                raise SystemExit("FREEZE_BLOCKED map version")
        single_path = work / "single.npy"
        np.save(single_path, coordinates[0])
        single_out = work / "single.json"
        single = run_cli(single_path, single_out)
        if single.returncode != 0:
            raise SystemExit(f"single mapping failed: {single.stderr}")
        single_row = json.loads(single_out.read_text())["assignments"][0]
        if single_row["niche_id"] != rows[0]["niche_id"]:
            raise SystemExit("FREEZE_BLOCKED single niche")
        if max_abs(single_row["scaled_d3_features"], rows[0]["scaled_d3_features"]) > TOLERANCE:
            raise SystemExit("FREEZE_BLOCKED single scaled features")
        invalid_path = work / "invalid.npy"
        np.save(invalid_path, np.zeros((99, 2)))
        invalid = run_cli(invalid_path, work / "invalid.json")
        if invalid.returncode == 0 or "error" not in json.loads((work / "invalid.json").read_text()):
            raise SystemExit("FREEZE_BLOCKED invalid shape was accepted")
        nan_path = work / "nan.npy"
        np.save(nan_path, np.full((100, 2), np.nan))
        nonfinite = run_cli(nan_path, work / "nan.json")
        if nonfinite.returncode == 0 or "error" not in json.loads((work / "nan.json").read_text()):
            raise SystemExit("FREEZE_BLOCKED nonfinite input was accepted")
    report = {
        "passed": True,
        "tolerance": TOLERANCE,
        "max_abs_difference": worst,
        "cases": [case["name"] for case in expected["cases"]],
        "niche_ids": [row["niche_id"] for row in rows],
        "single_niche_id": rows[0]["niche_id"],
        "batch_count": len(rows),
        "invalid_rejected": True,
        "nonfinite_rejected": True,
    }
    (PACKAGE / "tests/regression_result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
