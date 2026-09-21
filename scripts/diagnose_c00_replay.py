#!/usr/bin/env python
"""Locate the first deterministic C00 mutation replay mismatch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit.uniform_bootstrap import child_seed
from mutation import apply_operator
from qd.roots import recover_uniform_roots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=110_000)
    args = parser.parse_args()
    roots = recover_uniform_roots()
    audit = pd.read_parquet(ROOT / "audits/uniform_rooted_qd_bootstrap_v1/02_offspring_manifest.parquet")
    mismatch = None
    checked = 0
    for row in audit.itertuples(index=False):
        seed = child_seed(str(row.root_hash), str(row.operator_id))
        result = apply_operator(
            str(row.operator_id),
            roots.coordinates[int(row.root_index)],
            np.random.default_rng(seed),
        )
        checked += 1
        bad = []
        if seed != int(row.child_seed):
            bad.append("seed")
        if result.raw_output_hash != str(row.raw_child_hash):
            bad.append("raw_hash")
        if result.final_output_hash != str(row.child_hash):
            bad.append("final_hash")
        if bad:
            mismatch = {
                "row": checked - 1,
                "root_index": int(row.root_index),
                "operator": str(row.operator_id),
                "bad": bad,
                "seed": seed,
                "expected_seed": int(row.child_seed),
                "raw": result.raw_output_hash,
                "expected_raw": str(row.raw_child_hash),
                "final": result.final_output_hash,
                "expected_final": str(row.child_hash),
                "repair_count": result.repair_count,
                "expected_repair_count": int(row.repair_count),
            }
            break
        if checked >= args.max_rows:
            break
    print(json.dumps({
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "checked": checked,
        "first_mismatch": mismatch,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
