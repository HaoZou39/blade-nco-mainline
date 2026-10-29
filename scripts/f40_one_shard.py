#!/usr/bin/env python
"""Extract one F40 shard from a pickled job on stdin.

This process must not import PyTorch. Forking Rscript from a process that
already owns Torch threads segfaults, which is what stopped the HGB arms.
"""

from __future__ import annotations

import csv
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd


def extract(job: dict) -> str:
    out = Path(job["out"])
    coords = np.asarray(job["coordinates"], dtype=np.float64)
    hashes = list(job["hashes"])
    names = list(job["names"])
    if out.is_file():
        try:
            cached = pd.read_parquet(out, columns=["instance_hash"]).instance_hash.astype(str).tolist()
        except Exception:
            cached = []
        if cached == hashes:
            return str(out)
    with tempfile.TemporaryDirectory(prefix="s4b_f40_") as temporary:
        work = Path(temporary)
        inp, raw, schema = work / "coordinates.csv", work / "features.csv", work / "schema.csv"
        with inp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["state_id", "node_idx", "x", "y"])
            for instance_hash, points in zip(hashes, coords, strict=True):
                for node, (x, y) in enumerate(points):
                    writer.writerow([instance_hash, node, repr(float(x)), repr(float(y))])
        process = subprocess.run(
            [job["rscript"], job["r_script"], str(inp), str(raw), str(schema)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900,
        )
        if process.returncode or not raw.is_file():
            raise RuntimeError(f"F40 extraction failed rc={process.returncode}: {process.stderr[-1500:]}")
        # pandas.read_csv's C parser segfaults on these salesperson tables.
        with raw.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        if len(rows) < 2:
            raise RuntimeError("F40 extraction returned no rows")
        data = pd.DataFrame(rows[1:], columns=rows[0])
        for name in data.columns:
            if name != "state_id":
                data[name] = pd.to_numeric(data[name], errors="coerce")
        if data.state_id.astype(str).tolist() != hashes:
            raise RuntimeError("F40 alignment failure")
        if "feature_error" in data.columns and data.feature_error.notna().any():
            raise RuntimeError("F40 feature_error reported")
        missing = [name for name in names if name not in data.columns]
        if missing:
            raise RuntimeError(f"F40 missing columns {missing[:8]}")
        frame = data[["state_id", *names]].rename(columns={"state_id": "instance_hash"})
        params = job["params"]
        for name in names:
            center, scale = params[name]
            frame[name] = np.arcsinh((frame[name].to_numpy(float) - center) / scale)
        values = frame[names].to_numpy(float)
        if np.isinf(values).any():
            bad = np.isinf(values)
            counts = {name: int(bad[:, i].sum()) for i, name in enumerate(names) if bad[:, i].any()}
            raise RuntimeError(f"infinite F40 features {counts}")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp, index=False)
    tmp.replace(out)
    return str(out)


def main() -> None:
    job = pickle.load(sys.stdin.buffer)
    print(extract(job), flush=True)


if __name__ == "__main__":
    main()
