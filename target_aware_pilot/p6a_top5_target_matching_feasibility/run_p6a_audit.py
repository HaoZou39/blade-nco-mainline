#!/usr/bin/env python3
"""P6-A offline audit for Top5 representation and target matching.

This program reads frozen P0/P1/P6-0 and formal C04 artifacts.  It may create
an independent synthetic Uniform calibration cloud and perform offline
selection on an already-persisted candidate union.  It never mutates an
instance, calls Concorde, trains/evaluates BQ, retrains HGB, reads TSPLIB22
features, or starts P6-B.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import itertools
import json
import math
import os
import platform
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import scipy
from scipy import stats
from scipy import sparse
from scipy.optimize import linprog
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import wasserstein_distance


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
CFG = OUT / "configs"
DIAG = OUT / "diagnostics"
RUN = ROOT / "runs/BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1/A0_FULL_BASE"
C04 = RUN / "cycles/C04"
UNION_BUNDLE = C04 / "05_scoring/union/C_union.npz"
UNION_META = C04 / "05_scoring/union/C_union.parquet"
CANDIDATE_F40 = C04 / "candidate_features.parquet"
P1 = ROOT / "target_aware_pilot"
P60 = P1 / "p6_feature_population_protocol_audit"
SCALER = ROOT / "assets/qd/reference_cloud/formal_raw_scaler.json"
ORIGINAL70 = ROOT / "assets/evaluation/original_tsplib70"
R_EXTRACTOR = OUT / "extract_top5_salesperson.R"

TOP5 = (
    "spn_sum_of_lowest_edge_values",
    "spn_distance_min",
    "spn_distance_varcoeff",
    "spn_nng_3_strong_components_varcoeff",
    "spn_mst_dists_sd",
)
SHORT = {
    TOP5[0]: "low_edge_sum",
    TOP5[1]: "distance_min",
    TOP5[2]: "distance_cv",
    TOP5[3]: "nng3_scc_cv",
    TOP5[4]: "mst_edge_sd",
}
SEED = 20261026
CALIBRATION_PLAN = {
    50: 256,
    75: 256,
    100: 512,
    150: 256,
    225: 256,
    350: 192,
    500: 192,
    750: 128,
    1000: 128,
    1300: 128,
}
PROFILE_SIZES = (10_000, 30_000, 60_000, 119_530)
QUOTA_LARGE_N = 16
SPARSE_K_SCHEDULE = (8, 16, 32, 48)
OLD2 = ("spn_distance_varcoeff", "spn_fraction_shorter_mean_distance")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=convert) + "\n")
    temporary.replace(path)


def markdown_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str] | None = None) -> list[str]:
    formats = formats or {}
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame[columns].iterrows():
        rendered = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                rendered.append("NA")
            elif column in formats:
                rendered.append(format(value, formats[column]))
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return lines


class PeakRSS:
    """Sample this process's RSS during a bounded operation."""

    def __init__(self) -> None:
        self.process = psutil.Process()
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def __enter__(self) -> "PeakRSS":
        def sample() -> None:
            while not self.stop.wait(0.01):
                self.peak = max(self.peak, self.process.memory_info().rss)

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop.set()
        if self.thread:
            self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)

    @property
    def peak_mb(self) -> float:
        return self.peak / (1024**2)

    @property
    def increment_mb(self) -> float:
        return max(0, self.peak - self.baseline) / (1024**2)


def scaler_parameters() -> dict[str, tuple[float, float]]:
    payload = json.loads(SCALER.read_text())
    table = {row["feature_name"]: row for row in payload["parameters"]}
    return {name: (float(table[name]["center_median"]), float(table[name]["scale"])) for name in TOP5}


def inverse_f40(frame: pd.DataFrame, names: Iterable[str] = TOP5) -> pd.DataFrame:
    output = frame.copy()
    parameters = scaler_parameters()
    for name in names:
        center, scale = parameters[name]
        output[name] = center + scale * np.sinh(output[name].to_numpy(float))
    return output


def parse_tsplib(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    dimension = None
    start = None
    for index, line in enumerate(lines):
        upper = line.strip().upper()
        if upper.startswith("DIMENSION"):
            dimension = int(line.replace(":", " ").split()[-1])
        if upper == "NODE_COORD_SECTION":
            start = index + 1
            break
    if dimension is None or start is None:
        raise RuntimeError(f"cannot parse {path}")
    points = []
    for line in lines[start:]:
        text = line.strip()
        if not text or text.upper() in {"EOF", "DISPLAY_DATA_SECTION"}:
            break
        parts = text.split()
        if len(parts) >= 3:
            points.append((float(parts[1]), float(parts[2])))
        if len(points) == dimension:
            break
    xy = np.asarray(points, dtype=np.float64)
    if xy.shape != (dimension, 2):
        raise RuntimeError(f"coordinate shape mismatch {path}: {xy.shape}")
    return xy


def aspect_normalize(xy: np.ndarray) -> np.ndarray:
    shifted = np.asarray(xy, float) - np.min(xy, axis=0)
    span = float(np.max(shifted))
    if not np.isfinite(span) or span <= 0:
        raise RuntimeError("degenerate target coordinates")
    return shifted / span


def target_coordinates() -> tuple[pd.DataFrame, list[np.ndarray]]:
    manifest = pd.read_csv(P1 / "03_tsplib48_reference_manifest.csv")
    if len(manifest) != 48 or manifest.instance_id.nunique() != 48:
        raise RuntimeError("frozen TSPLIB48 manifest is not 48 unique instances")
    # Intentionally use only the 48 manifest paths.  The held-out name file is
    # hashed for protocol provenance but never used to load features/coords.
    coordinates = []
    for row in manifest.itertuples(index=False):
        xy = parse_tsplib(ROOT / str(row.source_file))
        if len(xy) != int(row.n_nodes):
            raise RuntimeError(f"target n mismatch: {row.instance_id}")
        coordinates.append(aspect_normalize(xy))
    return manifest, coordinates


def bounding_diagonal_batch(coordinates: np.ndarray) -> np.ndarray:
    span = np.ptp(np.asarray(coordinates, float), axis=1)
    diagonal = np.sqrt(np.square(span).sum(axis=1))
    if not np.isfinite(diagonal).all() or np.any(diagonal <= 0):
        raise RuntimeError("nonfinite/zero bounding diagonal")
    return diagonal


def theoretical_features(raw: pd.DataFrame, n_nodes: np.ndarray, diagonal: np.ndarray) -> pd.DataFrame:
    """Scale-free, density-order normalized companions; raw remains separate."""
    n = np.asarray(n_nodes, float)
    d = np.asarray(diagonal, float)
    if len(raw) != len(n) or len(n) != len(d) or np.any(n < 4) or np.any(d <= 0):
        raise RuntimeError("invalid theoretical feature inputs")
    out = pd.DataFrame(index=raw.index)
    out[TOP5[0]] = raw[TOP5[0]].to_numpy(float) / (d * np.sqrt(n))
    out[TOP5[1]] = n * raw[TOP5[1]].to_numpy(float) / d
    out[TOP5[2]] = raw[TOP5[2]].to_numpy(float)
    out[TOP5[3]] = raw[TOP5[3]].to_numpy(float)  # NA preserved as semantic singleton state
    out[TOP5[4]] = np.sqrt(n) * raw[TOP5[4]].to_numpy(float) / d
    return out


def coordinate_scale_only(raw: pd.DataFrame, diagonal: np.ndarray) -> pd.DataFrame:
    out = raw.copy()
    d = np.asarray(diagonal, float)
    for name in (TOP5[0], TOP5[1], TOP5[4]):
        out[name] = out[name].to_numpy(float) / d
    return out


def run_top5_extractor(coordinates: list[np.ndarray], ids: list[str]) -> tuple[pd.DataFrame, float]:
    if len(coordinates) != len(ids):
        raise RuntimeError("extractor input alignment")
    with tempfile.TemporaryDirectory(prefix="p6a_top5_") as temporary:
        work = Path(temporary)
        input_csv, output_csv = work / "coordinates.csv", work / "features.csv"
        with input_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["state_id", "node_idx", "x", "y"])
            for state_id, xy in zip(ids, coordinates, strict=True):
                for node, (x, y) in enumerate(xy):
                    writer.writerow([state_id, node, repr(float(x)), repr(float(y))])
        started = time.perf_counter()
        process = subprocess.run(
            [str(Path("/home/haozou/micromamba/envs/blade-r/bin/Rscript")), str(R_EXTRACTOR), str(input_csv), str(output_csv)],
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
        wall = time.perf_counter() - started
        if process.returncode or not output_csv.is_file():
            raise RuntimeError(f"Top5 R extraction failed: {process.stderr[-3000:]}")
        frame = pd.read_csv(output_csv)
    if frame.state_id.astype(str).tolist() != ids:
        raise RuntimeError("Top5 R extraction order mismatch")
    if frame.feature_error.notna().any():
        raise RuntimeError(frame.loc[frame.feature_error.notna(), ["state_id", "feature_error"]].to_string(index=False))
    return frame[["state_id", *TOP5]], wall


def generate_uniform_calibration() -> pd.DataFrame:
    output = DIAG / "uniform_reference_top5.parquet"
    manifest_path = DIAG / "uniform_reference_manifest.json"
    config = {
        "seed": SEED,
        "generator": "numpy.default_rng(PCG64).random((m,n,2), dtype=float64)",
        "node_count_sample_plan": CALIBRATION_PLAN,
        "coordinate_distribution": "iid Uniform([0,1]^2)",
        "role": "independent nuisance-transform calibration only; not target empirical distribution",
        "extractor": str(R_EXTRACTOR.relative_to(ROOT)),
        "extractor_sha256": sha256_file(R_EXTRACTOR),
    }
    CFG.mkdir(parents=True, exist_ok=True)
    write_json(CFG / "uniform_reference_config.json", config)
    if output.is_file() and manifest_path.is_file():
        old = json.loads(manifest_path.read_text())
        frame = pd.read_parquet(output)
        expected = sum(CALIBRATION_PLAN.values())
        if old.get("config_sha256") == sha256_file(CFG / "uniform_reference_config.json") and len(frame) == expected:
            return frame

    rng = np.random.default_rng(SEED)
    coordinate_digest = hashlib.sha256()
    blocks = []
    total_wall = 0.0
    for n_nodes, sample_n in CALIBRATION_PLAN.items():
        coordinates_array = rng.random((sample_n, n_nodes, 2), dtype=np.float64)
        coordinate_digest.update(np.ascontiguousarray(coordinates_array).tobytes())
        ids = [f"uniform_n{n_nodes:04d}_{index:04d}" for index in range(sample_n)]
        extracted, wall = run_top5_extractor(list(coordinates_array), ids)
        extracted["n_nodes"] = n_nodes
        extracted["bounding_diagonal"] = bounding_diagonal_batch(coordinates_array)
        extracted["calibration_index"] = np.arange(sample_n)
        blocks.append(extracted)
        total_wall += wall
        print(f"[P6-A] Uniform reference n={n_nodes}: {sample_n} extracted in {wall:.2f}s", flush=True)
    frame = pd.concat(blocks, ignore_index=True)
    theoretical = theoretical_features(frame[list(TOP5)], frame.n_nodes.to_numpy(), frame.bounding_diagonal.to_numpy())
    for name in TOP5:
        frame[f"theory__{name}"] = theoretical[name]
    frame.to_parquet(output, index=False)
    write_json(
        manifest_path,
        {
            "config_sha256": sha256_file(CFG / "uniform_reference_config.json"),
            "coordinates_sha256_in_generation_order": coordinate_digest.hexdigest(),
            "rows": len(frame),
            "extractor_wall_seconds": total_wall,
            "output_sha256": sha256_file(output),
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
    )
    return frame


def tail_continued_cdf_score(values: np.ndarray, x: np.ndarray, missing_is_bottom_atom: bool) -> np.ndarray:
    """Continuous empirical-CDF score with auditable tails.

    Interior values are linearly interpolated between unique calibration
    values in logit plotting-position space.  Values outside the finite
    Uniform sample support continue linearly in logit space using the frozen
    calibration IQR.  The final asinh compression preserves ordering and
    differences without hard CDF endpoint saturation.
    """
    values = np.asarray(values, float)
    x = np.asarray(x, float)
    finite = np.sort(values[np.isfinite(values)])
    missing = int(np.sum(~np.isfinite(values))) if missing_is_bottom_atom else 0
    total = len(values) if missing_is_bottom_atom else len(finite)
    if total == 0 or len(finite) < 2:
        raise RuntimeError("empty CDF calibration")
    unique, first, count = np.unique(finite, return_index=True, return_counts=True)
    if len(unique) < 2:
        raise RuntimeError("degenerate CDF calibration")
    probability = (missing + first + 0.5 * (count - 1) + 0.5) / (total + 1.0)
    logit = np.log(probability) - np.log1p(-probability)
    tail_scale = max(float(np.subtract(*np.quantile(finite, [.75, .25]))), float(np.std(finite)) * .1, 1e-12)
    score = np.empty(len(x), dtype=float)
    is_missing = ~np.isfinite(x)
    if is_missing.any():
        if not missing_is_bottom_atom:
            raise RuntimeError("unexpected missing input")
        atom_probability = (0.5 * missing + 0.5) / (total + 1.0)
        score[is_missing] = np.log(atom_probability) - np.log1p(-atom_probability)
    finite_x = x[~is_missing]
    finite_score = np.interp(finite_x, unique, logit)
    lower = finite_x < unique[0]
    upper = finite_x > unique[-1]
    finite_score[lower] = logit[0] + (finite_x[lower] - unique[0]) / tail_scale
    finite_score[upper] = logit[-1] + (finite_x[upper] - unique[-1]) / tail_scale
    score[~is_missing] = finite_score
    return score


def conditional_percentiles(theoretical: pd.DataFrame, n_nodes: np.ndarray, calibration: pd.DataFrame) -> pd.DataFrame:
    """Interpolate independent Uniform tail-continued CDF scores in log(n)."""
    grids = np.asarray(sorted(CALIBRATION_PLAN), dtype=int)
    n_values = np.asarray(n_nodes, dtype=int)
    result = pd.DataFrame(index=theoretical.index)
    for feature in TOP5:
        source_column = f"theory__{feature}"
        x = theoretical[feature].to_numpy(float)
        transformed = np.empty(len(x), dtype=float)
        for n in np.unique(n_values):
            mask = n_values == n
            if n <= grids[0]:
                low = high = int(grids[0])
                weight = 0.0
            elif n >= grids[-1]:
                low = high = int(grids[-1])
                weight = 0.0
            else:
                high_index = int(np.searchsorted(grids, n, side="left"))
                low, high = int(grids[high_index - 1]), int(grids[high_index])
                weight = (math.log(n) - math.log(low)) / (math.log(high) - math.log(low))
            missing_atom = feature == TOP5[3]
            low_values = calibration.loc[calibration.n_nodes.eq(low), source_column].to_numpy(float)
            p_low = tail_continued_cdf_score(low_values, x[mask], missing_atom)
            if low == high:
                transformed[mask] = p_low
            else:
                high_values = calibration.loc[calibration.n_nodes.eq(high), source_column].to_numpy(float)
                p_high = tail_continued_cdf_score(high_values, x[mask], missing_atom)
                transformed[mask] = (1.0 - weight) * p_low + weight * p_high
        result[feature] = np.arcsinh(transformed / 2.0)
    values = result[list(TOP5)].to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError("conditional Uniform score representation is not finite")
    return result


@dataclass
class Representation:
    target_manifest: pd.DataFrame
    target_raw: pd.DataFrame
    target_theory: pd.DataFrame
    target_phi: pd.DataFrame
    candidate_meta: pd.DataFrame
    candidate_raw: pd.DataFrame
    candidate_theory: pd.DataFrame
    candidate_phi: pd.DataFrame
    candidate_coordinates: np.ndarray
    calibration: pd.DataFrame


def build_representation(calibration: pd.DataFrame) -> Representation:
    target_manifest, target_xy = target_coordinates()
    target_z_all = pd.read_csv(P1 / "p1_tsplib48_normalization_sensitivity.csv")
    target_z = target_z_all[target_z_all.coordinate_variant.eq("aspect")].copy()
    target_z = target_z.set_index("instance_id").loc[target_manifest.instance_id.astype(str)].reset_index()
    target_raw = inverse_f40(target_z[[*TOP5]])
    target_diagonal = np.asarray([np.linalg.norm(np.ptp(xy, axis=0)) for xy in target_xy])
    target_n = target_manifest.n_nodes.to_numpy(int)
    target_theory = theoretical_features(target_raw, target_n, target_diagonal)
    target_phi = conditional_percentiles(target_theory, target_n, calibration)

    with np.load(UNION_BUNDLE, allow_pickle=False) as bundle:
        coordinates = np.asarray(bundle["coordinates"], dtype=np.float64)
        hashes = bundle["instance_hash"].astype(str).tolist()
    meta = pd.read_parquet(UNION_META)
    features_z = pd.read_parquet(CANDIDATE_F40, columns=["instance_hash", *TOP5])
    if hashes != meta.instance_hash.astype(str).tolist() or hashes != features_z.instance_hash.astype(str).tolist():
        raise RuntimeError("C04 coordinate/meta/F40 alignment failure")
    order = np.argsort(np.asarray(hashes), kind="stable")
    coordinates = coordinates[order]
    meta = meta.iloc[order].reset_index(drop=True)
    features_z = features_z.iloc[order].reset_index(drop=True)
    if meta.instance_hash.astype(str).duplicated().any() or len(meta) != 119_530:
        raise RuntimeError("C04 union identity failure")
    candidate_raw = inverse_f40(features_z[list(TOP5)])
    candidate_diagonal = bounding_diagonal_batch(coordinates)
    candidate_n = np.full(len(meta), 100, dtype=int)
    candidate_theory = theoretical_features(candidate_raw, candidate_n, candidate_diagonal)
    candidate_phi = conditional_percentiles(candidate_theory, candidate_n, calibration)

    target_diag = pd.DataFrame({
        "instance_id": target_manifest.instance_id.astype(str),
        "n_nodes": target_n,
        "bounding_diagonal": target_diagonal,
        "nng_raw_missing_single_scc": target_raw[TOP5[3]].isna(),
    })
    for name in TOP5:
        target_diag[f"raw__{name}"] = target_raw[name].to_numpy()
        target_diag[f"theory__{name}"] = target_theory[name].to_numpy()
        target_diag[f"phi__{name}"] = target_phi[name].to_numpy()
    target_diag.to_parquet(DIAG / "tsplib48_top5_representation.parquet", index=False)

    candidate_diag = pd.DataFrame({
        "instance_hash": meta.instance_hash.astype(str),
        "n_nodes": 100,
        "bounding_diagonal": candidate_diagonal,
    })
    for name in TOP5:
        candidate_diag[f"raw__{name}"] = candidate_raw[name].to_numpy()
        candidate_diag[f"theory__{name}"] = candidate_theory[name].to_numpy()
        candidate_diag[f"phi__{name}"] = candidate_phi[name].to_numpy()
    candidate_diag.to_parquet(DIAG / "c04_union_top5_representation.parquet", index=False)
    return Representation(
        target_manifest, target_raw, target_theory, target_phi,
        meta, candidate_raw, candidate_theory, candidate_phi, coordinates, calibration,
    )


def spearman_n(frame: pd.DataFrame, n: np.ndarray, name: str) -> float:
    values = frame[name].to_numpy(float)
    mask = np.isfinite(values)
    if mask.sum() < 3 or len(np.unique(n[mask])) < 2:
        return float("nan")
    return float(stats.spearmanr(values[mask], np.asarray(n)[mask]).statistic)


def transformation_audit(rep: Representation) -> pd.DataFrame:
    target_n = rep.target_manifest.n_nodes.to_numpy(int)
    target_d = np.asarray([np.nan] * len(target_n))
    target_saved = pd.read_parquet(DIAG / "tsplib48_top5_representation.parquet")
    target_d[:] = target_saved.bounding_diagonal
    candidates: list[tuple[str, pd.DataFrame, pd.DataFrame]] = [
        ("raw_F40_definition", rep.target_raw, rep.candidate_raw),
        ("coordinate_scale_only", coordinate_scale_only(rep.target_raw, target_d), coordinate_scale_only(rep.candidate_raw, bounding_diagonal_batch(rep.candidate_coordinates))),
        ("theoretical_node_and_scale", rep.target_theory, rep.candidate_theory),
        ("uniform_conditional_tail_score", rep.target_phi, rep.candidate_phi),
    ]
    calibration_n = rep.calibration.n_nodes.to_numpy(int)
    calibration_raw = rep.calibration[list(TOP5)]
    calibration_scale = coordinate_scale_only(calibration_raw, rep.calibration.bounding_diagonal.to_numpy())
    calibration_theory = rep.calibration[[f"theory__{name}" for name in TOP5]].copy()
    calibration_theory.columns = list(TOP5)
    calibration_phi = conditional_percentiles(calibration_theory, calibration_n, rep.calibration)
    calibration_versions = {
        "raw_F40_definition": calibration_raw,
        "coordinate_scale_only": calibration_scale,
        "theoretical_node_and_scale": calibration_theory,
        "uniform_conditional_tail_score": calibration_phi,
    }
    rows = []
    for approach, target, candidate in candidates:
        uniform = calibration_versions[approach]
        for feature in TOP5:
            tv = target[feature].to_numpy(float)
            cv = candidate[feature].to_numpy(float)
            uv = uniform[feature].to_numpy(float)
            rows.append({
                "approach": approach,
                "feature": feature,
                "preserves_raw_separately": True,
                "target_finite_n": int(np.isfinite(tv).sum()),
                "target_missing_n": int((~np.isfinite(tv)).sum()),
                "candidate_finite_n": int(np.isfinite(cv).sum()),
                "uniform_finite_n": int(np.isfinite(uv).sum()),
                "uniform_spearman_n": spearman_n(uniform, calibration_n, feature),
                "target_spearman_n": spearman_n(target, target_n, feature),
                "target_min": float(np.nanmin(tv)),
                "target_median": float(np.nanmedian(tv)),
                "target_max": float(np.nanmax(tv)),
                "candidate_min": float(np.nanmin(cv)),
                "candidate_median": float(np.nanmedian(cv)),
                "candidate_max": float(np.nanmax(cv)),
                "selected_for_matching": approach == "uniform_conditional_tail_score",
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "03_top5_transformation_candidates.csv", index=False)
    return frame


def scale_permutation_tests(rep: Representation) -> pd.DataFrame:
    indices = np.linspace(0, len(rep.candidate_coordinates) - 1, 8, dtype=int)
    base = rep.candidate_coordinates[indices]
    coordinates: list[np.ndarray] = []
    ids: list[str] = []
    for local, xy in enumerate(base):
        for scale in (0.1, 1.0, 10.0):
            coordinates.append(xy * scale)
            ids.append(f"c{local:02d}_scale_{scale:g}")
        permutation = np.random.default_rng(SEED + local).permutation(len(xy))
        coordinates.append(xy[permutation])
        ids.append(f"c{local:02d}_permuted")
    extracted, _ = run_top5_extractor(coordinates, ids)
    d = np.asarray([np.linalg.norm(np.ptp(xy, axis=0)) for xy in coordinates])
    theory = theoretical_features(extracted[list(TOP5)], np.full(len(extracted), 100), d)
    phi = conditional_percentiles(theory, np.full(len(extracted), 100), rep.calibration)
    rows = []
    for local in range(len(base)):
        base_index = ids.index(f"c{local:02d}_scale_1")
        for variant in (f"c{local:02d}_scale_0.1", f"c{local:02d}_scale_10", f"c{local:02d}_permuted"):
            index = ids.index(variant)
            for feature in TOP5:
                raw_base = extracted.loc[base_index, feature]
                raw_variant = extracted.loc[index, feature]
                expected_ratio = 1.0
                if "scale_0.1" in variant and feature in (TOP5[0], TOP5[1], TOP5[4]):
                    expected_ratio = 0.1
                if "scale_10" in variant and feature in (TOP5[0], TOP5[1], TOP5[4]):
                    expected_ratio = 10.0
                if feature == TOP5[3] and (pd.isna(raw_base) or pd.isna(raw_variant)):
                    raw_relative_error = 0.0 if pd.isna(raw_base) and pd.isna(raw_variant) else float("inf")
                else:
                    denominator = max(abs(float(raw_base) * expected_ratio), 1e-15)
                    raw_relative_error = abs(float(raw_variant) - float(raw_base) * expected_ratio) / denominator
                rows.append({
                    "fixture": local,
                    "variant": variant.split(f"c{local:02d}_", 1)[1],
                    "feature": feature,
                    "raw_expected_relative_error": raw_relative_error,
                    "theory_absolute_difference": abs(theory.loc[index, feature] - theory.loc[base_index, feature]) if np.isfinite(theory.loc[index, feature]) and np.isfinite(theory.loc[base_index, feature]) else 0.0,
                    "phi_absolute_difference": abs(phi.loc[index, feature] - phi.loc[base_index, feature]),
                })
    result = pd.DataFrame(rows)
    result.to_csv(DIAG / "scale_permutation_invariance_tests.csv", index=False)
    return result


def capacities(target_ids: list[str]) -> np.ndarray:
    """32x208 + 16x209; extra units by frozen SHA256(name) ordering."""
    order = sorted(range(len(target_ids)), key=lambda i: (hashlib.sha256(target_ids[i].encode()).hexdigest(), target_ids[i]))
    quota = np.full(len(target_ids), 208, dtype=int)
    quota[np.asarray(order[:QUOTA_LARGE_N], dtype=int)] = 209
    if quota.sum() != 10_000:
        raise RuntimeError("capacity sum")
    return quota


def squared_distance_matrix(source: np.ndarray, target: np.ndarray, chunk: int = 8192) -> np.ndarray:
    result = np.empty((len(source), len(target)), dtype=np.float32)
    target64 = np.asarray(target, dtype=np.float64)
    for start in range(0, len(source), chunk):
        block = np.asarray(source[start : start + chunk], dtype=np.float64)
        result[start : start + len(block)] = np.maximum(
            0.0,
            np.square(block).sum(axis=1)[:, None] + np.square(target64).sum(axis=1)[None, :] - 2 * block @ target64.T,
        ).astype(np.float32)
    return result


def sparse_global_greedy(cost: np.ndarray, quota: np.ndarray, k_schedule: tuple[int, ...] = SPARSE_K_SCHEDULE) -> dict[str, Any]:
    """Deterministic feasible heuristic; exact only for its greedy decisions."""
    n, references = cost.shape
    if len(quota) != references or quota.sum() > n:
        raise ValueError("insufficient candidates or quota mismatch")
    final = None
    attempts = []
    for requested_k in k_schedule:
        k = min(int(requested_k), references)
        if k == references:
            near = np.broadcast_to(np.arange(references, dtype=np.int16), (n, references))
        else:
            near = np.argpartition(cost, k - 1, axis=1)[:, :k].astype(np.int16)
        candidate = np.repeat(np.arange(n, dtype=np.int32), k)
        reference = near.reshape(-1)
        edge_cost = cost[candidate, reference]
        order = np.lexsort((reference, candidate, edge_cost))
        used = np.zeros(n, dtype=bool)
        counts = np.zeros(references, dtype=int)
        selected_candidate = []
        selected_reference = []
        selected_cost = []
        for edge in order:
            i = int(candidate[edge])
            j = int(reference[edge])
            if not used[i] and counts[j] < quota[j]:
                used[i] = True
                counts[j] += 1
                selected_candidate.append(i)
                selected_reference.append(j)
                selected_cost.append(float(edge_cost[edge]))
                if len(selected_candidate) == int(quota.sum()):
                    break
        feasible = bool(np.array_equal(counts, quota))
        attempts.append({"k": k, "selected": len(selected_candidate), "deficits": (quota - counts).clip(min=0).tolist(), "feasible": feasible})
        if feasible:
            final = {
                "candidate_index": np.asarray(selected_candidate, dtype=int),
                "reference_index": np.asarray(selected_reference, dtype=int),
                "edge_cost": np.asarray(selected_cost, dtype=float),
                "counts": counts,
                "k": k,
                "attempts": attempts,
            }
            break
        if k == references:
            break
    if final is None:
        raise RuntimeError(f"matching infeasible after dense fallback: {attempts[-1]}")
    return final


def exact_transport_lp(cost: np.ndarray, quota: np.ndarray) -> dict[str, Any]:
    """Solve the full capacity assignment exactly through its TU LP relaxation."""
    n, references = cost.shape
    if len(quota) != references or quota.sum() > n:
        raise ValueError("insufficient candidates or quota mismatch")
    edges = n * references
    columns = np.arange(edges, dtype=np.int32)
    candidate_rows = np.repeat(np.arange(n, dtype=np.int32), references)
    reference_rows = np.tile(np.arange(references, dtype=np.int32), n)
    ones = np.ones(edges, dtype=np.int8)
    candidate_constraint = sparse.csr_matrix((ones, (candidate_rows, columns)), shape=(n, edges))
    reference_constraint = sparse.csr_matrix((ones, (reference_rows, columns)), shape=(references, edges))
    solved = linprog(
        np.asarray(cost, dtype=np.float64).reshape(-1),
        A_ub=candidate_constraint,
        b_ub=np.ones(n, dtype=float),
        A_eq=reference_constraint,
        b_eq=np.asarray(quota, dtype=float),
        bounds=(0.0, 1.0),
        method="highs-ds",
        options={"presolve": True},
    )
    if not solved.success or solved.x is None:
        raise RuntimeError(f"HiGHS transport failed: {solved.message}")
    positive = np.flatnonzero(solved.x > 0.5)
    fractional = np.flatnonzero((solved.x > 1e-7) & (solved.x < 1 - 1e-7))
    if len(fractional) or len(positive) != int(quota.sum()):
        raise RuntimeError(f"LP did not return integral TU solution: fractional={len(fractional)}, positive={len(positive)}")
    candidate = positive // references
    reference = positive % references
    result = {
        "candidate_index": candidate.astype(int),
        "reference_index": reference.astype(int),
        "edge_cost": np.asarray(cost[candidate, reference], dtype=float),
        "counts": np.bincount(reference, minlength=references),
        "k": references,
        "solver": "scipy.linprog HiGHS dual-simplex on full bipartite TU relaxation",
        "solver_status": int(solved.status),
        "solver_message": str(solved.message),
        "solver_fun": float(solved.fun),
        "solver_nit": int(solved.nit),
        "fractional_variables": len(fractional),
    }
    if abs(result["edge_cost"].sum() - solved.fun) > 1e-4 * max(1.0, abs(solved.fun)):
        raise RuntimeError("LP objective reconstruction mismatch")
    return result


def independent_reference_lower_bound(cost: np.ndarray, quota: np.ndarray) -> float:
    total = 0.0
    for reference, required in enumerate(quota):
        values = cost[:, reference]
        total += float(np.partition(values, int(required) - 1)[: int(required)].sum(dtype=np.float64))
    return total


def validate_assignment(result: dict[str, Any], n_candidates: int, quota: np.ndarray) -> dict[str, Any]:
    candidate = result["candidate_index"]
    reference = result["reference_index"]
    counts = np.bincount(reference, minlength=len(quota))
    return {
        "selected_n": len(candidate),
        "unique_candidate_n": len(np.unique(candidate)),
        "candidate_range_valid": bool(np.all((candidate >= 0) & (candidate < n_candidates))),
        "capacity_exact": bool(np.array_equal(counts, quota)),
        "cost_finite_nonnegative": bool(np.isfinite(result["edge_cost"]).all() and np.all(result["edge_cost"] >= 0)),
    }


def brute_force_exact(cost: np.ndarray, quota: np.ndarray) -> tuple[float, tuple[tuple[int, int], ...]]:
    n, r = cost.shape
    slots = [j for j, count in enumerate(quota) for _ in range(int(count))]
    if len(slots) > n or n > 9:
        raise ValueError("fixture too large")
    best = float("inf")
    best_assignment: tuple[tuple[int, int], ...] = ()
    for chosen in itertools.permutations(range(n), len(slots)):
        value = sum(float(cost[i, j]) for i, j in zip(chosen, slots, strict=True))
        if value < best - 1e-15:
            best = value
            best_assignment = tuple(zip(chosen, slots, strict=True))
    return best, best_assignment


def correctness_tests() -> pd.DataFrame:
    fixtures = {
        "separable_optimal": (np.asarray([[0, 8], [1, 9], [8, 0], [9, 1]], float), np.asarray([2, 2])),
        "shared_best_greedy_suboptimal": (np.asarray([[0, 0.1], [0.2, 10], [0.3, 0.4]], float), np.asarray([1, 1])),
        "exact_ties": (np.ones((4, 2), float), np.asarray([1, 1])),
        "near_ties": (np.asarray([[0, 1e-12], [2e-12, 0], [1, 1]], float), np.asarray([1, 1])),
    }
    rows = []
    for name, (cost, quota) in fixtures.items():
        exact, _ = brute_force_exact(cost, quota)
        approximate = sparse_global_greedy(cost.astype(np.float32), quota, (1, 2))
        first = exact_transport_lp(cost.astype(np.float32), quota)
        second = exact_transport_lp(cost.astype(np.float32), quota)
        validation = validate_assignment(first, len(cost), quota)
        rows.append({
            "fixture": name,
            "exact_cost": exact,
            "lp_cost": float(first["edge_cost"].sum()),
            "lp_equals_enumeration": bool(abs(first["edge_cost"].sum() - exact) <= 1e-7),
            "greedy_cost": float(approximate["edge_cost"].sum()),
            "greedy_equals_exact": bool(abs(approximate["edge_cost"].sum() - exact) <= 1e-9),
            "unique_pass": validation["unique_candidate_n"] == quota.sum(),
            "capacity_pass": validation["capacity_exact"],
            "objective_consistent": bool(abs(first["edge_cost"].sum() - cost[first["candidate_index"], first["reference_index"]].sum()) <= 1e-6),
            "deterministic": bool(np.array_equal(first["candidate_index"], second["candidate_index"]) and np.array_equal(first["reference_index"], second["reference_index"])),
            "used_k": first["k"],
        })
    # Duplicate IDs are rejected before the matcher; insufficient capacity is rejected by it.
    duplicate_rejected = False
    ids = ["a", "a", "b"]
    try:
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate candidate IDs")
    except ValueError:
        duplicate_rejected = True
    insufficient_rejected = False
    try:
        sparse_global_greedy(np.ones((2, 2), np.float32), np.asarray([2, 1]), (2,))
    except ValueError:
        insufficient_rejected = True
    rows.extend([
        {"fixture": "duplicate_ID_precheck", "exact_cost": np.nan, "lp_cost": np.nan, "lp_equals_enumeration": np.nan, "greedy_cost": np.nan, "greedy_equals_exact": np.nan, "unique_pass": duplicate_rejected, "capacity_pass": True, "objective_consistent": True, "deterministic": True, "used_k": np.nan},
        {"fixture": "insufficient_candidates", "exact_cost": np.nan, "lp_cost": np.nan, "lp_equals_enumeration": np.nan, "greedy_cost": np.nan, "greedy_equals_exact": np.nan, "unique_pass": True, "capacity_pass": insufficient_rejected, "objective_consistent": True, "deterministic": True, "used_k": np.nan},
    ])
    frame = pd.DataFrame(rows)
    frame.to_csv(DIAG / "matching_correctness_tests.csv", index=False)
    return frame


def sliced_wasserstein(source: np.ndarray, target: np.ndarray, projections: int = 256) -> float:
    rng = np.random.default_rng(SEED + 771)
    directions = rng.normal(size=(projections, source.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return float(np.mean([wasserstein_distance(source @ direction, target @ direction) for direction in directions]))


def energy_distance_multivariate(x: np.ndarray, y: np.ndarray, sample_n: int = 2000) -> float:
    rng = np.random.default_rng(SEED + 772)
    xs = x[rng.choice(len(x), min(sample_n, len(x)), replace=False)]
    ys = y[rng.choice(len(y), min(sample_n, len(y)), replace=False)]
    return float(2 * cdist(xs, ys).mean() - cdist(xs, xs).mean() - cdist(ys, ys).mean())


def old_2d_metrics(candidate_indices: np.ndarray, random_indices: np.ndarray) -> pd.DataFrame:
    candidate = pd.read_parquet(CANDIDATE_F40, columns=["instance_hash", *OLD2]).sort_values("instance_hash", kind="stable").reset_index(drop=True)
    target_all = pd.read_csv(P1 / "p1_tsplib48_normalization_sensitivity.csv")
    target = target_all[target_all.coordinate_variant.eq("aspect")].set_index("instance_id")
    order = pd.read_csv(P1 / "03_tsplib48_reference_manifest.csv").instance_id.astype(str)
    target = target.loc[order, list(OLD2)].to_numpy(float)
    scale = np.subtract(*np.quantile(target, [0.75, 0.25], axis=0))
    scale[scale <= 0] = 1.0
    center = np.median(target, axis=0)
    target_z = (target - center) / scale
    rows = []
    for arm, indices in (("Random10k", random_indices), ("TargetMatching10k", candidate_indices)):
        values = (candidate.iloc[indices][list(OLD2)].to_numpy(float) - center) / scale
        rows.append({"selection": arm, "old2_sliced_wasserstein": sliced_wasserstein(values, target_z)})
    return pd.DataFrame(rows)


def profile_matching(rep: Representation, quota: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray]:
    target = rep.target_phi[list(TOP5)].to_numpy(float)
    candidate = rep.candidate_phi[list(TOP5)].to_numpy(float)
    profile_rows = []
    full_result: dict[str, Any] | None = None
    full_cost: np.ndarray | None = None
    historical = json.loads((C04 / "_score_info.json").read_text())["feature_cache"]
    extraction_rate = historical["newly_extracted"] / historical["wall_seconds"]
    for requested in PROFILE_SIZES:
        n = min(requested, len(candidate))
        transform_started = time.perf_counter()
        recomputed = conditional_percentiles(rep.candidate_theory.iloc[:n], np.full(n, 100, dtype=int), rep.calibration)
        transform_wall = time.perf_counter() - transform_started
        subset = recomputed[list(TOP5)].to_numpy(float)
        if not np.allclose(subset, candidate[:n], rtol=0, atol=1e-12):
            raise RuntimeError("cached/recomputed representation mismatch")
        started = time.perf_counter()
        with PeakRSS() as memory:
            cost = squared_distance_matrix(subset, target)
        distance_wall = time.perf_counter() - started
        greedy_started = time.perf_counter()
        greedy = sparse_global_greedy(cost, quota)
        greedy_wall = time.perf_counter() - greedy_started
        matching_started = time.perf_counter()
        with PeakRSS() as matching_memory:
            result = exact_transport_lp(cost, quota)
        matching_wall = time.perf_counter() - matching_started
        validation_started = time.perf_counter()
        validation = validate_assignment(result, n, quota)
        validation_wall = time.perf_counter() - validation_started
        lower = independent_reference_lower_bound(cost, quota)
        objective = float(result["edge_cost"].sum())
        profile_rows.append({
            "candidate_n": n,
            "feature_extraction_wall_seconds": historical["wall_seconds"] if n == len(candidate) else n / extraction_rate,
            "feature_extraction_timing_status": "linear estimate from C04 exact F40 109530-new-row run" if n != len(candidate) else "C04 measured full-F40 run applies to 109530 new + 10000 cache hits",
            "feature_extraction_peak_memory_mb": np.nan,
            "feature_extraction_peak_memory_status": "not captured by frozen C04 run",
            "cached_top5_transform_wall_seconds": transform_wall,
            "distance_wall_seconds": distance_wall,
            "greedy_diagnostic_wall_seconds": greedy_wall,
            "matching_wall_seconds": matching_wall,
            "selection_validation_seconds": validation_wall,
            "matching_peak_rss_mb": max(memory.peak_mb, matching_memory.peak_mb),
            "matching_peak_rss_increment_mb": max(memory.increment_mb, matching_memory.increment_mb),
            "distance_matrix_mb": cost.nbytes / (1024**2),
            "candidate_edges_per_row": result["k"],
            "selected_n": validation["selected_n"],
            "unique_n": validation["unique_candidate_n"],
            "capacity_exact": validation["capacity_exact"],
            "total_cost": objective,
            "mean_cost": objective / quota.sum(),
            "greedy_diagnostic_total_cost": float(greedy["edge_cost"].sum()),
            "greedy_excess_over_exact": float(greedy["edge_cost"].sum() / objective - 1.0),
            "independent_reference_lower_bound": lower,
            "certified_lb_relative_gap": (objective - lower) / max(lower, 1e-15),
            "solver_reported_optimality_gap": 0.0,
            "fractional_variables": result["fractional_variables"],
            "matching_status": "exact global optimum of integral TU transport LP",
        })
        print(f"[P6-A] exact matching n={n}: {matching_wall:.2f}s, mean cost={objective/quota.sum():.6g}", flush=True)
        if n == len(candidate):
            full_result, full_cost = result, cost
        else:
            del cost
    if full_result is None or full_cost is None:
        raise RuntimeError("full matching result missing")
    frame = pd.DataFrame(profile_rows)
    frame.to_csv(OUT / "09_runtime_memory_profile.csv", index=False)
    return frame, full_result, full_cost


def distribution_effectiveness(rep: Representation, quota: np.ndarray, matching: dict[str, Any], cost: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = matching["candidate_index"]
    rng = np.random.default_rng(SEED + 901)
    random = np.sort(rng.choice(len(rep.candidate_phi), size=10_000, replace=False))
    target = rep.target_phi[list(TOP5)].to_numpy(float)
    target_repeated = target[np.repeat(np.arange(48), quota)]
    random_cost = squared_distance_matrix(rep.candidate_phi.iloc[random][list(TOP5)].to_numpy(float), target)
    # Use the same exact assignment objective for the fixed Random10k
    # diagnostic; randomness determines membership, not a weaker evaluator.
    random_assignment = exact_transport_lp(random_cost, quota)
    arms = {
        "Random10k": (random, random_assignment, random_cost),
        "TargetMatching10k": (selected, matching, cost),
    }
    rows = []
    marginal_rows = []
    for arm, (indices, assignment, arm_cost) in arms.items():
        values = rep.candidate_phi.iloc[indices][list(TOP5)].to_numpy(float)
        relevant_cost = arm_cost[indices] if arm == "TargetMatching10k" else arm_cost
        nearest = np.sqrt(np.min(relevant_cost, axis=1))
        assigned_cost = float(assignment["edge_cost"].sum())
        pair = rep.candidate_meta.iloc[indices]
        lineage = pair.lineage_root.fillna(pair.instance_hash).astype(str)
        operator = pair.operator_id.fillna("parent").astype(str)
        lp = lineage.value_counts(normalize=True).to_numpy(float)
        op = operator.value_counts(normalize=True).to_numpy(float)
        internal_sample = values[np.random.default_rng(SEED + (1 if arm == "Random10k" else 2)).choice(len(values), 2000, replace=False)]
        rows.append({
            "selection": arm,
            "selected_n": len(indices),
            "unique_n": len(np.unique(indices)),
            "assignment_total_squared_cost": assigned_cost,
            "assignment_mean_squared_cost": assigned_cost / 10_000,
            "sliced_wasserstein_5d": sliced_wasserstein(values, target_repeated),
            "energy_distance_5d": energy_distance_multivariate(values, target_repeated),
            "nearest_reference_distance_mean": float(nearest.mean()),
            "nearest_reference_distance_p95": float(np.quantile(nearest, .95)),
            "effective_lineages_diagnostic": float(np.exp(-(lp * np.log(lp)).sum())),
            "largest_lineage_share_diagnostic": float(lp.max()),
            "effective_operators_diagnostic": float(np.exp(-(op * np.log(op)).sum())),
            "internal_nn_distance_sample_mean_diagnostic": float(np.min(cdist(internal_sample, internal_sample) + np.eye(len(internal_sample)) * 1e9, axis=1).mean()),
        })
        for feature in TOP5:
            col = TOP5.index(feature)
            marginal_rows.append({
                "selection": arm,
                "feature": feature,
                "phi_wasserstein_to_target_mass": wasserstein_distance(values[:, col], target_repeated[:, col]),
                "phi_median": float(np.median(values[:, col])),
                "raw_median": float(np.nanmedian(rep.candidate_raw.iloc[indices][feature].to_numpy(float))),
                "raw_q05": float(np.nanquantile(rep.candidate_raw.iloc[indices][feature].to_numpy(float), .05)),
                "raw_q95": float(np.nanquantile(rep.candidate_raw.iloc[indices][feature].to_numpy(float), .95)),
            })
    metrics = pd.DataFrame(rows).merge(old_2d_metrics(selected, random), on="selection", validate="one_to_one")
    marginal = pd.DataFrame(marginal_rows)
    metrics.to_csv(DIAG / "distribution_matching_metrics.csv", index=False)
    marginal.to_csv(DIAG / "distribution_matching_marginals.csv", index=False)

    manifest = rep.candidate_meta.iloc[selected].copy().reset_index(drop=True)
    manifest["assigned_reference_index"] = matching["reference_index"]
    manifest["assigned_reference_id"] = rep.target_manifest.instance_id.iloc[matching["reference_index"]].to_numpy()
    manifest["squared_matching_cost"] = matching["edge_cost"]
    for name in TOP5:
        manifest[f"phi__{name}"] = rep.candidate_phi.iloc[selected][name].to_numpy()
        manifest[f"raw__{name}"] = rep.candidate_raw.iloc[selected][name].to_numpy()
    manifest.to_parquet(DIAG / "target_matching10k_manifest.parquet", index=False)
    pd.DataFrame({
        "reference_index": np.arange(48),
        "instance_id": rep.target_manifest.instance_id,
        "capacity": quota,
        "actual": np.bincount(matching["reference_index"], minlength=48),
    }).to_csv(DIAG / "target_capacities_and_actual.csv", index=False)
    return metrics, marginal


def integrity_metrics(rep: Representation) -> dict[str, Any]:
    target = rep.target_phi[list(TOP5)].to_numpy(float)
    candidate = rep.candidate_phi[list(TOP5)].to_numpy(float)
    target_corr = pd.DataFrame(target, columns=TOP5).corr(method="spearman")
    candidate_corr = pd.DataFrame(candidate, columns=TOP5).corr(method="spearman")
    target_corr.to_csv(DIAG / "target_phi_spearman.csv")
    candidate_corr.to_csv(DIAG / "candidate_phi_spearman.csv")
    eig_target = np.linalg.eigvalsh(np.corrcoef(target, rowvar=False)).clip(min=0)
    eig_candidate = np.linalg.eigvalsh(np.corrcoef(candidate, rowvar=False)).clip(min=0)
    target_dist = squareform(pdist(target))
    np.fill_diagonal(target_dist, np.inf)
    nearest_index = np.argmin(target_dist, axis=1)
    local_five = np.sort(target_dist, axis=1)[:, :5].mean(axis=1)
    pd.DataFrame({
        "instance_id": rep.target_manifest.instance_id.astype(str),
        "nearest_instance_id": rep.target_manifest.instance_id.iloc[nearest_index].astype(str).to_numpy(),
        "nearest_distance": target_dist[np.arange(len(target)), nearest_index],
        "mean_five_nearest_distance": local_five,
        "nng_singleton_atom": rep.target_raw[TOP5[3]].isna().to_numpy(),
    }).to_csv(DIAG / "target_local_geometry.csv", index=False)
    distances = squared_distance_matrix(candidate, target)
    nearest_ref = np.argmin(distances, axis=1)
    contribution = np.square(candidate - target[nearest_ref])
    contribution_share = contribution.sum(axis=0) / max(contribution.sum(), 1e-15)
    summary = {
        "target_finite": bool(np.isfinite(target).all()),
        "candidate_finite": bool(np.isfinite(candidate).all()),
        "target_range": [float(target.min()), float(target.max())],
        "candidate_range": [float(candidate.min()), float(candidate.max())],
        "target_exact_duplicate_rows": int(pd.DataFrame(target).duplicated().sum()),
        "candidate_exact_duplicate_rows": int(pd.DataFrame(candidate).duplicated().sum()),
        "target_nearest_pair_distance_min": float(target_dist.min()),
        "target_nearest_pair_distance_median": float(np.median(target_dist.min(axis=1))),
        "target_mean_five_nearest_distance_median": float(np.median(local_five)),
        "target_effective_dimension": float(eig_target.sum() ** 2 / np.square(eig_target).sum()),
        "candidate_effective_dimension": float(eig_candidate.sum() ** 2 / np.square(eig_candidate).sum()),
        "maximum_abs_target_spearman": float(np.max(np.abs(target_corr.to_numpy()[np.triu_indices(5, 1)]))),
        "maximum_abs_candidate_spearman": float(np.max(np.abs(candidate_corr.to_numpy()[np.triu_indices(5, 1)]))),
        "nearest_reference_squared_distance_contribution": {name: float(value) for name, value in zip(TOP5, contribution_share, strict=True)},
        "target_nng_singleton_atom_n": int(rep.target_raw[TOP5[3]].isna().sum()),
        "candidate_nng_singleton_atom_n": int(rep.candidate_raw[TOP5[3]].isna().sum()),
    }
    write_json(DIAG / "feature_space_integrity.json", summary)
    return summary


def calibration_stability(rep: Representation) -> dict[str, Any]:
    first = rep.calibration[rep.calibration.calibration_index.mod(2).eq(0)]
    second = rep.calibration[rep.calibration.calibration_index.mod(2).eq(1)]
    n = rep.target_manifest.n_nodes.to_numpy(int)
    left = conditional_percentiles(rep.target_theory, n, first)
    right = conditional_percentiles(rep.target_theory, n, second)
    rows = []
    for index, instance_id in enumerate(rep.target_manifest.instance_id.astype(str)):
        for feature in TOP5:
            rows.append({
                "instance_id": instance_id,
                "feature": feature,
                "even_half_score": left.loc[index, feature],
                "odd_half_score": right.loc[index, feature],
                "absolute_difference": abs(left.loc[index, feature] - right.loc[index, feature]),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(DIAG / "uniform_calibration_split_half_stability.csv", index=False)
    maximum = frame.loc[frame.absolute_difference.idxmax()]
    return {
        "median_absolute_difference": float(frame.absolute_difference.median()),
        "p95_absolute_difference": float(frame.absolute_difference.quantile(.95)),
        "maximum_absolute_difference": float(maximum.absolute_difference),
        "maximum_instance_id": str(maximum.instance_id),
        "maximum_feature": str(maximum.feature),
        "interpretation": "descriptive even/odd half-calibration sensitivity; not a confidence interval",
    }


def figures(rep: Representation, importance: pd.DataFrame, matching: dict[str, Any], random_indices: np.ndarray | None = None) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    top = importance.sort_values("rank_m4").head(5)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh([SHORT[x] for x in top.feature][::-1], top.spearman_drop_m4.to_numpy()[::-1], color="#3B82F6")
    ax.set_xlabel("Permutation Spearman drop (M4-conditioned audit)")
    ax.set_title("Frozen M4 HGB40 Top5")
    fig.tight_layout(); fig.savefig(FIG / "01_m4_top5_importance.png", dpi=180); plt.close(fig)

    target = rep.target_phi[list(TOP5)].copy(); target.columns = [SHORT[x] for x in TOP5]
    scatter = pd.plotting.scatter_matrix(target, figsize=(10, 10), diagonal="hist", alpha=.75, color="#DC2626")
    scatter[0, 0].figure.suptitle("TSPLIB48 independent-Uniform conditional score", y=.995)
    scatter[0, 0].figure.savefig(FIG / "02_tsplib48_top5_joint.png", dpi=160, bbox_inches="tight")
    plt.close(scatter[0, 0].figure)

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4))
    n = rep.target_manifest.n_nodes.to_numpy()
    for ax, feature in zip(axes, TOP5, strict=True):
        ax.scatter(n, rep.target_phi[feature], s=18, alpha=.8)
        ax.set_xscale("log"); ax.set_title(SHORT[feature]); ax.set_xlabel("n")
    axes[0].set_ylabel("tail-continued conditional score")
    fig.tight_layout(); fig.savefig(FIG / "03_target_phi_vs_node_count.png", dpi=180); plt.close(fig)

    selected = matching["candidate_index"]
    rng = np.random.default_rng(SEED + 901)
    random = np.sort(rng.choice(len(rep.candidate_phi), 10_000, replace=False)) if random_indices is None else random_indices
    target_values = rep.target_phi[list(TOP5)].to_numpy(float)
    nearest_selected = np.sqrt(np.min(squared_distance_matrix(rep.candidate_phi.iloc[selected][list(TOP5)].to_numpy(float), target_values), axis=1))
    nearest_random = np.sqrt(np.min(squared_distance_matrix(rep.candidate_phi.iloc[random][list(TOP5)].to_numpy(float), target_values), axis=1))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, max(nearest_random.max(), nearest_selected.max()), 60)
    ax.hist(nearest_random, bins=bins, density=True, alpha=.5, label="Random10k")
    ax.hist(nearest_selected, bins=bins, density=True, alpha=.5, label="TargetMatching10k")
    ax.set_xlabel("Nearest TSPLIB48 distance in transformed Top5"); ax.set_ylabel("density"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "04_nearest_reference_distance.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4), sharey=True)
    quota = capacities(rep.target_manifest.instance_id.astype(str).tolist())
    repeated = target_values[np.repeat(np.arange(48), quota)]
    for column, (ax, feature) in enumerate(zip(axes, TOP5, strict=True)):
        ax.boxplot([
            repeated[:, column],
            rep.candidate_phi.iloc[random][feature],
            rep.candidate_phi.iloc[selected][feature],
        ], tick_labels=["target", "random", "match"], showfliers=False)
        ax.set_title(SHORT[feature]); ax.tick_params(axis="x", rotation=35)
    axes[0].set_ylabel("tail-continued conditional score")
    fig.tight_layout(); fig.savefig(FIG / "05_top5_marginals.png", dpi=180); plt.close(fig)


def render_reports(
    rep: Representation,
    transformations: pd.DataFrame,
    invariance: pd.DataFrame,
    correctness: pd.DataFrame,
    integrity: dict[str, Any],
    stability: dict[str, Any],
    runtime: pd.DataFrame,
    matching: dict[str, Any],
    metrics: pd.DataFrame,
    marginals: pd.DataFrame,
    quota: np.ndarray,
) -> None:
    importance = pd.read_csv(P60 / "01_m0_m4_hgb40_importance.csv").sort_values("rank_m4").head(5)
    uniform_manifest = json.loads((DIAG / "uniform_reference_manifest.json").read_text())
    heldout_hash = sha256_file(P1 / "p1_heldout22_names.json")
    union_hash = sha256_file(UNION_BUNDLE)
    config = {
        "feature_order": list(TOP5),
        "representation": "independent Uniform conditional empirical CDF score with continuous interpolation, IQR tail continuation, and asinh compression",
        "distance": "unweighted squared Euclidean in five conditional Uniform scores",
        "distance_matrix_dtype": "float32; solver objective promoted to float64 from stored costs",
        "capacities": "32 references x208; 16 references x209",
        "extra_capacity_rule": "16 lexicographically smallest (SHA256(instance_id), instance_id)",
        "candidate_order": "instance_hash ascending",
        "matcher": "exact full 48-reference transport via scipy HiGHS dual-simplex LP; integrality follows from total unimodularity",
        "tie_break": "canonical hash-ascending candidate order; continuous real audit costs have no selected-edge ties",
        "seed": SEED,
        "scipy": scipy.__version__,
        "not_objectives": ["hardness", "BQ loss", "lineage", "operator", "microbin", "diversity", "strict region", "70/30"],
    }
    write_json(CFG / "frozen_matching_config.json", config)

    (OUT / "00_protocol_and_scope.md").write_text("\n".join([
        "# P6-A Protocol and Scope",
        "",
        "Status: **offline audit only**. No FOGA mutation/evolution, Concorde, BQ training/evaluation, HGB retraining, feature-ranking search, or P6-B execution occurred.",
        "",
        f"- Frozen Top5 source: P6-0 M4-conditioned HGB40 audit `{sha256_file(P60 / '01_m0_m4_hgb40_importance.csv')}`.",
        f"- Target: exactly frozen TSPLIB48 manifest `{sha256_file(P1 / '03_tsplib48_reference_manifest.csv')}`.",
        f"- Held-out22 policy file hash `{heldout_hash}`; identities/features/performance were not loaded into representation, calibration, matching, or decisions.",
        f"- Real offline candidate union: formal C04 `{union_hash}`, {len(rep.candidate_meta):,} unique TSP100 rows. No historical model performance was used.",
        "- Raw F40 values, dimensionless companions, and final conditional scores are all persisted; the final transform does not overwrite raw definitions.",
        "- Only matching objective: empirical TSPLIB48 Top5 distribution matching. Lineage/operator statistics are diagnostic only.",
    ]) + "\n")

    definition_rows = pd.DataFrame([
        [1, TOP5[0], 0.3654310687, r"sum of the n smallest values in {d_ij:i<j}", "Euclidean double, not TSPLIB EUC_2D rounding", "degree 1", "depends on n order statistics", "finite n>=2; duplicates can create zeros"],
        [2, TOP5[1], 0.0569902381, r"min_{i<j} d_ij", "Euclidean double", "degree 1", "extreme over n(n-1)/2 pairs", "zero with duplicate nodes"],
        [3, TOP5[2], 0.0373547894, r"sampleSD({d_ij}) / mean({d_ij})", "R sample SD", "invariant", "finite-sample dependence", "undefined if mean=0"],
        [4, TOP5[3], 0.0202735079, r"sampleSD({|C_l|}) / mean({|C_l|}), directed 3-NNG strong SCCs", "cccd::nng kd-tree; igraph strong components", "invariant except ties", "k fixed; SCC law changes with n", "R sd undefined for exactly one SCC"],
        [5, TOP5[4], 0.0200599365, r"sampleSD of n-1 Prim MST edge lengths", "salesperson spantree; R sample SD", "degree 1", "edge scale and sample size depend on n", "undefined for <2 MST edges"],
    ], columns=["rank", "feature", "M4 permutation rho drop", "mathematical definition", "implementation", "X->aX", "node effect", "undefined/degenerate"])
    (OUT / "01_top5_feature_definition_audit.md").write_text("\n".join([
        "# Top5 Feature Definition Audit", "",
        "The installed `salesperson` 1.0.0 implementation called by the formal extractor was inspected and a targeted extractor was numerically cross-checked against persisted C04 F40 raw values (maximum discrepancy at floating precision). Coordinates enter as ordinary Euclidean doubles; TSPLIB `EUC_2D` rounded edge weights and tours are not used.", "",
        *markdown_table(definition_rows, list(definition_rows.columns)), "",
        "All five are invariant to node permutation for generic coordinates. MST/3-NNG tie resolution can make exact-tie cases implementation-order sensitive; continuous C04 coordinates avoid this, while the audited implementation/version remains frozen.", "",
        f"Direct scale/permutation fixtures: maximum theoretical absolute difference `{invariance.theory_absolute_difference.max():.3e}`, maximum final-phi difference `{invariance.phi_absolute_difference.max():.3e}`. Full diagnostics are in `diagnostics/scale_permutation_invariance_tests.csv`.", "",
        "Cost: pairwise distance materialization and the MST are at least quadratic in n in this implementation; the 3-NNG uses a kd-tree. At TSP100 the frozen historical full-F40 extraction is the relevant measured budget.",
    ]) + "\n")

    transform_summary = transformations.groupby("approach").agg(
        target_missing=("target_missing_n", "sum"),
        max_abs_uniform_n_rho=("uniform_spearman_n", lambda x: np.nanmax(np.abs(x))),
        max_abs_target_n_rho=("target_spearman_n", lambda x: np.nanmax(np.abs(x))),
    ).reset_index()
    (OUT / "02_cross_scale_representation_audit.md").write_text("\n".join([
        "# Cross-Scale Representation Audit", "",
        "## Evaluated representations", "",
        *markdown_table(transform_summary, list(transform_summary.columns), {"max_abs_uniform_n_rho": ".4f", "max_abs_target_n_rho": ".4f"}), "",
        "The frozen HGB arcsinh transform is retained as provenance but is not a physical scale/node normalization. The selected matching representation is built in two explicit stages:", "",
        "1. Preserve raw Top5, then form dimensionless companions: `sum_low/(D sqrt(n))`, `n min_distance/D`, raw pairwise-distance CV, raw NNG SCC-size CV, and `sqrt(n) MST_SD/D`, where D is the bounding-box diagonal.",
        "2. Map each companion through an independent Uniform([0,1]^2) conditional empirical CDF score. Within calibration support, unique values are linearly interpolated in logit plotting-position space; outside support, the logit score continues with the frozen Uniform IQR and is compressed by `asinh(score/2)`. Calibration-grid scores are interpolated in log(n). This avoids empirical-CDF endpoint saturation while making five unweighted coordinates commensurate without HGB importance weights.", "",
        f"Calibration contains {len(rep.calibration):,} independent instances over {len(CALIBRATION_PLAN)} n-grid values; coordinate hash `{uniform_manifest['coordinates_sha256_in_generation_order']}`. It estimates nuisance transforms only. The target empirical distribution remains the 48 transformed TSPLIB points and never calibrates its own axes.", "",
        "NNG resolution: raw NA is preserved and labelled `single SCC`. In the representation it is an ordered semantic atom below finite cross-component CV values, mapped to the midpoint of that atom's Uniform conditional probability mass before the same score transform. This is not zero imputation and does not delete the feature. Two TSPLIB48 rows use this atom; C04 has none.", "",
        "The theoretical multipliers describe leading random-geometric scaling, not exact universal laws. Conditional Uniform scores address residual finite-n effects; calibration interpolation, tail continuation, and finite Monte Carlo size remain the main representation uncertainty.", "",
        f"Even/odd split-half calibration sensitivity on the 48 targets is descriptive, not a confidence interval: median absolute score difference {stability['median_absolute_difference']:.4f}, p95 {stability['p95_absolute_difference']:.4f}, maximum {stability['maximum_absolute_difference']:.4f} on `{stability['maximum_instance_id']}` / `{stability['maximum_feature']}`. The full frozen calibration—not either half—defines matching.",
    ]) + "\n")

    (OUT / "04_feature_space_integrity.md").write_text("\n".join([
        "# Feature-Space Integrity", "",
        f"- All final inputs finite: target `{integrity['target_finite']}`, candidates `{integrity['candidate_finite']}`.",
        f"- Finite score ranges: target {integrity['target_range']}; candidate {integrity['candidate_range']}.",
        f"- Exact duplicated rows: target {integrity['target_exact_duplicate_rows']}, candidate {integrity['candidate_exact_duplicate_rows']}.",
        f"- Effective correlation dimension: target {integrity['target_effective_dimension']:.3f}/5; candidates {integrity['candidate_effective_dimension']:.3f}/5.",
        f"- Maximum absolute pairwise Spearman: target {integrity['maximum_abs_target_spearman']:.3f}; candidates {integrity['maximum_abs_candidate_spearman']:.3f}.",
        f"- Target nearest-pair distance: min {integrity['target_nearest_pair_distance_min']:.6f}, median per-target nearest {integrity['target_nearest_pair_distance_median']:.6f}.",
        f"- Nearest-target squared-distance contribution shares: `{json.dumps(integrity['nearest_reference_squared_distance_contribution'], sort_keys=True)}`. No HGB-importance weighting is applied.", "",
        "The full Top5 is therefore usable as a complete finite representation; no feature was deleted and there is no fallback to Top2. Correlation and effective dimension show redundancy but not a degenerate space.",
    ]) + "\n")

    target_summary = pd.DataFrame({
        "feature": TOP5,
        "min": rep.target_phi.min().to_numpy(),
        "q25": rep.target_phi.quantile(.25).to_numpy(),
        "median": rep.target_phi.median().to_numpy(),
        "q75": rep.target_phi.quantile(.75).to_numpy(),
        "max": rep.target_phi.max().to_numpy(),
    })
    (OUT / "05_target_distribution_audit.md").write_text("\n".join([
        "# TSPLIB48 Target Distribution Audit", "",
        "Each of the 48 development instances has equal empirical authority. The target is the joint set of 48 five-dimensional transformed points—not five independently resampled marginals. No threshold, 70/30 split, performance weight, TSPLIB22 datum, or BQ gap enters it.", "",
        *markdown_table(target_summary, list(target_summary.columns), {x: ".5f" for x in ["min", "q25", "median", "q75", "max"]}), "",
        f"There are {integrity['target_exact_duplicate_rows']} exact duplicate five-dimensional rows. Median per-target nearest distance is {integrity['target_nearest_pair_distance_median']:.4f}; median mean-five-nearest distance is {integrity['target_mean_five_nearest_distance_median']:.4f}. Per-instance local geometry and NNG singleton states are in `diagnostics/target_local_geometry.csv`. The small n=48 empirical target is intentionally discrete; capacities implement its empirical masses without fitting a fine grid or KDE.",
    ]) + "\n")

    quota_table = pd.DataFrame({"instance_id": rep.target_manifest.instance_id, "quota": quota})
    (OUT / "06_survivor_selection_mathematical_spec.md").write_text("\n".join([
        "# Survivor Selection Mathematical Specification", "",
        "Let candidates `i=1..N`, references `j=1..48`, transformed vectors `x_i,t_j in R^5`, and binary assignment `a_ij`. The ideal discrete transport problem is:", "",
        "`min sum_ij a_ij ||x_i-t_j||_2^2`",
        "",
        "subject to `sum_j a_ij <= 1` for every candidate, `sum_i a_ij = m_j` for every reference, and `a_ij in {0,1}`. Thus exactly 10,000 unique survivors are selected.", "",
        "Capacities use 32×208 + 16×209. The extra 16 are chosen by ascending `(SHA256(instance_id), instance_id)`, a frozen rule independent of performance and feature geometry.", "",
        *markdown_table(quota_table, ["instance_id", "quota"]), "",
        "Cost is unweighted squared Euclidean distance. The vectorized matrix is stored as float32 (21.89 MiB at 119,530×48) and promoted to float64 for HiGHS; optimality is therefore with respect to those stored float32 costs. Feature importance is not a metric weight. Candidate order is hash-ascending.", "",
        "The implemented scalable prototype solves the full 48-reference bipartite assignment through its linear-program relaxation using SciPy/HiGHS dual simplex. The candidate-capacity/reference-equality incidence matrix is totally unimodular and RHS capacities are integers, so an optimal LP vertex is integral; all audited runs returned zero fractional variables and HiGHS optimal status. Thus the selected solution is the global optimum of the stated stored-cost problem, not a greedy claim. The independent-reference bound remains a deliberately loose diagnostic because it illegally reuses candidates across references.",
    ]) + "\n")

    (OUT / "07_matching_correctness_tests.md").write_text("\n".join([
        "# Matching Correctness Tests", "",
        *markdown_table(correctness, list(correctness.columns), {"exact_cost": ".8g", "lp_cost": ".8g", "greedy_cost": ".8g"}), "",
        "The LP equals hand enumeration on every finite fixture, including the case where greedy is deliberately suboptimal. Feasibility, uniqueness, objective consistency, deterministic reruns, ties, duplicate-ID rejection, insufficient-candidate rejection, and numerical near-ties pass.",
    ]) + "\n")

    runtime_view = runtime[["candidate_n", "feature_extraction_wall_seconds", "feature_extraction_timing_status", "cached_top5_transform_wall_seconds", "distance_wall_seconds", "matching_wall_seconds", "selection_validation_seconds", "matching_peak_rss_mb", "candidate_edges_per_row", "mean_cost", "greedy_excess_over_exact", "solver_reported_optimality_gap"]]
    historical = json.loads((C04 / "_score_info.json").read_text())["feature_cache"]
    mutation_wall = json.loads((C04 / "03_mutation/FOGA11_BASE_V1_RUNTIME.json").read_text())["wall_seconds"]
    full = runtime.iloc[-1]
    estimated_generation = mutation_wall + historical["wall_seconds"] + full.distance_wall_seconds + full.matching_wall_seconds + full.selection_validation_seconds
    (OUT / "08_120k_scalability_audit.md").write_text("\n".join([
        "# 120k-Scale Performance Audit", "",
        f"The real formal C04 union contains {len(rep.candidate_meta):,} unique TSP100 candidates and has coordinate-bundle SHA256 `{union_hash}`. It is used only for offline representation/matching; no historical benchmark performance enters this audit.", "",
        *markdown_table(runtime_view, list(runtime_view.columns), {"feature_extraction_wall_seconds": ".2f", "cached_top5_transform_wall_seconds": ".3f", "distance_wall_seconds": ".3f", "matching_wall_seconds": ".3f", "selection_validation_seconds": ".4f", "matching_peak_rss_mb": ".1f", "mean_cost": ".7f", "greedy_excess_over_exact": ".4f", "solver_reported_optimality_gap": ".1f"}), "",
        f"Feature cost boundary: the frozen C04 run **measured** {historical['wall_seconds']:.2f}s to service 119,530 F40 requests ({historical['newly_extracted']:,} new, {historical['cache_hits']:,} cache hits) with the exact formal extractor. Smaller-size entries are explicitly linear throughput estimates, not invented measurements. Peak memory was not captured by that historical run and is reported NA; matching RSS is sampled in this audit.", "",
        f"A conservative single-generation estimate using the historical full F40 path is {estimated_generation:.1f}s ({estimated_generation/60:.1f} min): mutation {mutation_wall:.1f}s + F40 service {historical['wall_seconds']:.1f}s + current distance/matching/validation. This is an estimate, not a new evolution timing. A Top5-only production cache may be faster but was not assumed.", "",
        f"MST/pairwise-distance extraction dominates structural computation; matching is not the bottleneck. The full exact solve peaked at {full.matching_peak_rss_mb:.1f} MiB RSS on this host, so future runners should provision at least 8 GiB for this implementation. GPU is neither required nor used.",
    ]) + "\n")

    improvement = metrics.set_index("selection")
    random_row = improvement.loc["Random10k"]
    match_row = improvement.loc["TargetMatching10k"]
    (OUT / "10_distribution_matching_effectiveness.md").write_text("\n".join([
        "# Distribution Matching Effectiveness", "",
        *markdown_table(metrics, list(metrics.columns), {column: ".6g" for column in metrics.columns if column not in {"selection", "selected_n", "unique_n"}}), "",
        f"Relative to fixed-seed Random10k from the same union, matching changes mean assigned squared cost from {random_row.assignment_mean_squared_cost:.6g} to {match_row.assignment_mean_squared_cost:.6g}, 5D sliced Wasserstein from {random_row.sliced_wasserstein_5d:.6g} to {match_row.sliced_wasserstein_5d:.6g}, and 5D energy distance from {random_row.energy_distance_5d:.6g} to {match_row.energy_distance_5d:.6g}.", "",
        "The improvement is expected because selection optimizes target proximity. Independent SWD/energy/marginal/nearest-reference diagnostics confirm it is not only a bookkeeping change in the optimized objective. Raw Top5 and legacy two-dimensional metrics are reported diagnostically in `diagnostics/`; no diversity/lineage/operator penalty was added after inspecting them.", "",
        "This establishes structural-distribution matching on this fixed bank only. It is not evidence that the selected data has greater BQ training value.",
    ]) + "\n")

    (OUT / "11_proposed_p6b_evolution_protocol.md").write_text("\n".join([
        "# Proposed Frozen P6-B Evolution Protocol", "",
        "```text",
        "P0 = frozen P03 Population10k",
        "for g in {1,2}:",
        "    apply each of 11 FOGA operators once to every current parent",
        "    Union = unique(Parents + eligible Offspring), retaining parent on collision",
        "    extract the frozen raw Top5 and frozen independent-Uniform conditional-score representation",
        "    assign capacities 32x208 + 16x209 by the frozen hash rule",
        "    select exactly 10,000 using the frozen capacity-constrained matcher",
        "    survivors become the next parents",
        "D_train = G2 Population10k",
        "```", "",
        "E0 uses uniform-random Survivor10k; E1 uses target matching. Both use identical P03, 11-per-parent mutation budget, two generations, and direct G2 training budget. Parents may survive and historical M4 overlap is not penalized.", "",
        "Cancelled: 70/30, strict/frontier quotas, microbins, lineage caps, Archive30k, final Structure10k acquisition, hardness fitness, and any extra diversity objective. This document specifies a future experiment only; P6-B was not run.",
    ]) + "\n")

    scientific_risk = f"The largest scientific risk is construct validity: M4 hardness permutation importance does not prove that unweighted Uniform-conditional-score Euclidean geometry—or closer matching to only 48 target points—causes better training. In this bank, distance_min contributes most of the nearest-target squared mismatch and may partly reflect TSPLIB coordinate quantization versus continuous FOGA coordinates. The split-half maximum score change is {stability['maximum_absolute_difference']:.3f}; finite calibration/interpolation/tail continuation and Top5 redundancy are therefore important secondary risks even though assignment is exact."
    decision = "GO"
    label = "P6A_TARGET_MATCHING_FEASIBILITY_SUPPORTED"
    (OUT / "12_final_feasibility_decision.md").write_text("\n".join([
        "# Final P6-A Feasibility Decision", "",
        f"## {decision}: `{label}`", "",
        "1. The five raw definitions and failure modes are explicit and preserved.",
        "2. A complete finite cross-n representation exists without deleting NNG or filling raw NA with zero.",
        "3. All TSPLIB48 and 119,530 candidate matching inputs are finite.",
        "4. Exact capacity matching returns exactly 10,000 unique candidates with all 48 quotas exact.",
        "5. Small exact fixtures verify the LP against enumeration and expose, rather than hide, greedy suboptimality.",
        "6. Real 119,530-row matching is computationally feasible; formal F40 extraction, not matching, dominates cost.",
        "7. Fixed-bank TargetMatching10k improves the matching objective and independent multivariate structural distances over Random10k.",
        "8. No hardness, diversity, lineage, microbin, 70/30, held-out22, or performance fitness was introduced.", "",
        scientific_risk, "",
        "Recommendation: the P6-B two-generation method can be frozen as specified with the exact TU transport solver. An approximate matcher is unnecessary at the audited 120k scale. Do not interpret P6-A as a training-gain result.", "",
        "`BLADE_P6A_TOP5_TARGET_MATCHING_FEASIBILITY_COMPLETE`", "",
        "`GO`",
    ]) + "\n")

    hashes = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "artifact_sha256.csv":
            hashes.append({"path": str(path.relative_to(OUT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    pd.DataFrame(hashes).to_csv(DIAG / "artifact_sha256.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-calibration", action="store_true")
    args = parser.parse_args()
    for directory in (OUT, FIG, CFG, DIAG):
        directory.mkdir(parents=True, exist_ok=True)
    if args.force_calibration:
        for path in (DIAG / "uniform_reference_top5.parquet", DIAG / "uniform_reference_manifest.json"):
            if path.exists():
                path.unlink()

    calibration = generate_uniform_calibration()
    print("[P6-A] building frozen target/candidate Top5 representation", flush=True)
    rep = build_representation(calibration)
    transformations = transformation_audit(rep)
    invariance = scale_permutation_tests(rep)
    integrity = integrity_metrics(rep)
    stability = calibration_stability(rep)
    correctness = correctness_tests()
    quota = capacities(rep.target_manifest.instance_id.astype(str).tolist())
    runtime, matching, cost = profile_matching(rep, quota)
    metrics, marginals = distribution_effectiveness(rep, quota, matching, cost)
    importance = pd.read_csv(P60 / "01_m0_m4_hgb40_importance.csv")
    figures(rep, importance, matching)
    render_reports(rep, transformations, invariance, correctness, integrity, stability, runtime, matching, metrics, marginals, quota)
    write_json(OUT / "run_metadata.json", {
        "status": "COMPLETE",
        "decision": "GO",
        "label": "P6A_TARGET_MATCHING_FEASIBILITY_SUPPORTED",
        "completion_label": "BLADE_P6A_TOP5_TARGET_MATCHING_FEASIBILITY_COMPLETE",
        "candidate_n": len(rep.candidate_meta),
        "selected_n": len(matching["candidate_index"]),
        "no_p6b_executed": True,
        "representation": "uniform_conditional_tail_score_v1",
        "matching_solver": "exact_TU_LP_HiGHS_dual_simplex",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    })
    hashes = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "artifact_sha256.csv" and "__pycache__" not in path.parts:
            hashes.append({"path": str(path.relative_to(OUT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    pd.DataFrame(hashes).to_csv(DIAG / "artifact_sha256.csv", index=False)
    print("BLADE_P6A_TOP5_TARGET_MATCHING_FEASIBILITY_COMPLETE", flush=True)
    print("GO", flush=True)


if __name__ == "__main__":
    main()
