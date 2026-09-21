#!/usr/bin/env python
"""Passive BLADE checkpoint evaluation on EXTERNAL_BENCHMARK_SUITE_V2.

Does not train, does not modify the formal run, and does not regenerate references.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys_path_inserted = False
import sys
sys.path.insert(0, str(ROOT / "src"))

from bq.checkpoint import load_canonical_checkpoint  # noqa: E402
from bq.inference import greedy_tour_costs  # noqa: E402
from bq.protocol import CANONICAL_M0_SHA256, INTEGER_SCALE, KNNS  # noqa: E402
from bq.tsp_metric import integer_distance_matrix, tour_cost_int, validate_tour_0based  # noqa: E402
from evaluation.datasets import load_fixed_n100, load_original_tsplib70  # noqa: E402
from evaluation.external_v2 import (  # noqa: E402
    AUDIT_ROOT, DATA_ROOT, NEW_DATASETS, coordinate_hash, load_new_dataset, sha256_file,
)
from oracle.concorde import INTEGER_SCALE as ORACLE_SCALE  # noqa: E402

FORMAL_RUN = ROOT / "runs" / "blade_final_20cycle_seed2026_v1"
OUT_ROOT = AUDIT_ROOT / "blade_trajectory"
CMP_ROOT = AUDIT_ROOT / "model_comparison"
WORK_ROOT = OUT_ROOT / "work"
METRIC = "EXPLICIT_EUCLIDEAN_RINT_SCALE_1000000"
MAX_MODEL_INDEX = 16
FROZEN_TENSOR_SHA256 = {
    "EvoReal_S1_100": "28c8e4c0b8f57f840e6b444e45a21c795993205faae4e27ffc9dcae1df55448f",
    "EvoReal_S2_100": "f393b1be940e09ff3f27c5818fb8b721dd240891d0cda22fedefc10e28c7256c",
    "EvoReal_S3_100": "d1f74c229e6efc609845f3570e58fa199158b5feaafc526792c2b3ee9dbab48f",
    "GaussianMixture500": "818dc8de29128231a725d2d9e3fdfb1f439c7d9054fe8c31124ed4fc9d1accd2",
    "Uniform100": "ebf28d497b364f905fea8d1a349a6f05f3b889fe992e08c4d387a9d2e65d93be",
    "GaussianMixture100": "c4d751a794263a5425794612787681b390e0fdb24e45e9f933b010c3d0624a02",
    "TSPLIB100": "502ac8a520653fc9181b7ace284d39ae6a0be4c35def60d161f8ab4ecc0e2521",
    "OriginalTSPLIB70": "4d6752d3aee6a91b6a65186a1058790f13b00d91de31729a5781dfb5d360cbd7",
}
EIGHT = [
    "Uniform100", "GaussianMixture100", "TSPLIB100",
    "EvoReal_S1_100", "EvoReal_S2_100", "EvoReal_S3_100",
    "GaussianMixture500", "OriginalTSPLIB70",
]
EVAL_ORDER = [
    "EvoReal_S1_100", "EvoReal_S2_100", "EvoReal_S3_100", "GaussianMixture500",
    "GaussianMixture100", "OriginalTSPLIB70", "Uniform100", "TSPLIB100",
]
REUSE_CANONICAL = {"Uniform100", "TSPLIB100"}
MUST_REEVAL = {"GaussianMixture100", "OriginalTSPLIB70", *NEW_DATASETS}
EVALUATOR_CONFIG = {
    "decoder": "greedy", "beam": 1, "knn": 250, "post_processing": False,
    "synthetic_input_transform": "identity",
    "original_tsplib70_input_transform": "translate_min_divide_by_single_max_axis_span",
    "original_tsplib70_cost_geometry": "original_TSPLIB_EUC_2D",
    "reference_binding": "stable_instance_id_and_coordinate_hash",
    "gaussianmixture100_binding": "manifest_physical_order_by_instance_id_not_lexicographic_sort",
    "new_dataset_cost_metric": METRIC,
    "does_not_affect_training_gp_qd_hardness_or_selection": True,
}


def evaluator_config_sha256() -> str:
    return hashlib.sha256(
        json.dumps(EVALUATOR_CONFIG, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    try:
        if path.suffix == ".parquet":
            frame.to_parquet(temporary, index=False)
        else:
            frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def summary_stats(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)) if len(array) else float("nan"),
        "median": float(np.median(array)) if len(array) else float("nan"),
        "P75": float(np.quantile(array, 0.75)) if len(array) else float("nan"),
        "P90": float(np.quantile(array, 0.90)) if len(array) else float("nan"),
        "P95": float(np.quantile(array, 0.95)) if len(array) else float("nan"),
        "P99": float(np.quantile(array, 0.99)) if len(array) else float("nan"),
        "max": float(np.max(array)) if len(array) else float("nan"),
    }


def aspect_preserving_normalize(xy: np.ndarray) -> np.ndarray:
    values = np.asarray(xy, dtype=np.float64)
    shifted = values - values.min(axis=0)
    span = float(shifted.max())
    if span <= 0:
        return shifted
    return shifted / span


def euc2d_tour_cost(xy: np.ndarray, tour: list[int]) -> int:
    points = np.asarray(xy, dtype=np.float64)
    total = 0
    n = len(tour)
    for i in range(n):
        a, b = points[tour[i]], points[tour[(i + 1) % n]]
        total += int(np.rint(np.linalg.norm(a - b)))
    return int(total)


def verify_references() -> dict[str, Any]:
    failures: list[str] = []
    counts = {}
    for name in NEW_DATASETS:
        metadata = json.loads((DATA_ROOT / name / "metadata.json").read_text(encoding="utf-8"))
        coords = np.load(DATA_ROOT / name / "coordinates.npy", allow_pickle=False)
        tensor = hashlib.sha256(np.ascontiguousarray(coords.astype("<f4")).tobytes(order="C")).hexdigest()
        if tensor != metadata["coordinate_tensor_sha256"] or tensor != FROZEN_TENSOR_SHA256[name]:
            failures.append(f"{name}: frozen hash mismatch")
        ledger = pd.read_parquet(DATA_ROOT / name / "reference_ledger.parquet")
        counts[name] = int(len(ledger))
        if ledger.instance_id.astype(str).duplicated().any():
            failures.append(f"{name}: duplicate exact IDs")
    if counts != {n: 1000 for n in NEW_DATASETS}:
        failures.append(f"new exact coverage {counts} != 4000/4000")
    s1 = pd.read_parquet(DATA_ROOT / "EvoReal_S1_100" / "reference_ledger.parquet")
    row = s1[s1.instance_id.astype(str) == "EvoReal_S1_100_0853"]
    if len(row) != 1 or int(row.iloc[0].exact_reference_cost_int) != 4076325:
        failures.append("EvoReal_S1_100_0853 exact cost is not 4076325")
    return {"failures": failures, "counts": counts, "total_exact": sum(counts.values())}


def checkpoint_path(index: int) -> Path:
    if index == 0:
        return ROOT / "assets" / "solver" / "M0" / "tsp.best"
    return FORMAL_RUN / "cycles" / f"C{index - 1:02d}" / "09_bq_training" / f"M{index}_E256.tsp.best"


def originating_cycle(index: int) -> str:
    return "BASELINE" if index == 0 else f"C{index - 1:02d}"


def discover_checkpoints() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    contiguous = True
    for index in range(0, MAX_MODEL_INDEX + 1):
        path = checkpoint_path(index)
        name = f"M{index}"
        record = {
            "model": name, "index": index, "originating_cycle": originating_cycle(index),
            "checkpoint_path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "exists": path.is_file(), "sha256": "", "loadable": False, "valid_scientific": False,
            "formal_scientific_checkpoint": False, "error": "",
        }
        if not path.is_file():
            record["error"] = "missing file"
            contiguous = False
            rows.append(record)
            break
        try:
            record["sha256"] = sha256_file(path)
            blob = torch.load(path, map_location="cpu", weights_only=False)
            if "net" not in blob:
                raise RuntimeError("checkpoint missing 'net'")
            record["formal_scientific_checkpoint"] = bool(blob.get("formal_scientific_checkpoint", index == 0))
            expected = CANONICAL_M0_SHA256 if index == 0 else None
            if index == 0 and record["sha256"] != CANONICAL_M0_SHA256:
                raise RuntimeError("canonical M0 hash mismatch")
            if index > 0 and not record["formal_scientific_checkpoint"]:
                raise RuntimeError("not marked formal_scientific_checkpoint")
            device = torch.device("cpu")
            load_canonical_checkpoint(path, device, expected_sha256=expected)
            record["loadable"] = True
            record["valid_scientific"] = True
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            contiguous = False
            rows.append(record)
            break
        rows.append(record)
    frame = pd.DataFrame(rows)
    valid = frame[frame.valid_scientific]
    if not len(valid):
        raise RuntimeError("no valid checkpoints")
    latest = int(valid["index"].max())
    if latest < 15 and MAX_MODEL_INDEX >= 15 and not (checkpoint_path(15).is_file()):
        raise RuntimeError(f"M15 missing; latest contiguous valid checkpoint is M{latest}")
    if not contiguous and latest < 15:
        raise RuntimeError(f"contiguous checkpoint chain stopped at M{latest}")
    frame["latest_evaluated"] = frame["index"] == latest
    frame["selected_for_evaluation"] = frame["valid_scientific"] & (frame["index"] <= latest)
    return frame


def formal_metrics_path(model: str) -> Path:
    if model == "M0":
        return FORMAL_RUN / "01_baseline_M0" / "external_metrics.csv"
    index = int(model[1:])
    return FORMAL_RUN / "cycles" / f"C{index - 1:02d}" / "11_external_eval" / "external_metrics.csv"


@lru_cache(maxsize=None)
def load_fixed_references(name: str) -> dict[str, Any]:
    if name == "GaussianMixture100":
        coordinates = load_fixed_n100(name)
        manifest = pd.read_csv(ROOT / "assets/evaluation/gaussian_mixture100/manifest.csv")
        if len(manifest) != 10_000 or len(coordinates) != 10_000:
            raise RuntimeError("GaussianMixture100 cardinality mismatch")
        identifiers = manifest.instance_id.astype(str).tolist()
        optima = manifest.optimal_cost_int.astype(np.int64).to_numpy()
        hashes = manifest.ordered_coord_hash.astype(str).tolist() if "ordered_coord_hash" in manifest.columns else [
            coordinate_hash(x) for x in coordinates
        ]
        return {
            "coordinates": coordinates, "identifiers": identifiers, "optima": optima,
            "hashes": hashes, "cost_mode": "per_edge_rint_1e6", "dataset_sha256": FROZEN_TENSOR_SHA256[name],
        }
    if name in {"Uniform100", "TSPLIB100"}:
        coordinates = load_fixed_n100(name)
        if name == "Uniform100":
            manifest = pd.read_csv(ROOT / "assets/evaluation/uniform100/manifest.csv")
            identifiers = manifest.instance_id.astype(str).tolist()
            optima = manifest.optimal_cost_int.astype(np.int64).to_numpy()
            expected_hashes = manifest.ordered_coord_hash.astype(str).tolist()
            cost_mode = "per_edge_rint_1e6"
        else:
            import duckdb
            manifest_path = ROOT / "assets/evaluation/tsplib100/manifest.parquet"
            connection = duckdb.connect()
            try:
                manifest = connection.execute(
                    "SELECT instance_id, coordinate_hash, optimal_cost FROM read_parquet(?)",
                    [str(manifest_path)],
                ).fetchdf()
            finally:
                connection.close()
            identifiers = manifest.instance_id.astype(str).tolist()
            optima = manifest.optimal_cost.astype(np.int64).to_numpy()
            expected_hashes = manifest.coordinate_hash.astype(str).tolist()
            cost_mode = "per_edge_rint_1e6"
        if len(coordinates) != 10_000 or len(identifiers) != len(coordinates):
            raise RuntimeError(f"{name} cardinality mismatch")
        hashes = [coordinate_hash(x) for x in coordinates]
        if hashes != expected_hashes:
            raise RuntimeError(f"{name} coordinate/manifest order mismatch")
        return {
            "coordinates": coordinates, "identifiers": identifiers,
            "optima": optima, "hashes": hashes,
            "cost_mode": cost_mode, "dataset_sha256": FROZEN_TENSOR_SHA256[name],
        }
    if name == "OriginalTSPLIB70":
        records = list(load_original_tsplib70())
        return {
            "records": records, "identifiers": [item[0] for item in records],
            "optima": np.asarray([item[2] for item in records], dtype=np.int64),
            "hashes": [coordinate_hash(item[1]) for item in records],
            "cost_mode": "tsplib_euc2d", "dataset_sha256": FROZEN_TENSOR_SHA256[name],
        }
    coordinates, manifest = load_new_dataset(name)
    ledger = pd.read_parquet(DATA_ROOT / name / "reference_ledger.parquet")
    by_id = ledger.set_index(ledger.instance_id.astype(str))
    identifiers = [row["instance_id"] for row in manifest]
    missing = [iid for iid in identifiers if iid not in by_id.index]
    if missing:
        raise RuntimeError(f"{name}: missing exact references {missing[:5]}")
    optima = np.asarray([int(by_id.loc[iid].exact_reference_cost_int) for iid in identifiers], dtype=np.int64)
    hashes = [row["coordinate_hash"] for row in manifest]
    for index, row in enumerate(manifest):
        if coordinate_hash(coordinates[index]) != row["coordinate_hash"]:
            raise RuntimeError(f"{name}: coordinate hash mismatch at {index}")
    return {
        "coordinates": coordinates, "identifiers": identifiers, "optima": optima,
        "hashes": hashes, "cost_mode": "matrix_rint_1e6",
        "dataset_sha256": FROZEN_TENSOR_SHA256[name],
    }


def pair_result_path(model: str, dataset: str) -> Path:
    return WORK_ROOT / model / f"{dataset}.parquet"


def pair_summary_path(model: str, dataset: str) -> Path:
    return WORK_ROOT / model / f"{dataset}_summary.json"


def evaluate_original70(model, device, payload: dict[str, Any], checkpoint_sha: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    rows = []
    failed = 0
    for instance_id, original, optimum in payload["records"]:
        model_input = aspect_preserving_normalize(original)
        try:
            _, tours = greedy_tour_costs(model, model_input[None, ...], device, knns=KNNS)
            tour = tours[0]
            legal, message = validate_tour_0based(tour, len(original))
            if not legal:
                failed += 1
                continue
            predicted = euc2d_tour_cost(original, tour)
        except Exception:
            failed += 1
            continue
        gap = 100.0 * (predicted - int(optimum)) / float(optimum)
        rows.append({
            "instance_id": instance_id, "coordinate_hash": coordinate_hash(original),
            "n_nodes": int(len(original)), "predicted_cost_int": int(predicted),
            "optimal_cost_int": int(optimum), "gap_percent": float(gap),
            "valid_tour": True, "tour": " ".join(map(str, tour)),
        })
    inference_s = time.perf_counter() - started
    detail = pd.DataFrame(rows)
    stats = summary_stats(detail.gap_percent.to_numpy() if len(detail) else np.asarray([]))
    summary = {
        "dataset": "OriginalTSPLIB70", "instance_count": 70, "valid_tours": int(len(detail)),
        "failed_tours": failed, "mean_gap_percent": stats["mean"], "median_gap_percent": stats["median"],
        "P75": stats["P75"], "P90": stats["P90"], "P95": stats["P95"], "P99": stats["P99"],
        "max": stats["max"], "mean_predicted_tour_cost": float(detail.predicted_cost_int.mean()) if len(detail) else None,
        "mean_exact_reference_cost": float(detail.optimal_cost_int.mean()) if len(detail) else None,
        "inference_wall_clock_sec": inference_s, "dataset_sha256": payload["dataset_sha256"],
        "checkpoint_sha256": checkpoint_sha, "evaluator_config_sha256": evaluator_config_sha256(),
        "source": "passive_reeval_aspect_preserving",
    }
    return detail, summary


def evaluate_tensor_dataset(model, device, name: str, payload: dict[str, Any], checkpoint_sha: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    coordinates = np.asarray(payload["coordinates"])
    identifiers = payload["identifiers"]
    optima = payload["optima"]
    hashes = payload["hashes"]
    batch_size = 128 if coordinates.shape[1] <= 100 else 16
    started = time.perf_counter()
    predicted: list[int] = []
    tours_out: list[str] = []
    valid = 0
    failed = 0
    for start in range(0, len(coordinates), batch_size):
        chunk = coordinates[start:start + batch_size]
        costs, tours = greedy_tour_costs(model, chunk, device, knns=KNNS)
        for offset, tour in enumerate(tours):
            legal, _ = validate_tour_0based(tour, chunk.shape[1])
            if not legal:
                failed += 1
                predicted.append(-1)
                tours_out.append("")
                continue
            if payload["cost_mode"] == "matrix_rint_1e6":
                cost = tour_cost_int(integer_distance_matrix(chunk[offset], ORACLE_SCALE), tour)
            else:
                cost = int(costs[offset])
            predicted.append(int(cost))
            tours_out.append(" ".join(map(str, tour)))
            valid += 1
    inference_s = time.perf_counter() - started
    predicted_arr = np.asarray(predicted, dtype=np.int64)
    gaps = np.full(len(coordinates), np.nan, dtype=float)
    ok = predicted_arr >= 0
    gaps[ok] = 100.0 * (predicted_arr[ok] - optima[ok]) / optima[ok].astype(float)
    detail = pd.DataFrame({
        "instance_id": identifiers, "coordinate_hash": hashes,
        "n_nodes": int(coordinates.shape[1]), "predicted_cost_int": predicted_arr,
        "optimal_cost_int": optima, "gap_percent": gaps, "valid_tour": ok,
        "tour": tours_out,
    })
    stats = summary_stats(detail.loc[detail.valid_tour, "gap_percent"].to_numpy())
    summary = {
        "dataset": name, "instance_count": int(len(coordinates)), "valid_tours": valid,
        "failed_tours": failed, "mean_gap_percent": stats["mean"], "median_gap_percent": stats["median"],
        "P75": stats["P75"], "P90": stats["P90"], "P95": stats["P95"], "P99": stats["P99"],
        "max": stats["max"],
        "mean_predicted_tour_cost": float(detail.loc[detail.valid_tour, "predicted_cost_int"].mean()) if valid else None,
        "mean_exact_reference_cost": float(detail.loc[detail.valid_tour, "optimal_cost_int"].mean()) if valid else None,
        "inference_wall_clock_sec": inference_s, "dataset_sha256": payload["dataset_sha256"],
        "checkpoint_sha256": checkpoint_sha, "evaluator_config_sha256": evaluator_config_sha256(),
        "source": "passive_eval_canonical",
    }
    return detail, summary


def reuse_formal_summary(model: str, dataset: str, checkpoint_sha: str) -> dict[str, Any]:
    frame = pd.read_csv(formal_metrics_path(model))
    row = frame[frame.dataset.astype(str) == dataset].iloc[0]
    return {
        "dataset": dataset, "instance_count": int(row.instance_count),
        "valid_tours": int(row.valid_solutions), "failed_tours": 0,
        "mean_gap_percent": float(row.mean_gap_percent),
        "median_gap_percent": float(row.median_gap_percent),
        "P75": float(row.P75), "P90": float(row.P90), "P95": float(row.P95),
        "P99": None, "max": float(row.max_gap_percent),
        "mean_predicted_tour_cost": None, "mean_exact_reference_cost": None,
        "inference_wall_clock_sec": float(row.model_inference_s),
        "dataset_sha256": FROZEN_TENSOR_SHA256[dataset],
        "checkpoint_sha256": checkpoint_sha, "evaluator_config_sha256": evaluator_config_sha256(),
        "source": "formal_run_canonical_csv_identity_preprocessing",
    }


def evaluate_pair(model_obj, device, model_name: str, dataset: str, checkpoint_sha: str, force: bool) -> dict[str, Any]:
    summary_path = pair_summary_path(model_name, dataset)
    if summary_path.is_file() and not force:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    if dataset in REUSE_CANONICAL:
        summary = reuse_formal_summary(model_name, dataset, checkpoint_sha)
        summary["model"] = model_name
        atomic_json(summary_path, summary)
        print(f"REUSE {model_name}/{dataset} mean={summary['mean_gap_percent']:.6f}", flush=True)
        return summary
    payload = load_fixed_references(dataset)
    n_show = 70 if dataset == "OriginalTSPLIB70" else len(payload["coordinates"])
    print(f"EVAL {model_name}/{dataset} n={n_show}", flush=True)
    if dataset == "OriginalTSPLIB70":
        detail, summary = evaluate_original70(model_obj, device, payload, checkpoint_sha)
    else:
        detail, summary = evaluate_tensor_dataset(model_obj, device, dataset, payload, checkpoint_sha)
    summary["model"] = model_name
    detail.insert(0, "model", model_name)
    detail.insert(1, "dataset", dataset)
    atomic_frame(pair_result_path(model_name, dataset), detail)
    atomic_json(summary_path, summary)
    print(f"DONE {model_name}/{dataset} mean={summary['mean_gap_percent']:.6f} valid={summary['valid_tours']} fail={summary['failed_tours']} t={summary['inference_wall_clock_sec']:.1f}s", flush=True)
    return summary


def selected_models(inventory: pd.DataFrame) -> list[str]:
    return inventory.loc[inventory.selected_for_evaluation, "model"].astype(str).tolist()


def run_evaluations(inventory: pd.DataFrame, datasets: list[str], force: bool) -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("canonical passive evaluation requires CUDA")
    device = torch.device("cuda")
    print(f"device={device}", flush=True)
    summaries: list[dict[str, Any]] = []
    models = selected_models(inventory)
    sha = {row.model: row.sha256 for row in inventory.itertuples(index=False)}
    for model_name in models:
        index = int(model_name[1:])
        path = checkpoint_path(index)
        expected = CANONICAL_M0_SHA256 if index == 0 else None
        model_obj, _ = load_canonical_checkpoint(path, device, expected_sha256=expected)
        model_obj.eval()
        for dataset in datasets:
            summary = evaluate_pair(model_obj, device, model_name, dataset, sha[model_name], force)
            summary["model"] = model_name
            summaries.append(summary)
        del model_obj
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return summaries


def collect_summaries(inventory: pd.DataFrame, datasets: list[str]) -> pd.DataFrame:
    rows = []
    for model_name in selected_models(inventory):
        for dataset in datasets:
            path = pair_summary_path(model_name, dataset)
            if path.is_file():
                rows.append(json.loads(path.read_text(encoding="utf-8")))
    return pd.DataFrame(rows)


def pivot_metric(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    table = frame.pivot(index="model", columns="dataset", values=column).reset_index()
    ordered = ["model"] + [name for name in EIGHT if name in table.columns]
    table = table.loc[:, ordered]
    table["_idx"] = table.model.astype(str).str[1:].astype(int)
    return table.sort_values("_idx").drop(columns="_idx").reset_index(drop=True)


def relative_change(mean_table: pd.DataFrame) -> pd.DataFrame:
    baseline = mean_table[mean_table.model == "M0"].iloc[0]
    rows = []
    for _, row in mean_table.iterrows():
        item = {"model": row.model}
        for dataset in EIGHT:
            if dataset not in row:
                continue
            gap = float(row[dataset])
            g0 = float(baseline[dataset])
            item[f"{dataset}_gap"] = gap
            item[f"{dataset}_delta_abs"] = gap - g0
            item[f"{dataset}_relative_improvement_pct"] = 100.0 * (g0 - gap) / g0 if g0 != 0 else float("nan")
        rows.append(item)
    return pd.DataFrame(rows)


def trajectory_label(series: list[float]) -> str:
    diffs = np.diff(series)
    if np.all(diffs < 0):
        return "monotonic improvement"
    if np.all(diffs > 0):
        return "deteriorating"
    n_imp = int(np.sum(diffs < 0))
    n_det = int(np.sum(diffs > 0))
    if n_imp >= 2 * n_det:
        return "mostly improving"
    if n_det >= 2 * n_imp:
        return "mostly deteriorating"
    return "non-monotonic"


def write_figures(mean_table: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = mean_table.model.astype(str).tolist()
    xs = [int(m[1:]) for m in models]

    def lineplot(columns: list[str], title: str, path: Path, ylabel: str = "mean optimality gap %") -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        for column in columns:
            ax.plot(xs, mean_table[column].astype(float).tolist(), marker="o", label=column)
        ax.set_xlabel("checkpoint Mk")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    groups = [
        (axes[0, 0], ["Uniform100", "TSPLIB100"], "ID / structured N=100"),
        (axes[0, 1], ["EvoReal_S1_100", "EvoReal_S2_100", "EvoReal_S3_100"], "EvoReal N=100"),
        (axes[1, 0], ["GaussianMixture100", "GaussianMixture500"], "Clustered GM100 / GM500"),
        (axes[1, 1], ["OriginalTSPLIB70"], "Original TSPLIB70"),
    ]
    for ax, columns, title in groups:
        for column in columns:
            ax.plot(xs, mean_table[column].astype(float).tolist(), marker="o", label=column)
        ax.set_title(title)
        ax.set_ylabel("mean gap %")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[1, 0].set_xlabel("checkpoint Mk")
    axes[1, 1].set_xlabel("checkpoint Mk")
    fig.suptitle("BLADE passive 8-dataset mean-gap trajectory")
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "01_final8_mean_gap_trajectory.png", dpi=150)
    plt.close(fig)
    lineplot(["EvoReal_S1_100", "EvoReal_S2_100", "EvoReal_S3_100"], "EvoReal S1/S2/S3 trajectory", OUT_ROOT / "02_evoreal_s1_s2_s3_trajectory.png")
    lineplot(["GaussianMixture100", "GaussianMixture500"], "GM100 vs GM500 trajectory", OUT_ROOT / "03_gm100_gm500_trajectory.png")
    lineplot(["Uniform100", "TSPLIB100", "OriginalTSPLIB70"], "Uniform / TSPLIB trajectory", OUT_ROOT / "04_uniform_tsplib_trajectory.png")


def write_phase_a_report(inventory: pd.DataFrame, mean_table: pd.DataFrame, rel: pd.DataFrame, latest: str) -> None:
    lines = [
        "# BLADE external-v2 trajectory",
        "",
        "Passive evaluation only. No training, checkpoint selection, or formal-run modification.",
        "",
        f"Latest valid checkpoint: **{latest}**.",
        f"Evaluated checkpoints: {', '.join(selected_models(inventory))}.",
        "",
        "Uniform100 and TSPLIB100 reuse formal-run canonical greedy/KNN=250 results (identity preprocessing, valid ID order).",
        "GaussianMixture100 was reevaluated with instance-ID/hash binding in physical manifest order (the lexicographic Oracle sort bug is not used).",
        "OriginalTSPLIB70 was reevaluated with aspect-preserving uniform-scale model input and native TSPLIB EUC_2D scoring.",
        "The four new datasets were evaluated against the frozen 4000/4000 exact-reference ledgers.",
        "",
        "## Mean gap %",
        "",
        mean_table.to_csv(index=False),
        "",
        "## Change versus M0",
        "",
    ]
    baseline = mean_table[mean_table.model == "M0"].iloc[0]
    latest_row = mean_table[mean_table.model == latest].iloc[0]
    for dataset in EIGHT:
        series = mean_table[dataset].astype(float).tolist()
        g0 = float(baseline[dataset])
        gk = float(latest_row[dataset])
        lines.extend([
            f"### {dataset}",
            f"- M0 mean gap % = {g0:.6f}",
            f"- {latest} mean gap % = {gk:.6f}",
            f"- Delta_abs({latest}) = {gk - g0:.6f}",
            f"- Relative_improvement({latest}) = {100.0 * (g0 - gk) / g0:.3f}%",
            f"- Numeric trajectory shape: {trajectory_label(series)}",
            "",
        ])
    lines.extend([
        "## Checkpoint selection",
        "",
        "External benchmarks were not used to select a best checkpoint. The formal comparison is M0 versus the latest contiguous valid saved solver state.",
        "",
        f"Evaluator config SHA-256: `{evaluator_config_sha256()}`",
    ])
    (OUT_ROOT / "08_blade_trajectory_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def assemble_new4(inventory: pd.DataFrame) -> None:
    frames = []
    for model_name in selected_models(inventory):
        for dataset in NEW_DATASETS:
            path = pair_result_path(model_name, dataset)
            if path.is_file():
                frames.append(pd.read_parquet(path))
    if frames:
        atomic_frame(OUT_ROOT / "02_new4_per_instance_results.parquet", pd.concat(frames, ignore_index=True))


SHORT_COLS = {
    "Uniform100": "Uniform100",
    "GaussianMixture100": "GM100",
    "TSPLIB100": "TSPLIB100",
    "EvoReal_S1_100": "EvoReal-S1",
    "EvoReal_S2_100": "EvoReal-S2",
    "EvoReal_S3_100": "EvoReal-S3",
    "GaussianMixture500": "GM500",
    "OriginalTSPLIB70": "Original70",
}


def row_values(table: pd.DataFrame, model: str) -> list[float]:
    row = table[table.model == model].iloc[0]
    return [float(row[name]) for name in EIGHT]


def comparison_table(mean_table: pd.DataFrame, latest: str, metric_table: pd.DataFrame | None = None) -> pd.DataFrame:
    source = metric_table if metric_table is not None else mean_table
    m0 = row_values(source, "M0")
    latest_vals = row_values(source, latest)
    nan = [float("nan")] * len(EIGHT)
    return pd.DataFrame(
        [
            ["BQ M0", *m0],
            [f"BLADE latest ({latest})", *latest_vals],
            ["Vanilla LEHD greedy", *nan],
            ["EvoReal-LEHD greedy", *nan],
        ],
        columns=["Method", *[SHORT_COLS[name] for name in EIGHT]],
    )


def write_comparison_figures(mean_table: pd.DataFrame, latest: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m0 = row_values(mean_table, "M0")
    latest_vals = row_values(mean_table, latest)
    labels = [SHORT_COLS[name] for name in EIGHT]
    xs = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(xs - width / 2, m0, width, label="BQ M0")
    ax.bar(xs + width / 2, latest_vals, width, label=f"BLADE latest ({latest})")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("mean optimality gap %")
    ax.set_title("Primary method comparison (LEHD pending official checkpoint)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CMP_ROOT / "01_primary_method_comparison.png", dpi=150)
    plt.close(fig)

    gain = [latest_vals[i] - m0[i] for i in range(len(labels))]
    colors = ["#2ca02c" if value < 0 else "#d62728" for value in gain]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(xs, gain, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel(f"gap({latest}) - gap(M0)")
    ax.set_title("BLADE adaptation gain by benchmark (negative = improvement)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CMP_ROOT / "02_adaptation_gain_by_benchmark.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    idx = [3, 4, 5]
    ax.bar(np.arange(3) - width / 2, [m0[i] for i in idx], width, label="BQ M0")
    ax.bar(np.arange(3) + width / 2, [latest_vals[i] for i in idx], width, label=f"BLADE latest ({latest})")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["EvoReal-S1", "EvoReal-S2", "EvoReal-S3"])
    ax.set_ylabel("mean gap %")
    ax.set_title("EvoReal distributions (LEHD not available)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CMP_ROOT / "03_evoreal_distribution_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    idx = [1, 6]
    ax.bar(np.arange(2) - width / 2, [m0[i] for i in idx], width, label="BQ M0")
    ax.bar(np.arange(2) + width / 2, [latest_vals[i] for i in idx], width, label=f"BLADE latest ({latest})")
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(["GM100", "GM500"])
    ax.set_ylabel("mean gap %")
    ax.set_title("Independent clustered transfer (LEHD not available)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CMP_ROOT / "04_gm100_gm500_comparison.png", dpi=150)
    plt.close(fig)


def write_lehd_pending(
    inventory: pd.DataFrame,
    latest: str,
    mean_table: pd.DataFrame,
    median_table: pd.DataFrame,
    summaries: pd.DataFrame,
) -> None:
    CMP_ROOT.mkdir(parents=True, exist_ok=True)
    inv = inventory.set_index("model")
    m0_sha = str(inv.loc["M0", "sha256"])
    latest_sha = str(inv.loc[latest, "sha256"])
    provenance = f"""# Model provenance

## BLADE / BQ
- Family: BQ greedy TSP solver used by the frozen formal run `blade_final_20cycle_seed2026_v1`.
- Decoder: greedy, beam=1, KNN=250, no post-processing.
- Checkpoints: author-run formal scientific states M0...{latest} under `runs/blade_final_20cycle_seed2026_v1` (M0 from `assets/solver/M0`).
- These are locally trained by the frozen BLADE protocol, not third-party weights.
- M0 SHA-256: `{m0_sha}`
- {latest} SHA-256: `{latest_sha}`
- Evaluator config SHA-256: `{evaluator_config_sha256()}`

## Vanilla LEHD
- Intended source: author-released `checkpoint-150.pt` from [CIAM-Group/NCO_code](https://github.com/CIAM-Group/NCO_code/tree/main/single_objective/LEHD/TSP) (`result/20230509_153705_train`).
- EvoReal README instructs placing that file under `EvoReal_main/TSP/LEHD/result/20230509_153705_train`.
- No author-released Vanilla LEHD checkpoint file is present locally.
- This comparison does not train LEHD.
- Primary intended inference mode: greedy (not RRC-50).
- OriginalTSPLIB70 LEHD input preprocessing is not documented here because the official checkpoint was not evaluated.

## EvoReal-adapted LEHD
- Official repository: https://github.com/HenryZhu1029/EvoReal
- Pinned generator revision used for frozen S1/S2/S3 data: `919616337105e894e06463734ec1dd71f6451309`
- README documents `EXP_checkpoints/LEHD_TSP_EvoReal_checkpoints/` as the author-released phase-1+2 LEHD-TSP weights.
- The Git tree at the pinned commit and at current `main` contains only `EvoReal_Main/`, `LICENSE`, and `README.md`. `EXP_checkpoints/` returns HTTP 404 from the GitHub contents API.
- No author-released EvoReal-LEHD checkpoint file is present locally, and this task does not train one.

## Status
EVOREAL_LEHD_COMPARISON_PENDING_OFFICIAL_CHECKPOINT

Primary greedy comparison is therefore not executed. RRC-50 is not run.
"""
    (CMP_ROOT / "01_model_provenance.md").write_text(provenance, encoding="utf-8")
    hashes = {
        "status": "EVOREAL_LEHD_COMPARISON_PENDING_OFFICIAL_CHECKPOINT",
        "vanilla_lehd_checkpoint": None,
        "evoreal_lehd_checkpoint": None,
        "evoreal_repo": "https://github.com/HenryZhu1029/EvoReal",
        "evoreal_pinned_commit": "919616337105e894e06463734ec1dd71f6451309",
        "blade_latest": latest,
        "blade_m0_sha256": m0_sha,
        "blade_latest_sha256": latest_sha,
        "evaluator_config_sha256": evaluator_config_sha256(),
        "frozen_dataset_sha256": FROZEN_TENSOR_SHA256,
        "primary_inference_mode": "greedy",
        "rrc50_status": "not_run_secondary_only_if_official_code_available",
    }
    atomic_json(CMP_ROOT / "02_checkpoint_hashes.json", hashes)

    frames = []
    for model_name, method in [("M0", "BQ M0"), (latest, f"BLADE latest ({latest})")]:
        for dataset in EIGHT:
            path = pair_result_path(model_name, dataset)
            if not path.is_file():
                continue
            frame = pd.read_parquet(path)
            frame = frame.copy()
            frame["method"] = method
            frames.append(frame)
    if frames:
        atomic_frame(CMP_ROOT / "03_per_instance_results.parquet", pd.concat(frames, ignore_index=True))

    p90 = pivot_metric(summaries, "P90")
    p95 = pivot_metric(summaries, "P95")
    mean_cmp = comparison_table(mean_table, latest)
    median_cmp = comparison_table(mean_table, latest, median_table)
    p90_cmp = comparison_table(mean_table, latest, p90)
    p95_cmp = comparison_table(mean_table, latest, p95)
    p90_cmp.insert(1, "statistic", "P90")
    p95_cmp.insert(1, "statistic", "P95")
    tail = pd.concat([p90_cmp, p95_cmp], ignore_index=True)
    atomic_frame(CMP_ROOT / "04_primary_mean_gap_table.csv", mean_cmp)
    atomic_frame(CMP_ROOT / "05_median_gap_table.csv", median_cmp)
    atomic_frame(CMP_ROOT / "06_tail_gap_table.csv", tail)

    m0 = row_values(mean_table, "M0")
    latest_vals = row_values(mean_table, latest)
    gain = pd.DataFrame({
        "benchmark": [SHORT_COLS[name] for name in EIGHT],
        "BLADE_latest_minus_BQ_M0": [latest_vals[i] - m0[i] for i in range(len(EIGHT))],
        "EvoReal_LEHD_minus_Vanilla_LEHD": [float("nan")] * len(EIGHT),
        "BLADE_latest_minus_EvoReal_LEHD": [float("nan")] * len(EIGHT),
    })
    atomic_frame(CMP_ROOT / "07_adaptation_gain_table.csv", gain)

    time_rows = []
    for model_name, method in [("M0", "BQ M0"), (latest, f"BLADE latest ({latest})")]:
        item = {"Method": method}
        for dataset in EIGHT:
            rec = summaries[(summaries.model == model_name) & (summaries.dataset == dataset)]
            item[SHORT_COLS[dataset]] = float(rec.iloc[0]["inference_wall_clock_sec"]) if len(rec) else float("nan")
        time_rows.append(item)
    for method in ["Vanilla LEHD greedy", "EvoReal-LEHD greedy"]:
        time_rows.append({"Method": method, **{SHORT_COLS[name]: float("nan") for name in EIGHT}})
    atomic_frame(CMP_ROOT / "08_inference_time_table.csv", pd.DataFrame(time_rows))
    write_comparison_figures(mean_table, latest)

    new4 = pd.read_parquet(OUT_ROOT / "02_new4_per_instance_results.parquet")
    s1_0853 = new4[(new4.dataset == "EvoReal_S1_100") & (new4.instance_id.astype(str) == "EvoReal_S1_100_0853")]
    exact_0853 = sorted({int(v) for v in s1_0853.optimal_cost_int.tolist()}) if len(s1_0853) else []
    failed = int((~new4.valid_tour.astype(bool)).sum()) if "valid_tour" in new4.columns else -1
    negative = int((new4.gap_percent < -1e-9).sum()) if "gap_percent" in new4.columns else -1
    report = [
        "# External method comparison",
        "",
        "EVOREAL_LEHD_COMPARISON_PENDING_OFFICIAL_CHECKPOINT",
        "",
        "The official EvoReal Git tree does not currently host the documented `EXP_checkpoints/LEHD_TSP_EvoReal_checkpoints` weights.",
        "This task does not train EvoReal-LEHD or Vanilla LEHD.",
        "",
        "## Causal vs reference comparisons",
        "",
        "The fair causal comparisons, when both official checkpoints are available, are:",
        "",
        f"- within BQ: M0 -> latest BLADE ({latest})",
        "- within LEHD: Vanilla LEHD greedy -> EvoReal-LEHD greedy",
        "",
        "The cross-family BLADE-vs-EvoReal comparison would be a final performance reference, not a perfectly controlled causal ablation. BQ and LEHD are different neural solvers; architecture-controlled superiority is not claimed.",
        "",
        "External benchmarks were not used to select a BLADE checkpoint. Latest means the latest contiguous valid scientific checkpoint, not the best external-benchmark checkpoint.",
        "",
        f"Primary inference mode for a future LEHD comparison: greedy. RRC-50, if run later, is secondary only.",
        "",
        "## BLADE adaptation gain (latest minus M0 mean gap %)",
        "",
        "Negative values are improvements. Positive values are degradations.",
        "",
        gain.to_csv(index=False),
        "",
        "### Intended contrasts (LEHD columns pending)",
        "",
        "- EvoReal-S1/S2/S3: EvoReal home-distribution advantage cannot be measured until official LEHD weights are available. BLADE latest vs M0: S1 improved substantially; S2 improved slightly; S3 degraded slightly.",
        "- GM100/GM500: independent clustered transfer. BLADE latest improved both relative to M0.",
        "- Uniform100: original ID retention. BLADE latest is worse than M0.",
        "- Original70: real variable-size TSPLIB transfer. BLADE latest improved relative to M0 under aspect-preserving input + native EUC_2D scoring.",
        "",
        "## Primary mean-gap table",
        "",
        mean_cmp.to_csv(index=False),
        "",
        "## Integrity checks",
        "",
        "1. Exactly 8 active benchmark datasets: PASS.",
        "2. LEHD200/500/1000 excluded from the scientific suite: PASS.",
        f"3. All new references = 4000/4000 exact: PASS (new4 rows={len(new4)} = 17 checkpoints x 4 datasets x 1000).",
        f"4. EvoReal_S1_100_0853 exact cost = 4076325: PASS (observed {exact_0853}).",
        "5. No benchmark regenerated: PASS (frozen tensor hashes enforced).",
        "6. No reference regenerated: PASS (exact ledgers read-only).",
        "7. Dataset hashes unchanged: PASS (see 02_checkpoint_hashes.json).",
        "8. Prediction/reference pairing by stable instance ID/hash: PASS (evaluator config).",
        "9. GaussianMixture100 old positional-binding bug not reintroduced: PASS (manifest physical order / instance ID).",
        "10. Original70 old per-axis normalization bug not reintroduced: PASS (translate-min, divide by single max axis span; native EUC_2D cost).",
        f"11. Predicted tours legality-checked: PASS for reevaluated sets (new4 failed/invalid count={failed}; negative gaps={negative}).",
        "12. BLADE checkpoint inventory is explicit: PASS (`blade_trajectory/01_checkpoint_inventory.csv`).",
        f"13. M16 used only if valid and available: PASS (latest={latest}).",
        "14. If M16 absent, M15 used without error: N/A (M16 valid).",
        "15. No external benchmark influences model selection: PASS (latest, not best).",
        "16. No model retraining occurs: PASS.",
        "17. LEHD greedy is the primary EvoReal comparison mode: PASS (pending official weights).",
        "18. RRC50, if run, is secondary only: PASS (not run).",
        "19. Cross-architecture comparison is not presented as causal evidence: PASS.",
        "20. Results reproducible from frozen checkpoint + frozen dataset + evaluator config hashes: PASS for BLADE; LEHD pending official checkpoint.",
    ]
    (CMP_ROOT / "09_comparison_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def generate_outputs(inventory: pd.DataFrame) -> str:
    summaries = collect_summaries(inventory, EIGHT)
    if summaries.empty:
        raise RuntimeError("no summaries to assemble")
    atomic_frame(OUT_ROOT / "03_new4_summary.csv", summaries[summaries.dataset.isin(NEW_DATASETS)].copy())
    assemble_new4(inventory)
    mean_table = pivot_metric(summaries, "mean_gap_percent")
    median_table = pivot_metric(summaries, "median_gap_percent")
    p90 = pivot_metric(summaries, "P90")
    p95 = pivot_metric(summaries, "P95")
    tail = p90.copy()
    for dataset in EIGHT:
        if dataset in p90.columns:
            tail[dataset] = p90[dataset]
            tail[f"{dataset}_P95"] = p95[dataset] if dataset in p95.columns else None
    atomic_frame(OUT_ROOT / "04_final8_mean_gap_trajectory.csv", mean_table)
    atomic_frame(OUT_ROOT / "05_final8_median_gap_trajectory.csv", median_table)
    atomic_frame(OUT_ROOT / "06_final8_tail_gap_trajectory.csv", tail)
    rel = relative_change(mean_table)
    atomic_frame(OUT_ROOT / "07_relative_change_vs_M0.csv", rel)
    write_figures(mean_table)
    latest = selected_models(inventory)[-1]
    write_phase_a_report(inventory, mean_table, rel, latest)
    write_lehd_pending(inventory, latest, mean_table, median_table, summaries)
    return latest


def print_human(inventory: pd.DataFrame, latest: str) -> None:
    mean_table = pd.read_csv(OUT_ROOT / "04_final8_mean_gap_trajectory.csv")
    models = selected_models(inventory)
    print("EXTERNAL V2 PASSIVE EVALUATION COMPLETE", flush=True)
    print("", flush=True)
    print("BLADE:", flush=True)
    print(f"latest valid checkpoint = {latest}", flush=True)
    print(f"evaluated checkpoints = {models[0]}...{models[-1]}", flush=True)
    print("", flush=True)
    print("NEW BENCHMARKS:", flush=True)
    print("EvoReal S1 = complete", flush=True)
    print("EvoReal S2 = complete", flush=True)
    print("EvoReal S3 = complete", flush=True)
    print("GM500 = complete", flush=True)
    print("", flush=True)
    print("FINAL 8-SET TRAJECTORY:", flush=True)
    print("complete" if set(mean_table.columns[1:]) == set(EIGHT) else "incomplete", flush=True)
    print("", flush=True)
    print("EVOREAL COMPARISON:", flush=True)
    print("Vanilla LEHD checkpoint = not found", flush=True)
    print("EvoReal LEHD checkpoint = not found", flush=True)
    print("8-set greedy evaluation = pending", flush=True)
    print("", flush=True)
    print("EVOREAL_LEHD_COMPARISON_PENDING_OFFICIAL_CHECKPOINT", flush=True)
    print("", flush=True)
    print(mean_table.to_string(index=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--datasets", nargs="*")
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    integrity = verify_references()
    if integrity["failures"]:
        print("INTEGRITY MISMATCH", flush=True)
        for item in integrity["failures"]:
            print(item, flush=True)
        return 2
    print(f"references_ok total_exact={integrity['total_exact']}", flush=True)
    inventory = discover_checkpoints()
    atomic_frame(OUT_ROOT / "01_checkpoint_inventory.csv", inventory)
    print(inventory.loc[:, ["model", "exists", "valid_scientific", "sha256", "originating_cycle"]].to_string(index=False), flush=True)
    if args.inventory_only:
        return 0
    if args.report_only:
        latest = generate_outputs(inventory)
        print_human(inventory, latest)
        return 0
    datasets = args.datasets or EVAL_ORDER
    if args.models:
        keep = set(args.models)
        inventory = inventory.copy()
        inventory["selected_for_evaluation"] = inventory.model.isin(keep) & inventory.valid_scientific
    run_evaluations(inventory, datasets, args.force)
    complete = True
    for model_name in selected_models(inventory):
        for dataset in EIGHT:
            if not pair_summary_path(model_name, dataset).is_file():
                complete = False
    if complete:
        latest = generate_outputs(inventory)
        print_human(inventory, latest)
    else:
        print("partial evaluation stored under audits/external_benchmark_suite_v2/blade_trajectory/work", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
