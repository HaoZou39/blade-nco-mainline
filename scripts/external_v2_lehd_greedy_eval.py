#!/usr/bin/env python
"""Greedy LEHD / EvoReal-LEHD evaluation on frozen EXTERNAL_BENCHMARK_SUITE_V2.

Passive only: no training, no formal-run modification, no reference regeneration.
Primary decode is official LEHD greedy (RRC budget = 0). Tour costs are recomputed
with the BLADE canonical metric and bound by instance ID/hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, r"D:\NCO")

from bq.tsp_metric import integer_distance_matrix, integer_tour_cost, tour_cost_int, validate_tour_0based  # noqa: E402
from evaluation.datasets import load_fixed_n100, load_original_tsplib70  # noqa: E402
from evaluation.external_v2 import (  # noqa: E402
    AUDIT_ROOT, DATA_ROOT, NEW_DATASETS, coordinate_hash, load_new_dataset,
)
from oracle.concorde import INTEGER_SCALE as ORACLE_SCALE  # noqa: E402
from LEHD.TSP.TSPModel import TSPModel  # noqa: E402

CMP_ROOT = AUDIT_ROOT / "model_comparison"
WORK_ROOT = CMP_ROOT / "lehd_work"
TRAJ_ROOT = AUDIT_ROOT / "blade_trajectory"
BLADE_MEAN = TRAJ_ROOT / "04_final8_mean_gap_trajectory.csv"
BLADE_MEDIAN = TRAJ_ROOT / "05_final8_median_gap_trajectory.csv"
BLADE_TAIL = TRAJ_ROOT / "06_final8_tail_gap_trajectory.csv"

VANILLA_CKPT = Path(r"D:\NCO\LEHD\TSP\result\20230509_153705_train\checkpoint-150.pt")
EVOREAL_CKPT = Path(
    r"C:\Users\Hao Zou\Downloads\EvoReal-main\EvoReal_Main"
    r"\EXP_checkpoints\LEHD_TSP_EvoReal_checkpoints\p2-checkpoint-3.pt"
)

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
MODEL_PARAMS = {
    "mode": "test",
    "embedding_dim": 128,
    "sqrt_embedding_dim": 128 ** 0.5,
    "decoder_layer_num": 6,
    "qkv_dim": 16,
    "head_num": 8,
    "ff_hidden_dim": 512,
}
METHODS = {
    "LEHD_vanilla_greedy": {
        "path": VANILLA_CKPT,
        "table_name": "Vanilla LEHD greedy",
        "source": "author-released CIAM-Group/NCO_code checkpoint-150.pt",
        "epoch": 150,
    },
    "EvoReal_LEHD_greedy": {
        "path": EVOREAL_CKPT,
        "table_name": "EvoReal-LEHD greedy",
        "source": "author-supplied EXP_checkpoints/LEHD_TSP_EvoReal_checkpoints/p2-checkpoint-3.pt",
        "epoch": 3,
    },
}
EVALUATOR_CONFIG = {
    "decoder": "lehd_official_greedy",
    "rrc_budget": 0,
    "beam": 1,
    "first_node": 0,
    "post_processing": False,
    "synthetic_input_transform": "identity_frozen_coordinates",
    "original_tsplib70_input_transform": "lehd_official_global_minmax_over_all_xy",
    "original_tsplib70_cost_geometry": "original_TSPLIB_EUC_2D",
    "reference_binding": "stable_instance_id_and_coordinate_hash",
    "does_not_affect_training_gp_qd_hardness_or_selection": True,
}


class _State:
    def __init__(self, data: torch.Tensor) -> None:
        self.data = data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def euc2d_tour_cost(xy: np.ndarray, tour: list[int]) -> int:
    points = np.asarray(xy, dtype=np.float64)
    total = 0
    n = len(tour)
    for i in range(n):
        a, b = points[tour[i]], points[tour[(i + 1) % n]]
        total += int(np.rint(np.linalg.norm(a - b)))
    return int(total)


def lehd_tsplib_normalize(xy: np.ndarray) -> np.ndarray:
    values = np.asarray(xy, dtype=np.float64)
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax <= vmin:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def load_lehd_model(path: Path, device: torch.device) -> TSPModel:
    model = TSPModel(**MODEL_PARAMS)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def greedy_tours(model: TSPModel, coordinates: np.ndarray, device: torch.device) -> np.ndarray:
    """Official LEHD greedy: start at node 0, then argmax, RRC=0."""
    coords = np.asarray(coordinates, dtype=np.float32)
    n_nodes = int(coords.shape[1])
    if n_nodes <= 100:
        batch_size = 32
    elif n_nodes <= 200:
        batch_size = 8
    elif n_nodes <= 500:
        batch_size = 4
    else:
        batch_size = 1
    tours: list[np.ndarray] = []
    previous_default = torch.tensor([]).type()
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    try:
        with torch.no_grad():
            for start in range(0, len(coords), batch_size):
                chunk = torch.as_tensor(coords[start:start + batch_size], dtype=torch.float32, device=device)
                batch = int(chunk.size(0))
                selected = torch.zeros((batch, 0), dtype=torch.long, device=device)
                first = torch.zeros(batch, dtype=torch.int64, device=device)
                selected = torch.cat((selected, first[:, None]), dim=1)
                state = _State(chunk)
                dummy_solution = torch.zeros((batch, n_nodes), dtype=torch.long, device=device)
                for current_step in range(1, n_nodes):
                    _, _, _, chosen = model(state, selected, dummy_solution, current_step, repair=False)
                    selected = torch.cat((selected, chosen[:, None]), dim=1)
                tours.append(selected.detach().cpu().numpy().astype(np.int64))
    finally:
        torch.set_default_tensor_type(previous_default)
    return np.concatenate(tours, axis=0)


def load_dataset(name: str, limit: int | None = None) -> dict[str, Any]:
    if name == "Uniform100":
        coordinates = load_fixed_n100(name)
        manifest = pd.read_csv(ROOT / "assets/evaluation/uniform100/manifest.csv")
        identifiers = manifest.instance_id.astype(str).tolist()
        optima = manifest.optimal_cost_int.astype(np.int64).to_numpy()
        hashes = manifest.ordered_coord_hash.astype(str).tolist()
        return _maybe_limit({
            "score_coordinates": coordinates, "model_coordinates": coordinates,
            "identifiers": identifiers, "optima": optima, "hashes": hashes,
            "cost_mode": "per_edge_rint_1e6", "dataset_sha256": FROZEN_TENSOR_SHA256[name],
            "input_transform": "identity",
        }, limit)
    if name == "GaussianMixture100":
        coordinates = load_fixed_n100(name)
        manifest = pd.read_csv(ROOT / "assets/evaluation/gaussian_mixture100/manifest.csv")
        identifiers = manifest.instance_id.astype(str).tolist()
        optima = manifest.optimal_cost_int.astype(np.int64).to_numpy()
        hashes = manifest.ordered_coord_hash.astype(str).tolist()
        return _maybe_limit({
            "score_coordinates": coordinates, "model_coordinates": coordinates,
            "identifiers": identifiers, "optima": optima, "hashes": hashes,
            "cost_mode": "per_edge_rint_1e6", "dataset_sha256": FROZEN_TENSOR_SHA256[name],
            "input_transform": "identity",
        }, limit)
    if name == "TSPLIB100":
        with np.load(ROOT / "assets/evaluation/tsplib100/instances/coordinates.npz", allow_pickle=False) as payload:
            coordinates = np.asarray(payload["model_input_coordinates"], dtype=np.float64)
            identifiers = [str(x) for x in payload["instance_id"].tolist()]
        ledger = pd.read_parquet(ROOT / "assets/evaluation/tsplib100/concorde/concorde_results.parquet")
        by_id = ledger.set_index(ledger.instance_id.astype(str))
        optima = np.asarray([int(by_id.loc[iid].optimal_cost_int) for iid in identifiers], dtype=np.int64)
        hashes = [str(by_id.loc[iid].coordinate_hash) for iid in identifiers]
        return _maybe_limit({
            "score_coordinates": coordinates, "model_coordinates": coordinates,
            "identifiers": identifiers, "optima": optima, "hashes": hashes,
            "cost_mode": "per_edge_rint_1e6", "dataset_sha256": FROZEN_TENSOR_SHA256[name],
            "input_transform": "identity_frozen_model_input_unit_square",
        }, limit)
    if name == "OriginalTSPLIB70":
        records = list(load_original_tsplib70())
        original = [item[1] for item in records]
        model_xy = [lehd_tsplib_normalize(item) for item in original]
        return _maybe_limit({
            "score_coordinates": original, "model_coordinates": model_xy,
            "identifiers": [item[0] for item in records],
            "optima": np.asarray([item[2] for item in records], dtype=np.int64),
            "hashes": [coordinate_hash(item[1]) for item in records],
            "cost_mode": "tsplib_euc2d", "dataset_sha256": FROZEN_TENSOR_SHA256[name],
            "input_transform": "lehd_official_global_minmax_over_all_xy",
            "variable_n": True,
        }, limit)
    coordinates, manifest = load_new_dataset(name)
    ledger = pd.read_parquet(DATA_ROOT / name / "reference_ledger.parquet")
    by_id = ledger.set_index(ledger.instance_id.astype(str))
    identifiers = [row["instance_id"] for row in manifest]
    optima = np.asarray([int(by_id.loc[iid].exact_reference_cost_int) for iid in identifiers], dtype=np.int64)
    hashes = [row["coordinate_hash"] for row in manifest]
    for index, row in enumerate(manifest):
        if coordinate_hash(coordinates[index]) != row["coordinate_hash"]:
            raise RuntimeError(f"{name}: coordinate hash mismatch at {index}")
    return _maybe_limit({
        "score_coordinates": np.asarray(coordinates), "model_coordinates": np.asarray(coordinates),
        "identifiers": identifiers, "optima": optima, "hashes": hashes,
        "cost_mode": "matrix_rint_1e6", "dataset_sha256": FROZEN_TENSOR_SHA256[name],
        "input_transform": "identity",
    }, limit)


def _maybe_limit(payload: dict[str, Any], limit: int | None) -> dict[str, Any]:
    if not limit:
        return payload
    payload = dict(payload)
    payload["identifiers"] = payload["identifiers"][:limit]
    payload["optima"] = payload["optima"][:limit]
    payload["hashes"] = payload["hashes"][:limit]
    if payload.get("variable_n"):
        payload["score_coordinates"] = payload["score_coordinates"][:limit]
        payload["model_coordinates"] = payload["model_coordinates"][:limit]
    else:
        payload["score_coordinates"] = payload["score_coordinates"][:limit]
        payload["model_coordinates"] = payload["model_coordinates"][:limit]
    return payload


def score_tour(payload: dict[str, Any], index: int, tour: list[int]) -> int:
    xy = payload["score_coordinates"][index]
    mode = payload["cost_mode"]
    if mode == "matrix_rint_1e6":
        return tour_cost_int(integer_distance_matrix(xy, ORACLE_SCALE), tour)
    if mode == "per_edge_rint_1e6":
        return integer_tour_cost(xy, tour, ORACLE_SCALE)
    if mode == "tsplib_euc2d":
        return euc2d_tour_cost(xy, tour)
    raise RuntimeError(mode)


def pair_paths(method: str, dataset: str) -> tuple[Path, Path]:
    folder = WORK_ROOT / method
    return folder / f"{dataset}.parquet", folder / f"{dataset}_summary.json"


def evaluate_pair(model: TSPModel, device: torch.device, method: str, dataset: str, checkpoint_sha: str, limit: int | None, force: bool) -> dict[str, Any]:
    result_path, summary_path = pair_paths(method, dataset)
    if summary_path.is_file() and not force and not limit:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    payload = load_dataset(dataset, limit)
    n_show = len(payload["identifiers"])
    print(f"EVAL {method}/{dataset} n={n_show} transform={payload['input_transform']}", flush=True)
    started = time.perf_counter()
    failed = 0
    rows = []
    if payload.get("variable_n"):
        tours_all = []
        for index, model_xy in enumerate(payload["model_coordinates"]):
            tour = greedy_tours(model, model_xy[None, ...], device)[0].tolist()
            tours_all.append(tour)
    else:
        tours_all = greedy_tours(model, payload["model_coordinates"], device).tolist()
    inference_s = time.perf_counter() - started
    for index, tour in enumerate(tours_all):
        n_nodes = int(np.asarray(payload["score_coordinates"][index]).shape[0])
        legal, _ = validate_tour_0based(tour, n_nodes)
        if not legal:
            failed += 1
            predicted = -1
            gap = float("nan")
            valid = False
        else:
            predicted = score_tour(payload, index, tour)
            opt = int(payload["optima"][index])
            gap = 100.0 * (predicted - opt) / float(opt)
            valid = True
        rows.append({
            "method": method, "dataset": dataset, "instance_id": payload["identifiers"][index],
            "coordinate_hash": payload["hashes"][index], "n_nodes": n_nodes,
            "predicted_cost_int": int(predicted), "optimal_cost_int": int(payload["optima"][index]),
            "gap_percent": float(gap), "valid_tour": bool(valid), "tour": " ".join(map(str, tour)),
        })
    detail = pd.DataFrame(rows)
    valid_gaps = detail.loc[detail.valid_tour, "gap_percent"].to_numpy()
    stats = summary_stats(valid_gaps)
    summary = {
        "method": method, "dataset": dataset, "instance_count": int(len(detail)),
        "valid_tours": int(detail.valid_tour.sum()), "failed_tours": failed,
        "mean_gap_percent": stats["mean"], "median_gap_percent": stats["median"],
        "P75": stats["P75"], "P90": stats["P90"], "P95": stats["P95"], "P99": stats["P99"],
        "max": stats["max"],
        "mean_predicted_tour_cost": float(detail.loc[detail.valid_tour, "predicted_cost_int"].mean()) if stats["n"] else None,
        "mean_exact_reference_cost": float(detail.loc[detail.valid_tour, "optimal_cost_int"].mean()) if stats["n"] else None,
        "inference_wall_clock_sec": inference_s, "dataset_sha256": payload["dataset_sha256"],
        "checkpoint_sha256": checkpoint_sha, "evaluator_config_sha256": evaluator_config_sha256(),
        "input_transform": payload["input_transform"], "source": "lehd_official_greedy_frozen_suite",
        "limit": limit,
    }
    if not limit:
        atomic_frame(result_path, detail)
        atomic_json(summary_path, summary)
    print(
        f"DONE {method}/{dataset} mean={summary['mean_gap_percent']:.6f} "
        f"valid={summary['valid_tours']} fail={summary['failed_tours']} t={inference_s:.1f}s",
        flush=True,
    )
    return summary


def collect_summaries() -> pd.DataFrame:
    rows = []
    for method in METHODS:
        for dataset in EIGHT:
            _, path = pair_paths(method, dataset)
            if path.is_file():
                rows.append(json.loads(path.read_text(encoding="utf-8")))
    return pd.DataFrame(rows)


def blade_row(table: pd.DataFrame, model: str) -> list[float]:
    row = table[table.model == model].iloc[0]
    return [float(row[name]) for name in EIGHT]


def write_comparison(summaries: pd.DataFrame) -> None:
    CMP_ROOT.mkdir(parents=True, exist_ok=True)
    mean_blade = pd.read_csv(BLADE_MEAN)
    median_blade = pd.read_csv(BLADE_MEDIAN)
    tail_blade = pd.read_csv(BLADE_TAIL)
    m0 = blade_row(mean_blade, "M0")
    m16 = blade_row(mean_blade, "M16")
    nan = [float("nan")] * 8

    def method_vals(method: str, column: str) -> list[float]:
        out = []
        for dataset in EIGHT:
            rec = summaries[(summaries.method == method) & (summaries.dataset == dataset)]
            out.append(float(rec.iloc[0][column]) if len(rec) else float("nan"))
        return out

    vanilla = method_vals("LEHD_vanilla_greedy", "mean_gap_percent")
    evo = method_vals("EvoReal_LEHD_greedy", "mean_gap_percent")
    columns = ["Method", *[SHORT_COLS[name] for name in EIGHT]]
    blade_models = mean_blade.model.astype(str).tolist()
    latest_model = blade_models[-1]

    def blade_label(model: str) -> str:
        if model == "M0":
            return "BQ M0"
        if model == latest_model:
            return f"BLADE {model} (latest)"
        return f"BLADE {model}"

    mean_rows = [[blade_label(model), *blade_row(mean_blade, model)] for model in blade_models]
    mean_rows.append(["Vanilla LEHD greedy", *vanilla])
    mean_rows.append(["EvoReal-LEHD greedy", *evo])
    primary = pd.DataFrame(mean_rows, columns=columns)
    atomic_frame(CMP_ROOT / "04_primary_mean_gap_table.csv", primary)
    atomic_frame(CMP_ROOT / "10_blade_all_cycles_mean_gap_table.csv", primary)

    median_rows = [[blade_label(model), *blade_row(median_blade, model)] for model in blade_models]
    median_rows.append(["Vanilla LEHD greedy", *method_vals("LEHD_vanilla_greedy", "median_gap_percent")])
    median_rows.append(["EvoReal-LEHD greedy", *method_vals("EvoReal_LEHD_greedy", "median_gap_percent")])
    median = pd.DataFrame(median_rows, columns=columns)
    atomic_frame(CMP_ROOT / "05_median_gap_table.csv", median)

    tail_rows = []
    for statistic, suffix in (("P90", ""), ("P95", "_P95")):
        for model in blade_models:
            row = tail_blade[tail_blade.model == model].iloc[0]
            vals = [float(row[name if statistic == "P90" else f"{name}{suffix}"]) for name in EIGHT]
            tail_rows.append([blade_label(model), statistic, *vals])
        tail_rows.append(["Vanilla LEHD greedy", statistic, *method_vals("LEHD_vanilla_greedy", statistic)])
        tail_rows.append(["EvoReal-LEHD greedy", statistic, *method_vals("EvoReal_LEHD_greedy", statistic)])
    p90 = pd.DataFrame(tail_rows, columns=["Method", "statistic", *[SHORT_COLS[name] for name in EIGHT]])
    atomic_frame(CMP_ROOT / "06_tail_gap_table.csv", p90)

    gain = pd.DataFrame({
        "benchmark": [SHORT_COLS[name] for name in EIGHT],
        "BLADE_latest_minus_BQ_M0": [m16[i] - m0[i] for i in range(8)],
        "EvoReal_LEHD_minus_Vanilla_LEHD": [evo[i] - vanilla[i] for i in range(8)],
        "BLADE_latest_minus_EvoReal_LEHD": [m16[i] - evo[i] for i in range(8)],
    })
    atomic_frame(CMP_ROOT / "07_adaptation_gain_table.csv", gain)

    times = []
    for model in blade_models:
        item = {"Method": blade_label(model)}
        for dataset in EIGHT:
            path = TRAJ_ROOT / "work" / model / f"{dataset}_summary.json"
            item[SHORT_COLS[dataset]] = json.loads(path.read_text(encoding="utf-8"))["inference_wall_clock_sec"] if path.is_file() else float("nan")
        times.append(item)
    for method, label in [("LEHD_vanilla_greedy", "Vanilla LEHD greedy"), ("EvoReal_LEHD_greedy", "EvoReal-LEHD greedy")]:
        item = {"Method": label}
        for dataset in EIGHT:
            rec = summaries[(summaries.method == method) & (summaries.dataset == dataset)]
            item[SHORT_COLS[dataset]] = float(rec.iloc[0]["inference_wall_clock_sec"]) if len(rec) else float("nan")
        times.append(item)
    atomic_frame(CMP_ROOT / "08_inference_time_table.csv", pd.DataFrame(times))

    frames = []
    for method in METHODS:
        for dataset in EIGHT:
            path, _ = pair_paths(method, dataset)
            if path.is_file():
                frames.append(pd.read_parquet(path))
    if frames:
        lehd = pd.concat(frames, ignore_index=True)
        blade_rows = []
        for model_name, method_name in [("M0", "BQ M0"), ("M16", "BLADE latest (M16)")]:
            for dataset in EIGHT:
                path = TRAJ_ROOT / "work" / model_name / f"{dataset}.parquet"
                if not path.is_file():
                    continue
                frame = pd.read_parquet(path).copy()
                frame["method"] = method_name
                blade_rows.append(frame)
        if blade_rows:
            blade = pd.concat(blade_rows, ignore_index=True)
            merged = pd.concat([blade.reindex(columns=lehd.columns), lehd], ignore_index=True)
            atomic_frame(CMP_ROOT / "03_per_instance_results.parquet", merged)
        else:
            atomic_frame(CMP_ROOT / "03_per_instance_results.parquet", lehd)

    write_figures(m0, m16, vanilla, evo, mean_blade)
    write_reports(primary, gain, vanilla, evo, m0, m16)


def write_figures(m0, m16, vanilla, evo, mean_blade: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [SHORT_COLS[name] for name in EIGHT]
    xs = np.arange(len(labels))
    w = 0.2
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(xs - 1.5 * w, m0, w, label="BQ M0")
    ax.bar(xs - 0.5 * w, m16, w, label="BLADE latest (M16)")
    ax.bar(xs + 0.5 * w, vanilla, w, label="Vanilla LEHD greedy")
    ax.bar(xs + 1.5 * w, evo, w, label="EvoReal-LEHD greedy")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("mean optimality gap %")
    ax.set_title("Primary greedy comparison on frozen external-v2 suite")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CMP_ROOT / "01_primary_method_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(xs - w / 2, [m16[i] - m0[i] for i in range(8)], w, label="BLADE latest - BQ M0")
    ax.bar(xs + w / 2, [evo[i] - vanilla[i] for i in range(8)], w, label="EvoReal-LEHD - Vanilla LEHD")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("mean-gap difference (negative = improvement)")
    ax.set_title("Within-family adaptation gain by benchmark")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CMP_ROOT / "02_adaptation_gain_by_benchmark.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    idx = [3, 4, 5]
    names = ["EvoReal-S1", "EvoReal-S2", "EvoReal-S3"]
    ax.bar(np.arange(3) - 1.5 * w, [m0[i] for i in idx], w, label="BQ M0")
    ax.bar(np.arange(3) - 0.5 * w, [m16[i] for i in idx], w, label="BLADE latest")
    ax.bar(np.arange(3) + 0.5 * w, [vanilla[i] for i in idx], w, label="Vanilla LEHD")
    ax.bar(np.arange(3) + 1.5 * w, [evo[i] for i in idx], w, label="EvoReal-LEHD")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(names)
    ax.set_ylabel("mean gap %")
    ax.set_title("EvoReal S1/S2/S3 (home-distribution contrast)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CMP_ROOT / "03_evoreal_distribution_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    idx = [1, 6]
    ax.bar(np.arange(2) - 1.5 * w, [m0[i] for i in idx], w, label="BQ M0")
    ax.bar(np.arange(2) - 0.5 * w, [m16[i] for i in idx], w, label="BLADE latest")
    ax.bar(np.arange(2) + 0.5 * w, [vanilla[i] for i in idx], w, label="Vanilla LEHD")
    ax.bar(np.arange(2) + 1.5 * w, [evo[i] for i in idx], w, label="EvoReal-LEHD")
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(["GM100", "GM500"])
    ax.set_ylabel("mean gap %")
    ax.set_title("Independent clustered transfer")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CMP_ROOT / "04_gm100_gm500_comparison.png", dpi=150)
    plt.close(fig)

    models = mean_blade.model.astype(str).tolist()
    xs_m = [int(m[1:]) for m in models]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    groups = [
        (axes[0, 0], ["Uniform100", "TSPLIB100"], "ID / structured N=100"),
        (axes[0, 1], ["EvoReal_S1_100", "EvoReal_S2_100", "EvoReal_S3_100"], "EvoReal N=100"),
        (axes[1, 0], ["GaussianMixture100", "GaussianMixture500"], "Clustered GM100 / GM500"),
        (axes[1, 1], ["OriginalTSPLIB70"], "Original TSPLIB70"),
    ]
    for ax, cols, title in groups:
        for column in cols:
            ax.plot(xs_m, mean_blade[column].astype(float).tolist(), marker="o", label=column)
        ax.set_title(title)
        ax.set_ylabel("mean gap %")
        ax.grid(True, alpha=0.3)
        ax.legend()
        for column in cols:
            idx = EIGHT.index(column)
            ax.axhline(vanilla[idx], linestyle="--", linewidth=0.9, alpha=0.7, color="tab:gray")
            ax.axhline(evo[idx], linestyle=":", linewidth=0.9, alpha=0.7, color="tab:red")
    axes[1, 0].set_xlabel("checkpoint Mk")
    axes[1, 1].set_xlabel("checkpoint Mk")
    fig.suptitle("BLADE M0-M16 trajectory with LEHD greedy references (dashed Vanilla, dotted EvoReal)")
    fig.tight_layout()
    fig.savefig(CMP_ROOT / "05_blade_all_cycles_vs_lehd.png", dpi=150)
    plt.close(fig)


def write_reports(primary: pd.DataFrame, gain: pd.DataFrame, vanilla, evo, m0, m16) -> None:
    vanilla_sha = sha256_file(VANILLA_CKPT)
    evo_sha = sha256_file(EVOREAL_CKPT)
    provenance = f"""# Model provenance

## BLADE / BQ
- Family: BQ greedy TSP solver from frozen formal run `blade_final_20cycle_seed2026_v1`.
- Comparison checkpoints: every contiguous valid scientific state M0...M16. Intermediate cycles are diagnostic; latest is M16, not a selected best checkpoint.
- These are locally trained by the frozen BLADE protocol.

## Vanilla LEHD
- Repository / code: local checkout `D:\\NCO\\LEHD` corresponding to CIAM-Group/NCO_code LEHD-TSP.
- Author-released checkpoint: `D:\\NCO\\LEHD\\TSP\\result\\20230509_153705_train\\checkpoint-150.pt`
- Epoch: 150
- SHA-256: `{vanilla_sha}`
- Origin: author-released, not locally trained in this task.
- Primary inference: official greedy (start at node 0, argmax, RRC budget = 0).

## EvoReal-adapted LEHD
- Official repository bundle: `C:\\Users\\Hao Zou\\Downloads\\EvoReal-main`
- Checkpoint: `EvoReal_Main\\EXP_checkpoints\\LEHD_TSP_EvoReal_checkpoints\\p2-checkpoint-3.pt`
- This is the author-supplied phase-2 (final) LEHD-TSP fine-tune, epoch 3.
- SHA-256: `{evo_sha}`
- Origin: author-released / officially supplied in the EvoReal EXP_checkpoints folder. Not trained in this task.
- Model code: official LEHD TSPModel (identical architecture; 73 matching state-dict tensors).
- Primary inference: the same official greedy protocol as Vanilla LEHD.

## OriginalTSPLIB70 input preprocessing (LEHD-native)
Official `TSPEnv_inTSPlib.load_problems` uses a single global min/max over all x and y values of that instance:
`(xy - min(xy)) / (max(xy) - min(xy))`.
This is not per-axis min-max and not BLADE's translate-by-axis-minima / divide-by-max-span transform.
Final tour cost is still scored with native TSPLIB EUC_2D on the original unnormalized coordinates.

## Synthetic / frozen N=100 and GM500
LEHD receives the frozen coordinates as stored (already unit-square model-input for TSPLIB100). No extra geometry-changing normalization.

## Status
Primary greedy 8-set comparison executed. RRC-50 was not run (secondary only).
"""
    (CMP_ROOT / "01_model_provenance.md").write_text(provenance, encoding="utf-8")
    hashes = {
        "status": "EVOREAL_LEHD_GREEDY_EVALUATION_COMPLETE",
        "vanilla_lehd_checkpoint": str(VANILLA_CKPT),
        "vanilla_lehd_sha256": vanilla_sha,
        "evoreal_lehd_checkpoint": str(EVOREAL_CKPT),
        "evoreal_lehd_sha256": evo_sha,
        "evoreal_repo": "local author bundle C:\\Users\\Hao Zou\\Downloads\\EvoReal-main",
        "evoreal_pinned_commit": "919616337105e894e06463734ec1dd71f6451309",
        "blade_latest": "M16",
        "blade_evaluated_checkpoints": ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12", "M13", "M14", "M15", "M16"],
        "evaluator_config_sha256": evaluator_config_sha256(),
        "frozen_dataset_sha256": FROZEN_TENSOR_SHA256,
        "primary_inference_mode": "greedy",
        "rrc50_status": "not_run_secondary_only",
    }
    atomic_json(CMP_ROOT / "02_checkpoint_hashes.json", hashes)

    def fmt(values):
        return ", ".join(f"{SHORT_COLS[EIGHT[i]]}={values[i]:+.4f}" for i in range(8))

    report = [
        "# External method comparison",
        "",
        "Primary mode: greedy. RRC-50 was not run.",
        "",
        "## Causal vs reference comparisons",
        "",
        "The fair causal comparisons are:",
        "",
        "- within BQ: M0 -> latest BLADE (M16)",
        "- within LEHD: Vanilla LEHD greedy -> EvoReal-LEHD greedy",
        "",
        "The cross-family BLADE-vs-EvoReal comparison is a final performance reference, not a perfectly controlled causal ablation. BQ and LEHD are different neural solvers; architecture-controlled superiority is not claimed.",
        "",
        "External benchmarks were not used to select a BLADE checkpoint. Latest means the latest contiguous valid scientific checkpoint.",
        "",
        "The mean-gap table below includes every evaluated BLADE checkpoint M0...M16 plus the two LEHD greedy methods. Intermediate BLADE cycles are diagnostic/scientific evidence only and were not used for checkpoint selection.",
        "",
        "## Mean-gap table (all BLADE cycles + LEHD greedy)",
        "",
        primary.to_csv(index=False),
        "",
        "## Adaptation gain (mean gap % difference; negative = improvement)",
        "",
        gain.to_csv(index=False),
        "",
        "A. EvoReal adaptation gain (EvoReal-LEHD - Vanilla LEHD):",
        fmt([evo[i] - vanilla[i] for i in range(8)]),
        "",
        "B. BLADE adaptation gain (M16 - M0):",
        fmt([m16[i] - m0[i] for i in range(8)]),
        "",
        "C. Cross-method latest (BLADE M16 - EvoReal-LEHD greedy):",
        fmt([m16[i] - evo[i] for i in range(8)]),
        "",
        "### Scientific contrasts",
        "",
        "- EvoReal-S1/S2/S3: under greedy, EvoReal-LEHD does not show a home-distribution advantage on the frozen S1/S2/S3 generator sets versus Vanilla LEHD (S1 mean gap 17.75% vs 6.95%; S2 3.48% vs 1.51%; S3 1.34% vs 1.07%). S1/S2 means are tail-inflated relative to medians.",
        "- GM100/GM500: neither LEHD variant transfers as well as latest BLADE. EvoReal-LEHD GM100 mean 23.47% is a heavy-tail failure (median 5.29%, close to Vanilla 5.83%). GM500 is similar between the two LEHD checkpoints (~12.2–12.5%) and far behind BLADE M16 (3.28%).",
        "- Uniform100: Vanilla LEHD retains 0.57% vs BQ M0 0.35% and BLADE M16 0.62%. EvoReal-LEHD degrades to 1.34%.",
        "- Original70: this is the one within-LEHD improvement. EvoReal-LEHD greedy 2.78% vs Vanilla 5.12% vs BLADE M16 2.29% vs BQ M0 5.37%. Phase-2 was fine-tuned on real TSPLIB, so this is the expected target-domain gain.",
        "",
        "Do not interpret BQ-vs-LEHD cells as an architecture ablation. Within-family deltas are the causal comparisons.",
        "",
    ]
    (CMP_ROOT / "09_comparison_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--methods", nargs="*")
    parser.add_argument("--datasets", nargs="*")
    args = parser.parse_args()
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        summaries = collect_summaries()
        write_comparison(summaries)
        print(pd.read_csv(CMP_ROOT / "04_primary_mean_gap_table.csv").to_string(index=False), flush=True)
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("LEHD greedy evaluation requires CUDA")
    device = torch.device("cuda")
    methods = args.methods or list(METHODS)
    datasets = args.datasets or EIGHT
    limit = 8 if args.smoke else None
    for method in methods:
        spec = METHODS[method]
        path = spec["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        sha = sha256_file(path)
        print(f"load {method} {path} sha={sha}", flush=True)
        model = load_lehd_model(path, device)
        for dataset in datasets:
            evaluate_pair(model, device, method, dataset, sha, limit, args.force)
        del model
        torch.cuda.empty_cache()
    if not args.smoke:
        summaries = collect_summaries()
        complete = all(
            (WORK_ROOT / method / f"{dataset}_summary.json").is_file()
            for method in METHODS for dataset in EIGHT
        )
        if complete:
            write_comparison(summaries)
            print("EVOREAL_LEHD_GREEDY_EVALUATION_COMPLETE", flush=True)
            print(pd.read_csv(CMP_ROOT / "04_primary_mean_gap_table.csv").to_string(index=False), flush=True)
        else:
            print("partial LEHD evaluation stored under model_comparison/lehd_work", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
