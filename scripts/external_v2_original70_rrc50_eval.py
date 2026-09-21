#!/usr/bin/env python
"""Passive OriginalTSPLIB70 RRC50 evaluation.

Starts from already-computed canonical greedy tours. Does not train, does not
modify the formal run, and does not regenerate references.

RRC50 is LEHD-only (official destroy-subpath / repair, budget = 50). BQ/BLADE
have no RRC decoder and remain canonical greedy in the comparison table.
Accept/reject uses native TSPLIB EUC_2D integer cost.
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
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, r"D:\NCO")

from bq.checkpoint import load_canonical_checkpoint  # noqa: E402
from bq.decoding import greedy_decode  # noqa: E402
from bq.protocol import CANONICAL_M0_SHA256, KNNS  # noqa: E402
from bq.tsp_metric import decode_tour_to_permutation, validate_tour_0based  # noqa: E402
from evaluation.datasets import load_original_tsplib70  # noqa: E402
from evaluation.external_v2 import AUDIT_ROOT, coordinate_hash  # noqa: E402
from LEHD.TSP.TSPModel import TSPModel  # noqa: E402

FORMAL_RUN = ROOT / "runs" / "blade_final_20cycle_seed2026_v1"
CMP_ROOT = AUDIT_ROOT / "model_comparison"
WORK_ROOT = CMP_ROOT / "rrc50_work"
TRAJ_ROOT = AUDIT_ROOT / "blade_trajectory"

PRIORITY_IDS = [
    "berlin52", "st70", "pr76", "eil76", "pr124",
    "rl1304", "rl1323", "nrw1379", "fl1400", "u1432",
    "fl1577", "d1655", "vm1748", "u1817", "rl1889",
    "d2103", "u2152", "u2319", "pr2392", "pcb3038",
    "fl3795", "fnl4461",
]
RRC_BUDGET = 50
RRC_SEED = 2026
VANILLA_CKPT = Path(r"D:\NCO\LEHD\TSP\result\20230509_153705_train\checkpoint-150.pt")
EVOREAL_CKPT = Path(
    r"C:\Users\Hao Zou\Downloads\EvoReal-main\EvoReal_Main"
    r"\EXP_checkpoints\LEHD_TSP_EvoReal_checkpoints\p2-checkpoint-3.pt"
)
LEHD_MODEL_PARAMS = {
    "mode": "test",
    "embedding_dim": 128,
    "sqrt_embedding_dim": 128 ** 0.5,
    "decoder_layer_num": 6,
    "qkv_dim": 16,
    "head_num": 8,
    "ff_hidden_dim": 512,
}
METHODS = {
    "LEHD_vanilla_rrc50": {
        "family": "lehd",
        "table_name": "Vanilla LEHD RRC50",
        "greedy_parquet": CMP_ROOT / "lehd_work" / "LEHD_vanilla_greedy" / "OriginalTSPLIB70.parquet",
        "checkpoint": VANILLA_CKPT,
        "expected_sha": None,
        "input_transform": "lehd_official_global_minmax_over_all_xy",
    },
    "EvoReal_LEHD_rrc50": {
        "family": "lehd",
        "table_name": "EvoReal-LEHD RRC50",
        "greedy_parquet": CMP_ROOT / "lehd_work" / "EvoReal_LEHD_greedy" / "OriginalTSPLIB70.parquet",
        "checkpoint": EVOREAL_CKPT,
        "expected_sha": None,
        "input_transform": "lehd_official_global_minmax_over_all_xy",
    },
}
EVALUATOR_CONFIG = {
    "decoder": "rrc50_from_canonical_greedy_init",
    "rrc_budget": RRC_BUDGET,
    "rrc_seed": RRC_SEED,
    "destroy": "official_lehd_uniform_subpath_length_4_to_N_optional_reverse",
    "accept_metric": "native_TSPLIB_EUC_2D_integer",
    "lehd_repair": "official_lehd_repair_True_fixed_subpath_endpoints",
    "bq_not_evaluated_with_rrc": True,
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


def euc2d_tour_cost(xy: np.ndarray, tour: Sequence[int]) -> int:
    points = np.asarray(xy, dtype=np.float64)
    total = 0
    n = len(tour)
    for i in range(n):
        a, b = points[tour[i]], points[tour[(i + 1) % n]]
        total += int(np.rint(np.linalg.norm(a - b)))
    return int(total)


def aspect_preserving_normalize(xy: np.ndarray) -> np.ndarray:
    values = np.asarray(xy, dtype=np.float64)
    shifted = values - values.min(axis=0)
    span = float(shifted.max())
    if span <= 0:
        return shifted
    return shifted / span


def lehd_tsplib_normalize(xy: np.ndarray) -> np.ndarray:
    values = np.asarray(xy, dtype=np.float64)
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax <= vmin:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def instance_rng(method: str, instance_id: str) -> np.random.Generator:
    payload = f"original70_rrc50|{RRC_SEED}|{method}|{instance_id}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31 - 1)
    return np.random.default_rng(seed)


def parse_tour(text: str) -> list[int]:
    return [int(token) for token in str(text).split()]


def load_records() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for instance_id, original, optimum in load_original_tsplib70():
        out[str(instance_id)] = {
            "instance_id": str(instance_id),
            "original": np.asarray(original, dtype=np.float64),
            "optimum": int(optimum),
            "hash": coordinate_hash(original),
            "bq_input": aspect_preserving_normalize(original),
            "lehd_input": lehd_tsplib_normalize(original),
        }
    return out


def load_greedy_tours(path: Path) -> dict[str, list[int]]:
    frame = pd.read_parquet(path, columns=["instance_id", "tour", "valid_tour"])
    out: dict[str, list[int]] = {}
    for row in frame.itertuples(index=False):
        if not bool(row.valid_tour):
            continue
        out[str(row.instance_id)] = parse_tour(row.tour)
    return out


def splice_subpath(tour: np.ndarray, first: int, length: int, new_sub: np.ndarray) -> np.ndarray:
    double = np.concatenate([tour, tour])
    new_double = np.concatenate([double[:first], np.asarray(new_sub, dtype=np.int64), double[first + length :]])
    return new_double[first:first + len(tour)].copy()


def lehd_repair_subpath(
    model: TSPModel,
    sub_coords: np.ndarray,
    first_rank: int,
    last_rank: int,
    device: torch.device,
) -> np.ndarray:
    coords = torch.as_tensor(sub_coords, dtype=torch.float32, device=device)[None]
    batch, length, _ = coords.shape
    selected = torch.zeros((batch, 0), dtype=torch.long, device=device)
    dummy = torch.zeros((batch, length), dtype=torch.long, device=device)
    state = _State(coords)
    last_t = torch.tensor([last_rank], dtype=torch.int64, device=device)
    first_t = torch.tensor([first_rank], dtype=torch.int64, device=device)
    for step in range(length):
        if step == 0:
            chosen = last_t
        elif step == 1:
            chosen = first_t
        else:
            _, _, _, chosen = model(state, selected, dummy, step, repair=True)
        selected = torch.cat((selected, chosen[:, None]), dim=1)
    rolled = torch.roll(selected, shifts=-1, dims=1)
    return rolled[0].detach().cpu().numpy().astype(np.int64)


def bq_repair_subpath(model, model_xy: np.ndarray, sub_ids: np.ndarray, device: torch.device) -> np.ndarray:
    order = np.asarray(sub_ids, dtype=np.int64)
    length = int(len(order))
    xy = np.ascontiguousarray(model_xy[order], dtype=np.float32)
    node_coords = torch.from_numpy(xy)[None].to(device)
    dist = torch.cdist(node_coords, node_coords)
    tours = greedy_decode(node_coords, dist, model, knns=KNNS)
    local = decode_tour_to_permutation(tours[0].detach().cpu().numpy(), length - 1)
    local_full = np.asarray(local + [length - 1], dtype=np.int64)
    return order[local_full]


def rrc_step_lehd(model, model_xy, score_xy, tour, rng, device) -> tuple[list[int], int, bool]:
    n = len(tour)
    current = np.asarray(tour, dtype=np.int64)
    if int(rng.integers(0, 100)) >= 50:
        current = current[::-1].copy()
    first = int(rng.integers(0, n))
    length = int(rng.integers(4, n + 1))
    double = np.concatenate([current, current])
    sub_ids = double[first:first + length]
    rank = np.argsort(sub_ids, kind="stable")
    solution_rank = np.argsort(rank, kind="stable")
    ascending = np.sort(sub_ids)
    repaired = lehd_repair_subpath(
        model, model_xy[ascending], int(solution_rank[0]), int(solution_rank[-1]), device
    )
    new_sub = ascending[repaired]
    candidate = splice_subpath(current, first, length, new_sub)
    before = euc2d_tour_cost(score_xy, current.tolist())
    after = euc2d_tour_cost(score_xy, candidate.tolist())
    if after < before:
        return candidate.tolist(), after, True
    return current.tolist(), before, False


def rrc_step_bq(model, model_xy, score_xy, tour, rng, device) -> tuple[list[int], int, bool]:
    n = len(tour)
    current = np.asarray(tour, dtype=np.int64)
    if int(rng.integers(0, 100)) >= 50:
        current = current[::-1].copy()
    first = int(rng.integers(0, n))
    length = int(rng.integers(4, n + 1))
    double = np.concatenate([current, current])
    sub_ids = double[first:first + length]
    new_sub = bq_repair_subpath(model, model_xy, sub_ids, device)
    candidate = splice_subpath(current, first, length, new_sub)
    before = euc2d_tour_cost(score_xy, current.tolist())
    after = euc2d_tour_cost(score_xy, candidate.tolist())
    if after < before:
        return candidate.tolist(), after, True
    return current.tolist(), before, False


def load_lehd_model(path: Path, device: torch.device) -> TSPModel:
    model = TSPModel(**LEHD_MODEL_PARAMS)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def pair_paths(method: str) -> tuple[Path, Path]:
    folder = WORK_ROOT / method
    return folder / "OriginalTSPLIB70.parquet", folder / "OriginalTSPLIB70_summary.json"


def evaluate_method(
    method: str,
    spec: dict[str, Any],
    records: dict[str, dict[str, Any]],
    instance_ids: list[str],
    device: torch.device,
    force: bool,
) -> pd.DataFrame:
    result_path, summary_path = pair_paths(method)
    existing = pd.DataFrame()
    if result_path.is_file() and not force:
        existing = pd.read_parquet(result_path)
    done = set(existing.instance_id.astype(str)) if len(existing) else set()
    greedy_tours = load_greedy_tours(spec["greedy_parquet"])
    family = spec["family"]
    model = None
    previous_default = None
    if any(iid not in done for iid in instance_ids):
        if family == "bq":
            model, _ = load_canonical_checkpoint(
                spec["checkpoint"], device, expected_sha256=spec["expected_sha"]
            )
            model.eval()
        else:
            previous_default = torch.tensor([]).type()
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
            model = load_lehd_model(spec["checkpoint"], device)
    rows = existing.to_dict("records") if len(existing) else []
    checkpoint_sha = sha256_file(spec["checkpoint"])
    try:
        for index, instance_id in enumerate(instance_ids, 1):
            if instance_id in done:
                rec = existing[existing.instance_id.astype(str) == instance_id].iloc[0]
                print(
                    f"[{method}] {index}/{len(instance_ids)} {instance_id} "
                    f"N={int(rec.n_nodes)} skip existing rrc_gap={float(rec.gap_percent):.6f}%",
                    flush=True,
                )
                continue
            item = records[instance_id]
            original = item["original"]
            model_xy = item["bq_input"] if family == "bq" else item["lehd_input"]
            tour0 = greedy_tours[instance_id]
            legal, message = validate_tour_0based(tour0, len(original))
            if not legal:
                raise RuntimeError(f"{method}/{instance_id} greedy tour invalid: {message}")
            greedy_cost = euc2d_tour_cost(original, tour0)
            rng = instance_rng(method, instance_id)
            tour = list(tour0)
            cost = greedy_cost
            accepted = 0
            started = time.perf_counter()
            with torch.no_grad():
                for _ in range(RRC_BUDGET):
                    if family == "bq":
                        tour, cost, improved = rrc_step_bq(model, model_xy, original, tour, rng, device)
                    else:
                        tour, cost, improved = rrc_step_lehd(model, model_xy, original, tour, rng, device)
                    accepted += int(improved)
            elapsed = time.perf_counter() - started
            legal, message = validate_tour_0based(tour, len(original))
            if not legal:
                raise RuntimeError(f"{method}/{instance_id} RRC tour invalid: {message}")
            opt = item["optimum"]
            gap = 100.0 * (cost - opt) / float(opt)
            greedy_gap = 100.0 * (greedy_cost - opt) / float(opt)
            row = {
                "method": method,
                "dataset": "OriginalTSPLIB70",
                "instance_id": instance_id,
                "coordinate_hash": item["hash"],
                "n_nodes": int(len(original)),
                "predicted_cost_int": int(cost),
                "optimal_cost_int": int(opt),
                "gap_percent": float(gap),
                "greedy_cost_int": int(greedy_cost),
                "greedy_gap_percent": float(greedy_gap),
                "accepted_repairs": int(accepted),
                "rrc_budget": RRC_BUDGET,
                "valid_tour": True,
                "tour": " ".join(map(str, tour)),
                "inference_wall_clock_sec": float(elapsed),
                "checkpoint_sha256": checkpoint_sha,
                "evaluator_config_sha256": evaluator_config_sha256(),
            }
            rows.append(row)
            frame = pd.DataFrame(rows)
            atomic_frame(result_path, frame)
            print(
                f"[{method}] {index}/{len(instance_ids)} {instance_id} N={len(original)} "
                f"greedy_gap={greedy_gap:.6f}% rrc_gap={gap:.6f}% "
                f"accepted={accepted}/{RRC_BUDGET} t={elapsed:.1f}s",
                flush=True,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        if previous_default is not None:
            torch.set_default_tensor_type(previous_default)
    frame = pd.DataFrame(rows)
    if len(frame):
        gaps = frame.gap_percent.to_numpy(dtype=float)
        summary = {
            "method": method,
            "dataset": "OriginalTSPLIB70",
            "instance_count": int(len(frame)),
            "valid_tours": int(frame.valid_tour.sum()),
            "mean_gap_percent": float(np.mean(gaps)),
            "median_gap_percent": float(np.median(gaps)),
            "mean_greedy_gap_percent": float(np.mean(frame.greedy_gap_percent)),
            "mean_accepted_repairs": float(np.mean(frame.accepted_repairs)),
            "rrc_budget": RRC_BUDGET,
            "rrc_seed": RRC_SEED,
            "checkpoint_sha256": checkpoint_sha,
            "evaluator_config_sha256": evaluator_config_sha256(),
            "input_transform": spec["input_transform"],
        }
        atomic_json(summary_path, summary)
    return frame


def write_selected_table(records: dict[str, dict[str, Any]]) -> Path:
    greedy_map = {
        "BQ_M0": TRAJ_ROOT / "work" / "M0" / "OriginalTSPLIB70.parquet",
        "BLADE_M16": TRAJ_ROOT / "work" / "M16" / "OriginalTSPLIB70.parquet",
        "Vanilla_LEHD": CMP_ROOT / "lehd_work" / "LEHD_vanilla_greedy" / "OriginalTSPLIB70.parquet",
        "EvoReal_LEHD": CMP_ROOT / "lehd_work" / "EvoReal_LEHD_greedy" / "OriginalTSPLIB70.parquet",
    }
    rrc_map = {
        "Vanilla_LEHD": WORK_ROOT / "LEHD_vanilla_rrc50" / "OriginalTSPLIB70.parquet",
        "EvoReal_LEHD": WORK_ROOT / "EvoReal_LEHD_rrc50" / "OriginalTSPLIB70.parquet",
    }

    def load_subset(path: Path) -> pd.DataFrame:
        if not path.is_file():
            return pd.DataFrame(columns=["instance_id", "predicted_cost_int", "gap_percent"])
        frame = pd.read_parquet(path, columns=["instance_id", "predicted_cost_int", "gap_percent"])
        frame["instance_id"] = frame.instance_id.astype(str)
        return frame

    rows = []
    for instance_id in PRIORITY_IDS:
        item = records[instance_id]
        row = {"instance_id": instance_id, "N": int(len(item["original"])), "opt": int(item["optimum"])}
        for name, path in greedy_map.items():
            sub = load_subset(path)
            hit = sub[sub.instance_id == instance_id]
            row[f"{name}_greedy_obj"] = int(hit.iloc[0].predicted_cost_int) if len(hit) else np.nan
            row[f"{name}_greedy_gap"] = float(hit.iloc[0].gap_percent) if len(hit) else np.nan
        for name, path in rrc_map.items():
            sub = load_subset(path)
            hit = sub[sub.instance_id == instance_id]
            row[f"{name}_rrc50_obj"] = int(hit.iloc[0].predicted_cost_int) if len(hit) else np.nan
            row[f"{name}_rrc50_gap"] = float(hit.iloc[0].gap_percent) if len(hit) else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    numeric = [column for column in table.columns if column not in {"instance_id"}]
    avg = {"instance_id": "AVERAGE_22", "N": float(table.N.mean()), "opt": np.nan}
    for column in numeric:
        if column == "N":
            continue
        series = table[column]
        avg[column] = float(series.mean()) if series.notna().all() else np.nan
    table = pd.concat([table, pd.DataFrame([avg])], ignore_index=True)
    dest = CMP_ROOT / "12_selected_original70_greedy_and_rrc50.csv"
    atomic_frame(dest, table)
    return dest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="*", default=["LEHD_vanilla_rrc50", "EvoReal_LEHD_rrc50"])
    parser.add_argument("--instances", choices=["priority", "all"], default="priority")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    records = load_records()
    if args.instances == "priority":
        instance_ids = list(PRIORITY_IDS)
    else:
        rest = [iid for iid in records if iid not in set(PRIORITY_IDS)]
        rest.sort()
        instance_ids = list(PRIORITY_IDS) + rest
    if args.limit:
        instance_ids = instance_ids[: args.limit]
    missing = [iid for iid in instance_ids if iid not in records]
    if missing:
        raise RuntimeError(f"missing Original70 IDs: {missing}")
    print(
        f"Original70 RRC50 methods={args.methods} n_instances={len(instance_ids)} "
        f"budget={RRC_BUDGET} seed={RRC_SEED}",
        flush=True,
    )
    for method in args.methods:
        evaluate_method(method, METHODS[method], records, instance_ids, device, args.force)
        write_selected_table(records)
    dest = write_selected_table(records)
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
