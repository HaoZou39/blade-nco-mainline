"""Passive canonical OriginalTSPLIB70 reevaluation for the frozen M0--M20 chain.

This diagnostic never writes into the formal run.  It uses the historical
aspect-ratio-preserving model-input transform, greedy BQ decoding with K=250,
and native TSPLIB EUC_2D nint scoring on the unnormalised coordinates.
Per-checkpoint caches make the job safely resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "blade_final_20cycle_seed2026_v1"
OUT = ROOT / "audits" / "original70_canonical_trajectory_v1"
COMPLETE_MARKER = RUN / "BLADE_FINAL_20CYCLE_EXECUTION_COMPLETE"
BQ_ROOT = Path(r"D:\NCO\bq-nco-main\bq-nco-main")
sys.path.insert(0, str(ROOT / "src"))

from bq.utils import sha256_file  # noqa: E402
from evaluation.datasets import load_original_tsplib70  # noqa: E402


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def aspect_preserving_model_input(raw: np.ndarray) -> np.ndarray:
    points = np.asarray(raw, dtype=np.float64)
    minima = points.min(axis=0)
    span = points.max(axis=0) - minima
    scale = float(np.max(span))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("invalid Original70 coordinate span")
    return np.ascontiguousarray((points - minima) / scale, dtype=np.float64)


def tsplib_euc2d_cost(raw: np.ndarray, tour: list[int]) -> int:
    points = np.asarray(raw, dtype=np.float64)
    order = np.asarray(tour, dtype=np.int64)
    if not np.array_equal(np.sort(order), np.arange(len(points))):
        raise RuntimeError("decoded tour is not a permutation")
    nxt = np.roll(order, -1)
    distances = np.linalg.norm(points[order] - points[nxt], axis=1)
    return int(np.floor(distances + 0.5).astype(np.int64).sum())


def tour_sha256(tour: list[int]) -> str:
    return hashlib.sha256(" ".join(map(str, tour)).encode("ascii")).hexdigest()


def historical_model_instance(device: torch.device):
    if not (BQ_ROOT / "model" / "model.py").is_file():
        raise FileNotFoundError(f"historical BQ implementation is absent: {BQ_ROOT}")
    sys.path.insert(0, str(BQ_ROOT))
    from model.model import BQModel

    return BQModel(2, 192, 512, "relu", 9, 12, "softmax", 0.0, False, "tsp").to(device)


def historical_decode_original_knn(
    raw: np.ndarray,
    net: torch.nn.Module,
    device: torch.device,
    knns: int = 250,
) -> list[int]:
    """Exact decoder call path used by the authoritative 5.375617545% run."""
    normalized = aspect_preserving_model_input(raw)
    original_n = len(normalized)
    state_coords = np.concatenate([normalized, normalized[:1]], axis=0)
    state_ids = np.concatenate(
        [np.arange(original_n, dtype=np.int64), np.asarray([0], dtype=np.int64)]
    )
    path = [0]
    for _ in range(1, original_n):
        state_n = len(state_ids)
        if 0 < knns < state_n:
            squared = np.sum((state_coords[:-1] - state_coords[0]) ** 2, axis=1)
            candidate = np.argsort(squared, kind="stable")[: knns - 1]
            local = np.concatenate([candidate, np.asarray([state_n - 1])])
        else:
            local = np.arange(state_n)
        inputs = torch.as_tensor(
            state_coords[local][None], dtype=torch.float32, device=device
        )
        with torch.inference_mode():
            scores = net(inputs)
        selected_input = int(torch.argmax(scores, dim=1).item())
        selected_state = int(local[selected_input])
        if selected_state in (0, state_n - 1):
            raise RuntimeError("BQ selected a masked endpoint")
        selected_id = int(state_ids[selected_state])
        path.append(selected_id)
        keep = np.ones(state_n, dtype=bool)
        keep[selected_state] = False
        remaining_coords = state_coords[keep][1:]
        remaining_ids = state_ids[keep][1:]
        state_coords = np.concatenate(
            [state_coords[selected_state : selected_state + 1], remaining_coords], axis=0
        )
        state_ids = np.concatenate(
            [state_ids[selected_state : selected_state + 1], remaining_ids], axis=0
        )
    if sorted(path) != list(range(original_n)):
        raise RuntimeError("historical decoder did not return a complete permutation")
    return path


def checkpoints(through: int) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = [("M0", ROOT / "assets" / "solver" / "M0" / "tsp.best")]
    for model_index in range(1, through + 1):
        cycle = model_index - 1
        path = RUN / "cycles" / f"C{cycle:02d}" / "09_bq_training" / f"M{model_index}_E256.tsp.best"
        found.append((f"M{model_index}", path))
    missing = [str(path) for _, path in found if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing checkpoints:\n" + "\n".join(missing))
    return found


def load_cache(path: Path, model: str, checkpoint_hash: str) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    required = {
        "model", "checkpoint_sha256", "instance_id", "dimension", "optimal_cost_int",
        "predicted_cost_int", "gap_percent", "tour_0based", "tour_sha256", "runtime_sec",
        "knns", "preprocessing", "cost_metric", "evaluator",
    }
    if not required.issubset(frame.columns):
        raise RuntimeError(f"invalid cache schema: {path}")
    frame = frame.loc[
        frame["model"].astype(str).eq(model)
        & frame["checkpoint_sha256"].astype(str).eq(checkpoint_hash)
        & frame["knns"].astype(int).eq(250)
    ].copy()
    return frame.drop_duplicates("instance_id", keep="last")


def evaluate_checkpoint(
    model_name: str,
    checkpoint_path: Path,
    instances: list[tuple[str, np.ndarray, int]],
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_hash = sha256_file(checkpoint_path)
    cache_path = OUT / "per_checkpoint" / f"{model_name}.parquet"
    cache = load_cache(cache_path, model_name, checkpoint_hash)
    rows = cache.to_dict("records")
    completed = set(cache.get("instance_id", pd.Series(dtype=str)).astype(str))
    blob = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "net" not in blob:
        raise RuntimeError(f"checkpoint missing 'net': {checkpoint_path}")
    net = historical_model_instance(device)
    net.load_state_dict(blob["net"], strict=True)
    net.eval()
    fresh = 0
    for instance_id, raw, optimum in instances:
        if instance_id in completed:
            continue
        started = time.perf_counter()
        tour = historical_decode_original_knn(raw, net, device, knns=250)
        predicted = tsplib_euc2d_cost(raw, tour)
        gap = 100.0 * (predicted - int(optimum)) / int(optimum)
        elapsed = time.perf_counter() - started
        if gap < -1e-12:
            raise RuntimeError(f"negative gap for {model_name}/{instance_id}: {gap}")
        tour_text = " ".join(map(str, tour))
        rows.append({
            "model": model_name,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "instance_id": instance_id,
            "dimension": int(len(raw)),
            "optimal_cost_int": int(optimum),
            "predicted_cost_int": predicted,
            "gap_percent": gap,
            "tour_0based": tour_text,
            "tour_sha256": tour_sha256(tour),
            "runtime_sec": elapsed,
            "knns": 250,
            "preprocessing": "translate_min_divide_by_max_axis_span_aspect_preserving",
            "cost_metric": "TSPLIB_EUC_2D_floor_distance_plus_0.5_on_raw_coordinates",
            "evaluator": "historical_authoritative_external_BQModel_stepwise_decoder",
        })
        fresh += 1
        atomic_parquet(pd.DataFrame(rows), cache_path)
        print(
            f"[{model_name}] {len(rows)}/70 {instance_id} N={len(raw)} "
            f"gap={gap:.6f}% elapsed={elapsed:.2f}s",
            flush=True,
        )
    del net, blob
    torch.cuda.empty_cache()
    final = load_cache(cache_path, model_name, checkpoint_hash)
    if len(final) != 70 or final["instance_id"].nunique() != 70:
        raise RuntimeError(f"incomplete cache for {model_name}: {len(final)}/70")
    values = final["gap_percent"].astype(float).to_numpy()
    return {
        "model": model_name,
        "checkpoint_sha256": checkpoint_hash,
        "instance_count": 70,
        "mean_gap_percent": float(np.mean(values)),
        "median_gap_percent": float(np.median(values)),
        "P75": float(np.percentile(values, 75)),
        "P90": float(np.percentile(values, 90)),
        "P95": float(np.percentile(values, 95)),
        "max_gap_percent": float(np.max(values)),
        "total_inference_sec": float(final["runtime_sec"].astype(float).sum()),
        "freshly_evaluated": fresh,
        "reused": 70 - fresh,
        "cache_path": str(cache_path.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--through", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--require-formal-complete", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not 0 <= args.through <= 20:
        raise ValueError("--through must be in [0,20]")
    if args.require_formal_complete and not COMPLETE_MARKER.is_file():
        raise RuntimeError(f"formal completion marker is absent: {COMPLETE_MARKER}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Original70 canonical reevaluation")

    selected = list(load_original_tsplib70())
    if len(selected) != 70 or len({item[0] for item in selected}) != 70:
        raise RuntimeError("OriginalTSPLIB70 identity set is not exactly 70 unique instances")
    chain = checkpoints(args.through)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")

    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json({
        "status": "RUNNING",
        "formal_run": str(RUN.resolve()),
        "formal_completion_marker": str(COMPLETE_MARKER.resolve()),
        "through": args.through,
        "models": [name for name, _ in chain],
        "checkpoint_hashes": {name: sha256_file(path) for name, path in chain},
        "instance_count": 70,
        "knns": 250,
        "preprocessing": "translate_min_divide_by_max_axis_span_aspect_preserving",
        "cost_metric": "TSPLIB_EUC_2D_floor_distance_plus_0.5_on_raw_coordinates",
        "gap_formula": "100*(predicted-optimal)/optimal; arithmetic mean over 70",
        "evaluator": "historical_authoritative_external_BQModel_stepwise_decoder",
        "bq_implementation_root": str(BQ_ROOT.resolve()),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "seed": args.seed,
    }, OUT / "run_manifest.json")

    summaries = []
    for model_name, checkpoint_path in chain:
        summaries.append(evaluate_checkpoint(model_name, checkpoint_path, selected, device))
        frame = pd.DataFrame(summaries)
        frame["model_index"] = frame["model"].str.extract(r"(\d+)").astype(int)
        frame = frame.sort_values("model_index").drop(columns="model_index")
        atomic_csv(frame, OUT / "original70_canonical_trajectory.csv")

    combined = pd.concat(
        [pd.read_parquet(OUT / "per_checkpoint" / f"{name}.parquet") for name, _ in chain],
        ignore_index=True,
    )
    atomic_parquet(combined, OUT / "original70_canonical_per_instance.parquet")
    expected_m0 = 5.375617545006024
    observed_m0 = float(pd.DataFrame(summaries).set_index("model").loc["M0", "mean_gap_percent"])
    atomic_json({
        "status": "COMPLETE",
        "models": len(chain),
        "records": len(combined),
        "m0_expected_historical": expected_m0,
        "m0_observed_fresh": observed_m0,
        "m0_delta_percentage_points": observed_m0 - expected_m0,
        "trajectory": str((OUT / "original70_canonical_trajectory.csv").resolve()),
        "per_instance": str((OUT / "original70_canonical_per_instance.parquet").resolve()),
    }, OUT / "run_manifest.json")
    scientific_outputs = [
        *(OUT / "per_checkpoint").glob("*.parquet"),
        OUT / "original70_canonical_trajectory.csv",
        OUT / "original70_canonical_per_instance.parquet",
        OUT / "run_manifest.json",
    ]
    files = [
        {"path": path.relative_to(OUT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(scientific_outputs)
    ]
    atomic_json({"files": files, "count": len(files)}, OUT / "output_hash_manifest.json")
    print(f"ORIGINAL70_CANONICAL_TRAJECTORY_COMPLETE models={len(chain)} M0={observed_m0:.9f}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
