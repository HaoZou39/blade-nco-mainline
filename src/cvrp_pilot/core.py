"""Core, problem-specific contracts for the BQ-CVRP feasibility pilot.

The code in this module deliberately does not reuse the TSP hardness path.
BQ-CVRP represents each candidate customer with two actions (direct and via
depot), and training operates on expert-ordered suffixes with capacity state.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, distance
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import spearmanr
import torch
from torch.nn import functional as F


RUN_ID = "BLADE_BQ_CVRP_PILOT_V1"
SEED = 731
N_CUSTOMERS = 100
CAPACITY = 50
# HGS v2.0.0 rejects max distances above 100,000 as numerically unsafe.
# 10,000 preserves 1e-4 unit-square resolution while staying well inside
# the official guard. BQ continues to consume the original [0,1] floats.
HGS_SCALE = 10_000
M0_SHA256 = "9a21eb23eca1fa0d2b2b37bcd3b0c93335cd79e74002b89a39aeee8dd70aab60"
BQ_COMMIT = "b7f927faf1d87dca699361312471c10ecc8dd346"
HGS_COMMIT = "b062d5fccb1f64864dd65a94dac35929059ab089"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**32 - 1)


def instance_hash(depot: np.ndarray, customers: np.ndarray, demands: np.ndarray, capacity: int) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(depot, dtype="<f8").tobytes())
    digest.update(np.asarray(customers, dtype="<f8").tobytes())
    digest.update(np.asarray(demands, dtype="<i4").tobytes())
    digest.update(np.asarray([capacity], dtype="<i4").tobytes())
    return digest.hexdigest()


def coordinate_hash(coordinates: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(coordinates, dtype="<f8").tobytes()).hexdigest()


def atomic_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def load_bundle(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as blob:
        return {key: blob[key] for key in blob.files}


def save_bundle(path: str | Path, **arrays: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, target)


def bq_arrays(depot: np.ndarray, customers: np.ndarray, demands: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.concatenate([depot[None, :], customers, depot[None, :]], axis=0)
    demand = np.concatenate([np.array([0]), demands, np.array([0])], axis=0)
    return coords.astype(np.float32), demand.astype(np.float32)


def generate_role(
    role: str,
    count: int,
    operator_names: Sequence[str],
    apply_operator: Any,
    *,
    seed: int = SEED,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Generate isolated roots and one accepted FOGA child per instance.

    Only customer coordinates are passed to FOGA. Depot, demand vector,
    capacity and customer ordering remain byte-identical to their root.
    """

    depots = np.empty((count, 2), dtype=np.float64)
    customers = np.empty((count, N_CUSTOMERS, 2), dtype=np.float64)
    demands = np.empty((count, N_CUSTOMERS), dtype=np.int16)
    capacities = np.full(count, CAPACITY, dtype=np.int16)
    rows: list[dict[str, Any]] = []
    for index in range(count):
        root_rng = np.random.default_rng(stable_seed(RUN_ID, seed, role, "root", index))
        root_customers = root_rng.random((N_CUSTOMERS, 2))
        depot = root_rng.random(2)
        demand = root_rng.integers(1, 10, N_CUSTOMERS, dtype=np.int16)
        root_hash = instance_hash(depot, root_customers, demand, CAPACITY)
        operator = operator_names[index % len(operator_names)]
        accepted = None
        invocation_seed = None
        attempt = 0
        while accepted is None and attempt < 100:
            invocation_seed = stable_seed(RUN_ID, seed, role, "mutation", index, attempt)
            result = apply_operator(operator, root_customers, np.random.default_rng(invocation_seed))
            if result.proposal_status == "accepted":
                accepted = result
            attempt += 1
        if accepted is None:
            raise RuntimeError(f"could not generate accepted {operator} mutation for {role}/{index}")
        final = np.asarray(accepted.final_coordinates, dtype=np.float64)
        if final.shape != root_customers.shape:
            raise RuntimeError("FOGA changed customer cardinality")
        depots[index] = depot
        customers[index] = final
        demands[index] = demand
        child_hash = instance_hash(depot, final, demand, CAPACITY)
        rows.append(
            {
                "role": role,
                "instance_id": f"{role}_{index:06d}",
                "instance_hash": child_hash,
                "coordinate_hash": coordinate_hash(final),
                "root_id": f"{role}_root_{index:06d}",
                "root_hash": root_hash,
                "parent_id": f"{role}_root_{index:06d}",
                "operator": operator,
                "mutation_depth": 1,
                "mutation_seed": int(invocation_seed),
                "mutation_attempts": attempt,
                "mutation_parameters": json.dumps(accepted.sampled_parameters, sort_keys=True),
                "repair_count": int(accepted.repair_count),
                "depot_unchanged": True,
                "demands_unchanged": True,
                "capacity": CAPACITY,
            }
        )
    manifest = pd.DataFrame(rows)
    if manifest.instance_hash.nunique() != count or manifest.root_hash.nunique() != count:
        raise RuntimeError(f"{role} generation did not produce unique roots and instances")
    return {
        "depot": depots,
        "customers": customers,
        "demands": demands,
        "capacities": capacities,
    }, manifest


def gini(values: np.ndarray) -> float:
    x = np.sort(np.asarray(values, dtype=np.float64))
    if x.size == 0 or float(x.sum()) == 0.0:
        return 0.0
    indices = np.arange(1, x.size + 1, dtype=np.float64)
    return float((2.0 * np.sum(indices * x) / (x.size * x.sum())) - (x.size + 1.0) / x.size)


FEATURE_NAMES = (
    "x_mean", "y_mean", "x_std", "y_std", "x_range", "y_range", "bbox_area", "convex_hull_area",
    "centroid_radial_mean", "centroid_radial_std", "centroid_radial_q90", "centroid_radial_max",
    "nn_mean", "nn_std", "nn_min", "nn_q10", "nn_median", "nn_q90", "nn_max",
    "pair_mean", "pair_std", "pair_cv", "pair_q10", "pair_median", "pair_q90", "pair_max",
    "mst_total", "mst_mean", "mst_std", "mst_max",
    "depot_centroid_distance", "depot_dist_mean", "depot_dist_std", "depot_dist_cv", "depot_dist_min",
    "depot_dist_q10", "depot_dist_median", "depot_dist_q90", "depot_dist_max",
    "demand_mean", "demand_std", "demand_cv", "demand_min", "demand_q10", "demand_median",
    "demand_q90", "demand_max", "demand_gini", "demand_entropy",
    "total_demand_capacity_ratio", "vehicle_count_lower_bound", "max_demand_capacity_ratio",
    "mean_demand_capacity_ratio", "demand_x_corr", "demand_y_corr", "demand_depot_distance_corr",
    "demand_centroid_radius_corr", "weighted_centroid_offset", "nearest_neighbor_demand_difference",
    "high_demand_spatial_dispersion", "low_demand_spatial_dispersion",
)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else 0.0


def structural_features(depot: np.ndarray, customers: np.ndarray, demands: np.ndarray, capacity: float) -> np.ndarray:
    xy = np.asarray(customers, dtype=np.float64)
    q = np.asarray(demands, dtype=np.float64)
    centroid = xy.mean(axis=0)
    radial = np.linalg.norm(xy - centroid, axis=1)
    matrix = distance.squareform(distance.pdist(xy))
    np.fill_diagonal(matrix, np.inf)
    nearest_idx = np.argmin(matrix, axis=1)
    nearest = matrix[np.arange(len(xy)), nearest_idx]
    pair = distance.pdist(xy)
    finite_matrix = matrix.copy()
    np.fill_diagonal(finite_matrix, 0.0)
    mst_edges = np.asarray(minimum_spanning_tree(finite_matrix).data, dtype=np.float64)
    depot_dist = np.linalg.norm(xy - depot[None, :], axis=1)
    hull_area = float(ConvexHull(xy).volume) if len(xy) >= 3 else 0.0
    probs = np.bincount(q.astype(int), minlength=10)[1:10].astype(np.float64)
    probs = probs[probs > 0] / len(q)
    entropy = float(-np.sum(probs * np.log(probs)))
    weights = q / q.sum()
    weighted_centroid = np.sum(xy * weights[:, None], axis=0)
    high = xy[q >= np.quantile(q, 0.75)]
    low = xy[q <= np.quantile(q, 0.25)]

    def dispersion(points: np.ndarray) -> float:
        return float(np.mean(np.linalg.norm(points - points.mean(axis=0), axis=1))) if len(points) else 0.0

    values = (
        xy[:, 0].mean(), xy[:, 1].mean(), xy[:, 0].std(), xy[:, 1].std(), np.ptp(xy[:, 0]), np.ptp(xy[:, 1]),
        np.ptp(xy[:, 0]) * np.ptp(xy[:, 1]), hull_area,
        radial.mean(), radial.std(), np.quantile(radial, 0.9), radial.max(),
        nearest.mean(), nearest.std(), nearest.min(), np.quantile(nearest, 0.1), np.median(nearest),
        np.quantile(nearest, 0.9), nearest.max(), pair.mean(), pair.std(), pair.std() / max(pair.mean(), 1e-12),
        np.quantile(pair, 0.1), np.median(pair), np.quantile(pair, 0.9), pair.max(),
        mst_edges.sum(), mst_edges.mean(), mst_edges.std(), mst_edges.max(),
        np.linalg.norm(depot - centroid), depot_dist.mean(), depot_dist.std(), depot_dist.std() / max(depot_dist.mean(), 1e-12),
        depot_dist.min(), np.quantile(depot_dist, 0.1), np.median(depot_dist), np.quantile(depot_dist, 0.9), depot_dist.max(),
        q.mean(), q.std(), q.std() / max(q.mean(), 1e-12), q.min(), np.quantile(q, 0.1), np.median(q),
        np.quantile(q, 0.9), q.max(), gini(q), entropy, q.sum() / capacity, math.ceil(q.sum() / capacity),
        q.max() / capacity, q.mean() / capacity, _safe_corr(q, xy[:, 0]), _safe_corr(q, xy[:, 1]),
        _safe_corr(q, depot_dist), _safe_corr(q, radial), np.linalg.norm(weighted_centroid - centroid),
        np.mean(np.abs(q - q[nearest_idx])), dispersion(high), dispersion(low),
    )
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise RuntimeError("non-finite or malformed structural feature vector")
    return result


def extract_feature_frame(bundle: dict[str, np.ndarray], manifest: pd.DataFrame) -> pd.DataFrame:
    matrix = np.vstack(
        [
            structural_features(bundle["depot"][i], bundle["customers"][i], bundle["demands"][i], bundle["capacities"][i])
            for i in range(len(manifest))
        ]
    )
    frame = pd.DataFrame(matrix, columns=FEATURE_NAMES)
    frame.insert(0, "instance_hash", manifest.instance_hash.to_numpy())
    frame.insert(0, "instance_id", manifest.instance_id.to_numpy())
    return frame


@dataclass(frozen=True)
class ReferenceRecord:
    instance_id: str
    instance_hash: str
    routes: list[list[int]]
    hgs_cost_scaled: int
    rounded_cost: float
    euclidean_cost: float
    route_count: int
    wall_s: float
    feasible: bool
    status: str
    error: str = ""


def write_vrplib(path: str | Path, depot: np.ndarray, customers: np.ndarray, demands: np.ndarray, capacity: int) -> None:
    target = Path(path)
    coords = np.vstack([depot, customers])
    lines = [
        f"NAME : {target.stem}", "COMMENT : BLADE_BQ_CVRP_PILOT_V1", "TYPE : CVRP",
        f"DIMENSION : {len(coords)}", "EDGE_WEIGHT_TYPE : EUC_2D", f"CAPACITY : {int(capacity)}", "NODE_COORD_SECTION",
    ]
    for index, point in enumerate(coords, 1):
        scaled = np.rint(point * HGS_SCALE).astype(np.int64)
        lines.append(f"{index} {scaled[0]} {scaled[1]}")
    lines.append("DEMAND_SECTION")
    lines.append("1 0")
    lines.extend(f"{index + 2} {int(value)}" for index, value in enumerate(demands))
    lines.extend(["DEPOT_SECTION", "1", "-1", "EOF"])
    target.write_text("\n".join(lines) + "\n", encoding="ascii")


def parse_hgs_solution(path: str | Path) -> tuple[list[list[int]], int]:
    routes: list[list[int]] = []
    cost = None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if raw.startswith("Route #"):
            routes.append([int(value) for value in raw.split(":", 1)[1].split()])
        elif raw.startswith("Cost "):
            cost = int(round(float(raw.split()[1])))
    if not routes or cost is None:
        raise ValueError("HGS solution lacks routes or cost")
    return routes, cost


def validate_routes(
    depot: np.ndarray,
    customers: np.ndarray,
    demands: np.ndarray,
    capacity: int,
    routes: Sequence[Sequence[int]],
) -> tuple[int, float, bool, str]:
    flat = [int(node) for route in routes for node in route]
    expected = list(range(1, len(customers) + 1))
    if sorted(flat) != expected:
        return 0, float("nan"), False, "customers are not visited exactly once"
    for route in routes:
        load = int(sum(int(demands[node - 1]) for node in route))
        if load > capacity:
            return 0, float("nan"), False, f"route load {load} exceeds {capacity}"
    scaled = np.rint(np.vstack([depot, customers]) * HGS_SCALE).astype(np.int64)
    rounded_cost = 0
    euclidean_cost = 0.0
    for route in routes:
        path = [0] + [int(node) for node in route] + [0]
        for left, right in zip(path, path[1:]):
            rounded_cost += int(round(float(np.linalg.norm(scaled[left] - scaled[right]))))
            euclidean_cost += float(np.linalg.norm(customers[left - 1] - customers[right - 1])) if left and right else (
                float(np.linalg.norm(depot - customers[right - 1])) if left == 0 and right else float(np.linalg.norm(customers[left - 1] - depot))
            )
    return rounded_cost, euclidean_cost, True, ""


def solve_hgs_one(
    hgs_binary: Path,
    work_dir: Path,
    instance_id: str,
    instance_hash_value: str,
    depot: np.ndarray,
    customers: np.ndarray,
    demands: np.ndarray,
    capacity: int,
    *,
    seed: int,
    time_limit: float,
) -> ReferenceRecord:
    work_dir.mkdir(parents=True, exist_ok=True)
    cache = work_dir / f"{instance_id}.json"
    if cache.is_file():
        payload = json.loads(cache.read_text(encoding="utf-8"))
        if payload.get("instance_hash") == instance_hash_value and payload.get("status") == "ok":
            return ReferenceRecord(**payload)
    vrp = work_dir / f"{instance_id}.vrp"
    sol = work_dir / f"{instance_id}.sol"
    write_vrplib(vrp, depot, customers, demands, capacity)
    started = time.perf_counter()
    command = [str(hgs_binary), str(vrp), str(sol), "-seed", str(seed), "-t", str(time_limit), "-round", "1", "-log", "0"]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=max(60, time_limit * 5))
    wall = time.perf_counter() - started
    try:
        if process.returncode != 0:
            raise RuntimeError(f"HGS exit {process.returncode}: {process.stdout[-1000:]}")
        if not sol.is_file():
            raise RuntimeError(f"HGS produced no solution file; output: {process.stdout[-1000:]}")
        routes, hgs_cost = parse_hgs_solution(sol)
        rounded, euclidean, feasible, error = validate_routes(depot, customers, demands, capacity, routes)
        if not feasible:
            raise RuntimeError(error)
        if rounded != hgs_cost:
            raise RuntimeError(f"HGS cost {hgs_cost} != recomputed rounded cost {rounded}")
        record = ReferenceRecord(instance_id, instance_hash_value, routes, hgs_cost, rounded / HGS_SCALE, euclidean, len(routes), wall, True, "ok")
    except Exception as exc:
        record = ReferenceRecord(instance_id, instance_hash_value, [], 0, float("nan"), float("nan"), 0, wall, False, "failed", str(exc))
    atomic_json(cache, record.__dict__)
    if record.status != "ok":
        raise RuntimeError(f"HGS failed for {instance_id}: {record.error}")
    return record


def solve_hgs_bundle(
    hgs_binary: Path,
    work_dir: Path,
    bundle: dict[str, np.ndarray],
    manifest: pd.DataFrame,
    *,
    workers: int,
    time_limit: float,
    seed: int = SEED,
) -> list[ReferenceRecord]:
    records: list[ReferenceRecord | None] = [None] * len(manifest)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for index, row in manifest.reset_index(drop=True).iterrows():
            future = executor.submit(
                solve_hgs_one, hgs_binary, work_dir, str(row.instance_id), str(row.instance_hash),
                bundle["depot"][index], bundle["customers"][index], bundle["demands"][index], int(bundle["capacities"][index]),
                seed=stable_seed(seed, row.instance_id) % (2**31 - 1), time_limit=time_limit,
            )
            futures[future] = index
        for done_index, future in enumerate(as_completed(futures), 1):
            records[futures[future]] = future.result()
            if done_index % max(1, min(100, len(manifest))) == 0:
                print(f"HGS_PROGRESS={done_index}/{len(manifest)}", flush=True)
    return [record for record in records if record is not None]


def official_trajectory(
    depot: np.ndarray,
    customers: np.ndarray,
    demands: np.ndarray,
    capacity: int,
    routes: Sequence[Sequence[int]],
) -> dict[str, np.ndarray]:
    """Convert HGS routes to the exact BQ-CVRP reordered trajectory format."""

    residual = np.asarray([capacity - sum(int(demands[node - 1]) for node in route) for route in routes])
    order = np.argsort(residual, kind="stable")
    ordered_routes = [list(routes[index]) for index in order]
    flattened = [node for route in ordered_routes for node in route]
    if sorted(flattened) != list(range(1, len(customers) + 1)):
        raise ValueError("reference routes are not a customer permutation")
    trajectory_coords = np.vstack([depot, customers[np.asarray(flattened) - 1], depot]).astype(np.float32)
    trajectory_demands = np.concatenate([[0], demands[np.asarray(flattened) - 1], [0]]).astype(np.float32)
    remaining = [float(capacity)]
    via = [0]
    for route_index, route in enumerate(ordered_routes):
        current = float(capacity)
        for node_index, node in enumerate(route):
            current -= float(demands[node - 1])
            remaining.append(current)
            via.append(1 if route_index > 0 and node_index == 0 else 0)
    remaining.append(float(capacity))
    via.append(0)
    remaining_array = np.asarray(remaining, dtype=np.float32)
    via_array = np.asarray(via, dtype=np.int64)
    # The official converter detects route separators by capacity increases.
    official_via = np.zeros_like(via_array)
    official_via[0] = 1
    official_via[1:-1] = remaining_array[1:-1] > remaining_array[:-2]
    if not np.array_equal(via_array[1:-1], official_via[1:-1]):
        raise RuntimeError("route ordering does not reproduce official capacity-increase depot separators")
    return {
        "coords": trajectory_coords,
        "demands": trajectory_demands,
        "remaining_capacities": remaining_array,
        "via_depots": official_via,
        "route_order": order.astype(np.int64),
    }


def trajectory_bundle(
    bundle: dict[str, np.ndarray], references: Sequence[ReferenceRecord]
) -> dict[str, np.ndarray]:
    trajectories = [
        official_trajectory(bundle["depot"][i], bundle["customers"][i], bundle["demands"][i], int(bundle["capacities"][i]), record.routes)
        for i, record in enumerate(references)
    ]
    return {
        "coords": np.stack([item["coords"] for item in trajectories]),
        "demands": np.stack([item["demands"] for item in trajectories]),
        "remaining_capacities": np.stack([item["remaining_capacities"] for item in trajectories]),
        "via_depots": np.stack([item["via_depots"] for item in trajectories]),
        "capacities": bundle["capacities"].astype(np.float32),
    }


@torch.no_grad()
def per_instance_hardness(
    model: torch.nn.Module,
    trajectories: dict[str, np.ndarray],
    device: torch.device,
    *,
    batch_size: int = 128,
) -> np.ndarray:
    """Expected official sampled-suffix CE, exhaustively averaged per instance."""

    model.eval()
    n_instances, n_nodes, _ = trajectories["coords"].shape
    # Official collate: randint(0, nb_nodes - 3), hence starts [0, nb_nodes-4].
    starts = range(0, n_nodes - 3)
    sums = np.zeros(n_instances, dtype=np.float64)
    for begin in starts:
        for left in range(0, n_instances, batch_size):
            right = min(n_instances, left + batch_size)
            coords = torch.as_tensor(trajectories["coords"][left:right, begin:], dtype=torch.float32, device=device)
            demands = torch.as_tensor(trajectories["demands"][left:right, begin:], dtype=torch.float32, device=device)
            capacities = torch.as_tensor(trajectories["capacities"][left:right], dtype=torch.float32, device=device)
            remaining = torch.as_tensor(trajectories["remaining_capacities"][left:right, begin], dtype=torch.float32, device=device)
            inputs = torch.cat(
                [coords, (demands / capacities[:, None]).unsqueeze(-1), (remaining / capacities)[:, None, None].expand(-1, coords.shape[1], 1)], dim=-1
            )
            scores = model(inputs, demands=demands, remaining_capacities=remaining)
            via = torch.as_tensor(trajectories["via_depots"][left:right, begin + 1], dtype=torch.long, device=device)
            target = torch.full((right - left,), 2, dtype=torch.long, device=device) + via
            loss = F.cross_entropy(scores, target, reduction="none")
            sums[left:right] += loss.detach().cpu().double().numpy()
    return sums / float(len(starts))


def official_loss_equivalence(
    model: torch.nn.Module,
    trajectories: dict[str, np.ndarray],
    device: torch.device,
    *,
    begin: int = 17,
) -> dict[str, float | bool]:
    n = min(8, len(trajectories["coords"]))
    coords = torch.as_tensor(trajectories["coords"][:n, begin:], dtype=torch.float32, device=device)
    demands = torch.as_tensor(trajectories["demands"][:n, begin:], dtype=torch.float32, device=device)
    capacities = torch.as_tensor(trajectories["capacities"][:n], dtype=torch.float32, device=device)
    remaining = torch.as_tensor(trajectories["remaining_capacities"][:n, begin], dtype=torch.float32, device=device)
    via = torch.as_tensor(trajectories["via_depots"][:n, begin + 1], dtype=torch.long, device=device)
    inputs = torch.cat([coords, (demands / capacities[:, None]).unsqueeze(-1), (remaining / capacities)[:, None, None].expand(-1, coords.shape[1], 1)], dim=-1)
    scores = model(inputs, demands=demands, remaining_capacities=remaining)
    target = torch.full((n,), 2, dtype=torch.long, device=device) + via
    official = torch.nn.CrossEntropyLoss()(scores, target)
    audited = F.cross_entropy(scores, target, reduction="none").mean()
    delta = float(abs(official.detach().item() - audited.detach().item()))
    return {"official_loss": float(official.item()), "audited_loss": float(audited.item()), "absolute_delta": delta, "pass": delta <= 1e-12}


def hardness_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, top_fraction: float = 0.1) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    count = max(1, int(math.ceil(len(y_true) * top_fraction)))
    true_top = set(np.argsort(y_true)[-count:])
    pred_top = set(np.argsort(y_pred)[-count:])
    global_rate = count / len(y_true)
    hit_rate = len(true_top & pred_top) / count
    threshold = float(np.partition(y_true, -count)[-count])
    selected_rate = float(np.mean(y_true[list(pred_top)] >= threshold))
    return {
        "n": float(len(y_true)),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "true_top10_hit_rate": float(hit_rate),
        "true_top10_enrichment": float(selected_rate / global_rate),
        "predicted_top10_true_hardness_mean": float(np.mean(y_true[list(pred_top)])),
        "overall_true_hardness_mean": float(np.mean(y_true)),
    }


def bootstrap_mean_ci(values: np.ndarray, *, replicates: int = 10_000, seed: int = SEED) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    chunk = 500
    for begin in range(0, replicates, chunk):
        end = min(replicates, begin + chunk)
        indices = rng.integers(0, len(x), size=(end - begin, len(x)))
        means[begin:end] = x[indices].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def parse_cvrplib(path: str | Path) -> dict[str, Any]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    dimension = int(next(line.split(":", 1)[1] for line in lines if line.strip().startswith("DIMENSION")))
    capacity = int(float(next(line.split(":", 1)[1] for line in lines if line.strip().startswith("CAPACITY"))))
    coord_start = next(i for i, line in enumerate(lines) if line.strip() == "NODE_COORD_SECTION") + 1
    demand_start = next(i for i, line in enumerate(lines) if line.strip() == "DEMAND_SECTION") + 1
    coords = np.asarray([[float(value) for value in lines[coord_start + i].split()[1:3]] for i in range(dimension)])
    demands = np.asarray([int(float(lines[demand_start + i].split()[1])) for i in range(dimension)], dtype=np.int16)
    return {"name": Path(path).stem, "depot": coords[0], "customers": coords[1:], "demands": demands[1:], "capacity": capacity}


def normalize_cvrplib_coords(depot: np.ndarray, customers: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    all_coords = np.vstack([depot, customers]).astype(np.float64)
    minimum = all_coords.min(axis=0)
    scale = float(np.max(np.ptp(all_coords, axis=0)))
    if scale <= 0:
        raise ValueError("degenerate CVRPLIB coordinate scale")
    normalized = (all_coords - minimum) / scale
    return normalized[0], normalized[1:], {"offset_x": float(minimum[0]), "offset_y": float(minimum[1]), "isotropic_scale": scale}


def route_cost_from_path(coords: np.ndarray, path: Sequence[int], *, rounded: bool) -> float:
    total = 0.0
    for left, right in zip(path, path[1:]):
        value = float(np.linalg.norm(coords[left] - coords[right]))
        total += round(value) if rounded else value
    return total


def paired_gap_frame(frame: pd.DataFrame, reference_column: str = "reference_cost") -> pd.DataFrame:
    out = frame.copy()
    for model in ("M0", "M1_RANDOM", "M1_HGB"):
        out[f"gap_{model}"] = 100.0 * (out[f"cost_{model}"] - out[reference_column]) / out[reference_column]
    out["delta_gap_hgb_minus_random"] = out["gap_M1_HGB"] - out["gap_M1_RANDOM"]
    out["paired_improvement_hgb_over_random"] = out["gap_M1_RANDOM"] - out["gap_M1_HGB"]
    return out
