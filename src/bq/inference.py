"""Greedy and CRN stochastic BQ inference."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch
from torch.nn import Module

from .decoding import greedy_decode, sampling_decode
from .protocol import INTEGER_SCALE, KNNS, START_NODE
from .tsp_metric import decode_tour_to_permutation, integer_tour_cost


def closed_inference_coordinates(coords: np.ndarray) -> np.ndarray:
    xy = np.asarray(coords, dtype=np.float64)
    if xy.ndim != 3 or xy.shape[-1] != 2:
        raise ValueError(f"expected coords (B,N,2), got {xy.shape}")
    return np.concatenate([xy, xy[:, :1, :]], axis=1)


CRN_SEED_FIELDS = ("sampling_seed", "instance_id", "sample_id", "shuffle_seed")
CRN_SEED_FIELDS_EVAL = ("sampling_seed", "instance_id", "sample_id")
CRN_SEED_FIELDS_DATASET = ("sampling_seed", "dataset_role", "instance_id", "sample_id")


def crn_sampling_seed(
    sampling_seed: int,
    instance_id: str,
    sample_id: int,
    shuffle_seed: int,
) -> int:
    payload = f"{int(sampling_seed)}|{instance_id}|{int(sample_id)}|{int(shuffle_seed)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31 - 1)


def crn_sampling_seed_eval(
    sampling_seed: int,
    instance_id: str,
    sample_id: int,
) -> int:
    payload = f"{int(sampling_seed)}|{instance_id}|{int(sample_id)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31 - 1)


def crn_sampling_seed_dataset(
    sampling_seed: int,
    dataset_role: str,
    instance_id: str,
    sample_id: int,
) -> int:
    payload = f"{int(sampling_seed)}|{dataset_role}|{instance_id}|{int(sample_id)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31 - 1)


def crn_seed_matrix(
    instance_ids: list[str],
    num_samples: int,
    sampling_seed: int,
    shuffle_seed: int,
) -> np.ndarray:
    seeds = np.empty((len(instance_ids), num_samples), dtype=np.int64)
    for i, instance_id in enumerate(instance_ids):
        for sample_id in range(num_samples):
            seeds[i, sample_id] = crn_sampling_seed(
                sampling_seed, str(instance_id), sample_id, shuffle_seed
            )
    return seeds


def crn_seed_matrix_eval(
    instance_ids: list[str],
    num_samples: int,
    sampling_seed: int,
) -> np.ndarray:
    seeds = np.empty((len(instance_ids), num_samples), dtype=np.int64)
    for i, instance_id in enumerate(instance_ids):
        for sample_id in range(num_samples):
            seeds[i, sample_id] = crn_sampling_seed_eval(
                sampling_seed, str(instance_id), sample_id
            )
    return seeds


def crn_seed_matrix_dataset(
    instance_ids: list[str],
    num_samples: int,
    sampling_seed: int,
    dataset_role: str,
) -> np.ndarray:
    seeds = np.empty((len(instance_ids), num_samples), dtype=np.int64)
    role = str(dataset_role)
    for i, instance_id in enumerate(instance_ids):
        for sample_id in range(num_samples):
            seeds[i, sample_id] = crn_sampling_seed_dataset(
                sampling_seed, role, str(instance_id), sample_id
            )
    return seeds


def crn_uniforms(
    instance_ids: list[str],
    num_samples: int,
    n_steps: int,
    sampling_seed: int,
    shuffle_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return uniforms (B, S, n_steps) and CRN seeds (B, S). Independent of K."""

    seeds = crn_seed_matrix(instance_ids, num_samples, sampling_seed, shuffle_seed)
    batch = np.empty((len(instance_ids), num_samples, n_steps), dtype=np.float64)
    for i, instance_id in enumerate(instance_ids):
        for sample_id in range(num_samples):
            rng = np.random.default_rng(int(seeds[i, sample_id]))
            batch[i, sample_id] = rng.random(n_steps)
    return batch, seeds


def crn_uniforms_eval(
    instance_ids: list[str],
    num_samples: int,
    n_steps: int,
    sampling_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluation CRN uniforms. Independent of LR, K, and adaptation seed."""

    seeds = crn_seed_matrix_eval(instance_ids, num_samples, sampling_seed)
    batch = np.empty((len(instance_ids), num_samples, n_steps), dtype=np.float64)
    for i, instance_id in enumerate(instance_ids):
        for sample_id in range(num_samples):
            rng = np.random.default_rng(int(seeds[i, sample_id]))
            batch[i, sample_id] = rng.random(n_steps)
    return batch, seeds


def crn_uniforms_dataset(
    instance_ids: list[str],
    num_samples: int,
    n_steps: int,
    sampling_seed: int,
    dataset_role: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Dataset-role evaluation CRN. Independent of LR, K, and before/after."""

    seeds = crn_seed_matrix_dataset(instance_ids, num_samples, sampling_seed, dataset_role)
    batch = np.empty((len(instance_ids), num_samples, n_steps), dtype=np.float64)
    for i in range(len(instance_ids)):
        for sample_id in range(num_samples):
            rng = np.random.default_rng(int(seeds[i, sample_id]))
            batch[i, sample_id] = rng.random(n_steps)
    return batch, seeds


@torch.no_grad()
def greedy_tour_costs(
    model: Module,
    coords: np.ndarray,
    device: torch.device,
    *,
    knns: int = KNNS,
    mask_destination: bool = False,
) -> tuple[np.ndarray, list[list[int]]]:
    model.eval()
    xy = np.ascontiguousarray(coords, dtype=np.float64)
    closed = np.ascontiguousarray(closed_inference_coordinates(xy), dtype=np.float32)
    node_coords = torch.from_numpy(closed).to(device)
    dist_matrices = torch.cdist(node_coords, node_coords)
    tours = greedy_decode(node_coords, dist_matrices, model, knns=knns, mask_destination=mask_destination)
    tours_np = tours.detach().cpu().numpy()
    n_cities = xy.shape[1]
    costs = np.empty(len(xy), dtype=np.int64)
    permutations: list[list[int]] = []
    for index, raw in enumerate(tours_np):
        tour = decode_tour_to_permutation(raw, n_cities)
        if tour[0] != START_NODE:
            raise RuntimeError("greedy tour does not start at node 0")
        costs[index] = integer_tour_cost(xy[index], tour, INTEGER_SCALE)
        permutations.append(tour)
    return costs, permutations


@torch.no_grad()
def sample_mean_tour_costs(
    model: Module,
    coords: np.ndarray,
    instance_ids: list[str],
    device: torch.device,
    *,
    num_samples: int,
    temperature: float,
    sampling_seed: int,
    shuffle_seed: int | None = None,
    uniforms: np.ndarray | None = None,
    crn_seeds: np.ndarray | None = None,
    sample_chunk_size: int | None = None,
    knns: int = KNNS,
    mask_destination: bool = False,
) -> dict[str, Any]:
    """Monte Carlo mean tour cost. Never returns min/best-of-S as the estimate.

    One B×S batched sampling decode by default. ``sample_chunk_size`` splits the
    sample axis only when set and smaller than S; default is a single launch.
    """

    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    model.eval()
    xy = np.ascontiguousarray(coords, dtype=np.float64)
    n_instances, n_cities, _ = xy.shape
    if len(instance_ids) != n_instances:
        raise ValueError("instance_ids length must match coords")
    closed = np.ascontiguousarray(closed_inference_coordinates(xy), dtype=np.float32)
    n_steps = int(closed.shape[1] - 2)
    if uniforms is None:
        if shuffle_seed is None:
            uniforms, generated_seeds = crn_uniforms_eval(
                instance_ids, num_samples, n_steps, sampling_seed
            )
        else:
            uniforms, generated_seeds = crn_uniforms(
                instance_ids, num_samples, n_steps, sampling_seed, int(shuffle_seed)
            )
        if crn_seeds is None:
            crn_seeds = generated_seeds
    elif crn_seeds is None:
        if shuffle_seed is None:
            crn_seeds = crn_seed_matrix_eval(instance_ids, num_samples, sampling_seed)
        else:
            crn_seeds = crn_seed_matrix(instance_ids, num_samples, sampling_seed, int(shuffle_seed))
    if uniforms.shape != (n_instances, num_samples, n_steps):
        raise RuntimeError(f"CRN uniforms shape {uniforms.shape} != {(n_instances, num_samples, n_steps)}")
    if crn_seeds.shape != (n_instances, num_samples):
        raise RuntimeError(f"CRN seeds shape {crn_seeds.shape} != {(n_instances, num_samples)}")

    chunk = int(num_samples) if sample_chunk_size is None else int(sample_chunk_size)
    if chunk < 1:
        raise ValueError("sample_chunk_size must be positive when set")
    chunk = min(chunk, int(num_samples))

    tours_np = np.empty((n_instances, num_samples, n_cities), dtype=np.int64)
    n_launches = 0
    for sample_start in range(0, num_samples, chunk):
        sample_end = min(sample_start + chunk, num_samples)
        width = sample_end - sample_start
        closed_rep = np.repeat(closed, width, axis=0)
        uniforms_flat = np.ascontiguousarray(
            uniforms[:, sample_start:sample_end, :].reshape(n_instances * width, n_steps),
            dtype=np.float32,
        )
        node_coords = torch.from_numpy(closed_rep).to(device)
        dist_matrices = torch.cdist(node_coords, node_coords)
        u_tensor = torch.from_numpy(uniforms_flat).to(device)
        tours = sampling_decode(
            node_coords,
            dist_matrices,
            model,
            u_tensor,
            temperature,
            knns=knns,
            mask_destination=mask_destination,
        )
        n_launches += 1
        tours_np[:, sample_start:sample_end, :] = tours.detach().cpu().numpy().reshape(
            n_instances, width, n_cities
        )

    sample_costs = np.empty((n_instances, num_samples), dtype=np.float64)
    n_unique = np.empty(n_instances, dtype=np.int32)
    for i in range(n_instances):
        keys = []
        for s in range(num_samples):
            tour = decode_tour_to_permutation(tours_np[i, s], n_cities)
            if tour[0] != START_NODE:
                raise RuntimeError("sampled tour does not start at node 0")
            sample_costs[i, s] = float(integer_tour_cost(xy[i], tour, INTEGER_SCALE))
            keys.append(tuple(tour))
        n_unique[i] = len(set(keys))

    mean_cost = sample_costs.mean(axis=1)
    return {
        "mean_cost": mean_cost,
        "std_cost": sample_costs.std(axis=1, ddof=1) if num_samples > 1 else np.zeros(n_instances),
        "min_cost": sample_costs.min(axis=1),
        "max_cost": sample_costs.max(axis=1),
        "n_unique_tours": n_unique,
        "sample_costs": sample_costs,
        "uniforms": uniforms,
        "crn_seeds": crn_seeds,
        "num_samples": int(num_samples),
        "temperature": float(temperature),
        "estimator": "mean_of_S_stochastic_rollouts",
        "sampling_decode_launches": int(n_launches),
        "instance_rollouts": int(n_instances * num_samples),
        "replicated_batch_size": int(n_instances * chunk) if chunk == num_samples else int(n_instances * chunk),
        "sample_chunk_size": None if sample_chunk_size is None else int(sample_chunk_size),
        "crn_seed_fields": list(CRN_SEED_FIELDS_EVAL if shuffle_seed is None else CRN_SEED_FIELDS),
    }
