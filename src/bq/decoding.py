"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

Migrated greedy TSP decoding. The hardness pilot uses beam_size=1 and knns=-1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module


@dataclass
class DecodingSubPb:
    node_coords: Tensor
    original_idxs: Tensor
    dist_matrices: Tensor


def greedy_decode(
    node_coords: Tensor,
    dist_matrices: Tensor,
    net: Module,
    knns: int = -1,
    mask_destination: bool = False,
    score_observer: Callable[[Tensor, Tensor], None] | None = None,
) -> Tensor:
    tours = greedy_decoding_loop(
        node_coords,
        dist_matrices,
        net,
        knns,
        mask_destination=mask_destination,
        score_observer=score_observer,
    )
    tours = tours[:, :-1]
    num_nodes = tours.shape[1]
    expected = int(tours.shape[0] * (num_nodes - 1) * num_nodes // 2)
    if int(tours.sum().item()) != expected:
        raise RuntimeError("greedy decode did not return a permutation of remaining nodes")
    return tours


def greedy_decoding_loop(
    node_coords: Tensor,
    dist_matrices: Tensor,
    net: Module,
    knns: int,
    mask_destination: bool = False,
    score_observer: Callable[[Tensor, Tensor], None] | None = None,
) -> Tensor:
    num_nodes = node_coords.shape[1]
    bs = node_coords.shape[0]
    original_idxs = torch.tensor(list(range(num_nodes)), device=node_coords.device)[None, :].repeat(bs, 1)
    paths = torch.zeros((bs, num_nodes), dtype=torch.long, device=node_coords.device)
    paths[:, -1] = num_nodes - 1
    sub_problem = DecodingSubPb(node_coords, original_idxs, dist_matrices)
    for dec_pos in range(1, num_nodes - 1):
        idx_selected, sub_problem = greedy_decoding_step(
            sub_problem,
            net,
            knns,
            mask_destination=mask_destination,
            score_observer=score_observer,
        )
        paths[:, dec_pos] = idx_selected
    return paths


def prepare_input_and_forward_pass(sub_problem: DecodingSubPb, net: Module, knns: int) -> Tensor:
    bs, num_nodes, node_dim = sub_problem.node_coords.shape
    if 0 < knns < num_nodes:
        _, sorted_nodes_idx = torch.sort(sub_problem.dist_matrices[:, :-1, 0], dim=-1)
        knn_indices = sorted_nodes_idx[:, : knns - 1]
        input_nodes = torch.cat(
            [knn_indices, torch.full([bs, 1], num_nodes - 1, device=knn_indices.device)], dim=-1
        )
        node_coords = torch.gather(
            sub_problem.node_coords, 1, input_nodes.unsqueeze(dim=-1).repeat(1, 1, node_dim)
        )
        knn_scores = net(node_coords)
        scores = torch.full(sub_problem.node_coords.shape[:-1], -np.inf, device=node_coords.device)
        scores = torch.scatter(scores, 1, knn_indices, knn_scores)
    else:
        scores = net(sub_problem.node_coords)
    return scores


def greedy_decoding_step(
    sub_problem: DecodingSubPb,
    net: Module,
    knns: int,
    mask_destination: bool = False,
    score_observer: Callable[[Tensor, Tensor], None] | None = None,
) -> tuple[Tensor, DecodingSubPb]:
    scores = prepare_input_and_forward_pass(sub_problem, net, knns)
    observed_scores = scores
    scores = torch.nan_to_num(scores, nan=torch.finfo(scores.dtype).min, posinf=torch.finfo(scores.dtype).max, neginf=torch.finfo(scores.dtype).min)
    if mask_destination and scores.shape[1] > 2:
        scores = scores.clone()
        scores[:, -1] = torch.finfo(scores.dtype).min
    if score_observer is not None:
        if mask_destination and observed_scores.shape[1] > 2:
            observed_scores = observed_scores.clone()
            observed_scores[:, -1] = -torch.inf
        score_observer(observed_scores.detach(), sub_problem.original_idxs.detach())
    idx_selected = torch.argmax(scores, dim=1, keepdim=True)
    idx_selected_original = torch.gather(sub_problem.original_idxs, 1, idx_selected)
    return idx_selected_original.squeeze(1), reformat_subproblem_for_next_step(sub_problem, idx_selected, knns)


def inverse_cdf_indices(scores: Tensor, uniforms: Tensor, temperature: float) -> Tensor:
    """Sample one index per row from softmax(scores / T) using CRN uniforms in (0, 1)."""

    if temperature <= 0:
        raise ValueError("sampling temperature must be positive")
    scores = torch.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
    logits = scores / float(temperature)
    logits = logits - logits.max(dim=-1, keepdim=True).values
    probs = torch.softmax(logits, dim=-1)
    probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
    denom = probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    probs = probs / denom
    cdf = torch.cumsum(probs, dim=-1)
    cdf[:, -1] = 1.0
    u = uniforms.to(dtype=probs.dtype, device=probs.device).clamp(1e-12, 1.0 - 1e-12).unsqueeze(-1)
    idx = torch.searchsorted(cdf, u).squeeze(-1).clamp(0, scores.shape[1] - 1)
    return idx.unsqueeze(1)


def sampling_decode(
    node_coords: Tensor,
    dist_matrices: Tensor,
    net: Module,
    uniforms: Tensor,
    temperature: float,
    knns: int = -1,
    mask_destination: bool = False,
) -> Tensor:
    """Stochastic BQ decode. ``uniforms`` has shape (batch, n_decode_steps)."""

    tours = sampling_decoding_loop(
        node_coords, dist_matrices, net, uniforms, temperature, knns, mask_destination=mask_destination
    )
    tours = tours[:, :-1]
    num_nodes = tours.shape[1]
    expected = int(tours.shape[0] * (num_nodes - 1) * num_nodes // 2)
    if int(tours.sum().item()) != expected:
        raise RuntimeError("sampling decode did not return a permutation of remaining nodes")
    return tours


def sampling_decoding_loop(
    node_coords: Tensor,
    dist_matrices: Tensor,
    net: Module,
    uniforms: Tensor,
    temperature: float,
    knns: int,
    mask_destination: bool = False,
) -> Tensor:
    num_nodes = node_coords.shape[1]
    bs = node_coords.shape[0]
    n_steps = num_nodes - 2
    if uniforms.shape != (bs, n_steps):
        raise RuntimeError(f"uniforms shape {tuple(uniforms.shape)} != ({bs}, {n_steps})")
    original_idxs = torch.tensor(list(range(num_nodes)), device=node_coords.device)[None, :].repeat(bs, 1)
    paths = torch.zeros((bs, num_nodes), dtype=torch.long, device=node_coords.device)
    paths[:, -1] = num_nodes - 1
    sub_problem = DecodingSubPb(node_coords, original_idxs, dist_matrices)
    for step, dec_pos in enumerate(range(1, num_nodes - 1)):
        scores = prepare_input_and_forward_pass(sub_problem, net, knns)
        remaining = scores.shape[1]
        if mask_destination and remaining > 2:
            idx_selected = inverse_cdf_indices(scores[:, :-1], uniforms[:, step], temperature)
        else:
            idx_selected = inverse_cdf_indices(scores, uniforms[:, step], temperature)
        idx_selected_original = torch.gather(sub_problem.original_idxs, 1, idx_selected)
        paths[:, dec_pos] = idx_selected_original.squeeze(1)
        sub_problem = reformat_subproblem_for_next_step(sub_problem, idx_selected, knns)
    return paths


def reformat_subproblem_for_next_step(
    sub_problem: DecodingSubPb, idx_selected: Tensor, knns: int
) -> DecodingSubPb:
    subpb_size = sub_problem.node_coords.shape[1]
    bs = sub_problem.node_coords.shape[0]
    is_selected = (
        torch.arange(subpb_size, device=sub_problem.node_coords.device).unsqueeze(dim=0).repeat(bs, 1)
        == idx_selected.repeat(1, subpb_size)
    )
    next_begin_node_coord = sub_problem.node_coords[is_selected].unsqueeze(dim=1)
    next_begin_original_idx = sub_problem.original_idxs[is_selected].unsqueeze(dim=1)
    next_remaining_node_coords = sub_problem.node_coords[~is_selected].reshape((bs, -1, 2))[:, 1:, :]
    next_remaining_original_idxs = sub_problem.original_idxs[~is_selected].reshape((bs, -1))[:, 1:]
    next_node_coords = torch.cat([next_begin_node_coord, next_remaining_node_coords], dim=1)
    next_original_idxs = torch.cat([next_begin_original_idx, next_remaining_original_idxs], dim=1)
    if knns != -1:
        num_nodes = sub_problem.dist_matrices.shape[1]
        next_row_column = sub_problem.dist_matrices[is_selected]
        next_row_column = next_row_column[~is_selected].reshape((bs, -1))[:, 1:]
        next_dist_matrices = sub_problem.dist_matrices[~is_selected].reshape(bs, -1, num_nodes)[:, 1:, :]
        next_dist_matrices = next_dist_matrices.transpose(1, 2)[~is_selected].reshape(bs, num_nodes - 1, -1)[:, 1:, :]
        next_dist_matrices = torch.cat([next_row_column.unsqueeze(dim=1), next_dist_matrices], dim=1)
        next_row_column = torch.cat(
            [torch.zeros(idx_selected.shape, device=next_row_column.device), next_row_column], dim=1
        )
        next_dist_matrices = torch.cat([next_row_column.unsqueeze(dim=2), next_dist_matrices], dim=2)
    else:
        next_dist_matrices = sub_problem.dist_matrices
    return DecodingSubPb(next_node_coords, next_original_idxs, next_dist_matrices)
