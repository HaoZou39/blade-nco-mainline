"""Teacher-forced, per-instance BQ imitation-hardness diagnostics.

Unlike greedy decoding diagnostics, this module presents the model with the
ground-truth expert prefix at every step.  For the BQ suffix representation,
that means evaluating every expert-ordered suffix and using index 1 as the
expert next-action target.  No model parameters are updated.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from .tsp_metric import expert_closed_coordinates


MIN_DECISION_SUFFIX = 3


def closed_expert_coordinates(coordinates: np.ndarray, tours: np.ndarray) -> np.ndarray:
    xy = np.asarray(coordinates, dtype=np.float64)
    expert = np.asarray(tours, dtype=np.int64)
    if xy.ndim != 3 or xy.shape[-1] != 2:
        raise ValueError(f"expected coordinates (B,N,2), got {xy.shape}")
    if expert.shape != xy.shape[:2]:
        raise ValueError(f"tour shape {expert.shape} does not match coordinates {xy.shape[:2]}")
    rows = [expert_closed_coordinates(xy[index], expert[index]) for index in range(len(xy))]
    return np.ascontiguousarray(np.stack(rows, axis=0), dtype=np.float32)


@torch.no_grad()
def teacher_forced_step_diagnostics(
    model: nn.Module,
    coordinates: np.ndarray,
    tours: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    """Evaluate expert-action diagnostics at every teacher-forced decision.

    Each suffix begins with the current expert node, has the expert next node at
    index 1, and ends with the fixed destination token.  Successive suffixes
    therefore encode ground-truth previous actions rather than model rollouts.

    Returned arrays have shape ``(B, N - 1)`` and use chronological decoding
    order.  This is the canonical trace path for audits that need to reconstruct
    the stored instance mean without changing the original state convention.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    closed = closed_expert_coordinates(coordinates, tours)
    n_instances, closed_length, _ = closed.shape
    n_nodes = closed_length - 1
    if n_nodes < 2:
        raise ValueError("TSP instances must have at least two nodes")

    n_steps = n_nodes - 1
    step_ce = np.empty((n_instances, n_steps), dtype=np.float64)
    step_entropy = np.empty_like(step_ce)
    expert_probability = np.empty_like(step_ce)
    step_margin = np.empty_like(step_ce)
    remaining_candidate_count = np.empty(n_steps, dtype=np.int64)

    model.eval()
    # closed_length -> 3 covers all N-1 next-city decisions.  The final suffix
    # has one feasible action, so its entropy is zero and margin is one.
    for step_index, suffix_length in enumerate(
        range(closed_length, MIN_DECISION_SUFFIX - 1, -1)
    ):
        suffix = np.ascontiguousarray(closed[:, closed_length - suffix_length :], dtype=np.float32)
        remaining_candidate_count[step_index] = suffix_length - 2
        for start in range(0, n_instances, batch_size):
            end = min(start + batch_size, n_instances)
            inputs = torch.from_numpy(suffix[start:end]).to(device)
            scores = model(inputs)
            if scores.ndim != 2 or scores.shape[0] != end - start or scores.shape[1] != suffix_length:
                raise RuntimeError(f"unexpected BQ score shape {tuple(scores.shape)} for suffix {suffix_length}")
            feasible = torch.isfinite(scores)
            if not bool(feasible[:, 1].all()):
                raise RuntimeError("expert next action at index 1 is not feasible")
            log_normalizer = torch.logsumexp(scores, dim=1, keepdim=True)
            log_probabilities = scores - log_normalizer
            probabilities = torch.where(feasible, torch.exp(log_probabilities), torch.zeros_like(scores))
            losses = -log_probabilities[:, 1]
            entropy = -torch.where(
                feasible, probabilities * log_probabilities, torch.zeros_like(probabilities)
            ).sum(dim=1)
            top = torch.topk(probabilities, k=min(2, suffix_length), dim=1).values
            second = top[:, 1] if top.shape[1] > 1 else torch.zeros_like(top[:, 0])
            margin = top[:, 0] - second
            arrays = [losses, entropy, probabilities[:, 1], margin]
            if not all(bool(torch.isfinite(value).all()) for value in arrays):
                raise RuntimeError("non-finite teacher-forced BQ diagnostic")
            step_ce[start:end, step_index] = losses.cpu().numpy().astype(np.float64)
            step_entropy[start:end, step_index] = entropy.cpu().numpy().astype(np.float64)
            expert_probability[start:end, step_index] = probabilities[:, 1].cpu().numpy().astype(np.float64)
            step_margin[start:end, step_index] = margin.cpu().numpy().astype(np.float64)

    result = {
        "step_ce": step_ce,
        "expert_prob": expert_probability,
        "step_entropy": step_entropy,
        "step_margin": step_margin,
        "remaining_candidate_count": remaining_candidate_count,
    }
    if not all(np.isfinite(value).all() for value in result.values()):
        raise RuntimeError("non-finite teacher-forced BQ step diagnostic")
    if np.any(step_ce < 0) or np.any((expert_probability < 0) | (expert_probability > 1)):
        raise RuntimeError("teacher-forced BQ hardness violates probability bounds")
    return result


def teacher_forced_training_hardness(
    model: nn.Module,
    coordinates: np.ndarray,
    tours: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    """Evaluate mean expert-action NLL and uncertainty over all decisions."""

    trace = teacher_forced_step_diagnostics(
        model, coordinates, tours, device, batch_size=batch_size
    )
    n_instances, num_steps = trace["step_ce"].shape
    loss_raw = trace["step_ce"].mean(axis=1)
    result = {
        "N": np.full(n_instances, num_steps + 1, dtype=np.int64),
        "loss_raw": loss_raw,
        "loss_log": np.log1p(loss_raw),
        "num_steps": np.full(n_instances, num_steps, dtype=np.int64),
        "mean_entropy": trace["step_entropy"].mean(axis=1),
        # Per task definition, this is the probability assigned to the expert
        # action, even though the requested output column is named top1.
        "mean_top1_prob": trace["expert_prob"].mean(axis=1),
        "mean_margin": trace["step_margin"].mean(axis=1),
    }
    if not all(np.isfinite(value).all() for value in result.values()):
        raise RuntimeError("non-finite aggregate teacher-forced BQ hardness")
    return result


def protocol_manifest() -> dict[str, Any]:
    return {
        "target": "mean negative log probability of expert next action",
        "formula": "-(1/T) * sum_t log p_theta(a_t^* | s_t^*)",
        "conditioning": "teacher_forcing_ground_truth_previous_actions",
        "representation": "expert_ordered_closed_tour_suffix",
        "target_index": 1,
        "minimum_suffix_length": MIN_DECISION_SUFFIX,
        "greedy_rollout_used": False,
        "parameter_updates": 0,
    }
