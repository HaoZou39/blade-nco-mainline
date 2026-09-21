"""Read-only diagnostics for the canonical BQ greedy forward pass.

The decoder observer and encoder hook in this module only retain tensors that
the unmodified greedy path already computes. They never alter logits, masks,
actions, costs, parameters, or model mode.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .decoding import greedy_decode
from .inference import closed_inference_coordinates
from .protocol import INTEGER_SCALE, KNNS, START_NODE
from .tsp_metric import decode_tour_to_permutation, integer_tour_cost


DECODER_SCALAR_FEATURES = (
    "entropy_mean",
    "entropy_std",
    "entropy_p90",
    "entropy_max",
    "normalized_entropy_mean",
    "normalized_entropy_p90",
    "top1_prob_mean",
    "top1_prob_p10",
    "top1_prob_min",
    "prob_margin_mean",
    "prob_margin_p10",
    "prob_margin_min",
    "logit_margin_mean",
    "logit_margin_p10",
    "logit_margin_min",
)

EMBEDDING_SCALAR_FEATURES = (
    "embedding_node_norm_mean",
    "embedding_node_norm_std",
    "embedding_feature_variance_mean",
)


def _aggregate(values: np.ndarray, operation: str) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        if operation == "mean":
            return np.nanmean(values, axis=1)
        if operation == "std":
            return np.nanstd(values, axis=1, ddof=0)
        if operation == "p90":
            return np.nanpercentile(values, 90, axis=1)
        if operation == "p10":
            return np.nanpercentile(values, 10, axis=1)
        if operation == "max":
            return np.nanmax(values, axis=1)
        if operation == "min":
            return np.nanmin(values, axis=1)
    raise ValueError(f"unsupported aggregate {operation!r}")


@torch.no_grad()
def instrumented_greedy_tour_costs(
    model: nn.Module,
    coords: np.ndarray,
    device: torch.device,
    *,
    knns: int = KNNS,
    mask_destination: bool = False,
) -> dict[str, Any]:
    """Run canonical greedy decoding and retain cheap forward-only diagnostics."""

    model.eval()
    xy = np.ascontiguousarray(coords, dtype=np.float64)
    if xy.ndim != 3 or xy.shape[-1] != 2:
        raise ValueError(f"expected coords (B,N,2), got {xy.shape}")
    closed = np.ascontiguousarray(closed_inference_coordinates(xy), dtype=np.float32)
    node_coords = torch.from_numpy(closed).to(device)
    dist_matrices = torch.cdist(node_coords, node_coords)

    encoder_state: Tensor | None = None
    step_entropy: list[Tensor] = []
    step_normalized_entropy: list[Tensor] = []
    step_top1_probability: list[Tensor] = []
    step_probability_margin: list[Tensor] = []
    step_logit_margin: list[Tensor] = []
    feasible_action_counts: list[Tensor] = []

    def capture_encoder(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        nonlocal encoder_state
        if encoder_state is None:
            encoder_state = output.detach().clone()

    def observe_scores(scores: Tensor, _original_indices: Tensor) -> None:
        feasible = torch.isfinite(scores)
        counts = feasible.sum(dim=1)
        masked = torch.where(feasible, scores, torch.full_like(scores, -torch.inf))
        log_normalizer = torch.logsumexp(masked, dim=1, keepdim=True)
        log_probabilities = masked - log_normalizer
        probabilities = torch.where(feasible, torch.exp(log_probabilities), torch.zeros_like(scores))
        entropy = -torch.where(feasible, probabilities * log_probabilities, torch.zeros_like(scores)).sum(dim=1)
        normalized = torch.where(
            counts > 1,
            entropy / torch.log(counts.to(dtype=entropy.dtype)),
            torch.full_like(entropy, torch.nan),
        )
        top_probabilities = torch.topk(probabilities, k=min(2, probabilities.shape[1]), dim=1).values
        top_logits = torch.topk(masked, k=min(2, masked.shape[1]), dim=1).values
        top1 = top_probabilities[:, 0]
        if top_probabilities.shape[1] == 1:
            probability_margin = torch.full_like(top1, torch.nan)
            logit_margin = torch.full_like(top1, torch.nan)
        else:
            probability_margin = torch.where(
                counts > 1, top_probabilities[:, 0] - top_probabilities[:, 1], torch.full_like(top1, torch.nan)
            )
            logit_margin = torch.where(
                counts > 1, top_logits[:, 0] - top_logits[:, 1], torch.full_like(top1, torch.nan)
            )
        step_entropy.append(entropy)
        step_normalized_entropy.append(normalized)
        step_top1_probability.append(top1)
        step_probability_margin.append(probability_margin)
        step_logit_margin.append(logit_margin)
        feasible_action_counts.append(counts)

    if not hasattr(model, "encoder") or len(model.encoder) == 0:
        raise RuntimeError("BQ model has no encoder layers to instrument")
    hook = model.encoder[-1].register_forward_hook(capture_encoder)
    try:
        tours = greedy_decode(
            node_coords,
            dist_matrices,
            model,
            knns=knns,
            mask_destination=mask_destination,
            score_observer=observe_scores,
        )
    finally:
        hook.remove()
    if encoder_state is None:
        raise RuntimeError("encoder hook did not capture the initial final-layer state")
    if not step_entropy:
        raise RuntimeError("decoder observer captured no greedy steps")

    arrays = {
        "entropy": torch.stack(step_entropy, dim=1).cpu().numpy().astype(np.float64),
        "normalized_entropy": torch.stack(step_normalized_entropy, dim=1).cpu().numpy().astype(np.float64),
        "top1_prob": torch.stack(step_top1_probability, dim=1).cpu().numpy().astype(np.float64),
        "prob_margin": torch.stack(step_probability_margin, dim=1).cpu().numpy().astype(np.float64),
        "logit_margin": torch.stack(step_logit_margin, dim=1).cpu().numpy().astype(np.float64),
        "feasible_action_count": torch.stack(feasible_action_counts, dim=1).cpu().numpy().astype(np.int32),
    }
    scalar_features = {
        "entropy_mean": _aggregate(arrays["entropy"], "mean"),
        "entropy_std": _aggregate(arrays["entropy"], "std"),
        "entropy_p90": _aggregate(arrays["entropy"], "p90"),
        "entropy_max": _aggregate(arrays["entropy"], "max"),
        "normalized_entropy_mean": _aggregate(arrays["normalized_entropy"], "mean"),
        "normalized_entropy_p90": _aggregate(arrays["normalized_entropy"], "p90"),
        "top1_prob_mean": _aggregate(arrays["top1_prob"], "mean"),
        "top1_prob_p10": _aggregate(arrays["top1_prob"], "p10"),
        "top1_prob_min": _aggregate(arrays["top1_prob"], "min"),
        "prob_margin_mean": _aggregate(arrays["prob_margin"], "mean"),
        "prob_margin_p10": _aggregate(arrays["prob_margin"], "p10"),
        "prob_margin_min": _aggregate(arrays["prob_margin"], "min"),
        "logit_margin_mean": _aggregate(arrays["logit_margin"], "mean"),
        "logit_margin_p10": _aggregate(arrays["logit_margin"], "p10"),
        "logit_margin_min": _aggregate(arrays["logit_margin"], "min"),
    }
    for name in DECODER_SCALAR_FEATURES:
        scalar_features[f"{name}_missing"] = ~np.isfinite(scalar_features[name])

    pooled = encoder_state.mean(dim=1)
    node_norms = torch.linalg.vector_norm(encoder_state, dim=2)
    scalar_features["embedding_node_norm_mean"] = node_norms.mean(dim=1).cpu().numpy().astype(np.float64)
    scalar_features["embedding_node_norm_std"] = node_norms.std(dim=1, unbiased=False).cpu().numpy().astype(np.float64)
    scalar_features["embedding_feature_variance_mean"] = (
        encoder_state.var(dim=1, unbiased=False).mean(dim=1).cpu().numpy().astype(np.float64)
    )
    for name in EMBEDDING_SCALAR_FEATURES:
        scalar_features[f"{name}_missing"] = ~np.isfinite(scalar_features[name])

    tours_np = tours.detach().cpu().numpy()
    costs = np.empty(len(xy), dtype=np.int64)
    permutations: list[list[int]] = []
    for index, raw in enumerate(tours_np):
        tour = decode_tour_to_permutation(raw, xy.shape[1])
        if tour[0] != START_NODE:
            raise RuntimeError("instrumented greedy tour does not start at node 0")
        costs[index] = integer_tour_cost(xy[index], tour, INTEGER_SCALE)
        permutations.append(tour)

    return {
        "costs": costs,
        "tours": permutations,
        "selected_action_sequences": tours_np.astype(np.int64, copy=False),
        "scalar_features": scalar_features,
        "pooled_encoder_embedding": pooled.cpu().numpy().astype(np.float32),
        "encoder_embedding_dim": int(pooled.shape[1]),
        "encoder_node_count": int(encoder_state.shape[1]),
        "decoder_steps": int(arrays["entropy"].shape[1]),
        "feasible_action_counts": arrays["feasible_action_count"],
    }
