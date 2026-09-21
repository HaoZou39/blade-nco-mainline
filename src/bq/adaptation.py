"""Frozen 1:1 hard/anchor BQ adaptation protocol.

This module intentionally has no early stopping, scheduler, AMP, checkpoint
selection, or intermediate scientific checkpoint support.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from .supervised import create_optimizer, sample_suffix_length, supervised_optimizer_step
from .training_hardness import closed_expert_coordinates


LEARNING_RATE = 7.5e-5
EPOCHS = 256
HARD_COUNT = 10_000
ANCHOR_COUNT = 10_000
BRANCH_BATCH_SIZE = 128


def fine_tune_fixed_budget(
    model: nn.Module,
    hard_coordinates: np.ndarray,
    hard_tours: np.ndarray,
    anchor_coordinates: np.ndarray,
    anchor_tours: np.ndarray,
    device: torch.device,
    *,
    seed: int = 2026,
    learning_rate: float = LEARNING_RATE,
    epoch_callback: Callable[[int, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run exactly 256 epochs with a fresh Adam and equal branch sampling."""

    if len(hard_coordinates) != HARD_COUNT or len(anchor_coordinates) != ANCHOR_COUNT:
        raise ValueError("frozen adaptation requires exactly 10,000 hard and 10,000 anchor instances")
    hard_closed = closed_expert_coordinates(hard_coordinates, hard_tours)
    anchor_closed = closed_expert_coordinates(anchor_coordinates, anchor_tours)
    if hard_closed.shape[1:] != anchor_closed.shape[1:]:
        raise ValueError("hard and anchor instances must have the same problem size")

    rng = np.random.default_rng(seed)
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    optimizer = create_optimizer(model, float(learning_rate))
    history: list[dict[str, Any]] = []
    for epoch in range(1, EPOCHS + 1):
        epoch_started = time.perf_counter()
        hard_order = rng.permutation(HARD_COUNT)
        anchor_order = rng.permutation(ANCHOR_COUNT)
        losses: list[float] = []
        updates = 0
        dataloader_wait_wall_s = 0.0
        timing = {
            "host_to_device_transfer_wall_s": 0.0,
            "forward_and_loss_wall_s": 0.0,
            "backward_wall_s": 0.0,
            "optimizer_step_wall_s": 0.0,
        }
        for start in range(0, HARD_COUNT, BRANCH_BATCH_SIZE):
            batch_started = time.perf_counter()
            h_idx = hard_order[start : start + BRANCH_BATCH_SIZE]
            a_idx = anchor_order[start : start + BRANCH_BATCH_SIZE]
            batch = np.concatenate([hard_closed[h_idx], anchor_closed[a_idx]], axis=0)
            suffix_length = sample_suffix_length(batch.shape[1], rng)
            dataloader_wait_wall_s += time.perf_counter() - batch_started
            result = supervised_optimizer_step(model, optimizer, batch, suffix_length, device)
            losses.append(float(result["training_loss"]))
            updates += 1
            for key in timing:
                timing[key] += float(result[key])
        record = {
            "epoch": epoch,
            "mean_training_loss": float(np.mean(losses)),
            "optimizer_updates": updates,
            "hard_instances_seen": HARD_COUNT,
            "anchor_instances_seen": ANCHOR_COUNT,
            "samples_processed": HARD_COUNT + ANCHOR_COUNT,
            "batches_processed": updates,
            "epoch_wall_s": time.perf_counter() - epoch_started,
            "dataloader_wait_wall_s": dataloader_wait_wall_s,
            **timing,
            "internal_timing_type": "NON_ADDITIVE_DIAGNOSTIC",
        }
        history.append(record)
        if epoch_callback is not None:
            epoch_callback(epoch, record)
    return history


def save_e256(model: nn.Module, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
    """Persist the sole formal scientific checkpoint for one adaptation cycle."""

    target = Path(path).resolve()
    runs = Path(__file__).resolve().parents[2] / "runs"
    if runs.resolve() not in target.parents:
        raise ValueError("generated checkpoints must be written under runs/")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"net": model.state_dict(), "epoch": EPOCHS, "formal_scientific_checkpoint": True}
    if metadata:
        payload["metadata"] = dict(metadata)
    torch.save(payload, target)
