"""Supervised BQ short-adaptation step matching the validated v2 protocol."""

from __future__ import annotations

from typing import Any
import time

import numpy as np
import torch
from torch import nn

from .protocol import (
    GRADIENT_CLIPPING,
    LOSS_NAME,
    MICROBATCH_SIZE,
    OPTIMIZER_TYPE,
    SUFFIX_LENGTH_MIN,
    WEIGHT_DECAY,
)


def create_optimizer(model: nn.Module, learning_rate: float) -> torch.optim.Optimizer:
    if OPTIMIZER_TYPE != "Adam":
        raise RuntimeError(f"unsupported optimizer {OPTIMIZER_TYPE}")
    if GRADIENT_CLIPPING is not None:
        raise RuntimeError("validated BQ protocol does not clip gradients")
    return torch.optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=float(WEIGHT_DECAY))


def sample_suffix_length(closed_len: int, rng: np.random.Generator) -> int:
    # Official collate: np.random.randint(4, nb_nodes + 1) with nb_nodes = N+1.
    return int(rng.integers(SUFFIX_LENGTH_MIN, closed_len + 1))


def suffix_batch(closed_coords: np.ndarray, suffix_length: int) -> np.ndarray:
    if closed_coords.ndim != 3:
        raise ValueError(f"expected closed coords (B,N+1,2), got {closed_coords.shape}")
    closed_len = closed_coords.shape[1]
    if not SUFFIX_LENGTH_MIN <= suffix_length <= closed_len:
        raise ValueError(f"suffix length {suffix_length} outside [{SUFFIX_LENGTH_MIN}, {closed_len}]")
    begin = closed_len - suffix_length
    return np.ascontiguousarray(closed_coords[:, begin:], dtype=np.float32)


def _gradient_l2(model: nn.Module) -> float:
    total = torch.zeros((), dtype=torch.float64, device=next(model.parameters()).device)
    for param in model.parameters():
        if param.grad is not None:
            total = total + param.grad.detach().double().pow(2).sum()
    return float(torch.sqrt(total).item())


def supervised_optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    closed_coords: np.ndarray,
    suffix_length: int,
    device: torch.device,
    *,
    group_split: int | None = None,
) -> dict[str, Any]:
    """One Adam update on a batch of expert-ordered closed tours.

    Loss, target index, suffix sampler, micro-batch accumulation, and absence of
    gradient clipping follow ``finetune_worker_v2.py``.
    """

    if LOSS_NAME != "torch.nn.CrossEntropyLoss":
        raise RuntimeError(f"unexpected loss {LOSS_NAME}")
    batch = suffix_batch(closed_coords, suffix_length)
    n_instances = int(batch.shape[0])
    micro = min(int(MICROBATCH_SIZE), n_instances)
    n_micro = int(np.ceil(n_instances / micro))
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    raw_losses: list[float] = []
    group_losses: list[tuple[str, float]] = []
    host_to_device_wall_s = 0.0
    forward_and_loss_wall_s = 0.0
    backward_wall_s = 0.0
    optimizer_step_wall_s = 0.0
    for offset in range(0, n_instances, micro):
        started = time.perf_counter()
        chunk = torch.from_numpy(batch[offset : offset + micro]).to(device)
        target = torch.ones(len(chunk), dtype=torch.long, device=device)
        host_to_device_wall_s += time.perf_counter() - started
        started = time.perf_counter()
        loss = loss_fn(model(chunk), target)
        forward_and_loss_wall_s += time.perf_counter() - started
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite supervised BQ loss")
        started = time.perf_counter()
        (loss / n_micro).backward()
        backward_wall_s += time.perf_counter() - started
        raw_losses.append(float(loss.item()))
        if group_split is not None:
            if offset < group_split < offset + len(chunk):
                raise ValueError("group_split must align with a microbatch boundary")
            group_losses.append(("candidate" if offset < group_split else "anchor", float(loss.item())))
    grad_norm = _gradient_l2(model)
    started = time.perf_counter()
    optimizer.step()
    optimizer_step_wall_s += time.perf_counter() - started
    result = {
        "training_loss": float(np.mean(raw_losses)),
        "gradient_norm": grad_norm,
        "suffix_length": int(suffix_length),
        "microbatch_size": int(micro),
        "n_microbatches": int(n_micro),
        "n_instances": n_instances,
        "optimizer_updates": 1,
        "host_to_device_transfer_wall_s": host_to_device_wall_s,
        "forward_and_loss_wall_s": forward_and_loss_wall_s,
        "backward_wall_s": backward_wall_s,
        "optimizer_step_wall_s": optimizer_step_wall_s,
        "timing_type": "CPU_WALL_DIAGNOSTIC_NON_ADDITIVE",
    }
    if group_split is not None:
        if not 0 < group_split < n_instances:
            raise ValueError("group_split outside batch")
        for name in ("candidate", "anchor"):
            values = [value for group, value in group_losses if group == name]
            if not values:
                raise RuntimeError(f"missing detached {name} loss")
            result[f"{name}_loss"] = float(np.mean(values))
    return result
