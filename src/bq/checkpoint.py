"""Canonical BQ checkpoint loading, cloning, and byte-level integrity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .model import BQModel
from .protocol import (
    CANONICAL_M0_SHA256,
    CHECKPOINT_STATE_KEY,
    EXPECTED_PARAMETER_COUNT,
    FULL_PARAMETER_UPDATE,
    architecture_constructor_args,
)
from .utils import sha256_file


def build_bq_model(device: torch.device) -> BQModel:
    model = BQModel(*architecture_constructor_args()).to(device)
    n_params = sum(item.numel() for item in model.parameters())
    if n_params != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"BQ parameter count {n_params} != {EXPECTED_PARAMETER_COUNT}")
    if FULL_PARAMETER_UPDATE and not all(item.requires_grad for item in model.parameters()):
        raise RuntimeError("BQ protocol requires full-parameter training")
    return model


def load_canonical_checkpoint(
    path: Path,
    device: torch.device,
    *,
    expected_sha256: str | None = CANONICAL_M0_SHA256,
) -> tuple[BQModel, dict[str, Any]]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"BQ checkpoint not found: {path}")
    file_hash = sha256_file(path)
    if expected_sha256 is not None and file_hash != expected_sha256:
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch: got {file_hash}, expected canonical M0 {expected_sha256}"
        )
    blob = torch.load(path, map_location=device, weights_only=False)
    if CHECKPOINT_STATE_KEY not in blob:
        raise RuntimeError(f"checkpoint missing {CHECKPOINT_STATE_KEY!r}")
    model = build_bq_model(device)
    model.load_state_dict(blob[CHECKPOINT_STATE_KEY], strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(True)
    meta = {
        "path": str(path),
        "sha256": file_hash,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "load_protocol": "direct_strict_load_of_checkpoint['net']",
        "full_parameter": True,
    }
    return model, meta


def clone_model_from(source: nn.Module, device: torch.device) -> BQModel:
    clone = build_bq_model(device)
    clone.load_state_dict(source.state_dict(), strict=True)
    clone.train()
    for param in clone.parameters():
        param.requires_grad_(True)
    return clone


def state_dict_sha256(model: nn.Module) -> str:
    import hashlib

    digest = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
