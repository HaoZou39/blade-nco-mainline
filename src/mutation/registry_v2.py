"""Opt-in Unified-12 Structural V2; default registry remains frozen V1."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import numpy as np
from . import registry as v1_registry
from . import stripe_reorganization, motif_assembly_v3
from .parameters import load_operator_config
from .result import MutationResult

PROFILE_NAME = "UNIFIED12_STRUCTURAL_V2"
BASE_10 = v1_registry.OPERATOR_NAMES[:10]
OPERATOR_NAMES = BASE_10 + ("stripe_reorganization", "motif_assembly_v3")
PROFILE_FILE = Path(__file__).with_name("unified12_structural_v2.yaml")

def load_profile() -> dict[str, Any]:
    profile = load_operator_config(PROFILE_FILE)
    if profile["profile_name"] != PROFILE_NAME or set(profile["operators"]) != set(OPERATOR_NAMES):
        raise ValueError("Structural V2 manifest mismatch")
    return profile

def sample_operator_name(rng: np.random.Generator) -> str:
    return str(rng.choice(np.asarray(OPERATOR_NAMES)))

def apply_operator(name: str, coordinates: np.ndarray, rng: np.random.Generator,
                   params: Mapping[str, Any] | None = None) -> MutationResult:
    if name not in OPERATOR_NAMES:
        raise v1_registry.UnknownOperatorError(f"not a Structural V2 operator: {name}")
    if name == "stripe_reorganization":
        return stripe_reorganization.apply_operator(coordinates, rng, params)
    if name == "motif_assembly_v3":
        return motif_assembly_v3.apply_operator(coordinates, rng, params)
    if params is not None:
        return v1_registry.apply_operator(name, coordinates, rng, params)
    return v1_registry.apply_operator_with_config(name, coordinates, rng, load_profile())
