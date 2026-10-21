"""Central loading and sampling of configurable operator parameters."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


NAMED_PROFILES = {
    "FOGA11_BASE_V1": "foga2019_operators.yaml",
    "FOGA11_HIGH_V1": "foga11_high_v1.yaml",
}

@lru_cache(maxsize=1)
def load_operator_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the single formal FOGA 2019 operator configuration."""

    config_path = (
        Path(path)
        if path is not None
        else Path(__file__).with_name("foga2019_operators.yaml")
    )
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("operators"), dict):
        raise ValueError(f"invalid operator configuration: {config_path}")
    return payload


def load_named_profile(name: str) -> dict[str, Any]:
    """Load a versioned production mutation profile by its frozen name."""

    try:
        filename = NAMED_PROFILES[name]
    except KeyError as exc:
        raise KeyError(f"unknown mutation profile {name!r}; expected one of {sorted(NAMED_PROFILES)}") from exc
    payload = load_operator_config(Path(__file__).with_name(filename))
    declared = payload.get("profile_name")
    if declared is not None and declared != name:
        raise ValueError(f"mutation profile identity mismatch: requested {name}, file declares {declared}")
    return payload


def resolve_operator_config(
    operator_name: str,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve one operator's configuration from a full or local mapping."""

    payload: Mapping[str, Any] = config or load_operator_config()
    if "operators" in payload:
        operators = payload["operators"]
        if operator_name not in operators:
            raise KeyError(f"missing configuration for {operator_name}")
        return deepcopy(dict(operators[operator_name]))
    return deepcopy(dict(payload))


def repair_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return project-level repair settings."""

    payload = config or load_operator_config()
    if "repair" not in payload:
        payload = load_operator_config()
    return deepcopy(dict(payload["repair"]))


def sample_scalar(spec: Mapping[str, Any], rng: np.random.Generator) -> float:
    """Sample one scalar from a declarative distribution."""

    distribution = str(spec["distribution"])
    if distribution == "uniform":
        return float(rng.uniform(float(spec["low"]), float(spec["high"])))
    if distribution == "log_uniform":
        low = float(spec["low"])
        high = float(spec["high"])
        return float(np.exp(rng.uniform(np.log(low), np.log(high))))
    if distribution == "choice":
        return float(rng.choice(np.asarray(spec["values"], dtype=float)))
    raise ValueError(f"unsupported scalar distribution: {distribution}")


def sample_integer(spec: Mapping[str, Any], rng: np.random.Generator) -> int:
    """Sample one inclusive integer or one discrete configured choice."""

    distribution = str(spec["distribution"])
    if distribution == "integer_uniform":
        return int(rng.integers(int(spec["low"]), int(spec["high"]) + 1))
    if distribution == "choice":
        return int(rng.choice(np.asarray(spec["values"], dtype=int)))
    raise ValueError(f"unsupported integer distribution: {distribution}")


def sample_repair_seed(rng: np.random.Generator) -> int:
    """Materialize the repair RNG seed into replayable parameters."""

    return int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
