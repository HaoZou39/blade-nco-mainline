"""FOGA 2019-inspired adapted TSP instance mutation library."""

from .registry import (
    OPERATOR_NAMES,
    OPERATOR_REGISTRY,
    UnknownOperatorError,
    apply_operator,
    apply_operator_with_config,
    get_operator,
    list_available_operators,
    sample_operator_parameters,
)
from .result import MutationResult

__all__ = [
    "MutationResult",
    "OPERATOR_NAMES",
    "OPERATOR_REGISTRY",
    "UnknownOperatorError",
    "apply_operator",
    "apply_operator_with_config",
    "get_operator",
    "list_available_operators",
    "sample_operator_parameters",
]
