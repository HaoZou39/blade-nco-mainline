"""Default Unified-12 mutation library. FOGA11 remains available by name."""

from .registry import (
    FOGA11_OPERATOR_NAMES,
    OPERATOR_NAMES,
    OPERATOR_REGISTRY,
    UnknownOperatorError,
    apply_foga11_operator,
    apply_operator,
    apply_operator_with_config,
    get_operator,
    list_available_operators,
    sample_operator_name,
    sample_operator_parameters,
)
from .result import MutationResult

__all__ = [
    "FOGA11_OPERATOR_NAMES",
    "MutationResult",
    "OPERATOR_NAMES",
    "OPERATOR_REGISTRY",
    "UnknownOperatorError",
    "apply_foga11_operator",
    "apply_operator",
    "apply_operator_with_config",
    "get_operator",
    "list_available_operators",
    "sample_operator_name",
    "sample_operator_parameters",
]
