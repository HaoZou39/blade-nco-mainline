"""Structured output returned by every mutation operator."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

import numpy as np


def coordinate_hash(coordinates: np.ndarray) -> str:
    """Return the SHA-256 digest of a canonical float64 coordinate array."""

    array = np.ascontiguousarray(coordinates, dtype=np.float64)
    return hashlib.sha256(array.tobytes()).hexdigest()


@dataclass(frozen=True)
class MutationResult:
    """Complete raw/final mutation record used for replay and auditing."""

    operator_name: str
    input_coordinates: np.ndarray
    raw_coordinates: np.ndarray
    final_coordinates: np.ndarray
    sampled_parameters: Mapping[str, Any]
    affected_indices: np.ndarray
    affected_mask: np.ndarray
    n_affected: int
    repair_indices: np.ndarray
    repair_count: int
    repair_fraction: float
    valid_before_repair: bool
    valid_after_repair: bool
    proposal_status: str
    failure_reason: str
    input_hash: str
    raw_output_hash: str
    final_output_hash: str
    lineage_metadata: Mapping[str, Any]
    timing: Mapping[str, float]

    def to_summary(self) -> dict[str, Any]:
        """Return a compact JSON-serializable summary."""

        return {
            "operator_name": self.operator_name,
            "sampled_parameters": self.sampled_parameters,
            "affected_indices": self.affected_indices.tolist(),
            "n_affected": self.n_affected,
            "repair_indices": self.repair_indices.tolist(),
            "repair_count": self.repair_count,
            "repair_fraction": self.repair_fraction,
            "valid_before_repair": self.valid_before_repair,
            "valid_after_repair": self.valid_after_repair,
            "proposal_status": self.proposal_status,
            "failure_reason": self.failure_reason,
            "input_hash": self.input_hash,
            "raw_output_hash": self.raw_output_hash,
            "final_output_hash": self.final_output_hash,
            "lineage_metadata": self.lineage_metadata,
            "timing": dict(self.timing),
        }
