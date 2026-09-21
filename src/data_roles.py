"""Explicitly distinct data-role abstractions for the final architecture."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Generic, Iterable, TypeVar


class DataRole(str, Enum):
    UNIFORM_ROOT = "uniform_root"
    CANDIDATE_OFFSPRING = "candidate_offspring"
    QD_ELITE = "qd_elite"
    ACTIVE_HARD = "active_hard"
    SURROGATE_PROBE = "surrogate_probe"
    UNIFORM_ANCHOR = "uniform_anchor"


@dataclass(frozen=True)
class InstanceMetadata:
    instance_hash: str
    root_hash: str | None
    parent_hash: str | None
    operator_id: str | None
    cycle: int
    qd4: tuple[float, float, float, float] | None
    normalized_qd4: tuple[float, float, float, float] | None
    niche_id: int | None
    current_surrogate_score: float | None
    expert_tour_available: bool
    current_exact_hardness_available: bool
    data_roles: frozenset[DataRole]

    def with_role(self, role: DataRole) -> "InstanceMetadata":
        """Add a historical role without discarding prior provenance."""

        return replace(self, data_roles=self.data_roles | {role})


T = TypeVar("T", bound=InstanceMetadata)


@dataclass(frozen=True)
class _RoleCollection(Generic[T]):
    records: tuple[T, ...]

    @classmethod
    def from_iterable(cls, records: Iterable[T]):
        return cls(tuple(records))


@dataclass(frozen=True)
class UniformRootSet(_RoleCollection[InstanceMetadata]):
    pass


@dataclass(frozen=True)
class CandidateOffspringPool(_RoleCollection[InstanceMetadata]):
    pass


@dataclass(frozen=True)
class ActiveHardPopulation(_RoleCollection[InstanceMetadata]):
    budget: int = 10_000


@dataclass(frozen=True)
class SurrogateLabelArchive(_RoleCollection[InstanceMetadata]):
    pass

