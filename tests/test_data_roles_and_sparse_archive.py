from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_roles import (  # noqa: E402
    ActiveHardPopulation,
    CandidateOffspringPool,
    DataRole,
    InstanceMetadata,
    SurrogateLabelArchive,
    UniformRootSet,
)
from qd.archive import ArchiveEntry, CurrentSurrogateScore, SparseQDArchive, update_sparse_qd_archive  # noqa: E402


def entry(name: str, niche: int, score: float = 0.0, version: str = "s0") -> ArchiveEntry:
    return ArchiveEntry(name, name, niche, (0.1, 0.2, 0.3, 0.4), score, version)


def scores(**values: float):
    return {name: CurrentSurrogateScore(value, "s1") for name, value in values.items()}


def test_data_roles_are_distinct_and_history_is_additive() -> None:
    record = InstanceMetadata("a", "a", None, None, 0, None, None, None, None, False, False, frozenset({DataRole.UNIFORM_ROOT}))
    promoted = record.with_role(DataRole.SURROGATE_PROBE)
    assert promoted.data_roles == {DataRole.UNIFORM_ROOT, DataRole.SURROGATE_PROBE}
    assert isinstance(UniformRootSet((record,)), UniformRootSet)
    assert isinstance(CandidateOffspringPool(()), CandidateOffspringPool)
    assert isinstance(ActiveHardPopulation(()), ActiveHardPopulation)
    assert isinstance(SurrogateLabelArchive(()), SurrogateLabelArchive)


def test_sparse_archive_empty_and_new_niche_discovery() -> None:
    assert len(update_sparse_qd_archive([], [], {}, {}, current_score_version="s1")) == 0
    archive = update_sparse_qd_archive([], [entry("a", 4)], scores(a=1), {"a": 4}, current_score_version="s1")
    assert len(archive) == 1 and archive.entries[0].niche_id == 4


def test_sparse_archive_replacement_and_non_replacement() -> None:
    incumbent = entry("b", 7, score=99)
    weak = entry("c", 7)
    strong = entry("a", 7)
    archive = update_sparse_qd_archive([incumbent], [weak], scores(b=2, c=1), {"b": 7, "c": 7}, current_score_version="s1")
    assert archive.entries[0].instance_hash == "b"
    archive = update_sparse_qd_archive([incumbent], [strong], scores(b=2, a=3), {"b": 7, "a": 7}, current_score_version="s1")
    assert archive.entries[0].instance_hash == "a"


def test_sparse_archive_child_migration_and_tie_break() -> None:
    child = entry("a", 1)
    incumbent = entry("b", 9)
    archive = update_sparse_qd_archive([incumbent], [child], scores(a=2, b=2), {"a": 9, "b": 9}, current_score_version="s1")
    assert archive.entries[0].instance_hash == "a"
    assert archive.entries[0].niche_id == 9


def test_sparse_archive_missing_and_stale_score_fail() -> None:
    with pytest.raises(ValueError, match="missing current score"):
        update_sparse_qd_archive([], [entry("a", 1)], {}, {"a": 1}, current_score_version="s1")
    with pytest.raises(ValueError, match="stale score"):
        update_sparse_qd_archive([], [entry("a", 1)], {"a": CurrentSurrogateScore(1, "s0")}, {"a": 1}, current_score_version="s1")


def test_sparse_archive_duplicate_hash_is_safe() -> None:
    duplicate = entry("a", 1)
    archive = update_sparse_qd_archive([duplicate], [replace(duplicate, current_surrogate_score=10)], scores(a=2), {"a": 1}, current_score_version="s1")
    assert len(archive) == 1
    with pytest.raises(ValueError, match="one elite per niche"):
        SparseQDArchive([entry("a", 1), entry("b", 1)])

