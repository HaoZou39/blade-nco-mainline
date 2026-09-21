from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qd.acquisition import qd_balanced_acquire  # noqa: E402
from qd.archive import CurrentSurrogateScore  # noqa: E402
from qd.probes import StructuralCandidate, select_centroid_nearest_probes  # noqa: E402


def test_round_robin_is_layered_not_globally_ranked() -> None:
    names = ["a", "b", "c", "d"]
    scores = {name: CurrentSurrogateScore(value, "s1") for name, value in zip(names, [10, 9, 100, 1], strict=True)}
    niches = {"a": 0, "b": 0, "c": 1, "d": 1}
    chosen = qd_balanced_acquire(names, scores, niches, 4, current_score_version="s1")
    assert [item.instance_hash for item in chosen] == ["a", "c", "b", "d"]
    assert [item.round_robin_layer for item in chosen] == [1, 1, 2, 2]


def test_round_robin_deduplicates_and_uses_hash_ties() -> None:
    names = ["b", "a", "a", "c"]
    scores = {name: CurrentSurrogateScore(1, "s1") for name in set(names)}
    niches = {"a": 0, "b": 0, "c": 1}
    chosen = qd_balanced_acquire(names, scores, niches, 3, current_score_version="s1")
    assert [item.instance_hash for item in chosen] == ["a", "c", "b"]
    assert len({item.instance_hash for item in chosen}) == 3


def test_round_robin_exact_10k_budget_and_shortfall_failure() -> None:
    names = [f"h{i:05d}" for i in range(10_000)]
    scores = {name: CurrentSurrogateScore(float(i), "s1") for i, name in enumerate(names)}
    niches = {name: i % 1000 for i, name in enumerate(names)}
    chosen = qd_balanced_acquire(names, scores, niches, current_score_version="s1")
    assert len(chosen) == len({item.instance_hash for item in chosen}) == 10_000
    with pytest.raises(ValueError, match="unique eligible"):
        qd_balanced_acquire(names[:-1], scores, niches, current_score_version="s1")


def test_probe_selection_is_centroid_nearest_then_hash() -> None:
    centers = np.zeros((1000, 4))
    candidates = [
        StructuralCandidate("b", 4, (0.1, 0, 0, 0)),
        StructuralCandidate("a", 4, (-0.1, 0, 0, 0)),
        StructuralCandidate("c", 9, (0.2, 0, 0, 0)),
    ]
    selected = select_centroid_nearest_probes(candidates, centers)
    assert [(item.niche_id, item.instance_hash) for item in selected] == [(4, "a"), (9, "c")]

