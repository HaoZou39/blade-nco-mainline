"""Geometry, failure replay, sampling, and base-10 compatibility for Unified-12."""

from __future__ import annotations

import subprocess
from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mutation import OPERATOR_NAMES, apply_operator, apply_operator_with_config, sample_operator_name  # noqa: E402
from mutation.parameters import load_named_profile, load_operator_config, resolve_operator_config  # noqa: E402
from mutation.structural_geometry import motif_sites  # noqa: E402
import mutation.stripe_reorganization as stripes  # noqa: E402


PREDECESSOR = "0aa106ac48b39d9b2098805acce1985285ec688b"
BASE10_HASHES = {
    "uniform_mutation": "7fd4863994651a7521d3332ac29adc6f9188e3a6cb7a4982b8bdf07d96ad0f6b",
    "normal_mutation": "ff06464947fb075d6ead6eb4b6ff2c0577b7533ff35e787eb80034962734e71e",
    "rotation_mutation": "e5adbe293fc2335667094e535c46cbf19e85fb71d3365eed9e5908d36e12d3df",
    "implosion_mutation": "dc4050634ef4ef9690595727dd130e8fa66cd604f19f4e97f32fbe0b80066004",
    "explosion_mutation": "ef1ec200e6971f379fa7129c9bb24835e438686cc68b44d2bf3d48a2b03ab3a2",
    "cluster_mutation": "4929a5b5b78f842245f7dbdd7a0236b785bf991d88f3e843a91c06a393ce6065",
    "expansion_mutation": "acacffe819cede783f0a259dee5633ee01a29085016f3529ee9ad90ac2e4b099",
    "compression_mutation": "a1e12106dae59ea36355b93dc7c889c2a086ba1d9f356ff849b6760b2a43c311",
    "axis_projection_mutation": "aefb39b9e89522285d1225ec28fb379bb76eb81403460e9ae1db6187e9d74c4c",
    "grid_mutation": "381f9b0eb3a19cf7e504864580ac23e083841f41c51937a1ddf4c3cd4c683f6c",
}


def _directions(sites: np.ndarray, center: np.ndarray) -> list[np.ndarray]:
    relative = sites - center
    distance = np.linalg.norm(relative, axis=1)
    groups: list[np.ndarray] = []
    for vector in relative[distance > 1e-8]:
        vector = vector / np.linalg.norm(vector)
        if not any(float(np.dot(vector, group)) > 1.0 - 1e-6 for group in groups):
            groups.append(vector)
    return groups


def test_registry_names_sampling_and_callable_structural_ops() -> None:
    assert OPERATOR_NAMES == (
        "uniform_mutation", "normal_mutation", "rotation_mutation", "implosion_mutation",
        "explosion_mutation", "cluster_mutation", "expansion_mutation", "compression_mutation",
        "axis_projection_mutation", "grid_mutation", "stripe_reorganization", "motif_assembly",
    )
    assert "lattice_reorganization" not in OPERATOR_NAMES
    profile = load_operator_config()
    assert "grid_block" not in profile["operators"]["motif_assembly"]["motif"]["values"]
    assert profile["operators"]["motif_assembly"]["star_arms"]["low"] == 5
    parent = np.random.default_rng(3).random((100, 2))
    for name in ("stripe_reorganization", "motif_assembly"):
        result = apply_operator_with_config(name, parent, np.random.default_rng(9), profile)
        assert result.operator_name == name
        assert result.final_coordinates.shape == (100, 2)
    with pytest.raises(KeyError):
        apply_operator("lattice_reorganization", parent, np.random.default_rng(1))
    with pytest.raises(KeyError):
        load_named_profile("UNIFIED12_FROZEN_V1")
    draws = [sample_operator_name(np.random.default_rng(8000 + index)) for index in range(12000)]
    counts = Counter(draws)
    assert set(counts) == set(OPERATOR_NAMES)
    assert "lattice_reorganization" not in counts
    assert all(700 <= count <= 1300 for count in counts.values())


def test_global_stripes_share_bounds_and_local_windows_stay() -> None:
    parent = np.random.default_rng(4).random((100, 2))
    settings = resolve_operator_config(stripes.OPERATOR_NAME, None)
    seen: list[np.ndarray] = []
    real_rotate = stripes.rotate

    def capture(sites, center, degrees):
        seen.append(np.asarray(sites, dtype=np.float64).copy())
        return real_rotate(sites, center, degrees)

    stripes.rotate = capture
    try:
        observed = {"global": set(), "local": 0, "regional": 0}
        for index in range(240):
            seen.clear()
            drawn = stripes._once(parent, np.random.default_rng(6000 + index), settings)
            if not seen:
                continue
            sites = seen[0]
            center = np.asarray(drawn["center"], dtype=np.float64)
            extent = float(np.max(np.abs(sites - center)))
            if drawn["scope"] == "global":
                assert np.allclose(sites.min(axis=0), 0.02)
                assert np.allclose(sites.max(axis=0), 0.98)
                observed["global"].add(drawn["stripe_axis"])
            elif drawn["scope"] == "local":
                assert 0.125 - 1e-9 <= extent <= 0.225 + 1e-9
                observed["local"] += 1
            else:
                assert 0.275 - 1e-9 <= extent <= 0.425 + 1e-9
                observed["regional"] += 1
    finally:
        stripes.rotate = real_rotate
    assert observed["global"] == {"x", "y"}
    assert observed["local"] > 0 and observed["regional"] > 0


def test_cross_star_and_l_shape_geometry() -> None:
    center = np.array([0.4, 0.55])
    settings = load_operator_config()["operators"]["motif_assembly"]
    rng = np.random.default_rng(7)
    for count in (5, 20, 21, 22, 23):
        for orientation in (0.0, 20.0, 37.0):
            sites = motif_sites("cross", count, center, 0.2, orientation, rng, settings)
            assert sites.shape == (count, 2)
            assert np.any(np.linalg.norm(sites - center, axis=1) <= 1e-12)
            groups = _directions(sites, center)
            assert len(groups) == 4
            counts = []
            relative = sites - center
            distance = np.linalg.norm(relative, axis=1)
            for group in groups:
                projection = relative[distance > 1e-8] @ group
                counts.append(int(np.sum(projection > 1e-6)))
            assert sum(counts) == count - 1
            assert max(counts) - min(counts) <= 1
            for index, group in enumerate(groups):
                assert abs(float(np.dot(group, groups[(index + 1) % 4]))) < 1e-6
    for _ in range(30):
        sites = motif_sites("star", 24, center, 0.2, 15.0, np.random.default_rng(np.random.SeedSequence([11, _])), settings)
        assert len(_directions(sites, center)) in {5, 6}
    sites = motif_sites("l_shape", 21, center, 0.2, 25.0, np.random.default_rng(3), settings)
    groups = _directions(sites, center)
    assert len(groups) == 2
    assert abs(abs(float(np.dot(groups[0], groups[1]))) ) < 1e-6
    with pytest.raises(ValueError):
        motif_sites("grid_block", 21, center, 0.2, 0.0, np.random.default_rng(1), settings)


def test_failure_replay_for_small_inputs_and_empty_regions() -> None:
    small = np.random.default_rng(1).random((4, 2))
    failed = apply_operator("motif_assembly", small, np.random.default_rng(2))
    assert failed.proposal_status == "failed"
    replay = apply_operator("motif_assembly", small, np.random.default_rng(0), failed.sampled_parameters)
    assert replay.proposal_status == "failed"
    assert replay.failure_reason == failed.failure_reason
    assert np.array_equal(replay.final_coordinates, small)
    assert replay.final_output_hash == failed.final_output_hash

    profile = deepcopy(load_operator_config())
    profile["operators"]["stripe_reorganization"]["scope"] = {
        "distribution": "choice", "values": ["local"], "probabilities": [1.0],
    }
    profile["operators"]["stripe_reorganization"]["local_side"] = {"distribution": "uniform", "low": 0.05, "high": 0.05}
    parent = np.full((100, 2), 0.97)
    found = None
    for index in range(100):
        result = apply_operator_with_config("stripe_reorganization", parent, np.random.default_rng(100 + index), profile)
        if result.failure_reason == "empty_region":
            found = result
            break
    assert found is not None
    replay = apply_operator("stripe_reorganization", parent, np.random.default_rng(0), found.sampled_parameters)
    assert replay.proposal_status == "failed"
    assert replay.failure_reason == "empty_region"
    assert np.array_equal(replay.final_coordinates, parent)
    assert replay.final_output_hash == found.final_output_hash


def test_first_ten_match_predecessor_distributions_and_hashes() -> None:
    old_text = subprocess.check_output(["git", "show", f"{PREDECESSOR}:src/mutation/unified12_frozen.yaml"], cwd=ROOT)
    old = yaml.safe_load(old_text)
    new = load_operator_config()
    assert list(OPERATOR_NAMES[:10]) == list(BASE10_HASHES)
    for name in OPERATOR_NAMES[:10]:
        assert new["operators"][name] == old["operators"][name]
    assert new["repair"] == old["repair"]
    parent = np.random.default_rng(4).random((100, 2))
    for index, name in enumerate(OPERATOR_NAMES[:10]):
        result = apply_operator(name, parent, np.random.default_rng(100 + index))
        assert result.final_output_hash == BASE10_HASHES[name]


def test_tsp100_stripe_and_motif_smoke() -> None:
    parent = np.random.default_rng(11).random((100, 2))
    rates = {}
    for name in ("stripe_reorganization", "motif_assembly"):
        accepted = 0
        for index in range(40):
            first = apply_operator(name, parent, np.random.default_rng(3000 + index))
            second = apply_operator(name, parent, np.random.default_rng(3000 + index))
            assert first.final_output_hash == second.final_output_hash
            assert first.proposal_status == second.proposal_status
            coords = first.final_coordinates
            assert coords.shape == (100, 2)
            assert np.isfinite(coords).all()
            if first.proposal_status == "accepted":
                accepted += 1
                assert np.all((coords >= 0.0) & (coords <= 1.0))
                delta = coords[:, None, :] - coords[None, :, :]
                distance = np.sqrt(np.sum(delta * delta, axis=2))
                np.fill_diagonal(distance, np.inf)
                assert float(distance.min()) >= 1e-4
                replay = apply_operator(name, parent, np.random.default_rng(0), first.sampled_parameters)
                assert replay.proposal_status == "accepted"
                assert replay.final_output_hash == first.final_output_hash
            else:
                replay = apply_operator(name, parent, np.random.default_rng(0), first.sampled_parameters)
                assert replay.proposal_status == "failed"
                assert np.array_equal(replay.final_coordinates, parent)
        rates[name] = accepted / 40
        assert accepted > 0
    print("SMOKE_ACCEPTANCE", rates)
