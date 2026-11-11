"""Default Unified-12 registry, frozen CURRENT parameters, and historical replay."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mutation import (  # noqa: E402
    FOGA11_OPERATOR_NAMES,
    OPERATOR_NAMES,
    apply_foga11_operator,
    apply_operator,
    get_operator,
    sample_operator_name,
)
from mutation.parameters import load_named_profile, load_operator_config  # noqa: E402


EXPECTED = (
    "uniform_mutation",
    "normal_mutation",
    "rotation_mutation",
    "implosion_mutation",
    "explosion_mutation",
    "cluster_mutation",
    "expansion_mutation",
    "compression_mutation",
    "axis_projection_mutation",
    "grid_mutation",
    "stripe_reorganization",
    "motif_assembly",
)


def test_default_registry_is_the_twelve_frozen_operators() -> None:
    assert OPERATOR_NAMES == EXPECTED
    assert "linear_projection_mutation" not in OPERATOR_NAMES
    assert "outer_ring_mutation" not in OPERATOR_NAMES


def test_new_operators_and_all_twelve_accept_n100() -> None:
    parent = np.random.default_rng(4).random((100, 2))
    original = parent.copy()
    for index, name in enumerate(OPERATOR_NAMES):
        result = apply_operator(name, parent, np.random.default_rng(100 + index))
        assert result.final_coordinates.shape == (100, 2)
        assert np.isfinite(result.final_coordinates).all()
        assert result.operator_name == name
        if result.proposal_status == "accepted":
            assert np.all((result.final_coordinates >= 0.0) & (result.final_coordinates <= 1.0))
            assert result.valid_after_repair
        assert np.array_equal(parent, original)


def test_same_seed_and_explicit_parameters_replay() -> None:
    parent = np.random.default_rng(8).random((100, 2))
    for name in OPERATOR_NAMES:
        first = apply_operator(name, parent, np.random.default_rng(23))
        second = apply_operator(name, parent, np.random.default_rng(23))
        assert first.final_output_hash == second.final_output_hash
        assert first.raw_output_hash == second.raw_output_hash
        if first.proposal_status == "accepted":
            replay = apply_operator(name, parent, np.random.default_rng(0), first.sampled_parameters)
            assert replay.final_output_hash == first.final_output_hash


def test_default_config_is_unified12_and_ranges_match_the_frozen_manifest() -> None:
    loaded = load_operator_config()
    named = load_named_profile("UNIFIED12_DEFAULT")
    assert loaded["profile_name"] == "UNIFIED12_DEFAULT"
    assert loaded["predecessor_profile"] == "UNIFIED12_FROZEN_V1"
    assert named["operators"]["rotation_mutation"]["affected_fraction"] == {"distribution": "uniform", "low": 0.25, "high": 0.5}
    operators = loaded["operators"]
    assert operators["implosion_mutation"]["radius"]["low"] == 0.2
    assert operators["implosion_mutation"]["radius"]["high"] == 0.4
    assert operators["implosion_mutation"]["strength"] == {"distribution": "uniform", "low": 0.2, "high": 0.8}
    assert operators["cluster_mutation"]["affected_count"] == {"distribution": "integer_uniform", "low": 15, "high": 40}
    assert operators["grid_mutation"]["box_width"]["low"] == 0.35
    assert operators["grid_mutation"]["box_height"]["high"] == 0.65
    assert operators["stripe_reorganization"]["alpha"] == {"distribution": "uniform", "low": 0.35, "high": 1.0}
    assert "template" not in operators["stripe_reorganization"]
    assert "grid_half_span" not in operators["stripe_reorganization"]
    assert operators["motif_assembly"]["version"] == "v3"
    assert operators["motif_assembly"]["motif"]["values"] == ["arc", "cross", "l_shape", "star"]
    assert operators["motif_assembly"]["star_arms"] == {"distribution": "integer_uniform", "low": 5, "high": 6}
    assert "cross_fraction" not in operators["motif_assembly"]
    assert operators["motif_assembly"]["affected_count"] == {"distribution": "integer_uniform", "low": 20, "high": 40}
    assert operators["motif_assembly"]["alpha"]["low"] == 0.45
    assert "lattice_reorganization" not in operators
    assert "linear_projection_mutation" not in operators
    text = (ROOT / "src/mutation/unified12_frozen.yaml").read_text()
    assert "low: 0.20" in text or "low: 0.2" in text


# UNIFIED12_FROZEN_V1 at 0aa106ac. The fixture is kept. These two rows describe
# the predecessor lattice and motif calls and are not replayed on the new operators.
ARCHIVED_STRUCTURAL_HASHES = {
    "motif_assembly": "6e84d78085498e4be8ad601aaaf1c4cbf41282b541cba3128036b0b4d0bb0468",
    "lattice_reorganization": "a2cb2b55b26425222840386493f2cb5edb659e4f67724c01c436a4529bb5a27a",
}


def test_historical_unified12_examples_replay() -> None:
    fixture = json.loads((ROOT / "tests/fixtures/unified12_historical_replay.json").read_text())
    parents = np.load(ROOT / "tests/fixtures/unified12_historical_replay.npz")["parents"]
    assert len(fixture) == 12
    archived = {row["operator"]: row["final_hash"] for row in fixture if row["operator"] in ARCHIVED_STRUCTURAL_HASHES}
    assert archived == ARCHIVED_STRUCTURAL_HASHES
    assert "lattice_reorganization" not in OPERATOR_NAMES
    for row, parent in zip(fixture, parents):
        if row["operator"] in ARCHIVED_STRUCTURAL_HASHES:
            continue
        result = apply_operator(row["operator"], parent, np.random.default_rng(np.random.SeedSequence(row["seed"])))
        assert result.proposal_status == "accepted"
        assert result.final_output_hash == row["final_hash"]
        assert result.final_coordinates.shape == (100, 2)


def test_ten_random_steps_and_nonzero_sampling_probability() -> None:
    parent = np.random.default_rng(5).random((100, 2))
    rng = np.random.default_rng(20261112)
    seen = set()
    for _ in range(10):
        name = sample_operator_name(rng)
        seen.add(name)
        result = apply_operator(name, parent, rng)
        assert result.final_coordinates.shape == (100, 2)
        if result.proposal_status == "accepted":
            parent = result.final_coordinates.copy()
    draws = [sample_operator_name(np.random.default_rng(20261111 + index)) for index in range(2400)]
    assert set(draws) == set(OPERATOR_NAMES)


def test_foga11_profile_and_linear_projection_remain_explicit() -> None:
    profile = load_named_profile("FOGA11_BASE_V1")
    assert set(profile["operators"]) == set(FOGA11_OPERATOR_NAMES)
    assert len(FOGA11_OPERATOR_NAMES) == 11
    assert "linear_projection_mutation" in FOGA11_OPERATOR_NAMES
    parent = np.random.default_rng(6).random((100, 2))
    result = apply_foga11_operator("linear_projection_mutation", parent, np.random.default_rng(6))
    assert result.operator_name == "linear_projection_mutation"
    assert result.final_coordinates.shape == (100, 2)
    spec = get_operator("linear_projection_mutation")
    assert spec.name == "linear_projection_mutation"
    disk = yaml.safe_load((ROOT / "src/mutation/foga2019_operators.yaml").read_text())
    assert disk["operators"]["implosion_mutation"]["radius"]["high"] == 0.3
    assert disk["operators"]["rotation_mutation"]["affected_fraction"]["high"] == 0.35
