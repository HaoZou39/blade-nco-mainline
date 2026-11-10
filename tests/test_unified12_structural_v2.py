"""Structural V2 test suite. Does not mutate frozen V1."""
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mutation import OPERATOR_NAMES as V1_NAMES, apply_operator as v1_apply  # noqa: E402
from mutation.registry_v2 import OPERATOR_NAMES as V2_NAMES, apply_operator as v2_apply, load_profile, sample_operator_name  # noqa: E402
from mutation.structural_geometry_v2 import motif_sites_v3  # noqa: E402

def test_registry_and_no_overlapping_grid_templates():
    assert len(V1_NAMES) == len(V2_NAMES) == 12
    assert V2_NAMES[:10] == V1_NAMES[:10]
    assert V2_NAMES[-2:] == ("stripe_reorganization", "motif_assembly_v3")
    assert "lattice_reorganization" in V1_NAMES and "motif_assembly" in V1_NAMES
    assert "lattice_reorganization" not in V2_NAMES and "motif_assembly" not in V2_NAMES
    cfg = load_profile()["operators"]
    assert "grid_mutation" in cfg
    assert "template" not in cfg["stripe_reorganization"]
    assert cfg["motif_assembly_v3"]["motif"]["values"] == ["arc", "cross", "l_shape", "star"]
    assert cfg["motif_assembly_v3"]["version"] == "v3"

def test_cross_has_four_arms():
    center = np.array([0.5, 0.5]); rel = motif_sites_v3("cross", 21, center, 0.15, 0., np.random.default_rng(7), {}) - center
    assert rel.shape == (21, 2)
    assert np.count_nonzero((rel[:, 0] > 1e-8) & np.isclose(rel[:, 1], 0)) == 5
    assert np.count_nonzero((rel[:, 0] < -1e-8) & np.isclose(rel[:, 1], 0)) == 5
    assert np.count_nonzero((rel[:, 1] > 1e-8) & np.isclose(rel[:, 0], 0)) == 5
    assert np.count_nonzero((rel[:, 1] < -1e-8) & np.isclose(rel[:, 0], 0)) == 5
    with pytest.raises(ValueError):
        motif_sites_v3("grid_block", 21, center, 0.15, 0., np.random.default_rng(7), {})

def test_deterministic_replay_and_v1_retention():
    parent = np.random.default_rng(4).random((100, 2)); original = parent.copy()
    for index, name in enumerate(V2_NAMES):
        a = v2_apply(name, parent, np.random.default_rng(100 + index))
        b = v2_apply(name, parent, np.random.default_rng(100 + index))
        assert a.final_output_hash == b.final_output_hash
        assert a.operator_name == name and a.final_coordinates.shape == (100, 2)
        assert np.isfinite(a.final_coordinates).all()
        assert np.array_equal(parent, original)
        if a.proposal_status == "accepted":
            replay = v2_apply(name, parent, np.random.default_rng(0), a.sampled_parameters)
            assert replay.final_output_hash == a.final_output_hash
    assert v1_apply("motif_assembly", parent, np.random.default_rng(9)).operator_name == "motif_assembly"

def test_uniform_operator_sampling():
    draws = {sample_operator_name(np.random.default_rng(8000 + j)) for j in range(1000)}
    assert draws == set(V2_NAMES)
