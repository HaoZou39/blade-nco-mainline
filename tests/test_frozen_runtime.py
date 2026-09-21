from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mutation import OPERATOR_NAMES, apply_operator
from bq.protocol import KNNS
from qd import QD4_FEATURE_NAMES, assign_nearest, load_centroids, normalize_qd4


def test_frozen_qd_coordinate_system() -> None:
    assert QD4_FEATURE_NAMES == (
        "spn_sum_of_lowest_edge_values",
        "spn_distance_varcoeff",
        "spn_mst_depth_mean",
        "spn_mst_dists_median",
    )
    centroids = load_centroids()
    assert centroids.shape == (1000, 4)
    points = normalize_qd4(np.zeros((2, 4)))
    niches = assign_nearest(points, centroids)
    assert niches.shape == (2,)


def test_large_scale_bq_uses_paper_knn_limit() -> None:
    assert KNNS == 250


def test_foga11_registry_and_legality() -> None:
    assert len(OPERATOR_NAMES) == 11
    coordinates = np.random.default_rng(2026).random((100, 2))
    for index, name in enumerate(OPERATOR_NAMES):
        result = apply_operator(name, coordinates, np.random.default_rng(index))
        assert result.final_coordinates.shape == (100, 2)
        assert np.isfinite(result.final_coordinates).all()
        assert result.valid_after_repair
