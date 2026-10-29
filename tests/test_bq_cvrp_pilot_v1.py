from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "bq_nco_official"))

from cvrp_pilot.core import (  # noqa: E402
    FEATURE_NAMES,
    official_loss_equivalence,
    official_trajectory,
    per_instance_hardness,
    structural_features,
    trajectory_bundle,
    validate_routes,
)
from cvrp_pilot.core import ReferenceRecord  # noqa: E402
from model.model import BQModel  # noqa: E402


def tiny_problem():
    depot = np.array([0.5, 0.5])
    customers = np.array([[0.1, 0.1], [0.2, 0.1], [0.8, 0.8], [0.9, 0.8]])
    demands = np.array([3, 4, 5, 2])
    routes = [[1, 2], [3, 4]]
    return depot, customers, demands, routes


def test_official_trajectory_encodes_depot_separator_and_preserves_ids():
    depot, customers, demands, routes = tiny_problem()
    trajectory = official_trajectory(depot, customers, demands, 10, routes)
    assert trajectory["coords"].shape == (6, 2)
    assert trajectory["demands"].tolist() == [0, 3, 4, 5, 2, 0]
    assert trajectory["remaining_capacities"].tolist() == [10, 7, 3, 5, 3, 10]
    assert trajectory["via_depots"].tolist() == [1, 0, 0, 1, 0, 0]


def test_route_validation_uses_all_customers_and_capacity():
    depot, customers, demands, routes = tiny_problem()
    rounded, exact, feasible, error = validate_routes(depot, customers, demands, 10, routes)
    assert feasible and not error
    assert rounded > 0 and exact > 0
    _, _, feasible, error = validate_routes(depot, customers, demands, 6, routes)
    assert not feasible and "exceeds" in error


def test_structural_features_are_finite_and_named():
    rng = np.random.default_rng(731)
    values = structural_features(rng.random(2), rng.random((100, 2)), rng.integers(1, 10, 100), 50)
    assert values.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(values).all()


def test_per_instance_loss_matches_official_cross_entropy_formula():
    depot, customers, demands, routes = tiny_problem()
    reference = ReferenceRecord("i", "h", routes, 1, 1.0, 1.0, 2, 0.1, True, "ok")
    bundle = {
        "depot": depot[None],
        "customers": customers[None],
        "demands": demands[None],
        "capacities": np.array([10]),
    }
    trajectories = trajectory_bundle(bundle, [reference])
    torch.manual_seed(731)
    model = BQModel(4, 32, 64, "relu", 2, 4, "softmax", 0.0, False, "cvrp")
    audit = official_loss_equivalence(model, trajectories, torch.device("cpu"), begin=0)
    assert audit["pass"]
    hardness = per_instance_hardness(model, trajectories, torch.device("cpu"), batch_size=1)
    assert hardness.shape == (1,)
    assert np.isfinite(hardness).all()
