"""Unit tests for EvoReal Algorithm 1 normalization used by the corrected structure audit."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = ROOT / "scripts" / "evoreal_structure_alignment_audit_corrected_v2.py"
_SPEC = importlib.util.spec_from_file_location("evoreal_structure_v2", _SCRIPT)
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules["evoreal_structure_v2"] = _MOD
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)

SPLITS = _MOD.SPLITS
alg1_normalize = _MOD.alg1_normalize
fft_energy = _MOD.fft_energy
nn_ratio_stats = _MOD.nn_ratio_stats
tensor_sha256 = _MOD.tensor_sha256
FROZEN_SHA = _MOD.FROZEN_SHA


def test_algorithm1_shared_max_diff_and_aspect_ratio():
    rng = np.random.default_rng(2026)
    data_root = ROOT / "datasets" / "external_v2"
    tested = 0
    for split, dataset in SPLITS.items():
        coords = np.load(data_root / dataset / "coordinates.npy", allow_pickle=False)
        assert tensor_sha256(coords) == FROZEN_SHA[dataset]
        idx = rng.choice(len(coords), size=4, replace=False)
        for i in idx:
            xy = np.asarray(coords[i], dtype=np.float64)
            raw_ptp = xy.max(axis=0) - xy.min(axis=0)
            raw_aspect = raw_ptp[0] / raw_ptp[1]
            norm = alg1_normalize(xy)
            ptp = norm.max(axis=0) - norm.min(axis=0)
            assert np.isclose(max(ptp[0], ptp[1]), 1.0, atol=1e-12)
            assert np.isclose(ptp[0] / ptp[1], raw_aspect, atol=1e-12)
            independent = (xy - xy.min(axis=0)) / np.clip(xy.max(axis=0) - xy.min(axis=0), 1e-12, None)
            # Shared-max_diff must not independently squash both axes to 1 unless the box is square.
            if not np.isclose(raw_aspect, 1.0, atol=1e-6):
                assert not np.allclose(ptp, independent.max(axis=0) - independent.min(axis=0))
            tested += 1
    assert tested >= 10


def test_nn_ratio_is_cv_not_pairwise_scale():
    rng = np.random.default_rng(0)
    # Regular grid: CV of NN distances should be near 0.
    xs = np.linspace(0, 1, 10)
    xx, yy = np.meshgrid(xs, xs)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    grid_nn = nn_ratio_stats(grid)["nn_ratio"]
    # Clustered: two tight blobs far apart -> high CV.
    blob_a = rng.normal(loc=0.1, scale=0.01, size=(50, 2))
    blob_b = rng.normal(loc=0.9, scale=0.01, size=(50, 2))
    clustered = np.concatenate([blob_a, blob_b], axis=0)
    cluster_nn = nn_ratio_stats(clustered)["nn_ratio"]
    assert grid_nn < 0.15
    assert cluster_nn > 0.5


def test_fft_not_parseval_total_energy_scale():
    rng = np.random.default_rng(1)
    xy = rng.random((100, 2))
    val = fft_energy(xy)
    assert np.isfinite(val)
    assert abs(val - 16284) > 1000
