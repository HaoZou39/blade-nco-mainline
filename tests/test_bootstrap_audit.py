from __future__ import annotations

from pathlib import Path
import json
import os
import stat
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audits" / "uniform_rooted_qd_bootstrap_v1"
sys.path.insert(0, str(ROOT / "src"))

from audit.uniform_bootstrap import child_seed  # noqa: E402
from features.qd4 import QD4_FEATURE_NAMES, extract_qd4_exact  # noqa: E402
from mutation import OPERATOR_NAMES, apply_operator  # noqa: E402
from qd.coordinates import assign_nearest, load_centroids, load_normalization, normalize_qd4  # noqa: E402
from qd.roots import _duckdb, recover_uniform_roots  # noqa: E402


def test_canonical_10k_root_recovery() -> None:
    roots = recover_uniform_roots()
    assert roots.coordinates.shape == (10_000, 100, 2)
    assert len(set(roots.root_hashes)) == 10_000
    assert roots.root_hashes == roots.coordinate_hashes


def test_exactly_11_operators_per_root_in_audit() -> None:
    frame = pd.read_parquet(AUDIT / "02_offspring_manifest.parquet", columns=["root_index", "operator_id"])
    assert len(frame) == 110_000
    assert frame.groupby("root_index").size().eq(11).all()
    assert frame.groupby(["root_index", "operator_id"]).size().eq(1).all()
    assert set(frame.operator_id) == set(OPERATOR_NAMES)


def test_deterministic_mutation_replay_matches_manifest() -> None:
    roots = recover_uniform_roots()
    manifest = pd.read_parquet(AUDIT / "02_offspring_manifest.parquet")
    expected = manifest[manifest.root_index.eq(0)].set_index("operator_id")
    for operator_id in OPERATOR_NAMES:
        result = apply_operator(operator_id, roots.coordinates[0], np.random.default_rng(child_seed(roots.root_hashes[0], operator_id)))
        assert result.final_output_hash == expected.loc[operator_id, "child_hash"]
        assert result.raw_output_hash == expected.loc[operator_id, "raw_child_hash"]


def test_qd4_exact_against_frozen_reference_and_assignments() -> None:
    roots = recover_uniform_roots()
    got = extract_qd4_exact(roots.coordinates[:32])
    db = _duckdb()
    path = (ROOT / "assets/qd/reference_cloud/deep_reference_qd4.parquet").as_posix()
    placeholders = ",".join(f"'{item}'" for item in roots.root_hashes[:32])
    rows = db.sql(f"SELECT coordinate_hash,{','.join(QD4_FEATURE_NAMES)} FROM read_parquet('{path}') WHERE depth=0 AND coordinate_hash IN ({placeholders})").fetchall()
    expected = {row[0]: np.asarray(row[1:], dtype=float) for row in rows}
    assert max(np.max(np.abs(got[i] - expected[roots.root_hashes[i]])) for i in range(32)) < 2e-12
    frozen = pd.read_parquet(AUDIT / "04_root_qd_assignments.parquet").iloc[:32]
    assert np.allclose(got, frozen[list(QD4_FEATURE_NAMES)], rtol=0, atol=2e-12)


def test_normalization_exactness_and_k1000_assignment() -> None:
    lower, upper = load_normalization()
    sample = np.vstack([lower - 1, (lower + upper) / 2, upper + 1])
    normalized = normalize_qd4(sample)
    assert np.array_equal(normalized[0], np.zeros(4))
    assert np.allclose(normalized[1], 0.5)
    assert np.array_equal(normalized[2], np.ones(4))
    centers = load_centroids()
    assert centers.shape == (1000, 4)
    assert np.array_equal(assign_nearest(centers[[0, 123, 999]], centers), [0, 123, 999])


def test_gp0_probe_manifests_are_deterministic_and_unlabeled() -> None:
    for name, source in [
        ("10_gp0_probe_candidates_offspring_only.parquet", "05_offspring_qd_assignments.parquet"),
        ("11_gp0_probe_candidates_union.parquet", None),
    ]:
        probes = pd.read_parquet(AUDIT / name)
        assert probes.niche_id.is_unique
        assert (~probes.oracle_labeled).all()
        assert (~probes.expert_tour_available).all()
        assert (~probes.current_exact_hardness_available).all()
        assert probes.provisional.all()
        centers = load_centroids()
        normalized = probes[[f"normalized_{item}" for item in QD4_FEATURE_NAMES]].to_numpy()
        assert np.allclose(np.linalg.norm(normalized - centers[probes.niche_id], axis=1), probes.centroid_distance)


def test_no_runtime_old_paths_or_symlinks() -> None:
    forbidden = "D:" + "\\NCO\\" + "BLADE-NCO\\"
    checked = [ROOT / "src", ROOT / "scripts", ROOT / "configs", ROOT / "README.md", ROOT / "METHOD_FROZEN.md", ROOT / "PROTOCOL_STATUS.yaml"]
    for base in checked:
        paths = [base] if base.is_file() else base.rglob("*")
        for path in paths:
            if path.is_file() and path.suffix.lower() in {".py", ".yaml", ".md"}:
                assert forbidden not in path.read_text(encoding="utf-8-sig")
    for path in ROOT.rglob("*"):
        info = os.lstat(path)
        assert not path.is_symlink()
        assert not (getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))

