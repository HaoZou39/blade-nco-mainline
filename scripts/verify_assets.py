#!/usr/bin/env python
"""Strict, read-only verification of every frozen mainline asset."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bq.checkpoint import load_canonical_checkpoint  # noqa: E402
from evaluation.datasets import load_lehd_cross_size, load_original_tsplib70  # noqa: E402
from qd.coordinates import QD4_FEATURE_NAMES, load_centroids, load_normalization  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_manifest_hashes() -> int:
    manifest = read_json(ROOT / "ASSET_MANIFEST.json")
    records = manifest["assets"]
    declared = {record["relative_path"] for record in records}
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "assets").rglob("*")
        if path.is_file()
    }
    require(declared == actual, f"asset manifest coverage mismatch: missing={actual-declared}, stale={declared-actual}")
    require(manifest["asset_count"] == len(records) == len(declared), "asset_count mismatch")
    for record in records:
        path = ROOT / record["relative_path"]
        digest = sha256_file(path)
        require(path.stat().st_size == record["size_bytes"] == record["destination_size"], f"size mismatch: {path}")
        require(digest == record["copied_sha256"] == record["destination_sha256"], f"hash mismatch: {path}")
        require(record["source_sha256"] == record["copied_sha256"], f"copy-time source hash mismatch: {path}")
        require(record["frozen"] is True and record["read_only"] is True, f"asset is not logically frozen: {path}")
    return len(records)


def verify_dataset_manifest(directory: Path) -> dict[str, Any]:
    manifest = read_json(directory / "manifest.json")
    require(manifest["read_only"] is True, f"dataset is not read-only: {directory.name}")
    require(manifest["source_hashes"] == manifest["destination_hashes"], f"dataset copy hashes differ: {directory.name}")
    for relative, expected in manifest["destination_hashes"].items():
        require(sha256_file(directory / relative) == expected, f"dataset file hash mismatch: {directory.name}/{relative}")
    return manifest


def verify_fixed_n100(directory: str, key: str) -> int:
    path = ROOT / "assets" / "evaluation" / directory / "instances" / "coordinates.npz"
    with np.load(path, allow_pickle=False) as payload:
        shape = payload[key].shape
    require(shape == (10_000, 100, 2), f"{directory} coordinate shape mismatch: {shape}")
    rows = pq.ParquetFile(ROOT / "assets" / "evaluation" / directory / "manifest.parquet").metadata.num_rows
    require(rows == 10_000, f"{directory} manifest row count mismatch: {rows}")
    return rows


def verify_all() -> dict[str, int]:
    asset_count = verify_manifest_hashes()
    required = [
        "assets/solver/M0/tsp.best",
        "assets/retention/uniform10k/fixed_uniform_anchor_10k_labels.parquet",
        "assets/qd/reference_cloud/valid_unique_saved_states.parquet",
        "assets/qd/reference_cloud/deep_reference_qd4.parquet",
        "assets/qd/normalization/normalization_A_P1P99.json",
        "assets/qd/cvt/centroids.csv",
        "assets/qd/roots/exact_uniform_roots_manifest.json",
        "assets/qd/roots/root_hashes.txt",
    ]
    require(all((ROOT / item).is_file() for item in required), "one or more required frozen assets are missing")

    model, meta = load_canonical_checkpoint(ROOT / "assets" / "solver" / "M0" / "tsp.best", torch.device("cpu"))
    require(meta["sha256"] == "6826d7a71b30d882e4673a26ced162b7391157795931efe230f4d019292bba15", "M0 identity mismatch")
    require(sum(parameter.numel() for parameter in model.parameters()) == 3_107_539, "M0 architecture mismatch")
    del model

    lower, upper = load_normalization()
    require(lower.shape == upper.shape == (4,) and np.all(upper > lower), "Normalization A failed to load")
    require(load_centroids().shape == (1000, 4), "K1000 centroids failed to load")
    qd4_schema = tuple(pq.ParquetFile(ROOT / "assets" / "qd" / "reference_cloud" / "deep_reference_qd4.parquet").schema.names[-4:])
    require(qd4_schema == QD4_FEATURE_NAMES, f"QD4 feature schema mismatch: {qd4_schema}")
    require(pq.ParquetFile(ROOT / "assets" / "qd" / "reference_cloud" / "deep_reference_qd4.parquet").metadata.num_rows == 210_000, "QD4 cloud count mismatch")
    require(pq.ParquetFile(ROOT / "assets" / "qd" / "reference_cloud" / "valid_unique_saved_states.parquet").metadata.num_rows == 210_000, "deep state count mismatch")
    root_manifest = read_json(ROOT / "assets" / "qd" / "roots" / "exact_uniform_roots_manifest.json")
    require(root_manifest["root_count"] == root_manifest["unique_coordinate_hashes"] == 10_000, "QD root count mismatch")
    require(root_manifest["source"]["path"] == "assets/qd/reference_cloud/valid_unique_saved_states.parquet", "QD roots are not localized to the frozen deep state artifact")
    require(root_manifest["source"]["sha256"] == sha256_file(ROOT / root_manifest["source"]["path"]), "QD root source identity mismatch")
    require(root_manifest["root_selection"] == "saved_depth == 0", "QD root selection mismatch")
    root_hashes = (ROOT / "assets" / "qd" / "roots" / "root_hashes.txt").read_text(encoding="utf-8").splitlines()
    require(len(root_hashes) == 10_000 and len(set(root_hashes)) == 10_000, "QD root hash membership mismatch")
    require(root_manifest["root_hashes_sha256"] == sha256_file(ROOT / "assets" / "qd" / "roots" / "root_hashes.txt"), "QD root hash file identity mismatch")

    anchor = ROOT / "assets" / "retention" / "uniform10k"
    with np.load(anchor / "fixed_uniform_anchor_10k_coordinates.npz", allow_pickle=False) as payload:
        require(payload["coordinates"].shape == (10_000, 100, 2), "Uniform anchor coordinate count mismatch")
    require(pq.ParquetFile(anchor / "fixed_uniform_anchor_10k_labels.parquet").metadata.num_rows == 10_000, "Uniform anchor labels count mismatch")
    require(pq.ParquetFile(anchor / "fixed_uniform_anchor_10k_concorde_ledger.parquet").metadata.num_rows == 10_000, "Uniform anchor expert ledger count mismatch")

    counts = {
        "Uniform100": verify_fixed_n100("uniform100", "coordinates"),
        "GaussianMixture100": verify_fixed_n100("gaussian_mixture100", "coordinates"),
        "TSPLIB100": verify_fixed_n100("tsplib100", "model_input_coordinates"),
    }
    original = list(load_original_tsplib70())
    require(len(original) == 70, "OriginalTSPLIB count mismatch")
    counts["OriginalTSPLIB70"] = len(original)
    for size in (200, 500, 1000):
        coordinates, tours = load_lehd_cross_size(size)
        require(coordinates.shape == (128, size, 2) and tours.shape == (128, size + 1), f"LEHD TSP{size} parse failure")
        counts[f"LEHD_TSP{size}"] = len(coordinates)

    for directory in ("uniform100", "gaussian_mixture100", "tsplib100", "original_tsplib70", "lehd_cross_size"):
        verify_dataset_manifest(ROOT / "assets" / "evaluation" / directory)
    inventory = read_json(ROOT / "assets" / "evaluation" / "lehd_cross_size" / "lehd_data_inventory.json")
    require({item["filename"] for item in inventory["files"]} == {"test_TSP200_n128.txt", "test_TSP500_n128.txt", "test_TSP1000_n128.txt", "TSPlib_70instances.txt"}, "LEHD inventory is incomplete")
    counts["asset_count"] = asset_count
    return counts


if __name__ == "__main__":
    result = verify_all()
    print("ASSET_VERIFICATION=PASS")
    for name, count in result.items():
        print(f"{name}={count}")
