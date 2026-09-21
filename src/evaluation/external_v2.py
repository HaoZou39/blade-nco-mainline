"""Independent data/oracle utilities for EXTERNAL_BENCHMARK_SUITE_V2.

Outputs from this module are passive: they cannot affect the frozen formal run.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "external_benchmark_suite_v2.yaml"
SOURCE_ROOT = ROOT / "third_party" / "evoreal"
ASSET_ROOT = ROOT / "assets" / "evaluation"
DATA_ROOT = ROOT / "datasets" / "external_v2"
AUDIT_ROOT = ROOT / "audits" / "external_benchmark_suite_v2"
FORMAL_RUN = ROOT / "runs" / "blade_final_20cycle_seed2026_v1"

ACTIVE_DATASETS = (
    "Uniform100", "GaussianMixture100", "TSPLIB100",
    "EvoReal_S1_100", "EvoReal_S2_100", "EvoReal_S3_100",
    "GaussianMixture500", "OriginalTSPLIB70",
)
NEW_DATASETS = (
    "EvoReal_S1_100", "EvoReal_S2_100", "EvoReal_S3_100",
    "GaussianMixture500",
)
EVOREAL_FUNCTIONS = {
    "EvoReal_S1_100": ("gpt_S1.py", "generate_s1_type"),
    "EvoReal_S2_100": ("gpt_S2.py", "generate_s2_type"),
    "EvoReal_S3_100": ("gpt_S3.py", "generate_s3_type"),
}
EXPECTED_SOURCE_SHA256 = {
    "gpt_S1.py": "918bf7568ab4bcbe26042d0c8d02a221d4f9b2fc838e2b415992a55f31218fe0",
    "gpt_S2.py": "1129b5b902bc7aeee65c226e97e7180a601c2e35ed8b9810c3b8ead0a18b71f8",
    "gpt_S3.py": "b9eee5d038ca999a30d7aec80be365cacf5ee332264d6fe00e1a1c454d98c364",
}
FIXED_PATHS = {
    "Uniform100": ASSET_ROOT / "uniform100" / "instances" / "coordinates.npz",
    "GaussianMixture100": ASSET_ROOT / "gaussian_mixture100" / "instances" / "coordinates.npz",
    "TSPLIB100": ASSET_ROOT / "tsplib100" / "instances" / "coordinates.npz",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def coordinate_hash(coordinates: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(coordinates, dtype="<f8"))
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def coordinate_set_hash(coordinates: np.ndarray) -> str:
    arr = np.asarray(coordinates, dtype="<f8")
    return coordinate_hash(arr[np.lexsort((arr[:, 1], arr[:, 0]))])


def dataset_tensor_hash(coordinates: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(coordinates, dtype="<f4"))
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def derived_seed(dataset_name: str, master_seed: int = 2026) -> int:
    text = f"external_benchmark_suite_v2::{dataset_name}::{master_seed}"
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".npy", dir=path.parent)
    os.close(fd)
    try:
        np.save(temporary, value, allow_pickle=False)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def verify_evoreal_sources() -> dict[str, str]:
    manifest_path = SOURCE_ROOT / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("pinned_commit") != "919616337105e894e06463734ec1dd71f6451309":
        raise RuntimeError("EvoReal source manifest has the wrong pinned commit")
    actual = {name: sha256_file(SOURCE_ROOT / name) for name in EXPECTED_SOURCE_SHA256}
    for name, expected in EXPECTED_SOURCE_SHA256.items():
        if actual[name] != expected:
            raise RuntimeError(f"EvoReal source hash mismatch: {name}")
    return actual


def _load_source_function(filename: str, function_name: str):
    path = SOURCE_ROOT / filename
    spec = importlib.util.spec_from_file_location(f"external_v2_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)


def generate_evoreal(dataset_name: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    filename, function_name = EVOREAL_FUNCTIONS[dataset_name]
    seed = derived_seed(dataset_name)
    seed_all(seed)
    fn = _load_source_function(filename, function_name)
    coordinates = fn(batch_size=1000, problem_size=100).detach().cpu().numpy().astype(np.float32, copy=False)
    rows = [{
        "instance_id": f"{dataset_name}_{index:04d}",
        "dataset": dataset_name,
        "index": index,
        "seed": seed,
        "coordinate_hash": coordinate_hash(instance),
        "coordinate_set_hash": coordinate_set_hash(instance),
    } for index, instance in enumerate(coordinates)]
    return coordinates, rows


def generate_gaussian_mixture500() -> tuple[np.ndarray, list[dict[str, Any]]]:
    dataset_name = "GaussianMixture500"
    base_seed = derived_seed(dataset_name)
    batches: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for cdist_index, cdist in enumerate((10, 30, 50, 70, 100)):
        for candidate_index in range(200):
            seed = base_seed + cdist_index * 100_000 + candidate_index
            rng = np.random.Generator(np.random.PCG64(seed))
            cluster_count = int(rng.integers(3, 8))
            centers = rng.uniform(0.0, float(cdist), size=(cluster_count, 2))
            assignment = rng.integers(0, cluster_count, size=500)
            raw = rng.normal(loc=centers[assignment], scale=1.0)
            minima, maxima = raw.min(axis=0), raw.max(axis=0)
            coordinates = ((raw - minima) / (maxima - minima)).astype(np.float32)
            index = len(batches)
            batches.append(coordinates)
            rows.append({
                "instance_id": f"GaussianMixture500_cdist{cdist}_{candidate_index:04d}",
                "dataset": dataset_name,
                "index": index,
                "seed": seed,
                "cdist": cdist,
                "cdist_index": cdist_index,
                "candidate_index": candidate_index,
                "cluster_count": cluster_count,
                "cluster_size_vector": json.dumps(np.bincount(assignment, minlength=cluster_count).tolist()),
                "coordinate_hash": coordinate_hash(coordinates),
                "coordinate_set_hash": coordinate_set_hash(coordinates),
            })
    return np.stack(batches), rows


def generate_dataset(dataset_name: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if dataset_name in EVOREAL_FUNCTIONS:
        return generate_evoreal(dataset_name)
    if dataset_name == "GaussianMixture500":
        return generate_gaussian_mixture500()
    raise KeyError(dataset_name)


def validate_coordinates(dataset_name: str, coordinates: np.ndarray) -> dict[str, Any]:
    expected = (1000, 500, 2) if dataset_name == "GaussianMixture500" else (1000, 100, 2)
    if coordinates.shape != expected or coordinates.dtype != np.float32:
        raise RuntimeError(f"{dataset_name}: got {coordinates.shape}/{coordinates.dtype}; expected {expected}/float32")
    if not np.isfinite(coordinates).all():
        raise RuntimeError(f"{dataset_name}: non-finite coordinates")
    if float(coordinates.min()) < 0.0 or float(coordinates.max()) > 1.0:
        raise RuntimeError(f"{dataset_name}: coordinates outside [0,1]")
    ordered = [coordinate_hash(x) for x in coordinates]
    unordered = [coordinate_set_hash(x) for x in coordinates]
    if len(set(ordered)) != len(ordered) or len(set(unordered)) != len(unordered):
        raise RuntimeError(f"{dataset_name}: duplicate whole instances")
    duplicate_instances = 0
    duplicate_nodes = 0
    for instance in coordinates:
        unique_count = np.unique(instance, axis=0).shape[0]
        if unique_count != instance.shape[0]:
            duplicate_instances += 1
            duplicate_nodes += instance.shape[0] - unique_count
    return {
        "shape": list(coordinates.shape), "dtype": str(coordinates.dtype),
        "minimum": float(coordinates.min()), "maximum": float(coordinates.max()),
        "ordered_instance_hashes_unique": True, "coordinate_set_hashes_unique": True,
        "duplicate_node_instances_diagnostic": duplicate_instances,
        "duplicate_node_count_diagnostic": duplicate_nodes,
        "duplicate_node_policy": "preserve exact official output; no repair or filtering",
    }


def load_fixed_coordinates(dataset_name: str) -> np.ndarray:
    with np.load(FIXED_PATHS[dataset_name], allow_pickle=False) as archive:
        key = "model_input_coordinates" if dataset_name == "TSPLIB100" else "coordinates"
        return np.asarray(archive[key], dtype=np.float32)


def formal_archive_hashes() -> set[str]:
    result: set[str] = set()
    for path in sorted((FORMAL_RUN / "cycles").glob("C*/07_oracle/D_selected.npz")):
        try:
            with np.load(path, allow_pickle=False) as archive:
                if "instance_hash" in archive:
                    result.update(str(x) for x in archive["instance_hash"].tolist())
                elif "coordinates" in archive:
                    result.update(coordinate_hash(x) for x in archive["coordinates"])
        except (OSError, ValueError):
            continue
    return result


def overlap_reference_hashes() -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for name in FIXED_PATHS:
        coordinates = load_fixed_coordinates(name)
        result[name] = {coordinate_hash(x) for x in coordinates}
        result[name + "_unordered"] = {coordinate_set_hash(x) for x in coordinates}
    result["formal_archive"] = formal_archive_hashes()
    return result


def build_new_datasets() -> dict[str, dict[str, Any]]:
    verify_evoreal_sources()
    references = overlap_reference_hashes()
    accumulated_ordered: set[str] = set()
    accumulated_unordered: set[str] = set()
    reports: dict[str, dict[str, Any]] = {}
    for dataset_name in NEW_DATASETS:
        coordinates, rows = generate_dataset(dataset_name)
        replay_coordinates, replay_rows = generate_dataset(dataset_name)
        if not np.array_equal(coordinates, replay_coordinates) or rows != replay_rows:
            raise RuntimeError(f"{dataset_name}: reproducibility replay mismatch")
        validation = validate_coordinates(dataset_name, coordinates)
        ordered = {row["coordinate_hash"] for row in rows}
        unordered = {row["coordinate_set_hash"] for row in rows}
        overlaps = {}
        for reference_name, hashes in references.items():
            candidate = unordered if reference_name.endswith("_unordered") else ordered
            overlaps[reference_name] = len(candidate.intersection(hashes))
        overlaps["new_suite_ordered"] = len(ordered.intersection(accumulated_ordered))
        overlaps["new_suite_unordered"] = len(unordered.intersection(accumulated_unordered))
        if any(overlaps.values()):
            raise RuntimeError(f"{dataset_name}: overlap detected: {overlaps}")
        accumulated_ordered.update(ordered)
        accumulated_unordered.update(unordered)
        output_dir = DATA_ROOT / dataset_name
        coordinates_path = output_dir / "coordinates.npy"
        manifest_path = output_dir / "instances.csv"
        metadata_path = output_dir / "metadata.json"
        atomic_npy(coordinates_path, coordinates)
        atomic_csv(manifest_path, rows, list(rows[0]))
        metadata = {
            "schema_version": "EXTERNAL_BENCHMARK_SUITE_V2_DATASET",
            "dataset": dataset_name, "derived_seed": derived_seed(dataset_name),
            "coordinates_file_sha256": sha256_file(coordinates_path),
            "coordinate_tensor_sha256": dataset_tensor_hash(coordinates),
            "instance_manifest_sha256": sha256_file(manifest_path),
            "validation": validation, "overlap_counts": overlaps,
            "generation_policy": {"augmentation": False, "mutation": False, "repair": False,
                                  "filtering": False, "replay_bitwise_equal": True},
        }
        atomic_json(metadata_path, metadata)
        reports[dataset_name] = metadata
    return reports


def load_new_dataset(dataset_name: str) -> tuple[np.ndarray, list[dict[str, str]]]:
    coordinates = np.load(DATA_ROOT / dataset_name / "coordinates.npy", allow_pickle=False)
    with (DATA_ROOT / dataset_name / "instances.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return coordinates, rows


def source_and_config_hashes() -> dict[str, Any]:
    return {
        "config_sha256": sha256_file(CONFIG_PATH),
        "source_manifest_sha256": sha256_file(SOURCE_ROOT / "SOURCE_MANIFEST.json"),
        "evoreal_source_sha256": verify_evoreal_sources(),
        "module_sha256": sha256_file(Path(__file__)),
    }


