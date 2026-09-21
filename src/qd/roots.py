"""Recovery of exact canonical Uniform roots from frozen mainline assets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STATE_ASSET = ROOT / "assets" / "qd" / "reference_cloud" / "valid_unique_saved_states.parquet"
HASH_MANIFEST = ROOT / "assets" / "qd" / "roots" / "root_hashes.txt"


@dataclass(frozen=True)
class UniformRoots:
    root_indices: np.ndarray
    root_hashes: tuple[str, ...]
    coordinate_hashes: tuple[str, ...]
    coordinates: np.ndarray


def _duckdb():
    try:
        import duckdb
    except ModuleNotFoundError:
        vendor = ROOT / "vendor"
        if str(vendor) not in sys.path:
            sys.path.insert(0, str(vendor))
        import duckdb
    return duckdb


def coordinate_hash(coordinates: np.ndarray) -> str:
    array = np.ascontiguousarray(coordinates, dtype="<f8")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def recover_uniform_roots() -> UniformRoots:
    """Require saved_depth=0 and exact ordered root-hash membership."""

    if not STATE_ASSET.is_file() or not HASH_MANIFEST.is_file():
        raise FileNotFoundError("canonical root assets are missing")
    expected = tuple(line.strip() for line in HASH_MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip())
    if len(expected) != 10_000 or len(set(expected)) != 10_000:
        raise RuntimeError("canonical root-hash manifest is not exactly 10,000 unique hashes")
    db = _duckdb()
    path = STATE_ASSET.as_posix().replace("'", "''")
    rows = db.sql(
        f"""SELECT root_index, root_hash, coordinate_hash, coordinates,
                   coordinate_encoding, validity_flag, duplicate_flag
            FROM read_parquet('{path}')
            WHERE saved_depth = 0
            ORDER BY root_index"""
    ).fetchall()
    if len(rows) != 10_000:
        raise RuntimeError(f"saved_depth=0 returned {len(rows)} roots")
    indices = np.asarray([row[0] for row in rows], dtype=np.int64)
    hashes = tuple(str(row[1]) for row in rows)
    coordinate_hashes = tuple(str(row[2]) for row in rows)
    if not np.array_equal(indices, np.arange(10_000)):
        raise RuntimeError("root indices are not canonical 0..9999")
    if hashes != expected or set(hashes) != set(expected):
        raise RuntimeError("saved_depth=0 roots do not match the canonical hash manifest")
    if hashes != coordinate_hashes:
        raise RuntimeError("root hashes differ from coordinate hashes")
    coordinates = np.empty((10_000, 100, 2), dtype=np.float64)
    for index, row in enumerate(rows):
        if row[4] != "little-endian float64 C-order bytes with shape (100,2)":
            raise RuntimeError(f"unexpected coordinate encoding at root {index}")
        if row[5] is not True or row[6] is not False:
            raise RuntimeError(f"invalid or duplicate canonical root at index {index}")
        points = np.frombuffer(row[3], dtype="<f8")
        if points.size != 200:
            raise RuntimeError(f"root {index} is not N=100")
        coordinates[index] = points.reshape(100, 2)
    if not np.isfinite(coordinates).all() or np.any((coordinates < 0) | (coordinates > 1)):
        raise RuntimeError("canonical roots contain illegal coordinates")
    computed = tuple(coordinate_hash(points) for points in coordinates)
    if computed != coordinate_hashes:
        raise RuntimeError("coordinate bytes do not reproduce canonical hashes")
    return UniformRoots(indices, hashes, coordinate_hashes, coordinates)

