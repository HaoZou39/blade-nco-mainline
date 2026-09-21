"""Frozen sparse-QD utilities (no GP or final evolution runner)."""

from .acquisition import AcquisitionSelection, qd_balanced_acquire
from .archive import ArchiveEntry, CurrentSurrogateScore, SparseQDArchive, update_sparse_qd_archive
from .coordinates import QD4_FEATURE_NAMES, assign_nearest, load_centroids, normalize_qd4
from .probes import ProbeSelection, StructuralCandidate, select_centroid_nearest_probes
from .roots import UniformRoots, recover_uniform_roots

__all__ = [
    "ArchiveEntry",
    "CurrentSurrogateScore",
    "SparseQDArchive",
    "update_sparse_qd_archive",
    "AcquisitionSelection",
    "qd_balanced_acquire",
    "ProbeSelection",
    "StructuralCandidate",
    "select_centroid_nearest_probes",
    "UniformRoots",
    "recover_uniform_roots",
    "QD4_FEATURE_NAMES",
    "assign_nearest",
    "load_centroids",
    "normalize_qd4",
]
