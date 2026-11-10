"""V2-only motif geometry; V1 motif_sites remains immutable."""
from __future__ import annotations
from typing import Any, Mapping
import numpy as np
from .structural_geometry import motif_sites as legacy_motif_sites

def motif_sites_v3(kind: str, count: int, center: np.ndarray, scale: float,
                   orientation: float, rng: np.random.Generator,
                   settings: Mapping[str, Any]) -> np.ndarray:
    """A true cross uses a center and four equally angled arms."""
    if kind == "cross":
        if count < 5:
            raise ValueError("cross requires >=5 nodes")
        theta = np.deg2rad(orientation)
        per_arm = np.full(4, (count - 1) // 4, dtype=int)
        per_arm[: (count - 1) % 4] += 1
        sites = [np.asarray(center, dtype=np.float64)]
        for arm, n_arm in enumerate(per_arm):
            angle = theta + arm * np.pi / 2.0
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
            for step in range(1, n_arm + 1):
                sites.append(center + direction * (scale * step / n_arm))
        return np.asarray(sites, dtype=np.float64)
    if kind not in ("arc", "l_shape", "star"):
        raise ValueError(f"unsupported V3 motif: {kind}")
    return legacy_motif_sites(kind, count, center, scale, orientation, rng, settings)
