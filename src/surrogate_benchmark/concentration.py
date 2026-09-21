"""Frozen QD niche concentration on a selected candidate subset."""
from __future__ import annotations

import numpy as np


def niche_metrics(niches: np.ndarray) -> dict:
    counts = np.bincount(np.asarray(niches, dtype=int), minlength=1000).astype(float)
    nonzero = counts[counts > 0]
    total = float(nonzero.sum()) if nonzero.size else 0.0
    if total <= 0:
        return {
            "occupied_niches": 0, "N_eff": 0.0, "entropy": 0.0, "gini": float("nan"),
            "largest_niche_share": float("nan"), "top5_niche_share": float("nan"),
            "top10_niche_share": float("nan"),
        }
    probabilities = nonzero / total
    shannon = float(-np.sum(probabilities * np.log(probabilities)))
    ordered = np.sort(counts)
    gini = float(2 * np.sum(np.arange(1, 1001) * ordered) / (1000 * total) - 1001 / 1000)
    return {
        "occupied_niches": int(len(nonzero)),
        "N_eff": float(np.exp(shannon)),
        "entropy": shannon,
        "gini": gini,
        "largest_niche_share": float(nonzero.max() / total),
        "top5_niche_share": float(np.sort(nonzero)[-5:].sum() / total),
        "top10_niche_share": float(np.sort(nonzero)[-10:].sum() / total),
    }


def set_jaccard(a, b) -> float:
    a = set(a)
    b = set(b)
    if not a and not b:
        return float("nan")
    return float(len(a & b) / len(a | b))


def topk_indices(scores: np.ndarray, hashes: np.ndarray, k: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    hashes = np.asarray(hashes)
    k = min(int(k), scores.size)
    return np.lexsort((hashes, -scores))[:k]
