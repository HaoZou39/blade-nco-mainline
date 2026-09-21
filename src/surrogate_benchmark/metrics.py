"""Acquisition metrics shared by S2-A and later feature/sample audits."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def k_q(n: int, q: float) -> int:
    return max(1, int(round(q * n)))


def enr(true_h: np.ndarray, pred: np.ndarray, q: float) -> float:
    true_h = np.asarray(true_h, dtype=float)
    pred = np.asarray(pred, dtype=float)
    k = k_q(true_h.size, q)
    idx = np.argsort(-pred)[:k]
    denom = float(np.mean(true_h))
    if denom == 0 or not np.isfinite(denom):
        return float("nan")
    return float(np.mean(true_h[idx]) / denom)


def top_recall(true_h: np.ndarray, pred: np.ndarray, q: float) -> float:
    n = true_h.size
    k = k_q(n, q)
    pred_idx = set(np.argsort(-pred)[:k].tolist())
    true_idx = set(np.argsort(-true_h)[:k].tolist())
    return float(len(pred_idx & true_idx) / k)


def capture_rate(model_enr: float, oracle_enr: float, random_mean: float, eps: float = 1e-8):
    available = oracle_enr - random_mean
    gain = model_enr - random_mean
    if not np.isfinite(available) or abs(available) < eps:
        return gain, available, "CAPTURE_RATE_UNDEFINED_LOW_HEADROOM"
    return gain, available, float(gain / available)


def recall_capture(recall: float, q: float) -> float:
    return float((recall - q) / (1.0 - q)) if (1.0 - q) != 0 else float("nan")


def safe_spearman(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.unique(x[mask]).size < 2 or np.unique(y[mask]).size < 2:
        return float("nan")
    return float(spearmanr(x[mask], y[mask]).statistic)


def tail_ranking(pred: np.ndarray, true_h: np.ndarray) -> dict:
    pred = np.asarray(pred, dtype=float)
    true_h = np.asarray(true_h, dtype=float)
    n = pred.size
    k20 = k_q(n, 0.20)
    k10 = k_q(n, 0.10)
    k5 = k_q(n, 0.05)
    pred_order = np.argsort(-pred)
    true_order = np.argsort(-true_h)
    return {
        "n": int(n),
        "n_tail20": k20,
        "rho_tail20": safe_spearman(pred[pred_order[:k20]], true_h[pred_order[:k20]]),
        "n_tail10": k10,
        "rho_tail10": safe_spearman(pred[pred_order[:k10]], true_h[pred_order[:k10]]),
        "n_tail5": k5,
        "rho_tail5": safe_spearman(pred[pred_order[:k5]], true_h[pred_order[:k5]]),
        "n_true_tail10": k10,
        "true_tail_rho10": safe_spearman(pred[true_order[:k10]], true_h[true_order[:k10]]),
    }


def decile_rows(pred: np.ndarray, true_h: np.ndarray) -> list[dict]:
    pred = np.asarray(pred, dtype=float)
    true_h = np.asarray(true_h, dtype=float)
    order = np.argsort(pred)
    bins = np.array_split(order, 10)
    means = []
    rows = []
    panel_mean = float(np.mean(true_h))
    for d, bin_idx in enumerate(bins, start=1):
        true_bin = true_h[bin_idx]
        pred_bin = pred[bin_idx]
        means.append(float(np.mean(true_bin)))
        rows.append({
            "decile": d,
            "n": int(len(bin_idx)),
            "pred_mean": float(np.mean(pred_bin)),
            "true_h_mean": float(np.mean(true_bin)),
            "true_h_median": float(np.median(true_bin)),
        })
    top_over_full = means[-1] / panel_mean if panel_mean else float("nan")
    top_over_bottom = means[-1] / means[0] if means[0] else float("nan")
    for row in rows:
        row["top_decile_over_full_mean_h"] = float(top_over_full)
        row["top_decile_over_bottom_mean_h"] = float(top_over_bottom)
    return rows


def set_jaccard(a, b) -> float:
    a = set(a)
    b = set(b)
    if not a and not b:
        return float("nan")
    return float(len(a & b) / len(a | b))


def paired_bootstrap_ci(deltas: np.ndarray, seed: int = 2026, repeats: int = 10000) -> dict:
    deltas = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(deltas, size=(repeats, deltas.size), replace=True).mean(axis=1)
    return {
        "mean": float(deltas.mean()),
        "median": float(np.median(deltas)),
        "n_positive": int(np.sum(deltas > 0)),
        "n_negative": int(np.sum(deltas < 0)),
        "n_zero": int(np.sum(deltas == 0)),
        "ci95_low": float(np.percentile(draws, 2.5)),
        "ci95_high": float(np.percentile(draws, 97.5)),
        "bootstrap_seed": seed,
        "bootstrap_repeats": repeats,
    }
