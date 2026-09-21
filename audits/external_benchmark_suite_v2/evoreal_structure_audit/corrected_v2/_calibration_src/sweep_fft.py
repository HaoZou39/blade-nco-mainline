"""Sweep FFT parameterizations on author-labeled TSPLIB S1/S2/S3."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
OUT = SRC.parent
FFT_THRESH = 35.0
NN_THRESH = 0.5


def parse_type_file(path: Path, split: str) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = ast.literal_eval(line)
        xy = np.asarray(rec[2:], dtype=np.float64).reshape(-1, 2)
        rows.append({"name": str(rec[0]), "split": split, "N": int(xy.shape[0]), "xy": xy})
    return rows


def alg1_normalize(xy: np.ndarray) -> np.ndarray:
    x = np.asarray(xy, dtype=np.float64)
    min_val = x.min(axis=0)
    max_val = x.max(axis=0)
    max_diff = float(np.max(max_val - min_val))
    if max_diff <= 0:
        return np.zeros_like(x)
    return (x - min_val) / max_diff


def nn_stats(xy: np.ndarray, ddof: int) -> dict:
    pts = alg1_normalize(xy)
    n = pts.shape[0]
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    d_nn = d.min(axis=1)
    mean = float(d_nn.mean())
    std = float(d_nn.std(ddof=ddof))
    ratio = std / mean if mean > 0 else float("nan")
    return {"nn_mean": mean, "nn_std": std, "nn_ratio": ratio}


def density_map(pts: np.ndarray, grid: int, hist_norm: str) -> np.ndarray:
    H, _, _ = np.histogram2d(
        pts[:, 0], pts[:, 1], bins=grid, range=[[0.0, 1.0], [0.0, 1.0]]
    )
    H = H.astype(np.float64)
    if hist_norm == "counts":
        return H
    if hist_norm == "occupancy":
        return (H > 0).astype(np.float64)
    if hist_norm == "prob":
        s = H.sum()
        return H / s if s > 0 else H
    if hist_norm == "density":
        # numpy density=True equivalent
        area = (1.0 / grid) ** 2
        s = H.sum()
        return H / (s * area) if s > 0 else H
    if hist_norm == "max":
        m = H.max()
        return H / m if m > 0 else H
    raise ValueError(hist_norm)


def fft_energy(xy: np.ndarray, grid: int, hist_norm: str, region: str, energy: str) -> float:
    pts = alg1_normalize(xy)
    H = density_map(pts, grid, hist_norm)
    F = np.fft.fft2(H)
    mag2 = np.abs(F) ** 2
    mag = np.abs(F)
    # DC is [0,0] before shift
    mag2[0, 0] = 0.0
    mag[0, 0] = 0.0
    shifted2 = np.fft.fftshift(mag2)
    shifted = np.fft.fftshift(mag)
    cy = cx = grid // 2
    if region == "all_except_dc":
        mask = np.ones((grid, grid), dtype=bool)
        mask[cy, cx] = False
    elif region.startswith("lowfrac_"):
        frac = float(region.split("_")[1])
        half = max(1, int(round(grid * frac / 2)))
        mask = np.zeros((grid, grid), dtype=bool)
        mask[cy - half: cy + half, cx - half: cx + half] = True
        mask[cy, cx] = False
    elif region.startswith("lowpix_"):
        half = int(region.split("_")[1])
        mask = np.zeros((grid, grid), dtype=bool)
        mask[cy - half: cy + half, cx - half: cx + half] = True
        mask[cy, cx] = False
    elif region.startswith("annulus_"):
        r_in, r_out = [int(x) for x in region.split("_")[1:]]
        yy, xx = np.ogrid[:grid, :grid]
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        mask = (r > r_in) & (r <= r_out)
    else:
        raise ValueError(region)
    if mask.sum() == 0:
        return float("nan")
    if energy == "mean_abs2":
        return float(shifted2[mask].mean())
    if energy == "mean_abs":
        return float(shifted[mask].mean())
    if energy == "sum_abs2":
        return float(shifted2[mask].sum())
    if energy == "rms_abs":
        return float(np.sqrt(shifted2[mask].mean()))
    if energy == "frac_non_dc_pct":
        spec2 = np.abs(np.fft.fft2(H)) ** 2
        total = float(spec2.sum())
        dc = float(spec2[0, 0])
        return 100.0 * (1.0 - dc / total) if total > 0 else 0.0
    if energy == "mean_abs2_over_dc":
        dc = float(np.abs(np.fft.fft2(H)[0, 0]) ** 2)
        return float(shifted2[mask].mean() / dc) if dc > 0 else 0.0
    raise ValueError(energy)


def score_fft(values_s1, values_non):
    s1 = np.asarray(values_s1, dtype=np.float64)
    non = np.asarray(values_non, dtype=np.float64)
    if np.any(~np.isfinite(s1)) or np.any(~np.isfinite(non)):
        return None
    med_all = float(np.median(np.concatenate([s1, non])))
    # reject the old 16000-scale total-energy collapse
    if np.median(np.abs(np.concatenate([s1, non]) - 16284)) < 50:
        return None
    s1_gt = float((s1 > FFT_THRESH).mean())
    non_le = float((non <= FFT_THRESH).mean())
    direction = float(np.mean(s1) > np.mean(non))
    span = float(np.percentile(np.concatenate([s1, non]), 95) - np.percentile(np.concatenate([s1, non]), 5))
    # 35 should sit inside the observed range
    lo, hi = float(np.min(np.concatenate([s1, non]))), float(np.max(np.concatenate([s1, non])))
    thresh_inside = lo < FFT_THRESH < hi
    return {
        "s1_gt35": s1_gt,
        "non_le35": non_le,
        "sep_score": s1_gt + non_le,
        "s1_mean": float(np.mean(s1)),
        "s1_median": float(np.median(s1)),
        "non_mean": float(np.mean(non)),
        "non_median": float(np.median(non)),
        "all_median": med_all,
        "min": lo,
        "max": hi,
        "p05": float(np.percentile(np.concatenate([s1, non]), 5)),
        "p95": float(np.percentile(np.concatenate([s1, non]), 95)),
        "span_p05_p95": span,
        "direction_s1_gt_non": bool(direction),
        "thresh_inside_range": bool(thresh_inside),
    }


def main():
    inst = []
    for split in ["S1", "S2", "S3"]:
        inst.extend(parse_type_file(SRC / f"tsplib_type{split}.txt", split))
    print("loaded", len(inst), "author validation instances")

    # NN-ratio first
    nn_rows = []
    for rec in inst:
        for ddof in (0, 1):
            st = nn_stats(rec["xy"], ddof)
            nn_rows.append({**{k: rec[k] for k in ("name", "split", "N")}, "ddof": ddof, **st})
    nn_df = pd.DataFrame(nn_rows)
    nn_df.to_csv(OUT / "calibration_nn_author_tsplib.csv", index=False)
    print("\n=== NN-ratio on author labels ===")
    for ddof in (0, 1):
        sub = nn_df[nn_df.ddof == ddof]
        print(f"\nddof={ddof}")
        for split in ["S1", "S2", "S3"]:
            v = sub.loc[sub.split == split, "nn_ratio"]
            print(
                f"  {split}: n={len(v)} mean={v.mean():.3f} median={v.median():.3f} "
                f"share<0.5={(v<0.5).mean():.3f} share>=0.5={(v>=0.5).mean():.3f} "
                f"min={v.min():.3f} max={v.max():.3f}"
            )

    grids = [8, 16, 32, 64]
    hist_norms = ["counts", "occupancy", "prob", "density", "max"]
    regions = [
        "all_except_dc",
        "lowfrac_0.25",
        "lowfrac_0.5",
        "lowpix_2",
        "lowpix_4",
        "annulus_0_2",
        "annulus_0_4",
        "annulus_1_4",
    ]
    energies = ["mean_abs2", "mean_abs", "sum_abs2", "rms_abs", "frac_non_dc_pct"]

    results = []
    for grid in grids:
        for hist_norm in hist_norms:
            for region in regions:
                if region.startswith("lowpix_") and int(region.split("_")[1]) * 2 > grid:
                    continue
                if region.startswith("annulus_"):
                    r_out = int(region.split("_")[-1])
                    if r_out * 2 > grid:
                        continue
                for energy in energies:
                    s1_vals, non_vals, s2_vals, s3_vals = [], [], [], []
                    bad = False
                    for rec in inst:
                        try:
                            val = fft_energy(rec["xy"], grid, hist_norm, region, energy)
                        except Exception:
                            bad = True
                            break
                        if rec["split"] == "S1":
                            s1_vals.append(val)
                        else:
                            non_vals.append(val)
                        if rec["split"] == "S2":
                            s2_vals.append(val)
                        if rec["split"] == "S3":
                            s3_vals.append(val)
                    if bad:
                        continue
                    scored = score_fft(s1_vals, non_vals)
                    if scored is None:
                        continue
                    results.append({
                        "grid": grid,
                        "hist_norm": hist_norm,
                        "region": region,
                        "energy": energy,
                        "s2_mean": float(np.mean(s2_vals)),
                        "s3_mean": float(np.mean(s3_vals)),
                        "s2_gt35": float((np.asarray(s2_vals) > FFT_THRESH).mean()),
                        "s3_gt35": float((np.asarray(s3_vals) > FFT_THRESH).mean()),
                        **scored,
                    })

    rdf = pd.DataFrame(results)
    rdf.to_csv(OUT / "calibration_fft_sweep.csv", index=False)
    rdf = rdf.sort_values(["sep_score", "direction_s1_gt_non", "thresh_inside_range"], ascending=[False, False, False])
    print("\n=== TOP FFT configs by S1>35 + non-S1<=35 ===")
    cols = ["grid", "hist_norm", "region", "energy", "sep_score", "s1_gt35", "non_le35",
            "s1_median", "non_median", "s1_mean", "non_mean", "min", "max",
            "direction_s1_gt_non", "thresh_inside_range"]
    print(rdf[cols].head(30).to_string(index=False))

    # also show configs where 35 is inside range AND direction holds, even if sep is imperfect
    cand = rdf[(rdf.thresh_inside_range) & (rdf.direction_s1_gt_non) & (rdf.s1_median > rdf.non_median)]
    print("\n=== direction-correct, threshold-inside, S1 median > non median (top 20 by sep) ===")
    print(cand[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
