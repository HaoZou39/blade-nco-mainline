"""Presentation-only schematics for PPT Step 2 (v4).

LEFT  = discrete structural coverage / QD niches
RIGHT = continuous solver-relative hardness field

Both views read the frozen candidate CSV. Coordinates, region assignment,
and hardness values are never regenerated.

Does not touch experiment code or the main training pipeline.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from PIL import Image
from scipy.ndimage import gaussian_filter, label
from scipy.spatial import Voronoi, cKDTree
from scipy.stats import gaussian_kde

SEED = 2026
COMMON_FIGSIZE = (4.6, 2.5)
COMMON_DPI = 300
POINT_SIZE_LEFT = 10.4
POINT_SIZE_RIGHT = 12.0
REP_SIZE = 24.5
GRID_N = 720

N_OCCUPIED_CENTROIDS = 25
N_EMPTY_CENTROIDS = 4
N_REPRESENTATIVES = 7

POINT_COLOR_LEFT = "#7F93A3"
POINT_COLOR_RIGHT = "#526A7C"
OCCUPIED_FILL = "#DDEFE8"
NICHE_EDGE = "#9BAEBC"
REP_EDGE = "#1F6970"

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

CSV_PATH = ASSETS / "step2_candidate_coordinates.csv"
FIGA_PNG = ASSETS / "step2_structural_coverage_v4.png"
FIGA_SVG = ASSETS / "step2_structural_coverage_v4.svg"
FIGB_PNG = ASSETS / "step2_predicted_hardness_v4.png"
FIGB_SVG = ASSETS / "step2_predicted_hardness_v4.svg"
PREVIEW_PNG = ASSETS / "step2_two_views_preview_v4.png"


def hardness_raw(x, y):
    # Hotspots deliberately do NOT coincide with structural-region centers.
    h = 1.00 * np.exp(-((x - 0.98) ** 2 + (y - 1.14) ** 2) / 0.28)
    h += 0.70 * np.exp(-((x - 2.18) ** 2 + (y - 1.58) ** 2) / 0.36)
    h += 0.66 * np.exp(-((x - 3.02) ** 2 + (y - 0.52) ** 2) / 0.26)
    h += 0.13 * np.exp(-((x - 1.90) ** 2 / 4.2 + (y - 0.98) ** 2 / 2.1))
    h += 0.040 * (1.0 + np.sin(0.95 * x + 0.55 * y))
    return h


def shared_limits(x, y):
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    dx, dy = xmax - xmin, ymax - ymin
    pad_x = 0.20 + 0.035 * dx
    pad_y = 0.16 + 0.035 * dy
    xmin, xmax = xmin - pad_x, xmax + pad_x
    ymin, ymax = ymin - pad_y, ymax + pad_y
    target_aspect = COMMON_FIGSIZE[0] / COMMON_FIGSIZE[1]
    dx, dy = xmax - xmin, ymax - ymin
    data_aspect = dx / dy
    if data_aspect < target_aspect:
        extra = 0.5 * (target_aspect * dy - dx)
        xmin -= extra
        xmax += extra
    else:
        extra = 0.5 * (dx / target_aspect - dy)
        ymin -= extra
        ymax += extra
    return (xmin, xmax), (ymin, ymax)


def make_grid(xlim, ylim):
    nx = GRID_N
    ny = int(round(GRID_N * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])))
    gx = np.linspace(xlim[0], xlim[1], nx)
    gy = np.linspace(ylim[0], ylim[1], ny)
    return np.meshgrid(gx, gy)


def make_canvas(xlim, ylim):
    fig = plt.figure(figsize=COMMON_FIGSIZE, dpi=COMMON_DPI, facecolor="none")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor="none")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig, ax


def save_figure(fig, png_path, svg_path):
    kw = dict(
        dpi=COMMON_DPI,
        transparent=True,
        facecolor="none",
        edgecolor="none",
        bbox_inches=None,
        pad_inches=0,
    )
    fig.savefig(png_path, **kw)
    fig.savefig(svg_path, **kw)
    plt.close(fig)


def kde_on_grid(x, y, xx, yy, bw):
    kde = gaussian_kde(np.vstack([x, y]), bw_method=bw)
    return kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)


def large_components(mask, min_frac=0.03, min_pixels=180):
    labeled, nlab = label(mask)
    if nlab == 0:
        return mask
    counts = np.bincount(labeled.ravel())
    keep = np.zeros_like(mask, dtype=bool)
    total = max(int(counts[1:].sum()), 1)
    for i in range(1, nlab + 1):
        if counts[i] >= min_frac * total or counts[i] >= min_pixels:
            keep |= labeled == i
    return keep


def lloyd_centroids(xy, k, rng, n_iter=18):
    n = len(xy)
    cents = np.empty((k, 2), dtype=float)
    cents[0] = xy[int(rng.integers(n))]
    for i in range(1, k):
        d2 = ((xy[:, None, :] - cents[None, :i, :]) ** 2).sum(axis=2).min(axis=1)
        p = d2 / max(float(d2.sum()), 1e-12)
        cents[i] = xy[int(rng.choice(n, p=p))]
    for _ in range(n_iter):
        d2 = ((xy[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
        lab = d2.argmin(axis=1)
        new = cents.copy()
        for i in range(k):
            m = lab == i
            if np.any(m):
                new[i] = xy[m].mean(axis=0)
        cents = new
    return cents


def peripheral_empty_centroids(xy, xlim, ylim, n, rng):
    """Place unoccupied niches only near the frame, never in the interior hole."""
    tree = cKDTree(xy)
    xs = np.linspace(xlim[0], xlim[1], 8)
    ys = np.linspace(ylim[0], ylim[1], 5)
    inset_x = 0.07 * (xlim[1] - xlim[0])
    inset_y = 0.08 * (ylim[1] - ylim[0])
    frame = []
    for xv in xs:
        frame.append([xv, ylim[0] + inset_y])
        frame.append([xv, ylim[1] - inset_y])
    for yv in ys[1:-1]:
        frame.append([xlim[0] + inset_x, yv])
        frame.append([xlim[1] - inset_x, yv])
    frame = np.asarray(frame, dtype=float)
    dist = tree.query(frame, k=1)[0]
    order = np.argsort(-dist)
    picked = []
    min_sep = 0.28 * max(xlim[1] - xlim[0], ylim[1] - ylim[0]) / max(n, 1)
    for i in order:
        if dist[i] < 0.22 * max(xlim[1] - xlim[0], 1e-6):
            continue
        p = frame[i]
        if picked and np.min(np.linalg.norm(np.asarray(picked) - p, axis=1)) < min_sep:
            continue
        picked.append(p)
        if len(picked) >= n:
            break
    return np.asarray(picked, dtype=float) if picked else np.zeros((0, 2))


def clip_poly_to_box(poly, xlim, ylim):
    xmin, xmax = xlim
    ymin, ymax = ylim

    def clip(pts, inside, intersect):
        if len(pts) == 0:
            return pts
        out = []
        s = pts[-1]
        for e in pts:
            ein, sin = inside(e), inside(s)
            if ein:
                if not sin:
                    out.append(intersect(s, e))
                out.append(e)
            elif sin:
                out.append(intersect(s, e))
            s = e
        return out

    def lerp(s, e, t):
        return (s[0] + t * (e[0] - s[0]), s[1] + t * (e[1] - s[1]))

    pts = [tuple(p) for p in poly]
    pts = clip(pts, lambda p: p[0] >= xmin, lambda s, e: lerp(s, e, (xmin - s[0]) / (e[0] - s[0] + 1e-15)))
    pts = clip(pts, lambda p: p[0] <= xmax, lambda s, e: lerp(s, e, (xmax - s[0]) / (e[0] - s[0] + 1e-15)))
    pts = clip(pts, lambda p: p[1] >= ymin, lambda s, e: lerp(s, e, (ymin - s[1]) / (e[1] - s[1] + 1e-15)))
    pts = clip(pts, lambda p: p[1] <= ymax, lambda s, e: lerp(s, e, (ymax - s[1]) / (e[1] - s[1] + 1e-15)))
    if len(pts) < 3:
        return None
    return np.asarray(pts, dtype=float)


def voronoi_cells(centroids, xlim, ylim):
    xmid = 0.5 * (xlim[0] + xlim[1])
    ymid = 0.5 * (ylim[0] + ylim[1])
    span = 18.0 * max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    dummy = np.array(
        [
            [xmid - span, ymid - span],
            [xmid - span, ymid + span],
            [xmid + span, ymid - span],
            [xmid + span, ymid + span],
            [xmid - span, ymid],
            [xmid + span, ymid],
            [xmid, ymid - span],
            [xmid, ymid + span],
        ]
    )
    vor = Voronoi(np.vstack([centroids, dummy]))
    # Clip to a box slightly larger than the axes so internal ridges meet the
    # canvas at an angle instead of leaving extra edges glued to the border.
    grow_x = 0.08 * (xlim[1] - xlim[0])
    grow_y = 0.08 * (ylim[1] - ylim[0])
    clip_xlim = (xlim[0] - grow_x, xlim[1] + grow_x)
    clip_ylim = (ylim[0] - grow_y, ylim[1] + grow_y)
    polys = []
    for i in range(len(centroids)):
        region_idx = vor.point_region[i]
        verts = vor.regions[region_idx]
        if not verts or -1 in verts:
            polys.append(None)
            continue
        poly = clip_poly_to_box(vor.vertices[verts], clip_xlim, clip_ylim)
        polys.append(poly)
    return polys


def farthest_point_indices(pts, k):
    n = len(pts)
    k = min(k, n)
    chosen = [int(np.argmin(np.sum((pts - pts.mean(axis=0)) ** 2, axis=1)))]
    for _ in range(k - 1):
        dmin = np.min(((pts[:, None, :] - pts[chosen][None, :, :]) ** 2).sum(axis=2), axis=1)
        dmin[np.array(chosen)] = -1.0
        chosen.append(int(np.argmax(dmin)))
    return np.asarray(chosen, dtype=int)


def hardness_cmap():
    return LinearSegmentedColormap.from_list(
        "pred_hardness_v3",
        [
            (0.00, "#9DBFD3"),
            (0.38, "#D5E6EE"),
            (0.54, "#F7F7F3"),
            (0.70, "#E39286"),
            (1.00, "#DD6A5D"),
        ],
        N=256,
    )


def manifold_support_alpha(x, y, xx, yy):
    dens = kde_on_grid(x, y, xx, yy, bw=0.19)
    dens = gaussian_filter(dens, sigma=1.35)
    dmax = float(dens.max()) + 1e-12
    xy = np.column_stack([x, y])
    tree = cKDTree(xy)
    spacing = float(np.median(tree.query(xy, k=2)[0][:, 1]))
    nn_dist = tree.query(np.column_stack([xx.ravel(), yy.ravel()]), k=1)[0].reshape(xx.shape)
    dens_term = np.clip((dens - 0.07 * dmax) / (0.28 * dmax), 0.0, 1.0)
    nn_term = np.clip(1.0 - nn_dist / (3.15 * spacing), 0.0, 1.0)
    alpha = np.sqrt(np.clip(dens_term, 0.0, 1.0) * np.clip(nn_term, 0.0, 1.0))
    alpha = gaussian_filter(alpha, sigma=4.2)
    keep = large_components(alpha > 0.055, min_frac=0.03, min_pixels=180)
    alpha = np.where(keep, alpha, 0.0)
    alpha = gaussian_filter(alpha, sigma=2.6)
    alpha = np.clip(alpha, 0.0, 1.0)
    if alpha.max() > 0:
        alpha = alpha / alpha.max()
    alpha = np.where(alpha < 0.035, 0.0, alpha)
    return alpha


def draw_fig_a(x, y, xlim, ylim, rng):
    fig, ax = make_canvas(xlim, ylim)
    xy = np.column_stack([x, y])

    occupied_cents = lloyd_centroids(xy, N_OCCUPIED_CENTROIDS, rng)
    empty_cents = peripheral_empty_centroids(xy, xlim, ylim, N_EMPTY_CENTROIDS, rng)
    centroids = np.vstack([occupied_cents, empty_cents]) if len(empty_cents) else occupied_cents

    assign = cKDTree(centroids).query(xy, k=1)[1]
    occupied_ids = np.unique(assign)
    polys = voronoi_cells(centroids, xlim, ylim)

    occ_polys = [polys[i] for i in occupied_ids if polys[i] is not None]
    if occ_polys:
        ax.add_collection(
            PolyCollection(
                occ_polys,
                facecolors=to_rgba(OCCUPIED_FILL, 0.24),
                edgecolors="none",
                zorder=1,
            )
        )

    for poly in polys:
        if poly is None:
            continue
        closed = np.vstack([poly, poly[:1]])
        ax.plot(
            closed[:, 0],
            closed[:, 1],
            color=NICHE_EDGE,
            lw=0.70,
            alpha=0.34,
            solid_joinstyle="round",
            solid_capstyle="round",
            zorder=2,
        )

    ax.scatter(
        x,
        y,
        s=POINT_SIZE_LEFT,
        facecolors=to_rgba(POINT_COLOR_LEFT, 0.52),
        edgecolors="none",
        linewidths=0,
        zorder=10,
    )

    occ_cent_xy = centroids[occupied_ids]
    fps = farthest_point_indices(occ_cent_xy, N_REPRESENTATIVES)
    rep_cent = occ_cent_xy[fps]
    rep_idx = []
    used = set()
    for c in rep_cent:
        d = np.linalg.norm(xy - c, axis=1)
        for idx in np.argsort(d):
            if int(idx) not in used:
                used.add(int(idx))
                rep_idx.append(int(idx))
                break
    rx, ry = x[rep_idx], y[rep_idx]
    ax.scatter(
        rx,
        ry,
        s=REP_SIZE,
        facecolors="white",
        edgecolors=REP_EDGE,
        linewidths=2.2,
        zorder=12,
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    display_xy = ax.transData.transform(xy)
    used_xlim = ax.get_xlim()
    used_ylim = ax.get_ylim()
    save_figure(fig, FIGA_PNG, FIGA_SVG)
    return display_xy, used_xlim, used_ylim, len(centroids), int(len(occupied_ids)), len(rep_idx)


def draw_fig_b(x, y, xlim, ylim):
    fig, ax = make_canvas(xlim, ylim)
    xx, yy = make_grid(xlim, ylim)

    h_grid = hardness_raw(xx, yy)
    h_grid = gaussian_filter(h_grid, sigma=3.4)
    h_pts = hardness_raw(x, y)
    hmin = float(min(h_grid.min(), h_pts.min()))
    hmax = float(max(h_grid.max(), h_pts.max()))
    h_grid_n = (h_grid - hmin) / (hmax - hmin + 1e-12)

    alpha = manifold_support_alpha(x, y, xx, yy)
    rgba = hardness_cmap()(h_grid_n)
    rgba[..., 3] = alpha * 0.94
    ax.imshow(
        rgba,
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        origin="lower",
        interpolation="bilinear",
        aspect="equal",
        zorder=1,
    )

    ax.scatter(
        x,
        y,
        s=POINT_SIZE_RIGHT,
        facecolors=to_rgba(POINT_COLOR_RIGHT, 0.60),
        edgecolors="none",
        linewidths=0,
        zorder=10,
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    display_xy = ax.transData.transform(np.column_stack([x, y]))
    used_xlim = ax.get_xlim()
    used_ylim = ax.get_ylim()
    save_figure(fig, FIGB_PNG, FIGB_SVG)
    return display_xy, used_xlim, used_ylim, float(h_pts.min()), float(h_pts.max())


def draw_preview():
    left = Image.open(FIGA_PNG).convert("RGBA")
    right = Image.open(FIGB_PNG).convert("RGBA")
    white_l = Image.new("RGBA", left.size, (255, 255, 255, 255))
    white_r = Image.new("RGBA", right.size, (255, 255, 255, 255))
    left = Image.alpha_composite(white_l, left)
    right = Image.alpha_composite(white_r, right)

    fig = plt.figure(figsize=(10.6, 3.35), dpi=220, facecolor="#F6F7F8")
    gs = fig.add_gridspec(
        2, 3, height_ratios=[0.22, 1.0], width_ratios=[1.0, 0.10, 1.0], hspace=0.02, wspace=0.04
    )
    ax_tl = fig.add_subplot(gs[0, 0])
    ax_tm = fig.add_subplot(gs[0, 1])
    ax_tr = fig.add_subplot(gs[0, 2])
    ax_bl = fig.add_subplot(gs[1, 0])
    ax_bm = fig.add_subplot(gs[1, 1])
    ax_br = fig.add_subplot(gs[1, 2])
    for ax in (ax_tl, ax_tm, ax_tr, ax_bl, ax_bm, ax_br):
        ax.axis("off")
        ax.set_facecolor("#F6F7F8")

    ax_tl.text(0.5, 0.72, "Structural diversity", ha="center", va="center", fontsize=12.5, color="#2C3A45", fontweight=500)
    ax_tl.text(0.5, 0.18, "Where to explore?", ha="center", va="center", fontsize=10.5, color="#6A7680", style="italic")
    ax_tr.text(0.5, 0.72, "Predicted hardness", ha="center", va="center", fontsize=12.5, color="#2C3A45", fontweight=500)
    ax_tr.text(0.5, 0.18, "What is hard now?", ha="center", va="center", fontsize=10.5, color="#6A7680", style="italic")
    ax_bm.text(0.5, 0.50, "+", ha="center", va="center", fontsize=28, color="#8A939B")

    ax_bl.imshow(left)
    ax_br.imshow(right)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.06)
    fig.savefig(PREVIEW_PNG, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def svg_ok(path):
    try:
        tree = ET.parse(path)
        return "svg" in tree.getroot().tag.lower()
    except Exception:
        return False


def png_info(path):
    with Image.open(path) as im:
        return im.size, im.mode, im.info.get("dpi")


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Frozen candidate CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    ids = df["candidate_id"].to_numpy(int)
    x = df["x"].to_numpy(float)
    y = df["y"].to_numpy(float)
    region_id = df["region_id"].to_numpy(int)
    common_xlim, common_ylim = shared_limits(x, y)
    rng = np.random.default_rng(SEED)

    xy_a, xlim_a, ylim_a, n_cells, n_occ, n_rep = draw_fig_a(x, y, common_xlim, common_ylim, rng)
    xy_b, xlim_b, ylim_b, hmin, hmax = draw_fig_b(x, y, common_xlim, common_ylim)
    draw_preview()

    df_again = pd.read_csv(CSV_PATH)
    same_ids = np.array_equal(ids, df_again["candidate_id"].to_numpy(int))
    same_xy_data = np.allclose(x, df_again["x"].to_numpy(float)) and np.allclose(y, df_again["y"].to_numpy(float))
    same_regions = np.array_equal(region_id, df_again["region_id"].to_numpy(int))
    same_px = np.allclose(xy_a, xy_b, atol=1e-6)
    same_xlim = np.allclose(xlim_a, xlim_b) and np.allclose(xlim_a, common_xlim)
    same_ylim = np.allclose(ylim_a, ylim_b) and np.allclose(ylim_a, common_ylim)
    size_a, mode_a, dpi_a = png_info(FIGA_PNG)
    size_b, mode_b, dpi_b = png_info(FIGB_PNG)
    expected = (int(round(COMMON_FIGSIZE[0] * COMMON_DPI)), int(round(COMMON_FIGSIZE[1] * COMMON_DPI)))
    same_canvas = size_a == size_b == expected
    inside = np.all((x >= common_xlim[0]) & (x <= common_xlim[1]) & (y >= common_ylim[0]) & (y <= common_ylim[1]))

    print("==== Step 2 schematic QC (v4) ====")
    print(f"candidate_count = {len(df)}")
    print(f"visible_niches = {n_cells}")
    print(f"occupied_visible_niches = {n_occ}")
    print(f"representative_count = {n_rep}")
    print(f"same_coordinates_with_hardness = {'PASS' if (same_xy_data and same_px) else 'FAIL'}")
    print(f"same_candidate_ids = {'PASS' if same_ids else 'FAIL'}")
    print(f"same_region_assignment = {'PASS' if same_regions else 'FAIL'}")
    print(f"same_xlim = {'PASS' if same_xlim else 'FAIL'}")
    print(f"same_ylim = {'PASS' if same_ylim else 'FAIL'}")
    print(f"same_canvas_size = {'PASS' if same_canvas else 'FAIL'}  {size_a} vs {size_b}")
    print(f"hardness_min = {hmin:.6f}")
    print(f"hardness_max = {hmax:.6f}")
    print(f"no_clipping = {'PASS' if inside else 'FAIL'}")
    print(f"svg_openable = {'PASS' if (svg_ok(FIGA_SVG) and svg_ok(FIGB_SVG)) else 'FAIL'}")
    print(f"png_transparent = {'PASS' if ('RGBA' in mode_a and 'RGBA' in mode_b) else 'FAIL'}")
    print(f"png_dpi = {dpi_a} / {dpi_b}")
    print("output PNG/SVG paths:")
    print(f"  {FIGA_PNG}")
    print(f"  {FIGA_SVG}")
    print(f"  {FIGB_PNG}")
    print(f"  {FIGB_SVG}")
    print(f"  {PREVIEW_PNG}")
    print(f"frozen CSV (not rewritten): {CSV_PATH}")


if __name__ == "__main__":
    main()
