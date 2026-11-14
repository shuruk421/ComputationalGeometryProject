"""
Small-multiple maps of the consenting MEB of Chicago crimes on map tiles.

One panel per consent threshold: consenting vs non-consenting incidents, the
consenting-only MEB (a real geographic disk: the plot is in local km, and the
Web-Mercator basemap is resampled row by row into that frame), its support
points, the all-points MEB as a dashed reference, and a label with the
threshold, % consenting and the consent requests of both algorithms.

The panels draw exactly the benchmark sample for seed 0, so the request counts
come from results/benchmark_runs.json (recomputed if that file is missing).
"""

import argparse
import json
import math
import os
import sys
import urllib.request

import numpy as np
from PIL import Image

import consent as C
import crime_data as cd
import viz_style as V

V.apply()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
TILE_ZOOM = 11


# --------------------------------------------------------------------------
# Tiles (CARTO Positron, cached in tile_cache/)
# --------------------------------------------------------------------------

def lat_lng_to_tile(lat, lng, zoom):
    """Convert latitude/longitude to OSM tile coordinates."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lng + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return xtile, ytile


def tile_to_lat_lng(xtile, ytile, zoom):
    """Convert OSM tile coordinates back to latitude/longitude."""
    n = 2.0 ** zoom
    lng = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * ytile / n)))
    lat = math.degrees(lat_rad)
    return lat, lng


def fetch_and_stitch_tiles(lat_min, lat_max, lng_min, lng_max, zoom=TILE_ZOOM):
    """Fetch map tiles covering the bounding box and stitch them together.

    Returns (image, [lng_left, lng_right, lat_bottom, lat_top])."""
    xtile_min, ytile_min = lat_lng_to_tile(lat_max, lng_min, zoom)
    xtile_max, ytile_max = lat_lng_to_tile(lat_min, lng_max, zoom)

    cache_dir = os.path.join(HERE, "tile_cache")
    os.makedirs(cache_dir, exist_ok=True)

    tile_size = 256
    width = (xtile_max - xtile_min + 1) * tile_size
    height = (ytile_max - ytile_min + 1) * tile_size
    stitched_img = Image.new("RGBA", (width, height), (245, 245, 243, 255))

    for x in range(xtile_min, xtile_max + 1):
        for y in range(ytile_min, ytile_max + 1):
            tile_file = os.path.join(cache_dir, f"light_{zoom}_{x}_{y}.png")
            if not os.path.exists(tile_file):
                url = f"https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{zoom}/{x}/{y}.png"
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ChicagoCrimeMEBPlotter/1.0"}
                try:
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req) as response:
                        with open(tile_file, "wb") as f:
                            f.write(response.read())
                except Exception as e:
                    print(f"  Failed to fetch tile {zoom}/{x}/{y}: {e}")
                    continue
            try:
                tile_img = Image.open(tile_file).convert("RGBA")
                stitched_img.paste(tile_img, ((x - xtile_min) * tile_size, (y - ytile_min) * tile_size))
            except Exception as e:
                print(f"  Failed to process tile {tile_file}: {e}")

    lat_top, lng_left = tile_to_lat_lng(xtile_min, ytile_min, zoom)
    lat_bottom, lng_right = tile_to_lat_lng(xtile_max + 1, ytile_max + 1, zoom)
    return stitched_img, [lng_left, lng_right, lat_bottom, lat_top]


def _merc(lat):
    return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def basemap_km(proj, xlim, ylim, zoom=TILE_ZOOM, fade=0.25):
    """Basemap resampled into the local km frame. Returns (rgb array, extent)."""
    lat_lo, lng_lo = proj.to_latlng(xlim[0], ylim[0])
    lat_hi, lng_hi = proj.to_latlng(xlim[1], ylim[1])
    img, (lng_l, lng_r, lat_b, lat_t) = fetch_and_stitch_tiles(lat_lo, lat_hi, lng_lo, lng_hi, zoom)
    src = np.asarray(img.convert("RGB")).astype(np.float32)
    h, w = src.shape[:2]
    out_w = max(2, int(round(w * (lng_hi - lng_lo) / (lng_r - lng_l))))
    m_t, m_b = _merc(lat_t), _merc(lat_b)
    out_h = max(2, int(round(h * (_merc(lat_hi) - _merc(lat_lo)) / (m_t - m_b))))
    xs = xlim[0] + (np.arange(out_w) + 0.5) / out_w * (xlim[1] - xlim[0])
    ys = ylim[1] - (np.arange(out_h) + 0.5) / out_h * (ylim[1] - ylim[0])
    cols = [(proj.to_latlng(x, 0.0)[1] - lng_l) / (lng_r - lng_l) * w for x in xs]
    rows = [(m_t - _merc(proj.to_latlng(0.0, y)[0])) / (m_t - m_b) * h for y in ys]
    ci = np.clip(np.array(cols).astype(int), 0, w - 1)
    ri = np.clip(np.array(rows).astype(int), 0, h - 1)
    out = src[ri][:, ci]
    out = out * (1 - fade) + 255.0 * fade  # quieten the basemap so the data leads
    return out.astype(np.uint8), [xlim[0], xlim[1], ylim[0], ylim[1]]


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------

def load_seed0_calls(exp_id):
    """{(cutoff, algorithm): calls} for seed 0 of a sweep, if results exist."""
    path = os.path.join(RESULTS, "benchmark_runs.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        runs = json.load(f)["runs"]
    return {(r["cutoff"], r["algorithm"]): r["calls"]
            for r in runs if r["experiment"] == exp_id and r["seed"] == 0}


def pick_panels(cutoffs, k):
    if len(cutoffs) <= k:
        return list(cutoffs)
    idx = sorted({int(round(i * (len(cutoffs) - 1) / (k - 1))) for i in range(k)})
    return [cutoffs[i] for i in idx]


def draw_panel(ax, base, coords, flags, gt_c, gt_r, all_c, all_r, title, lines):
    img, extent = base
    ax.imshow(img, extent=extent, origin="upper", interpolation="bilinear", zorder=0)
    xs_n = [p[0] for p, f in zip(coords, flags) if not f]
    ys_n = [p[1] for p, f in zip(coords, flags) if not f]
    xs_c = [p[0] for p, f in zip(coords, flags) if f]
    ys_c = [p[1] for p, f in zip(coords, flags) if f]
    ax.scatter(xs_n, ys_n, s=2.2, c=V.NONCONSENT, alpha=0.55, linewidths=0, zorder=2)
    ax.scatter(xs_c, ys_c, s=3.2, c=V.CONSENT, alpha=0.9, linewidths=0, zorder=3)
    ax.add_patch(Circle(all_c, all_r, fill=False, ec=V.INK2, lw=1.0, ls=(0, (4, 3)), zorder=4))
    if gt_c is not None:
        ax.add_patch(Circle(gt_c, gt_r, fill=False, ec=V.INK, lw=1.8, zorder=5))
        ax.plot([gt_c[0]], [gt_c[1]], marker="+", ms=7, mew=1.6, color=V.INK, zorder=6)
        sup = C.support_points([p for p, f in zip(coords, flags) if f], gt_c, gt_r, 1e-7)
        ax.scatter([p[0] for p in sup], [p[1] for p in sup], s=46, c=V.INK, edgecolors=V.SURFACE,
                   linewidths=1.6, zorder=7)
    # Labels sit above the map so they never hide data (O'Hare, in the NW
    # corner, often carries the support points).
    stat, req = lines
    ax.set_title(title, loc="left", pad=17, fontsize=11, fontweight="semibold", color=V.INK)
    ax.set_title(stat, loc="right", pad=17, fontsize=9.5, fontweight="normal", color=V.INK2)
    t = ax.text(0, 1.012, "requests ", transform=ax.transAxes, fontsize=9, color=V.INK2,
                va="bottom", ha="left")
    for algo in C.ALGORITHMS:
        t = ax.annotate("●", xy=(1, 0), xycoords=t, va="bottom", ha="left", fontsize=9,
                        color=V.ALGO_COLOR[algo])
        t = ax.annotate(f" {algo} {req[algo]:,}    ", xy=(1, 0), xycoords=t, va="bottom", ha="left",
                        fontsize=9, color=V.INK2)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_color(V.AXIS)
        s.set_linewidth(0.8)


def scale_bar(ax, extent, km=10):
    x0 = extent[1] - (extent[1] - extent[0]) * 0.05 - km
    y0 = extent[2] + (extent[3] - extent[2]) * 0.05
    ax.plot([x0, x0 + km], [y0, y0], color=V.INK, lw=2, solid_capstyle="butt", zorder=9)
    ax.text(x0 + km / 2, y0 + 0.8, f"{km} km", ha="center", va="bottom", fontsize=8.5, color=V.INK, zorder=9)


def plot_sweep_maps(records, proj, family, exp_id, n, targets, out_path, ncols=3, max_panels=9,
                    title=None, subtitle=None, seed=0):
    fam = C.THRESHOLD_FAMILIES[family]
    cutoffs = []
    for t in targets:
        c = fam.cutoff_for_fraction(records, t)
        if c not in cutoffs:
            cutoffs.append(c)
    cutoffs = pick_panels(cutoffs, max_panels)
    rows, coords = cd.draw_points(records, proj, n, seed=seed)
    all_c, all_r = C.exact_meb(coords)
    pad = 1.06
    xlim = (all_c[0] - all_r * pad, all_c[0] + all_r * pad)
    ylim = (all_c[1] - all_r * pad, all_c[1] + all_r * pad)
    base = basemap_km(proj, xlim, ylim)
    calls = load_seed0_calls(exp_id)

    nrows = int(math.ceil(len(cutoffs) / ncols))
    fig_h = 4.75 * nrows + 1.45
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, fig_h))
    axes = np.atleast_1d(axes).ravel()
    for ax, cut in zip(axes, cutoffs):
        rule = fam.rule(cut)
        flags = [bool(rule(r)) for r in rows]
        cons = [p for p, f in zip(coords, flags) if f]
        gt_c, gt_r = C.exact_meb(cons, seed=seed) if cons else (None, 0.0)
        pts = list(zip(coords, flags))
        req = {}
        for a in C.ALGORITHMS:
            if (cut, a) in calls:
                req[a] = calls[(cut, a)]
            else:
                req[a] = C.run_algorithm(a, pts, seed=1000 + seed)["calls"]
        pct = 100.0 * len(cons) / len(rows)
        stat = f"{pct:.0f}% consenting · r = {gt_r:.1f} km"
        panel_title = fam.label(cut).replace(">=", "≥").replace("<=", "≤")
        draw_panel(ax, base, coords, flags, gt_c, gt_r, all_c, all_r, panel_title, (stat, req))
    for ax in axes[len(cutoffs):]:
        ax.axis("off")
    scale_bar(axes[len(cutoffs) - 1], base[1])

    handles = [
        Line2D([], [], ls="none", marker="o", ms=6, mfc=V.CONSENT, mec="none", label="Consenting incident"),
        Line2D([], [], ls="none", marker="o", ms=6, mfc=V.NONCONSENT, mec="none", label="Non-consenting incident"),
        Line2D([], [], color=V.INK, lw=1.8, label="MEB of consenting incidents"),
        Line2D([], [], ls="none", marker="o", ms=8, mfc=V.INK, mec=V.SURFACE, mew=1.5, label="Support points"),
        Line2D([], [], color=V.INK2, lw=1.0, ls=(0, (4, 3)), label="MEB of all incidents"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.012), fontsize=9.5,
               handlelength=2.2, columnspacing=1.6)
    fig.text(0.02, 1 - 0.18 / fig_h, title or f"Consenting MEB by {fam.column} threshold", ha="left", va="top",
             fontsize=15, fontweight="semibold", color=V.INK)
    fig.text(0.02, 1 - 0.55 / fig_h, subtitle or "", ha="left", va="top", fontsize=10.5, color=V.INK2)
    fig.text(0.98, 0.003, "Basemap © OpenStreetMap contributors © CARTO", ha="right", va="bottom",
             fontsize=7.5, color=V.MUTED)
    fig.subplots_adjust(left=0.02, right=0.98, top=1 - 1.25 / fig_h, bottom=0.6 / fig_h, wspace=0.04,
                        hspace=0.47 / 4.3)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[maps] wrote {out_path}")
    return out_path


def make_all(records, proj, out_dir=HERE):

    n = 2000
    outs = []
    outs.append(plot_sweep_maps(
        records, proj, "DISTRICT", "district_n2000", n,
        (0.05, 0.10, 0.20, 0.40, 0.50, 0.60, 0.70, 0.80, 0.95),  # subset of benchmark_crimes.SWEEP_TARGETS
        os.path.join(out_dir, "chicago_district_rollout_maps.png"),
        title="Consent rolled out police district by district: the consenting MEB moves and grows",
        subtitle=(f"Chicago crimes 2002–2020, {n:,} distinct incident locations sampled across the whole file. "
                  "Consent = record's police DISTRICT ≤ k. Requests = unique consent (oracle) calls.")))
    outs.append(plot_sweep_maps(
        records, proj, "TIMESTAMP", "timestamp_n2000", n, (0.05, 0.20, 0.40, 0.60, 0.80, 0.95),
        os.path.join(out_dir, "chicago_timestamp_cutoff_maps.png"), max_panels=6,
        title="Consent by policy date is spatially random: the MEB barely moves",
        subtitle=(f"Same {n:,} locations. Consent = incident reported on or after the cutoff date. "
                  "Requests = unique consent (oracle) calls. At 5% only ~100 points consent, "
                  "so the city's edges are simply missed.")))
    return outs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", default=cd.DEFAULT_CSV)
    ap.add_argument("--out-dir", default=HERE)
    a = ap.parse_args()
    recs, meta, proj = cd.load(a.csv)
    make_all(recs, proj, a.out_dir)
