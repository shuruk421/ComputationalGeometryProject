"""
Figures for the Amsterdam consent study.

Colour roles (fixed across every figure; validated with the dataviz palette
checker for all-pairs use on a light surface):
    incremental  -> slot 1 blue    #2a78d6  (circle markers)
    decremental  -> slot 2 orange  #eb6834  (square markers)
    consenting   -> slot 3 aqua    #1baf7a  (map points / disc wash)
Everything else is ink or grey so it never competes with those three.
"""

import hashlib
import math
import os
import textwrap
import urllib.request
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker  # noqa: E402
from matplotlib.patches import Circle, Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator  # noqa: E402
from PIL import Image  # noqa: E402

from consent_data import Listings  # noqa: E402

# ---- palette / tokens --------------------------------------------------------
INC = "#2a78d6"
DEC = "#eb6834"
CONSENT = "#1baf7a"
ALGO_COLOR = {"incremental": INC, "decremental": DEC}
ALGO_MARKER = {"incremental": "o", "decremental": "s"}
ALGOS = ("incremental", "decremental")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NONCONSENT = "#8f8d87"

TILE_URLS = [
    "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
    "https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
]
# CARTO now answers keyless requests with this "API KEY REQUIRED" placeholder tile
PLACEHOLDER_TILE_MD5 = {"502fc5f6793fad87dcf8ba3fa646c38f"}
TILE_SIZE = 256
_tile_downloads_blocked = False


def setup_style():
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 9.5,
        "text.color": INK,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10.5,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.facecolor": SURFACE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.labelsize": 8.6,
        "ytick.labelsize": 8.6,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "legend.frameon": False,
    })


# ---------------------------------------------------------------------------
# Basemap: Web-Mercator tiles resampled onto the local km grid
# ---------------------------------------------------------------------------
def _merc_px(lat, lng, zoom):
    scale = TILE_SIZE * 2 ** zoom
    X = (np.asarray(lng) + 180.0) / 360.0 * scale
    Y = (1.0 - np.arcsinh(np.tan(np.radians(lat))) / math.pi) / 2.0 * scale
    return X, Y


def _load_tile(cache_dir, z, x, y):
    global _tile_downloads_blocked
    path = os.path.join(cache_dir, f"light_{z}_{x}_{y}.png")
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = f.read()
        if hashlib.md5(data).hexdigest() in PLACEHOLDER_TILE_MD5:  # poisoned cache entry
            os.remove(path)
    if not os.path.exists(path) and not _tile_downloads_blocked:
        for url in TILE_URLS:
            try:
                req = urllib.request.Request(url.format(z=z, x=x, y=y),
                                             headers={"User-Agent": "Mozilla/5.0 (compatible; meb-plot/1.0)"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = resp.read()
            except Exception:
                continue
            if hashlib.md5(data).hexdigest() in PLACEHOLDER_TILE_MD5:
                continue
            with open(path, "wb") as f:
                f.write(data)
            break
        if not os.path.exists(path):
            _tile_downloads_blocked = True
            print(f"  basemap tile {z}/{x}/{y} not cached and the tile servers refused it "
                  "(CARTO now requires an API key); further downloads skipped, map fades out there")
    if not os.path.exists(path):
        return None
    try:
        return np.asarray(Image.open(path).convert("RGB"))
    except Exception as e:
        print(f"  tile {path} unreadable: {e}")
        return None


def basemap_km(listings: Listings, extent_km, cache_dir: str, zoom: int = 13, px_km: float = 0.012,
               fade_km: float = 0.45):
    """RGBA image covering extent_km=(x0, x1, y0, y1) in the local km projection,
    resampled (nearest) from Web-Mercator tiles. x->lng and y->lat are separable,
    so this is two 1D index lookups. Where tiles are unavailable the map fades
    smoothly into the page instead of showing a hard blank edge."""
    x0, x1, y0, y1 = extent_km
    w = int(math.ceil((x1 - x0) / px_km))
    h = int(math.ceil((y1 - y0) / px_km))
    xs = x0 + (np.arange(w) + 0.5) * (x1 - x0) / w
    ys = y1 - (np.arange(h) + 0.5) * (y1 - y0) / h  # top row first
    lat, _ = listings.to_latlng(np.zeros_like(ys), ys)
    _, lng = listings.to_latlng(xs, np.zeros_like(xs))
    X, _ = _merc_px(np.full_like(lng, listings.lat0), lng, zoom)
    _, Y = _merc_px(lat, np.full_like(lat, listings.lng0), zoom)
    tx0, tx1 = int(X.min() // TILE_SIZE), int(X.max() // TILE_SIZE)
    ty0, ty1 = int(Y.min() // TILE_SIZE), int(Y.max() // TILE_SIZE)
    os.makedirs(cache_dir, exist_ok=True)
    canvas = np.zeros(((ty1 - ty0 + 1) * TILE_SIZE, (tx1 - tx0 + 1) * TILE_SIZE, 3), dtype=np.uint8)
    valid = np.zeros(canvas.shape[:2], dtype=bool)
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            tile = _load_tile(cache_dir, zoom, tx, ty)
            if tile is not None:
                sl = (slice((ty - ty0) * TILE_SIZE, (ty - ty0 + 1) * TILE_SIZE),
                      slice((tx - tx0) * TILE_SIZE, (tx - tx0 + 1) * TILE_SIZE))
                canvas[sl] = tile[:TILE_SIZE, :TILE_SIZE, :3]
                valid[sl] = True
    iy = np.clip((Y - ty0 * TILE_SIZE).astype(int), 0, canvas.shape[0] - 1)
    ix = np.clip((X - tx0 * TILE_SIZE).astype(int), 0, canvas.shape[1] - 1)
    rgb = canvas[iy[:, None], ix[None, :]].astype(float) / 255.0
    ok = valid[iy[:, None], ix[None, :]]
    if ok.all():
        alpha = np.ones(ok.shape)
    elif not ok.any():
        alpha = np.zeros(ok.shape)
    else:
        from scipy.ndimage import distance_transform_edt
        alpha = np.clip(distance_transform_edt(ok) * px_km / fade_km, 0.0, 1.0)
    return np.dstack([rgb, alpha]), float(ok.mean())


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def aggregate(rows: List[Dict], experiment: str) -> Dict[tuple, Dict]:
    """(column, threshold, n, algorithm) -> stats over seeds."""
    groups = defaultdict(list)
    for r in rows:
        if r["experiment"] == experiment:
            groups[(r["column"], float(r["threshold"]), int(r["n"]), r["algorithm"])].append(r)
    out = {}
    for k, rr in groups.items():
        q = np.array([float(r["oracle_calls"]) for r in rr])
        t = np.array([float(r["wall_s"]) for r in rr])
        out[k] = dict(
            seeds=len(rr),
            consent_frac=float(np.mean([float(r["consent_frac"]) for r in rr])),
            n_consenting=float(np.mean([float(r["n_consenting"]) for r in rr])),
            threshold_label=rr[0]["threshold_label"],
            calls_mean=q.mean(), calls_min=q.min(), calls_max=q.max(),
            calls_std=q.std(ddof=1) if len(q) > 1 else 0.0,
            wall_mean=t.mean(), wall_min=t.min(), wall_max=t.max(),
            wall_std=t.std(ddof=1) if len(t) > 1 else 0.0,
            wall_median=float(np.median(t)),
            cpu_mean=float(np.mean([float(r["cpu_s"]) for r in rr])),
            noncons_outside=float(np.mean([float(r["noncons_outside_gt"]) for r in rr])),
            gt_radius_km=float(np.mean([float(r["gt_radius_km"]) for r in rr])),
            n_incorrect=int(sum(not bool(r["correct"]) for r in rr)),
        )
    return out


def _series(agg, column, n, algo):
    pts = sorted(((v["consent_frac"], v) for (c, t, nn, a), v in agg.items()
                  if c == column and nn == n and a == algo), key=lambda p: p[0])
    return np.array([p[0] * 100 for p in pts]), [p[1] for p in pts]


def _num_fmt(v, _=None):
    if v >= 1:
        return f"{v:,.0f}"
    return f"{v:.2g}"


def _log_y(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(_num_fmt))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_locator(NullLocator())


def _log_pct_x(ax, ticks: Sequence[Tuple[float, str]]):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([t for t, _ in ticks]))
    labels = {round(t, 9): s for t, s in ticks}
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: labels.get(round(v, 9), "")))
    ax.xaxis.set_minor_locator(NullLocator())


def _legend_handles():
    return [Line2D([], [], color=ALGO_COLOR[a], marker=ALGO_MARKER[a], mec=SURFACE, lw=1.7, ms=6, label=a)
            for a in ALGOS]


# ---------------------------------------------------------------------------
# Figure: small-multiple maps
# ---------------------------------------------------------------------------
def _marker_box(color, marker, size=7.0):
    da = DrawingArea(11, 10, 0, 0)
    da.add_artist(Line2D([5.5], [5.0], marker=marker, markersize=size, color=color,
                         markeredgecolor="white", markeredgewidth=1.0, linestyle="none"))
    return da


def plot_maps(listings: Listings, panels: List[Dict], all_disc: Dict, out_path: str, cache_dir: str,
              zoom: int = 13, title: str = "", subtitle: str = "", footnote: str = ""):
    """panels: dicts with label, consent (bool array), center, radius, support
    (bool array over consenting points), calls {algo: mean}, seeds, frac, n_consenting."""
    setup_style()
    xy = listings.xy
    c_all = np.array([all_disc["center_x_km"], all_disc["center_y_km"]])
    r_all = all_disc["radius_km"]

    # one shared, square extent that contains every disc drawn
    discs = [(c_all, r_all)] + [(np.asarray(p["center"]), p["radius"]) for p in panels]
    lo = np.min([c - r for c, r in discs], axis=0)
    hi = np.max([c + r for c, r in discs], axis=0)
    mid, half = (lo + hi) / 2, (hi - lo).max() / 2 + 0.5
    extent = (mid[0] - half, mid[0] + half, mid[1] - half, mid[1] + half)
    bg, coverage = basemap_km(listings, extent, cache_dir, zoom=zoom)

    ncols = 3
    nrows = int(math.ceil(len(panels) / ncols))
    W, side, gap_x, margin = 15.0, 4.7, 0.2, 0.25
    head, top, foot, gap_y = 0.98, 1.3, 0.5, 0.12
    H = top + nrows * (head + side) + (nrows - 1) * gap_y + foot
    fig = plt.figure(figsize=(W, H))

    for i, p in enumerate(panels):
        r_, c_ = divmod(i, ncols)
        left = margin + c_ * (side + gap_x)
        ytop = top + r_ * (head + side + gap_y) + head
        ax = fig.add_axes([left / W, (H - ytop - side) / H, side / W, side / H])
        ax.set_facecolor(SURFACE)
        ax.imshow(bg, extent=extent, origin="upper", interpolation="antialiased", zorder=0, alpha=0.92)
        cons = p["consent"]
        ax.scatter(xy[~cons, 0], xy[~cons, 1], s=1.3, c=NONCONSENT, alpha=0.45, linewidths=0, zorder=2,
                   rasterized=True)
        # sparse consenting sets get slightly larger dots so they stay visible
        dot = float(np.clip(3.2 * (2000.0 / max(cons.sum(), 1)) ** 0.35, 3.2, 7.0))
        ax.scatter(xy[cons, 0], xy[cons, 1], s=dot, c=CONSENT, alpha=0.9, linewidths=0, zorder=3,
                   rasterized=True)
        ax.add_patch(Circle(tuple(c_all), r_all, fill=False, ec=INK2, lw=1.0, ls=(0, (4, 3)), zorder=4))
        c = np.asarray(p["center"])
        ax.add_patch(Circle(tuple(c), p["radius"], fc=CONSENT, alpha=0.10, ec="none", zorder=1.5))
        ax.add_patch(Circle(tuple(c), p["radius"], fill=False, ec=INK, lw=1.6, zorder=5))
        ax.plot([c_all[0]], [c_all[1]], marker="+", color=INK2, ms=7, mew=1.0, zorder=5)
        ax.plot([c[0]], [c[1]], marker="+", color=INK, ms=9, mew=1.6, zorder=6)
        sup = xy[cons][p["support"]]
        ax.scatter(sup[:, 0], sup[:, 1], s=46, c=INK, edgecolors="white", linewidths=1.6, zorder=7)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(AXIS)
            s.set_linewidth(0.8)

        shift = float(np.linalg.norm(c - c_all))
        rows = [
            TextArea(p["label"], textprops=dict(color=INK, fontsize=11.5, fontweight="bold")),
            TextArea(f"{p['frac'] * 100:.1f}% consenting  ({p['n_consenting']:,} of {listings.n:,} listings)",
                     textprops=dict(color=INK2, fontsize=9)),
            TextArea(f"disc radius {p['radius']:.2f} km (all listings {r_all:.2f}),  centre moved {shift:.2f} km",
                     textprops=dict(color=INK2, fontsize=9)),
            HPacker(children=[
                TextArea(f"oracle calls, mean of {p['seeds']} runs:", textprops=dict(color=INK2, fontsize=9)),
                _marker_box(INC, "o"),
                TextArea(f"incremental {p['calls']['incremental']:,.0f}",
                         textprops=dict(color=INK, fontsize=9, fontweight="semibold")),
                _marker_box(DEC, "s"),
                TextArea(f"decremental {p['calls']['decremental']:,.0f}",
                         textprops=dict(color=INK, fontsize=9, fontweight="semibold")),
            ], align="center", pad=0, sep=3),
        ]
        box = AnchoredOffsetbox(loc="lower left", child=VPacker(children=rows, align="left", pad=0, sep=3.5),
                                pad=0.05, borderpad=0.0, frameon=False, bbox_to_anchor=(0.0, 1.0),
                                bbox_transform=ax.transAxes)
        ax.add_artist(box)
        if i == 0:  # all panels share one scale
            x0, y0 = extent[0] + 0.7, extent[2] + 0.7
            ax.plot([x0, x0 + 5], [y0, y0], color=INK, lw=2.2, solid_capstyle="butt", zorder=8)
            ax.text(x0 + 2.5, y0 + 0.25, "5 km", ha="center", va="bottom", fontsize=8.5, color=INK, zorder=8)

    handles = [
        Line2D([], [], marker="o", ls="none", color=CONSENT, ms=5, label="consenting listing"),
        Line2D([], [], marker="o", ls="none", color=NONCONSENT, ms=4, alpha=0.7, label="non-consenting listing"),
        Patch(facecolor=(0.106, 0.686, 0.478, 0.12), edgecolor=INK, lw=1.4,
              label="minimum enclosing disc of consenting listings"),
        Line2D([], [], marker="o", ls="none", color=INK, mec="white", mew=1.4, ms=7.5,
               label="its support (boundary) points"),
        Line2D([], [], color=INK2, lw=1.0, ls=(0, (4, 3)), label="disc of all listings (reference)"),
        Line2D([], [], marker="o", ls="none", color=INC, mec="white", ms=7, label="incremental"),
        Line2D([], [], marker="s", ls="none", color=DEC, mec="white", ms=7, label="decremental"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(margin / W - 0.004, 1 - 0.86 / H), ncol=7,
               fontsize=9, handletextpad=0.4, columnspacing=1.3, labelcolor=INK2)
    fig.text(margin / W, 1 - 0.22 / H, title, fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(margin / W, 1 - 0.58 / H, subtitle, fontsize=10, color=INK2, va="top")
    if coverage < 0.999:
        footnote += (f"  Basemap tiles cover {coverage * 100:.0f}% of the panel; uncached tiles could not be fetched, "
                     "so the map fades out there.")
    fig.text(margin / W, 0.12 / H, textwrap.fill(footnote, 235), fontsize=7.8, color=MUTED, va="bottom")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Performance figures (manual grid so row titles never collide)
# ---------------------------------------------------------------------------
def _perf_grid(nrows, title, subtitle):
    W, H_top, row_h, plot_h = 15.5, 1.05, 4.75, 3.25
    H = H_top + nrows * row_h + 0.1
    fig = plt.figure(figsize=(W, H))
    left, right, gap = 0.75, 0.25, 0.85
    pw = (W - left - right - 2 * gap) / 3
    axes = []
    for r in range(nrows):
        row_top = H_top + r * row_h
        ax_top = row_top + 0.42 + 0.36
        axes.append([fig.add_axes([(left + j * (pw + gap)) / W, (H - ax_top - plot_h) / H, pw / W, plot_h / H])
                     for j in range(3)])
    fig.text(0.25 / W, 1 - 0.22 / H, title, fontsize=14.5, fontweight="bold", color=INK, va="top")
    fig.text(0.25 / W, 1 - 0.58 / H, subtitle, fontsize=9.5, color=INK2, va="top")
    fig.legend(handles=_legend_handles(), loc="upper right", bbox_to_anchor=(1 - 0.2 / W, 1 - 0.16 / H), ncol=2,
               fontsize=9.5, labelcolor=INK2)

    def row_title(r, text):
        fig.text(0.25 / W, 1 - (H_top + r * row_h + 0.1) / H, text, fontsize=11.5, fontweight="bold",
                 color=INK, va="top")
    return fig, np.array(axes), row_title


def _draw_algo(ax, x, stats, key, algo, scale=1.0):
    color = ALGO_COLOR[algo]
    scale = np.broadcast_to(np.asarray(scale, dtype=float), (len(stats),))
    mean = np.array([s[f"{key}_mean"] for s in stats]) * scale
    lo = np.array([s[f"{key}_min"] for s in stats]) * scale
    hi = np.array([s[f"{key}_max"] for s in stats]) * scale
    ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0, zorder=2)
    ax.plot(x, mean, color=color, lw=1.7, zorder=3, solid_joinstyle="round", solid_capstyle="round")
    ax.plot(x, mean, ls="none", marker=ALGO_MARKER[algo], ms=5.2, color=color, mec=SURFACE, mew=1.1, zorder=4)
    return mean


def _direct_labels(ax, x, means: Dict[str, np.ndarray]):
    """Direct-label both algorithm lines, each label placed in the gap between
    the two lines. Candidate positions are tried from the widest gap down and
    a candidate is kept only if neither label touches either line (checked in
    display coordinates after the axis scales are final)."""
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    names = list(means)
    lines = []
    for k in names:
        pts = ax.transData.transform(np.c_[x, means[k]])
        t = np.linspace(0, 1, 25)[:, None]
        lines.append(np.vstack([p0 + t * (p1 - p0) for p0, p1 in zip(pts[:-1], pts[1:])]) if len(pts) > 1 else pts)
    poly = np.vstack(lines)
    axbox = ax.get_window_extent(renderer)
    gap = np.abs(np.log(means[names[0]] / means[names[1]]))
    order = [0, len(x) - 1] + [int(i) for i in np.argsort(-gap) if 0 < i < len(x) - 1]
    order = sorted(dict.fromkeys(order), key=lambda i: -gap[i])
    fallback = None
    for i in order:
        ha = "left" if i == 0 else ("right" if i == len(x) - 1 else "center")
        dx = 6 if ha == "left" else (-6 if ha == "right" else 0)
        top = max(names, key=lambda k: means[k][i])
        for off in (9, 14, 20):
            anns = []
            for k in names:
                up = k == top  # upper line: label below it; lower line: label above it
                anns.append(ax.annotate(k, (x[i], means[k][i]), xytext=(dx, -off if up else off),
                                        textcoords="offset points", ha=ha, va="top" if up else "bottom",
                                        fontsize=8.8, color=INK2, zorder=6))
            ok = True
            for a in anns:
                bb = a.get_window_extent(renderer).expanded(1.0, 1.0).padded(2)
                inside = ((poly[:, 0] >= bb.x0) & (poly[:, 0] <= bb.x1) &
                          (poly[:, 1] >= bb.y0) & (poly[:, 1] <= bb.y1))
                if inside.any() or not (axbox.x0 <= bb.x0 and bb.x1 <= axbox.x1 and
                                        axbox.y0 <= bb.y0 and bb.y1 <= axbox.y1):
                    ok = False
            if ok:
                return
            if fallback is None:
                fallback = (i, off)
            for a in anns:
                a.remove()
    i, off = fallback  # nothing collision-free: use the widest gap
    top = max(names, key=lambda k: means[k][i])
    for k in names:
        up = k == top
        ax.annotate(k, (x[i], means[k][i]), xytext=(0, -off if up else off), textcoords="offset points",
                    ha="center", va="top" if up else "bottom", fontsize=8.8, color=INK2, zorder=6)


def plot_threshold_perf(agg: Dict, columns: Sequence[str], column_titles: Dict[str, str], n_full: int,
                        out_path: str, tick_thresholds: Dict[str, Sequence[float]], seeds: int,
                        correctness_note: str = ""):
    setup_style()
    fig, axes, row_title = _perf_grid(
        len(columns),
        f"Incremental vs decremental consent-aware MEB, Amsterdam Airbnb (n = {n_full:,} listings, 2D km)",
        f"Lines: mean over {seeds} seeds; shaded band: min–max over seeds. Both algorithms get identical "
        f"inputs and alternate which runs first per seed. {correctness_note}")
    for r, column in enumerate(columns):
        x, inc = _series(agg, column, n_full, "incremental")
        _, dec = _series(agg, column, n_full, "decremental")
        stats = {"incremental": inc, "decremental": dec}
        tk = []
        for s in inc:
            lab = s["threshold_label"]
            if any(abs(s["consent_frac"] - agg[(column, t, n_full, "incremental")]["consent_frac"]) < 1e-12
                   for t in tick_thresholds.get(column, []) if (column, t, n_full, "incremental") in agg):
                op = "≤" if "≤" in lab else "≥"
                val = lab.split(op)[-1].strip().replace(" d/yr", "").replace(" d ago", "d")
                pct = s["consent_frac"] * 100
                tk.append((pct, f"{pct:.0f}%\n{op}{val}" if pct >= 1.5 else f"{pct:.1f}%\n{op}{val}"))
        row_title(r, column_titles[column])
        panels = [
            ("calls", 1.0, "oracle calls (unique consent requests)", "Consent requests"),
            ("calls", 100.0 / n_full, "oracle calls, % of all listings", "Consent requests as a share of n"),
            ("wall", 1.0, "run time, seconds (wall clock)", "Run time"),
        ]
        for j, (key, scale, ylab, ptitle) in enumerate(panels):
            ax = axes[r, j]
            means = {a: _draw_algo(ax, x, stats[a], key, a, scale) for a in ALGOS}
            if j == 1:
                ref = np.array([s["noncons_outside"] for s in inc]) * scale
                pos = ref > 0
                ref_line, = ax.plot(x[pos], ref[pos], color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=2.5,
                                    label="reference: non-consenters outside the consenting disc")
                ax.legend(handles=[ref_line], loc="upper right", fontsize=8, labelcolor=INK2,
                          handlelength=2.2, borderaxespad=0.2)
            _log_y(ax)
            _log_pct_x(ax, tk)
            ax.set_ylabel(ylab)
            ax.set_title(ptitle)
            ax.set_xlabel("listings consenting (% of n), consent threshold", fontsize=8.8)
            if j != 1:
                _direct_labels(ax, x, means)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_size_perf(agg: Dict, column: str, row_titles: Dict[float, str], out_path: str, seeds: int):
    setup_style()
    thresholds = sorted({t for (c, t, n, a) in agg if c == column})
    fig, axes, row_title = _perf_grid(
        len(thresholds), "How consent requests and run time scale with the number of listings",
        f"Each seed draws a fresh random subsample of listings (same consent rule); lines: mean over {seeds} "
        "seeds, band: min–max. At full n every seed uses the same listings, so only the shuffle varies.")
    for r, t in enumerate(thresholds):
        stats = {}
        for algo in ALGOS:
            pts = sorted(((n, v) for (c, tt, n, a), v in agg.items() if c == column and tt == t and a == algo))
            ns = np.array([p[0] for p in pts], dtype=float)
            stats[algo] = [p[1] for p in pts]
        frac = np.mean([s["consent_frac"] for s in stats["incremental"]])
        row_title(r, f"{row_titles.get(t, str(t))}   (≈{frac * 100:.0f}% consenting)")
        panels = [("calls", "oracle calls (unique consent requests)", "Consent requests vs n"),
                  ("calls", "oracle calls, % of n", "Consent requests as a share of n"),
                  ("wall", "run time, seconds (wall clock)", "Run time vs n")]
        for j, (key, ylab, ptitle) in enumerate(panels):
            ax = axes[r, j]
            scale = (100.0 / ns) if j == 1 else 1.0
            means = {a: _draw_algo(ax, ns, stats[a], key, a, scale) for a in ALGOS}
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(FixedLocator(ns))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
            ax.xaxis.set_minor_locator(NullLocator())
            _log_y(ax)
            ax.set_xlabel("n (random subsample of listings)", fontsize=8.8)
            ax.set_ylabel(ylab)
            ax.set_title(ptitle)
            _direct_labels(ax, ns, means)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Consent-column analysis figure
# ---------------------------------------------------------------------------
def plot_column_analysis(rows: List[Dict], labels: Dict[str, str], primary: str, recommended: str,
                         achievable: Dict[str, Tuple[float, float]], r_all: float, out_path: str, draws: int):
    setup_style()
    thr_cols = [c for c in labels if any(r["column"] == c and r["kind"] == "threshold" for r in rows)]
    binary = [r for r in rows if r["kind"] == "binary"]
    npanels = len(thr_cols) + (1 if binary else 0)
    ncols = 4
    nrows = int(math.ceil(npanels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.9 * nrows + 1.55), sharey=True, squeeze=False)
    H = fig.get_size_inches()[1]
    fig.subplots_adjust(left=0.05, right=0.99, top=1 - 1.45 / H, bottom=0.55 / H, wspace=0.08, hspace=0.45)
    axes = axes.ravel()
    ticks = [(1, "1%"), (2, "2%"), (5, "5%"), (10, "10%"), (25, "25%"), (50, "50%"), (90, "90%")]
    for ax in axes[npanels:]:
        ax.set_visible(False)

    for ax, col in zip(axes, thr_cols):
        rr = sorted((r for r in rows if r["column"] == col and r["kind"] == "threshold"),
                    key=lambda r: r["consent_frac"])
        x = np.array([r["consent_frac"] * 100 for r in rr])
        ax.fill_between(x, [r["random_hausdorff_p10"] for r in rr], [r["random_hausdorff_p90"] for r in rr],
                        color=MUTED, alpha=0.2, lw=0, zorder=1)
        ax.plot(x, [r["random_hausdorff_p50"] for r in rr], color=MUTED, lw=1.2, zorder=2)
        ax.plot(x, [r["hausdorff_km"] for r in rr], color=INK, lw=1.7, zorder=3)
        ax.plot(x, [r["hausdorff_km"] for r in rr], ls="none", marker="o", ms=4.6, color=INK, mec=SURFACE,
                mew=1.0, zorder=4)
        ax.set_xlim(0.8, 100)
        _log_pct_x(ax, ticks)
        tag = "  — primary (requested)" if col == primary else (
            "  — best alternative" if col == recommended else "")
        ax.set_title(labels[col] + tag, fontsize=10)
        lo, hi = achievable.get(col, (x.min() / 100, x.max() / 100))
        lo_txt = f"{lo * 100:.1f}" if lo >= 0.001 else "<0.1"
        ax.text(0.98, 0.96, f"thresholds reach {lo_txt}–{hi * 100:.0f}% consenting",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
        ax.set_xlabel("listings consenting", fontsize=8.5)

    if binary:
        ax = axes[len(thr_cols)]
        for r in binary:
            xf = r["consent_frac"] * 100
            ax.plot([xf, xf], [r["random_hausdorff_p10"], r["random_hausdorff_p90"]], color=MUTED, lw=6,
                    alpha=0.35, solid_capstyle="round", zorder=1)
            ax.plot([xf], [r["hausdorff_km"]], marker="o", ms=6, color=INK, mec=SURFACE, mew=1.0, zorder=3)
            ax.annotate(r["rule"].split(" (")[0], (xf, r["hausdorff_km"]), xytext=(-7, 0),
                        textcoords="offset points", fontsize=7.8, color=INK2, ha="right", va="center")
        ax.set_xlim(0.8, 100)
        _log_pct_x(ax, ticks)
        ax.set_title("binary columns (one fraction each)", fontsize=10)
        ax.set_xlabel("listings consenting", fontsize=8.5)

    for ax in axes[::ncols]:
        if ax.get_visible():
            ax.set_ylabel("disc displacement, km")
    handles = [
        Line2D([], [], color=INK, lw=1.7, marker="o", ms=5, mec=SURFACE, label="consent = column passes a threshold"),
        Line2D([], [], color=MUTED, lw=1.2, label=f"random subset of the same size (median of {draws})"),
        Patch(facecolor=MUTED, alpha=0.3, label="random subset, 10th–90th percentile"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.045, 1 - 0.92 / H), ncol=3, fontsize=9,
               labelcolor=INK2)
    fig.text(0.012, 1 - 0.22 / H, "Which column makes a consent signal that actually moves the disc?",
             fontsize=14.5, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 1 - 0.56 / H,
             "Displacement = Hausdorff distance between the consenting disc and the all-listings disc "
             f"(centre shift + radius change; all-listings radius {r_all:.2f} km).  Below the grey band the "
             "consenters reach the city edge more than chance would (the disc is 'sticky');\nabove it they "
             "cluster away from it. Thresholds are quantiles of each column.", fontsize=9.3, color=INK2, va="top")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
