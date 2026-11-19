"""
Figures for the flight consent-MEB study (reads results/*.json; PNGs are
gitignored repo-wide and written next to this file).

    flight_map_<column>.png        small multiples on map tiles, one per threshold
    flight_perf_vs_fraction.png    oracle calls / calls per n / run time vs % consenting
    flight_perf_vs_n.png           oracle calls / calls per n / run time vs n
    flight_consent_candidates.png  MEB shift of every candidate consent column

Colour roles are fixed across every figure: incremental = blue, decremental =
orange (categorical slots 1 and 2 of the dataviz reference palette, validated
for CVD separation); consenting points = aqua, non-consenting = muted grey.

Usage:
    python plot_flight_figures.py [--csv PATH] [--zoom 5]
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

import flight_data as fd  # noqa: E402
import map_tiles  # noqa: E402
from flight_meb_benchmark import subsample  # noqa: E402
from meb_tools import exact_meb  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

# ---- palette / chrome (dataviz reference palette, light mode) -------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
C_INC = "#2a78d6"      # incremental (slot 1, blue)
C_DEC = "#eb6834"      # decremental (slot 2, orange)
C_CONS = "#1baf7a"     # consenting points (slot 3, aqua)
C_NONC = "#898781"     # non-consenting points (muted)
C_PRIMARY = "#008300"  # candidate chart: chosen column (emphasis)
C_CONTRAST = "#6250d6"  # candidate chart: contrast column
C_OTHER = "#c3c2b7"

ALGO_COLOR = {"incremental": C_INC, "decremental": C_DEC}

COLUMN_TITLE = {
    "optin_date": "Opt-in date (flight legs flown on/after T consent)",
    "leg_random": "Random flight legs (each leg consents with probability p)",
}
COLUMN_SHORT = {"optin_date": "opt-in date", "leg_random": "random legs",
                "point_random": "random positions"}
CANDIDATE_NAME = {
    "optin_date": "opt-in date, per leg (TIME)",
    "flightnum_top": "busiest flight numbers (CALLSIGN)",
    "hour_of_day": "UTC time of day (TIME)",
    "geo_above": "GEOALTITUDE above h",
    "flightnum_random": "random flight numbers",
    "velocity_above": "VELOCITY above v",
    "leg_random": "random flight legs",
    "squawk_below": "SQUAWK below s",
    "baro_above": "BAROALTITUDE above h",
    "point_random": "random positions",
    "baro_below": "BAROALTITUDE below h",
}


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "legend.frameon": False,
        "savefig.facecolor": SURFACE,
    })


def _load(name):
    with open(os.path.join(RESULTS_DIR, name)) as fh:
        return json.load(fh)


def _fmt_int(v):
    return f"{int(round(v)):,}"


# --------------------------------------------------------------------------
# Maps
# --------------------------------------------------------------------------

def _panel_header(ax, title, sub, calls):
    """Header above a map panel (keeps the map itself unobstructed).
    Text stays in ink; algorithm identity comes from the coloured dots."""
    ax.text(0.0, 1.118, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=10.5,
            fontweight="semibold", color=INK)
    ax.text(0.0, 1.066, sub, transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
            color=INK2)
    x = 0.0
    for algo, n_calls in calls:
        ax.plot([x + 0.01], [1.033], marker="o", ms=6, color=ALGO_COLOR[algo], mec=SURFACE,
                mew=1.0, transform=ax.transAxes, clip_on=False)
        ax.text(x + 0.032, 1.033, f"{algo}: {_fmt_int(n_calls)} oracle calls",
                transform=ax.transAxes, ha="left", va="center", fontsize=9, color=INK2)
        x += 0.5


def plot_maps(fs, sweep, column, out_path, zoom=5):
    runs = sorted([r for r in sweep["runs"] if r["column"] == column and r["seed"] == 0],
                  key=lambda r: r["target_frac"])
    n = sweep["n"]
    idx = subsample(fs, n, 0)
    xy = fs.xy_km[idx]
    lat, lon = fs.c["lat"][idx], fs.c["lon"][idx]
    my = map_tiles.merc_y(lat)
    fn = fd.SWEEP_CANDIDATES[column][0]

    c_all, r_all, _ = exact_meb(xy, unit=fd.ALGO_UNIT_KM)
    clat_all, clon_all = fs.proj.circle_latlon(c_all[0], c_all[1], r_all)
    masks, mebs = [], []
    for rec in runs:
        m = fn(fs, rec["target_frac"])[0][idx]
        assert int(m.sum()) == rec["n_consenting"], "subsample/mask drifted from sweep.json"
        c, r, sup = exact_meb(xy[m], unit=fd.ALGO_UNIT_KM)
        assert abs(r - rec["gt_radius_km"]) <= 1e-6 * r
        masks.append(m)
        mebs.append((c, r, sup))
    # extent: every point and every circle drawn in any panel
    circ = [fs.proj.circle_latlon(c[0], c[1], r) for c, r, _ in mebs] + [(clat_all, clon_all)]
    lat_lo = min([lat.min()] + [a.min() for a, _ in circ]) - 0.6
    lat_hi = max([lat.max()] + [a.max() for a, _ in circ]) + 0.6
    lon_lo = min([lon.min()] + [b.min() for _, b in circ]) - 0.8
    lon_hi = max([lon.max()] + [b.max() for _, b in circ]) + 0.8
    img, ext = map_tiles.basemap(lat_lo, lat_hi, lon_lo, lon_hi, zoom=zoom)

    ncols = 3
    nrows = int(np.ceil(len(runs) / ncols))
    aspect = (map_tiles.merc_y(lat_hi) - map_tiles.merc_y(lat_lo)) / (lon_hi - lon_lo)
    pw = 5.0
    fig, axes = plt.subplots(nrows, ncols, figsize=(pw * ncols, (pw * aspect + 0.7) * nrows + 1.5),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.17})
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(runs):]:
        ax.axis("off")

    for ax, rec, m, (c, r, sup) in zip(axes, runs, masks, mebs):
        ax.imshow(img, extent=ext, origin="upper", interpolation="lanczos", zorder=0)
        ax.scatter(lon[~m], my[~m], s=1.6, c=C_NONC, alpha=0.55, lw=0, zorder=2, rasterized=True)
        ax.scatter(lon[m], my[m], s=2.4, c=C_CONS, alpha=0.8, lw=0, zorder=3, rasterized=True)
        ax.plot(clon_all, map_tiles.merc_y(clat_all), color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=4)
        clat, clon = fs.proj.circle_latlon(c[0], c[1], r)
        ax.plot(clon, map_tiles.merc_y(clat), color=INK, lw=1.7, zorder=5, solid_capstyle="round")
        cy, cx = fs.proj.inverse(c[0], c[1])
        ax.plot([cx], [map_tiles.merc_y(cy)], marker="+", ms=8, mew=1.6, color=INK, zorder=6)
        slat, slon = fs.proj.inverse(sup[:, 0], sup[:, 1])
        ax.scatter(slon, map_tiles.merc_y(slat), s=46, c=INK, edgecolors=SURFACE, linewidths=1.6,
                   zorder=7)
        title = rec["label"].replace("legs since", "Legs since").replace(
            "random flight legs", "Random legs,")
        _panel_header(ax, title,
                      f"{rec['consent_pct']:.1f}% of positions consenting · MEB radius "
                      f"{_fmt_int(r)} km (all positions: {_fmt_int(r_all)} km)",
                      [("incremental", rec["incremental_calls"]),
                       ("decremental", rec["decremental_calls"])])
        ax.set_xlim(lon_lo, lon_hi)
        ax.set_ylim(map_tiles.merc_y(lat_lo), map_tiles.merc_y(lat_hi))
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(AXIS); s.set_linewidth(0.8)

    handles = [
        Line2D([], [], ls="", marker="o", ms=6, color=C_CONS, label="consenting position"),
        Line2D([], [], ls="", marker="o", ms=6, color=C_NONC, alpha=0.6, label="non-consenting position"),
        Line2D([], [], color=INK, lw=1.7, label="MEB of consenting positions"),
        Line2D([], [], ls="", marker="o", ms=7, color=INK, mec=SURFACE, mew=1.4, label="support points"),
        Line2D([], [], color=INK2, lw=1.0, ls=(0, (4, 3)), label="MEB of all positions"),
        Line2D([], [], ls="", marker="o", ms=6.5, color=C_INC, label="incremental"),
        Line2D([], [], ls="", marker="o", ms=6.5, color=C_DEC, label="decremental"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=9, bbox_to_anchor=(0.5, 0.012),
               handletextpad=0.4, columnspacing=1.4, labelcolor=INK2)
    fig.text(0.99, 0.004, map_tiles.ATTRIBUTION, fontsize=7, color=MUTED, ha="right", va="bottom")
    fig.suptitle(f"Consenting-point MEB: {COLUMN_TITLE.get(column, column)}", x=0.012, ha="left",
                 y=0.995, fontsize=14, color=INK, fontweight="semibold")
    fig.text(0.012, 0.972,
             f"n = {n:,} positions of aircraft 471f63, sampled uniformly from the whole file "
             "(Mar 2016 – Nov 2019); oracle calls are for this drawn sample (seed 0).\n"
             "The ball is computed in 2D (altitude ignored) on an azimuthal-equidistant km plane; "
             "each outline is that plane circle mapped back onto the Web-Mercator basemap, "
             "so it is not drawn as a perfect circle.", fontsize=9, color=INK2, ha="left", va="top",
             linespacing=1.5)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.905, bottom=0.045)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# Performance charts
# --------------------------------------------------------------------------

def _series(summary, column, xkey, transform=None):
    rows = sorted([r for r in summary if r["column"] == column], key=lambda r: r[xkey])
    return rows


def _band_line(ax, x, stats, color, label, scale=1.0):
    ok = [i for i, s in enumerate(stats) if s is not None]
    if not ok:
        return
    xs = np.asarray([x[i] for i in ok], dtype=float)
    mean = np.asarray([stats[i]["mean"] for i in ok]) * scale
    lo = np.asarray([stats[i]["min"] for i in ok]) * scale
    hi = np.asarray([stats[i]["max"] for i in ok]) * scale
    ax.fill_between(xs, lo, hi, color=color, alpha=0.13, lw=0, zorder=2)
    ax.plot(xs, mean, color=color, lw=2.0, marker="o", ms=5.5, mec=SURFACE, mew=1.4,
            label=label, zorder=4, solid_capstyle="round", solid_joinstyle="round")
    return xs, mean


def _lb_line(ax, x, stats, scale=1.0):
    xs = np.asarray(x, dtype=float)
    ax.plot(xs, np.asarray([s["mean"] for s in stats]) * scale, color=MUTED, lw=1.2,
            ls=(0, (1.5, 2.5)), label="lower bound (points outside consenting MEB + support)",
            zorder=3)


def _log_axis(ax, which="y"):
    axis = ax.yaxis if which == "y" else ax.xaxis
    getattr(ax, f"set_{which}scale")("log")
    axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.set_minor_formatter(NullFormatter())
    axis.set_major_formatter(FuncFormatter(lambda v, _: (f"{v:,.0f}" if v >= 1 else f"{v:g}")))


def _end_label(ax, x, y, text, color, dy):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", fontsize=9,
                color=INK2, va="center", ha="left")
    ax.plot([x], [y], marker="o", ms=5.5, color=color, mec=SURFACE, mew=1.4, zorder=5)


def plot_perf_vs_fraction(sweep, out_path):
    summ = sweep["summary"]
    cols = [sweep["primary_column"], sweep["contrast_column"]]
    fig, axes = plt.subplots(3, 2, figsize=(12, 11.5), sharex=True)
    for j, col in enumerate(cols):
        rows = _series(summ, col, "target_frac")
        x = [r["consent_pct"]["mean"] for r in rows]
        n = rows[0]["n"]
        ax = axes[0, j]
        ends = {}
        for algo in ("incremental", "decremental"):
            ends[algo] = _band_line(ax, x, [r[f"{algo}_calls"] for r in rows], ALGO_COLOR[algo], algo)
        _lb_line(ax, x, [r["lower_bound_calls"] for r in rows])
        _log_axis(ax)
        ax.set_ylabel("Oracle calls (log)")
        ax.set_title(COLUMN_TITLE.get(col, col), loc="left", fontsize=11, pad=10)
        for algo in ("incremental", "decremental"):
            xs, ys = ends[algo]
            ax.annotate(algo, (xs[-2], ys[-2]), xytext=(0, 9), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, color=INK2)

        ax = axes[1, j]
        for algo in ("incremental", "decremental"):
            _band_line(ax, x, [r[f"{algo}_calls_per_n"] for r in rows], ALGO_COLOR[algo], algo,
                       scale=100.0)
        _lb_line(ax, x, [r["lower_bound_calls"] for r in rows], scale=100.0 / n)
        ax.set_ylabel(f"Oracle calls, % of n (n = {n:,})")
        ax.set_ylim(bottom=0)

        ax = axes[2, j]
        for algo in ("incremental", "decremental"):
            _band_line(ax, x, [r[f"{algo}_time_s"] for r in rows], ALGO_COLOR[algo], algo)
        _log_axis(ax)
        ax.set_ylabel("Run time, s (log)")
        ax.set_xlabel("Consenting positions, % of n")
        ax.set_xticks([5, 20, 35, 50, 65, 80, 95])
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", ncol=3, bbox_to_anchor=(0.01, 0.955), fontsize=9.5,
               labelcolor=INK2)
    nseeds = summ[0]["seeds"]
    fig.suptitle("Incremental vs decremental as the consenting fraction changes", x=0.01, ha="left",
                 y=0.995, fontsize=14, fontweight="semibold", color=INK)
    fig.text(0.01, 0.968, f"Lines: mean of {nseeds} seeds (fresh subsample + shuffle each); "
             "bands: min–max. Pure-Python db_consents, 2D positions in 1000-km units.",
             fontsize=9.5, color=INK2, ha="left", va="top")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.06, hspace=0.18, wspace=0.18)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_perf_vs_n(scaling, out_path):
    summ = scaling["summary"]
    fracs = scaling["fractions"]
    fig, axes = plt.subplots(3, len(fracs), figsize=(12, 11.5), sharex=True)
    for j, f in enumerate(fracs):
        rows = sorted([r for r in summ if r["target_frac"] == f], key=lambda r: r["n"])
        x = [r["n"] for r in rows]
        ax = axes[0, j]
        for algo in ("incremental", "decremental"):
            _band_line(ax, x, [r[f"{algo}_calls"] for r in rows], ALGO_COLOR[algo], algo)
        _lb_line(ax, x, [r["lower_bound_calls"] for r in rows])
        _log_axis(ax); _log_axis(ax, "x")
        ax.set_ylabel("Oracle calls (log)")
        cons = np.mean([r["consent_pct"]["mean"] for r in rows])
        ax.set_title(f"{rows[0]['label'].replace('legs since', 'Legs since')} "
                     f"(~{cons:.0f}% consenting)", loc="left", fontsize=11, pad=10)
        skipped = [r["n"] for r in rows if r["decremental_skipped"]]
        if skipped:
            ax.text(0.99, 0.03, "decremental not run at n = " + ", ".join(f"{s:,}" for s in skipped)
                    + "\n(predicted > 60 s per run)", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=8.5, color=INK2)

        ax = axes[1, j]
        for algo in ("incremental", "decremental"):
            _band_line(ax, x, [r[f"{algo}_calls_per_n"] for r in rows], ALGO_COLOR[algo], algo,
                       scale=100.0)
        ax.plot(x, [100.0 * r["lower_bound_calls"]["mean"] / r["n"] for r in rows], color=MUTED,
                lw=1.2, ls=(0, (1.5, 2.5)), zorder=3)
        _log_axis(ax); _log_axis(ax, "x")
        ax.set_ylabel("Oracle calls, % of n (log)")

        ax = axes[2, j]
        for algo in ("incremental", "decremental"):
            _band_line(ax, x, [r[f"{algo}_time_s"] for r in rows], ALGO_COLOR[algo], algo)
        _log_axis(ax); _log_axis(ax, "x")
        ax.set_ylabel("Run time, s (log)")
        ax.set_xlabel("n (positions, log)")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
        ax.set_xticks(x)
    for ax in axes.ravel():
        ax.grid(True, axis="both", which="major")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", ncol=3, bbox_to_anchor=(0.01, 0.955), fontsize=9.5,
               labelcolor=INK2)
    fig.suptitle("Scaling with n at two fixed opt-in dates", x=0.01, ha="left", y=0.995,
                 fontsize=14, fontweight="semibold", color=INK)
    fig.text(0.01, 0.968, f"Lines: mean of {summ[0]['seeds']} seeds; bands: min–max. "
             "Each n is a fresh uniform subsample of the cleaned whole-file sample.",
             fontsize=9.5, color=INK2, ha="left", va="top")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.06, hspace=0.18, wspace=0.18)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_candidates(cand, primary, contrast, out_path):
    sweep = cand["sweep"]
    names = sorted(sweep, key=lambda k: sweep[k]["mean_hausdorff_pct_r0"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1, 1.15]})
    ys = np.arange(len(names))
    vals = [sweep[k]["mean_hausdorff_pct_r0"] for k in names]
    colors = [C_PRIMARY if k == primary else C_CONTRAST if k == contrast else C_OTHER for k in names]
    ax1.barh(ys, vals, height=0.55, color=colors, zorder=3)
    for y, v in zip(ys, vals):
        ax1.text(v + 0.8, y, f"{v:.1f}%", va="center", fontsize=8.8, color=INK2)
    ax1.set_yticks(ys)
    ax1.set_yticklabels([CANDIDATE_NAME.get(k, k) for k in names])
    base = np.mean(list(cand["random_baseline"].values()))
    ax1.axvline(base, color=MUTED, lw=1.0, ls=(0, (1.5, 2.5)), zorder=2)
    ax1.set_ylim(-0.6, len(names) + 0.1)
    ax1.text(base + 0.4, len(names) - 0.3, "random-position baseline (5 seeds)", fontsize=8.5,
             color=INK2, va="center")
    ax1.grid(True, axis="x"); ax1.grid(False, axis="y")
    ax1.set_xlabel("MEB shift vs all positions, % of all-points radius\n(mean over 5–95% consenting)")
    ax1.set_title("Candidate consent columns", loc="left", fontsize=11)

    for k, col, lw in ((primary, C_PRIMARY, 2.0), (contrast, C_CONTRAST, 2.0),
                       ("point_random", MUTED, 1.6)):
        rows = sweep[k]["rows"]
        x = [r["consent_pct"] for r in rows]
        y = [r["hausdorff_pct_r0"] for r in rows]
        ax2.plot(x, y, color=col, lw=lw, marker="o", ms=5.5, mec=SURFACE, mew=1.4, zorder=4,
                 label=COLUMN_SHORT.get(k, k))
    ax2.legend(loc="upper right", fontsize=9, labelcolor=INK2)
    ax2.set_xlabel("Consenting positions, % (whole cleaned sample)")
    ax2.set_ylabel("MEB shift, % of all-points radius")
    ax2.set_title("Chosen column vs contrast vs random positions", loc="left", fontsize=11)
    ax2.set_ylim(bottom=0)
    fx = cand["fixed"]
    note = ("Not sweepable: " + ", ".join(f"{k} {fx[k]['consent_pct']:.3g}%" for k in
                                          ("icao24", "airline_prefix", "alert", "spi", "onground")))
    fig.text(0.01, 0.015, note + "  (one aircraft, one airline; flags are rare)", fontsize=8.8,
             color=INK2)
    fig.suptitle("How far does the consenting MEB move? (exact MEBs on ~800k cleaned positions)",
                 x=0.01, ha="left", fontsize=13, fontweight="semibold", color=INK)
    fig.subplots_adjust(left=0.19, right=0.98, top=0.86, bottom=0.2, wspace=0.22)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run(csv_path, zoom=5, maps=True, log=print):
    _style()
    out = []
    sweep = _load("sweep.json")
    scaling = _load("scaling.json")
    cand = _load("consent_candidates.json")
    out.append(plot_perf_vs_fraction(sweep, os.path.join(HERE, "flight_perf_vs_fraction.png")))
    out.append(plot_perf_vs_n(scaling, os.path.join(HERE, "flight_perf_vs_n.png")))
    out.append(plot_candidates(cand, sweep["primary_column"], sweep["contrast_column"],
                               os.path.join(HERE, "flight_consent_candidates.png")))
    if maps:
        fs = fd.FlightSet(csv_path)
        for col in (sweep["primary_column"], sweep["contrast_column"]):
            out.append(plot_maps(fs, sweep, col, os.path.join(HERE, f"flight_map_{col}.png"),
                                 zoom=zoom))
    for p in out:
        log(f"Wrote {p}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=fd.DEFAULT_CSV)
    ap.add_argument("--zoom", type=int, default=5)
    ap.add_argument("--no-maps", action="store_true")
    args = ap.parse_args()
    run(args.csv, zoom=args.zoom, maps=not args.no_maps)


if __name__ == "__main__":
    main()
