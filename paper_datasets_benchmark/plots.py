"""
Figures for the paper-datasets benchmark (PNG, written to figures/; *.png is
gitignored repo-wide).

Per dataset:
  <key>_spatial.png      small multiples over consent thresholds (top row: the
                         chosen consent column, bottom row: random consent at
                         the same rate) with consenting MEB + support points
  <key>_performance.png  oracle calls, calls / n and run time vs % consenting,
                         incremental vs decremental (mean line, min-max band)
  <key>_vs_n.png         the same metrics vs n at a fixed ~50% threshold
Across datasets:
  overview_calls.png, overview_time.png

Colours (validated with the dataviz palette script, light surface):
incremental = slot 1 blue, decremental = slot 2 orange, consenting points =
slot 3 aqua, everything else neutral ink / greys.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

import datasets  # noqa: E402
from meb_common import (FIGURES_DIR, RESULTS_DIR, TILE_ATTRIBUTION, basemap_equirect,  # noqa: E402
                        consent_mask,
                        ground_truth_meb, subsample, support_indices,
                        threshold_for_fraction)

# ---- design tokens ----------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
C_INC = "#2a78d6"
C_DEC = "#eb6834"
C_CONSENT = "#1baf7a"
C_NON = "#b9b8b0"
ALGO_COLOR = {"incremental": C_INC, "decremental": C_DEC}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 9.5,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 10.5,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 9.5,
})

PANEL_FRACS = (0.05, 0.20, 0.50, 0.80, 0.95)


def fmt_compact(v):
    a = abs(v)
    if a >= 1e9:
        return f"{v/1e9:.1f}B"
    if a >= 1e6:
        return f"{v/1e6:.1f}M"
    if a >= 1e4:
        return f"{v/1e3:.0f}K"
    return datasets.fmt_num(v)


def rule_op(cand):
    return "≥" if cand.sign > 0 else "≤"


def rule_text(cand, t):
    return f"{cand.label} {rule_op(cand)} {fmt_compact(t)}"


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("seed", "n", "calls", "match", "n_consent", "outside_nonconsent"):
            r[k] = int(r[k])
        for k in ("target_frac", "threshold", "frac", "seconds", "radius", "gt_radius"):
            r[k] = float(r[k])
    return rows


def save(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", path)
    return path


def algo_legend_handles(bound=False):
    hs = [Line2D([], [], color=C_INC, lw=2, marker="o", ms=5.5, mec=SURFACE, mew=1.2,
                 label="incremental"),
          Line2D([], [], color=C_DEC, lw=2, marker="o", ms=5.5, mec=SURFACE, mew=1.2,
                 label="decremental")]
    if bound:
        hs.append(Line2D([], [], color=MUTED, lw=1.2,
                         label="non-consenters outside the consenting MEB "
                               "(decremental must query each)"))
    return hs


# ----------------------------------------------------------------------------
# spatial small multiples
# ----------------------------------------------------------------------------

def spatial(key, n=5000):
    ds = datasets.load(key)
    cand = ds.candidates[ds.chosen]
    N = len(ds.X)
    idx = subsample(N, min(n, N), 1000)          # == the seed-0 subsample of benchmark.py
    X = ds.X[idx]
    v = cand.values[idx]
    sweep = read_csv(os.path.join(RESULTS_DIR, f"{key}_sweep.csv"))
    calls = {(r["mechanism"], round(r["target_frac"], 4), r["algo"]): r
             for r in sweep if r["seed"] == 0}
    run_fracs = sorted({round(r["target_frac"], 4) for r in sweep})
    fracs = [f for f in PANEL_FRACS if ("column", round(f, 4), "incremental") in calls]
    c_all, r_all = ground_truth_meb(X)

    # ---- compute every panel first, so one extent covers all circles -------
    panels = {}
    for mech in ("column", "random"):
        for f in fracs:
            t, real = threshold_for_fraction(cand.values, cand.sign, f)
            if mech == "column":
                m = consent_mask(v, cand.sign, t)
            else:
                # same RNG stream as benchmark.py: default_rng([seed, frac index, 7])
                fi = run_fracs.index(round(f, 4))
                m = np.random.default_rng([0, fi, 7]).random(len(X)) < real
            c, r = ground_truth_meb(X[m])
            panels[(mech, f)] = (t, real, m, c, r)

    geo = ds.geo is not None
    scale = ds.geo["km_per_unit"] if geo else 1.0
    P = X * scale
    lo = np.minimum(P.min(0), (c_all - r_all) * scale)
    hi = np.maximum(P.max(0), (c_all + r_all) * scale)
    for (_, _, _, c, r) in panels.values():
        if c is not None:
            lo = np.minimum(lo, (c - r) * scale)
            hi = np.maximum(hi, (c + r) * scale)
    pad = 0.03 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    bg = None
    if geo:
        R = 6371.0088
        lat0, lon0 = ds.geo["lat0"], ds.geo["lon0"]
        coslat = np.cos(np.radians(lat0))
        try:
            bg = basemap_equirect(lat0 + np.degrees(lo[1] / R), lat0 + np.degrees(hi[1] / R),
                                  lon0 + np.degrees(lo[0] / (R * coslat)),
                                  lon0 + np.degrees(hi[0] / (R * coslat)), lat0, lon0, zoom=4)
        except Exception as e:  # offline: plain background
            print("  basemap unavailable:", e)

    ncol = len(fracs)
    aspect = (hi[1] - lo[1]) / (hi[0] - lo[0])
    fig, axes = plt.subplots(2, ncol, figsize=(3.6 * ncol, 2 * 3.6 * aspect + 2.4), squeeze=False)
    row_title = {"column": f"Consent = {cand.label} {rule_op(cand)} t",
                 "random": "Random consent at the same rate (contrast)"}
    for ri, mech in enumerate(("column", "random")):
        for ci, f in enumerate(fracs):
            ax = axes[ri, ci]
            t, real, m, c, r = panels[(mech, f)]
            if bg is not None:
                ax.imshow(bg[0], extent=bg[1], origin="upper", zorder=0, interpolation="bilinear")
            ax.scatter(P[~m, 0], P[~m, 1], s=4 if not geo else 10, c=C_NON if not geo else MUTED,
                       lw=0, zorder=2, alpha=0.75)
            ax.scatter(P[m, 0], P[m, 1], s=5 if not geo else 13, c=C_CONSENT, lw=0, zorder=3,
                       alpha=0.9)
            ax.add_patch(Circle(c_all * scale, r_all * scale, fill=False, ec=MUTED, lw=1.0,
                                zorder=4))
            if c is not None:
                ax.add_patch(Circle(c * scale, r * scale, fill=True, fc=C_CONSENT, alpha=0.06,
                                    lw=0, zorder=4))
                ax.add_patch(Circle(c * scale, r * scale, fill=False, ec=INK, lw=2, zorder=5))
                sup = support_indices(X[m], c, r)
                S = P[m][sup]
                ax.scatter(S[:, 0], S[:, 1], s=60, facecolor=C_CONSENT, edgecolor=INK, lw=1.4,
                           zorder=6)
                ax.plot(*(c * scale), marker="+", color=INK, ms=9, mew=1.6, zorder=6)
            ri_ = calls.get((mech, round(f, 4), "incremental"))
            rd_ = calls.get((mech, round(f, 4), "decremental"))
            head = f"t = {fmt_compact(t)}" if mech == "column" else f"Bernoulli p = {real:.2f}"
            txt = (f"{head}\n{100*m.mean():.0f}% consenting ({int(m.sum())}/{len(m)})\n"
                   f"oracle calls: inc {ri_['calls']} · dec {rd_['calls']}\n"
                   f"radius {r/r_all:.2f} × all-points MEB")
            ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8.0,
                    color=INK, zorder=10, linespacing=1.35,
                    bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec=GRID, lw=0.8, alpha=0.9))
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set_aspect("equal", adjustable="box")
            ax.grid(not geo)
            ax.tick_params(labelsize=7.5)
            if ci == 0:
                ax.set_ylabel(("north-south (km)" if geo else ds.axis_labels[1]))
            if ri == 1:
                ax.set_xlabel(("east-west (km)" if geo else ds.axis_labels[0]))
        axes[ri, 0].annotate(row_title[mech], xy=(0, 1.03), xycoords="axes fraction",
                             fontsize=10.5, color=INK, fontweight="semibold", ha="left", va="bottom")
    handles = [
        Line2D([], [], ls="", marker="o", ms=6, mfc=C_CONSENT, mec=C_CONSENT, label="consenting"),
        Line2D([], [], ls="", marker="o", ms=6, mfc=C_NON if not geo else MUTED,
               mec=C_NON if not geo else MUTED, label="not consenting"),
        Line2D([], [], color=INK, lw=2, label="MEB of consenting points"),
        Line2D([], [], ls="", marker="o", ms=8, mfc=C_CONSENT, mec=INK, mew=1.4,
               label="support points (on the MEB)"),
        Line2D([], [], color=MUTED, lw=1, label="MEB of all points"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0))
    n_txt = f"{len(X):,} of {N:,} points" if len(X) < N else f"all {N:,} points"
    fig.suptitle(f"{ds.code} — {ds.title}: consenting MEB across consent thresholds "
                 f"({n_txt}, seed 0)", x=0.01, y=0.995, ha="left", fontsize=12.5, color=INK,
                 fontweight="semibold")
    sub = ds.unit_note + (f"  {TILE_ATTRIBUTION}." if bg is not None else "")
    fig.text(0.01, 0.962, sub, fontsize=8.5, color=INK2, ha="left", va="top", wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    return save(fig, f"{key}_spatial.png")


# ----------------------------------------------------------------------------
# performance: calls, calls/n, time vs % consenting
# ----------------------------------------------------------------------------

def _agg(rows, xkey):
    """group by (algo, xkey) -> x mean, y stats per metric."""
    g = defaultdict(list)
    for r in rows:
        g[(r["algo"], r[xkey])].append(r)
    out = {}
    for algo in ("incremental", "decremental"):
        xs = sorted({k[1] for k in g if k[0] == algo})
        recs = [g[(algo, x)] for x in xs]
        out[algo] = {
            "x": np.array([np.mean([r["frac"] if xkey == "target_frac" else r["n"] for r in rr])
                           for rr in recs]),
            "calls": np.array([[r["calls"] for r in rr] for rr in recs], dtype=float),
            "per_n": np.array([[r["calls"] / r["n"] for r in rr] for rr in recs], dtype=float),
            "sec": np.array([[r["seconds"] for r in rr] for rr in recs], dtype=float),
            "bound": np.array([[r["outside_nonconsent"] for r in rr] for rr in recs], dtype=float),
            "bound_per_n": np.array([[r["outside_nonconsent"] / r["n"] for r in rr] for rr in recs],
                                    dtype=float),
        }
    return out


def _series(ax, x, Y, color, label=None, scale_y=1.0):
    mean = Y.mean(axis=1) * scale_y
    ax.fill_between(x, Y.min(axis=1) * scale_y, Y.max(axis=1) * scale_y, color=color,
                    alpha=0.13, lw=0, zorder=2)
    ax.plot(x, mean, color=color, lw=2, solid_capstyle="round", zorder=3, label=label)
    ax.scatter(x, mean, s=30, color=color, edgecolor=SURFACE, lw=1.2, zorder=4)


def _crossovers(x, a, b):
    """x positions where sign(a - b) changes (linear interpolation)."""
    d = a - b
    xs = []
    for i in range(len(d) - 1):
        if d[i] == 0:
            xs.append(x[i])
        elif d[i] * d[i + 1] < 0:
            xs.append(x[i] + (x[i + 1] - x[i]) * d[i] / (d[i] - d[i + 1]))
    return xs


def performance(key):
    ds = datasets.load(key)
    cand = ds.candidates[ds.chosen]
    sweep = read_csv(os.path.join(RESULTS_DIR, f"{key}_sweep.csv"))
    mechs = [m for m in ("column", "random") if any(r["mechanism"] == m for r in sweep)]
    n = sweep[0]["n"]
    seeds = len({r["seed"] for r in sweep})
    fig, axes = plt.subplots(len(mechs), 3, figsize=(14.5, 4.1 * len(mechs) + 0.6),
                             squeeze=False, sharey="col", sharex=True)
    titles = ("Consent requests (oracle calls)", "Consent requests per point (% of n)",
              "Run time (s, log scale)")
    for ri, mech in enumerate(mechs):
        A = _agg([r for r in sweep if r["mechanism"] == mech], "target_frac")
        for ci, metric in enumerate(("calls", "per_n", "sec")):
            ax = axes[ri, ci]
            if metric in ("calls", "per_n"):
                b = A["incremental"]["bound" if metric == "calls" else "bound_per_n"].mean(1)
                ax.plot(100 * A["incremental"]["x"], b * (100.0 if metric == "per_n" else 1.0),
                        color=MUTED, lw=1.2, zorder=2)
            for algo in ("incremental", "decremental"):
                _series(ax, 100 * A[algo]["x"], A[algo][metric], ALGO_COLOR[algo], algo,
                        scale_y=100.0 if metric == "per_n" else 1.0)
            if metric == "sec":
                ax.set_yscale("log")
            xs = _crossovers(100 * A["incremental"]["x"], A["incremental"][metric].mean(1),
                             A["decremental"][metric].mean(1))
            for xc in xs:
                ax.axvline(xc, color=AXIS, lw=1, zorder=1)
                ax.annotate(f"crossover\n≈{xc:.0f}%", xy=(xc, 1), xycoords=("data", "axes fraction"),
                            xytext=(3, -3), textcoords="offset points", va="top", fontsize=7.8,
                            color=INK2)
            if ri == 0:
                ax.set_title(titles[ci], loc="left")
            if ri == len(mechs) - 1:
                ax.set_xlabel("% of points consenting")
            ax.set_xlim(0, 100)
            if metric != "sec":
                ax.set_ylim(bottom=0)
        lab = (f"Consent = {cand.label} {rule_op(cand)} t  (t = quantiles of the column)"
               if mech == "column" else "Random consent at the same rates (contrast)")
        axes[ri, 0].annotate(lab, xy=(0, 1.13 if ri == 0 else 1.04), xycoords="axes fraction",
                             fontsize=10.5, fontweight="semibold", color=INK, ha="left", va="bottom")
    fig.legend(handles=algo_legend_handles(bound=True), loc="upper right", ncol=3,
               bbox_to_anchor=(0.995, 1.0))
    fig.suptitle(f"{ds.code} — {ds.title}: incremental vs decremental consent-aware MEB",
                 x=0.01, ha="left", fontsize=12.5, color=INK, fontweight="semibold")
    fig.text(0.01, 0.935 if len(mechs) == 2 else 0.9,
             f"n = {n} points per run, {seeds} seeds per point (fresh subsample + shuffle); "
             f"line = mean, band = min–max.",
             fontsize=8.8, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93 if len(mechs) == 2 else 0.88))
    return save(fig, f"{key}_performance.png")


def vs_n(key):
    path = os.path.join(RESULTS_DIR, f"{key}_vs_n.csv")
    if not os.path.exists(path):
        return None
    ds = datasets.load(key)
    rows = read_csv(path)
    A = _agg(rows, "n")
    frac = np.mean([r["frac"] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    titles = ("Consent requests (oracle calls)", "Consent requests per point (% of n)",
              "Run time (s)")
    for ci, metric in enumerate(("calls", "per_n", "sec")):
        ax = axes[ci]
        for algo in ("incremental", "decremental"):
            _series(ax, A[algo]["x"], A[algo][metric], ALGO_COLOR[algo], algo,
                    scale_y=100.0 if metric == "per_n" else 1.0)
        ax.set_xscale("log")
        if metric in ("calls", "sec"):
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
        ax.set_title(titles[ci] + (", log-log" if metric != "per_n" else ""), loc="left")
        ax.set_xlabel("n (points, log scale)")
        xs = sorted(set(A["incremental"]["x"].astype(int)))
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:,}" for x in xs], rotation=0)
        ax.minorticks_off()
    fig.legend(handles=algo_legend_handles(), loc="upper right", ncol=2, bbox_to_anchor=(0.995, 1.02))
    cand = ds.candidates[ds.chosen]
    fig.suptitle(f"{ds.code} — {ds.title}: scaling with n at a fixed threshold "
                 f"({rule_text(cand, rows[0]['threshold'])}, ≈{100*frac:.0f}% consenting)",
                 x=0.01, ha="left", fontsize=12.5, color=INK, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save(fig, f"{key}_vs_n.png")


# ----------------------------------------------------------------------------
# overview across datasets
# ----------------------------------------------------------------------------

def overview(keys, metric):
    keys = [k for k in keys if os.path.exists(os.path.join(RESULTS_DIR, f"{k}_sweep.csv"))]
    if not keys:
        return None
    nc = 3
    nr = int(np.ceil(len(keys) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(14.5, 3.9 * nr + 0.7), squeeze=False)
    for i, key in enumerate(keys):
        ax = axes[i // nc, i % nc]
        ds = datasets.load(key)
        sweep = read_csv(os.path.join(RESULTS_DIR, f"{key}_sweep.csv"))
        A = _agg([r for r in sweep if r["mechanism"] == "column"], "target_frac")
        for algo in ("incremental", "decremental"):
            _series(ax, 100 * A[algo]["x"], A[algo][metric], ALGO_COLOR[algo], algo,
                    scale_y=100.0 if metric == "per_n" else 1.0)
        if metric == "sec":
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
        ax.set_xlim(0, 100)
        ax.set_title(f"{ds.code} — {ds.title}  (n={sweep[0]['n']})", loc="left", fontsize=10)
        cand = ds.candidates[ds.chosen]
        ax.text(0.98, 0.95, f"consent: {cand.label} {rule_op(cand)} t",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
        if i // nc == nr - 1:
            ax.set_xlabel("% of points consenting")
    for j in range(len(keys), nr * nc):
        axes[j // nc, j % nc].set_visible(False)
    what = {"per_n": "Consent requests per point (% of n)", "sec": "Run time (s, log scale)",
            "calls": "Consent requests (oracle calls)"}[metric]
    fig.legend(handles=algo_legend_handles(), loc="upper right", ncol=2, bbox_to_anchor=(0.995, 1.0))
    fig.suptitle(f"All paper datasets — {what} vs % consenting (consent-column thresholds)",
                 x=0.01, ha="left", fontsize=12.5, color=INK, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save(fig, f"overview_{'calls' if metric == 'per_n' else 'time'}.png")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--skip-spatial", action="store_true")
    args = ap.parse_args(argv)
    keys = args.datasets or list(datasets.LOADERS)
    for key in keys:
        if not os.path.exists(os.path.join(RESULTS_DIR, f"{key}_sweep.csv")):
            print(f"[{key}] no results yet - run benchmark.py first")
            continue
        print(f"[{key}]")
        if not args.skip_spatial:
            spatial(key, args.n)
        performance(key)
        vs_n(key)
    overview(list(datasets.LOADERS), "per_n")
    overview(list(datasets.LOADERS), "sec")


if __name__ == "__main__":
    main()
