"""
Performance figures: incremental vs decremental on Chicago crimes.

Reads results/benchmark_summary.json (and results/consent_candidates.json) and
writes:
  crime_requests_vs_consent.png  consent requests (raw, and per point) vs % consenting
  crime_runtime_vs_consent.png   run time vs % consenting
  crime_scaling_vs_n.png         requests and run time vs n at ~50% consent
  crime_consent_candidates.png   MEB shift per candidate consent column vs random consent

Lines are the mean over seeds; the shaded band spans min..max across seeds.
"""

import argparse
import json
import os

import viz_style as V

V.apply()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

SWEEP_PANELS = [
    ("district_n2000", "DISTRICT ≤ k  ·  n = 2,000", "spatially clustered consent"),
    ("timestamp_n2000", "TIMESTAMP ≥ t  ·  n = 2,000", "spatially random consent"),
    ("timestamp_n20000", "TIMESTAMP ≥ t  ·  n = 20,000", "spatially random consent"),
]
SCALING_PANELS = [
    ("timestamp_scaling", "TIMESTAMP ≥ 2009-07-03 (≈50% consenting)"),
    ("district_scaling", "DISTRICT ≤ 10 (≈51% consenting)"),
]


def pretty(label):
    return label.replace(">=", "≥").replace("<=", "≤")


def _pct_label(p):
    return f"{p:.1f}%" if 99 < p < 100 or 0 < p < 1 else f"{p:.0f}%"


def load_summary():
    with open(os.path.join(RESULTS, "benchmark_summary.json")) as f:
        return json.load(f)


def series(points, exp, algo, xkey):
    pts = [p for p in points if p["experiment"] == exp and p["algorithm"] == algo]
    if xkey == "pct":
        pts.sort(key=lambda p: p["pct_consenting"]["mean"])
        xs = [p["pct_consenting"]["mean"] for p in pts]
    else:
        pts.sort(key=lambda p: p["n"])
        xs = [p["n"] for p in pts]
    return xs, pts


def _num(v, _pos=None):
    if v >= 1:
        if v >= 10000:
            return f"{v / 1000:,.0f}K"
        return f"{v:,.0f}"
    return f"{v:g}"


def _sec(v, _pos=None):
    if v >= 1:
        return f"{v:g} s"
    if v >= 0.001:
        return f"{v * 1000:g} ms"
    return f"{v * 1e6:g} µs"


def _pct(v, _pos=None):
    return f"{v * 100:g}%"


def log_axis(ax, which="y", fmt=_num):
    getattr(ax, f"set_{which}scale")("log")
    axis = ax.yaxis if which == "y" else ax.xaxis
    axis.set_major_locator(LogLocator(base=10))
    axis.set_minor_formatter(NullFormatter())
    axis.set_major_formatter(FuncFormatter(fmt))


def plot_algo(ax, xs, pts, algo, field, scale=1.0, label_end=False, label=None):
    col = V.ALGO_COLOR[algo]
    mean = [p[field]["mean"] * scale for p in pts]
    lo = [p[field]["min"] * scale for p in pts]
    hi = [p[field]["max"] * scale for p in pts]
    ax.fill_between(xs, lo, hi, color=col, alpha=0.14, linewidth=0, zorder=2)
    ax.plot(xs, mean, color=col, lw=1.8, zorder=4, marker=V.ALGO_MARKER[algo], ms=5.5,
            mec=V.SURFACE, mew=1.2, label=label or V.ALGO_LABEL[algo])
    if label_end and xs:
        ax.annotate(V.ALGO_LABEL[algo], (xs[-1], mean[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=9, color=V.INK2)


def plot_bound(ax, xs, pts, scale=1.0, label=True):
    lb = [p["lower_bound"]["mean"] * scale for p in pts]
    ax.plot(xs, lb, color=V.INK2, lw=1.1, ls=(0, (3, 2.5)), zorder=6)
    return lb


def legend_handles(bound=True):
    hs = [Line2D([], [], color=V.ALGO_COLOR[a], lw=1.8, marker=V.ALGO_MARKER[a], ms=5.5, mec=V.SURFACE,
                 mew=1.2, label=V.ALGO_LABEL[a]) for a in ("incremental", "decremental")]
    if bound:
        hs.append(Line2D([], [], color=V.INK2, lw=1.1, ls=(0, (3, 2.5)),
                         label="Lower bound (non-consenting points outside the answer + support)"))
    return hs


def header(fig, title, subtitle, y_title=0.985, y_sub=0.935):
    fig.text(0.015, y_title, title, ha="left", va="top", fontsize=14, fontweight="semibold", color=V.INK)
    fig.text(0.015, y_sub, subtitle, ha="left", va="top", fontsize=10, color=V.INK2)


def fig_requests(points, out):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), sharex=True)
    for j, (exp, ttl, sub) in enumerate(SWEEP_PANELS):
        for i, (field_scale, ylab) in enumerate([("raw", "Consent requests"), ("per_n", "Requests per point")]):
            ax = axes[i, j]
            V.despine(ax)
            for algo in ("decremental", "incremental"):
                xs, pts = series(points, exp, algo, "pct")
                if not pts:
                    continue
                scale = 1.0 if field_scale == "raw" else 1.0 / pts[0]["n"]
                field = "calls"
                plot_algo(ax, xs, pts, algo, field, scale=scale)
            xs, pts = series(points, exp, "incremental", "pct")
            if pts:
                scale = 1.0 if field_scale == "raw" else 1.0 / pts[0]["n"]
                plot_bound(ax, xs, pts, scale=scale)
            log_axis(ax, "y", _num if field_scale == "raw" else _pct)
            ax.set_xlim(0, 100)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0f}%"))
            if i == 0:
                ax.set_title(ttl, loc="left", pad=17)
                ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=9, color=V.MUTED, va="bottom")
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 1:
                ax.set_xlabel("Consenting incidents")
    # shared y range per row for the per-point row so panels compare directly
    lo = min(a.get_ylim()[0] for a in axes[1]); hi = max(a.get_ylim()[1] for a in axes[1])
    for a in axes[1]:
        a.set_ylim(lo, hi)
    lo = min(a.get_ylim()[0] for a in axes[0]); hi = max(a.get_ylim()[1] for a in axes[0])
    for a in axes[0]:
        a.set_ylim(lo, hi)
    header(fig, "Consent requests vs consenting fraction",
           "Mean of 5 seeds (band: min–max). Requests are unique oracle calls. Clustered consent forces both "
           "algorithms to ask almost every non-consenting point outside the answer;\nrandom consent needs only a "
           "handful, and there decremental sits on the lower bound while incremental asks 4–16× more.",
           y_sub=0.945)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.83, bottom=0.12, wspace=0.16, hspace=0.12)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[perf] wrote {out}")


def fig_runtime(points, out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.9), sharey=True)
    for j, (exp, ttl, sub) in enumerate(SWEEP_PANELS):
        ax = axes[j]
        V.despine(ax)
        for algo in ("decremental", "incremental"):
            xs, pts = series(points, exp, algo, "pct")
            if pts:
                plot_algo(ax, xs, pts, algo, "seconds")
        log_axis(ax, "y", _sec)
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0f}%"))
        ax.set_title(ttl, loc="left", pad=17)
        ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=9, color=V.MUTED, va="bottom")
        ax.set_xlabel("Consenting incidents")
        if j == 0:
            ax.set_ylabel("Run time per query (pure Python)")
    header(fig, "Run time vs consenting fraction",
           "Mean of 5 seeds (band: min–max). Decremental recomputes a full MEB after every peeling round: "
           "with clustered consent it is ~200–850× slower than incremental, with random consent 1.2–80×.",
           y_sub=0.915)
    fig.legend(handles=legend_handles(bound=False), loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.74, bottom=0.2, wspace=0.08)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[perf] wrote {out}")


def fig_scaling(points, meta, out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), sharex=True)
    for j, (exp, ttl) in enumerate(SCALING_PANELS):
        for i, (field, ylab, fmt) in enumerate([("calls", "Consent requests", _num),
                                                ("seconds", "Run time per query", _sec)]):
            ax = axes[i, j]
            V.despine(ax)
            for algo in ("decremental", "incremental"):
                xs, pts = series(points, exp, algo, "n")
                if pts:
                    plot_algo(ax, xs, pts, algo, field)
            if field == "calls":
                xs, pts = series(points, exp, "incremental", "n")
                plot_bound(ax, xs, pts)
            log_axis(ax, "y", fmt)
            log_axis(ax, "x", _num)
            if i == 0:
                ax.set_title(ttl, loc="left")
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 1:
                ax.set_xlabel("Points n (distinct incident locations)")
            xs_d, pts_d = series(points, exp, "decremental", "n")
            xs_i, _ = series(points, exp, "incremental", "n")
            if pts_d and xs_i and xs_d[-1] < xs_i[-1] and i == 1:
                ax.annotate(f"decremental stopped after n = {xs_d[-1]:,}\n(projected > {meta['dec_projected_cap_s']:.0f} s per run)",
                            (xs_d[-1], pts_d[-1]["seconds"]["mean"]), xytext=(14, -2), textcoords="offset points",
                            fontsize=8.8, color=V.INK2, va="center")
    for row in axes:
        lo = min(a.get_ylim()[0] for a in row); hi = max(a.get_ylim()[1] for a in row)
        for a in row:
            a.set_ylim(lo, hi)
    header(fig, "Scaling with n at ≈50% consent",
           "Mean of 5 seeds (band: min–max), log–log. Random consent: requests stay near-constant in n. "
           "Clustered consent: requests grow linearly for both;\ndecremental's run time grows ~n^1.7 "
           "(26 s per run at n = 4,000).",
           y_sub=0.948)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.085, right=0.98, top=0.85, bottom=0.12, wspace=0.12, hspace=0.12)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[perf] wrote {out}")


def fig_candidates(out):
    with open(os.path.join(RESULTS, "consent_candidates.json")) as f:
        cand = json.load(f)
    rows = []
    for e in cand["boolean"]:
        if "shift_km" in e:
            rows.append((f"{e['candidate']}", e))
    for e in cand["threshold"]:
        if e["target"] in (0.25, 0.5) and "shift_km" in e:
            rows.append((e["label"], e))
    rows.sort(key=lambda t: t[1]["shift_pct_of_radius"])
    chosen = lambda name: name.startswith("DISTRICT") or name.startswith("TIMESTAMP")  # noqa: E731
    r_all = cand["all_points_radius_km"]
    fig, ax = plt.subplots(figsize=(10.5, 0.33 * len(rows) + 2.2))
    V.despine(ax)
    ax.grid(axis="y", visible=False)
    for i, (name, e) in enumerate(rows):
        rnd = 100.0 * e["random_shift_km"] / r_all
        act = e["shift_pct_of_radius"]
        ax.plot([rnd, act], [i, i], color=V.AXIS, lw=1.5, zorder=2)
        ax.scatter([rnd], [i], s=34, c=V.SURFACE, edgecolors=V.MUTED, linewidths=1.4, zorder=3)
        ax.scatter([act], [i], s=40, c=V.INK, edgecolors=V.SURFACE, linewidths=1.4, zorder=4)
        ax.text(max(act, rnd) + 1.5, i, f"{act:.0f}%", va="center", fontsize=8.5, color=V.INK2)
    ax.set_yticks(range(len(rows)))
    labels = [f"{pretty(name)}   ({_pct_label(e['pct_consenting'])} consenting)" for name, e in rows]
    ax.set_yticklabels(labels, fontsize=9)
    for tl, (name, _) in zip(ax.get_yticklabels(), rows):
        if chosen(name):
            tl.set_fontweight("semibold")
            tl.set_color(V.INK)
    ax.set_xlabel("MEB shift: |Δcentre| + |Δradius| as % of the all-points radius")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0f}%"))
    ax.set_xlim(left=-2)
    handles = [Line2D([], [], ls="none", marker="o", ms=6.5, mfc=V.INK, mec=V.SURFACE, label="This consent rule"),
               Line2D([], [], ls="none", marker="o", ms=6, mfc=V.SURFACE, mec=V.MUTED, mew=1.4,
                      label="Random consent at the same fraction (mean of 20 draws)")]
    ax.legend(handles=handles, loc="lower right")
    fig.text(0.015, 0.99, "How far each candidate consent column moves the MEB", ha="left", va="top",
             fontsize=14, fontweight="semibold", color=V.INK)
    fig.text(0.015, 0.99 - 0.42 / fig.get_figheight(),
             f"n = {cand['n']:,} distinct locations; all-points MEB radius {r_all:.1f} km. Threshold columns shown "
             "at ≈25% and ≈50% consenting. Bold = the two sweeps used.",
             ha="left", va="top", fontsize=9.5, color=V.INK2)
    fig.subplots_adjust(left=0.36, right=0.97, top=1 - 0.95 / fig.get_figheight(),
                        bottom=0.75 / fig.get_figheight())
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[perf] wrote {out}")


def make_all(out_dir=HERE):
    summ = load_summary()
    pts, meta = summ["points"], summ["meta"]
    outs = [os.path.join(out_dir, f) for f in ("crime_requests_vs_consent.png", "crime_runtime_vs_consent.png",
                                               "crime_scaling_vs_n.png", "crime_consent_candidates.png")]
    fig_requests(pts, outs[0])
    fig_runtime(pts, outs[1])
    fig_scaling(pts, meta, outs[2])
    fig_candidates(outs[3])
    return outs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", default=HERE)
    a = ap.parse_args()
    make_all(a.out_dir)
