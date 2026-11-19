"""Spotify: minimum enclosing BOX of consenting tracks (orthogonal range query).

Same plane, consent column, thresholds and seed-0 subsample as the ball figure
(plots.spatial), but uses db_consents.incremental_orthogonal /
decremental_orthogonal. Writes spotify_spatial.png (box version) to --out and
results/spotify_box.json.

    python -I spotify_box.py [--out DIR] [--n 5000] [--seeds 5]
"""
import argparse, json, os, random, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plots  # noqa: E402  (style + helpers)
from plots import (plt, Line2D, datasets, SURFACE, INK, INK2, MUTED, GRID, C_CONSENT, C_NON,
                   C_INC, C_DEC, PANEL_FRACS, threshold_text, rule_op)
from matplotlib.patches import Rectangle  # noqa: E402
from meb_common import (FIGURES_DIR, RESULTS_DIR, consent_mask, subsample, threshold_for_fraction,
                        to_points, REPO)
sys.path.insert(0, REPO)
import db_consents  # noqa: E402
from db_consents import Oracle, incremental_orthogonal, decremental_orthogonal  # noqa: E402


def run(name, pts, seed):
    random.seed(seed)
    oracle = Oracle()
    t0 = time.perf_counter()
    lo, hi = (incremental_orthogonal if name == "incremental" else decremental_orthogonal)(pts, oracle)
    return np.array(lo, float), np.array(hi, float), oracle.get_call_count(), time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=FIGURES_DIR)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()

    ds = datasets.load("spotify")
    cand = ds.candidates[ds.chosen]
    N = len(ds.X)
    idx = subsample(N, min(a.n, N), 1000)
    X, v = ds.X[idx], cand.values[idx]
    lo_all, hi_all = X.min(0), X.max(0)

    rows, panels = [], []
    for f in PANEL_FRACS:
        t, _ = threshold_for_fraction(cand.values, cand.sign, f)
        m = consent_mask(v, cand.sign, t)
        gt_lo, gt_hi = X[m].min(0), X[m].max(0)
        pts = to_points(X, m)
        stats = {}
        for algo in ("incremental", "decremental"):
            calls, secs = [], []
            for s in range(a.seeds):
                order = ("incremental", "decremental") if s % 2 == 0 else ("decremental", "incremental")
                lo, hi, c, sec = run(algo, pts, s)
                ok = bool(np.array_equal(lo, gt_lo) and np.array_equal(hi, gt_hi))
                assert ok, (algo, f, s)
                calls.append(c); secs.append(sec)
                rows.append(dict(target_frac=f, threshold=float(t), frac=float(m.mean()), algo=algo,
                                 seed=s, calls=c, seconds=sec, matches_ground_truth=ok))
            stats[algo] = (float(np.mean(calls)), float(np.mean(secs)))
        panels.append((t, m, gt_lo, gt_hi, stats))
        print(f"{100*m.mean():5.1f}%  inc {stats['incremental']}  dec {stats['decremental']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json.dump(rows, open(os.path.join(RESULTS_DIR, "spotify_box.json"), "w"), indent=1)

    pad = 0.04 * (hi_all - lo_all)
    ncol = len(panels)
    fig, axes = plt.subplots(1, ncol, figsize=(3.6 * ncol, 5.6), layout="constrained", squeeze=False)
    for ci, (t, m, lo, hi, st) in enumerate(panels):
        ax = axes[0][ci]
        ax.scatter(X[~m, 0], X[~m, 1], s=4, c=C_NON, lw=0, alpha=0.75, zorder=2)
        ax.scatter(X[m, 0], X[m, 1], s=5, c=C_CONSENT, lw=0, alpha=0.9, zorder=3)
        ax.add_patch(Rectangle(lo_all, *(hi_all - lo_all), fill=False, ec=MUTED, lw=1.0, ls="--", zorder=4))
        ax.add_patch(Rectangle(lo, *(hi - lo), fill=True, fc=C_CONSENT, alpha=0.06, lw=0, zorder=4))
        ax.add_patch(Rectangle(lo, *(hi - lo), fill=False, ec=INK, lw=2, zorder=5))
        Xc = X[m]
        edge = ((Xc == lo) | (Xc == hi)).any(1)
        ax.scatter(Xc[edge, 0], Xc[edge, 1], s=45, facecolor=C_CONSENT, edgecolor=INK, lw=1.2, zorder=6)
        area = np.prod(hi - lo) / np.prod(hi_all - lo_all)
        (ic, isec), (dc, dsec) = st["incremental"], st["decremental"]
        ax.set_title(f"{threshold_text(ds, t)}\n{100*m.mean():.0f}% consenting ({int(m.sum())}/{len(m)})\n"
                     f"box area {area:.2f} × all-points box", loc="left", fontsize=9, color=INK)
        ax.text(0.0, -0.17, f"consent requests: inc {ic:.0f} · dec {dc:.0f}\n"
                            f"run time: inc {1000*isec:.1f} ms · dec {1000*dsec:.1f} ms",
                transform=ax.transAxes, va="top", ha="left", fontsize=8.5, color=INK, linespacing=1.4)
        ax.set_xlim(lo_all[0] - pad[0], hi_all[0] + pad[0])
        ax.set_ylim(lo_all[1] - pad[1], hi_all[1] + pad[1])
        ax.set_xlabel(ds.axis_labels[0])
        if ci == 0:
            ax.set_ylabel(ds.axis_labels[1])
        ax.tick_params(labelsize=7.5)
    handles = [
        Line2D([], [], ls="", marker="o", ms=6, mfc=C_CONSENT, mec=C_CONSENT, label="consenting"),
        Line2D([], [], ls="", marker="o", ms=6, mfc=C_NON, mec=C_NON, label="not consenting"),
        Line2D([], [], color=INK, lw=2, label="box of consenting points"),
        Line2D([], [], ls="", marker="o", ms=7, mfc=C_CONSENT, mec=INK, mew=1.2, label="consenting points on the box edge"),
        Line2D([], [], color=MUTED, lw=1, ls="--", label="box of all points"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=5)
    fig.suptitle(f"SP — Spotify tracks: minimum enclosing box of consenting tracks "
                 f"({len(X):,} of {N:,} points)\nConsent = {cand.label} {rule_op(cand)} t   ·   "
                 f"requests and run time are means of {a.seeds} runs", x=0.005, ha="left",
                 fontsize=12, color=INK, fontweight="semibold")
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "spotify_spatial.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    main()
