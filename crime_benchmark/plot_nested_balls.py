"""All DISTRICT <= k consenting balls on ONE map, coloured by k, with the path of the centre.

    python crime_benchmark/plot_nested_balls.py [--csv ...] [--out FILE] [--n 4000]
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import consent as C
import crime_data as cd
import viz_style as V
from plot_chicago_crimes import basemap_km, scale_bar

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", default=cd.DEFAULT_CSV)
    ap.add_argument("--out", default=os.path.join(HERE, "chicago_district_nested_balls.png"))
    ap.add_argument("--n", type=int, default=4000)
    a = ap.parse_args()
    V.apply()
    records, meta, proj = cd.load(a.csv)
    fam = C.THRESHOLD_FAMILIES["DISTRICT"]
    rows, coords = cd.draw_points(records, proj, a.n, seed=0)
    dist = np.array([fam.value(r) for r in rows])
    P = np.array(coords)
    ks = sorted(set(dist.tolist()))
    balls = []
    for k in ks:
        c, r = C.exact_meb([p for p, d in zip(coords, dist) if d <= k])
        balls.append((k, np.array(c), r, float((dist <= k).mean())))
    all_c, all_r = balls[-1][1], balls[-1][2]
    pad = 1.05
    xlim = (all_c[0] - all_r * pad, all_c[0] + all_r * pad)
    ylim = (all_c[1] - all_r * pad, all_c[1] + all_r * pad)
    base = basemap_km(proj, xlim, ylim, fade=0.45)

    cmap = plt.get_cmap("viridis_r")
    norm = Normalize(0, len(ks) - 1)
    rank = {k: i for i, k in enumerate(ks)}
    fig, ax = plt.subplots(figsize=(10.5, 9.6))
    ax.imshow(base[0], extent=base[1], origin="upper", interpolation="bilinear", zorder=0)
    ax.scatter(P[:, 0], P[:, 1], s=4, c=[cmap(norm(rank[d])) for d in dist], linewidths=0, alpha=0.8, zorder=2)
    prev_r = None
    for k, c, r, frac in balls:
        col = cmap(norm(rank[k]))
        ax.add_patch(Circle(c, r, fill=False, ec=col, lw=2.0, zorder=4))
        if prev_r is None or abs(r - prev_r) > 0.6:   # label only where the ball changes
            ang = np.radians(205 + 9 * rank[k])
            ax.text(c[0] + r * np.cos(ang), c[1] + r * np.sin(ang), f"≤{int(k)}", fontsize=8.5,
                    color=V.INK, ha="center", va="center", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=col, lw=1.2))
        prev_r = r
    cs = np.array([b[1] for b in balls])
    ax.plot(cs[:, 0], cs[:, 1], color=V.INK, lw=1.4, zorder=6)
    ax.scatter(cs[:, 0], cs[:, 1], s=42, c=[cmap(norm(i)) for i in range(len(ks))], edgecolors=V.INK,
               linewidths=1.0, zorder=7)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    scale_bar(ax, base[1])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, ticks=range(0, len(ks), 2))
    cb.ax.set_yticklabels([f"≤{int(ks[i])}  ({100*balls[i][3]:.0f}%)" for i in range(0, len(ks), 2)], fontsize=8.5)
    cb.set_label("consent rule: DISTRICT ≤ k  (% of incidents consenting)", fontsize=9.5)
    ax.text(0, 1.085, "Chicago: the enclosing ball as consent rolls out district by district",
            transform=ax.transAxes, fontsize=15, fontweight="semibold", color=V.INK)
    ax.set_title(f"One circle per rule DISTRICT ≤ k: the smallest ball containing every consenting incident "
                 f"({len(rows):,} sampled locations).\nPoints are coloured by their district; the black line "
                 f"follows the ball's centre as k grows.", loc="left", fontsize=10, color=V.INK2)
    fig.text(0.98, 0.005, "Basemap © OpenStreetMap contributors © CARTO", ha="right", fontsize=7.5, color=V.MUTED)
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print("wrote", a.out)
    for k, c, r, f in balls:
        print(int(k), f"{100*f:.0f}%", f"r={r:.1f}")


if __name__ == "__main__":
    main()
