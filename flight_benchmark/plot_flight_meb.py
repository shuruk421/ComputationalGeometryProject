"""
Single-map view: consenting vs non-consenting flight positions and the MEB of
the consenting ones, for one consent column and threshold.

Superseded for the full study by run_flight_suite.py (small multiples and
benchmarks); kept as a quick interactive tool.  Compared with the first
version of this script it now
  * samples the whole 9.3 GB file (cached, see flight_data.py) instead of
    the first 300k rows, which covered only a few flights from March 2016;
  * uses the consent columns from flight_data.SWEEP_CANDIDATES (ALERT/SPI are
    0.2% / 0.07% of positions and make poor consent signals);
  * computes the MEB in km (azimuthal-equidistant) instead of in raw degrees;
  * draws in Web-Mercator space so points register with the basemap tiles.

Usage:
    python plot_flight_meb.py --consent optin_date --frac 0.25 --algo decremental
"""

import argparse
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import flight_data as fd  # noqa: E402
import map_tiles  # noqa: E402
from flight_meb_benchmark import subsample  # noqa: E402
from meb_tools import support_points, to_points  # noqa: E402
from db_consents import Oracle, decremental_distance_based, incremental_distance_based  # noqa: E402
import plot_flight_figures as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=fd.DEFAULT_CSV)
    ap.add_argument("--consent", default="optin_date", choices=sorted(fd.SWEEP_CANDIDATES))
    ap.add_argument("--frac", type=float, default=0.25, help="target consenting fraction")
    ap.add_argument("--algo", choices=["decremental", "incremental"], default="decremental")
    ap.add_argument("--n-points", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zoom", type=int, default=5)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    P._style()
    fs = fd.FlightSet(args.csv)
    idx = subsample(fs, args.n_points, args.seed)
    xy = fs.xy_km[idx]
    mask, label, _ = fd.SWEEP_CANDIDATES[args.consent][0](fs, args.frac)
    mask = mask[idx]
    print(f"{label}: {mask.sum():,} / {len(mask):,} consenting ({100 * mask.mean():.1f}%)")

    random.seed(args.seed)
    oracle = Oracle()
    algo = incremental_distance_based if args.algo == "incremental" else decremental_distance_based
    center, radius = algo(to_points(xy, mask, fd.ALGO_UNIT_KM), oracle)
    c = np.asarray(center) * fd.ALGO_UNIT_KM
    r = float(radius) * fd.ALGO_UNIT_KM
    sup = support_points(xy[mask], c, r)
    print(f"MEB radius {r:,.1f} km, {oracle.get_call_count()} oracle calls ({args.algo})")

    lat, lon = fs.c["lat"][idx], fs.c["lon"][idx]
    clat, clon = fs.proj.circle_latlon(c[0], c[1], r)
    lat_lo, lat_hi = min(lat.min(), clat.min()) - 0.6, max(lat.max(), clat.max()) + 0.6
    lon_lo, lon_hi = min(lon.min(), clon.min()) - 0.8, max(lon.max(), clon.max()) + 0.8
    img, ext = map_tiles.basemap(lat_lo, lat_hi, lon_lo, lon_hi, zoom=args.zoom)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, extent=ext, origin="upper", interpolation="lanczos")
    my = map_tiles.merc_y(lat)
    ax.scatter(lon[~mask], my[~mask], s=2, c=P.C_NONC, alpha=0.55, lw=0,
               label=f"non-consenting ({(~mask).sum():,})")
    ax.scatter(lon[mask], my[mask], s=3, c=P.C_CONS, alpha=0.8, lw=0,
               label=f"consenting ({mask.sum():,})")
    ax.plot(clon, map_tiles.merc_y(clat), color=P.INK, lw=1.8, label=f"MEB, r = {r:,.0f} km")
    slat, slon = fs.proj.inverse(sup[:, 0], sup[:, 1])
    ax.scatter(slon, map_tiles.merc_y(slat), s=50, c=P.INK, edgecolors=P.SURFACE, linewidths=1.6,
               zorder=5, label="support points")
    ax.set_xlim(lon_lo, lon_hi)
    ax.set_ylim(map_tiles.merc_y(lat_lo), map_tiles.merc_y(lat_hi))
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.legend(loc="upper left", fontsize=9, frameon=True, facecolor=P.SURFACE, edgecolor="none",
              labelcolor=P.INK2)
    ax.set_title(f"{label} - {args.algo}: {oracle.get_call_count():,} oracle calls", loc="left")
    ax.text(0.995, 0.005, map_tiles.ATTRIBUTION, transform=ax.transAxes, fontsize=7,
            color=P.INK2, ha="right", va="bottom")
    out = args.output or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      f"flight_meb_{args.consent}_{args.algo}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
