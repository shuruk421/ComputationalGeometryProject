"""
Consent-column analysis for the flight dataset.

For every candidate consent column this reports the consenting fraction and
how far the MEB of the consenting points moves away from the MEB of all
points.  Sweepable columns are evaluated at target fractions 5%..95%
(thresholds picked by quantile); binary flags at their only setting.

MEB shift = Hausdorff distance between the two discs, |c0 - c1| + |r0 - r1|,
reported in km and as % of the all-points radius r0.  A random per-point
consent at the same fraction is the "spatially random" baseline.

All MEBs here are exact MEBs of the whole cleaned sample (~800k points), not a
subsample.  The script also measures the 2D-vs-3D and projection questions.

Usage:
    python consent_analysis.py [--csv PATH]
"""

import argparse
import json
import math
import os

import numpy as np

import flight_data as fd
from meb_tools import disc_hausdorff, exact_meb

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
TARGET_FRACS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
U = fd.ALGO_UNIT_KM


def meb_shift(fs, mask, ref):
    c0, r0 = ref
    if mask.sum() == 0:
        return None
    c1, r1, _ = exact_meb(fs.xy_km[mask], unit=U)
    return {
        "center_km": [round(float(c1[0]), 3), round(float(c1[1]), 3)],
        "radius_km": round(r1, 3),
        "center_shift_km": round(float(np.linalg.norm(c1 - c0)), 2),
        "radius_change_pct": round(100 * (r1 - r0) / r0, 2),
        "hausdorff_km": round(disc_hausdorff(c0, r0, c1, r1), 2),
        "hausdorff_pct_r0": round(100 * disc_hausdorff(c0, r0, c1, r1) / r0, 2),
    }


def geometry_checks(fs, ref):
    """2D AEQD vs true-scale 3D (x, y, altitude) and vs the exact spherical cap."""
    c0, r0 = ref
    alt_km = fs.c["baro"].astype(np.float64) / 1000.0
    xyz = np.column_stack([fs.xy_km, alt_km])
    c3, r3, _ = exact_meb(xyz, unit=U)
    # ECEF on the sphere: the Euclidean MEB of points on a sphere has the same
    # axis as the minimal spherical cap; cap great-circle radius = R*asin(rho/R)
    # when the centre lies inside the sphere.
    R = fd.EARTH_RADIUS_KM
    la, lo = np.radians(fs.c["lat"]), np.radians(fs.c["lon"])
    ecef = R * np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    ce, re, _ = exact_meb(ecef, unit=U)
    cap_gc_km = R * math.asin(min(re / R, 1.0))
    cdir = ce / np.linalg.norm(ce)
    cap_lat = math.degrees(math.asin(cdir[2]))
    cap_lon = math.degrees(math.atan2(cdir[1], cdir[0]))
    clat, clon = fs.proj.inverse(c0[0], c0[1])
    # great-circle distance between the AEQD MEB centre and the cap centre
    p1, p2 = math.radians(float(clat)), math.radians(cap_lat)
    dl = math.radians(cap_lon - float(clon))
    gc = R * math.acos(min(1.0, math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dl)))
    return {
        "radius_2d_km": round(r0, 3),
        "radius_3d_true_scale_altitude_km": round(r3, 3),
        "radius_3d_minus_2d_m": round(1000 * (r3 - r0), 1),
        "center_3d_vs_2d_horizontal_shift_m": round(1000 * float(np.linalg.norm(c3[:2] - c0)), 1),
        "altitude_range_km": [round(float(alt_km.min()), 3), round(float(alt_km.max()), 3)],
        "spherical_cap_radius_great_circle_km": round(cap_gc_km, 2),
        "aeqd_vs_cap_radius_pct": round(100 * (r0 - cap_gc_km) / cap_gc_km, 2),
        "aeqd_center_vs_cap_center_km": round(gc, 1),
    }


def run(csv_path, out_path=None, verbose=True):
    fs = fd.FlightSet(csv_path)
    info = fs.describe()
    c0, r0, sup0 = exact_meb(fs.xy_km, unit=U)
    ref = (c0, r0)
    slat, slon = fs.proj.inverse(sup0[:, 0], sup0[:, 1])
    out = {
        "dataset": info,
        "all_points_meb": {
            "center_km": [round(float(c0[0]), 3), round(float(c0[1]), 3)],
            "radius_km": round(r0, 3),
            "support_latlon": [[round(float(a), 4), round(float(b), 4)] for a, b in zip(slat, slon)],
        },
        "geometry": geometry_checks(fs, ref),
        "fixed": {},
        "sweep": {},
        "random_baseline": {},
    }
    if verbose:
        print(json.dumps(info, indent=1))
        print("All-points MEB radius %.1f km; support %s" % (r0, out["all_points_meb"]["support_latlon"]))
        print("Geometry:", out["geometry"])

    for name, (mask, story) in fd.fixed_candidates(fs).items():
        row = {"story": story, "consent_pct": round(100 * mask.mean(), 3),
               "n_consenting": int(mask.sum())}
        row.update(meb_shift(fs, mask, ref) or {})
        out["fixed"][name] = row

    for f in TARGET_FRACS:
        rows = []
        for seed in range(5):
            m, _, _ = fd.consent_point_random(fs, f, seed=seed)
            rows.append(meb_shift(fs, m, ref)["hausdorff_pct_r0"])
        out["random_baseline"][f"{f:.2f}"] = round(float(np.mean(rows)), 3)

    for name, (fn, story) in fd.SWEEP_CANDIDATES.items():
        rows = []
        for f in TARGET_FRACS:
            mask, label, thr = fn(fs, f)
            row = {"target_frac": f, "label": label,
                   "threshold": thr if isinstance(thr, (int, float)) else str(thr),
                   "consent_pct": round(100 * mask.mean(), 2)}
            row.update(meb_shift(fs, mask, ref) or {})
            rows.append(row)
        hs = [r["hausdorff_pct_r0"] for r in rows if "hausdorff_pct_r0" in r]
        base = [out["random_baseline"][f"{r['target_frac']:.2f}"] for r in rows if "hausdorff_pct_r0" in r]
        out["sweep"][name] = {
            "story": story,
            "rows": rows,
            "consent_pct_range": [min(r["consent_pct"] for r in rows), max(r["consent_pct"] for r in rows)],
            "mean_hausdorff_pct_r0": round(float(np.mean(hs)), 2),
            "mean_random_baseline_pct_r0": round(float(np.mean(base)), 2),
        }
        if verbose:
            print(f"\n{name}: {story}")
            for r in rows:
                rb = out["random_baseline"]["%.2f" % r["target_frac"]]
                print(f"  {r['label']:<32} consent {r['consent_pct']:6.2f}%  "
                      f"shift {r.get('center_shift_km', float('nan')):8.1f} km  "
                      f"radius {r.get('radius_change_pct', float('nan')):+7.2f}%  "
                      f"Hausdorff {r.get('hausdorff_pct_r0', float('nan')):6.2f}% r0  "
                      f"(random {rb:.2f}%)")

    if verbose:
        print("\nFixed columns:")
        for k, r in out["fixed"].items():
            print(f"  {k:<16} consent {r['consent_pct']:8.3f}%  Hausdorff {r.get('hausdorff_pct_r0', float('nan')):6.2f}% r0")

    out_path = out_path or os.path.join(RESULTS_DIR, "consent_candidates.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1)
    if verbose:
        print(f"\nWrote {out_path}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=fd.DEFAULT_CSV)
    args = ap.parse_args()
    run(args.csv)


if __name__ == "__main__":
    main()
