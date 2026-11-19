"""
Numerical-scale check for db_consents.

db_consents compares squared distances with an *absolute* tolerance
(GEOMETRY_TOLERANCE = 1e-10). decremental_distance_based only queries points
with |d^2 - r^2| <= 1e-10 as boundary points, so when coordinates are large
(e.g. kilometres, r^2 ~ 1e7, where one ulp of r^2 is ~1e-9) the true support
points can be missed and the loop stops early with a ball that still rests on
non-consenting points.

This script runs both algorithms on the same point sets at several coordinate
scales (FL airports in units of 1000 km, km and m; and the CC sample) and
records how many runs match the ground truth. Output:
results/scale_check.csv
"""

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import datasets  # noqa: E402
from meb_common import (RADIUS_RTOL, RESULTS_DIR, consent_mask, ground_truth_meb,  # noqa: E402
                        run_algo, subsample, threshold_for_fraction)


def main():
    rows = []
    cases = [("flights", [1.0, 1000.0, 1e6], ["1000 km", "km", "m"]),
             ("creditcard", [1.0, 1e3, 1e5], ["z-score", "z x 1e3", "z x 1e5"])]
    for key, scales, names in cases:
        ds = datasets.load(key)
        cand = ds.candidates[ds.chosen]
        for f in (0.2, 0.5, 0.8):
            t, _ = threshold_for_fraction(cand.values, cand.sign, f)
            for seed in range(5):
                idx = subsample(len(ds.X), 1000, 1000 + seed)
                m = consent_mask(cand.values[idx], cand.sign, t)
                for s, name in zip(scales, names):
                    X = ds.X[idx] * s
                    c_gt, r_gt = ground_truth_meb(X[m])
                    for algo in ("incremental", "decremental"):
                        res = run_algo(algo, X, m, seed=seed)
                        ok = abs(res["radius"] - r_gt) <= RADIUS_RTOL * r_gt
                        rows.append({"dataset": key, "units": name, "scale": s,
                                     "target_frac": f, "seed": seed, "algo": algo,
                                     "calls": res["calls"], "radius": res["radius"],
                                     "gt_radius": r_gt, "rel_err": (res["radius"] - r_gt) / r_gt,
                                     "match": int(ok)})
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "scale_check.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{'dataset':11s} {'units':9s} {'algo':12s} {'match':>7s} {'mean calls':>10s} {'max rel err':>12s}")
    for key, scales, names in cases:
        for name in names:
            for algo in ("incremental", "decremental"):
                sel = [r for r in rows if r["dataset"] == key and r["units"] == name and r["algo"] == algo]
                print(f"{key:11s} {name:9s} {algo:12s} {sum(r['match'] for r in sel):3d}/{len(sel):<3d} "
                      f"{np.mean([r['calls'] for r in sel]):10.1f} "
                      f"{max(abs(r['rel_err']) for r in sel):12.2e}")
    print("->", out)


if __name__ == "__main__":
    main()
