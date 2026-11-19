"""
Evaluate candidate consent columns for the Chicago crimes MEB experiments.

For every candidate (boolean rule, or threshold family at several target
fractions) on one deduplicated sample of n locations:

* consenting fraction (in the sample, and record-level in the full reservoir);
* MEB shift: Hausdorff distance between the consenting-only MEB and the
  all-points MEB (|dc| + |dr|), in km and as % of the all-points radius;
* the same shift for RANDOM consent at the same fraction (mean of a few seeds),
  so "spatially non-random" is measured rather than eyeballed;
* forced requests: share of points that are non-consenting AND outside the
  consenting MEB. Any correct algorithm has to ask every one of them (if one
  of them consented the answer would change), so it is a lower bound on
  oracle calls / n.

Writes results/consent_candidates.json.
"""

import json
import math
import os
import random
import time

import consent as C
import crime_data as cd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
TARGETS = (0.10, 0.25, 0.50, 0.75, 0.90)


def evaluate(rows, coords, rule, all_c, all_r, n_random=20, seed=0):
    flags = [bool(rule(r)) for r in rows]
    cons = [c for c, f in zip(coords, flags) if f]
    k = len(cons)
    out = {"n": len(rows), "n_consenting": k, "pct_consenting": 100.0 * k / len(rows)}
    if k == 0:
        return out
    c, r = C.exact_meb(cons)
    shift = C.disk_shift(c, r, all_c, all_r)
    r2 = r * r * (1 + 1e-9) + 1e-12
    forced = sum(1 for p, f in zip(coords, flags)
                 if not f and (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 > r2)
    rnd = []
    rng = random.Random(seed)
    for _ in range(n_random):
        sub = rng.sample(coords, k)
        rc, rr = C.exact_meb(sub)
        rnd.append(C.disk_shift(rc, rr, all_c, all_r))
    rnd_mean = sum(rnd) / len(rnd)
    out.update({
        "center_shift_km": math.hypot(c[0] - all_c[0], c[1] - all_c[1]),
        "radius_km": r,
        "radius_change_km": r - all_r,
        "shift_km": shift,
        "shift_pct_of_radius": 100.0 * shift / all_r,
        "random_shift_km": rnd_mean,
        "shift_vs_random": shift / rnd_mean if rnd_mean >= 0.01 else None,
        "forced_pct": 100.0 * forced / len(rows),
    })
    return out


def run(records, proj, n=20000, seed=0, meta=None, verbose=True):
    t0 = time.time()
    rows, coords = cd.draw_points(records, proj, n, seed=seed)
    all_c, all_r = C.exact_meb(coords)
    result = {"n": len(rows), "seed": seed, "all_points_radius_km": all_r,
              "all_points_center_km": all_c, "sample_meta": meta or {},
              "boolean": [], "threshold": []}

    for name, (desc, rule) in C.BOOLEAN_RULES.items():
        ev = evaluate(rows, coords, rule, all_c, all_r)
        ev.update({"candidate": name, "description": desc,
                   "pct_records": 100.0 * sum(1 for r in records if rule(r)) / len(records)})
        result["boolean"].append(ev)
        if verbose:
            _print(name, ev)

    for fam in C.THRESHOLD_FAMILIES.values():
        for t in TARGETS:
            cut = fam.cutoff_for_fraction(records, t)
            rule = fam.rule(cut)
            ev = evaluate(rows, coords, rule, all_c, all_r)
            ev.update({"candidate": fam.name, "story": fam.story, "target": t,
                       "cutoff": cut, "label": fam.label(cut),
                       "pct_records": 100.0 * fam.fraction(records, cut)})
            result["threshold"].append(ev)
            if verbose:
                _print(fam.label(cut), ev)
    result["seconds"] = time.time() - t0
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "consent_candidates.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    if verbose:
        print(f"[consent] all-points MEB radius {all_r:.2f} km; wrote {path} ({result['seconds']:.0f}s)")
    return result


def _print(name, ev):
    if "shift_km" not in ev:
        print(f"  {name:32s} {ev['pct_consenting']:5.1f}%  (no consenting points)")
        return
    print(f"  {name:32s} {ev['pct_consenting']:5.1f}%  shift {ev['shift_km']:5.2f} km "
          f"({ev['shift_pct_of_radius']:4.1f}%)  random {ev['random_shift_km']:5.2f} km  "
          f"x{_ratio(ev['shift_vs_random'])}  forced {ev['forced_pct']:4.1f}%", flush=True)


def _ratio(v):
    return "  n/a" if v is None else f"{v:5.1f}"


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", default=cd.DEFAULT_CSV)
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args()
    recs, meta, proj = cd.load(a.csv)
    run(recs, proj, n=a.n, meta=meta)
