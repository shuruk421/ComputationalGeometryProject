"""
Incremental vs decremental consent-MEB benchmark on Chicago crimes.

Experiments (each point: >= 5 seeds; a seed fixes the sample draw and the
algorithms' internal shuffles; incremental/decremental order alternates per
seed so background load hits both equally):

* consent sweeps: consenting fraction ~5% .. 95% via quantile cutoffs
    - DISTRICT <= k   (primary: spatially clustered consent), n = 2,000
    - TIMESTAMP >= t  (contrast: spatially random-looking consent), n = 2,000 and 20,000
* scaling with n at a fixed ~50% cutoff for both families.
  Decremental is skipped once its projected run time exceeds a cap (it is
  roughly quadratic when many non-consenting points must be peeled).

Correctness: every run's radius is compared with the ground-truth MEB of the
consenting points only (db_consents' Welzl on consenting-only points with a
throwaway Oracle), whose optimality is certified independently (encloses all
consenting points, and its centre lies in the convex hull of its support).
Also records a lower bound on consent requests: every non-consenting point
outside the consenting MEB, plus the support points, must be asked by any
correct algorithm.

Ablations (correctness hazards): duplicates kept; coordinates in feet.

Writes results/benchmark_runs.json, results/benchmark_summary.json,
results/correctness_ablations.json.
"""

import json
import math
import os
import statistics
import time

import consent as C
import crime_data as cd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

SWEEP_TARGETS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
SWEEPS = [
    {"id": "district_n2000", "family": "DISTRICT", "n": 2000},
    {"id": "timestamp_n2000", "family": "TIMESTAMP", "n": 2000},
    {"id": "timestamp_n20000", "family": "TIMESTAMP", "n": 20000},
]
SCALING = [
    {"id": "timestamp_scaling", "family": "TIMESTAMP", "target": 0.5,
     "sizes": [1000, 2000, 5000, 10000, 20000, 50000, 100000, 150000]},
    {"id": "district_scaling", "family": "DISTRICT", "target": 0.5,
     "sizes": [500, 1000, 2000, 4000, 10000, 20000, 50000, 100000, 150000]},
]
DEC_PROJECTED_CAP_S = 45.0  # skip decremental when projected per-run time exceeds this
RADIUS_RTOL = 1e-9


# --------------------------------------------------------------------------
# Ground truth and checks
# --------------------------------------------------------------------------

def certify_meb(coords, center, radius, rtol=1e-7):
    """Independent optimality certificate for a 2-D enclosing disk.

    A disk is THE minimum enclosing disk iff it encloses every point and its
    centre lies in the convex hull of the points on its boundary, i.e. no open
    half-plane through the centre misses all boundary points (max angular gap
    between boundary points, seen from the centre, <= pi)."""
    cx, cy = center[0], center[1]
    tol = max(radius * rtol, 1e-9)
    angles, worst = [], 0.0
    for p in coords:
        d = math.hypot(p[0] - cx, p[1] - cy)
        worst = max(worst, d - radius)
        if abs(d - radius) <= tol:
            angles.append(math.atan2(p[1] - cy, p[0] - cx))
    encloses = worst <= tol
    if len(coords) == 1:
        return encloses, 1
    angles.sort()
    gaps = [b - a for a, b in zip(angles, angles[1:])]
    if angles:
        gaps.append(angles[0] + 2 * math.pi - angles[-1])
    in_hull = len(angles) >= 2 and max(gaps) <= math.pi + 1e-6
    return encloses and in_hull, len(angles)


def ground_truth(rows, coords, flags, seed):
    cons = [c for c, f in zip(coords, flags) if f]
    if not cons:
        return {"center": None, "radius": 0.0, "certified": True, "support": 0, "forced": 0}
    center, radius = C.exact_meb(cons, seed=seed)
    ok, n_support = certify_meb(cons, center, radius)
    r2 = radius * radius * (1 + 2 * RADIUS_RTOL) + 1e-12
    forced = sum(1 for p, f in zip(coords, flags)
                 if not f and (p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2 > r2)
    return {"center": center, "radius": radius, "certified": ok, "support": n_support,
            "forced": forced}


def check_run(res, gt, cons_coords):
    if gt["center"] is None:
        return res["radius"] == 0.0
    if res["center"] is None:
        return False
    if abs(res["radius"] - gt["radius"]) > RADIUS_RTOL * gt["radius"] + 1e-9:
        return False
    worst = C.max_dist(cons_coords, res["center"])
    return worst <= res["radius"] * (1 + RADIUS_RTOL) + 1e-9


# --------------------------------------------------------------------------
# One measurement point
# --------------------------------------------------------------------------

class SampleCache:
    def __init__(self, records, proj):
        self.records, self.proj, self._c = records, proj, {}

    def get(self, n, seed):
        key = (n, seed)
        if key not in self._c:
            if len(self._c) > 24:
                self._c.clear()
            self._c[key] = cd.draw_points(self.records, self.proj, n, seed=seed)
        return self._c[key]


def run_point(samples, exp_id, fam, cutoff, target, n, seed, algorithms=C.ALGORITHMS):
    rows, coords = samples.get(n, seed)
    rule = fam.rule(cutoff)
    flags = [bool(rule(r)) for r in rows]
    pts = [(c, f) for c, f in zip(coords, flags)]
    cons_coords = [c for c, f in zip(coords, flags) if f]
    gt = ground_truth(rows, coords, flags, seed=seed)
    order = list(algorithms) if seed % 2 == 0 else list(reversed(algorithms))
    out = []
    for algo in order:
        res = C.run_algorithm(algo, pts, seed=1000 + seed)
        ok = check_run(res, gt, cons_coords)
        out.append({
            "experiment": exp_id, "family": fam.name, "label": fam.label(cutoff),
            "cutoff": cutoff, "target": target, "n": len(rows), "seed": seed,
            "n_consenting": len(cons_coords), "pct_consenting": 100.0 * len(cons_coords) / len(rows),
            "algorithm": algo, "calls": res["calls"], "calls_per_n": res["calls"] / len(rows),
            "seconds": res["seconds"], "radius": res["radius"], "center": res["center"],
            "gt_radius": gt["radius"], "gt_center": gt["center"], "gt_certified": gt["certified"],
            "support": gt["support"], "forced": gt["forced"],
            "lower_bound": gt["forced"] + gt["support"], "correct": ok,
        })
    return out


def _log(r):
    flag = "" if r["correct"] else "  <-- MISMATCH"
    print(f"  {r['experiment']:18s} {r['label']:26s} n={r['n']:6d} s={r['seed']} "
          f"{r['pct_consenting']:5.1f}% {r['algorithm']:11s} calls={r['calls']:6d} "
          f"t={r['seconds']:7.3f}s{flag}", flush=True)


def run_benchmark(records, proj, seeds=5, quick=False, verbose=True):
    t_start = time.time()
    samples = SampleCache(records, proj)
    runs = []
    sweeps = SWEEPS if not quick else [dict(s, n=min(s["n"], 1000)) for s in SWEEPS]
    targets = SWEEP_TARGETS if not quick else (0.05, 0.5, 0.95)
    for sw in sweeps:
        fam = C.THRESHOLD_FAMILIES[sw["family"]]
        cutoffs = []
        for t in targets:
            c = fam.cutoff_for_fraction(records, t)
            if c not in [x for x, _ in cutoffs]:
                cutoffs.append((c, t))
        for seed in range(seeds):
            for cutoff, t in cutoffs:
                for r in run_point(samples, sw["id"], fam, cutoff, t, sw["n"], seed):
                    runs.append(r)
                    if verbose:
                        _log(r)

    for sc in SCALING:
        fam = C.THRESHOLD_FAMILIES[sc["family"]]
        cutoff = fam.cutoff_for_fraction(records, sc["target"])
        sizes = sc["sizes"] if not quick else [s for s in sc["sizes"] if s <= 5000]
        dec_on, prev = True, None
        for n in sizes:
            if dec_on and prev is not None:
                pn, pt = prev
                projected = pt * (n / pn) ** 2
                if projected > DEC_PROJECTED_CAP_S:
                    dec_on = False
                    if verbose:
                        print(f"  {sc['id']}: skipping decremental from n={n} "
                              f"(projected {projected:.0f}s/run)", flush=True)
            algos = C.ALGORITHMS if dec_on else ("incremental",)
            dec_times = []
            for seed in range(seeds):
                for r in run_point(samples, sc["id"], fam, cutoff, sc["target"], n, seed, algos):
                    runs.append(r)
                    if r["algorithm"] == "decremental":
                        dec_times.append(r["seconds"])
                    if verbose:
                        _log(r)
            if dec_times:
                prev = (n, statistics.mean(dec_times))

    meta = {"seeds": seeds, "quick": quick, "sweeps": sweeps, "targets": list(targets),
            "scaling": SCALING, "dec_projected_cap_s": DEC_PROJECTED_CAP_S,
            "radius_rtol": RADIUS_RTOL, "seconds": time.time() - t_start}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "benchmark_runs.json"), "w") as f:
        json.dump({"meta": meta, "runs": runs}, f, indent=0, default=float)
    summary = summarize(runs)
    with open(os.path.join(RESULTS, "benchmark_summary.json"), "w") as f:
        json.dump({"meta": meta, "points": summary}, f, indent=1)
    n_bad = sum(1 for r in runs if not r["correct"])
    n_uncert = len({(r["experiment"], r["cutoff"], r["n"], r["seed"]) for r in runs if not r["gt_certified"]})
    if verbose:
        print(f"[bench] {len(runs)} runs in {meta['seconds']:.0f}s; mismatches: {n_bad}; "
              f"uncertified ground truths: {n_uncert}", flush=True)
    return runs, summary


def summarize(runs):
    groups = {}
    for r in runs:
        key = (r["experiment"], r["cutoff"], r["n"], r["algorithm"])
        groups.setdefault(key, []).append(r)
    out = []
    for (exp, cutoff, n, algo), rs in groups.items():
        def stats(field):
            v = [x[field] for x in rs]
            return {"mean": statistics.mean(v), "min": min(v), "max": max(v),
                    "sd": statistics.stdev(v) if len(v) > 1 else 0.0}
        out.append({
            "experiment": exp, "family": rs[0]["family"], "label": rs[0]["label"],
            "cutoff": cutoff, "target": rs[0]["target"], "n": n, "algorithm": algo,
            "seeds": len(rs), "pct_consenting": stats("pct_consenting"),
            "calls": stats("calls"), "calls_per_n": stats("calls_per_n"),
            "seconds": stats("seconds"), "lower_bound": stats("lower_bound"),
            "forced": stats("forced"), "radius": stats("gt_radius"),
            "mismatches": sum(1 for x in rs if not x["correct"]),
        })
    out.sort(key=lambda p: (p["experiment"], p["n"], p["pct_consenting"]["mean"], p["algorithm"]))
    return out


# --------------------------------------------------------------------------
# Ablations: why we dedupe and why we use kilometres
# --------------------------------------------------------------------------

def run_ablations(records, proj, seeds=4, verbose=True):
    t0 = time.time()
    fam_t = C.THRESHOLD_FAMILIES["TIMESTAMP"]
    rules = {
        "ARREST": C.BOOLEAN_RULES["ARREST"][1],
        "LOCATION outdoor public": C.BOOLEAN_RULES["LOCATION outdoor public"][1],
        "TIMESTAMP 50%": fam_t.rule(fam_t.cutoff_for_fraction(records, 0.5)),
        "TIMESTAMP 10%": fam_t.rule(fam_t.cutoff_for_fraction(records, 0.1)),
    }
    out = {"duplicates_kept": [], "feet_units": []}

    # (a) duplicates kept: n records, many sharing a geocode.
    n_dup = 50000
    for seed in range(seeds):
        rows, coords = cd.draw_points(records, proj, n_dup, seed=seed, dedupe=False)
        n_unique = len(set(map(tuple, coords)))
        for name, rule in rules.items():
            flags = [bool(rule(r)) for r in rows]
            pts = list(zip(coords, flags))
            cons = [c for c, f in pts if f]
            gc, gr = C.exact_meb(cons, seed=seed)
            entry = {"rule": name, "seed": seed, "n_records": len(rows), "n_unique_locations": n_unique,
                     "gt_radius": gr}
            for algo in C.ALGORITHMS:
                res = C.run_algorithm(algo, pts, seed=1000 + seed)
                entry[algo] = {"radius": res["radius"], "calls": res["calls"],
                               "correct": abs(res["radius"] - gr) <= RADIUS_RTOL * gr + 1e-9}
            out["duplicates_kept"].append(entry)
            if verbose:
                print(f"  [dup] seed={seed} {name:24s} r_true={gr:.3f} "
                      + " ".join(f"{a}={'ok' if entry[a]['correct'] else 'WRONG'}" for a in C.ALGORITHMS),
                      flush=True)

    # (b) same deduplicated points expressed in feet instead of km.
    ft = 1000.0 / 0.3048
    rule = rules["TIMESTAMP 50%"]
    for sample_seed in range(3):
        rows, coords = cd.draw_points(records, proj, 2000, seed=sample_seed)
        flags = [bool(rule(r)) for r in rows]
        for unit, scale in (("km", 1.0), ("feet", ft)):
            pts = [([c[0] * scale, c[1] * scale], f) for c, f in zip(coords, flags)]
            cons = [p for p, f in pts if f]
            gc, gr = C.exact_meb(cons)
            wrong = {a: 0 for a in C.ALGORITHMS}
            for s in range(10):
                for algo in C.ALGORITHMS:
                    res = C.run_algorithm(algo, pts, seed=s)
                    if abs(res["radius"] - gr) > RADIUS_RTOL * gr + 1e-9:
                        wrong[algo] += 1
            out["feet_units"].append({"sample_seed": sample_seed, "unit": unit, "runs_per_algo": 10,
                                      "gt_radius": gr, "wrong": wrong})
            if verbose:
                print(f"  [units] sample={sample_seed} {unit:4s} r_true={gr:.4g} wrong/10: {wrong}", flush=True)
    out["seconds"] = time.time() - t0
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "correctness_ablations.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Incremental vs decremental consent-MEB benchmark.")
    ap.add_argument("--csv", default=cd.DEFAULT_CSV)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test configuration")
    ap.add_argument("--skip-ablations", action="store_true")
    a = ap.parse_args()
    recs, meta, proj = cd.load(a.csv)
    run_benchmark(recs, proj, seeds=a.seeds, quick=a.quick)
    if not a.skip_ablations:
        run_ablations(recs, proj)
