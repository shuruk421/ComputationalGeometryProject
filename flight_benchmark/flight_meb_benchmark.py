"""
Incremental vs decremental consent-aware MEB on the flight dataset.

Three experiments, all on 2D azimuthal-equidistant positions (see flight_data):

* sweep    - primary and contrast consent columns, consenting fraction swept
             from 5% to 95% at a fixed subsample size n;
* scaling  - oracle calls and run time vs n at two fixed thresholds of the
             primary column;
* units    - the same instances fed to db_consents in km instead of 1000-km
             units, to document the absolute-tolerance bug (see report).

Every run is checked against the exact MEB of the consenting points only
(meb_tools.exact_meb): the radius must agree to 1e-9 relative and every
consenting point must lie inside the returned ball.

For each seed a fresh uniform subsample of n points is drawn; the same
subsample is used for every threshold of that seed (paired comparison), and
the algorithm order alternates per seed (even: incremental first, odd:
decremental first) because other benchmarks share the machine.

Also recorded per run is the certificate lower bound
    LB = (# points outside the consenting MEB) + (# support points):
any algorithm must ask about every point outside the final ball (each could
otherwise enlarge it) and about the support points (each could otherwise
shrink it).

Usage:
    python flight_meb_benchmark.py [--csv PATH] [--seeds 6] [--quick]
"""

import argparse
import gc
import json
import os
import random
import time

import numpy as np

import flight_data as fd
from meb_tools import exact_meb, max_excess, to_points
from db_consents import Oracle, decremental_distance_based, incremental_distance_based

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

PRIMARY = "optin_date"
CONTRAST = "leg_random"
SWEEP_FRACS = [0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95]
SWEEP_N = 10_000
SCALING_FRACS = [0.10, 0.50]
SCALING_NS = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000]
UNITS_N = 5_000
UNITS_SEEDS = 3
ALGOS = {"incremental": incremental_distance_based, "decremental": decremental_distance_based}
REL_TOL = 1e-9


def subsample(fs, n, seed):
    rng = np.random.default_rng(1000 + seed)
    return np.sort(rng.choice(fs.n, size=min(n, fs.n), replace=False))


def run_instance(xy_km, mask, seed, unit=fd.ALGO_UNIT_KM, algos=("incremental", "decremental"),
                 skip=()):
    """Run the requested algorithms on one instance and check them."""
    cons_xy = xy_km[mask]
    gc_, gr, gsup = exact_meb(cons_xy, unit=fd.ALGO_UNIT_KM)
    d = np.linalg.norm(xy_km - gc_, axis=1)
    lb = int((d > gr * (1 + REL_TOL)).sum()) + int(len(gsup))
    rec = {
        "seed": seed, "n": int(len(mask)), "n_consenting": int(mask.sum()),
        "consent_pct": round(100.0 * float(mask.mean()), 3),
        "gt_radius_km": gr, "gt_center_km": [float(gc_[0]), float(gc_[1])],
        "gt_support_km": [[float(a), float(b)] for a, b in gsup],
        "lower_bound_calls": lb,
    }
    pts = to_points(xy_km, mask, unit)
    order = list(algos) if seed % 2 == 0 else list(reversed(algos))
    for name in order:
        if name in skip:
            rec[f"{name}_skipped"] = True
            continue
        gc.collect()
        random.seed(10_000 * (seed + 1) + (0 if name == "incremental" else 1))
        oracle = Oracle()
        t0 = time.perf_counter()
        center, radius = ALGOS[name](pts, oracle)
        elapsed = time.perf_counter() - t0
        ok, err, excess = False, None, None
        if center is not None and not isinstance(center, (int, float)):
            c_km = np.asarray(center, dtype=np.float64) * unit
            r_km = float(radius) * unit
            err = abs(r_km - gr) / gr if gr > 0 else abs(r_km)
            excess = max_excess(cons_xy, c_km, r_km)
            ok = err <= REL_TOL and excess <= REL_TOL * max(gr, 1.0)
        else:
            r_km = None
        rec.update({
            f"{name}_calls": int(oracle.get_call_count()),
            f"{name}_time_s": elapsed,
            f"{name}_radius_km": r_km,
            f"{name}_rel_err": err,
            f"{name}_max_excess_km": excess,
            f"{name}_ok": bool(ok),
        })
    return rec


def _summ(vals):
    v = np.asarray([x for x in vals if x is not None], dtype=np.float64)
    if len(v) == 0:
        return None
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max()), "median": float(np.median(v)),
            "count": int(len(v))}


def aggregate(records, keys):
    groups = {}
    for r in records:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for key, rs in sorted(groups.items()):
        row = dict(zip(keys, key))
        row["label"] = rs[0].get("label")
        row["seeds"] = len(rs)
        for f in ("consent_pct", "lower_bound_calls", "gt_radius_km",
                  "incremental_calls", "incremental_time_s",
                  "decremental_calls", "decremental_time_s"):
            row[f] = _summ([r.get(f) for r in rs])
        for a in ALGOS:
            row[f"{a}_calls_per_n"] = _summ([r[f"{a}_calls"] / r["n"] for r in rs if f"{a}_calls" in r])
            row[f"{a}_mismatches"] = sum(1 for r in rs if f"{a}_ok" in r and not r[f"{a}_ok"])
            row[f"{a}_skipped"] = sum(1 for r in rs if r.get(f"{a}_skipped"))
        out.append(row)
    return out


def sweep(fs, seeds, n=SWEEP_N, fracs=SWEEP_FRACS, columns=(PRIMARY, CONTRAST), log=print):
    masks = {}
    for col in columns:
        fn = fd.SWEEP_CANDIDATES[col][0]
        for f in fracs:
            m, label, thr = fn(fs, f)
            masks[(col, f)] = (m, label, thr)
    records = []
    t_start = time.perf_counter()
    for seed in range(seeds):
        idx = subsample(fs, n, seed)
        xy = fs.xy_km[idx]
        for col in columns:
            for f in fracs:
                m, label, thr = masks[(col, f)]
                rec = run_instance(xy, m[idx], seed)
                rec.update({"column": col, "target_frac": f, "label": label,
                            "threshold": thr if isinstance(thr, (int, float)) else str(thr)})
                records.append(rec)
                log(f"  sweep seed={seed} {col:<11} {label:<28} cons={rec['consent_pct']:5.1f}% "
                    f"LB={rec['lower_bound_calls']:5d} inc={rec['incremental_calls']:5d} "
                    f"({rec['incremental_time_s']:.2f}s) dec={rec['decremental_calls']:5d} "
                    f"({rec['decremental_time_s']:.2f}s) ok={rec['incremental_ok']}/{rec['decremental_ok']}")
    log(f"  sweep done in {time.perf_counter() - t_start:.0f}s")
    return records


def scaling(fs, seeds, ns=SCALING_NS, fracs=SCALING_FRACS, column=PRIMARY,
            max_run_s=60.0, log=print):
    fn = fd.SWEEP_CANDIDATES[column][0]
    records = []
    t_start = time.perf_counter()
    for f in fracs:
        m, label, thr = fn(fs, f)
        prev = None  # (n, mean decremental time)
        for n in ns:
            skip = ()
            if prev is not None:
                predicted = prev[1] * (n / prev[0]) ** 2
                if predicted > max_run_s:
                    skip = ("decremental",)
                    log(f"  scaling {label} n={n}: skipping decremental "
                        f"(predicted {predicted:.0f}s/run > {max_run_s:.0f}s)")
            dec_times = []
            for seed in range(seeds):
                idx = subsample(fs, n, seed)
                rec = run_instance(fs.xy_km[idx], m[idx], seed, skip=skip)
                rec.update({"column": column, "target_frac": f, "label": label,
                            "threshold": thr if isinstance(thr, (int, float)) else str(thr)})
                records.append(rec)
                if "decremental_time_s" in rec:
                    dec_times.append(rec["decremental_time_s"])
                log(f"  scaling {label:<24} n={n:6d} seed={seed} cons={rec['consent_pct']:5.1f}% "
                    f"LB={rec['lower_bound_calls']:5d} inc={rec['incremental_calls']:5d} "
                    f"({rec['incremental_time_s']:.2f}s) dec={rec.get('decremental_calls', '-')} "
                    f"({rec.get('decremental_time_s', float('nan')):.2f}s)")
            if dec_times:
                prev = (n, float(np.mean(dec_times)))
    log(f"  scaling done in {time.perf_counter() - t_start:.0f}s")
    return records


def units_check(fs, seeds=UNITS_SEEDS, n=UNITS_N, fracs=SWEEP_FRACS, columns=(PRIMARY, CONTRAST),
                log=print):
    """Same instances with coordinates in km (unit=1) instead of 1000 km."""
    records = []
    for seed in range(seeds):
        idx = subsample(fs, n, seed)
        xy = fs.xy_km[idx]
        for col in columns:
            fn = fd.SWEEP_CANDIDATES[col][0]
            for f in fracs:
                m, label, _ = fn(fs, f)
                rec = run_instance(xy, m[idx], seed, unit=1.0)
                rec.update({"column": col, "target_frac": f, "label": label, "unit_km": 1.0})
                records.append(rec)
    summ = {}
    for a in ALGOS:
        bad = [r for r in records if not r[f"{a}_ok"]]
        summ[a] = {
            "runs": len(records),
            "mismatches": len(bad),
            "max_rel_radius_err": max((r[f"{a}_rel_err"] or 0.0) for r in records),
            "max_excess_km": max((r[f"{a}_max_excess_km"] or 0.0) for r in records),
            "example": None if not bad else {
                k: bad[0][k] for k in ("column", "label", "seed", "n", "gt_radius_km",
                                       f"{a}_radius_km", f"{a}_calls", "lower_bound_calls")},
        }
        log(f"  units=km {a}: {len(bad)}/{len(records)} runs disagree with ground truth "
            f"(max rel radius err {summ[a]['max_rel_radius_err']:.3g})")
    return summ, records


def correctness_summary(sweep_recs, scaling_recs):
    out = {}
    for a in ALGOS:
        runs = [r for r in sweep_recs + scaling_recs if f"{a}_ok" in r]
        bad = [r for r in runs if not r[f"{a}_ok"]]
        out[a] = {
            "runs_checked": len(runs),
            "mismatches": len(bad),
            "max_rel_radius_err": max((r[f"{a}_rel_err"] or 0.0) for r in runs) if runs else None,
            "max_excess_km": max((r[f"{a}_max_excess_km"] or 0.0) for r in runs) if runs else None,
            "mismatch_examples": [{k: r.get(k) for k in ("column", "label", "n", "seed",
                                                        "gt_radius_km", f"{a}_radius_km")}
                                  for r in bad[:5]],
        }
    return out


def run(csv_path, seeds=6, quick=False, log=print):
    fs = fd.FlightSet(csv_path)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    common = {
        "primary_column": PRIMARY, "contrast_column": CONTRAST,
        "algo_unit_km": fd.ALGO_UNIT_KM, "points_in_pool": fs.n,
        "projection": fs.describe()["projection"], "seeds": seeds,
    }
    sweep_n = 2_000 if quick else SWEEP_N
    ns = [1_000, 2_000, 5_000] if quick else SCALING_NS

    log(f"Sweep: n={sweep_n}, {seeds} seeds, fractions {SWEEP_FRACS}")
    s_recs = sweep(fs, seeds, n=sweep_n, log=log)
    log(f"Scaling: n in {ns}, fractions {SCALING_FRACS}")
    n_recs = scaling(fs, seeds, ns=ns, log=log)
    log("Units check (km coordinates)")
    u_summ, u_recs = units_check(fs, seeds=min(UNITS_SEEDS, seeds), n=min(UNITS_N, sweep_n), log=log)

    sweep_out = dict(common, n=sweep_n, fractions=SWEEP_FRACS,
                     summary=aggregate(s_recs, ["column", "target_frac", "n"]), runs=s_recs)
    scaling_out = dict(common, fractions=SCALING_FRACS, ns=ns,
                       summary=aggregate(n_recs, ["column", "target_frac", "n"]), runs=n_recs)
    corr = {
        "rule": f"|r - r_gt| / r_gt <= {REL_TOL} and every consenting point inside the returned ball "
                f"(excess <= {REL_TOL} * r_gt); r_gt = exact MEB of the consenting points only",
        "benchmark_runs": correctness_summary(s_recs, n_recs),
        "km_units_check": {"n": min(UNITS_N, sweep_n), "seeds": min(UNITS_SEEDS, seeds),
                           "summary": u_summ},
    }
    for name, obj in (("sweep.json", sweep_out), ("scaling.json", scaling_out),
                      ("correctness.json", corr)):
        with open(os.path.join(RESULTS_DIR, name), "w") as fh:
            json.dump(obj, fh, indent=1)
        log(f"Wrote {os.path.join(RESULTS_DIR, name)}")
    return sweep_out, scaling_out, corr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=fd.DEFAULT_CSV)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--quick", action="store_true", help="small n, for a smoke test")
    args = ap.parse_args()
    run(args.csv, seeds=args.seeds, quick=args.quick)


if __name__ == "__main__":
    main()
