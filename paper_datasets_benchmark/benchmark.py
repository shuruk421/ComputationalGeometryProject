"""
Incremental vs decremental consent-aware MEB on the paper's datasets.

For every dataset:
  * sweep: for each target consenting fraction (thresholds = quantiles of the
    chosen consent column over the whole dataset) and each seed (a fresh random
    subsample of n points), run incremental_distance_based and
    decremental_distance_based (order alternates by seed), under two consent
    mechanisms: 'column' (the threshold rule) and 'random' (Bernoulli consent at
    the same rate, as a spatially-random contrast);
  * vs_n: the same at one fixed threshold (~50%) for growing n.

Every run is checked against the ground-truth MEB of the consenting points
only, which is itself certified (enclosure + centre in the convex hull of its
boundary points). Raw rows go to results/<dataset>_sweep.csv and
results/<dataset>_vs_n.csv, a summary to results/<dataset>_summary.json.
"""

import argparse
import csv
import json
import os
import sys
import time
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import datasets  # noqa: E402
from meb_common import (ALGOS, RADIUS_RTOL, RESULTS_DIR, certify_meb, consent_mask,  # noqa: E402
                        ground_truth_meb, run_algo, subsample, threshold_for_fraction)

FRACS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
VS_N_SIZES = (250, 500, 1000, 2000, 4000, 8000, 16000, 32000)


def one_setting(X, m, seed, run_tag):
    """GT + both algorithms on points X with consent mask m. Returns rows."""
    Xc = X[m]
    c_gt, r_gt = ground_truth_meb(Xc)
    enc, opt = certify_meb(Xc, c_gt, r_gt)
    if c_gt is not None:
        d = np.linalg.norm(X - c_gt[None, :], axis=1)
        outside_nc = int(np.sum(~m & (d > r_gt * (1 + 1e-9))))
    else:
        outside_nc = int(np.sum(~m))
    order = ALGOS if seed % 2 == 0 else ALGOS[::-1]
    rows = []
    for k, algo in enumerate(order):
        res = run_algo(algo, X, m, seed=zlib.crc32(repr((run_tag, seed)).encode()))
        if c_gt is None:
            match = res["radius"] == 0.0
            cdist = 0.0
        else:
            match = abs(res["radius"] - r_gt) <= RADIUS_RTOL * max(r_gt, 1e-300)
            cdist = (float(np.linalg.norm(np.asarray(res["center"]) - c_gt))
                     if res["center"] is not None else float("inf"))
            match = match and cdist <= 1e-6 * max(r_gt, 1e-12)
        rows.append({
            "algo": algo, "run_order": k, "calls": res["calls"],
            "seconds": round(res["seconds"], 6), "radius": res["radius"],
            "gt_radius": r_gt, "center_err": cdist, "match": int(bool(match)),
            "gt_encloses": int(enc), "gt_optimal": int(opt),
            "n_consent": int(m.sum()), "outside_nonconsent": outside_nc,
            "gt_cx": None if c_gt is None else float(c_gt[0]),
            "gt_cy": None if c_gt is None else float(c_gt[1]),
        })
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run_dataset(key, n, seeds, fracs, mechanisms, vs_n_sizes, vs_n_frac, log=print):
    ds = datasets.load(key)
    N = len(ds.X)
    cand = ds.candidates[ds.chosen]
    n_eff = min(n, N)
    log(f"\n=== {ds.code} {ds.title}: N={N}, n={n_eff}, seeds={seeds}, consent: {cand.label}")
    thr = [threshold_for_fraction(cand.values, cand.sign, f) for f in fracs]
    sweep = []
    t_start = time.perf_counter()
    for seed in range(seeds):
        idx = subsample(N, n_eff, 1000 + seed)
        X = ds.X[idx]
        v = cand.values[idx]
        for i, (f, (t, real)) in enumerate(zip(fracs, thr)):
            for mech in mechanisms:
                if mech == "column":
                    m = consent_mask(v, cand.sign, t)
                else:
                    rng = np.random.default_rng([seed, i, 7])
                    m = rng.random(len(X)) < real
                t0 = time.perf_counter()
                rows = one_setting(X, m, seed, (key, mech, i))
                for r in rows:
                    r.update({"dataset": key, "mechanism": mech, "seed": seed, "n": len(X),
                              "target_frac": f, "threshold": t, "frac_full": round(real, 5),
                              "frac": round(float(m.mean()), 5)})
                sweep.extend(rows)
                inc = next(r for r in rows if r["algo"] == "incremental")
                dec = next(r for r in rows if r["algo"] == "decremental")
                log(f"  seed {seed} {mech:6s} {100*m.mean():5.1f}% consent | inc {inc['calls']:5d} calls "
                    f"{inc['seconds']:7.3f}s | dec {dec['calls']:5d} calls {dec['seconds']:7.3f}s"
                    f" | ok={inc['match']}{dec['match']} cert={inc['gt_optimal']} "
                    f"({time.perf_counter()-t0:.1f}s)")
    sweep_secs = time.perf_counter() - t_start
    cols = ["dataset", "mechanism", "seed", "n", "target_frac", "threshold", "frac_full", "frac",
            "algo", "run_order", "calls", "seconds", "radius", "gt_radius", "center_err", "match",
            "gt_encloses", "gt_optimal", "n_consent", "outside_nonconsent", "gt_cx", "gt_cy"]
    sweep = [{c: r[c] for c in cols} for r in sweep]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    write_csv(sweep, os.path.join(RESULTS_DIR, f"{key}_sweep.csv"))

    # ---- scaling with n at a fixed threshold -------------------------------
    vs_rows = []
    t_start = time.perf_counter()
    if vs_n_sizes:
        t, real = threshold_for_fraction(cand.values, cand.sign, vs_n_frac)
        sizes = sorted({min(s, N) for s in vs_n_sizes})
        if N < 1000:  # tiny datasets (flights: one point per airport)
            sizes = sorted({max(20, int(N * q)) for q in (0.125, 0.25, 0.5, 0.75)} | {N})
        for size in sizes:
            for seed in range(seeds):
                idx = subsample(N, size, 5000 + 97 * seed + size)
                X = ds.X[idx]
                m = consent_mask(cand.values[idx], cand.sign, t)
                rows = one_setting(X, m, seed, (key, "vs_n", size))
                for r in rows:
                    r.update({"dataset": key, "mechanism": "column", "seed": seed, "n": size,
                              "target_frac": vs_n_frac, "threshold": t, "frac_full": round(real, 5),
                              "frac": round(float(m.mean()), 5)})
                vs_rows.extend(rows)
                inc = next(r for r in rows if r["algo"] == "incremental")
                dec = next(r for r in rows if r["algo"] == "decremental")
                log(f"  vs_n n={size:5d} seed {seed} | inc {inc['calls']:5d} {inc['seconds']:7.3f}s | "
                    f"dec {dec['calls']:5d} {dec['seconds']:7.3f}s | ok={inc['match']}{dec['match']}")
        vs_rows = [{c: r[c] for c in cols} for r in vs_rows]
        write_csv(vs_rows, os.path.join(RESULTS_DIR, f"{key}_vs_n.csv"))
    vs_secs = time.perf_counter() - t_start

    # ---- summary ------------------------------------------------------------
    allrows = sweep + vs_rows
    mism = [r for r in allrows if not r["match"]]
    agg = []
    for mech in mechanisms:
        for f in fracs:
            sel = [r for r in sweep if r["mechanism"] == mech and r["target_frac"] == f]
            row = {"mechanism": mech, "target_frac": f,
                   "threshold": sel[0]["threshold"],
                   "frac": round(float(np.mean([r["frac"] for r in sel])), 4)}
            for algo in ALGOS:
                s = [r for r in sel if r["algo"] == algo]
                row[f"{algo[:3]}_calls_mean"] = round(float(np.mean([r["calls"] for r in s])), 2)
                row[f"{algo[:3]}_calls_sd"] = round(float(np.std([r["calls"] for r in s])), 2)
                row[f"{algo[:3]}_sec_mean"] = round(float(np.mean([r["seconds"] for r in s])), 5)
                row[f"{algo[:3]}_sec_sd"] = round(float(np.std([r["seconds"] for r in s])), 5)
            agg.append(row)
    summary = {
        "dataset": key, "code": ds.code, "title": ds.title, "source": ds.source,
        "coordinates": ds.unit_note, "axis_labels": list(ds.axis_labels),
        "N_points": N, "n_raw_rows": ds.n_raw, "dropped_missing_or_filtered": ds.n_dropped_missing,
        "dropped_duplicate_points": ds.n_dropped_duplicate,
        "consent_column": cand.key, "consent_label": cand.label, "consent_sign": cand.sign,
        "consent_story": cand.story, "n_per_run": n_eff, "seeds": seeds,
        "runs": len(allrows), "mismatches": len(mism),
        "mismatch_rows": mism[:20],
        "gt_certificate_failures": sum(1 for r in allrows if not (r["gt_encloses"] and r["gt_optimal"])),
        "sweep_wall_seconds": round(sweep_secs, 1), "vs_n_wall_seconds": round(vs_secs, 1),
        "aggregate": agg,
    }
    with open(os.path.join(RESULTS_DIR, f"{key}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=float)
    log(f"  -> {len(allrows)} runs, {len(mism)} mismatches, sweep {sweep_secs:.0f}s, vs_n {vs_secs:.0f}s")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=5000, help="points per run (subsample)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--fracs", type=str, default=",".join(map(str, FRACS)))
    ap.add_argument("--mechanisms", type=str, default="column,random")
    ap.add_argument("--vs-n", type=str, default=",".join(map(str, VS_N_SIZES)),
                    help="sizes for the scaling experiment ('' to skip)")
    ap.add_argument("--vs-n-frac", type=float, default=0.5)
    args = ap.parse_args(argv)
    fracs = tuple(float(x) for x in args.fracs.split(",") if x)
    sizes = tuple(int(x) for x in args.vs_n.split(",") if x)
    for key in args.datasets or list(datasets.LOADERS):
        run_dataset(key, args.n, args.seeds, fracs, tuple(args.mechanisms.split(",")),
                    sizes, args.vs_n_frac)


if __name__ == "__main__":
    main()
