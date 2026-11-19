"""
Amsterdam Airbnb: minimum number of reviews as consent.

Single entry point that regenerates every result and figure of the study:

  1. consent-column analysis (which column makes a good consent signal?)
  2. threshold sweep: incremental vs decremental consent-aware MEB, consent =
     number_of_reviews >= t (t at quantiles), plus the same sweep for the
     best alternative column
  3. size sweep: subsample n at two fixed review thresholds
  4. correctness check of every run against the certified MEB of the
     consenting listings only
  5. figures (maps on CartoDB Positron tiles + performance charts)

Geometry is 2D: (lat, lng) projected to local equirectangular km, so the
enclosing ball is a real geographic disc (see consent_data.py).

Usage (from the repository root):
  python airbnb_benchmark/visualize_review_consent.py \
      --csv airbnb_benchmark/listings_amsterdam.csv --out airbnb_benchmark/figures

  --plots-only   redraw the figures from airbnb_benchmark/results/*.csv
  --quick        smoke test (few thresholds, 2 seeds)
"""

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from typing import Dict, List

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import consent_bench as bench  # noqa: E402  (also puts the repo root on sys.path)
import consent_plots as plots  # noqa: E402
from consent_data import achievable_range, load_listings, thresholds_for_fractions  # noqa: E402

PRIMARY = "number_of_reviews"

SWEEP_TARGETS = {
    PRIMARY: [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.07, 0.05, 0.035, 0.02, 0.01],
    "alt": [0.60, 0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.07, 0.05, 0.035, 0.02],
}
MAP_TARGETS = {
    PRIMARY: [0.90, 0.50, 0.25, 0.10, 0.05, 0.02],
    "alt": [0.60, 0.40, 0.25, 0.10, 0.05, 0.02],
}
COLUMN_TARGETS = [0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.07, 0.05,
                  0.035, 0.02, 0.015, 0.01]
SIZE_TARGETS = [0.50, 0.05]
SIZES = [1000, 2000, 4000, 7000, 10**9]  # last = all listings

RUN_FIELDS = ["experiment", "column", "threshold", "threshold_label", "n", "n_consenting", "consent_frac",
              "subsample_seed", "seed", "algorithm", "run_position", "oracle_calls", "wall_s", "cpu_s",
              "radius_km", "gt_radius_km", "radius_rel_err", "center_err_km", "correct",
              "gt_support_points", "noncons_outside_gt", "gt_certified"]
COLUMN_FIELDS = ["column", "kind", "rule", "threshold", "n_consenting", "consent_frac", "radius_km",
                 "radius_ratio", "center_shift_km", "hausdorff_km", "noncons_outside", "random_hausdorff_p10",
                 "random_hausdorff_p50", "random_hausdorff_p90", "random_radius_ratio_p50",
                 "radius_pct_vs_random"]


def _fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.10g}"
    return v


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in fields})


def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k in ("experiment", "column", "threshold_label", "algorithm", "kind", "rule"):
                continue
            if v in ("True", "False"):
                r[k] = v == "True"
            else:
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    pass
    return rows


def excess_displacement(column_rows: List[Dict], col: str, lo: float = 0.05, hi: float = 0.50) -> float:
    """Mean (disc displacement - median displacement of an equally sized random
    subset) over thresholds giving lo..hi consenting. > 0: the column moves the
    disc more than chance; < 0: the disc is stickier than chance."""
    rr = [r for r in column_rows if r["column"] == col and r["kind"] == "threshold"
          and lo <= r["consent_frac"] <= hi]
    return float(np.mean([r["hausdorff_km"] - r["random_hausdorff_p50"] for r in rr])) if rr else float("nan")


def pick_alternative(column_rows: List[Dict], listings) -> str:
    """Best alternative = column whose disc moves most beyond chance over 5-50%
    consenting, among columns with a plausible story whose thresholds reach at
    least 50% consenting."""
    best, best_score = None, -np.inf
    for name, col in listings.columns.items():
        if name == PRIMARY or col.weak_story or achievable_range(col.score)[1] < 0.5:
            continue
        score = excess_displacement(column_rows, name)
        if score > best_score:
            best, best_score = name, score
    return best


def git_commit():
    try:
        return subprocess.check_output(["git", "describe", "--always", "--dirty"], cwd=_HERE,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=os.path.join(_HERE, "listings_amsterdam.csv"))
    ap.add_argument("--out", default=os.path.join(_HERE, "figures"), help="directory for PNG figures")
    ap.add_argument("--results", default=os.path.join(_HERE, "results"), help="directory for CSV/JSON results")
    ap.add_argument("--tile-cache", default=os.path.join(_HERE, "tile_cache"))
    ap.add_argument("--zoom", type=int, default=13, help="basemap tile zoom (default 13)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--alt-column", default="auto",
                    help="second consent column to benchmark and map ('auto' = best by column analysis)")
    ap.add_argument("--dup-jitter-m", type=float, default=1.0,
                    help="move listings with identical coordinates by this many metres (0 = off)")
    ap.add_argument("--column-draws", type=int, default=25, help="random subsets per size in column analysis")
    ap.add_argument("--plots-only", action="store_true", help="skip benchmarks, redraw from results/*.csv")
    ap.add_argument("--skip-size-sweep", action="store_true")
    ap.add_argument("--quick", action="store_true", help="smoke test: few thresholds, 2 seeds")
    args = ap.parse_args()
    try:  # rules contain >= / <= signs; Windows consoles default to cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.results, exist_ok=True)
    runs_path = os.path.join(args.results, "review_consent_runs.csv")
    cols_path = os.path.join(args.results, "consent_column_analysis.csv")
    summary_path = os.path.join(args.results, "review_consent_summary.json")

    t_start = time.time()
    L = load_listings(args.csv, dup_jitter_m=args.dup_jitter_m)
    bench.log(f"loaded {L.n:,} listings; centroid {L.lat0:.5f}N {L.lng0:.5f}E; "
              f"{L.n_duplicate_rows} rows share exact coordinates with an earlier row "
              f"({L.n_duplicate_groups} groups) -> jitter {args.dup_jitter_m} m")

    seeds = list(range(2 if args.quick else args.seeds))
    sweep_targets = {k: (v[::4] if args.quick else v) for k, v in SWEEP_TARGETS.items()}

    # ---------------- 1. consent column analysis ----------------------------
    if args.plots_only:
        column_rows = read_csv(cols_path)
        all_disc = json.load(open(summary_path, encoding="utf-8"))["all_listings_disc"]
    else:
        bench.log("consent-column analysis ...")
        column_rows, all_disc = bench.analyse_columns(
            L, COLUMN_TARGETS[::3] if args.quick else COLUMN_TARGETS, draws=5 if args.quick else args.column_draws)
        write_csv(cols_path, column_rows, COLUMN_FIELDS)
    achievable = {name: achievable_range(c.score) for name, c in L.columns.items()}
    for name in L.columns:
        bench.log(f"  {name:<32} thresholds reach {achievable[name][0] * 100:5.2f}-{achievable[name][1] * 100:5.1f}%"
                  f" consenting; excess disc displacement vs random (5-50%): "
                  f"{excess_displacement(column_rows, name):+.2f} km")
    alt = pick_alternative(column_rows, L) if args.alt_column == "auto" else args.alt_column
    bench.log(f"all-listings disc r = {all_disc['radius_km']:.3f} km; alternative consent column: {alt}")

    # ---------------- 2./3. benchmarks ---------------------------------------
    if args.plots_only:
        runs = read_csv(runs_path)
    else:
        runs = []
        bench.log(f"threshold sweep, {len(seeds)} seeds per point ...")
        runs += bench.threshold_sweep(L, PRIMARY, sweep_targets[PRIMARY], seeds)
        runs += bench.threshold_sweep(L, alt, sweep_targets["alt"], seeds)
        if not args.skip_size_sweep:
            bench.log("size sweep ...")
            runs += bench.size_sweep(L, PRIMARY, SIZE_TARGETS, SIZES[::2] if args.quick else SIZES, seeds)
        write_csv(runs_path, runs, RUN_FIELDS)

    # ---------------- 4. correctness -----------------------------------------
    bad = [r for r in runs if not r["correct"]]
    uncertified = {(r["column"], r["threshold"], r["n"]) for r in runs if not r["gt_certified"]}
    bench.log(f"correctness: {len(runs) - len(bad)}/{len(runs)} runs match the certified ground truth "
              f"(radius rel. tol {bench.RADIUS_REL_TOL:g}, centre tol {bench.CENTER_TOL_KM * 1e6:g} mm); "
              f"{len(uncertified)} ground truths failed certification")
    for r in bad[:20]:
        bench.log(f"  MISMATCH {r['algorithm']} {r['threshold_label']} n={r['n']} seed={r['seed']}: "
                  f"r={r['radius_km']} vs {r['gt_radius_km']} (rel {r['radius_rel_err']:.2e}), "
                  f"centre off {r['center_err_km']:.2e} km")

    # ---------------- summary JSON ---------------------------------------------
    agg_t = plots.aggregate(runs, "threshold_sweep")
    agg_s = plots.aggregate(runs, "size_sweep")
    if not args.plots_only:
        def table(agg):
            out = []
            for (col, t, n) in sorted({k[:3] for k in agg}, key=lambda k: (k[0], k[2], -agg[k + ("incremental",)]
                                                                            ["consent_frac"])):
                i, d = agg[(col, t, n, "incremental")], agg[(col, t, n, "decremental")]
                out.append(dict(
                    column=col, threshold=t, rule=i["threshold_label"], n=n, seeds=i["seeds"],
                    consent_frac=round(i["consent_frac"], 5), gt_radius_km=round(i["gt_radius_km"], 4),
                    noncons_outside_disc=i["noncons_outside"],
                    incremental=dict(calls_mean=i["calls_mean"], calls_std=i["calls_std"], calls_min=i["calls_min"],
                                     calls_max=i["calls_max"], wall_s_mean=i["wall_mean"], wall_s_std=i["wall_std"],
                                     cpu_s_mean=i["cpu_mean"]),
                    decremental=dict(calls_mean=d["calls_mean"], calls_std=d["calls_std"], calls_min=d["calls_min"],
                                     calls_max=d["calls_max"], wall_s_mean=d["wall_mean"], wall_s_std=d["wall_std"],
                                     cpu_s_mean=d["cpu_mean"]),
                    incorrect_runs=i["n_incorrect"] + d["n_incorrect"],
                ))
            return out

        summary = dict(
            generated=time.strftime("%Y-%m-%d %H:%M:%S"),
            git_commit=git_commit(),
            python=platform.python_version(),
            numpy=np.__version__,
            data=dict(csv=os.path.basename(args.csv), listings=L.n, snapshot_last_review=str(L.snapshot_date),
                      centroid_lat=L.lat0, centroid_lng=L.lng0, duplicate_coordinate_rows=L.n_duplicate_rows,
                      duplicate_coordinate_groups=L.n_duplicate_groups, dup_jitter_m=args.dup_jitter_m),
            geometry="2D local equirectangular km: x = R*dlng*cos(lat0), y = R*dlat, R = 6371.0088 km",
            seeds=seeds,
            primary_column=PRIMARY,
            alternative_column=alt,
            all_listings_disc=all_disc,
            correctness=dict(runs=len(runs), mismatches=len(bad), uncertified_ground_truths=len(uncertified),
                             radius_rel_tol=bench.RADIUS_REL_TOL, center_tol_km=bench.CENTER_TOL_KM),
            consent_columns={name: dict(story=c.story, achievable_frac=achievable[name],
                                        excess_displacement_km_5_50pct=excess_displacement(column_rows, name))
                             for name, c in L.columns.items()},
            threshold_sweep=table(agg_t),
            size_sweep=table(agg_s),
            runtime_s=round(time.time() - t_start, 1),
        )
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1)

    # ---------------- 5. figures ------------------------------------------------
    labels = {name: c.label for name, c in L.columns.items()}
    p_cols = os.path.join(args.out, "consent_column_analysis.png")
    plots.plot_column_analysis(column_rows, labels, PRIMARY, alt, achievable, all_disc["radius_km"], p_cols,
                               draws=5 if args.quick else args.column_draws)
    bench.log(f"wrote {p_cols}")

    def map_panels(column, targets):
        col = L.columns[column]
        panels = []
        for t in thresholds_for_fractions(col.score, targets):
            consent = col.consent(t)
            c, r = bench.exact_meb(L.xy[consent])
            calls = {a: agg_t[(column, float(t), L.n, a)]["calls_mean"] for a in bench.ALGO_ORDER
                     if (column, float(t), L.n, a) in agg_t}
            if len(calls) < 2:
                continue
            panels.append(dict(threshold=t, label=col.describe(t), consent=consent, center=c, radius=r,
                               support=bench.boundary_mask(L.xy[consent], c, r), calls=calls,
                               seeds=agg_t[(column, float(t), L.n, "incremental")]["seeds"],
                               frac=float(consent.mean()), n_consenting=int(consent.sum())))
        return panels

    foot = (f"Basemap © OpenStreetMap contributors © CARTO.  Local equirectangular projection in km "
            f"centred on {L.lat0:.4f}°N {L.lng0:.4f}°E (distances within ~0.2%); the drawn circle is the "
            f"exact 2D minimum enclosing disc.  {L.n_duplicate_rows} listings sharing exact coordinates with "
            f"another were moved {args.dup_jitter_m:g} m so the coordinate-keyed oracle cache treats them as "
            f"distinct.")
    for column, key, fname in ((PRIMARY, PRIMARY, "amsterdam_review_consent_maps.png"),
                               (alt, "alt", f"amsterdam_{alt}_consent_maps.png")):
        panels = map_panels(column, MAP_TARGETS[key])
        if not panels:
            continue
        path = os.path.join(args.out, fname)
        plots.plot_maps(
            L, panels, all_disc, path, args.tile_cache, zoom=args.zoom,
            title=f"Amsterdam Airbnb: enclosing disc of consenting listings, consent = {L.columns[column].label} "
                  f"above a threshold",
            subtitle=("Thresholds sit at quantiles of the column so the consenting share runs from "
                      f"{panels[0]['frac'] * 100:.0f}% down to {panels[-1]['frac'] * 100:.0f}%. "
                      "Both algorithms find the same disc; they differ in how many hosts they must ask."),
            footnote=foot)
        bench.log(f"wrote {path}")

    titles = {name: f"Consent = {c.label} ≥ threshold" for name, c in L.columns.items()}
    titles["last_review_recency"] = "Consent = reviewed within the last D days"
    tick_thr = {}
    for column, key in ((PRIMARY, PRIMARY), (alt, "alt")):
        # label the map thresholds plus the lowest-consent point of the sweep
        tick_thr[column] = [float(t) for t in thresholds_for_fractions(
            L.columns[column].score, MAP_TARGETS[key] + [min(SWEEP_TARGETS[key])])]
    p_perf = os.path.join(args.out, "perf_threshold_sweep.png")
    plots.plot_threshold_perf(agg_t, [c for c in (PRIMARY, alt) if any(k[0] == c for k in agg_t)], titles, L.n,
                              p_perf, tick_thr, seeds=len(seeds),
                              correctness_note=("Every run returned the certified ground-truth disc." if not bad else
                                                f"{len(bad)} of {len(runs)} runs did NOT match the ground truth."))
    bench.log(f"wrote {p_perf}")
    if agg_s:
        p_size = os.path.join(args.out, "perf_size_sweep.png")
        row_titles = {t: f"Consent = {L.columns[PRIMARY].describe(t)}" for t in {k[1] for k in agg_s}}
        plots.plot_size_perf(agg_s, PRIMARY, row_titles, p_size, seeds=len(seeds))
        bench.log(f"wrote {p_size}")

    # ---------------- console table ------------------------------------------
    print()
    print(f"{'column':<24}{'rule':<28}{'consent':>8}{'r km':>7}{'calls inc':>11}{'calls dec':>11}"
          f"{'time inc s':>12}{'time dec s':>12}")
    for (col, t, n) in sorted({k[:3] for k in agg_t}, key=lambda k: (k[0] != PRIMARY, k[0], -k[1] if k[0] ==
                                                                      "last_review_recency" else k[1])):
        i, d = agg_t[(col, t, n, "incremental")], agg_t[(col, t, n, "decremental")]
        print(f"{col:<24}{i['threshold_label']:<28}{i['consent_frac'] * 100:7.1f}%{i['gt_radius_km']:7.2f}"
              f"{i['calls_mean']:11.1f}{d['calls_mean']:11.1f}{i['wall_mean']:12.3f}{d['wall_mean']:12.3f}")
    bench.log(f"done in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
