"""
Consent-column analysis: for every candidate consent attribute of a dataset,
how far does thresholding it move the consenting MEB away from the
all-points MEB, compared with consent drawn at random at the same rate?

Writes results/<dataset>_consent_candidates.csv and prints a table.

Metrics (computed on a fixed random subsample of <= 20,000 points):
  frac        realised consenting fraction (whole dataset) at the threshold
  r_ratio     radius(consenting MEB) / radius(all-points MEB)
  shift       |centre(consenting MEB) - centre(all MEB)| / radius(all MEB)
  rand_*      the same two numbers for random consent at that fraction (mean of 3 draws)
  rho_radial  Spearman correlation between (sign * value) and distance from the
              all-points MEB centre; < 0 means consenters sit in the middle.
"""

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import datasets  # noqa: E402
from meb_common import (RESULTS_DIR, consent_mask, ground_truth_meb, subsample,  # noqa: E402
                        threshold_for_fraction)

FRACS = (0.10, 0.25, 0.50, 0.75, 0.90)
MAX_N = 20000


def spearman(a, b):
    from scipy.stats import spearmanr
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    return float(spearmanr(a[ok], b[ok]).correlation)


def analyse(ds, max_n=MAX_N, seed=0):
    idx = subsample(len(ds.X), max_n, seed)
    X = ds.X[idx]
    c_all, r_all = ground_truth_meb(X)
    radial = np.linalg.norm(X - c_all[None, :], axis=1)
    rng = np.random.default_rng(seed)
    rows = []
    for cand in ds.candidates.values():
        v_full = cand.values
        v = v_full[idx]
        rho = spearman(cand.sign * v, radial)
        fracs = FRACS
        if cand.binary:
            fracs = (float(np.nanmean(v_full)),)
        for f in fracs:
            if cand.binary:
                t, real = 1.0, float(np.mean(v_full >= 1))
            else:
                t, real = threshold_for_fraction(v_full, cand.sign, f)
            m = consent_mask(v, cand.sign, t)
            c, r = ground_truth_meb(X[m])
            rr, sh = [], []
            for _ in range(3):
                mr = rng.random(len(X)) < m.mean()
                cr, rrad = ground_truth_meb(X[mr])
                rr.append(rrad / r_all)
                sh.append(np.linalg.norm(cr - c_all) / r_all)
            rows.append({
                "dataset": ds.key, "candidate": cand.key, "label": cand.label,
                "rule": cand.rule(t), "chosen": int(cand.key == ds.chosen),
                "binary": int(cand.binary), "target_frac": round(f, 4),
                "threshold": t, "frac": round(real, 4),
                "r_ratio": round(r / r_all, 4) if c is not None else 0.0,
                "shift": round(float(np.linalg.norm(c - c_all) / r_all), 4) if c is not None else float("nan"),
                "rand_r_ratio": round(float(np.mean(rr)), 4),
                "rand_shift": round(float(np.mean(sh)), 4),
                "rho_radial": round(rho, 3),
                "n_unique_values": int(len(np.unique(v_full[np.isfinite(v_full)]))),
                "missing_frac": round(float(np.mean(~np.isfinite(v_full))), 4),
                "story": cand.story,
            })
    return rows


def write(rows, key):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{key}_consent_candidates.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def print_table(rows):
    print(f"{'candidate':13s} {'rule':42s} {'frac':>5s} {'r/r0':>6s} {'shift':>6s} "
          f"{'rnd r/r0':>8s} {'rnd sh':>6s} {'rho':>6s}")
    for r in rows:
        print(f"{r['candidate']:13s} {r['rule'][:42]:42s} {r['frac']:5.2f} {r['r_ratio']:6.3f} "
              f"{r['shift']:6.3f} {r['rand_r_ratio']:8.3f} {r['rand_shift']:6.3f} {r['rho_radial']:6.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", default=list(datasets.LOADERS))
    args = ap.parse_args(argv)
    for key in args.datasets or list(datasets.LOADERS):
        ds = datasets.load(key)
        print(f"\n=== {ds.code} {ds.title}: N={len(ds.X)} (raw {ds.n_raw}, "
              f"-{ds.n_dropped_missing} missing/filtered, -{ds.n_dropped_duplicate} duplicate points)")
        rows = analyse(ds)
        print_table(rows)
        print("->", write(rows, key))


if __name__ == "__main__":
    main()
