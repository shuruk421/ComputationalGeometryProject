"""
Benchmarks for the Amsterdam consent study: incremental vs decremental
consent-aware minimum enclosing ball (db_consents), plus the geometric
consent-column analysis.

Every algorithm run is checked against the ground-truth MEB of the consenting
points only. The ground truth itself is certified independently of Welzl: all
consenting points must lie inside it and its centre must lie in the convex
hull of its boundary points (no open half-circle contains all of them), which
is the optimality condition for a minimum enclosing disc.
"""

import gc
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import db_consents  # noqa: E402

db_consents.USE_CPP_BACKEND = False  # the C++ module is a Linux .so
from db_consents import (  # noqa: E402
    Oracle,
    decremental_distance_based,
    incremental_distance_based,
    welzl,
)

from consent_data import (  # noqa: E402
    BINARY_CANDIDATES,
    Listings,
    thresholds_for_fractions,
)

ALGORITHMS = {
    "incremental": incremental_distance_based,
    "decremental": decremental_distance_based,
}
ALGO_ORDER = ("incremental", "decremental")

RADIUS_REL_TOL = 1e-9
CENTER_TOL_KM = 1e-6  # 1 mm


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
def exact_meb(xy: np.ndarray, seed: int = 0) -> Tuple[np.ndarray, float]:
    """MEB of ``xy`` (all treated as consenting) using db_consents.welzl with a
    throwaway Oracle. Points are shuffled first (Welzl needs random order for
    its expected linear running time)."""
    pts = [([float(a), float(b)], True) for a, b in xy]
    random.Random(seed).shuffle(pts)
    c, r2 = welzl(pts, [], Oracle(), 2, len(pts))
    return np.asarray(c, dtype=float), math.sqrt(max(r2, 0.0))


def boundary_mask(xy: np.ndarray, center: np.ndarray, radius: float, rel: float = 1e-7) -> np.ndarray:
    d = np.linalg.norm(xy - center, axis=1)
    return np.abs(d - radius) <= rel * max(radius, 1e-3)


def certify_meb(xy: np.ndarray, center: np.ndarray, radius: float) -> Tuple[bool, str]:
    """Check the MEB optimality conditions without using Welzl."""
    d = np.linalg.norm(xy - center, axis=1)
    if d.max() > radius * (1 + 1e-9) + 1e-12:
        return False, f"point outside ball by {d.max() - radius:.3e} km"
    if radius == 0.0:
        return True, "single location"
    on = boundary_mask(xy, center, radius)
    v = xy[on] - center
    ang = np.sort(np.arctan2(v[:, 1], v[:, 0]))
    gaps = np.diff(np.r_[ang, ang[0] + 2 * math.pi])
    if gaps.max() > math.pi + 1e-6:
        return False, f"centre not in hull of {int(on.sum())} boundary points (max gap {gaps.max():.4f} rad)"
    return True, f"{int(on.sum())} boundary points"


# ---------------------------------------------------------------------------
# One configuration = one point set + consent vector, several seeds
# ---------------------------------------------------------------------------
def run_config(xy: np.ndarray, consent: np.ndarray, seeds: Sequence[int], meta: Dict) -> Tuple[List[Dict], Dict]:
    consent = np.asarray(consent, dtype=bool)
    points = [([float(a), float(b)], bool(c)) for (a, b), c in zip(xy, consent)]
    cons_xy = xy[consent]
    gt_c, gt_r = exact_meb(cons_xy)
    cert_ok, cert_msg = certify_meb(cons_xy, gt_c, gt_r)
    d_all = np.linalg.norm(xy - gt_c, axis=1)
    noncons_outside = int(((d_all > gt_r * (1 + 1e-9)) & ~consent).sum())
    info = dict(
        meta,
        n=int(len(xy)),
        n_consenting=int(consent.sum()),
        consent_frac=float(consent.mean()),
        gt_radius_km=gt_r,
        gt_center_x_km=float(gt_c[0]),
        gt_center_y_km=float(gt_c[1]),
        gt_certified=bool(cert_ok),
        gt_certificate=cert_msg,
        gt_support_points=int(boundary_mask(cons_xy, gt_c, gt_r).sum()),
        noncons_outside_gt=noncons_outside,
    )
    rows = []
    for s in seeds:
        # alternate which algorithm goes first so background load hits both equally
        order = ALGO_ORDER if s % 2 == 0 else ALGO_ORDER[::-1]
        for pos, algo in enumerate(order):
            random.seed(s)
            oracle = Oracle()
            gc.collect()
            w0 = time.perf_counter()
            c0 = time.process_time()
            center, radius = ALGORITHMS[algo](points, oracle)
            c1 = time.process_time()
            w1 = time.perf_counter()
            center = np.asarray(center, dtype=float)
            r_err = abs(radius - gt_r) / max(gt_r, 1e-12)
            c_err = float(np.linalg.norm(center - gt_c))
            rows.append(dict(
                info,
                seed=int(s),
                algorithm=algo,
                run_position=pos,
                oracle_calls=int(oracle.get_call_count()),
                wall_s=w1 - w0,
                cpu_s=c1 - c0,
                radius_km=float(radius),
                center_x_km=float(center[0]),
                center_y_km=float(center[1]),
                radius_rel_err=r_err,
                center_err_km=c_err,
                correct=bool(r_err <= RADIUS_REL_TOL and c_err <= CENTER_TOL_KM),
            ))
    return rows, info


def _summarise(rows: List[Dict]) -> str:
    parts = []
    for algo in ALGO_ORDER:
        rr = [r for r in rows if r["algorithm"] == algo]
        q = np.mean([r["oracle_calls"] for r in rr])
        t = np.mean([r["wall_s"] for r in rr])
        bad = sum(not r["correct"] for r in rr)
        parts.append(f"{algo[:3]} q={q:7.1f} t={t:7.3f}s" + (f" MISMATCH x{bad}" if bad else ""))
    return " | ".join(parts)


def threshold_sweep(listings: Listings, column: str, targets: Sequence[float], seeds: Sequence[int]) -> List[Dict]:
    col = listings.columns[column]
    rows: List[Dict] = []
    for t in thresholds_for_fractions(col.score, targets):
        consent = col.consent(t)
        meta = dict(experiment="threshold_sweep", column=column, threshold=float(t),
                    threshold_label=col.describe(t), subsample_seed=-1)
        r, info = run_config(listings.xy, consent, seeds, meta)
        rows += r
        log(f"[{column}] {col.describe(t):>26}  consent={info['consent_frac']*100:5.1f}%  "
            f"r={info['gt_radius_km']:.3f}km  {_summarise(r)}")
    return rows


def size_sweep(listings: Listings, column: str, target_fracs: Sequence[float], sizes: Sequence[int],
               seeds: Sequence[int]) -> List[Dict]:
    col = listings.columns[column]
    rows: List[Dict] = []
    for t in thresholds_for_fractions(col.score, target_fracs):
        consent_full = col.consent(t)
        for n in sizes:
            n = min(int(n), listings.n)
            chunk = []
            for s in seeds:
                if n < listings.n:
                    idx = np.sort(np.random.default_rng(100_003 * n + s).choice(listings.n, n, replace=False))
                else:
                    idx = np.arange(listings.n)
                meta = dict(experiment="size_sweep", column=column, threshold=float(t),
                            threshold_label=col.describe(t), subsample_seed=int(s))
                r, _ = run_config(listings.xy[idx], consent_full[idx], [s], meta)
                chunk += r
            rows += chunk
            log(f"[size {column}] {col.describe(t):>22}  n={n:6d}  {_summarise(chunk)}")
    return rows


# ---------------------------------------------------------------------------
# Consent-column analysis (geometry only, no oracle)
# ---------------------------------------------------------------------------
def _disc_metrics(c, r, c_all, r_all):
    shift = float(np.linalg.norm(c - c_all))
    return dict(radius_km=r, radius_ratio=r / r_all, center_shift_km=shift,
                # Hausdorff distance between two discs
                hausdorff_km=shift + abs(r_all - r))


def analyse_columns(listings: Listings, targets: Sequence[float], draws: int = 25,
                    seed: int = 12345) -> Tuple[List[Dict], Dict]:
    xy = listings.xy
    c_all, r_all = exact_meb(xy)
    rng = np.random.default_rng(seed)
    baseline_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def random_baseline(k: int):
        if k not in baseline_cache:
            hs, rs = [], []
            for _ in range(draws):
                idx = rng.choice(listings.n, k, replace=False)
                c, r = exact_meb(xy[idx])
                m = _disc_metrics(c, r, c_all, r_all)
                hs.append(m["hausdorff_km"])
                rs.append(r)
            baseline_cache[k] = (np.array(hs), np.array(rs))
        return baseline_cache[k]

    def measure(name, kind, rule, threshold, consent):
        k = int(consent.sum())
        c, r = exact_meb(xy[consent])
        m = _disc_metrics(c, r, c_all, r_all)
        d = np.linalg.norm(xy - c, axis=1)
        hs, rs = random_baseline(k)
        return dict(
            column=name, kind=kind, rule=rule, threshold=threshold,
            n_consenting=k, consent_frac=k / listings.n, **m,
            noncons_outside=int(((d > r * (1 + 1e-9)) & ~consent).sum()),
            random_hausdorff_p10=float(np.percentile(hs, 10)),
            random_hausdorff_p50=float(np.percentile(hs, 50)),
            random_hausdorff_p90=float(np.percentile(hs, 90)),
            random_radius_ratio_p50=float(np.percentile(rs, 50) / r_all),
            # share of equally sized random subsets whose disc is smaller:
            # ~0 => consenters reach the city edge more than random (MEB sticky)
            # ~1 => consenters are more central than random (MEB shrinks)
            radius_pct_vs_random=float(np.mean(rs < r)),
        )

    out: List[Dict] = []
    for name, col in listings.columns.items():
        for t in thresholds_for_fractions(col.score, targets):
            consent = col.consent(t)
            if consent.all():
                continue
            out.append(measure(name, "threshold", col.describe(t), float(t), consent))
        log(f"column analysis: {name} done ({len(baseline_cache)} random sizes cached)")
    for name, rule in BINARY_CANDIDATES.items():
        consent = listings.raw[name] > 0.5
        out.append(measure(name, "binary", rule, 1.0, consent))
    return out, dict(center_x_km=float(c_all[0]), center_y_km=float(c_all[1]), radius_km=r_all)
