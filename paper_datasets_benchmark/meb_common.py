"""
Shared helpers: consent thresholds, running the two db_consents algorithms,
ground-truth MEBs with an independent optimality certificate, and map tiles.

Only reads db_consents; never modifies it.
"""

import math
import os
import random
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import db_consents  # noqa: E402
from db_consents import Oracle, decremental_distance_based, incremental_distance_based  # noqa: E402

db_consents.USE_CPP_BACKEND = False  # the C++ module is a Linux .so

RESULTS_DIR = os.path.join(HERE, "results")
FIGURES_DIR = os.path.join(HERE, "figures")
TILE_DIR = os.path.join(HERE, "tile_cache")

ALGOS = ("incremental", "decremental")
RADIUS_RTOL = 1e-7        # |r - r_gt| <= RADIUS_RTOL * r_gt counts as a match
SUPPORT_RTOL = 1e-7       # |d - r| <= SUPPORT_RTOL * r -> point is on the sphere


# ----------------------------------------------------------------------------
# consent
# ----------------------------------------------------------------------------

def consent_mask(values, sign, t):
    """consent <=> sign * value >= sign * t ; NaN never consents."""
    v = np.asarray(values, dtype=float)
    with np.errstate(invalid="ignore"):
        m = (sign * v) >= (sign * t)
    return m & np.isfinite(v)


def threshold_for_fraction(values, sign, frac):
    """Threshold t (a data value) whose consenting fraction over *all* rows is
    closest to ``frac``. With ties the realised fraction may differ."""
    v = np.asarray(values, dtype=float)
    s = np.sort(sign * v[np.isfinite(v)])           # ascending in sign*value
    n_all = len(v)
    u = np.unique(s)
    # number of rows with sign*v >= u_i
    cnt = len(s) - np.searchsorted(s, u, side="left")
    i = int(np.argmin(np.abs(cnt / n_all - frac)))
    return float(sign * u[i]), float(cnt[i] / n_all)


# ----------------------------------------------------------------------------
# algorithms + ground truth
# ----------------------------------------------------------------------------

def to_points(X, consent):
    """db_consents format: [(coords_list, consent_bool), ...] with Python floats."""
    return [(row, bool(k)) for row, k in zip(np.asarray(X, dtype=float).tolist(), consent)]


def run_algo(name, X, consent, seed):
    """Run one db_consents algorithm. Returns dict(calls, seconds, center, radius)."""
    pts = to_points(X, consent)
    random.seed(seed)
    oracle = Oracle()
    fn = incremental_distance_based if name == "incremental" else decremental_distance_based
    t0 = time.perf_counter()
    center, radius = fn(pts, oracle)
    dt = time.perf_counter() - t0
    if center is None or (not isinstance(center, (list, tuple, np.ndarray))):
        center = None
    return {"calls": oracle.get_call_count(), "seconds": dt,
            "center": None if center is None else [float(c) for c in center],
            "radius": float(radius)}


def ground_truth_meb(Xc, seed=12345):
    """MEB of the consenting points only (everyone 'consents'; throwaway Oracle)."""
    if len(Xc) == 0:
        return None, 0.0
    pts = [([float(c) for c in row], True) for row in Xc]
    random.seed(seed)
    c, r = incremental_distance_based(pts, Oracle())
    return np.array(c, dtype=float), float(r)


def support_indices(X, center, r, rtol=SUPPORT_RTOL):
    d = np.linalg.norm(X - np.asarray(center)[None, :], axis=1)
    return np.flatnonzero(np.abs(d - r) <= rtol * max(r, 1e-300))


def certify_meb(Xc, center, r, rtol=1e-7):
    """Independent optimality certificate for an enclosing ball:
    (1) every point lies inside (1 + rtol) * r, and (2) the centre lies in the
    convex hull of the points on the sphere (KKT condition of the MEB).
    Returns (encloses, optimal)."""
    if len(Xc) == 0 or center is None:
        return True, True
    center = np.asarray(center, dtype=float)
    d = np.linalg.norm(Xc - center[None, :], axis=1)
    encloses = bool(np.all(d <= r * (1 + rtol) + 1e-12))
    if len(Xc) == 1:
        return encloses, bool(r <= 1e-12)
    S = Xc[np.abs(d - r) <= 1e-6 * max(r, 1e-300)]
    if len(S) == 0:
        return encloses, False
    # centre in conv(S)? solve min ||S^T w - c||, w >= 0, sum w = 1 via NNLS on
    # an augmented system (scipy.optimize.nnls).
    from scipy.optimize import nnls
    big = 1e3
    A = np.vstack([S.T, big * np.ones((1, len(S)))])
    b = np.concatenate([center, [big]])
    w, res = nnls(A, b)
    optimal = bool(res <= 1e-6 * max(r, 1.0))
    return encloses, optimal


def subsample(N, n, seed):
    if n >= N:
        return np.arange(N)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(N, size=n, replace=False))


# ----------------------------------------------------------------------------
# map tiles, adapted from airbnb_benchmark/plot_amsterdam_meb.py
# ----------------------------------------------------------------------------

TILE_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
            "World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}")
TILE_ATTRIBUTION = "Basemap: Esri World Light Gray Canvas (Esri, HERE, Garmin, OpenStreetMap contributors)"

def lat_lng_to_tile(lat, lng, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lng + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return xtile, ytile


def _tile_y_to_lat(y, zoom):
    n = 2.0 ** zoom
    return math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))


def fetch_and_stitch_tiles(lat_min, lat_max, lng_min, lng_max, zoom):
    """Fetch Web-Mercator tiles covering the box and stitch them (cached)."""
    from PIL import Image
    x0, y0 = lat_lng_to_tile(lat_max, lng_min, zoom)
    x1, y1 = lat_lng_to_tile(lat_min, lng_max, zoom)
    os.makedirs(TILE_DIR, exist_ok=True)
    ts = 256
    img = Image.new("RGB", ((x1 - x0 + 1) * ts, (y1 - y0 + 1) * ts), (242, 242, 240))
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            fn = os.path.join(TILE_DIR, f"esri_lightgray_{zoom}_{x}_{y}.jpg")
            if not os.path.exists(fn):
                # CARTO's Positron endpoint (used by airbnb_benchmark) now answers every
                # request with an "API KEY REQUIRED" placeholder tile, so use Esri's
                # keyless World Light Gray Canvas instead (note: z/y/x order).
                url = TILE_URL.format(z=zoom, x=x, y=y)
                req = urllib.request.Request(url, headers={"User-Agent": "meb-consent-benchmark/1.0"})
                try:
                    with urllib.request.urlopen(req, timeout=30) as resp, open(fn, "wb") as f:
                        f.write(resp.read())
                except Exception as e:  # keep the grey background for that tile
                    print(f"  tile {zoom}/{x}/{y} failed: {e}")
                    continue
            try:
                img.paste(Image.open(fn).convert("RGB"), ((x - x0) * ts, (y - y0) * ts))
            except Exception as e:
                print(f"  tile {fn} unreadable: {e}")
    n = 2.0 ** zoom
    lng_left = x0 / n * 360.0 - 180.0
    lng_right = (x1 + 1) / n * 360.0 - 180.0
    return img, (x0, y0, x1, y1), (lng_left, lng_right)


def basemap_equirect(lat_min, lat_max, lng_min, lng_max, lat0, lon0, zoom=5, out_rows=900):
    """Stitch Mercator tiles and resample rows so latitude is linear, i.e. the
    image is in the same equirectangular (km) projection as the points.
    Returns (rgb array, extent_km [x_left, x_right, y_bottom, y_top])."""
    from datasets import project_equirect
    img, (x0, y0, x1, y1), (lng_left, lng_right) = fetch_and_stitch_tiles(
        lat_min, lat_max, lng_min, lng_max, zoom)
    arr = np.asarray(img)
    H = arr.shape[0]
    lat_top = _tile_y_to_lat(y0, zoom)
    lat_bot = _tile_y_to_lat(y1 + 1, zoom)
    lats = np.linspace(lat_top, lat_bot, out_rows)
    n = 2.0 ** zoom
    ytile = (1.0 - np.log(np.tan(np.radians(lats)) + 1.0 / np.cos(np.radians(lats))) / math.pi) / 2.0 * n
    rows = np.clip(((ytile - y0) * 256.0).astype(int), 0, H - 1)
    out = arr[rows]
    xl, yt = project_equirect(np.array([lat_top]), np.array([lng_left]), lat0, lon0)
    xr, yb = project_equirect(np.array([lat_bot]), np.array([lng_right]), lat0, lon0)
    return out, [float(xl[0]), float(xr[0]), float(yb[0]), float(yt[0])]
