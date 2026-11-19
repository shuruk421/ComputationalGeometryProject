"""
Small MEB helpers shared by the consent analysis, the benchmark and the plots.

``exact_meb`` is the ground truth used everywhere: the MEB of a point set is
the MEB of its convex-hull vertices, so we take the hull with scipy (qhull) and
run db_consents' own Welzl (``incremental_distance_based`` with every point
marked consenting and a throwaway Oracle) on the hull vertices only.  This is
mathematically identical to running it on all points and lets the analysis use
the whole ~800k-point cleaned sample.
"""

import math
import os
import random
import sys

import numpy as np
from scipy.spatial import ConvexHull, QhullError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import db_consents  # noqa: E402
from db_consents import Oracle, incremental_distance_based  # noqa: E402

db_consents.USE_CPP_BACKEND = False  # the C++ module is a Linux .so


def exact_meb(xy, unit=1.0):
    """
    Exact MEB of the rows of ``xy`` (any dimension).

    ``unit`` rescales coordinates before calling db_consents (see
    flight_data.ALGO_UNIT_KM); the result is returned in the input units.
    Returns (center ndarray, radius float, support ndarray of support points).
    """
    xy = np.unique(np.asarray(xy, dtype=np.float64), axis=0)
    if len(xy) == 0:
        return None, 0.0, np.empty((0, 2))
    verts = xy
    if len(xy) > xy.shape[1] + 1:
        try:
            verts = xy[ConvexHull(xy).vertices]
        except QhullError:  # degenerate (collinear) input
            verts = xy
    pts = [((v / unit).tolist(), True) for v in verts]
    state = random.getstate()  # do not disturb the caller's RNG stream
    random.seed(12345)
    try:
        center, radius = incremental_distance_based(pts, Oracle())
    finally:
        random.setstate(state)
    center = np.asarray(center, dtype=np.float64) * unit
    radius = float(radius) * unit
    d = np.linalg.norm(verts - center, axis=1)
    support = verts[np.abs(d - radius) <= 1e-7 * max(radius, 1.0)]
    return center, radius, support


def disc_hausdorff(c0, r0, c1, r1):
    """Hausdorff distance between two balls: |c0 - c1| + |r0 - r1|."""
    return float(np.linalg.norm(np.asarray(c0) - np.asarray(c1)) + abs(r0 - r1))


def to_points(xy, mask, unit=1.0):
    """db_consents input format: [([x, y], consent_bool), ...]."""
    scaled = np.asarray(xy, dtype=np.float64) / unit
    return [(row, bool(m)) for row, m in zip(scaled.tolist(), mask)]


def support_points(xy, center, radius, rel_tol=1e-7):
    """Rows of ``xy`` lying on the sphere (center, radius)."""
    d = np.linalg.norm(np.asarray(xy) - np.asarray(center), axis=1)
    return np.asarray(xy)[np.abs(d - radius) <= rel_tol * max(radius, 1.0)]


def max_excess(xy, center, radius):
    """Largest distance by which a row of ``xy`` lies outside the ball (<=0 if inside)."""
    if len(xy) == 0:
        return -math.inf
    d = np.linalg.norm(np.asarray(xy) - np.asarray(center), axis=1)
    return float(d.max() - radius)
