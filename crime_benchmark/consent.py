"""
Consent rules for the Chicago crimes MEB experiments, and MEB helpers.

A consent rule maps a crime record to True (the record may be used) or False.
Two kinds:

* boolean rules, e.g. ARREST, DOMESTIC, PRIMARY=THEFT, "not at the airport";
* threshold families, a column plus a direction, e.g. ``TIMESTAMP >= t`` or
  ``DISTRICT <= k``; the cutoff is picked from quantiles so the consenting
  fraction can be swept (roughly 5% .. 95%).
"""

import bisect
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import db_consents  # noqa: E402
from db_consents import Oracle, decremental_distance_based, incremental_distance_based  # noqa: E402

db_consents.USE_CPP_BACKEND = False  # the C++ module is a Linux .so

ALGORITHMS = ("incremental", "decremental")

# FBI UCR codes in the order of the UCR hierarchy (Part I most serious first,
# then Part II). Used as a severity rank.
FBI_ORDER = ["01A", "01B", "02", "03", "04A", "04B", "05", "06", "07", "09",
             "08A", "08B", "10", "11", "12", "13", "14", "15", "16", "17", "18",
             "19", "20", "22", "24", "26"]
FBI_RANK = {c: i for i, c in enumerate(FBI_ORDER)}
FBI_PART_I = set(FBI_ORDER[:10])

RESIDENTIAL_WORDS = ("RESIDENCE", "RESIDENTIAL", "APARTMENT", "CHA APARTMENT", "HOUSE",
                     "NURSING HOME", "COLLEGE/UNIVERSITY RESIDENCE HALL")
OUTDOOR_PUBLIC = {"STREET", "SIDEWALK", "ALLEY", "PARK PROPERTY", "HIGHWAY/EXPRESSWAY",
                  "VACANT LOT/LAND", "LAKEFRONT/WATERFRONT/RIVERBANK", "BRIDGE",
                  "CHA PARKING LOT/GROUNDS", "PARKING LOT/GARAGE(NON.RESID.)"}


def is_residential(r):
    loc = r["location_desc"]
    return any(w in loc for w in RESIDENTIAL_WORDS)


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------

BOOLEAN_RULES = {
    "ARREST": ("arrest made", lambda r: r["arrest"] == 1),
    "NOT ARREST": ("no arrest made", lambda r: r["arrest"] == 0),
    "DOMESTIC": ("domestic incident", lambda r: r["domestic"] == 1),
    "NOT DOMESTIC": ("not a domestic incident", lambda r: r["domestic"] == 0),
    "PRIMARY=THEFT": ("primary type THEFT", lambda r: r["primary"] == "THEFT"),
    "PRIMARY=BATTERY": ("primary type BATTERY", lambda r: r["primary"] == "BATTERY"),
    "PRIMARY=NARCOTICS": ("primary type NARCOTICS", lambda r: r["primary"] == "NARCOTICS"),
    "FBI Part II": ("FBI UCR Part II (less serious) offence", lambda r: r["fbi"] not in FBI_PART_I),
    "LOCATION non-residential": ("incident not at a residence", lambda r: not is_residential(r)),
    "LOCATION outdoor public": ("street, sidewalk, alley, park, lot ...",
                                lambda r: r["location_desc"] in OUTDOOR_PUBLIC),
    "LOCATION not airport": ("incident not on airport property",
                             lambda r: "AIRPORT" not in r["location_desc"]),
}


class ThresholdFamily:
    """consent(r) = value(r) >= cutoff  (direction '>=')  or  <= cutoff ('<=')."""

    def __init__(self, name, column, direction, value, story, fmt=str):
        self.name = name
        self.column = column
        self.direction = direction
        self.value = value
        self.story = story
        self.fmt = fmt

    def rule(self, cutoff):
        v = self.value
        if self.direction == ">=":
            return lambda r: v(r) >= cutoff
        return lambda r: v(r) <= cutoff

    def label(self, cutoff):
        return f"{self.column} {self.direction} {self.fmt(cutoff)}"

    def cutoff_for_fraction(self, records, target):
        """Cutoff whose consenting fraction over `records` is closest to target.

        Candidate cutoffs are the distinct values, so discrete columns
        (DISTRICT, YEAR) snap to the nearest achievable fraction."""
        vals = sorted(self.value(r) for r in records)
        n = len(vals)
        distinct = sorted(set(vals))
        best, best_err = None, None
        for c in distinct:
            if self.direction == ">=":
                frac = (n - bisect.bisect_left(vals, c)) / n
            else:
                frac = bisect.bisect_right(vals, c) / n
            err = abs(frac - target)
            if best_err is None or err < best_err:
                best, best_err = c, err
        return best

    def fraction(self, records, cutoff):
        f = self.rule(cutoff)
        return sum(1 for r in records if f(r)) / len(records)


def _fmt_ts(ms):
    return time.strftime("%Y-%m-%d", time.gmtime(ms / 1000.0))


THRESHOLD_FAMILIES = {
    "TIMESTAMP": ThresholdFamily(
        "TIMESTAMP", "TIMESTAMP", ">=", lambda r: r["timestamp"],
        "records reported on/after a consent-policy date are consented", fmt=_fmt_ts),
    "YEAR": ThresholdFamily(
        "YEAR", "YEAR", ">=", lambda r: r["year"],
        "records from year >= Y are consented"),
    "DISTRICT": ThresholdFamily(
        "DISTRICT", "DISTRICT", "<=", lambda r: r["district"],
        "consent programme rolled out police district by district (in district-number order)"),
    "COMMUNITY": ThresholdFamily(
        "COMMUNITY", "COMMUNITY", "<=", lambda r: r["community"],
        "community areas opt in in official numbering order (north to south, O'Hare = 76 last)"),
    "WARD": ThresholdFamily(
        "WARD", "WARD", "<=", lambda r: r["ward"],
        "aldermanic wards opt in in ward-number order"),
    "FBI severity": ThresholdFamily(
        "FBI severity", "FBI code (UCR order)", ">=", lambda r: FBI_RANK.get(r["fbi"], len(FBI_ORDER)),
        "less serious offences (higher UCR rank) are shareable",
        fmt=lambda k: FBI_ORDER[k] if k < len(FBI_ORDER) else "?"),
}


def make_points(rows, coords, rule):
    """[(coords, consent_bool), ...] in the format db_consents expects."""
    return [(c, bool(rule(r))) for r, c in zip(rows, coords)]


# --------------------------------------------------------------------------
# MEB helpers
# --------------------------------------------------------------------------

def exact_meb(coords, seed=0):
    """Ground-truth MEB of a plain list of points (no consent involved).

    Uses db_consents' Welzl with every point marked consenting and a throwaway
    Oracle, i.e. `incremental_distance_based` on consenting-only points."""
    if not coords:
        return None, 0.0
    state = random.getstate()
    random.seed(seed)
    try:
        center, radius = incremental_distance_based([(c, True) for c in coords], Oracle())
    finally:
        random.setstate(state)
    return list(center), radius


def max_dist(coords, center):
    cx, cy = center[0], center[1]
    return max(math.hypot(p[0] - cx, p[1] - cy) for p in coords)


def support_points(coords, center, radius, rel_tol=1e-9):
    cx, cy = center[0], center[1]
    tol = max(radius * rel_tol, 1e-9)
    return [p for p in coords if abs(math.hypot(p[0] - cx, p[1] - cy) - radius) <= tol]


def disk_shift(c1, r1, c2, r2):
    """Hausdorff distance between two disks = |c1 - c2| + |r1 - r2|."""
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1]) + abs(r1 - r2)


def run_algorithm(name, points, seed):
    """Run one algorithm; returns dict(center, radius, calls, seconds)."""
    fn = incremental_distance_based if name == "incremental" else decremental_distance_based
    oracle = Oracle()
    random.seed(seed)
    t0 = time.perf_counter()
    center, radius = fn(points, oracle)
    dt = time.perf_counter() - t0
    return {"center": None if center is None else list(center), "radius": float(radius),
            "calls": oracle.get_call_count(), "seconds": dt}
