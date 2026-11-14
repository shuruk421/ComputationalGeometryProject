"""
Data loading, geometry and consent columns for the Amsterdam review-consent study.

Geometry: every listing becomes a 2D point (x_km, y_km) in a local
equirectangular projection centred on the data centroid:

    x = R * d_lng * cos(lat0),   y = R * d_lat      (angles in radians)

so Euclidean distance is (to within ~0.2% over the 20 km city extent) the
ground distance in km and the minimum enclosing ball is a real geographic disc.

Consent: a listing "consents" when its score in a chosen column is >= a
threshold. Thresholds are chosen from the column's own quantiles so that the
consenting fraction spans a wide range.
"""

import csv
import datetime
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

EARTH_RADIUS_KM = 6371.0088
GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))


@dataclass
class ConsentColumn:
    """A thresholdable consent signal: consent <=> score >= threshold."""

    name: str
    label: str  # short label used in figure titles
    story: str  # why would such a host agree to share their data?
    score: np.ndarray  # higher = more likely to consent; -inf = never consents
    describe: Callable[[float], str]  # threshold -> human readable rule
    weak_story: bool = False  # excluded from the automatic "best alternative" pick

    def consent(self, threshold: float) -> np.ndarray:
        return self.score >= threshold


@dataclass
class Listings:
    n: int
    lat: np.ndarray
    lng: np.ndarray
    lat0: float
    lng0: float
    xy: np.ndarray  # (n, 2) km, after duplicate jitter
    raw: Dict[str, np.ndarray]
    snapshot_date: datetime.date
    n_duplicate_rows: int  # rows whose exact (lat, lng) was already taken
    n_duplicate_groups: int
    columns: Dict[str, ConsentColumn] = field(default_factory=dict)

    # ---- projection helpers -------------------------------------------------
    def to_km(self, lat, lng):
        lat = np.asarray(lat, dtype=float)
        lng = np.asarray(lng, dtype=float)
        x = EARTH_RADIUS_KM * np.radians(lng - self.lng0) * math.cos(math.radians(self.lat0))
        y = EARTH_RADIUS_KM * np.radians(lat - self.lat0)
        return x, y

    def to_latlng(self, x_km, y_km):
        x_km = np.asarray(x_km, dtype=float)
        y_km = np.asarray(y_km, dtype=float)
        lng = self.lng0 + np.degrees(x_km / (EARTH_RADIUS_KM * math.cos(math.radians(self.lat0))))
        lat = self.lat0 + np.degrees(y_km / EARTH_RADIUS_KM)
        return lat, lng


def _num(rows, col, missing=np.nan):
    out = np.empty(len(rows))
    for i, r in enumerate(rows):
        v = r[col].strip()
        out[i] = float(v) if v else missing
    return out


def load_listings(csv_path: str, dup_jitter_m: float = 1.0) -> Listings:
    """Load the Inside Airbnb summary CSV and project it to local km.

    Listings that share an exact (lat, lng) with an earlier listing are moved by
    ``dup_jitter_m`` metres (on a golden-angle spiral, so a group of k
    duplicates becomes k distinct points). This is needed because
    ``db_consents.Oracle`` caches consent answers by *coordinates*: without it,
    two co-located listings with different consent would share one cached
    answer. 1 m is far below Inside Airbnb's own location anonymisation
    (up to ~150 m). Pass 0 to disable.
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    lat = _num(rows, "latitude")
    lng = _num(rows, "longitude")
    keep = np.isfinite(lat) & np.isfinite(lng)
    rows = [r for r, k in zip(rows, keep) if k]
    lat, lng = lat[keep], lng[keep]
    n = len(rows)

    lat0, lng0 = float(lat.mean()), float(lng.mean())
    listings = Listings(
        n=n, lat=lat, lng=lng, lat0=lat0, lng0=lng0, xy=np.zeros((n, 2)), raw={},
        snapshot_date=datetime.date(1970, 1, 1), n_duplicate_rows=0, n_duplicate_groups=0,
    )
    x, y = listings.to_km(lat, lng)
    xy = np.c_[x, y]

    # --- jitter exact duplicate coordinates ----------------------------------
    seen: Dict[tuple, int] = {}
    dup_rows = 0
    for i in range(n):
        key = (lat[i], lng[i])
        k = seen.get(key, 0)
        if k and dup_jitter_m > 0:
            ang = k * GOLDEN_ANGLE
            xy[i, 0] += dup_jitter_m / 1000.0 * math.cos(ang)
            xy[i, 1] += dup_jitter_m / 1000.0 * math.sin(ang)
        if k:
            dup_rows += 1
        seen[key] = k + 1
    listings.xy = xy
    listings.n_duplicate_rows = dup_rows
    listings.n_duplicate_groups = sum(1 for v in seen.values() if v > 1)

    # --- raw columns ----------------------------------------------------------
    raw = {
        "number_of_reviews": _num(rows, "number_of_reviews"),
        "number_of_reviews_ltm": _num(rows, "number_of_reviews_ltm"),
        # missing reviews_per_month means the listing has no reviews at all
        "reviews_per_month": _num(rows, "reviews_per_month", missing=0.0),
        "availability_365": _num(rows, "availability_365"),
        "calculated_host_listings_count": _num(rows, "calculated_host_listings_count"),
        "price": _num(rows, "price"),
    }
    dates = [r["last_review"].strip() for r in rows]
    snapshot = max(datetime.date.fromisoformat(d) for d in dates if d)
    listings.snapshot_date = snapshot
    raw["days_since_last_review"] = np.array(
        [(snapshot - datetime.date.fromisoformat(d)).days if d else np.inf for d in dates]
    )
    lic = [r["license"].strip() for r in rows]
    raw["license_registered"] = np.array([1.0 if (s and "exempt" not in s.lower()) else 0.0 for s in lic])
    raw["license_exempt"] = np.array([1.0 if "exempt" in s.lower() else 0.0 for s in lic])
    raw["not_entire_home"] = np.array([0.0 if r["room_type"] == "Entire home/apt" else 1.0 for r in rows])
    listings.raw = raw
    listings.columns = build_consent_columns(raw)
    return listings


def _int_fmt(t):
    return f"{int(round(t)):,}"


def build_consent_columns(raw: Dict[str, np.ndarray]) -> Dict[str, ConsentColumn]:
    cols = [
        ConsentColumn(
            "number_of_reviews", "number of reviews",
            "Established hosts with a long public track record have a reputation to protect "
            "and are used to scrutiny, so they agree to share data.",
            raw["number_of_reviews"], lambda t: f"reviews ≥ {_int_fmt(t)}"),
        ConsentColumn(
            "number_of_reviews_ltm", "reviews in last 12 months",
            "Hosts who are active right now are reachable and engaged with the platform.",
            raw["number_of_reviews_ltm"], lambda t: f"reviews (12 mo) ≥ {_int_fmt(t)}"),
        ConsentColumn(
            "reviews_per_month", "reviews per month",
            "High-turnover hosts run the listing as a business and answer requests quickly.",
            raw["reviews_per_month"], lambda t: f"reviews/month ≥ {t:g}"),
        ConsentColumn(
            "availability_365", "availability (days/yr)",
            "Listings open most of the year are run professionally (B&Bs, hotel rooms, "
            "multi-listing operators) and already deal with the city's registration rules; "
            "occasional home-sharers under Amsterdam's 30-night whole-home cap are warier.",
            raw["availability_365"], lambda t: f"available ≥ {_int_fmt(t)} d/yr"),
        ConsentColumn(
            "last_review_recency", "last review recency",
            "A recent review means the host is currently operating and reachable.",
            -raw["days_since_last_review"], lambda t: f"reviewed ≤ {_int_fmt(-t)} d ago"),
        ConsentColumn(
            "calculated_host_listings_count", "host's listing count",
            "Multi-listing (commercial) hosts have staff and compliance processes.",
            raw["calculated_host_listings_count"], lambda t: f"host listings ≥ {_int_fmt(t)}"),
        ConsentColumn(
            "price", "nightly price (blank = no)",
            "Premium listings are run professionally. Weak story, and 44% of rows have no price "
            "(treated as not consenting), so it cannot exceed ~56% consenting.",
            np.where(np.isnan(raw["price"]), -np.inf, raw["price"]),
            lambda t: f"price ≥ €{_int_fmt(t)}", weak_story=True),
    ]
    return {c.name: c for c in cols}


BINARY_CANDIDATES = {
    "license_registered": "has a registration number (not exempt / blank)",
    "license_exempt": "license field says 'Exempt'",
    "not_entire_home": "private / shared / hotel room (not entire home)",
}


def thresholds_for_fractions(score: np.ndarray, targets: Sequence[float]) -> List[float]:
    """For each target consenting fraction pick the threshold (an actual value
    of the column) whose consenting fraction P(score >= t) is closest to it.
    Duplicate thresholds (from ties in the column) are dropped, order kept."""
    finite = np.sort(score[np.isfinite(score)])
    n = len(score)
    vals = np.unique(finite)
    # fraction with score >= v; -inf scores never consent
    frac = (len(finite) - np.searchsorted(finite, vals, side="left")) / n
    out: List[float] = []
    for t in targets:
        v = float(vals[int(np.argmin(np.abs(frac - t)))])
        if v not in out:
            out.append(v)
    return out


def achievable_range(score: np.ndarray) -> Tuple[float, float]:
    """(smallest, largest < 100%) consenting fraction any threshold can give."""
    finite = np.sort(score[np.isfinite(score)])
    vals = np.unique(finite)
    frac = (len(finite) - np.searchsorted(finite, vals, side="left")) / len(score)
    frac = frac[frac < 1 - 1e-12]
    return float(frac.min()), float(frac.max())
