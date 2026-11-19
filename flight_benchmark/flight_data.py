"""
Whole-file sampling, cleaning, projection and consent definitions for the
ADS-B flight telemetry dataset (flight.csv, ~9.3 GB, ~51M rows).

Sampling
--------
flight.csv holds one OpenSky-style state vector per second.  Consecutive rows
are near-identical positions of the same flight, so reading only the head of
the file covers a handful of flights.  ``build_sample`` makes ONE streaming
pass over the whole file and keeps every ``every``-th data row (systematic
sampling; with the default ``every=60`` that is one fix per minute of recorded
data, ~850k rows).  Only the kept rows are parsed, so the pass is I/O bound
(well under a minute on an SSD).  The result is cached as a compressed .npz
under ``flight_benchmark/cache/`` (gitignored) keyed by file size, mtime and
``every``; later runs load the cache in a fraction of a second.

Cleaning (``clean_sample``)
---------------------------
* rows with empty or out-of-range LAT/LON are dropped;
* rows with an empty BAROALTITUDE are dropped (the altitude consent column
  must be defined for every point);
* rows whose position is stale (TIME - LASTPOSUPDATE > ``max_pos_age_s``) are
  dropped: OpenSky repeats the last known position while the transponder is
  silent (e.g. parked at a stand), which produces runs of identical rows;
* exact duplicate (LAT, LON) pairs are collapsed to their earliest
  observation.  ``db_consents.Oracle`` caches by coordinate tuple, so two
  records at the same coordinates would be ONE consent request answered with
  the first label it sees.  Collapsing them up front keeps the ground truth
  consistent with what the oracle returns, makes oracle calls equal to
  "distinct locations asked", and keeps degenerate (zero-length) support sets
  out of Welzl.  The consent label of a collapsed location is that of its
  earliest observation.

Geometry
--------
Positions are projected with an azimuthal-equidistant (AEQD) projection
centred on the spherical centroid of the cleaned sample.  Distances from the
centre are exact and other distances are distorted by < ~3% over this extent,
whereas an equirectangular projection would distort east-west distances by
cos(lat) (~0.85 at 32N vs 0.5 at 60N).  The MEB is computed in 2D (see the
module docstring of run_flight_suite.py for the 2D-vs-3D justification).
"""

import json
import math
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "flight.csv")
CACHE_DIR = os.path.join(HERE, "cache")

EARTH_RADIUS_KM = 6371.0088

COLUMNS = [
    "TIME", "ICAO24", "LAT", "LON", "VELOCITY", "HEADING", "VERTRATE",
    "CALLSIGN", "ONGROUND", "ALERT", "SPI", "SQUAWK", "BAROALTITUDE",
    "GEOALTITUDE", "LASTPOSUPDATE", "LASTCONTACT", "TIMESTAMP",
]
_IDX = {name: i for i, name in enumerate(COLUMNS)}


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

def _f(tok):
    """Parse a float field; empty -> NaN."""
    if not tok:
        return math.nan
    try:
        return float(tok)
    except ValueError:
        return math.nan


def _i(tok, default=-1):
    """Parse an int field; empty/garbled -> default."""
    if not tok:
        return default
    try:
        return int(tok)
    except ValueError:
        return default


def cache_path(csv_path, every):
    st = os.stat(csv_path)
    tag = f"{st.st_size}_{int(st.st_mtime)}_k{every}"
    return os.path.join(CACHE_DIR, f"flight_sample_{tag}.npz")


def build_sample(csv_path, every=60, out_path=None, progress=True):
    """
    Stream the whole CSV once and keep every ``every``-th data row.

    Returns the path of the cached .npz.
    """
    out_path = out_path or cache_path(csv_path, every)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    size = os.path.getsize(csv_path)

    cols = {k: [] for k in (
        "t", "lat", "lon", "vel", "vrate", "baro", "geo", "onground",
        "alert", "spi", "squawk", "callsign", "icao24", "posage")}
    n_rows = 0
    n_bad = 0
    first_t = last_t = None
    t0 = time.perf_counter()
    next_report = 1 << 30
    nbytes = 0
    I = _IDX

    with open(csv_path, "rb", buffering=1 << 22) as fh:
        header = fh.readline().decode().strip().split(";")
        if header != COLUMNS:
            raise ValueError(f"Unexpected header: {header}")
        nbytes += 1
        for i, line in enumerate(fh):
            n_rows += 1
            if progress:
                nbytes += len(line)
                if nbytes >= next_report:
                    el = time.perf_counter() - t0
                    print(f"  ... {nbytes / 1e9:.1f} / {size / 1e9:.1f} GB "
                          f"({el:.0f}s, {n_rows / 1e6:.1f}M rows)", flush=True)
                    next_report += 1 << 30
            if i == 0 or i % every == 0:
                p = line.decode("utf-8", "replace").rstrip("\r\n").split(";")
                if len(p) != len(COLUMNS):
                    n_bad += 1
                    continue
                t_ms = _i(p[I["TIME"]])
                if t_ms < 0:
                    n_bad += 1
                    continue
                t_s = t_ms / 1000.0
                if first_t is None:
                    first_t = t_s
                cols["t"].append(t_s)
                cols["lat"].append(_f(p[I["LAT"]]))
                cols["lon"].append(_f(p[I["LON"]]))
                cols["vel"].append(_f(p[I["VELOCITY"]]))
                cols["vrate"].append(_f(p[I["VERTRATE"]]))
                cols["baro"].append(_f(p[I["BAROALTITUDE"]]))
                cols["geo"].append(_f(p[I["GEOALTITUDE"]]))
                cols["onground"].append(_i(p[I["ONGROUND"]]))
                cols["alert"].append(_i(p[I["ALERT"]]))
                cols["spi"].append(_i(p[I["SPI"]]))
                cols["squawk"].append(_i(p[I["SQUAWK"]]))
                cols["callsign"].append(p[I["CALLSIGN"]].strip())
                cols["icao24"].append(p[I["ICAO24"]].strip())
                lpu = _f(p[I["LASTPOSUPDATE"]])
                cols["posage"].append(t_s - lpu if lpu == lpu else math.nan)
            last_line = line
    # last data row of the file (for the time span)
    try:
        last_t = _i(last_line.decode().split(";")[0]) / 1000.0
    except Exception:  # pragma: no cover
        last_t = cols["t"][-1] if cols["t"] else None

    elapsed = time.perf_counter() - t0
    meta = {
        "csv_path": os.path.abspath(csv_path),
        "csv_bytes": size,
        "every": every,
        "n_rows_total": n_rows,
        "n_rows_sampled": len(cols["t"]),
        "n_rows_unparseable": n_bad,
        "first_time_s": first_t,
        "last_time_s": last_t,
        "pass_seconds": round(elapsed, 1),
    }
    arrays = {
        "t": np.asarray(cols["t"], dtype=np.float64),
        "lat": np.asarray(cols["lat"], dtype=np.float64),
        "lon": np.asarray(cols["lon"], dtype=np.float64),
        "vel": np.asarray(cols["vel"], dtype=np.float32),
        "vrate": np.asarray(cols["vrate"], dtype=np.float32),
        "baro": np.asarray(cols["baro"], dtype=np.float32),
        "geo": np.asarray(cols["geo"], dtype=np.float32),
        "onground": np.asarray(cols["onground"], dtype=np.int8),
        "alert": np.asarray(cols["alert"], dtype=np.int8),
        "spi": np.asarray(cols["spi"], dtype=np.int8),
        "squawk": np.asarray(cols["squawk"], dtype=np.int32),
        "callsign": np.asarray(cols["callsign"], dtype="<U8"),
        "icao24": np.asarray(cols["icao24"], dtype="<U6"),
        "posage": np.asarray(cols["posage"], dtype=np.float32),
    }
    np.savez_compressed(out_path, meta=np.array(json.dumps(meta)), **arrays)
    if progress:
        print(f"Sampled {meta['n_rows_sampled']:,} of {n_rows:,} rows in "
              f"{elapsed:.0f}s -> {out_path}", flush=True)
    return out_path


def load_sample(csv_path=DEFAULT_CSV, every=60, rebuild=False):
    """Load (building if needed) the cached whole-file systematic sample."""
    path = cache_path(csv_path, every)
    if rebuild or not os.path.exists(path):
        print(f"Building whole-file sample of {csv_path} (every {every}th row)...",
              flush=True)
        build_sample(csv_path, every=every, out_path=path)
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files if k != "meta"}
        meta = json.loads(str(z["meta"]))
    return data, meta


# --------------------------------------------------------------------------
# Cleaning and flight-leg segmentation
# --------------------------------------------------------------------------

def segment_legs(t, callsign, max_gap_s=30 * 60):
    """
    Assign a flight-leg id to each (time-sorted) row: a new leg starts when the
    callsign changes or there is a gap of more than ``max_gap_s`` seconds.
    """
    order = np.argsort(t, kind="stable")
    leg = np.empty(len(t), dtype=np.int32)
    cur = -1
    prev_t = None
    prev_cs = None
    for idx in order:
        ti = t[idx]
        cs = callsign[idx]
        if prev_t is None or cs != prev_cs or ti - prev_t > max_gap_s:
            cur += 1
        leg[idx] = cur
        prev_t, prev_cs = ti, cs
    return leg


def _jump_outliers(t, lat, lon, max_speed_ms=350.0, max_dt_s=600.0):
    """
    Flag garbled ADS-B positions (bad CPR decodes): a fix is an outlier when the
    ground speed implied by EVERY time-neighbour within ``max_dt_s`` exceeds
    ``max_speed_ms`` (an A320 rarely exceeds ~300 m/s ground speed).
    ``t`` must be sorted.  Fixes without a neighbour within ``max_dt_s`` are kept.
    """
    la, lo = np.radians(lat), np.radians(lon)
    # haversine distance between consecutive fixes (km)
    dla, dlo = np.diff(la), np.diff(lo)
    h = np.sin(dla / 2) ** 2 + np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(dlo / 2) ** 2
    d_m = 2 * EARTH_RADIUS_KM * 1000 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
    dt = np.diff(t)
    near = dt <= max_dt_s
    fast = near & (d_m > max_speed_ms * np.maximum(dt, 1.0))
    n = len(t)
    has_prev = np.zeros(n, bool); has_next = np.zeros(n, bool)
    prev_fast = np.zeros(n, bool); next_fast = np.zeros(n, bool)
    has_prev[1:] = near; prev_fast[1:] = fast
    has_next[:-1] = near; next_fast[:-1] = fast
    any_nb = has_prev | has_next
    all_fast = (~has_prev | prev_fast) & (~has_next | next_fast)
    return any_nb & all_fast


def clean_sample(data, max_pos_age_s=60.0, max_speed_ms=350.0,
                 baro_range=(-500.0, 13000.0), max_velocity_ms=350.0):
    """
    Apply the cleaning rules from the module docstring.

    Returns (clean, stats) where ``clean`` is a dict of equally long arrays
    (time-sorted, duplicates collapsed) and ``stats`` documents what was removed.
    """
    n0 = len(data["t"])
    lat, lon = data["lat"], data["lon"]
    ok_pos = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90) & (np.abs(lon) <= 180)
    # A handful of OpenSky fixes are exactly (0, 0) or similar junk.
    ok_pos &= ~((np.abs(lat) < 1e-6) & (np.abs(lon) < 1e-6))
    baro = data["baro"]
    ok_alt = np.isfinite(baro)
    ok_alt_range = ok_alt & (baro >= baro_range[0]) & (baro <= baro_range[1])
    vel = data["vel"]
    ok_vel = ~(np.isfinite(vel) & (vel > max_velocity_ms))
    age = data["posage"]
    ok_age = np.isfinite(age) & (age <= max_pos_age_s)
    stats = {
        "rows_sampled": int(n0),
        "missing_or_invalid_latlon": int((~ok_pos).sum()),
        "missing_baroaltitude": int((ok_pos & ~ok_alt).sum()),
        "missing_geoaltitude": int((ok_pos & ~np.isfinite(data["geo"])).sum()),
        "implausible_baroaltitude": int((ok_pos & ok_alt & ~ok_alt_range).sum()),
        "implausible_velocity": int((ok_pos & ok_alt_range & ~ok_vel).sum()),
        "stale_position_rows": int((ok_pos & ok_alt_range & ok_vel & ~ok_age).sum()),
        "max_pos_age_s": max_pos_age_s,
    }
    keep = ok_pos & ok_alt_range & ok_vel & ok_age
    idx = np.nonzero(keep)[0]
    idx = idx[np.argsort(data["t"][idx], kind="stable")]

    # Garbled positions: two passes so that pairs of consecutive bad fixes,
    # whose mutual implied speed is small, are caught once their good
    # neighbours are compared directly.
    n_jump = 0
    for _ in range(2):
        bad = _jump_outliers(data["t"][idx], lat[idx], lon[idx], max_speed_ms)
        n_jump += int(bad.sum())
        idx = idx[~bad]
    stats["position_jump_outliers"] = n_jump
    stats["max_implied_speed_ms"] = max_speed_ms

    # Collapse exact duplicate coordinates to their earliest observation.
    seen = set()
    first = []
    for j in idx:
        key = (lat[j], lon[j])
        if key in seen:
            continue
        seen.add(key)
        first.append(j)
    first = np.asarray(first, dtype=np.int64)
    stats["duplicate_latlon_rows"] = int(len(idx) - len(first))
    stats["points_after_cleaning"] = int(len(first))

    clean = {k: v[first] for k, v in data.items()}
    clean["leg"] = segment_legs(clean["t"], clean["callsign"])
    stats["flight_legs"] = int(clean["leg"].max() + 1) if len(first) else 0
    return clean, stats


# --------------------------------------------------------------------------
# Projection
# --------------------------------------------------------------------------

def spherical_centroid(lat, lon):
    """Lat/lon of the normalised mean of the unit vectors."""
    la, lo = np.radians(lat), np.radians(lon)
    x = np.mean(np.cos(la) * np.cos(lo))
    y = np.mean(np.cos(la) * np.sin(lo))
    z = np.mean(np.sin(la))
    return float(np.degrees(np.arctan2(z, math.hypot(x, y)))), float(np.degrees(np.arctan2(y, x)))


class AEQD:
    """Azimuthal-equidistant projection (spherical Earth), output in km."""

    def __init__(self, lat0, lon0, R=EARTH_RADIUS_KM):
        self.lat0, self.lon0, self.R = lat0, lon0, R
        self._p0 = math.radians(lat0)
        self._l0 = math.radians(lon0)

    def forward(self, lat, lon):
        p = np.radians(np.asarray(lat, dtype=np.float64))
        dl = np.radians(np.asarray(lon, dtype=np.float64)) - self._l0
        sp0, cp0 = math.sin(self._p0), math.cos(self._p0)
        cosc = np.clip(sp0 * np.sin(p) + cp0 * np.cos(p) * np.cos(dl), -1.0, 1.0)
        c = np.arccos(cosc)
        with np.errstate(invalid="ignore", divide="ignore"):
            k = np.where(c > 1e-12, c / np.sin(c), 1.0)
        x = self.R * k * np.cos(p) * np.sin(dl)
        y = self.R * k * (cp0 * np.sin(p) - sp0 * np.cos(p) * np.cos(dl))
        return x, y

    def inverse(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        rho = np.hypot(x, y)
        c = rho / self.R
        sp0, cp0 = math.sin(self._p0), math.cos(self._p0)
        with np.errstate(invalid="ignore", divide="ignore"):
            lat = np.arcsin(np.clip(np.cos(c) * sp0 + np.where(rho > 0, y * np.sin(c) * cp0 / rho, 0.0), -1, 1))
            lon = self._l0 + np.arctan2(x * np.sin(c), rho * cp0 * np.cos(c) - y * sp0 * np.sin(c))
        return np.degrees(lat), (np.degrees(lon) + 180.0) % 360.0 - 180.0

    def circle_latlon(self, cx, cy, r, n=361):
        """Lat/lon polyline of the projected-plane circle (centre cx, cy; radius r km)."""
        a = np.linspace(0, 2 * math.pi, n)
        return self.inverse(cx + r * np.cos(a), cy + r * np.sin(a))


# --------------------------------------------------------------------------
# Working set: cleaned sample + projection, in the units the algorithms see
# --------------------------------------------------------------------------

# db_consents compares SQUARED distances with an ABSOLUTE tolerance of 1e-10
# (GEOMETRY_TOLERANCE).  With coordinates in km (radius ~1800 km, r^2 ~ 3e6)
# the rounding error of r^2 (~1e-9) exceeds that tolerance, so
# decremental_distance_based fails to recognise support points as "on the
# boundary" and stops early with a too-large ball (see the correctness section
# of the report).  The algorithms are therefore fed coordinates in units of
# 1000 km; everything reported is converted back to km.
ALGO_UNIT_KM = 1000.0


class FlightSet:
    """Cleaned, projected flight positions plus the attributes consent uses."""

    def __init__(self, csv_path=DEFAULT_CSV, every=60, rebuild=False):
        data, meta = load_sample(csv_path, every=every, rebuild=rebuild)
        clean, stats = clean_sample(data)
        self.meta = meta
        self.clean_stats = stats
        self.c = clean
        self.n = len(clean["t"])
        self.lat0, self.lon0 = spherical_centroid(clean["lat"], clean["lon"])
        self.proj = AEQD(self.lat0, self.lon0)
        x, y = self.proj.forward(clean["lat"], clean["lon"])
        self.xy_km = np.column_stack([x, y])
        # leg start time (each leg consents as a unit for the date column)
        leg = clean["leg"]
        start = np.full(leg.max() + 1, np.inf)
        np.minimum.at(start, leg, clean["t"])
        self.leg_start = start[leg]

    def describe(self):
        import datetime as _dt
        c = self.c
        fmt = lambda s: _dt.datetime.utcfromtimestamp(s).strftime("%Y-%m-%d %H:%M UTC")
        return {
            "rows_in_file": self.meta["n_rows_total"],
            "sample_every_nth_row": self.meta["every"],
            "rows_sampled": self.meta["n_rows_sampled"],
            "points_after_cleaning": self.n,
            "cleaning": self.clean_stats,
            "time_first_row": fmt(self.meta["first_time_s"]),
            "time_last_row": fmt(self.meta["last_time_s"]),
            "distinct_icao24": int(len(np.unique(c["icao24"]))),
            "distinct_callsigns": int(len(np.unique(c["callsign"]))),
            "flight_legs": int(c["leg"].max() + 1),
            "lat_range": [float(c["lat"].min()), float(c["lat"].max())],
            "lon_range": [float(c["lon"].min()), float(c["lon"].max())],
            "projection": f"azimuthal equidistant centred on ({self.lat0:.3f}N, {self.lon0:.3f}E)",
            "max_distance_from_centre_km": float(np.hypot(*self.xy_km.T).max()),
            "baroaltitude_m_range": [float(c["baro"].min()), float(c["baro"].max())],
        }


# --------------------------------------------------------------------------
# Consent columns
# --------------------------------------------------------------------------

def rank_entities_by_size(labels):
    """Return entity labels sorted by number of points (desc), ties by label."""
    uniq, counts = np.unique(labels, return_counts=True)
    order = sorted(range(len(uniq)), key=lambda i: (-counts[i], uniq[i]))
    return uniq[order], counts[order]


def top_entities_mask(labels, target_frac):
    """
    Consent = point belongs to one of the k largest entities, with k the
    smallest number reaching ``target_frac`` of the points.
    """
    ents, counts = rank_entities_by_size(labels)
    cum = np.cumsum(counts) / counts.sum()
    k = int(np.searchsorted(cum, target_frac) + 1)
    return np.isin(labels, ents[:k]), k


def hashed_entity_mask(labels, target_frac, salt="consent"):
    """
    Pseudo-random per-entity consent: each entity consents iff a stable hash of
    its label falls below ``target_frac``.  All of an entity's points share
    the decision.
    """
    import hashlib
    uniq, inv = np.unique(labels, return_inverse=True)
    u = np.array([int.from_bytes(hashlib.blake2b(f"{salt}:{v}".encode(), digest_size=8).digest(),
                                 "big") / 2 ** 64 for v in uniq])
    return (u < target_frac)[inv]


def _ge_quantile(values, frac):
    """Mask values >= the (1-frac) quantile, plus the threshold used."""
    thr = float(np.quantile(values, 1.0 - frac, method="higher"))
    return values >= thr, thr


def _lt_quantile(values, frac):
    thr = float(np.quantile(values, frac, method="higher"))
    return values < thr, thr


def _utc_date(s):
    import datetime as _dt
    return _dt.datetime.utcfromtimestamp(s).strftime("%Y-%m-%d")


def consent_optin_date(fs, frac):
    """Flight legs that started on/after an opt-in date consent (per leg)."""
    m, thr = _ge_quantile(fs.leg_start, frac)
    return m, f"legs since {_utc_date(thr)}", thr


def consent_baro_above(fs, frac):
    m, thr = _ge_quantile(fs.c["baro"], frac)
    return m, f"baro alt >= {thr:,.0f} m", thr


def consent_baro_below(fs, frac):
    m, thr = _lt_quantile(fs.c["baro"], frac)
    return m, f"baro alt < {thr:,.0f} m", thr


def consent_geo_above(fs, frac):
    m, thr = _ge_quantile(fs.c["geo"], frac)
    return m, f"geo alt >= {thr:,.0f} m", thr


def consent_velocity_above(fs, frac):
    m, thr = _ge_quantile(fs.c["vel"], frac)
    return m, f"velocity >= {thr:.0f} m/s", thr


def consent_hour_of_day(fs, frac):
    """Positions recorded before a cut-off UTC time of day."""
    tod = np.mod(fs.c["t"], 86400.0) / 3600.0
    m, thr = _lt_quantile(tod, frac)
    hh, mm = int(thr), int((thr % 1) * 60)
    return m, f"before {hh:02d}:{mm:02d} UTC", thr


def consent_squawk_below(fs, frac):
    m, thr = _lt_quantile(fs.c["squawk"].astype(np.float64), frac)
    return m, f"squawk < {int(thr):04d}", thr


def consent_flightnum_top(fs, frac):
    """The k busiest flight numbers (callsigns) consent, k set by frac."""
    m, k = top_entities_mask(fs.c["callsign"], frac)
    return m, f"top {k} flight numbers", k


def consent_flightnum_random(fs, frac):
    m = hashed_entity_mask(fs.c["callsign"], frac, salt="flightnum")
    return m, f"random flight numbers p={frac:.2f}", frac


def consent_leg_random(fs, frac):
    m = hashed_entity_mask(fs.c["leg"], frac, salt="leg")
    return m, f"random flight legs p={frac:.2f}", frac


def consent_point_random(fs, frac, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random(fs.n) < frac, f"random points p={frac:.2f}", frac


# name -> (function, story).  Sweepable: f(fs, frac) -> (mask, label, threshold)
SWEEP_CANDIDATES = {
    "optin_date": (consent_optin_date,
                   "Opt-in date: every flight leg flown on/after date T is shared "
                   "(a leg consents as a unit); T swept by quantile."),
    "flightnum_top": (consent_flightnum_top,
                      "Per flight number (CALLSIGN ~ route): the k busiest flight numbers "
                      "have a data-sharing agreement."),
    "baro_above": (consent_baro_above,
                   "BAROALTITUDE >= h: only en-route/cruise positions are shareable; "
                   "low-altitude fixes near airports are withheld."),
    "baro_below": (consent_baro_below,
                   "BAROALTITUDE < h: only terminal-area (low) positions are shareable."),
    "geo_above": (consent_geo_above, "GEOALTITUDE >= h (GNSS height)."),
    "velocity_above": (consent_velocity_above, "VELOCITY >= v (ground speed)."),
    "hour_of_day": (consent_hour_of_day, "Positions recorded before a UTC time-of-day cut-off."),
    "squawk_below": (consent_squawk_below,
                     "SQUAWK < s (codes are allocated in blocks to ATC units)."),
    "flightnum_random": (consent_flightnum_random,
                         "Per flight number, pseudo-random (hashed) decision with probability p."),
    "leg_random": (consent_leg_random,
                   "Per flight leg, pseudo-random (hashed) decision with probability p."),
    "point_random": (consent_point_random, "Independent coin flip per position (baseline)."),
}


def fixed_candidates(fs):
    """Binary columns that cannot be swept: name -> (mask, story)."""
    c = fs.c
    uniq, counts = np.unique(c["icao24"], return_counts=True)
    top_icao = uniq[np.argmax(counts)]
    prefix = np.array([s[:3] for s in c["callsign"]])
    return {
        "icao24": (c["icao24"] == top_icao,
                   f"Per aircraft: ICAO24 {top_icao} consents (the file holds a single aircraft)."),
        "airline_prefix": (prefix == "WZZ",
                           "Per operator: CALLSIGN prefix WZZ (Wizz Air); all other prefixes are "
                           "garbled callsigns of the same aircraft."),
        "onground": (c["onground"] == 1, "ONGROUND == 1."),
        "alert": (c["alert"] == 1, "ALERT flag (previous attempt)."),
        "spi": (c["spi"] == 1, "SPI flag (previous attempt)."),
        "alert_or_spi": ((c["alert"] == 1) | (c["spi"] == 1), "ALERT or SPI (previous attempt)."),
    }
