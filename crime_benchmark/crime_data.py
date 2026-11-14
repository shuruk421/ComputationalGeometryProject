"""
Data access for the Chicago crimes MEB experiments.

* ``build_sample`` makes ONE streaming pass over ``Crimes_Ordered.csv`` (1.3 GB,
  ordered by time) and draws a uniform reservoir sample of rows from the whole
  file (Vitter's Algorithm L, so only the kept rows are parsed). Rows with a
  missing or out-of-bounds LATITUDE/LONGITUDE are dropped. The parsed sample is
  shuffled and cached to ``sample_cache/`` (gitignored) so later runs take a few
  seconds instead of a full pass.
* ``project_km`` maps lat/lng to a local equirectangular plane in kilometres
  (x east, y north) around a fixed reference point (the sample centroid), so a
  ball in these coordinates is a real geographic disk.
* ``draw_points`` draws n incidents for a seed and removes duplicate geocodes
  (see the docstring for why), returning rows aligned with projected points.
"""

import csv
import gzip
import itertools
import json
import math
import os
import random
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "Crimes_Ordered.csv")
CACHE_DIR = os.path.join(HERE, "sample_cache")

# Generous box around the City of Chicago (city limits are roughly
# 41.64..42.03 N, -87.94..-87.52 E). Rows outside it are bogus geocodes, e.g.
# the well-known (36.62, -91.69) placeholder.
LAT_MIN, LAT_MAX = 41.60, 42.10
LNG_MIN, LNG_MAX = -87.95, -87.50

EARTH_RADIUS_KM = 6371.0088
SNAP_KM = 0.001  # 1 m grid used to identify duplicate geocodes

# Header of the source file, used to locate columns.
SOURCE_COLUMNS = [
    "ID", "CASE", "BLOCK", "IUCR", "PRIMARY", "DESCRIPTION", "LOCATION_DESC",
    "ARREST", "DOMESTIC", "BEAT", "DISTRICT", "WARD", "COMMUNITY", "FBI", "X", "Y",
    "YEAR", "UPDATED", "LATITUDE", "LONGITUDE", "TIMESTAMP",
]

# Columns kept in the cached sample, in order.
CACHE_FIELDS = [
    "id", "lat", "lng", "timestamp", "year", "arrest", "domestic", "primary",
    "iucr", "fbi", "location_desc", "district", "ward", "community", "beat",
]


def _to_int(value):
    value = value.strip()
    if not value:
        return -1
    try:
        return int(float(value))
    except ValueError:
        return -1


def _parse_row(fields, col):
    """Parse one split source row into a cache record, or None if unusable."""
    lat_s, lng_s = fields[col["LATITUDE"]].strip(), fields[col["LONGITUDE"]].strip()
    if not lat_s or not lng_s:
        return None, "missing_coords"
    try:
        lat, lng = float(lat_s), float(lng_s)
    except ValueError:
        return None, "missing_coords"
    if not (LAT_MIN <= lat <= LAT_MAX and LNG_MIN <= lng <= LNG_MAX):
        return None, "out_of_bounds"
    ts = _to_int(fields[col["TIMESTAMP"]])
    if ts < 0:
        return None, "missing_timestamp"
    rec = {
        "id": fields[col["ID"]].strip(),
        "lat": lat,
        "lng": lng,
        "timestamp": ts,
        "year": _to_int(fields[col["YEAR"]]),
        "arrest": 1 if fields[col["ARREST"]].strip().lower() == "true" else 0,
        "domestic": 1 if fields[col["DOMESTIC"]].strip().lower() == "true" else 0,
        "primary": fields[col["PRIMARY"]].strip(),
        "iucr": fields[col["IUCR"]].strip(),
        "fbi": fields[col["FBI"]].strip(),
        "location_desc": fields[col["LOCATION_DESC"]].strip(),
        "district": _to_int(fields[col["DISTRICT"]]),
        "ward": _to_int(fields[col["WARD"]]),
        "community": _to_int(fields[col["COMMUNITY"]]),
        "beat": _to_int(fields[col["BEAT"]]),
    }
    return rec, "ok"


def _split_line(raw, encoding="utf-8"):
    line = raw.decode(encoding, errors="replace").rstrip("\r\n")
    fields = line.split(";")
    if len(fields) != len(SOURCE_COLUMNS):
        # Fall back to a real CSV parser for quoted fields containing ';'.
        fields = next(csv.reader([line], delimiter=";"))
    return fields


def reservoir_lines(path, k, seed):
    """Uniform sample of k data lines (every line after the header), one pass.

    Vitter's Algorithm L: the gaps between replacements are drawn directly,
    and skipped lines are consumed by ``itertools.islice`` without parsing.
    """
    rng = random.Random(seed)
    with open(path, "rb") as f:
        f.readline()  # header
        reservoir = list(itertools.islice(f, k))
        if len(reservoir) < k:
            return reservoir
        w = math.exp(math.log(rng.random()) / k)
        while True:
            skip = int(math.floor(math.log(rng.random()) / math.log(1.0 - w)))
            nxt = next(itertools.islice(f, skip, skip + 1), None)  # skip, then take one
            if nxt is None:
                return reservoir
            reservoir[rng.randrange(k)] = nxt
            w *= math.exp(math.log(rng.random()) / k)


def count_lines(path, chunk=1 << 24):
    n = 0
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            n += buf.count(b"\n")
    return n


def cache_paths(k, seed):
    stem = os.path.join(CACHE_DIR, f"sample_k{k}_s{seed}")
    return stem + ".csv.gz", stem + ".meta.json"


def build_sample(csv_path=DEFAULT_CSV, k=400_000, seed=12345, force=False, verbose=True):
    """Return (records, meta). Builds and caches the reservoir sample if needed."""
    data_path, meta_path = cache_paths(k, seed)
    if not force and os.path.exists(data_path) and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if os.path.abspath(meta.get("source", "")) == os.path.abspath(csv_path) or not os.path.exists(csv_path):
            return load_cached(data_path), meta

    os.makedirs(CACHE_DIR, exist_ok=True)
    t0 = time.time()
    if verbose:
        print(f"[data] streaming {csv_path} (reservoir k={k}, seed={seed}) ...", flush=True)
    with open(csv_path, "rb") as f:
        header = f.readline().decode("utf-8").strip().split(";")
    col = {h.split(":")[0].strip().upper(): i for i, h in enumerate(header)}
    missing = [c for c in SOURCE_COLUMNS if c not in col]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    lines = reservoir_lines(csv_path, k, seed)
    total = count_lines(csv_path) - 1  # minus header
    t1 = time.time()

    records, reasons = [], {"ok": 0, "missing_coords": 0, "out_of_bounds": 0, "missing_timestamp": 0}
    for raw in lines:
        rec, why = _parse_row(_split_line(raw), col)
        reasons[why] += 1
        if rec is not None:
            records.append(rec)
    random.Random(seed + 1).shuffle(records)  # reservoir slot order is not random

    lat0 = sum(r["lat"] for r in records) / len(records)
    lng0 = sum(r["lng"] for r in records) / len(records)
    meta = {
        "source": os.path.abspath(csv_path),
        "source_data_rows": total,
        "reservoir_size": k,
        "seed": seed,
        "kept_rows": len(records),
        "drop_reasons": reasons,
        "bounds": {"lat": [LAT_MIN, LAT_MAX], "lng": [LNG_MIN, LNG_MAX]},
        "ref_lat": lat0,
        "ref_lng": lng0,
        "years": [min(r["year"] for r in records), max(r["year"] for r in records)],
        "build_seconds": round(time.time() - t0, 1),
    }
    with gzip.open(data_path, "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CACHE_FIELDS)
        for r in records:
            w.writerow([r[c] for c in CACHE_FIELDS])
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    if verbose:
        print(f"[data] sampled {len(lines)} of {total} rows in {t1 - t0:.0f}s; kept {len(records)} "
              f"({reasons}); cached to {data_path}", flush=True)
    return records, meta


_INT_FIELDS = {"timestamp", "year", "arrest", "domestic", "district", "ward", "community", "beat"}
_FLOAT_FIELDS = {"lat", "lng"}


def load_cached(data_path):
    out = []
    with gzip.open(data_path, "rt", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        conv = []
        for h in header:
            if h in _INT_FIELDS:
                conv.append(int)
            elif h in _FLOAT_FIELDS:
                conv.append(float)
            else:
                conv.append(str)
        for row in reader:
            out.append({h: c(v) for h, c, v in zip(header, conv, row)})
    return out


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

class LocalProjection:
    """Equirectangular projection to km around (ref_lat, ref_lng).

    Over Chicago's ~45 km extent the scale error of this projection is below
    0.1%, far smaller than the geocoding precision (block level)."""

    def __init__(self, ref_lat, ref_lng):
        self.ref_lat = ref_lat
        self.ref_lng = ref_lng
        self.kx = EARTH_RADIUS_KM * math.cos(math.radians(ref_lat)) * math.pi / 180.0
        self.ky = EARTH_RADIUS_KM * math.pi / 180.0

    def to_km(self, lat, lng):
        return (lng - self.ref_lng) * self.kx, (lat - self.ref_lat) * self.ky

    def to_latlng(self, x, y):
        return self.ref_lat + y / self.ky, self.ref_lng + x / self.kx


def snap(v):
    """Snap a km coordinate to the 1 m grid (also the duplicate key)."""
    return round(v / SNAP_KM) * SNAP_KM


def draw_points(records, proj, n, seed, dedupe=True):
    """Draw up to n incidents for a seed.

    Records are visited in a seed-specific random order (so inclusion is
    record-weighted, like sampling incidents); each kept incident's location is
    snapped to a 1 m grid. With ``dedupe`` an incident whose snapped location
    is already taken is skipped, so every point is a distinct location.

    Why dedupe: crimes are geocoded to the block, so many incidents share exact
    coordinates. ``db_consents.Oracle`` caches answers by coordinate tuple, so
    duplicates would (a) collapse into one consent request and (b) return the
    FIRST queried record's consent for every record at that location, which is
    wrong whenever co-located records disagree. One record per location avoids
    both: one point = one record = one consent request.

    Returns (rows, coords) with coords[i] = [x_km, y_km] for rows[i].
    """
    rng = random.Random(seed)
    order = list(range(len(records)))
    rng.shuffle(order)
    rows, coords, seen = [], [], set()
    for idx in order:
        r = records[idx]
        x, y = proj.to_km(r["lat"], r["lng"])
        p = (snap(x), snap(y))
        if dedupe:
            if p in seen:
                continue
            seen.add(p)
        rows.append(r)
        coords.append([p[0], p[1]])
        if len(rows) >= n:
            break
    return rows, coords


VALID_DISTRICTS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 22, 24, 25}
DISTRICT_MAX_KM = 15.0


def clean_records(records, proj):
    """Drop records whose police district is invalid or contradicts the geocode.

    A handful of rows carry a district code that is not a CPD district (e.g. 31)
    or are geocoded > 15 km from the median location of their district (e.g. a
    district-8 / Clearing record placed in the far north). Either the district
    or the coordinates are wrong; a single such row can define the MEB of a
    district-based consent set, so they are removed. Returns (kept, stats)."""
    by_d = {}
    for r in records:
        by_d.setdefault(r["district"], []).append(r)
    med = {}
    for d, rs in by_d.items():
        lats = sorted(r["lat"] for r in rs)
        lngs = sorted(r["lng"] for r in rs)
        med[d] = proj.to_km(lats[len(lats) // 2], lngs[len(lngs) // 2])
    kept, invalid, far = [], 0, 0
    for r in records:
        d = r["district"]
        if d not in VALID_DISTRICTS:
            invalid += 1
            continue
        x, y = proj.to_km(r["lat"], r["lng"])
        mx, my = med[d]
        if math.hypot(x - mx, y - my) > DISTRICT_MAX_KM:
            far += 1
            continue
        kept.append(r)
    return kept, {"invalid_district": invalid, "far_from_district": far}


def load(csv_path=DEFAULT_CSV, k=400_000, seed=12345, force=False, verbose=True):
    """Convenience: (records, meta, projection), with clean_records applied."""
    records, meta = build_sample(csv_path, k=k, seed=seed, force=force, verbose=verbose)
    proj = LocalProjection(meta["ref_lat"], meta["ref_lng"])
    records, stats = clean_records(records, proj)
    meta = dict(meta, cleaning=stats, clean_rows=len(records))
    return records, meta, proj


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build/inspect the cached Chicago crimes sample.")
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--k", type=int, default=400_000)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    recs, meta, proj = load(a.csv, k=a.k, force=a.force)
    print(json.dumps(meta, indent=2))
    seen = set()
    for r in recs:
        x, y = proj.to_km(r["lat"], r["lng"])
        seen.add((snap(x), snap(y)))
    print(f"unique 1 m locations in sample: {len(seen)} of {len(recs)} rows")
