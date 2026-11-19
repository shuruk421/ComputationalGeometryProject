"""
Loaders for the (public copies of the) datasets used in
Amsterdamer et al., "Selecting Sub-tables for Data Exploration" (ICDE 2023).

Each loader turns one raw table into a ``Dataset``: an (N, 2) array of
points in *algorithm units* (O(1) magnitudes, see README "Numerical scale"),
plus a set of candidate consent attributes, each of which can be
thresholded to sweep the consenting fraction.

Rows whose coordinates repeat exactly are dropped (first one kept) because
``db_consents.Oracle`` caches consent answers by coordinate tuple: two rows
at the same point would otherwise share one consent answer.

All files are parsed as plain CSV/TSV text; nothing in them is executed.
"""

import csv
import io
import math
import os
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
EARTH_RADIUS_KM = 6371.0088

csv.field_size_limit(1 << 26)


@dataclass
class Candidate:
    """A numeric attribute that can act as consent: consent <=> sign*value >= t."""
    key: str
    label: str            # human label, e.g. "Amount (EUR)"
    values: np.ndarray    # float, NaN = missing (never consents)
    sign: int             # +1: higher values consent, -1: lower values consent
    story: str            # why would this record's owner agree?
    binary: bool = False  # 0/1 attribute: only one consenting fraction is possible

    def rule(self, t):
        op = ">=" if self.sign > 0 else "<="
        return f"{self.label} {op} {fmt_num(t)}"


@dataclass
class Dataset:
    key: str
    code: str                       # the paper's two-letter code
    title: str
    source: str                     # where the copy comes from + fidelity to the paper
    unit_note: str                  # what the coordinates mean
    X: np.ndarray                   # (N, 2) points in algorithm units
    axis_labels: Tuple[str, str]
    candidates: Dict[str, Candidate]
    chosen: str                     # key of the chosen consent candidate
    n_raw: int                      # rows read from the file(s)
    n_dropped_missing: int
    n_dropped_duplicate: int
    geo: Optional[dict] = None      # for map plots: lat, lon, lat0, lon0, km_per_unit
    names: Optional[np.ndarray] = None
    notes: List[str] = field(default_factory=list)


def fmt_num(v):
    a = abs(v)
    if a == 0:
        return "0"
    if a >= 1e5:
        return f"{v:,.0f}"
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.3g}"


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _f(s):
    """Parse a float; empty / NA / unparsable -> NaN."""
    try:
        v = float(s)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _zscore(a):
    a = np.asarray(a, dtype=float)
    return (a - a.mean()) / a.std()


def _finalize(key, code, title, source, unit_note, X, axis_labels, cands, chosen,
              n_raw, extra_keep=None, geo=None, names=None, notes=None):
    """Drop rows with missing coordinates / chosen consent value, then exact
    coordinate duplicates (first kept). ``extra_keep`` is an optional boolean
    mask of rows to keep. Coordinates are rounded to 12 significant digits so
    that duplicates are detected robustly."""
    X = np.asarray(X, dtype=float)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(cands[chosen].values)
    if extra_keep is not None:
        ok &= extra_keep
    n_missing = int((~ok).sum())
    idx = np.flatnonzero(ok)
    seen = set()
    keep = []
    for i in idx:
        t = (float(f"{X[i, 0]:.12g}"), float(f"{X[i, 1]:.12g}"))
        if t in seen:
            continue
        seen.add(t)
        keep.append(i)
    keep = np.array(keep, dtype=int)
    n_dup = len(idx) - len(keep)
    for c in cands.values():
        c.values = np.asarray(c.values, dtype=float)[keep]
    if geo is not None:
        geo = dict(geo)
        geo["lat"] = np.asarray(geo["lat"])[keep]
        geo["lon"] = np.asarray(geo["lon"])[keep]
    if names is not None:
        names = np.asarray(names)[keep]
    return Dataset(key=key, code=code, title=title, source=source, unit_note=unit_note,
                   X=X[keep], axis_labels=axis_labels, candidates=cands, chosen=chosen,
                   n_raw=n_raw, n_dropped_missing=n_missing, n_dropped_duplicate=n_dup,
                   geo=geo, names=names, notes=notes or [])


def _cache_path(key):
    return os.path.join(DATA_DIR, key, "_parsed_cache.npz")


def _load_columns(key, path_fn, columns, cache=True):
    """Read selected columns (as float, or str for names prefixed 's:') with
    an npz cache (no pickle) next to the raw data."""
    cp = _cache_path(key)
    if cache and os.path.exists(cp):
        z = np.load(cp, allow_pickle=False)
        if all(c.split(":", 1)[-1] in z.files for c in columns):
            return {c.split(":", 1)[-1]: z[c.split(":", 1)[-1]] for c in columns}
    out = path_fn(columns)
    if cache:
        np.savez_compressed(cp, **out)
    return out


def _read_csv_columns(path, columns, delimiter=","):
    want = [(c.split(":", 1)[-1], c.startswith("s:")) for c in columns]
    vals = {name: [] for name, _ in want}
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        r = csv.reader(f, delimiter=delimiter)
        header = next(r)
        pos = {h: i for i, h in enumerate(header)}
        missing = [name for name, _ in want if name not in pos]
        if missing:
            raise KeyError(f"{path}: missing columns {missing}")
        for row in r:
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            for name, is_str in want:
                v = row[pos[name]]
                vals[name].append(v if is_str else _f(v))
    return {name: (np.array(v, dtype=str) if is_str else np.array(v, dtype=float))
            for (name, is_str), v in zip(want, vals.values())}


# ----------------------------------------------------------------------------
# CC: credit card frauds
# ----------------------------------------------------------------------------

def load_creditcard():
    path = os.path.join(DATA_DIR, "creditcard", "creditcard.csv")
    cols = _load_columns("creditcard", lambda c: _read_csv_columns(path, c),
                         ["Time", "V1", "V2", "Amount", "Class"])
    n = len(cols["V1"])
    X = np.column_stack([_zscore(cols["V1"]), _zscore(cols["V2"])])
    cands = {
        "amount": Candidate(
            "amount", "Amount (EUR)", cols["Amount"], -1,
            "Cardholders agree to share low-value transactions but withhold consent "
            "for large purchases, which are more sensitive and identifying."),
        "time": Candidate(
            "time", "Time (s since first txn)", cols["Time"], +1,
            "Consent was collected by an opt-in campaign during the 48-hour window and "
            "is not retroactive: only transactions after the opt-in time t are covered."),
        "legit": Candidate(
            "legit", "Class = legitimate", 1.0 - cols["Class"], +1,
            "Transactions under fraud investigation (Class=1) cannot be released.",
            binary=True),
    }
    return _finalize(
        "creditcard", "CC", "Credit card frauds",
        "Kaggle mlg-ulb/creditcardfraud (paper [1]); identical file from TensorFlow's "
        "public mirror. 284,807 rows x 31 columns (paper: '250K x 31').",
        "V1, V2 = the two leading PCA components of the (anonymised) transaction "
        "features, each z-scored.",
        X, ("V1 (z-score)", "V2 (z-score)"), cands, "amount", n)


# ----------------------------------------------------------------------------
# SP: Spotify
# ----------------------------------------------------------------------------

def load_spotify():
    path = os.path.join(DATA_DIR, "spotify", "spotify_all.csv")
    cols = _load_columns("spotify", lambda c: _read_csv_columns(path, c),
                         ["valence", "energy", "popularity", "year", "explicit",
                          "duration_minutes"])
    n = len(cols["valence"])
    X = np.column_stack([_zscore(cols["valence"]), _zscore(cols["energy"])])
    cands = {
        # popularity is an integer with heavy ties (a quarter of the tracks have
        # popularity <= 2), so no threshold gives 80-99% consent. Ties are broken
        # by release year (newer first): value = popularity + (year - 1900) / 1000.
        "popularity": Candidate(
            "popularity", "popularity (+ year tie-break)",
            cols["popularity"] + (cols["year"] - 1900.0) / 1000.0, +1,
            "Rights holders of popular tracks opt in to sharing the track's audio "
            "analysis for its promotional value; owners of obscure catalogue tracks "
            "never answer the request (among equally popular tracks, newer releases, "
            "whose rights holders are still reachable, opt in first)."),
        "year": Candidate(
            "year", "release year", cols["year"], +1,
            "Only recordings released after year t are covered by the streaming-era "
            "licence that includes a data-sharing clause."),
        "duration": Candidate(
            "duration", "duration (min)", cols["duration_minutes"], -1,
            "Short-form releases (singles) are licensed under a standard agreement "
            "that permits sharing; long works are individually negotiated."),
        "clean": Candidate(
            "clean", "explicit = 0", 1.0 - cols["explicit"], +1,
            "Labels refuse to share metadata of explicit-content tracks.", binary=True),
    }
    return _finalize(
        "spotify", "SP", "Spotify tracks",
        "Paper [11] is Kaggle competition bfh-spotify-challenge (42K x 15). The "
        "authors' group ships the same competition data as spotify_all.csv in "
        "pd-explain (174,389 x 21); used here as the closest public version.",
        "valence (musical positiveness) and energy, each z-scored: the "
        "valence-arousal 'mood plane' of the track.",
        X, ("valence (z-score)", "energy (z-score)"), cands, "popularity", n)


# ----------------------------------------------------------------------------
# BL: bank loans
# ----------------------------------------------------------------------------

def load_bankloans():
    d = os.path.join(DATA_DIR, "bankloans")
    num = ["Current Loan Amount", "Credit Score", "Annual Income", "Monthly Debt",
           "Years of Credit History", "Number of Open Accounts"]

    def read(columns):
        parts = [_read_csv_columns(os.path.join(d, fn), ["s:Loan ID"] + num)
                 for fn in ("credit_train.csv", "credit_test.csv")]
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    cols = _load_columns("bankloans", read, ["s:Loan ID"] + num)
    n = len(cols["Loan ID"])
    # Kaggle data quirks: blank padding rows, repeated Loan IDs, 99999999 as a
    # "missing" loan amount, and some credit scores multiplied by 10.
    loan_id = cols["Loan ID"]
    first = np.zeros(n, dtype=bool)
    seen = set()
    for i, lid in enumerate(loan_id):
        if lid and lid not in seen:
            seen.add(lid)
            first[i] = True
    amount = cols["Current Loan Amount"].copy()
    amount[amount >= 99999999] = np.nan
    score = cols["Credit Score"].copy()
    score = np.where(score > 850, score / 10.0, score)
    income = cols["Annual Income"]
    debt = cols["Monthly Debt"]
    with np.errstate(divide="ignore", invalid="ignore"):
        dti = 100.0 * 12.0 * debt / income
        log_income = np.log10(income)
    ok = first & (income > 0) & (dti >= 0) & (dti < 100)
    X = np.column_stack([log_income, dti])
    Xz = np.full_like(X, np.nan)
    Xz[ok, 0] = _zscore(X[ok, 0])
    Xz[ok, 1] = _zscore(X[ok, 1])
    cands = {
        "credit_score": Candidate(
            "credit_score", "credit score", score, +1,
            "Borrowers above a credit-score cut-off were invited to the bank's "
            "open-data programme (and accepted); the others were never asked."),
        "history": Candidate(
            "history", "years of credit history", cols["Years of Credit History"], +1,
            "Long-standing customers signed the newer account terms that include a "
            "data-sharing clause."),
        "loan_amount": Candidate(
            "loan_amount", "loan amount", amount, -1,
            "Small loans are processed under a standard digital contract with a "
            "consent checkbox; large loans are negotiated individually and carry no "
            "data-sharing clause."),
    }
    return _finalize(
        "bankloans", "BL", "Bank loans",
        "Kaggle panamby/bank-loan-status-dataset (paper [6]); public GitHub mirror of "
        "credit_train.csv + credit_test.csv = 110,867 rows x 19 columns (paper: 110K x 19).",
        "x = log10(annual income), y = debt-to-income ratio (% = 12 * monthly debt / "
        "income), each z-scored. Rows with a repeated Loan ID, missing income/debt, "
        "DTI >= 100% or the 99,999,999 'missing' loan-amount sentinel are dropped.",
        Xz, ("log10 annual income (z-score)", "debt-to-income % (z-score)"), cands,
        "loan_amount", n, extra_keep=ok)


# ----------------------------------------------------------------------------
# USF: US mutual funds
# ----------------------------------------------------------------------------

def load_funds():
    path = os.path.join(DATA_DIR, "funds", "MutualFunds.csv")
    names = ["fund_stdev_3years", "fund_return_3years", "total_net_assets",
             "fund_annual_report_net_expense_ratio", "s:inception_date",
             "morningstar_overall_rating", "s:fund_symbol"]
    cols = _load_columns("funds", lambda c: _read_csv_columns(path, c), names)
    n = len(cols["fund_symbol"])
    age = np.full(n, np.nan)
    for i, s in enumerate(cols["inception_date"]):
        if len(s) >= 4 and s[:4].isdigit():
            y = int(s[:4]) + (int(s[5:7]) - 1) / 12.0 if s[5:7].isdigit() else int(s[:4])
            age[i] = 2021.25 - y
    tna = cols["total_net_assets"].copy()
    tna[tna <= 0] = np.nan
    X = np.column_stack([cols["fund_stdev_3years"], 100.0 * cols["fund_return_3years"]])
    ok = np.isfinite(X).all(axis=1)
    Xz = np.full_like(X, np.nan)
    Xz[ok, 0] = _zscore(X[ok, 0])
    Xz[ok, 1] = _zscore(X[ok, 1])
    cands = {
        "assets": Candidate(
            "assets", "total net assets (USD)", tna, +1,
            "Large funds have compliance teams that sign the data-sharing agreement; "
            "small funds do not respond."),
        "expense": Candidate(
            "expense", "net expense ratio", cols["fund_annual_report_net_expense_ratio"], -1,
            "Low-cost (index) funds publish everything by policy; high-fee active "
            "funds treat their data as proprietary."),
        "age": Candidate(
            "age", "fund age (years)", age, +1,
            "Funds launched before the regulator's opt-out window closed are "
            "grandfathered into disclosure."),
        "rating": Candidate(
            "rating", "Morningstar rating (1-5)", cols["morningstar_overall_rating"], +1,
            "Highly rated funds are happy to be benchmarked."),
    }
    return _finalize(
        "funds", "USF", "US mutual funds",
        "Kaggle stefanoleone992/mutual-funds-and-etfs, MutualFunds.csv (paper [5]); "
        "public GitHub mirror, 23,783 rows x 298 columns (paper: 23.5K x 298).",
        "x = 3-year volatility (stdev, %), y = 3-year annualised return (%), each "
        "z-scored: the classic risk-return plane.",
        Xz, ("3y volatility (z-score)", "3y annualised return (z-score)"), cands,
        "assets", n, names=cols["fund_symbol"])


# ----------------------------------------------------------------------------
# FL: flights (geographic; one point per origin airport)
# ----------------------------------------------------------------------------

NON_CONUS = {"AK", "HI", "PR", "VI", "TT", "GU", "AS", "MP"}


def _read_flights(columns):
    d = os.path.join(DATA_DIR, "flights")
    z = zipfile.ZipFile(os.path.join(d, "ontime_2015_01.zip"))
    member = [i for i in z.infolist() if i.filename.lower().endswith(".csv")][0]
    agg = {}
    with z.open(member) as fb:
        r = csv.DictReader(io.TextIOWrapper(fb, encoding="latin-1", newline=""))
        for row in r:
            a = row["Origin"]
            s = agg.setdefault(a, {"state": row["OriginState"], "n": 0, "canc": 0.0,
                                   "dep": 0, "del15": 0.0, "taxi": 0.0, "ntaxi": 0})
            s["n"] += 1
            s["canc"] += _f(row["Cancelled"]) or 0.0
            d15 = _f(row["DepDel15"])
            if not math.isnan(d15):
                s["dep"] += 1
                s["del15"] += d15
            t = _f(row["TaxiOut"])
            if not math.isnan(t):
                s["taxi"] += t
                s["ntaxi"] += 1
    coords = {}
    with open(os.path.join(d, "airports.dat"), encoding="utf-8", errors="replace",
              newline="") as f:
        for row in csv.reader(f):
            if len(row) > 7 and len(row[4]) == 3:
                coords[row[4]] = (_f(row[6]), _f(row[7]))
    codes = sorted(agg)
    out = {"code": np.array(codes, dtype=str),
           "state": np.array([agg[c]["state"] for c in codes], dtype=str),
           "lat": np.array([coords.get(c, (np.nan, np.nan))[0] for c in codes]),
           "lon": np.array([coords.get(c, (np.nan, np.nan))[1] for c in codes]),
           "flights": np.array([agg[c]["n"] for c in codes], dtype=float),
           "cancel_rate": np.array([agg[c]["canc"] / agg[c]["n"] for c in codes]),
           "ontime_rate": np.array([1 - agg[c]["del15"] / agg[c]["dep"] if agg[c]["dep"]
                                    else np.nan for c in codes]),
           "taxi_out": np.array([agg[c]["taxi"] / agg[c]["ntaxi"] if agg[c]["ntaxi"]
                                 else np.nan for c in codes])}
    return out


def project_equirect(lat, lon, lat0, lon0):
    """Equirectangular projection (km) around (lat0, lon0)."""
    x = EARTH_RADIUS_KM * np.radians(lon - lon0) * math.cos(math.radians(lat0))
    y = EARTH_RADIUS_KM * np.radians(lat - lat0)
    return x, y


def load_flights(km_per_unit=1000.0):
    names = ["s:code", "s:state", "lat", "lon", "flights", "cancel_rate", "ontime_rate",
             "taxi_out"]
    cols = _load_columns("flights", _read_flights, names)
    n = len(cols["code"])
    conus = np.array([s not in NON_CONUS for s in cols["state"]])
    has = np.isfinite(cols["lat"]) & np.isfinite(cols["lon"])
    keep = conus & has
    lat0 = float(np.mean(cols["lat"][keep]))
    lon0 = float(np.mean(cols["lon"][keep]))
    xk, yk = project_equirect(cols["lat"], cols["lon"], lat0, lon0)
    X = np.column_stack([xk, yk]) / km_per_unit
    cands = {
        "ontime": Candidate(
            "ontime", "on-time departure rate", cols["ontime_rate"], +1,
            "Airport operators whose January on-time departure rate is at least t "
            "agree to publish per-flight data; poor performers decline (reputational risk)."),
        "volume": Candidate(
            "volume", "departures in Jan 2015", cols["flights"], +1,
            "Hub airports have data offices that sign the sharing agreement; small "
            "regional airports never reply."),
        "cancel": Candidate(
            "cancel", "cancellation rate", cols["cancel_rate"], -1,
            "Airport operators whose January cancellation rate is at most t agree to "
            "publish per-flight data; airports hit hard by the late-January 2015 "
            "Northeast blizzard decline (reputational risk)."),
        "taxi": Candidate(
            "taxi", "mean taxi-out (min)", cols["taxi_out"], -1,
            "Uncongested airports agree; congested ones fear the comparison."),
    }
    return _finalize(
        "flights", "FL", "US flights (airports)",
        "SUBSTITUTE. Paper [2] is Kaggle usdot/flight-delays (2015, 5.8M flights x 31). "
        "Used here: the same BTS On-Time table for January 2015 from TranStats "
        "(469,968 flights), aggregated to one point per origin airport, with OpenFlights "
        "coordinates; contiguous US only.",
        f"Equirectangular projection around the airport centroid (lat {lat0:.2f}, lon "
        f"{lon0:.2f}); algorithm units = {km_per_unit:.0f} km.",
        X, (f"east-west ({km_per_unit:.0f} km)", f"north-south ({km_per_unit:.0f} km)"),
        cands, "cancel", n, extra_keep=keep,
        geo={"lat": cols["lat"], "lon": cols["lon"], "lat0": lat0, "lon0": lon0,
             "km_per_unit": km_per_unit},
        names=cols["code"],
        notes=["Alaska, Hawaii and the territories are excluded: an equirectangular "
               "projection is meaningless at trans-Pacific scale."])


# ----------------------------------------------------------------------------
# CY: cyber-security (Honeynet packet captures, REACT-IDA versions)
# ----------------------------------------------------------------------------

def _read_cyber(columns):
    import datetime
    d = os.path.join(DATA_DIR, "cyber")
    flows = {}
    for cap in (1, 2, 3, 4):
        with open(os.path.join(d, f"{cap}.tsv"), encoding="utf-8", errors="replace",
                  newline="") as f:
            r = csv.DictReader(f, delimiter="\t")
            t0 = None
            for row in r:
                ts = datetime.datetime.strptime(row["sniff_timestamp"], "%Y-%m-%d %H:%M:%S")
                t0 = ts if t0 is None else t0
                st = row["tcp_stream"]
                if not st:
                    continue
                k = (cap, int(float(st)))
                fl = flows.get(k)
                if fl is None:
                    flows[k] = fl = {"cap": cap, "sport": _f(row["tcp_srcport"]),
                                     "dport": _f(row["tcp_dstport"]),
                                     "t": (ts - t0).total_seconds(),
                                     "first": int(float(row["number"])),
                                     "pk": 0, "bytes": 0.0}
                fl["pk"] += 1
                fl["bytes"] += _f(row["length"])
    keys = sorted(flows)
    caps = np.array([flows[k]["cap"] for k in keys], dtype=float)
    # time rank within the capture (0..1), by first packet number
    rank = np.zeros(len(keys))
    for cap in (1, 2, 3, 4):
        idx = [i for i, k in enumerate(keys) if k[0] == cap]
        order = np.argsort([flows[keys[i]]["first"] for i in idx], kind="stable")
        for r_, j in enumerate(order):
            rank[idx[j]] = (r_ + 0.5) / len(idx)
    return {"cap": caps,
            "sport": np.array([flows[k]["sport"] for k in keys]),
            "dport": np.array([flows[k]["dport"] for k in keys]),
            "trank": rank,
            "packets": np.array([flows[k]["pk"] for k in keys], dtype=float),
            "bytes": np.array([flows[k]["bytes"] for k in keys])}


def load_cyber():
    names = ["cap", "sport", "dport", "trank", "packets", "bytes"]
    cols = _load_columns("cyber", _read_cyber, names)
    n = len(cols["cap"])
    X = np.column_stack([cols["sport"], cols["dport"]]) / 65535.0
    cands = {
        "time": Candidate(
            "time", "connection start (fraction of capture)", cols["trank"], +1,
            "Users accepted the monitoring banner at time t; consent is not "
            "retroactive, so only connections opened afterwards may be analysed."),
        "bytes": Candidate(
            "bytes", "bytes in connection", cols["bytes"], -1,
            "Short control connections carry no payload and are covered by the "
            "acceptable-use policy; payload-carrying connections need explicit consent."),
    }
    return _finalize(
        "cyber", "CY", "Cyber-security packets (TCP connections)",
        "Paper [3]: Honeynet challenge captures as used in REACT-IDA [46]. Used here: the "
        "four REACT packet tables published by the authors' group (TAU-DB/ATENA-A-EDA), "
        "23,366 packets x 17 columns (paper: 30K x 15), aggregated to 7,672 TCP connections.",
        "x = client (source) port / 65535, y = server (destination) port / 65535 of the "
        "connection's first packet: the 'port plane' used in network-scan analysis.",
        X, ("source port / 65535", "destination port / 65535"), cands, "time", n)


LOADERS = {
    "creditcard": load_creditcard,
    "spotify": load_spotify,
    "bankloans": load_bankloans,
    "funds": load_funds,
    "flights": load_flights,
    "cyber": load_cyber,
}


def load(key):
    return LOADERS[key]()
