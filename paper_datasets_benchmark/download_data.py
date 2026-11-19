"""
Download the public copies of the datasets used in
Amsterdamer et al., "Selecting Sub-tables for Data Exploration" (ICDE 2023).

Every file goes to ``data/<dataset>/`` next to this script (gitignored).
Downloaded files are treated as untrusted data: they are only ever parsed
as CSV/TSV text (zip archives are read member-by-member in memory, never
extracted to disk), nothing in them is executed.

Usage:
    python -I download_data.py                 # all datasets
    python -I download_data.py creditcard spotify
    python -I download_data.py --list
"""

import argparse
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

_GH = "https://raw.githubusercontent.com"

# dataset -> list of (url, local filename, approx size in MB)
SOURCES = {
    # Paper: Credit card frauds (CC) [1], Kaggle mlg-ulb/creditcardfraud.
    # TensorFlow's public mirror of the identical Kaggle file.
    "creditcard": [
        ("https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv",
         "creditcard.csv", 151),
    ],
    # Paper: Cyber-security (CY) [3], Honeynet challenges, 12 columns, used in
    # the REACT-IDA session study [46].  The same four packet tables are
    # published by the authors' group (TAU-DB) in the ATENA benchmark repo.
    "cyber": [
        (f"{_GH}/TAU-DB/ATENA-A-EDA/master/benchmark/atena/datasets/cyber/{i}.tsv",
         f"{i}.tsv", s) for i, s in ((1, 1.5), (2, 0.06), (3, 0.12), (4, 2.6))
    ],
    # Paper: Spotify (SP) [11], Kaggle competition bfh-spotify-challenge.
    # The authors' group ships the same competition data in pd-explain.
    "spotify": [
        (f"{_GH}/analysis-bots/pd-explain/main/Examples/Datasets/spotify_all.csv",
         "spotify_all.csv", 35),
    ],
    # Paper: Bank Loans (BL) [6], Kaggle panamby/bank-loan-status-dataset.
    # Public GitHub mirror of the two Kaggle files (train + test).
    "bankloans": [
        (f"{_GH}/NgTrDung/Bank-Loan-Status-Dataset/main/data/credit_train.csv",
         "credit_train.csv", 19),
        (f"{_GH}/NgTrDung/Bank-Loan-Status-Dataset/main/data/credit_test.csv",
         "credit_test.csv", 2),
    ],
    # Paper: US Funds (USF) [5], Kaggle stefanoleone992/mutual-funds-and-etfs
    # (MutualFunds.csv).  Public GitHub mirror of the Kaggle file.
    "funds": [
        (f"{_GH}/Pranavsharma13/-Fundwise---Deciphering-Mutual-Fund-Performance-Behavior-/main/MutualFunds.csv",
         "MutualFunds.csv", 77),
    ],
    # Paper: Flights (FL) [2], Kaggle usdot/flight-delays (2015 BTS on-time
    # data).  Substitute: the same BTS table straight from TranStats (one
    # month, January 2015) + OpenFlights airport coordinates (the Kaggle
    # bundle's airports.csv is not separately mirrored).
    "flights": [
        ("https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2015_1.zip",
         "ontime_2015_01.zip", 23),
        (f"{_GH}/jpatokal/openflights/master/data/airports.dat",
         "airports.dat", 1.1),
    ],
}


def download(url, dest):
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "meb-consent-benchmark/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", help="subset of: " + ", ".join(SOURCES))
    ap.add_argument("--list", action="store_true", help="print sources and exit")
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args()

    if args.list:
        for name, files in SOURCES.items():
            for url, fn, mb in files:
                print(f"{name:11s} {fn:28s} ~{mb:>6.1f} MB  {url}")
        return

    names = args.datasets or list(SOURCES)
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        sys.exit(f"unknown dataset(s): {unknown}")

    for name in names:
        d = os.path.join(DATA_DIR, name)
        os.makedirs(d, exist_ok=True)
        for url, fn, mb in SOURCES[name]:
            dest = os.path.join(d, fn)
            if os.path.exists(dest) and not args.force:
                print(f"[skip] {name}/{fn} exists")
                continue
            print(f"[get ] {name}/{fn} (~{mb} MB) <- {url}")
            download(url, dest)
            print(f"       {os.path.getsize(dest) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
