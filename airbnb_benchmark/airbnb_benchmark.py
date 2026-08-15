"""
Airbnb Minimum Enclosing Ball Benchmark

Computes the minimum enclosing ball (MEB) of "consenting" Airbnb listings
in 3D space: (latitude, longitude, normalized_price).

Scenario: A tourism regulation authority wants to determine the geographic and
price footprint of Airbnb hosts who comply with local regulations. Querying
each host for compliance (consent) is expensive, so the goal is to compute the
MEB while minimizing the number of consent queries (oracle calls).

Consent is simulated based on the number_of_reviews column: listings with at
least a threshold number of reviews (e.g. >= 10) are deemed to have consented.

Usage:
    python airbnb_benchmark.py --csv listings.csv [options]
"""

import argparse
import csv
import logging
import math
import os
import random
import sys
import time

import numpy as np

# Add parent directory to path to import db_consents
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import db_consents
from db_consents import (
    Oracle,
    incremental_distance_based,
    decremental_distance_based,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_airbnb_data(csv_path):
    """
    Loads the Inside Airbnb summary listings CSV and extracts
    (latitude, longitude, price, number_of_reviews) for each valid listing.
    """
    listings = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row["latitude"])
                lng = float(row["longitude"])
                price = float(row["price"])
                n_reviews = int(row["number_of_reviews"])
                if price > 0:
                    listings.append((lat, lng, price, n_reviews))
            except (ValueError, KeyError):
                continue
    return listings


def prepare_points(listings, price_scale_factor, consent_mode, consent_param):
    """
    Converts raw listing data into 3D points (lat, lng, normalized_price)
    with simulated consent values.

    Args:
        listings: List of (lat, lng, price, n_reviews)
        price_scale_factor: Factor to scale price to be comparable to lat/lng range
        consent_mode: How to assign consent. "review_based" uses static review count threshold,
                      "random" uses a fixed probability.
        consent_param: For "random" mode, the consent probability.
                       For "review_based" mode, the minimum review count for consent.
    """
    # Compute lat/lng ranges
    lats = [l[0] for l in listings]
    lngs = [l[1] for l in listings]
    prices = [l[2] for l in listings]

    lat_range = max(lats) - min(lats)
    lng_range = max(lngs) - min(lngs)
    geo_range = max(lat_range, lng_range)
    price_range = max(prices) - min(prices)

    if price_range == 0:
        price_range = 1.0

    # Scale price so its range is comparable to the geographic range
    # multiplied by a user-controlled factor
    scale = (geo_range / price_range) * price_scale_factor

    points_with_consent = []
    for lat, lng, price, n_reviews in listings:
        point = [lat, lng, price * scale]

        if consent_mode == "review_based":
            # Deterministic threshold: listings with >= consent_param reviews consent completely
            consent = (n_reviews >= consent_param)
        else:
            consent = random.random() < consent_param

        points_with_consent.append((point, consent))

    return points_with_consent, scale


def run_benchmark(csv_path, algo, backend, price_scale, consent_mode, consent_param,
                  n_runs, max_points):
    """Run the Airbnb MEB benchmark."""
    logger.info(f"Loading data from {csv_path}...")
    listings = load_airbnb_data(csv_path)
    logger.info(f"Loaded {len(listings)} valid listings.")

    if max_points and max_points < len(listings):
        logger.info(f"Sampling {max_points} listings from {len(listings)}...")
        listings = random.sample(listings, max_points)

    # Report statistics
    prices = [l[2] for l in listings]
    lats = [l[0] for l in listings]
    lngs = [l[1] for l in listings]
    reviews = [l[3] for l in listings]
    logger.info(f"  Latitude range:  [{min(lats):.4f}, {max(lats):.4f}]")
    logger.info(f"  Longitude range: [{min(lngs):.4f}, {max(lngs):.4f}]")
    logger.info(f"  Price range:     [{min(prices):.0f}, {max(prices):.0f}]")
    logger.info(f"  Reviews range:   [{min(reviews)}, {max(reviews)}]")
    logger.info(f"  Consent mode:    {consent_mode} (param={consent_param})")

    db_consents.USE_CPP_BACKEND = (backend == "cpp")

    algos = ["incremental", "decremental"] if algo == "both" else [algo]

    for algo_name in algos:
        all_times = []
        all_calls = []
        all_radii = []
        all_consenting = []

        for run in range(n_runs):
            random.seed(42 + run)
            np.random.seed(42 + run)

            points_with_consent, price_scale_used = prepare_points(
                listings, price_scale, consent_mode, consent_param
            )

            n_consenting = sum(1 for _, c in points_with_consent if c)
            all_consenting.append(n_consenting)

            oracle = Oracle()

            start = time.perf_counter()
            if algo_name == "incremental":
                center, radius = incremental_distance_based(points_with_consent, oracle)
            else:
                center, radius = decremental_distance_based(points_with_consent, oracle)
            elapsed = time.perf_counter() - start

            all_times.append(elapsed)
            all_calls.append(oracle.get_call_count())
            all_radii.append(radius)

        # Print results
        print("=" * 65)
        print(f"AIRBNB MEB BENCHMARK — {algo_name.upper()}")
        print(f"  Backend:     {backend.upper()}")
        print(f"  Listings:    {len(listings)}")
        print(f"  Dimensions:  3 (lat, lng, scaled_price)")
        print(f"  Runs:        {n_runs}")
        print("-" * 65)
        print(f"  Avg consenting:   {np.mean(all_consenting):.0f} / {len(listings)} "
              f"({100*np.mean(all_consenting)/len(listings):.1f}%)")
        print(f"  Avg oracle calls: {np.mean(all_calls):.1f}")
        print(f"  Avg time:         {np.mean(all_times):.4f}s")
        print(f"  Avg MEB radius:   {np.mean(all_radii):.6f}")
        if n_runs > 1:
            print(f"  Std oracle calls: {np.std(all_calls):.1f}")
            print(f"  Std time:         {np.std(all_times):.4f}s")
        print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="Airbnb Minimum Enclosing Ball Benchmark"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to Inside Airbnb listings.csv (summary/visualisations version)"
    )
    parser.add_argument(
        "--algo",
        choices=["incremental", "decremental", "both"],
        default="both",
        help="Algorithm to benchmark (default: both)"
    )
    parser.add_argument(
        "--backend",
        choices=["python", "cpp"],
        default="python",
        help="Execution backend (default: python)"
    )
    parser.add_argument(
        "--price-scale",
        type=float,
        default=1.0,
        help="Factor to scale the price dimension relative to lat/lng (default: 1.0)"
    )
    parser.add_argument(
        "--consent-mode",
        choices=["random", "review_based"],
        default="review_based",
        help="How to simulate consent (default: review_based)"
    )
    parser.add_argument(
        "--consent-param",
        type=float,
        default=10.0,
        help="For random: consent probability. For review_based: minimum number of reviews required for consent (default: 10.0)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of runs to average (default: 5)"
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Maximum number of listings to use (default: all)"
    )

    args = parser.parse_args()

    run_benchmark(
        csv_path=args.csv,
        algo=args.algo,
        backend=args.backend,
        price_scale=args.price_scale,
        consent_mode=args.consent_mode,
        consent_param=args.consent_param,
        n_runs=args.runs,
        max_points=args.max_points,
    )


if __name__ == "__main__":
    main()
