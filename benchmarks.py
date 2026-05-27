import argparse
import logging
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from db_consents import *

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Create results directory if it doesn't exist
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_box_vs_points(
    n_dim, consent_probability, num_points_list, low=-10, high=10, n_runs=10
):
    """
    Runs box algorithms over a range of point counts, measuring both oracle calls and running times,
    then saves both corresponding plots.
    """
    incremental_calls = []
    decremental_calls = []
    lower_bounds = []
    d_log_n_values = []
    incremental_times = []
    decremental_times = []

    pbar = tqdm(num_points_list, desc="Running box benchmarks vs points")
    for num_points in pbar:
        inc_calls_runs = []
        dec_calls_runs = []
        inc_time_runs = []
        dec_time_runs = []
        lb_runs = []
        for run_idx in range(n_runs):
            pbar.set_description(
                f"Box benchmarks vs points (N={num_points}, run {run_idx + 1}/{n_runs})"
            )
            # Generate random points in a box
            points = generate_in_box(n_dim, low=low, high=high, count=num_points)
            # Add consent values
            points_with_consent = [
                (point, random.random() < consent_probability) for point in points
            ]

            # Run incremental algorithm
            oracle_inc = Oracle()
            start = time.perf_counter()
            min_bounds_inc, max_bounds_inc = incremental_orthogonal(
                points_with_consent, oracle_inc
            )
            inc_time_runs.append(time.perf_counter() - start)
            inc_calls_runs.append(oracle_inc.get_call_count())

            # Run decremental algorithm
            oracle_dec = Oracle()
            start = time.perf_counter()
            min_bounds_dec, max_bounds_dec = decremental_orthogonal(
                points_with_consent, oracle_dec
            )
            dec_time_runs.append(time.perf_counter() - start)
            dec_calls_runs.append(oracle_dec.get_call_count())

            # Calculate lower bound
            min_bounds = min_bounds_inc if len(min_bounds_inc) > 0 else np.array([])
            max_bounds = max_bounds_inc if len(max_bounds_inc) > 0 else np.array([])

            points_outside = 0
            if len(min_bounds) > 0 and len(max_bounds) > 0:
                for point, _ in points_with_consent:
                    if not is_point_in_box(point, min_bounds, max_bounds):
                        points_outside += 1
            else:
                points_outside = num_points

            lower_bound = points_outside + 2 * n_dim
            lb_runs.append(lower_bound)

        incremental_calls.append(np.median(inc_calls_runs))
        decremental_calls.append(np.median(dec_calls_runs))
        lower_bounds.append(np.median(lb_runs))
        incremental_times.append(np.median(inc_time_runs))
        decremental_times.append(np.median(dec_time_runs))

        if num_points > 0:
            d_log_n = n_dim * np.log(num_points)
        else:
            d_log_n = 0
        d_log_n_values.append(d_log_n)

    # Plot 1: Oracle Calls vs Number of Points
    x = np.array(d_log_n_values)
    y = np.array(incremental_calls)
    if len(x) > 1 and np.any(x != x[0]):
        C, B = np.polyfit(x, y, 1)
    else:
        C, B = 1.0, 0.0
    scaled_d_log_n_values = [C * val + B for val in d_log_n_values]

    plt.figure(figsize=(12, 8))
    plt.plot(num_points_list, incremental_calls, label="Incremental box algorithm")
    plt.plot(num_points_list, decremental_calls, label="Decremental box algorithm")
    plt.plot(num_points_list, lower_bounds, label="Lower bound (points outside + 2*d)")
    plt.plot(
        num_points_list,
        scaled_d_log_n_values,
        label=f"d*log(n) scaled (y = {C:.2f}x + {B:.2f})",
    )
    plt.xlabel("Number of Points")
    plt.ylabel("Number of Consent Requests")
    plt.title(
        f"Oracle Calls vs Number of Points (Box)\n(d={n_dim}, p={consent_probability}, runs={n_runs})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "oracle_calls_vs_points_box.png"), dpi=150)
    plt.close()

    # Plot 2: Running Time vs Number of Points
    plt.figure(figsize=(12, 8))
    plt.plot(num_points_list, incremental_times, label="Incremental box algorithm")
    plt.plot(num_points_list, decremental_times, label="Decremental box algorithm")
    plt.xlabel("Number of Points")
    plt.ylabel("Running Time (seconds)")
    plt.title(
        f"Running Time vs Number of Points (Box)\n(d={n_dim}, p={consent_probability}, runs={n_runs})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "running_time_box_algorithms.png"), dpi=150)
    plt.close()


def plot_sphere_vs_points(
    n_dim,
    consent_probability,
    num_points_list,
    radius=10,
    center_bounds=(-10, 10),
    n_runs=10,
):
    """
    Runs sphere algorithms over a range of point counts, measuring both oracle calls and running times,
    then saves both corresponding plots.
    """
    incremental_calls = []
    decremental_calls = []
    lower_bounds = []
    factorial_bound_values = []
    p_bound_values = []
    incremental_times = []
    decremental_times = []

    pbar = tqdm(num_points_list, desc="Running sphere benchmarks vs points")
    for num_points in pbar:
        inc_calls_runs = []
        dec_calls_runs = []
        inc_time_runs = []
        dec_time_runs = []
        lb_runs = []
        for run_idx in range(n_runs):
            pbar.set_description(
                f"Sphere benchmarks vs points (N={num_points}, run {run_idx + 1}/{n_runs})"
            )
            points = generate_in_sphere(n_dim, radius, num_points)
            points_with_consent = [
                (p, random.random() < consent_probability) for p in points
            ]

            oracle_inc = Oracle()
            start = time.perf_counter()
            center_inc, radius_inc = incremental_distance_based(
                points_with_consent, oracle_inc
            )
            inc_time_runs.append(time.perf_counter() - start)
            inc_calls_runs.append(oracle_inc.get_call_count())

            oracle_dec = Oracle()
            start = time.perf_counter()
            center_dec, radius_dec = decremental_distance_based(
                points_with_consent, oracle_dec
            )
            dec_time_runs.append(time.perf_counter() - start)
            dec_calls_runs.append(oracle_dec.get_call_count())

            if center_inc is not None and radius_inc is not None and radius_inc > 0:
                center = np.array(center_inc)
                radius_sq = radius_inc**2
                points_outside = sum(
                    1
                    for p, _ in points_with_consent
                    if not is_point_in_sphere(p, center, radius_sq)
                )
            else:
                points_outside = num_points

            lb_runs.append(points_outside + n_dim + 1)

        incremental_calls.append(np.median(inc_calls_runs))
        decremental_calls.append(np.median(dec_calls_runs))
        lower_bounds.append(np.median(lb_runs))
        incremental_times.append(np.median(inc_time_runs))
        decremental_times.append(np.median(dec_time_runs))

        factorial_bound_values.append(
            np.log(num_points) ** (n_dim + 1) if num_points > 1 else 0
        )
        p_bound_values.append(
            (n_dim + 1) / (consent_probability ** (n_dim + 1))
            if consent_probability > 0
            else float("inf")
        )

    # Plot 1: Oracle Calls vs Number of Points
    xf = np.array(factorial_bound_values)
    yf = np.array(incremental_calls)
    if len(xf) > 1 and np.any(xf != xf[0]):
        Cf, Bf = np.polyfit(xf, yf, 1)
    else:
        Cf, Bf = 1.0, 0.0
    scaled_fact = [Cf * v + Bf for v in factorial_bound_values]

    plt.figure(figsize=(12, 8))
    plt.plot(num_points_list, incremental_calls, label="Incremental sphere algorithm")
    plt.plot(num_points_list, decremental_calls, label="Decremental sphere algorithm")
    plt.plot(
        num_points_list, lower_bounds, label="Lower bound (points outside + d + 1)"
    )
    plt.plot(
        num_points_list,
        scaled_fact,
        label=f"(d+1)! * ln^(d+1)(n) scaled (y = {Cf:.2f}x + {Bf:.2f})",
    )
    plt.xlabel("Number of Points")
    plt.ylabel("Number of Consent Requests")
    plt.title(
        f"Oracle Calls vs Number of Points (Sphere)\n(d={n_dim}, p={consent_probability}, runs={n_runs})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "oracle_calls_vs_points_sphere.png"), dpi=150)
    plt.close()

    # Plot 2: Running Time vs Number of Points
    plt.figure(figsize=(12, 8))
    plt.plot(num_points_list, incremental_times, label="Incremental sphere algorithm")
    plt.plot(num_points_list, decremental_times, label="Decremental sphere algorithm")
    plt.xlabel("Number of Points")
    plt.ylabel("Running Time (seconds)")
    plt.title(
        f"Running Time vs Number of Points (Sphere)\n(d={n_dim}, p={consent_probability}, runs={n_runs})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(RESULTS_DIR, "running_time_sphere_algorithms.png"), dpi=150
    )
    plt.close()


def plot_box_vs_consent(
    n_dim, num_points, consent_prob_list, low=-10, high=10, n_runs=10
):
    """
    Runs box algorithms over a range of consent probabilities, measuring both oracle calls and running times,
    then saves both corresponding plots.
    """
    incremental_calls = []
    decremental_calls = []
    lower_bounds = []
    incremental_times = []
    decremental_times = []

    pbar = tqdm(consent_prob_list, desc="Running box benchmarks vs consent prob")
    for consent_probability in pbar:
        inc_calls_runs = []
        dec_calls_runs = []
        inc_time_runs = []
        dec_time_runs = []
        lb_runs = []
        for run_idx in range(n_runs):
            pbar.set_description(
                f"Box benchmarks vs consent prob (p={consent_probability:.2f}, run {run_idx + 1}/{n_runs})"
            )
            points = generate_in_box(n_dim, low=low, high=high, count=num_points)
            points_with_consent = [
                (point, random.random() < consent_probability) for point in points
            ]

            # Run incremental algorithm
            oracle_inc = Oracle()
            start = time.perf_counter()
            min_bounds_inc, max_bounds_inc = incremental_orthogonal(
                points_with_consent, oracle_inc
            )
            inc_time_runs.append(time.perf_counter() - start)
            inc_calls_runs.append(oracle_inc.get_call_count())

            # Run decremental algorithm
            oracle_dec = Oracle()
            start = time.perf_counter()
            min_bounds_dec, max_bounds_dec = decremental_orthogonal(
                points_with_consent, oracle_dec
            )
            dec_time_runs.append(time.perf_counter() - start)
            dec_calls_runs.append(oracle_dec.get_call_count())

            # Calculate lower bound
            min_bounds = min_bounds_inc if len(min_bounds_inc) > 0 else np.array([])
            max_bounds = max_bounds_inc if len(max_bounds_inc) > 0 else np.array([])

            points_outside = 0
            if len(min_bounds) > 0 and len(max_bounds) > 0:
                for point, _ in points_with_consent:
                    if not is_point_in_box(point, min_bounds, max_bounds):
                        points_outside += 1
            else:
                points_outside = num_points

            lower_bound = points_outside + 2 * n_dim
            lb_runs.append(lower_bound)

        incremental_calls.append(np.median(inc_calls_runs))
        decremental_calls.append(np.median(dec_calls_runs))
        lower_bounds.append(np.median(lb_runs))
        incremental_times.append(np.median(inc_time_runs))
        decremental_times.append(np.median(dec_time_runs))

    # Plot 1: Oracle Calls vs Consent Probability
    plt.figure(figsize=(12, 8))
    plt.plot(consent_prob_list, incremental_calls, label="Incremental box algorithm")
    plt.plot(consent_prob_list, decremental_calls, label="Decremental box algorithm")
    plt.plot(
        consent_prob_list, lower_bounds, label="Lower bound (points outside + 2*d)"
    )
    plt.xlabel("Consent Probability")
    plt.ylabel("Number of Consent Requests")
    plt.title(
        f"Oracle Calls vs Consent Probability (Box)\n(d={n_dim}, n={num_points}, runs={n_runs})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "oracle_calls_vs_consent_box.png"), dpi=150)
    plt.close()

    # Plot 2: Running Time vs Consent Probability
    plt.figure(figsize=(12, 8))
    plt.plot(consent_prob_list, incremental_times, label="Incremental box algorithm")
    plt.plot(consent_prob_list, decremental_times, label="Decremental box algorithm")
    plt.xlabel("Consent Probability")
    plt.ylabel("Running Time (seconds)")
    plt.title(
        f"Running Time vs Consent Probability (Box)\n(d={n_dim}, n={num_points}, runs={n_runs})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "running_time_vs_consent_box.png"), dpi=150)
    plt.close()


def plot_sphere_vs_consent(
    n_dim, num_points, consent_prob_list, radius=10, n_runs=10
):
    """
    Runs sphere algorithms over a range of consent probabilities, measuring both oracle calls and running times,
    then saves both corresponding plots.
    """
    incremental_calls = []
    decremental_calls = []
    lower_bounds = []
    incremental_times = []
    decremental_times = []

    pbar = tqdm(consent_prob_list, desc="Running sphere benchmarks vs consent prob")
    for consent_probability in pbar:
        inc_calls_runs = []
        dec_calls_runs = []
        inc_time_runs = []
        dec_time_runs = []
        lb_runs = []
        for run_idx in range(n_runs):
            pbar.set_description(
                f"Sphere benchmarks vs consent prob (p={consent_probability:.2f}, run {run_idx + 1}/{n_runs})"
            )
            points = generate_in_sphere(n_dim, radius, num_points)
            points_with_consent = [
                (p, random.random() < consent_probability) for p in points
            ]

            oracle_inc = Oracle()
            start = time.perf_counter()
            center_inc, radius_inc = incremental_distance_based(
                points_with_consent, oracle_inc
            )
            inc_time_runs.append(time.perf_counter() - start)
            inc_calls_runs.append(oracle_inc.get_call_count())

            oracle_dec = Oracle()
            start = time.perf_counter()
            center_dec, radius_dec = decremental_distance_based(
                points_with_consent, oracle_dec
            )
            dec_time_runs.append(time.perf_counter() - start)
            dec_calls_runs.append(oracle_dec.get_call_count())

            if center_inc is not None and radius_inc is not None and radius_inc > 0:
                center = np.array(center_inc)
                radius_sq = radius_inc**2
                points_outside = sum(
                    1
                    for p, _ in points_with_consent
                    if not is_point_in_sphere(p, center, radius_sq)
                )
            else:
                points_outside = num_points

            lb_runs.append(points_outside + n_dim + 1)

        incremental_calls.append(np.median(inc_calls_runs))
        decremental_calls.append(np.median(dec_calls_runs))
        lower_bounds.append(np.median(lb_runs))
        incremental_times.append(np.median(inc_time_runs))
        decremental_times.append(np.median(dec_time_runs))

    # Plot 1: Oracle Calls vs Consent Probability
    plt.figure(figsize=(12, 8))
    plt.plot(consent_prob_list, incremental_calls, label="Incremental sphere algorithm")
    plt.plot(consent_prob_list, decremental_calls, label="Decremental sphere algorithm")
    plt.plot(
        consent_prob_list, lower_bounds, label="Lower bound (points outside + d + 1)"
    )
    plt.xlabel("Consent Probability")
    plt.ylabel("Number of Consent Requests")
    plt.title(
        f"Oracle Calls vs Consent Probability (Sphere)\n(d={n_dim}, n={num_points}, runs={n_runs})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(RESULTS_DIR, "oracle_calls_vs_consent_sphere.png"), dpi=150
    )
    plt.close()

    # Plot 2: Running Time vs Consent Probability
    plt.figure(figsize=(12, 8))
    plt.plot(consent_prob_list, incremental_times, label="Incremental sphere algorithm")
    plt.plot(consent_prob_list, decremental_times, label="Decremental sphere algorithm")
    plt.xlabel("Consent Probability")
    plt.ylabel("Running Time (seconds)")
    plt.title(
        f"Running Time vs Consent Probability (Sphere)\n(d={n_dim}, n={num_points}, runs={n_runs})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(RESULTS_DIR, "running_time_vs_consent_sphere.png"), dpi=150
    )
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmarks for db_consents orthogonal and distance-based algorithms."
    )
    parser.add_argument(
        "-n",
        "--runs",
        type=int,
        default=10,
        help="Number of repetitions to run each experiment and compute the median (default: 10)",
    )
    args = parser.parse_args()

    n_runs = args.runs
    n_dim = 3
    consent_probability = 0.7
    num_points_list = range(100, 10000, 500)

    logger.info(f"Running Box Benchmarks vs Number of Points (n_runs={n_runs})...")
    plot_box_vs_points(
        n_dim, consent_probability, num_points_list, n_runs=n_runs
    )

    logger.info(f"Running Sphere Benchmarks vs Number of Points (n_runs={n_runs})...")
    plot_sphere_vs_points(
        n_dim, consent_probability, num_points_list, n_runs=n_runs
    )

    # --- Consent probability on X-axis ---
    num_points_fixed = 500
    consent_prob_list = [p / 100 for p in range(5, 100, 5)]

    logger.info(
        f"Running Box Benchmarks vs Consent Probability (n_runs={n_runs})..."
    )
    plot_box_vs_consent(
        n_dim, num_points_fixed, consent_prob_list, n_runs=n_runs
    )

    logger.info(
        f"Running Sphere Benchmarks vs Consent Probability (n_runs={n_runs})..."
    )
    plot_sphere_vs_consent(
        n_dim, num_points_fixed, consent_prob_list, n_runs=n_runs
    )


if __name__ == "__main__":
    main()
