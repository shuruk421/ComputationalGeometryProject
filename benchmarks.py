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


def plot_oracle_calls_vs_points_box(
    n_dim, consent_probability, num_points_list, low=-10, high=10, n_runs=10
):
    """
    Draws a graph with Number of points on X-axis and Number of consent requests on Y-axis.
    """
    incremental_calls = []
    decremental_calls = []
    lower_bounds = []
    d_log_n_values = []

    pbar = tqdm(num_points_list, desc="Processing box algorithms")
    for num_points in pbar:
        inc_runs = []
        dec_runs = []
        lb_runs = []
        for run_idx in range(n_runs):
            pbar.set_description(
                f"Processing box algorithms (N={num_points}, run {run_idx + 1}/{n_runs})"
            )
            # Generate random points in a box
            points = generate_in_box(n_dim, low=low, high=high, count=num_points)
            # Add consent values
            points_with_consent = [
                (point, random.random() < consent_probability) for point in points
            ]

            # Run incremental algorithm
            oracle_inc = Oracle()
            min_bounds_inc, max_bounds_inc = incremental_orthogonal(
                points_with_consent, oracle_inc
            )
            inc_runs.append(oracle_inc.get_call_count())

            # Run decremental algorithm
            oracle_dec = Oracle()
            min_bounds_dec, max_bounds_dec = decremental_orthogonal(
                points_with_consent, oracle_dec
            )
            dec_runs.append(oracle_dec.get_call_count())

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

        incremental_calls.append(np.median(inc_runs))
        decremental_calls.append(np.median(dec_runs))
        lower_bounds.append(np.median(lb_runs))

        if num_points > 0:
            d_log_n = n_dim * np.log(num_points)
        else:
            d_log_n = 0
        d_log_n_values.append(d_log_n)

    # Calculate scaling constant and additive constant using linear regression
    # y = C * x + B where x is d_log_n_values and y is incremental_calls
    x = np.array(d_log_n_values)
    y = np.array(incremental_calls)

    if len(x) > 1 and np.any(x != x[0]):
        C, B = np.polyfit(x, y, 1)
    else:
        C, B = 1.0, 0.0

    # Scale d*log(n) values
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


def plot_oracle_calls_vs_points_sphere(
    n_dim,
    consent_probability,
    num_points_list,
    radius=10,
    center_bounds=(-10, 10),
    n_runs=10,
):
    """
    Draws a graph for sphere algorithms.
    """
    incremental_calls = []
    decremental_calls = []
    lower_bounds = []
    factorial_bound_values = []
    p_bound_values = []

    pbar = tqdm(num_points_list, desc="Processing sphere algorithms")
    for num_points in pbar:
        inc_runs = []
        dec_runs = []
        lb_runs = []
        for run_idx in range(n_runs):
            pbar.set_description(
                f"Processing sphere algorithms (N={num_points}, run {run_idx + 1}/{n_runs})"
            )
            points = generate_in_sphere(n_dim, radius, num_points)
            points_with_consent = [
                (p, random.random() < consent_probability) for p in points
            ]

            oracle_inc = Oracle()
            center_inc, radius_inc = incremental_distance_based(
                points_with_consent, oracle_inc
            )
            inc_runs.append(oracle_inc.get_call_count())

            oracle_dec = Oracle()
            center_dec, radius_dec = decremental_distance_based(
                points_with_consent, oracle_dec
            )
            dec_runs.append(oracle_dec.get_call_count())

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

        incremental_calls.append(np.median(inc_runs))
        decremental_calls.append(np.median(dec_runs))
        lower_bounds.append(np.median(lb_runs))

        factorial_bound_values.append(
            np.log(num_points) ** (n_dim + 1) if num_points > 1 else 0
        )
        p_bound_values.append(
            (n_dim + 1) / (consent_probability ** (n_dim + 1))
            if consent_probability > 0
            else float("inf")
        )

    # Linear regression for factorial bound (Incremental)
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


def plot_running_time_box_algorithms(
    n_dim, consent_probability, num_points_list, low=-10, high=10, n_runs=10
):
    """
    Draws a graph for running time of box algorithms.
    """
    incremental_times = []
    decremental_times = []

    pbar = tqdm(num_points_list, desc="Measuring box algorithm running times")
    for num_points in pbar:
        inc_runs = []
        dec_runs = []
        for run_idx in range(n_runs):
            pbar.set_description(
                f"Measuring box algorithm running times (N={num_points}, run {run_idx + 1}/{n_runs})"
            )
            points = generate_in_box(n_dim, low=low, high=high, count=num_points)
            points_with_consent = [
                (p, random.random() < consent_probability) for p in points
            ]

            oracle_inc = Oracle()
            start = time.perf_counter()
            incremental_orthogonal(points_with_consent, oracle_inc)
            inc_runs.append(time.perf_counter() - start)

            oracle_dec = Oracle()
            start = time.perf_counter()
            decremental_orthogonal(points_with_consent, oracle_dec)
            dec_runs.append(time.perf_counter() - start)

        incremental_times.append(np.median(inc_runs))
        decremental_times.append(np.median(dec_runs))

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


def plot_running_time_sphere_algorithms(
    n_dim, consent_probability, num_points_list, radius=10, n_runs=10
):
    """
    Draws a graph for running time of sphere algorithms.
    """
    incremental_times = []
    decremental_times = []

    pbar = tqdm(num_points_list, desc="Measuring sphere algorithm running times")
    for num_points in pbar:
        inc_runs = []
        dec_runs = []
        for run_idx in range(n_runs):
            pbar.set_description(
                f"Measuring sphere algorithm running times (N={num_points}, run {run_idx + 1}/{n_runs})"
            )
            points = generate_in_sphere(n_dim, radius, num_points)
            points_with_consent = [
                (p, random.random() < consent_probability) for p in points
            ]

            oracle_inc = Oracle()
            start = time.perf_counter()
            incremental_distance_based(points_with_consent, oracle_inc)
            inc_runs.append(time.perf_counter() - start)

            oracle_dec = Oracle()
            start = time.perf_counter()
            decremental_distance_based(points_with_consent, oracle_dec)
            dec_runs.append(time.perf_counter() - start)

        incremental_times.append(np.median(inc_runs))
        decremental_times.append(np.median(dec_runs))

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


def plot_oracle_calls_vs_consent_box(
    n_dim, num_points, consent_prob_list, low=-10, high=10, n_runs=10
):
    """
    Draws a graph with Consent Probability on X-axis and Number of consent requests on Y-axis (Box).
    """
    incremental_calls = []
    decremental_calls = []
    lower_bounds = []

    pbar = tqdm(consent_prob_list, desc="Processing box algorithms (vs consent prob)")
    for consent_probability in pbar:
        inc_runs = []
        dec_runs = []
        lb_runs = []
        for run_idx in range(n_runs):
            pbar.set_description(
                f"Processing box algorithms (p={consent_probability:.2f}, run {run_idx + 1}/{n_runs})"
            )
            points = generate_in_box(n_dim, low=low, high=high, count=num_points)
            points_with_consent = [
                (point, random.random() < consent_probability) for point in points
            ]

            oracle_inc = Oracle()
            min_bounds_inc, max_bounds_inc = incremental_orthogonal(
                points_with_consent, oracle_inc
            )
            inc_runs.append(oracle_inc.get_call_count())

            oracle_dec = Oracle()
            min_bounds_dec, max_bounds_dec = decremental_orthogonal(
                points_with_consent, oracle_dec
            )
            dec_runs.append(oracle_dec.get_call_count())

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

        incremental_calls.append(np.median(inc_runs))
        decremental_calls.append(np.median(dec_runs))
        lower_bounds.append(np.median(lb_runs))

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


def plot_oracle_calls_vs_consent_sphere(
    n_dim, num_points, consent_prob_list, radius=10, n_runs=10
):
    """
    Draws a graph with Consent Probability on X-axis and Number of consent requests on Y-axis (Sphere).
    """
    incremental_calls = []
    decremental_calls = []
    lower_bounds = []

    pbar = tqdm(
        consent_prob_list, desc="Processing sphere algorithms (vs consent prob)"
    )
    for consent_probability in pbar:
        inc_runs = []
        dec_runs = []
        lb_runs = []
        for run_idx in range(n_runs):
            pbar.set_description(
                f"Processing sphere algorithms (p={consent_probability:.2f}, run {run_idx + 1}/{n_runs})"
            )
            points = generate_in_sphere(n_dim, radius, num_points)
            points_with_consent = [
                (p, random.random() < consent_probability) for p in points
            ]

            oracle_inc = Oracle()
            center_inc, radius_inc = incremental_distance_based(
                points_with_consent, oracle_inc
            )
            inc_runs.append(oracle_inc.get_call_count())

            oracle_dec = Oracle()
            center_dec, radius_dec = decremental_distance_based(
                points_with_consent, oracle_dec
            )
            dec_runs.append(oracle_dec.get_call_count())

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

        incremental_calls.append(np.median(inc_runs))
        decremental_calls.append(np.median(dec_runs))
        lower_bounds.append(np.median(lb_runs))

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


def plot_running_time_vs_consent_box(
    n_dim, num_points, consent_prob_list, low=-10, high=10, n_runs=10
):
    """
    Draws a graph for running time of box algorithms vs consent probability.
    """
    incremental_times = []
    decremental_times = []

    pbar = tqdm(consent_prob_list, desc="Measuring box running times (vs consent prob)")
    for consent_probability in pbar:
        inc_runs = []
        dec_runs = []
        for run_idx in range(n_runs):
            pbar.set_description(
                f"Measuring box running times (p={consent_probability:.2f}, run {run_idx + 1}/{n_runs})"
            )
            points = generate_in_box(n_dim, low=low, high=high, count=num_points)
            points_with_consent = [
                (p, random.random() < consent_probability) for p in points
            ]

            oracle_inc = Oracle()
            start = time.perf_counter()
            incremental_orthogonal(points_with_consent, oracle_inc)
            inc_runs.append(time.perf_counter() - start)

            oracle_dec = Oracle()
            start = time.perf_counter()
            decremental_orthogonal(points_with_consent, oracle_dec)
            dec_runs.append(time.perf_counter() - start)

        incremental_times.append(np.median(inc_runs))
        decremental_times.append(np.median(dec_runs))

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


def plot_running_time_vs_consent_sphere(
    n_dim, num_points, consent_prob_list, radius=10, n_runs=10
):
    """
    Draws a graph for running time of sphere algorithms vs consent probability.
    """
    incremental_times = []
    decremental_times = []

    pbar = tqdm(
        consent_prob_list, desc="Measuring sphere running times (vs consent prob)"
    )
    for consent_probability in pbar:
        inc_runs = []
        dec_runs = []
        for run_idx in range(n_runs):
            pbar.set_description(
                f"Measuring sphere running times (p={consent_probability:.2f}, run {run_idx + 1}/{n_runs})"
            )
            points = generate_in_sphere(n_dim, radius, num_points)
            points_with_consent = [
                (p, random.random() < consent_probability) for p in points
            ]

            oracle_inc = Oracle()
            start = time.perf_counter()
            incremental_distance_based(points_with_consent, oracle_inc)
            inc_runs.append(time.perf_counter() - start)

            oracle_dec = Oracle()
            start = time.perf_counter()
            decremental_distance_based(points_with_consent, oracle_dec)
            dec_runs.append(time.perf_counter() - start)

        incremental_times.append(np.median(inc_runs))
        decremental_times.append(np.median(dec_runs))

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
    num_points_list = range(100, 1000, 10)

    logger.info(f"Running Box Oracle Calls Benchmark (n_runs={n_runs})...")
    plot_oracle_calls_vs_points_box(
        n_dim, consent_probability, num_points_list, n_runs=n_runs
    )

    logger.info(f"Running Sphere Oracle Calls Benchmark (n_runs={n_runs})...")
    plot_oracle_calls_vs_points_sphere(
        n_dim, consent_probability, num_points_list, n_runs=n_runs
    )

    logger.info(f"Running Box Running Time Benchmark (n_runs={n_runs})...")
    plot_running_time_box_algorithms(
        n_dim, consent_probability, num_points_list, n_runs=n_runs
    )

    logger.info(f"Running Sphere Running Time Benchmark (n_runs={n_runs})...")
    plot_running_time_sphere_algorithms(
        n_dim, consent_probability, num_points_list, n_runs=n_runs
    )

    # --- Consent probability on X-axis ---
    num_points_fixed = 500
    consent_prob_list = [p / 100 for p in range(5, 100, 5)]

    logger.info(
        f"Running Box Oracle Calls vs Consent Prob Benchmark (n_runs={n_runs})..."
    )
    plot_oracle_calls_vs_consent_box(
        n_dim, num_points_fixed, consent_prob_list, n_runs=n_runs
    )

    logger.info(
        f"Running Sphere Oracle Calls vs Consent Prob Benchmark (n_runs={n_runs})..."
    )
    plot_oracle_calls_vs_consent_sphere(
        n_dim, num_points_fixed, consent_prob_list, n_runs=n_runs
    )

    logger.info(
        f"Running Box Running Time vs Consent Prob Benchmark (n_runs={n_runs})..."
    )
    plot_running_time_vs_consent_box(
        n_dim, num_points_fixed, consent_prob_list, n_runs=n_runs
    )

    logger.info(
        f"Running Sphere Running Time vs Consent Prob Benchmark (n_runs={n_runs})..."
    )
    plot_running_time_vs_consent_sphere(
        n_dim, num_points_fixed, consent_prob_list, n_runs=n_runs
    )


if __name__ == "__main__":
    main()
