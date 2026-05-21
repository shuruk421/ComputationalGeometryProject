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
        f"Oracle Calls vs Number of Points (Box)\n(d={n_dim}, p={consent_probability})"
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

    # Linear regression for p-bound (Decremental)
    xp = np.array(p_bound_values)
    yp = np.array(decremental_calls)
    # Filter out infs for regression
    valid = ~np.isinf(xp)
    if np.sum(valid) > 1 and np.any(xp[valid] != xp[valid][0]):
        Cp, Bp = np.polyfit(xp[valid], yp[valid], 1)
    else:
        Cp, Bp = 1.0, 0.0
    scaled_p = [Cp * v + Bp for v in p_bound_values]

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
    plt.plot(
        num_points_list,
        scaled_p,
        label=f"(d+1)/(p^(d+1)) scaled (y = {Cp:.2f}x + {Bp:.2f})",
    )

    plt.xlabel("Number of Points")
    plt.ylabel("Number of Consent Requests")
    plt.title(
        f"Oracle Calls vs Number of Points (Sphere)\n(d={n_dim}, p={consent_probability})"
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
        f"Running Time vs Number of Points (Box)\n(d={n_dim}, p={consent_probability})"
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
        f"Running Time vs Number of Points (Sphere)\n(d={n_dim}, p={consent_probability})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(RESULTS_DIR, "running_time_sphere_algorithms.png"), dpi=150
    )

def plot_oracle_calls_vs_dim_box(
    num_points, consent_probability, dim_list, low=-10, high=10, n_runs=10
):
    """
    Draws a graph with Dimension on X-axis and Number of consent requests on Y-axis.
    """
    incremental_calls = []
    decremental_calls = []
    lower_bounds = []
    d_log_n_values = []

    pbar = tqdm(dim_list, desc="Processing box algorithms vs dimension")
    for n_dim in pbar:
        inc_runs = []
        dec_runs = []
        lb_runs = []
        for run_idx in range(n_runs):
            pbar.set_description(
                f"Processing box algorithms (d={n_dim}, run {run_idx + 1}/{n_runs})"
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
    x = np.array(d_log_n_values)
    y = np.array(incremental_calls)

    if len(x) > 1 and np.any(x != x[0]):
        C, B = np.polyfit(x, y, 1)
    else:
        C, B = 1.0, 0.0

    # Scale d*log(n) values
    scaled_d_log_n_values = [C * val + B for val in d_log_n_values]

    plt.figure(figsize=(12, 8))
    plt.plot(dim_list, incremental_calls, label="Incremental box algorithm")
    plt.plot(dim_list, decremental_calls, label="Decremental box algorithm")
    plt.plot(dim_list, lower_bounds, label="Lower bound (points outside + 2*d)")
    plt.plot(
        dim_list,
        scaled_d_log_n_values,
        label=f"d*log(n) scaled (y = {C:.2f}x + {B:.2f})",
    )

    plt.xlabel("Dimension (d)")
    plt.ylabel("Number of Consent Requests")
    plt.title(
        f"Oracle Calls vs Dimension (Box)\n(N={num_points}, p={consent_probability})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "oracle_calls_vs_dim_box.png"), dpi=150)


def plot_oracle_calls_vs_dim_sphere(
    num_points,
    consent_probability,
    dim_list,
    radius=10,
    center_bounds=(-10, 10),
    n_runs=10,
):
    """
    Draws a graph for sphere algorithms vs dimension.
    """
    import math
    incremental_calls = []
    decremental_calls = []
    lower_bounds = []
    factorial_bound_values = []
    p_bound_values = []

    pbar = tqdm(dim_list, desc="Processing sphere algorithms vs dimension")
    for n_dim in pbar:
        inc_runs = []
        dec_runs = []
        lb_runs = []
        for run_idx in range(n_runs):
            pbar.set_description(
                f"Processing sphere algorithms (d={n_dim}, run {run_idx + 1}/{n_runs})"
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

        factorial_val = math.factorial(n_dim + 1) * (np.log(num_points) ** (n_dim + 1)) if num_points > 1 else 0
        factorial_bound_values.append(factorial_val)
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

    # Linear regression for p-bound (Decremental)
    xp = np.array(p_bound_values)
    yp = np.array(decremental_calls)
    valid = ~np.isinf(xp)
    if np.sum(valid) > 1 and np.any(xp[valid] != xp[valid][0]):
        Cp, Bp = np.polyfit(xp[valid], yp[valid], 1)
    else:
        Cp, Bp = 1.0, 0.0
    scaled_p = [Cp * v + Bp for v in p_bound_values]

    plt.figure(figsize=(12, 8))
    plt.plot(dim_list, incremental_calls, label="Incremental sphere algorithm")
    plt.plot(dim_list, decremental_calls, label="Decremental sphere algorithm")
    plt.plot(
        dim_list, lower_bounds, label="Lower bound (points outside + d + 1)"
    )
    plt.plot(
        dim_list,
        scaled_fact,
        label=f"(d+1)! * ln^(d+1)(n) scaled (y = {Cf:.2f}x + {Bf:.2f})",
    )
    plt.plot(
        dim_list,
        scaled_p,
        label=f"(d+1)/(p^(d+1)) scaled (y = {Cp:.2f}x + {Bp:.2f})",
    )

    plt.xlabel("Dimension (d)")
    plt.ylabel("Number of Consent Requests")
    plt.title(
        f"Oracle Calls vs Dimension (Sphere)\n(N={num_points}, p={consent_probability})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "oracle_calls_vs_dim_sphere.png"), dpi=150)


def plot_running_time_box_algorithms_vs_dim(
    num_points, consent_probability, dim_list, low=-10, high=10, n_runs=10
):
    """
    Draws a graph for running time of box algorithms vs dimension.
    """
    incremental_times = []
    decremental_times = []

    pbar = tqdm(dim_list, desc="Measuring box algorithm running times vs dimension")
    for n_dim in pbar:
        inc_runs = []
        dec_runs = []
        for run_idx in range(n_runs):
            pbar.set_description(
                f"Measuring box algorithm running times (d={n_dim}, run {run_idx + 1}/{n_runs})"
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
    plt.plot(dim_list, incremental_times, label="Incremental box algorithm")
    plt.plot(dim_list, decremental_times, label="Decremental box algorithm")
    plt.xlabel("Dimension (d)")
    plt.ylabel("Running Time (seconds)")
    plt.title(
        f"Running Time vs Dimension (Box)\n(N={num_points}, p={consent_probability})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "running_time_box_algorithms_vs_dim.png"), dpi=150)


def plot_running_time_sphere_algorithms_vs_dim(
    num_points, consent_probability, dim_list, radius=10, n_runs=10
):
    """
    Draws a graph for running time of sphere algorithms vs dimension.
    """
    incremental_times = []
    decremental_times = []

    pbar = tqdm(dim_list, desc="Measuring sphere algorithm running times vs dimension")
    for n_dim in pbar:
        inc_runs = []
        dec_runs = []
        for run_idx in range(n_runs):
            pbar.set_description(
                f"Measuring sphere algorithm running times (d={n_dim}, run {run_idx + 1}/{n_runs})"
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
    plt.plot(dim_list, incremental_times, label="Incremental sphere algorithm")
    plt.plot(dim_list, decremental_times, label="Decremental sphere algorithm")
    plt.xlabel("Dimension (d)")
    plt.ylabel("Running Time (seconds)")
    plt.title(
        f"Running Time vs Dimension (Sphere)\n(N={num_points}, p={consent_probability})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(RESULTS_DIR, "running_time_sphere_algorithms_vs_dim.png"), dpi=150
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
    parser.add_argument(
        "-x",
        "--xaxis",
        type=str,
        choices=["points", "dimension", "both"],
        default="both",
        help="Which axis to vary for the benchmarks: points, dimension, or both (default: both)",
    )
    parser.add_argument(
        "-p",
        "--points",
        type=int,
        default=500,
        help="Fixed number of points to use when X-axis is dimension (default: 500)",
    )
    parser.add_argument(
        "--points-range",
        type=int,
        nargs=3,
        default=[100, 1000, 10],
        help="Range of points to vary (start, stop, step) (default: 100 1000 10)",
    )
    parser.add_argument(
        "-d",
        "--dim",
        type=int,
        default=3,
        help="Fixed dimension to use when X-axis is points (default: 3)",
    )
    parser.add_argument(
        "--dim-range",
        type=int,
        nargs=3,
        default=[1, 20, 1],
        help="Range of dimensions to vary (start, stop, step) (default: 1 20 1)",
    )
    parser.add_argument(
        "-c",
        "--consent-prob",
        type=float,
        default=0.7,
        help="Consent probability to use in experiments (default: 0.7)",
    )
    args = parser.parse_args()

    n_runs = args.runs
    consent_probability = args.consent_prob

    # 1. Runs points-based benchmarks if requested
    if args.xaxis in ["points", "both"]:
        n_dim = args.dim
        num_points_list = range(args.points_range[0], args.points_range[1], args.points_range[2])

        logger.info(f"Running Box Oracle Calls Benchmark (n_runs={n_runs}, d={n_dim})...")
        plot_oracle_calls_vs_points_box(
            n_dim, consent_probability, num_points_list, n_runs=n_runs
        )

        logger.info(f"Running Sphere Oracle Calls Benchmark (n_runs={n_runs}, d={n_dim})...")
        plot_oracle_calls_vs_points_sphere(
            n_dim, consent_probability, num_points_list, n_runs=n_runs
        )

        logger.info(f"Running Box Running Time Benchmark (n_runs={n_runs}, d={n_dim})...")
        plot_running_time_box_algorithms(
            n_dim, consent_probability, num_points_list, n_runs=n_runs
        )

        logger.info(f"Running Sphere Running Time Benchmark (n_runs={n_runs}, d={n_dim})...")
        plot_running_time_sphere_algorithms(
            n_dim, consent_probability, num_points_list, n_runs=n_runs
        )

    # 2. Runs dimension-based benchmarks if requested
    if args.xaxis in ["dimension", "both"]:
        num_points = args.points
        dim_list = list(range(args.dim_range[0], args.dim_range[1], args.dim_range[2]))

        logger.info(f"Running Box Oracle Calls vs Dimension Benchmark (n_runs={n_runs}, N={num_points})...")
        plot_oracle_calls_vs_dim_box(
            num_points, consent_probability, dim_list, n_runs=n_runs
        )

        logger.info(f"Running Sphere Oracle Calls vs Dimension Benchmark (n_runs={n_runs}, N={num_points})...")
        plot_oracle_calls_vs_dim_sphere(
            num_points, consent_probability, dim_list, n_runs=n_runs
        )

        logger.info(f"Running Box Running Time vs Dimension Benchmark (n_runs={n_runs}, N={num_points})...")
        plot_running_time_box_algorithms_vs_dim(
            num_points, consent_probability, dim_list, n_runs=n_runs
        )

        logger.info(f"Running Sphere Running Time vs Dimension Benchmark (n_runs={n_runs}, N={num_points})...")
        plot_running_time_sphere_algorithms_vs_dim(
            num_points, consent_probability, dim_list, n_runs=n_runs
        )


if __name__ == "__main__":
    main()

