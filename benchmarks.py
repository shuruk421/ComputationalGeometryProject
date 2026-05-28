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


def plot_sphere_vs_consent(n_dim, num_points, consent_prob_list, radius=10, n_runs=10):
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


def plot_box_vs_dimension(
    dimensions, consent_probability=0.7, num_points=1000, n_runs=10
):
    """
    Runs box algorithms over a range of dimensions, measuring both oracle calls and running times,
    then saves both corresponding plots.
    """
    incremental_calls = []
    decremental_calls = []
    incremental_times = []
    decremental_times = []

    pbar = tqdm(dimensions, desc="Running box benchmarks vs dimension")
    for d in pbar:
        inc_calls_runs = []
        dec_calls_runs = []
        inc_time_runs = []
        dec_time_runs = []
        for run_idx in range(n_runs):
            pbar.set_description(
                f"Box benchmarks vs dimension (d={d}, run {run_idx + 1}/{n_runs})"
            )
            # Generate random points in a box
            points = generate_in_box(d, low=-10, high=10, count=num_points)
            points_with_consent = [
                (point, random.random() < consent_probability) for point in points
            ]

            # Run incremental algorithm
            oracle_inc = Oracle()
            start = time.perf_counter()
            incremental_orthogonal(points_with_consent, oracle_inc)
            inc_time_runs.append(time.perf_counter() - start)
            inc_calls_runs.append(oracle_inc.get_call_count())

            # Run decremental algorithm
            oracle_dec = Oracle()
            start = time.perf_counter()
            decremental_orthogonal(points_with_consent, oracle_dec)
            dec_time_runs.append(time.perf_counter() - start)
            dec_calls_runs.append(oracle_dec.get_call_count())

        incremental_calls.append(np.median(inc_calls_runs))
        decremental_calls.append(np.median(dec_calls_runs))
        incremental_times.append(np.median(inc_time_runs))
        decremental_times.append(np.median(dec_time_runs))

    # Plot 1: Oracle Calls vs Dimension (Box)
    plt.figure(figsize=(12, 8))
    plt.plot(
        dimensions, incremental_calls, marker="o", label="Incremental box algorithm"
    )
    plt.plot(
        dimensions, decremental_calls, marker="s", label="Decremental box algorithm"
    )
    plt.xlabel("Dimension (d)")
    plt.ylabel("Number of Consent Requests")
    plt.title(
        f"Oracle Calls vs Dimension (Box)\n(N={num_points}, p={consent_probability}, runs={n_runs})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "oracle_calls_vs_dimension_box.png"), dpi=150)
    plt.close()

    # Plot 2: Running Time vs Dimension (Box)
    plt.figure(figsize=(12, 8))
    plt.plot(
        dimensions, incremental_times, marker="o", label="Incremental box algorithm"
    )
    plt.plot(
        dimensions, decremental_times, marker="s", label="Decremental box algorithm"
    )
    plt.xlabel("Dimension (d)")
    plt.ylabel("Running Time (seconds)")
    plt.title(
        f"Running Time vs Dimension (Box)\n(N={num_points}, p={consent_probability}, runs={n_runs})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "running_time_vs_dimension_box.png"), dpi=150)
    plt.close()


def plot_sphere_vs_dimension(
    dimensions, consent_probability=0.7, num_points=1000, radius=10, n_runs=10
):
    """
    Runs sphere algorithms over a range of dimensions, measuring both oracle calls and running times,
    then saves both corresponding plots.
    """
    incremental_calls = []
    decremental_calls = []
    incremental_times = []
    decremental_times = []

    pbar = tqdm(dimensions, desc="Running sphere benchmarks vs dimension")
    for d in pbar:
        inc_calls_runs = []
        dec_calls_runs = []
        inc_time_runs = []
        dec_time_runs = []
        for run_idx in range(n_runs):
            pbar.set_description(
                f"Sphere benchmarks vs dimension (d={d}, run {run_idx + 1}/{n_runs})"
            )
            # Generate random points in a sphere
            points = generate_in_sphere(d, radius, num_points)
            points_with_consent = [
                (point, random.random() < consent_probability) for point in points
            ]

            # Run incremental algorithm
            oracle_inc = Oracle()
            start = time.perf_counter()
            incremental_distance_based(points_with_consent, oracle_inc)
            inc_time_runs.append(time.perf_counter() - start)
            inc_calls_runs.append(oracle_inc.get_call_count())

            # Run decremental algorithm
            oracle_dec = Oracle()
            start = time.perf_counter()
            decremental_distance_based(points_with_consent, oracle_dec)
            dec_time_runs.append(time.perf_counter() - start)
            dec_calls_runs.append(oracle_dec.get_call_count())

        incremental_calls.append(np.median(inc_calls_runs))
        decremental_calls.append(np.median(dec_calls_runs))
        incremental_times.append(np.median(inc_time_runs))
        decremental_times.append(np.median(dec_time_runs))

    # Plot 1: Oracle Calls vs Dimension (Sphere)
    plt.figure(figsize=(12, 8))
    plt.plot(
        dimensions, incremental_calls, marker="o", label="Incremental sphere algorithm"
    )
    plt.plot(
        dimensions, decremental_calls, marker="s", label="Decremental sphere algorithm"
    )
    plt.xlabel("Dimension (d)")
    plt.ylabel("Number of Consent Requests")
    plt.title(
        f"Oracle Calls vs Dimension (Sphere)\n(N={num_points}, p={consent_probability}, runs={n_runs})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(RESULTS_DIR, "oracle_calls_vs_dimension_sphere.png"), dpi=150
    )
    plt.close()

    # Plot 2: Running Time vs Dimension (Sphere)
    plt.figure(figsize=(12, 8))
    plt.plot(
        dimensions, incremental_times, marker="o", label="Incremental sphere algorithm"
    )
    plt.plot(
        dimensions, decremental_times, marker="s", label="Decremental sphere algorithm"
    )
    plt.xlabel("Dimension (d)")
    plt.ylabel("Running Time (seconds)")
    plt.title(
        f"Running Time vs Dimension (Sphere)\n(N={num_points}, p={consent_probability}, runs={n_runs})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(RESULTS_DIR, "running_time_vs_dimension_sphere.png"), dpi=150
    )
    plt.close()


def plot_noisy_sphere_vs_points(
    n_dim,
    consent_probability,
    num_points_list,
    radius=10,
    noise_std=0.5,
    center_bounds=(-10, 10),
    n_runs=10,
):
    """
    Runs sphere algorithms over a range of point counts for points generated on a noisy sphere (disc in 2D),
    measuring both oracle calls and running times, then saves both corresponding plots.
    """
    incremental_calls = []
    decremental_calls = []
    lower_bounds = []
    factorial_bound_values = []
    p_bound_values = []
    incremental_times = []
    decremental_times = []

    pbar = tqdm(num_points_list, desc="Running noisy sphere benchmarks vs points")
    for num_points in pbar:
        inc_calls_runs = []
        dec_calls_runs = []
        inc_time_runs = []
        dec_time_runs = []
        lb_runs = []
        for run_idx in range(n_runs):
            pbar.set_description(
                f"Noisy sphere benchmarks vs points (N={num_points}, run {run_idx + 1}/{n_runs})"
            )
            # Generate points on a sphere with noise
            points = generate_noisy_sphere_points(
                n_dim, num_points, radius, noise_std, center_bounds=center_bounds
            )
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
        f"Oracle Calls vs Number of Points (Noisy Sphere)\n(d={n_dim}, p={consent_probability}, noise_std={noise_std}, runs={n_runs})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(RESULTS_DIR, "oracle_calls_vs_points_noisy_sphere.png"), dpi=150
    )
    plt.close()

    # Plot 2: Running Time vs Number of Points
    plt.figure(figsize=(12, 8))
    plt.plot(num_points_list, incremental_times, label="Incremental sphere algorithm")
    plt.plot(num_points_list, decremental_times, label="Decremental sphere algorithm")
    plt.xlabel("Number of Points")
    plt.ylabel("Running Time (seconds)")
    plt.title(
        f"Running Time vs Number of Points (Noisy Sphere)\n(d={n_dim}, p={consent_probability}, noise_std={noise_std}, runs={n_runs})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(RESULTS_DIR, "running_time_noisy_sphere_algorithms.png"), dpi=150
    )
    plt.close()


def plot_noisy_sphere_vs_consent(
    n_dim,
    num_points,
    consent_prob_list,
    radius=10,
    noise_std=0.5,
    center_bounds=(-10, 10),
    n_runs=10,
):
    """
    Runs sphere algorithms over a range of consent probabilities for points generated on a noisy sphere (disc in 2D),
    measuring both oracle calls and running times, then saves both corresponding plots.
    """
    incremental_calls = []
    decremental_calls = []
    lower_bounds = []
    incremental_times = []
    decremental_times = []

    pbar = tqdm(
        consent_prob_list, desc="Running noisy sphere benchmarks vs consent prob"
    )
    for consent_probability in pbar:
        inc_calls_runs = []
        dec_calls_runs = []
        inc_time_runs = []
        dec_time_runs = []
        lb_runs = []
        for run_idx in range(n_runs):
            pbar.set_description(
                f"Noisy sphere benchmarks vs consent prob (p={consent_probability:.2f}, run {run_idx + 1}/{n_runs})"
            )
            # Generate points on a sphere with noise
            points = generate_noisy_sphere_points(
                n_dim, num_points, radius, noise_std, center_bounds=center_bounds
            )
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
        f"Oracle Calls vs Consent Probability (Noisy Sphere)\n(d={n_dim}, n={num_points}, noise_std={noise_std}, runs={n_runs})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(RESULTS_DIR, "oracle_calls_vs_consent_noisy_sphere.png"), dpi=150
    )
    plt.close()

    # Plot 2: Running Time vs Consent Probability
    plt.figure(figsize=(12, 8))
    plt.plot(consent_prob_list, incremental_times, label="Incremental sphere algorithm")
    plt.plot(consent_prob_list, decremental_times, label="Decremental sphere algorithm")
    plt.xlabel("Consent Probability")
    plt.ylabel("Running Time (seconds)")
    plt.title(
        f"Running Time vs Consent Probability (Noisy Sphere)\n(d={n_dim}, n={num_points}, noise_std={noise_std}, runs={n_runs})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(RESULTS_DIR, "running_time_vs_consent_noisy_sphere.png"), dpi=150
    )
    plt.close()


def plot_noisy_sphere_vs_dimension(
    dimensions,
    consent_probability=0.7,
    num_points=1000,
    radius=10,
    noise_std=0.5,
    center_bounds=(-10, 10),
    n_runs=10,
):
    """
    Runs sphere algorithms over a range of dimensions for points generated on a noisy sphere (disc in 2D),
    measuring both oracle calls and running times, then saves both corresponding plots.
    """
    incremental_calls = []
    decremental_calls = []
    incremental_times = []
    decremental_times = []

    pbar = tqdm(dimensions, desc="Running noisy sphere benchmarks vs dimension")
    for d in pbar:
        inc_calls_runs = []
        dec_calls_runs = []
        inc_time_runs = []
        dec_time_runs = []
        for run_idx in range(n_runs):
            pbar.set_description(
                f"Noisy sphere benchmarks vs dimension (d={d}, run {run_idx + 1}/{n_runs})"
            )
            # Generate points on a sphere with noise
            points = generate_noisy_sphere_points(
                d, num_points, radius, noise_std, center_bounds=center_bounds
            )
            points_with_consent = [
                (point, random.random() < consent_probability) for point in points
            ]

            # Run incremental algorithm
            oracle_inc = Oracle()
            start = time.perf_counter()
            incremental_distance_based(points_with_consent, oracle_inc)
            inc_time_runs.append(time.perf_counter() - start)
            inc_calls_runs.append(oracle_inc.get_call_count())

            # Run decremental algorithm
            oracle_dec = Oracle()
            start = time.perf_counter()
            decremental_distance_based(points_with_consent, oracle_dec)
            dec_time_runs.append(time.perf_counter() - start)
            dec_calls_runs.append(oracle_dec.get_call_count())

        incremental_calls.append(np.median(inc_calls_runs))
        decremental_calls.append(np.median(dec_calls_runs))
        incremental_times.append(np.median(inc_time_runs))
        decremental_times.append(np.median(dec_time_runs))

    # Plot 1: Oracle Calls vs Dimension (Noisy Sphere)
    plt.figure(figsize=(12, 8))
    plt.plot(
        dimensions, incremental_calls, marker="o", label="Incremental sphere algorithm"
    )
    plt.plot(
        dimensions, decremental_calls, marker="s", label="Decremental sphere algorithm"
    )
    plt.xlabel("Dimension (d)")
    plt.ylabel("Number of Consent Requests")
    plt.title(
        f"Oracle Calls vs Dimension (Noisy Sphere)\n(N={num_points}, p={consent_probability}, noise_std={noise_std}, runs={n_runs})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(RESULTS_DIR, "oracle_calls_vs_dimension_noisy_sphere.png"), dpi=150
    )
    plt.close()

    # Plot 2: Running Time vs Dimension (Noisy Sphere)
    plt.figure(figsize=(12, 8))
    plt.plot(
        dimensions, incremental_times, marker="o", label="Incremental sphere algorithm"
    )
    plt.plot(
        dimensions, decremental_times, marker="s", label="Decremental sphere algorithm"
    )
    plt.xlabel("Dimension (d)")
    plt.ylabel("Running Time (seconds)")
    plt.title(
        f"Running Time vs Dimension (Noisy Sphere)\n(N={num_points}, p={consent_probability}, noise_std={noise_std}, runs={n_runs})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(RESULTS_DIR, "running_time_vs_dimension_noisy_sphere.png"), dpi=150
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
    # Shape flags
    parser.add_argument("--box", action="store_true", help="Run box shape benchmarks")
    parser.add_argument(
        "--sphere", action="store_true", help="Run sphere shape benchmarks"
    )
    parser.add_argument(
        "--noisy-sphere", action="store_true", help="Run noisy sphere shape benchmarks"
    )
    # Plot type flags
    parser.add_argument(
        "--vs-points", action="store_true", help="Run plots vs number of points"
    )
    parser.add_argument(
        "--vs-consent", action="store_true", help="Run plots vs consent probability"
    )
    parser.add_argument(
        "--vs-dimension", action="store_true", help="Run plots vs dimension"
    )
    # Points range arguments
    parser.add_argument(
        "--points-start",
        type=int,
        default=100,
        help="Starting number of points (default: 100)",
    )
    parser.add_argument(
        "--points-end",
        type=int,
        default=100000,
        help="Ending number of points (exclusive) (default: 100000)",
    )
    parser.add_argument(
        "--points-step",
        type=int,
        default=1000,
        help="Step size for points range (default: 1000)",
    )
    # Consent probability range arguments
    parser.add_argument(
        "--consent-start",
        type=float,
        default=0.05,
        help="Starting consent probability (default: 0.05)",
    )
    parser.add_argument(
        "--consent-end",
        type=float,
        default=1.00,
        help="Ending consent probability (exclusive) (default: 1.00)",
    )
    parser.add_argument(
        "--consent-step",
        type=float,
        default=0.02,
        help="Step size for consent probability range (default: 0.02)",
    )
    # Dimension range arguments
    parser.add_argument(
        "--dim-start", type=int, default=3, help="Starting dimension (default: 3)"
    )
    parser.add_argument(
        "--dim-end",
        type=int,
        default=5,
        help="Ending dimension (inclusive) (default: 5)",
    )
    args = parser.parse_args()

    # Determine which shapes to run
    # If none of the shape flags are set, run all shapes
    run_all_shapes = not (args.box or args.sphere or args.noisy_sphere)
    run_box = args.box or run_all_shapes
    run_sphere = args.sphere or run_all_shapes
    run_noisy_sphere = args.noisy_sphere or run_all_shapes

    # Determine which plot types to run
    # If none of the plot flags are set, run all plot types
    run_all_plots = not (args.vs_points or args.vs_consent or args.vs_dimension)
    run_vs_points = args.vs_points or run_all_plots
    run_vs_consent = args.vs_consent or run_all_plots
    run_vs_dimension = args.vs_dimension or run_all_plots

    n_runs = args.runs
    n_dim = 3
    consent_probability = 0.7

    # --- Points on X-axis ---
    if run_vs_points:
        num_points_list = range(args.points_start, args.points_end, args.points_step)
        if run_box:
            logger.info(
                f"Running Box Benchmarks vs Number of Points (n_runs={n_runs})..."
            )
            plot_box_vs_points(
                n_dim, consent_probability, num_points_list, n_runs=n_runs
            )

        if run_sphere:
            logger.info(
                f"Running Sphere Benchmarks vs Number of Points (n_runs={n_runs})..."
            )
            plot_sphere_vs_points(
                n_dim, consent_probability, num_points_list, n_runs=n_runs
            )

        if run_noisy_sphere:
            logger.info(
                f"Running Noisy Sphere Benchmarks vs Number of Points (n_runs={n_runs})..."
            )
            plot_noisy_sphere_vs_points(
                n_dim, consent_probability, num_points_list, n_runs=n_runs
            )

    # --- Consent probability on X-axis ---
    if run_vs_consent:
        num_points_fixed = 500
        consent_prob_list = []
        curr = args.consent_start
        while curr < args.consent_end - 1e-9:
            consent_prob_list.append(round(curr, 4))
            curr += args.consent_step

        if run_box:
            logger.info(
                f"Running Box Benchmarks vs Consent Probability (n_runs={n_runs})..."
            )
            plot_box_vs_consent(
                n_dim, num_points_fixed, consent_prob_list, n_runs=n_runs
            )

        if run_sphere:
            logger.info(
                f"Running Sphere Benchmarks vs Consent Probability (n_runs={n_runs})..."
            )
            plot_sphere_vs_consent(
                n_dim, num_points_fixed, consent_prob_list, n_runs=n_runs
            )

        if run_noisy_sphere:
            logger.info(
                f"Running Noisy Sphere Benchmarks vs Consent Probability (n_runs={n_runs})..."
            )
            plot_noisy_sphere_vs_consent(
                n_dim, num_points_fixed, consent_prob_list, n_runs=n_runs
            )

    # --- Dimension on X-axis ---
    if run_vs_dimension:
        dimensions = list(range(args.dim_start, args.dim_end + 1))

        if run_box:
            logger.info(f"Running Box Benchmarks vs Dimension (n_runs={n_runs})...")
            plot_box_vs_dimension(
                dimensions, consent_probability, num_points=10000, n_runs=n_runs
            )

        if run_sphere:
            logger.info(f"Running Sphere Benchmarks vs Dimension (n_runs={n_runs})...")
            plot_sphere_vs_dimension(
                dimensions, consent_probability, num_points=10000, n_runs=n_runs
            )

        if run_noisy_sphere:
            logger.info(
                f"Running Noisy Sphere Benchmarks vs Dimension (n_runs={n_runs})..."
            )
            plot_noisy_sphere_vs_dimension(
                dimensions, consent_probability, num_points=10000, n_runs=n_runs
            )


if __name__ == "__main__":
    main()
