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


def plot_oracle_calls_vs_points(
    n_dim, consent_probability, num_points_list, low=-10, high=10
):
    """
    Draws a graph with Number of points on X-axis and Number of consent requests on Y-axis.
    """
    incremental_calls = []
    decremental_calls = []
    lower_bounds = []
    d_log_n_values = []

    for num_points in tqdm(num_points_list, desc="Processing box algorithms"):
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
        incremental_calls.append(oracle_inc.get_call_count())

        # Run decremental algorithm
        oracle_dec = Oracle()
        min_bounds_dec, max_bounds_dec = decremental_orthogonal(
            points_with_consent, oracle_dec
        )
        decremental_calls.append(oracle_dec.get_call_count())

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
        lower_bounds.append(lower_bound)

        if num_points > 0:
            d_log_n = n_dim * np.log(num_points)
        else:
            d_log_n = 0
        d_log_n_values.append(d_log_n)

    ratios = [inc / d for inc, d in zip(incremental_calls, d_log_n_values) if d > 0]
    scaling_constant = np.mean(ratios) if ratios else 1.0
    scaled_d_log_n_values = [scaling_constant * val for val in d_log_n_values]

    plt.figure(figsize=(12, 8))
    plt.plot(num_points_list, incremental_calls, label="Incremental box algorithm")
    plt.plot(num_points_list, decremental_calls, label="Decremental box algorithm")
    plt.plot(num_points_list, lower_bounds, label="Lower bound (points outside + 2*d)")
    plt.plot(
        num_points_list,
        scaled_d_log_n_values,
        label=f"d*log(n) scaled (C={scaling_constant:.2f})",
    )

    plt.xlabel("Number of Points")
    plt.ylabel("Number of Consent Requests")
    plt.title(
        f"Oracle Calls vs Number of Points (Box)\n(d={n_dim}, p={consent_probability})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "oracle_calls_vs_points_box.png"), dpi=150)
    plt.show()


def plot_oracle_calls_vs_points_sphere(
    n_dim, consent_probability, num_points_list, radius=10, center_bounds=(-10, 10)
):
    """
    Draws a graph for sphere algorithms.
    """
    incremental_calls = []
    decremental_calls = []
    lower_bounds = []
    factorial_bound_values = []
    p_bound_values = []

    for num_points in tqdm(num_points_list, desc="Processing sphere algorithms"):
        points = generate_in_sphere(n_dim, radius, num_points)
        points_with_consent = [
            (p, random.random() < consent_probability) for p in points
        ]

        oracle_inc = Oracle()
        center_inc, radius_inc = incremental_distance_based(
            points_with_consent, oracle_inc
        )
        incremental_calls.append(oracle_inc.get_call_count())

        oracle_dec = Oracle()
        center_dec, radius_dec = decremental_distance_based(
            points_with_consent, oracle_dec
        )
        decremental_calls.append(oracle_dec.get_call_count())

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

        lower_bounds.append(points_outside + n_dim + 1)
        factorial_bound_values.append(
            np.log(num_points) ** (n_dim + 1) if num_points > 1 else 0
        )
        p_bound_values.append(
            (n_dim + 1) / (consent_probability ** (n_dim + 1))
            if consent_probability > 0
            else float("inf")
        )

    fact_ratios = [
        inc / f for inc, f in zip(incremental_calls, factorial_bound_values) if f > 0
    ]
    fact_scaling = np.mean(fact_ratios) if fact_ratios else 1.0
    scaled_fact = [fact_scaling * v for v in factorial_bound_values]

    p_ratios = [
        dec / p
        for dec, p in zip(decremental_calls, p_bound_values)
        if p > 0 and not np.isinf(p)
    ]
    p_scaling = np.mean(p_ratios) if p_ratios else 1.0
    scaled_p = [p_scaling * v for v in p_bound_values]

    plt.figure(figsize=(12, 8))
    plt.plot(num_points_list, incremental_calls, label="Incremental sphere algorithm")
    plt.plot(num_points_list, decremental_calls, label="Decremental sphere algorithm")
    plt.plot(
        num_points_list, lower_bounds, label="Lower bound (points outside + d + 1)"
    )
    plt.plot(
        num_points_list,
        scaled_fact,
        label=f"(d+1)! * ln^(d+1)(n) scaled (C={fact_scaling:.2f})",
    )
    plt.plot(
        num_points_list, scaled_p, label=f"(d+1)/(p^(d+1)) scaled (C={p_scaling:.2f})"
    )

    plt.xlabel("Number of Points")
    plt.ylabel("Number of Consent Requests")
    plt.title(
        f"Oracle Calls vs Number of Points (Sphere)\n(d={n_dim}, p={consent_probability})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "oracle_calls_vs_points_sphere.png"), dpi=150)
    plt.show()


def plot_running_time_box_algorithms(
    n_dim, consent_probability, num_points_list, low=-10, high=10
):
    """
    Draws a graph for running time of box algorithms.
    """
    incremental_times = []
    decremental_times = []

    for num_points in tqdm(
        num_points_list, desc="Measuring box algorithm running times"
    ):
        points = generate_in_box(n_dim, low=low, high=high, count=num_points)
        points_with_consent = [
            (p, random.random() < consent_probability) for p in points
        ]

        oracle_inc = Oracle()
        start = time.perf_counter()
        incremental_orthogonal(points_with_consent, oracle_inc)
        incremental_times.append(time.perf_counter() - start)

        oracle_dec = Oracle()
        start = time.perf_counter()
        decremental_orthogonal(points_with_consent, oracle_dec)
        decremental_times.append(time.perf_counter() - start)

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
    plt.show()


def plot_running_time_sphere_algorithms(
    n_dim, consent_probability, num_points_list, radius=10
):
    """
    Draws a graph for running time of sphere algorithms.
    """
    incremental_times = []
    decremental_times = []

    for num_points in tqdm(
        num_points_list, desc="Measuring sphere algorithm running times"
    ):
        points = generate_in_sphere(n_dim, radius, num_points)
        points_with_consent = [
            (p, random.random() < consent_probability) for p in points
        ]

        oracle_inc = Oracle()
        start = time.perf_counter()
        incremental_distance_based(points_with_consent, oracle_inc)
        incremental_times.append(time.perf_counter() - start)

        oracle_dec = Oracle()
        start = time.perf_counter()
        decremental_distance_based(points_with_consent, oracle_dec)
        decremental_times.append(time.perf_counter() - start)

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
    plt.show()


def main():
    n_dim = 3
    consent_probability = 0.3
    num_points_list = range(100, 2000, 10)

    logger.info("Running Box Oracle Calls Benchmark...")
    plot_oracle_calls_vs_points(n_dim, consent_probability, num_points_list)

    logger.info("Running Sphere Oracle Calls Benchmark...")
    plot_oracle_calls_vs_points_sphere(n_dim, consent_probability, num_points_list)

    logger.info("Running Box Running Time Benchmark...")
    plot_running_time_box_algorithms(n_dim, consent_probability, num_points_list)

    logger.info("Running Sphere Running Time Benchmark...")
    plot_running_time_sphere_algorithms(n_dim, consent_probability, num_points_list)


if __name__ == "__main__":
    main()
