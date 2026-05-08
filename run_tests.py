import random
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from db_consents import *


def test_decremental_vs_incremental(box_points, sphere_points, noisy_disc_points):
    # Create oracle instances for each algorithm run
    oracle1 = Oracle()
    decremental_box = decremental_orthogonal(box_points, oracle1)

    oracle2 = Oracle()
    incremental_box = incremental_orthogonal(box_points, oracle2)
    # Make sure the incremental and decremental algorithms return the same result
    assert np.allclose(decremental_box[0], incremental_box[0]) and np.allclose(
        decremental_box[1], incremental_box[1]
    )

    # print(f"decremental_box: {decremental_box}")
    # print(f"Oracle calls for decremental_orthogonal: {oracle1.get_call_count()}")
    # print(f"Oracle calls for incremental_orthogonal: {oracle2.get_call_count()}")

    oracle3 = Oracle()
    decremental_sphere = decremental_distance_based(sphere_points, oracle3)

    oracle4 = Oracle()
    incremental_sphere = incremental_distance_based(sphere_points, oracle4)
    assert np.allclose(decremental_sphere[0], incremental_sphere[0]) and np.allclose(
        decremental_sphere[1], incremental_sphere[1]
    )

    # print(f"decremental_sphere: {decremental_sphere}")
    # print(f"Oracle calls for decremental_distance_based: {oracle3.get_call_count()}")
    # print(f"Oracle calls for incremental_distance_based: {oracle4.get_call_count()}")

    oracle5 = Oracle()
    decremental_disc = decremental_distance_based(noisy_disc_points, oracle5)

    oracle6 = Oracle()
    incremental_disc = incremental_distance_based(noisy_disc_points, oracle6)
    assert np.allclose(decremental_disc[0], incremental_disc[0]) and np.allclose(
        decremental_disc[1], incremental_disc[1]
    )

    # print(f"decremental_disc: {decremental_disc}")
    # print(f"Oracle calls for decremental_distance_based (noisy): {oracle5.get_call_count()}")
    # print(f"Oracle calls for incremental_distance_based (noisy): {oracle6.get_call_count()}")


def test_incremental_distance_based_helper(test_sphere_points):
    oracle = Oracle()
    query_center, query_radius = incremental_distance_based(test_sphere_points, oracle)
    query_radius_sq = query_radius**2
    for tup in test_sphere_points:
        point, ground_truth = tup[0], tup[1]
        if ground_truth:
            # assert is_point_in_sphere(point, query_center, query_radius_sq), point
            if not is_point_in_sphere(point, query_center, query_radius_sq):
                print(query_center, query_radius)
                print("point", point)
                print("dist", np.linalg.norm(point - query_center))
                print("radius", query_radius)
                assert False


def test_incremental_distance_based(sphere_points):
    for i in range(10):
        test_incremental_distance_based_helper(
            [([0, 1], 1), ([1, 0], 1), ([-1, 0], 1), ([1, -1], 1)]
        )
        test_incremental_distance_based_helper(sphere_points)


def plot_oracle_calls_vs_points(
    n_dim, consent_probability, num_points_list, low=-10, high=10
):
    """
    Draws a graph with Number of points on X-axis and Number of consent requests on Y-axis.

    Args:
        n_dim (int): Fixed dimension
        consent_probability (float): Fixed consent probability (0-1)
        num_points_list (list): List of number of points to test
        low (float): Lower bound for box generation
        high (float): Upper bound for box generation
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

        # Calculate lower bound: number of points outside the box + 2 * dimension
        # Use the result from one of the algorithms (they should be the same)
        min_bounds = min_bounds_inc if len(min_bounds_inc) > 0 else np.array([])
        max_bounds = max_bounds_inc if len(max_bounds_inc) > 0 else np.array([])

        points_outside = 0
        if len(min_bounds) > 0 and len(max_bounds) > 0:
            for point, _ in points_with_consent:
                if not is_point_in_box(point, min_bounds, max_bounds):
                    points_outside += 1
                    assert not _
        else:
            # If box is empty, all points are outside
            points_outside = num_points

        lower_bound = points_outside + 2 * n_dim
        lower_bounds.append(lower_bound)

        # Calculate d * log(n)
        if num_points > 0:
            d_log_n = n_dim * np.log(num_points)
        else:
            d_log_n = 0
        d_log_n_values.append(d_log_n)

    # Calculate scaling constant to match incremental algorithm to d*log(n)
    # Find the average ratio of incremental_calls to d_log_n_values
    ratios = []
    for inc_calls, d_log_n in zip(incremental_calls, d_log_n_values):
        if d_log_n > 0:
            ratios.append(inc_calls / d_log_n)

    if ratios:
        scaling_constant = np.mean(ratios)
    else:
        scaling_constant = 1.0

    # Scale d*log(n) values by the constant
    scaled_d_log_n_values = [scaling_constant * val for val in d_log_n_values]

    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(
        num_points_list,
        incremental_calls,
        label="Incremental box algorithm",
        linewidth=2,
        markersize=6,
    )
    plt.plot(
        num_points_list,
        decremental_calls,
        label="Decremental box algorithm",
        linewidth=2,
        markersize=6,
    )
    plt.plot(
        num_points_list,
        lower_bounds,
        label=f"Lower bound (points outside + 2*d)",
        linewidth=2,
        markersize=6,
    )
    plt.plot(
        num_points_list,
        scaled_d_log_n_values,
        label=f"d*log(n) scaled (C={scaling_constant:.2f}, d={n_dim})",
        linewidth=2,
        markersize=6,
    )

    plt.xlabel("Number of Points", fontsize=12)
    plt.ylabel("Number of Consent Requests", fontsize=12)
    plt.title(
        f"Oracle Calls vs Number of Points\n(Dimension={n_dim}, Consent Probability={consent_probability})",
        fontsize=14,
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plt.savefig("oracle_calls_vs_points_box.png", dpi=150)
    print(f"Graph saved as 'oracle_calls_vs_points_box.png'")

    # Also show the plot
    plt.show()

    return {
        "num_points": num_points_list,
        "incremental_calls": incremental_calls,
        "decremental_calls": decremental_calls,
        "lower_bounds": lower_bounds,
        "d_log_n": d_log_n_values,
        "scaled_d_log_n": scaled_d_log_n_values,
        "scaling_constant": scaling_constant,
    }


def do_tests(repetitions=10):
    print("TESTING")

    for i in range(10):
        n_dim = 3
        num_samples = 50
        gen_radius = half_width = 10

        # Generate points
        box_points = generate_in_box(
            n_dim, low=-half_width, high=half_width, count=num_samples
        )
        sphere_points = generate_in_sphere(n_dim, gen_radius, num_samples)

        center_bounds = (-10, 10)
        noise_std = 0.5
        noisy_disc_points = generate_noisy_sphere_points(
            n_dim, num_samples, gen_radius, noise_std, center_bounds=center_bounds
        )

        # Add consent values
        consent_probability = 0.7
        box_points = [
            (point, random.random() < consent_probability) for point in box_points
        ]
        sphere_points = [
            (point, random.random() < consent_probability) for point in sphere_points
        ]
        noisy_disc_points = [
            (point, random.random() < consent_probability)
            for point in noisy_disc_points
        ]

        test_decremental_vs_incremental(box_points, sphere_points, noisy_disc_points)
        test_incremental_distance_based(sphere_points)

    print("TESTS PASSED SUCCESSFULY!")


def main():
    do_tests()

    # print("Plotting Graph 1")
    # plot_graph_1()

    print("Plotting Graph 2")
    plot_graph_2()

    print("Plotting Graph 3")
    plot_graph_3()

    print("Plotting Graph 4")
    plot_graph_4()


def plot_graph_1():
    n_dim = 4
    consent_probability = 0.3
    # Test with different numbers of points
    num_points_list = range(100, 10000, 100)

    results = plot_oracle_calls_vs_points(n_dim, consent_probability, num_points_list)
    return results


def plot_oracle_calls_vs_points_sphere(
    n_dim, consent_probability, num_points_list, radius=10, center_bounds=(-10, 10)
):
    """
    Draws a graph for sphere algorithms with Number of points on X-axis and Number of consent requests on Y-axis.

    Args:
        n_dim (int): Fixed dimension
        consent_probability (float): Fixed consent probability (0-1)
        num_points_list (list): List of number of points to test
        radius (float): Radius for sphere generation
        center_bounds (tuple): Bounds for sphere center generation
    """
    incremental_calls = []
    decremental_calls = []
    lower_bounds = []  # points outside + d + 1
    factorial_bound_values = []  # (d+1)! * ln^(d+1)(n)
    p_bound_values = []  # (d+1)/(p^(d+1))

    for num_points in tqdm(num_points_list, desc="Processing sphere algorithms"):
        # Generate random points in a sphere
        points = generate_in_sphere(n_dim, radius, num_points)
        # Add consent values
        points_with_consent = [
            (point, random.random() < consent_probability) for point in points
        ]

        # Run incremental sphere algorithm
        oracle_inc = Oracle()
        center_inc, radius_inc = incremental_distance_based(
            points_with_consent, oracle_inc
        )
        incremental_calls.append(oracle_inc.get_call_count())

        # Run decremental sphere algorithm
        oracle_dec = Oracle()
        center_dec, radius_dec = decremental_distance_based(
            points_with_consent, oracle_dec
        )
        decremental_calls.append(oracle_dec.get_call_count())

        # Calculate lower bound: number of points outside the disc + d + 1
        # Use the result from one of the algorithms (they should be the same)
        if center_inc is not None and radius_inc is not None and radius_inc > 0:
            center = np.array(center_inc)
            radius_sq = radius_inc**2
            points_outside = 0
            for point, _ in points_with_consent:
                if not is_point_in_sphere(point, center, radius_sq):
                    points_outside += 1
        else:
            # If sphere is empty/invalid, all points are outside
            points_outside = num_points

        lower_bound = points_outside + n_dim + 1
        lower_bounds.append(lower_bound)

        # Calculate (d+1)! * ln^(d+1)(n)
        if num_points > 1:  # ln(n) requires n > 1
            # No need to multiply by math.factorial(n_dim + 1) since we do scaling anyways
            factorial_bound = np.log(num_points) ** (n_dim + 1)
        else:
            factorial_bound = 0
        factorial_bound_values.append(factorial_bound)

        # Calculate (d+1)/(p^(d+1))
        if consent_probability > 0:
            p_bound = (n_dim + 1) / (consent_probability ** (n_dim + 1))
        else:
            p_bound = float("inf") if n_dim + 1 > 0 else 0
        p_bound_values.append(p_bound)

    # Calculate scaling constant to match incremental algorithm to factorial bound
    # Find the average ratio of incremental_calls to factorial_bound_values
    factorial_ratios = []
    for inc_calls, fact_bound in zip(incremental_calls, factorial_bound_values):
        if fact_bound > 0:
            factorial_ratios.append(inc_calls / fact_bound)

    if factorial_ratios:
        factorial_scaling_constant = np.mean(factorial_ratios)
    else:
        factorial_scaling_constant = 1.0

    # Scale factorial_bound values by the constant
    scaled_factorial_bound_values = [
        factorial_scaling_constant * val for val in factorial_bound_values
    ]

    # Calculate scaling constant to match decremental algorithm to p_bound
    # Find the average ratio of decremental_calls to p_bound_values
    p_ratios = []
    for dec_calls, p_bound in zip(decremental_calls, p_bound_values):
        if p_bound > 0 and not np.isinf(p_bound):
            p_ratios.append(dec_calls / p_bound)

    if p_ratios:
        p_scaling_constant = np.mean(p_ratios)
    else:
        p_scaling_constant = 1.0

    # Scale p_bound values by the constant
    scaled_p_bound_values = [p_scaling_constant * val for val in p_bound_values]

    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(
        num_points_list,
        incremental_calls,
        label="Incremental sphere algorithm",
        linewidth=2,
        markersize=6,
    )
    plt.plot(
        num_points_list,
        decremental_calls,
        label="Decremental sphere algorithm",
        linewidth=2,
        markersize=6,
    )
    plt.plot(
        num_points_list,
        lower_bounds,
        label="Lower bound (points outside + d + 1)",
        linewidth=2,
        markersize=6,
    )
    plt.plot(
        num_points_list,
        scaled_factorial_bound_values,
        label=f"(d+1)! * ln^(d+1)(n) scaled (C={factorial_scaling_constant:.2f}, d={n_dim})",
        linewidth=2,
        markersize=6,
    )
    plt.plot(
        num_points_list,
        scaled_p_bound_values,
        label=f"(d+1)/(p^(d+1)) scaled (C={p_scaling_constant:.2f}, d={n_dim}, p={consent_probability})",
        linewidth=2,
        markersize=6,
    )

    plt.xlabel("Number of Points", fontsize=12)
    plt.ylabel("Number of Consent Requests", fontsize=12)
    plt.title(
        f"Oracle Calls vs Number of Points (Sphere Algorithms)\n(Dimension={n_dim}, Consent Probability={consent_probability})",
        fontsize=14,
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plt.savefig("oracle_calls_vs_points_sphere.png", dpi=150)
    print(f"Graph saved as 'oracle_calls_vs_points_sphere.png'")

    # Also show the plot
    plt.show()

    return {
        "num_points": num_points_list,
        "incremental_calls": incremental_calls,
        "decremental_calls": decremental_calls,
        "lower_bounds": lower_bounds,
        "factorial_bound": factorial_bound_values,
        "scaled_factorial_bound": scaled_factorial_bound_values,
        "factorial_scaling_constant": factorial_scaling_constant,
        "p_bound": p_bound_values,
        "scaled_p_bound": scaled_p_bound_values,
        "p_scaling_constant": p_scaling_constant,
    }


def plot_graph_2():
    n_dim = 4
    consent_probability = 0.3
    # Test with different numbers of points
    num_points_list = range(100, 1000, 10)

    results = plot_oracle_calls_vs_points_sphere(
        n_dim, consent_probability, num_points_list
    )
    return results


def plot_running_time_box_algorithms(
    n_dim, consent_probability, num_points_list, low=-10, high=10
):
    """
    Draws a graph with Number of points on X-axis and Running time on Y-axis for box algorithms.

    Args:
        n_dim (int): Fixed dimension
        consent_probability (float): Fixed consent probability (0-1)
        num_points_list (list): List of number of points to test
        low (float): Lower bound for box generation
        high (float): Upper bound for box generation
    """
    incremental_times = []
    decremental_times = []

    for num_points in tqdm(
        num_points_list, desc="Measuring box algorithm running times"
    ):
        # Generate random points in a box
        points = generate_in_box(n_dim, low=low, high=high, count=num_points)
        # Add consent values
        points_with_consent = [
            (point, random.random() < consent_probability) for point in points
        ]

        # Measure incremental algorithm running time
        oracle_inc = Oracle()
        start_time = time.perf_counter()
        incremental_orthogonal(points_with_consent, oracle_inc)
        end_time = time.perf_counter()
        incremental_times.append(end_time - start_time)

        # Measure decremental algorithm running time
        oracle_dec = Oracle()
        start_time = time.perf_counter()
        decremental_orthogonal(points_with_consent, oracle_dec)
        end_time = time.perf_counter()
        decremental_times.append(end_time - start_time)

    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(
        num_points_list,
        incremental_times,
        label="Incremental box algorithm",
        linewidth=2,
        markersize=6,
    )
    plt.plot(
        num_points_list,
        decremental_times,
        label="Decremental box algorithm",
        linewidth=2,
        markersize=6,
    )

    plt.xlabel("Number of Points", fontsize=12)
    plt.ylabel("Running Time (seconds)", fontsize=12)
    plt.title(
        f"Running Time vs Number of Points (Box Algorithms)\n(Dimension={n_dim}, Consent Probability={consent_probability})",
        fontsize=14,
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plt.savefig("running_time_box_algorithms.png", dpi=150)
    print(f"Graph saved as 'running_time_box_algorithms.png'")

    # Also show the plot
    plt.show()

    return {
        "num_points": num_points_list,
        "incremental_times": incremental_times,
        "decremental_times": decremental_times,
    }


def plot_graph_3():
    n_dim = 4
    consent_probability = 0.3
    # Test with different numbers of points
    num_points_list = range(100, 10000, 100)

    results = plot_running_time_box_algorithms(
        n_dim, consent_probability, num_points_list
    )
    return results


def plot_running_time_sphere_algorithms(
    n_dim, consent_probability, num_points_list, radius=10
):
    """
    Draws a graph with Number of points on X-axis and Running time on Y-axis for sphere algorithms.

    Args:
        n_dim (int): Fixed dimension
        consent_probability (float): Fixed consent probability (0-1)
        num_points_list (list): List of number of points to test
        radius (float): Radius for sphere generation
    """
    incremental_times = []
    decremental_times = []

    for num_points in tqdm(
        num_points_list, desc="Measuring sphere algorithm running times"
    ):
        # Generate random points in a sphere
        points = generate_in_sphere(n_dim, radius, num_points)
        # Add consent values
        points_with_consent = [
            (point, random.random() < consent_probability) for point in points
        ]

        # Measure incremental algorithm running time
        oracle_inc = Oracle()
        start_time = time.perf_counter()
        incremental_distance_based(points_with_consent, oracle_inc)
        end_time = time.perf_counter()
        incremental_times.append(end_time - start_time)

        # Measure decremental algorithm running time
        oracle_dec = Oracle()
        start_time = time.perf_counter()
        decremental_distance_based(points_with_consent, oracle_dec)
        end_time = time.perf_counter()
        decremental_times.append(end_time - start_time)

    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(
        num_points_list,
        incremental_times,
        label="Incremental sphere algorithm",
        linewidth=2,
        markersize=6,
    )
    plt.plot(
        num_points_list,
        decremental_times,
        label="Decremental sphere algorithm",
        linewidth=2,
        markersize=6,
    )

    plt.xlabel("Number of Points", fontsize=12)
    plt.ylabel("Running Time (seconds)", fontsize=12)
    plt.title(
        f"Running Time vs Number of Points (Sphere Algorithms)\n(Dimension={n_dim}, Consent Probability={consent_probability})",
        fontsize=14,
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plt.savefig("running_time_sphere_algorithms.png", dpi=150)
    print("Graph saved as 'running_time_sphere_algorithms.png'")

    # Also show the plot
    plt.show()

    return {
        "num_points": num_points_list,
        "incremental_times": incremental_times,
        "decremental_times": decremental_times,
    }


def plot_graph_4():
    n_dim = 3
    consent_probability = 0.3
    # Test with different numbers of points
    num_points_list = range(100, 1000, 10)

    results = plot_running_time_sphere_algorithms(
        n_dim, consent_probability, num_points_list
    )
    return results


if __name__ == "__main__":
    main()
