import logging
import random

import numpy as np

from db_consents import *

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


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

    oracle3 = Oracle()
    decremental_sphere = decremental_distance_based(sphere_points, oracle3)

    oracle4 = Oracle()
    incremental_sphere = incremental_distance_based(sphere_points, oracle4)
    assert np.allclose(decremental_sphere[0], incremental_sphere[0]) and np.allclose(
        decremental_sphere[1], incremental_sphere[1]
    )

    oracle5 = Oracle()
    decremental_disc = decremental_distance_based(noisy_disc_points, oracle5)

    oracle6 = Oracle()
    incremental_disc = incremental_distance_based(noisy_disc_points, oracle6)
    assert np.allclose(decremental_disc[0], incremental_disc[0]) and np.allclose(
        decremental_disc[1], incremental_disc[1]
    )


def test_incremental_distance_based_helper(test_sphere_points):
    oracle = Oracle()
    query_center, query_radius = incremental_distance_based(test_sphere_points, oracle)
    query_radius_sq = query_radius**2
    for tup in test_sphere_points:
        point, ground_truth = tup[0], tup[1]
        if ground_truth:
            if not is_point_in_sphere(point, query_center, query_radius_sq):
                logger.error(
                    f"Point outside query sphere! Center: {query_center}, Radius: {query_radius}"
                )
                logger.error(
                    f"Point: {point}, Distance: {np.linalg.norm(point - query_center)}"
                )
                assert False


def test_incremental_distance_based(sphere_points):
    for i in range(10):
        test_incremental_distance_based_helper(
            [([0, 1], 1), ([1, 0], 1), ([-1, 0], 1), ([1, -1], 1)]
        )
        test_incremental_distance_based_helper(sphere_points)


def do_tests(repetitions=10):
    logger.info("TESTING")

    for i in range(repetitions):
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

    logger.info("TESTS PASSED SUCCESSFULY!")


def main():
    do_tests()


if __name__ == "__main__":
    main()
