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
    assert np.allclose(decremental_box[0], incremental_box[0], atol=1e-9) and np.allclose(
        decremental_box[1], incremental_box[1], atol=1e-9
    )

    oracle3 = Oracle()
    decremental_sphere = decremental_distance_based(sphere_points, oracle3)

    oracle4 = Oracle()
    incremental_sphere = incremental_distance_based(sphere_points, oracle4, debug=True)
    assert np.allclose(decremental_sphere[0], incremental_sphere[0], atol=1e-9) and np.allclose(
        decremental_sphere[1], incremental_sphere[1], atol=1e-9
    )

    oracle5 = Oracle()
    decremental_disc = decremental_distance_based(noisy_disc_points, oracle5)

    oracle6 = Oracle()
    incremental_disc = incremental_distance_based(noisy_disc_points, oracle6, debug=True)
    assert np.allclose(decremental_disc[0], incremental_disc[0], atol=1e-9) and np.allclose(
        decremental_disc[1], incremental_disc[1], atol=1e-9
    )


def test_incremental_distance_based_helper(test_sphere_points):
    oracle = Oracle()
    query_center, query_radius = incremental_distance_based(test_sphere_points, oracle, debug=True)
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


def test_welzl_vs_miniball():
    logger.info("Running Welzl vs Miniball sanity checks...")
    import miniball
    for _ in range(50):
        n_dim = random.randint(2, 6)
        num_samples = random.randint(10, 100)
        points = generate_in_sphere(n_dim, 10.0, num_samples)
        
        # Test: all points consenting
        points_all_consenting = [(p, True) for p in points]
        
        center_welzl, radius_sq_welzl = welzl(
            points_all_consenting, [], Oracle(), n_dim, len(points_all_consenting), debug=True
        )
        
        center_miniball, radius_sq_miniball = miniball.get_bounding_ball(np.array(points))
        
        assert np.allclose(center_welzl, center_miniball, atol=1e-8), (
            f"Center mismatch. Welzl: {center_welzl}, Miniball: {center_miniball}"
        )
        assert np.isclose(radius_sq_welzl, radius_sq_miniball, atol=1e-8), (
            f"Radius squared mismatch. Welzl: {radius_sq_welzl}, Miniball: {radius_sq_miniball}"
        )


def assert_cpp_vs_python_match(points_with_consent, n_dim):
    import db_consents
    # Run with Python backend
    db_consents.USE_CPP_BACKEND = False
    oracle_py = Oracle()
    res_py = welzl(points_with_consent, [], oracle_py, n_dim, len(points_with_consent))
    
    # Run with C++ backend
    db_consents.USE_CPP_BACKEND = True
    oracle_cpp = Oracle()
    res_cpp = welzl(points_with_consent, [], oracle_cpp, n_dim, len(points_with_consent))
    
    # Compare results
    if res_py[0] is None:
        assert res_cpp[0] is None, f"Python found None, but C++ found a sphere: {res_cpp[0]} (radius_sq={res_cpp[1]})"
    else:
        assert res_cpp[0] is not None, f"Python found a sphere: {res_py[0]}, but C++ found None"
        assert np.allclose(res_py[0], res_cpp[0], atol=1e-10), (
            f"Center mismatch. Python: {res_py[0]}, C++: {res_cpp[0]}"
        )
        assert np.isclose(res_py[1], res_cpp[1], atol=1e-10), (
            f"Radius squared mismatch. Python: {res_py[1]}, C++: {res_cpp[1]}"
        )
        
    # Verify oracle call counts match
    assert oracle_py.get_call_count() == oracle_cpp.get_call_count(), (
        f"Oracle call count mismatch! Python: {oracle_py.get_call_count()}, C++: {oracle_cpp.get_call_count()}"
    )


def test_cpp_vs_python_welzl():
    logger.info("Running Python vs C++ Welzl comparison tests...")
    import db_consents
    original_backend = db_consents.USE_CPP_BACKEND
    try:
        # Scenario 1: Edge cases (empty, single point, small points)
        logger.info("  Scenario 1: Edge cases...")
        for n_dim in [2, 3, 5]:
            # Empty points
            assert_cpp_vs_python_match([], n_dim)
            # Single point (consenting / non-consenting)
            assert_cpp_vs_python_match([([0.0]*n_dim, True)], n_dim)
            assert_cpp_vs_python_match([([0.0]*n_dim, False)], n_dim)
            # Dual points
            assert_cpp_vs_python_match([([0.0]*n_dim, True), ([1.0]*n_dim, True)], n_dim)
            assert_cpp_vs_python_match([([0.0]*n_dim, True), ([1.0]*n_dim, False)], n_dim)

        # Scenario 2: Varying consent probabilities (0.0, 0.3, 0.7, 1.0)
        logger.info("  Scenario 2: Varying consent probabilities...")
        for p in [0.0, 0.3, 0.7, 1.0]:
            for _ in range(15):
                n_dim = random.randint(2, 6)
                n_points = random.randint(10, 200)
                points = generate_in_sphere(n_dim, 10.0, n_points)
                points_with_consent = [(pt, random.random() < p) for pt in points]
                assert_cpp_vs_python_match(points_with_consent, n_dim)

        # Scenario 3: Different point distributions (Box, Sphere, Noisy Sphere)
        logger.info("  Scenario 3: Different distributions...")
        for dist in ["box", "sphere", "noisy_sphere"]:
            for _ in range(15):
                n_dim = random.randint(2, 5)
                n_points = random.randint(10, 200)
                if dist == "box":
                    points = generate_in_box(n_dim, -10.0, 10.0, n_points)
                elif dist == "sphere":
                    points = generate_in_sphere(n_dim, 10.0, n_points)
                else:
                    points = generate_noisy_sphere_points(n_dim, n_points, 10.0, 0.5)
                
                points_with_consent = [(pt, random.random() < 0.7) for pt in points]
                assert_cpp_vs_python_match(points_with_consent, n_dim)

        # Scenario 4: Larger datasets (up to 3000 points) and higher dimensions
        logger.info("  Scenario 4: Larger datasets and higher dimensions...")
        for _ in range(10):
            n_dim = random.randint(3, 6)
            n_points = random.randint(500, 3000)
            points = generate_in_sphere(n_dim, 10.0, n_points)
            points_with_consent = [(pt, random.random() < 0.7) for pt in points]
            assert_cpp_vs_python_match(points_with_consent, n_dim)

    finally:
        db_consents.USE_CPP_BACKEND = original_backend
    logger.info("Python vs C++ Welzl comparison tests passed successfully!")


def do_tests(repetitions=10):
    logger.info("TESTING")
    test_welzl_vs_miniball()
    test_cpp_vs_python_welzl()

    logger.info(f"Running {repetitions} repetitions of Incremental vs Decremental and correctness tests...")
    for i in range(repetitions):
        logger.info(f"  Repetition {i+1}/{repetitions}...")
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
