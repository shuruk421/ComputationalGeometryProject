import argparse
import time
import random
import sys
import numpy as np

# Import the geometry module
from db_consents import (
    incremental_distance_based,
    decremental_distance_based,
    generate_in_sphere,
    Oracle
)
import db_consents

def profile():
    parser = argparse.ArgumentParser(description="Profile incremental and decremental distance-based algorithms.")
    parser.add_argument(
        "--algo",
        choices=["incremental", "decremental", "both"],
        default="both",
        help="Algorithm to profile (default: both)"
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=100000,
        help="Number of points to generate (default: 100000)"
    )
    parser.add_argument(
        "--backend",
        choices=["python", "cpp", "both"],
        default="both",
        help="Execution backend (default: both)"
    )
    args = parser.parse_args()

    n_dim = 3
    num_samples = args.n_points
    gen_radius = 10.0
    consent_probability = 0.7

    print("Generating points...")
    # Seed generation once so points are consistent
    random.seed(42)
    np.random.seed(42)
    sphere_points = generate_in_sphere(n_dim, gen_radius, num_samples)
    points_with_consent = [
        (point, random.random() < consent_probability) for point in sphere_points
    ]

    algos = ["incremental", "decremental"] if args.algo == "both" else [args.algo]
    backends = ["python", "cpp"] if args.backend == "both" else [args.backend]

    results = {}

    for algo in algos:
        results[algo] = {}
        for backend in backends:
            db_consents.USE_CPP_BACKEND = (backend == "cpp")
            oracle = Oracle()
            
            print(f"Running {algo}_distance_based with {num_samples} points using {backend.upper()} backend...")
            
            # Clone list to ensure exact same inputs for side-by-side comparison
            run_points = list(points_with_consent)
            
            # Reset random seeds so the shuffle order inside the algorithms is identical
            random.seed(42)
            np.random.seed(42)
            
            start_time = time.perf_counter()
            if algo == "incremental":
                center, radius = incremental_distance_based(run_points, oracle)
            else:
                center, radius = decremental_distance_based(run_points, oracle)
            elapsed = time.perf_counter() - start_time
            
            results[algo][backend] = {
                "elapsed": elapsed,
                "center": center,
                "radius": radius,
                "oracle_calls": oracle.get_call_count()
            }

    # Print results
    print("=" * 60)
    print(f"PROFILING RESULTS (Points: {num_samples})")
    print("=" * 60)
    for algo in algos:
        print(f"\n===== Algorithm: {algo.capitalize()} =====")
        for backend in ["python", "cpp"]:
            if backend in results[algo]:
                res = results[algo][backend]
                print(f"--- {backend.upper()} Backend ---")
                print(f"Execution Time:      {res['elapsed']:.6f} seconds")
                print(f"Found Sphere Center:  {res['center']}")
                print(f"Found Sphere Radius:  {res['radius']:.6f}")
                print(f"Oracle Call Count:   {res['oracle_calls']}")
                print("-" * 60)

        if "cpp" in results[algo] and "python" in results[algo]:
            elapsed_py = results[algo]["python"]["elapsed"]
            elapsed_cpp = results[algo]["cpp"]["elapsed"]
            speedup = elapsed_py / elapsed_cpp if elapsed_cpp > 0 else 0
            print(f"Speedup: {speedup:.2f}x faster with C++ backend")
            print("=" * 60)

    if len(algos) > 1 and "cpp" in backends and "python" in backends:
        print("\n" + "=" * 65)
        print("SUMMARY SPEEDUP COMPARISON FOR 100K POINTS:")
        print("-" * 65)
        print(f"{'Algorithm':<15} | {'Python Time (s)':<15} | {'C++ Time (s)':<15} | {'Speedup':<12}")
        print("-" * 65)
        for algo in algos:
            t_py = results[algo]["python"]["elapsed"]
            t_cpp = results[algo]["cpp"]["elapsed"]
            speedup = t_py / t_cpp if t_cpp > 0 else 0
            print(f"{algo.capitalize():<15} | {t_py:<15.6f} | {t_cpp:<15.6f} | {speedup:.2f}x")
        print("=" * 65)

if __name__ == "__main__":
    profile()
