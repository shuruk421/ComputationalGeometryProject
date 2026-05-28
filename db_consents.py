import math
import random
from typing import Any, List, Optional, Tuple
import sys
import os

import numpy as np


# Tolerance constants for geometric checks
GEOMETRY_TOLERANCE = 1e-10
DEBUG_BOUNDARY_TOLERANCE = 1e-7

USE_CPP_BACKEND = False
_welzl_module = None

def import_welzl_cpp():
    global _welzl_module
    if _welzl_module is None:
        try:
            dir_path = os.path.dirname(os.path.abspath(__file__))
            cpp_dir = os.path.join(dir_path, "cpp_welzl")
            if cpp_dir not in sys.path:
                sys.path.append(cpp_dir)
            import welzl_cpp_module
            _welzl_module = welzl_cpp_module
        except Exception as e:
            # Fall back to False if loading fails
            _welzl_module = False
    return _welzl_module




def generate_in_box(n, low=0, high=1, count=1):
    """
    Generates 'count' random points of dimension 'n' within a box defined by [low, high].

    Args:
        n (int): Dimension of the space.
        low (float): Lower bound of the box.
        high (float): Upper bound of the box.
        count (int): Number of points to generate.

    Returns:
        list: List of lists containing coordinates.
    """
    # Uniform distribution for each coordinate independently
    return np.random.uniform(low, high, size=(count, n)).tolist()


def generate_in_sphere(n, radius, count):
    """
    Generates 'count' random points of dimension 'n' uniformly distributed inside a sphere.

    Args:
        n (int): Dimension of the space.
        radius (float): Radius of the sphere.
        count (int): Number of points to generate.

    Returns:
        list: List of lists containing coordinates.
    """
    # 1. Generate points using Gaussian distribution (spherically symmetric)
    # This places points with random directions but non-uniform distribution relative to distance from origin
    points = np.random.normal(size=(count, n))

    # 2. Normalize the points to project them onto the surface of the unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points_on_surface = points / norms

    # 3. Generate random radii to distribute points uniformly within the volume
    # For n-dimensions, the volume element scales with r^(n-1), so the CDF is r^n.
    # To sample uniformly, we pick u ~ U[0,1] and take r = u^(1/n).
    random_radii = np.random.random(size=(count, 1)) ** (1 / n)

    return (points_on_surface * random_radii * radius).tolist()


def generate_noisy_sphere_points(n, count, r, noise_std, center_bounds=(-10, 10)):
    """
    Generates 'count' random points on a sphere surface with added Gaussian noise.

    Returns:
        list: List of lists containing the noisy points.
    """
    # 1. Random Center
    center = np.random.uniform(center_bounds[0], center_bounds[1], size=(1, n))

    # 2. Points on unit sphere surface (Gaussian method)
    points = np.random.normal(size=(count, n))
    points /= np.linalg.norm(points, axis=1, keepdims=True)

    # 3. Scale to radius 'r' and shift to center
    points = center + (points * r)

    # 4. Add Noise
    noise = np.random.normal(scale=noise_std, size=(count, n))

    return (points + noise).tolist()


def is_point_in_box(point, min_bounds, max_bounds):
    if len(min_bounds) == 0 and len(max_bounds) == 0:
        return False
    for i in range(len(point)):
        if not (min_bounds[i] <= point[i] <= max_bounds[i]):
            return False
    return True


def update_box(point, min_bounds, max_bounds):
    """
    Expands the box defined by min_bounds and max_bounds to include the given point.

    Args:
        min_bounds (array-like): Current minimum coordinates [x_min, y_min, ...]
        max_bounds (array-like): Current maximum coordinates [x_max, y_max, ...]
        point (array-like): The point to include [p_1, p_2, ...]

    Returns:
        tuple: (new_min_bounds, new_max_bounds) as lists
    """
    new_mins = [min(min_bounds[i], point[i]) for i in range(len(point))]
    new_maxs = [max(max_bounds[i], point[i]) for i in range(len(point))]
    return new_mins, new_maxs


class Oracle:
    """
    Oracle class that tracks consent checks with caching.
    Multiple consent checks for the same point only count as one oracle call.
    """

    def __init__(self):
        """Initialize a new Oracle instance with empty cache and counter."""
        self._cache = {}  # Maps point tuples to consent values
        self._call_count = 0  # Counter for unique oracle calls

    def get_ground_truth(self, tup):
        """
        Oracle function that checks point consent. Uses caching to track unique oracle calls.
        Multiple calls for the same point only count as one oracle call.

        Args:
            tup: Tuple of (point, consent_value)

        Returns:
            bool: The consent value for the point
        """
        point, ground_truth = tup[0], tup[1]

        # Convert point to tuple for hashability (for caching)
        point_tuple = tuple(point)

        # Check if we've already queried this point
        if point_tuple not in self._cache:
            # This is a new unique oracle call
            self._call_count += 1
            self._cache[point_tuple] = ground_truth

        return self._cache[point_tuple]

    def get_call_count(self):
        """Returns the number of unique oracle calls made."""
        return self._call_count

    def add_call_count(self, count):
        """Adds to the number of oracle calls made."""
        self._call_count += count

    def reset(self):
        """Reset the cache and counter. Call this before running a new algorithm."""
        self._cache = {}
        self._call_count = 0


def incremental_orthogonal(relation, oracle):
    """
    Runs the incremental algorithm for finding a minimum orthogonal bounding range query for a relation.
    Returns a tuple containing the lower and upper bound in each coordinate of the query box.

    Args:
        relation: The relation (list of tuples)
        oracle: Oracle instance for tracking consent checks
    """
    relation_list = list(relation)
    random.shuffle(relation_list)

    query_min = query_max = []
    for tup in relation_list:
        point = tup[0]
        if not is_point_in_box(point, query_min, query_max):
            if oracle.get_ground_truth(tup):
                if len(query_min) == 0 and len(query_max) == 0:
                    # Query initialization
                    query_min = query_max = point
                else:
                    # Query update
                    query_min, query_max = update_box(point, query_min, query_max)

    return query_min, query_max


def is_point_in_sphere(
    point: List[float],
    center: List[float],
    radius_sq: float,
    tolerance=GEOMETRY_TOLERANCE,
):
    """
    Checks if a point is inside a sphere (inclusive).

    Args:
        point (array-like): The coordinates of the point [p_1, p_2, ...]
        center (array-like): The coordinates of the sphere center [c_1, c_2, ...]
        radius_sq (float): The radius of the sphere squared

    Returns:
        bool: True if inside or on boundary, False otherwise.
    """
    # Optimized check: avoid numpy function call overhead for small dimensions
    d = len(point)
    if d == 3:
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        dz = point[2] - center[2]
        distance_squared = dx * dx + dy * dy + dz * dz
    elif d == 2:
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        distance_squared = dx * dx + dy * dy
    elif d == 4:
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        dz = point[2] - center[2]
        dw = point[3] - center[3]
        distance_squared = dx * dx + dy * dy + dz * dz + dw * dw
    else:
        distance_squared = 0.0
        for i in range(d):
            diff = point[i] - center[i]
            distance_squared += diff * diff

    # Compare with tolerance
    return distance_squared <= radius_sq + tolerance


def is_row_in_matrix(target_array, list_of_arrays):
    """
    Helper function for checking if a target array is in a list of arrays
    """
    return target_array in list_of_arrays


def get_circum_ball(R: List[List[float]]) -> Tuple[List[float], float]:
    """
    Computes the circumsphere of a set of points R.
    The center is the unique point in the affine hull of R that is equidistant from all points in R.
    """
    if len(R) == 0:
        return None, -1.0

    # Ensure all points are numpy arrays for vector operations
    R_arr = [np.array(p) for p in R]

    if len(R_arr) == 1:
        return R_arr[0].tolist(), 0.0

    # Points v_i = p_{i+1} - p_1
    p1 = R_arr[0]
    vs = [p - p1 for p in R_arr[1:]]
    n = len(vs)

    # Solve 2 * vs[i] . delta_c = ||vs[i]||^2
    # where delta_c = sum(lambda_j * vs[j])
    # sum_j lambda_j * (2 * vs[i] . vs[j]) = ||vs[i]||^2

    M = np.zeros((n, n))
    B = np.zeros(n)
    for i in range(n):
        for j in range(n):
            M[i, j] = 2.0 * np.dot(vs[i], vs[j])
        B[i] = np.dot(vs[i], vs[i])

    try:
        lambdas = np.linalg.solve(M, B)
    except np.linalg.LinAlgError:
        # Fallback for degenerate cases: use Welzl
        R_P = [(p.tolist(), True) for p in R_arr]
        return welzl(R_P, [], Oracle(), len(R_arr[0]), len(R_P))

    delta_c = np.sum([lambdas[j] * vs[j] for j in range(n)], axis=0)

    center = p1 + delta_c
    radius_sq = np.sum(delta_c**2)
    return center.tolist(), radius_sq


def welzl_cpp(
    P: List[Tuple[List[float], bool]],
    oracle: Oracle,
    d: int,
    n: int,
) -> Tuple[Optional[List[float]], float]:
    module = import_welzl_cpp()
    if not module:
        raise ImportError(
            "C++ Welzl backend library 'welzl_cpp_module' could not be loaded. "
            "Please ensure the module is compiled successfully in 'cpp_welzl/' and that you are running Python inside WSL (Linux environment)."
        )
    
    # Extract points and consents directly as lists/vectors to pass to pybind11
    points = [P[i][0] for i in range(n)]
    consents = [P[i][1] for i in range(n)]
    
    res = module.welzl_consent_cpp(points, consents, d)
    
    if res.success:
        # Record oracle calls made in C++
        oracle.add_call_count(res.oracle_calls)
        return res.center, res.radius_sq
    else:
        return None, -1.0



def welzl_py(
    P: List[Tuple[List[float], bool]],
    R: List[List[float]],
    oracle: Oracle,
    d: int,
    n: int,
    debug: bool = False,
    P_coords: Optional[List[List[float]]] = None,
) -> Tuple[Optional[List[float]], float]:
    """
    Welzl's algorithm for finding the smallest enclosing ball in Python.
    """
    if n == 0 or len(R) == d + 1:
        if len(R) == 0:
            return None, -1.0
        center, radius_sq = get_circum_ball(R)
        if debug:
            for p in R:
                dist_sq = sum((pi - ci) ** 2 for pi, ci in zip(p, center))
                assert abs(dist_sq - radius_sq) < DEBUG_BOUNDARY_TOLERANCE, (
                    f"Point {p} not on boundary. Dist_sq: {dist_sq}, Radius_sq: {radius_sq}"
                )
        return center, radius_sq

    if P_coords is None:
        P_coords = [p[0] for p in P]

    center, radius_sq = get_circum_ball(R)

    for i in range(n):
        pt = P_coords[i]
        if center is not None and is_point_in_sphere(pt, center, radius_sq):
            continue

        p_tup = P[i]
        if oracle.get_ground_truth(p_tup):
            center, radius_sq = welzl(
                P, R + [pt], oracle, d, i, debug=debug, P_coords=P_coords
            )

    return center, radius_sq


def welzl(
    P: List[Tuple[List[float], bool]],
    R: List[List[float]],
    oracle: Oracle,
    d: int,
    n: int,
    debug: bool = False,
    P_coords: Optional[List[List[float]]] = None,
) -> Tuple[Optional[List[float]], float]:
    if USE_CPP_BACKEND and len(R) == 0:
        return welzl_cpp(P, oracle, d, n)
    return welzl_py(P, R, oracle, d, n, debug=debug, P_coords=P_coords)


def incremental_distance_based(relation, oracle, debug: bool = False):
    """
    Runs the incremental algorithm for finding a minimum distance-based bounding range query for a relation.
    Returns a tuple representing the center point of the sphere, and a radius.

    Args:
        relation: The relation (list of tuples)
        oracle: Oracle instance for tracking consent checks
        debug (bool): Run slow debug assertions within Welzl's algorithm
    """
    if not relation:
        return 0, 0.0

    relation_list = list(relation)
    random.shuffle(relation_list)

    d = len(relation_list[0][0])

    center, radius_sq = welzl(
        relation_list, [], oracle, d, len(relation_list), debug=debug
    )

    if center is None:
        return 0, 0.0

    return center, math.sqrt(radius_sq)


def decremental_orthogonal(points, oracle):
    """
    Finds the minimum bounding box containing all consenting points by iteratively
    removing non-consenting points that sit on the edges of the box.

    Args:
        points (list): List of tuples (array_like, bool).
                       Example: [([x1, y1], True), ([x2, y2], False)]
                       Where array_like is the point and bool is the consent.
        oracle: Oracle instance for tracking consent checks

    Returns:
        tuple: (min_bounds, max_bounds) - Arrays defining the box
    """
    # Separate coordinates and consents
    if not points:
        return ([], [])

    # Track which points are still in the box calculation
    # Initially, all points are candidates
    active_indices = set(range(len(points)))

    while True:
        # 1. Get currently active coordinates
        if not active_indices:
            return ([], [])

        # 2. Find Minimum Bounding Box of active points
        first_idx = next(iter(active_indices))
        d = len(points[first_idx][0])

        min_bounds = list(points[first_idx][0])
        max_bounds = list(points[first_idx][0])
        for idx in active_indices:
            pt = points[idx][0]
            for col in range(d):
                if pt[col] < min_bounds[col]:
                    min_bounds[col] = pt[col]
                if pt[col] > max_bounds[col]:
                    max_bounds[col] = pt[col]

        # 3. Identify Active Points on the Edge and check consent
        to_remove = []
        for idx in active_indices:
            pt = points[idx][0]
            is_on_edge = False
            for col in range(d):
                if pt[col] == min_bounds[col] or pt[col] == max_bounds[col]:
                    is_on_edge = True
                    break

            if is_on_edge:
                # check consent via oracle
                if not oracle.get_ground_truth(points[idx]):
                    to_remove.append(idx)

        # 4. If no points need removal, we are done
        if not to_remove:
            return (min_bounds, max_bounds)

        # 5. Remove non-consenting edge points and repeat
        for idx in to_remove:
            active_indices.remove(idx)


def decremental_distance_based(
    points: List[Tuple[List[float], bool]], oracle: Oracle
) -> Tuple[Optional[List[float]], float]:
    """
    Finds the minimum enclosing ball containing all consenting points by iteratively
    removing non-consenting points that sit on the boundary of the ball.

    Args:
        points (list): List of tuples (array_like, bool).
        oracle: Oracle instance for tracking consent checks

    Returns:
        tuple: (center, radius) - Definition of the ball
    """
    if not points:
        return None, 0.0

    # Track active points
    active_indices = set(range(len(points)))
    d = len(points[0][0])

    while True:
        if not active_indices:
            return None, 0.0

        # 1. Calculate Minimum Enclosing Ball for active points using our Welzl implementation
        # Pass all consents as True so that Welzl ignores real consent flags during MEB computation
        active_P = [(points[idx][0], True) for idx in active_indices]
        random.shuffle(active_P)
        center, radius_sq = welzl(active_P, [], Oracle(), d, len(active_P))

        # 2. Identify points on the boundary (Edge)
        to_remove = []
        for idx in active_indices:
            pt = points[idx][0]
            dist_sq = 0.0
            for col in range(d):
                diff = pt[col] - center[col]
                dist_sq += diff * diff

            if abs(dist_sq - radius_sq) <= GEOMETRY_TOLERANCE:
                # This point is on the boundary, check consent via oracle
                if not oracle.get_ground_truth(points[idx]):
                    to_remove.append(idx)

        # 3. If no non-consenting points are on the boundary, we are done
        if not to_remove:
            return center, math.sqrt(radius_sq)

        # 4. Remove non-consenting boundary points and repeat
        for idx in to_remove:
            active_indices.remove(idx)
