import numpy as np
import random


def generate_in_box(n, low=0, high=1, count=1):
    """
    Generates 'count' random points of dimension 'n' within a box defined by [low, high].
    
    Args:
        n (int): Dimension of the space.
        low (float): Lower bound of the box.
        high (float): Upper bound of the box.
        count (int): Number of points to generate.
        
    Returns:
        np.ndarray: Array of shape (count, n).
    """
    # Uniform distribution for each coordinate independently
    return np.random.uniform(low, high, size=(count, n))

def generate_in_sphere(n, radius, count):
    """
    Generates 'count' random points of dimension 'n' uniformly distributed inside a sphere.
    
    Args:
        n (int): Dimension of the space.
        radius (float): Radius of the sphere.
        count (int): Number of points to generate.
        
    Returns:
        np.ndarray: Array of shape (count, n).
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
    random_radii = np.random.random(size=(count, 1)) ** (1/n)
    
    return points_on_surface * random_radii * radius

def generate_noisy_sphere_points(n, count, r, noise_std, center_bounds=(-10, 10)):
    """
    Generates 'count' random points on a sphere surface with added Gaussian noise.
    
    Returns:
        np.ndarray: Array of shape (count, n) containing the noisy points.
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
    
    return points + noise

def is_point_in_box(point, min_bounds, max_bounds):
    if len(min_bounds) == 0 and len(max_bounds) == 0:
        return False
    
    point = np.array(point)
    mins = np.array(min_bounds)
    maxs = np.array(max_bounds)
    
    # logical_and checks both conditions element-wise
    # np.all ensures it is true for every dimension
    return np.all((point >= mins) & (point <= maxs))

def update_box(point, min_bounds, max_bounds):
    """
    Expands the box defined by min_bounds and max_bounds to include the given point.
    
    Args:
        min_bounds (array-like): Current minimum coordinates [x_min, y_min, ...]
        max_bounds (array-like): Current maximum coordinates [x_max, y_max, ...]
        point (array-like): The point to include [p_1, p_2, ...]
        
    Returns:
        tuple: (new_min_bounds, new_max_bounds) as numpy arrays
    """
    # Convert inputs to numpy arrays to ensure element-wise operations work
    mins = np.array(min_bounds)
    maxs = np.array(max_bounds)
    p = np.array(point)
    
    # Calculate new bounds element-wise
    new_mins = np.minimum(mins, p)
    new_maxs = np.maximum(maxs, p)
    
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
        point_tuple = tuple(point) if isinstance(point, (list, np.ndarray)) else point
        
        # Check if we've already queried this point
        if point_tuple not in self._cache:
            # This is a new unique oracle call
            self._call_count += 1
            self._cache[point_tuple] = ground_truth
        
        return self._cache[point_tuple]
    
    def get_call_count(self):
        """Returns the number of unique oracle calls made."""
        return self._call_count
    
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

import miniball

def is_point_in_sphere(point, center, radius_sq, tolerance=1e-10):
    """
    Checks if a point is inside a sphere (inclusive).
    
    Args:
        point (array-like): The coordinates of the point [p_1, p_2, ...]
        center (array-like): The coordinates of the sphere center [c_1, c_2, ...]
        radius_sq (float): The radius of the sphere squared
        
    Returns:
        bool: True if inside or on boundary, False otherwise.
    """
    p = np.array(point)
    c = np.array(center)
    
    # Calculate Euclidean distance squared: (x-cx)^2 + (y-cy)^2 + ...
    distance_squared = np.sum((p - c)**2)
    
    # Compare with tolerance
    return distance_squared <= radius_sq + tolerance

def is_row_in_matrix(target_array, list_of_arrays):
    """
    Helper function for checking if a target array is in a list of arrays
    """
    if len(list_of_arrays) == 0:
        return False
        
    # Convert list of arrays to a single 2D array (N x D)
    matrix = np.array(list_of_arrays)
    
    # 1. Compare target against every row (broadcasting) -> Result is matrix of bools
    # 2. Check if ALL columns match for a row (.all(axis=1)) -> Result is 1D array of bools
    # 3. Check if ANY row was a match (.any()) -> Result is single bool
    return np.any(np.all(matrix == target_array, axis=1))

def rec_incremental_distance_based(relation, S, oracle):
    """
    Recursive helper function for incremental_distance_based
    
    Args:
        relation: The relation (list of tuples)
        S: List of support points
        oracle: Oracle instance for tracking consent checks
    """
    relation_list = list(relation)
    # Traverse the tuples in a random order
    random.shuffle(relation_list)

    R = []
    
    query_point, query_radius_sq = 0, 0
    # Used instead of the dummy tuple
    pushed_to_S = False
    
    for tup in relation_list:
        if not is_row_in_matrix(tup[0], S):
            R.append(tup)
            # print(S, tup[0])
            if len(R) + len(S) + (not pushed_to_S) >= len(tup[0]) + 1 and not is_point_in_sphere(tup[0], query_point, query_radius_sq):
                ground_truth = oracle.get_ground_truth(tup)
                if ground_truth:
                    if pushed_to_S:
                        S.pop()
                    S.append(tup[0])
                    pushed_to_S = True
                    if len(S) <= len(tup[0]):
                        query_point, query_radius_sq = rec_incremental_distance_based(R, S, oracle)
                    else:
                        query_point, query_radius_sq = miniball.get_bounding_ball(np.array(S))

    if pushed_to_S:
        S.pop()
    return query_point, query_radius_sq


def incremental_distance_based(relation, oracle):
    """
    Runs the incremental algorithm for finding a minimum distance-based bounding range query for a relation.
    Returns a tuple representing the center point of the sphere, and a radius.
    
    Args:
        relation: The relation (list of tuples)
        oracle: Oracle instance for tracking consent checks
    """
    S = []
    center, radius_sq = rec_incremental_distance_based(relation, S, oracle)
    
    return center, np.sqrt(radius_sq)

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
        return (np.array([]), np.array([]))
        
    # Extract coordinates into a numpy array for speed
    coords = np.array([p[0] for p in points])
    # Store original tuples for oracle calls
    point_tuples = [(p[0], p[1]) for p in points]
    
    # Track which points are still in the box calculation
    # Initially, all points are candidates
    active_mask = np.ones(len(points), dtype=bool)
    
    while True:
        # 1. Get currently active coordinates
        active_coords = coords[active_mask]
        
        if active_coords.size == 0:
            return (np.array([]), np.array([]))

        # 2. Find Minimum Bounding Box of active points
        min_bounds = np.min(active_coords, axis=0)
        max_bounds = np.max(active_coords, axis=0)
        
        # 3. Identify Active Points on the Edge
        # Create masks relative to the *active* subset
        on_min = (active_coords == min_bounds)
        on_max = (active_coords == max_bounds)
        
        # A point is on the edge if it touches min or max in ANY dimension
        is_on_edge_subset = np.any(on_min | on_max, axis=1)
        
        # 4. Check Consents of these edge points using oracle
        active_indices = np.nonzero(active_mask)[0]
        active_consents_array = np.zeros(len(active_indices), dtype=bool)
        
        # Only check consent for points on the edge
        for subset_idx, global_idx in enumerate(active_indices):
            if is_on_edge_subset[subset_idx]:
                # This point is on the edge, check consent via oracle
                tup = point_tuples[global_idx]
                active_consents_array[subset_idx] = oracle.get_ground_truth(tup)
        
        # Identify points that are ON the edge AND DO NOT consent
        should_remove_subset = is_on_edge_subset & (~active_consents_array)
        
        # 5. If no points need removal, we are done
        if not np.any(should_remove_subset):
            return (min_bounds, max_bounds)
            
        # 6. Remove non-consenting edge points and repeat
        # Map subset indices back to global mask to disable them
        indices_to_remove = active_indices[should_remove_subset]
        active_mask[indices_to_remove] = False
    

def decremental_distance_based(points, oracle):
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

    # Prepare data: Extract coords and store original tuples for oracle calls
    coords = np.array([p[0] for p in points])
    point_tuples = [(p[0], p[1]) for p in points]
    
    # Track active points
    active_mask = np.ones(len(points), dtype=bool)

    while True:
        active_coords = coords[active_mask]
        
        if active_coords.size == 0:
            return None, 0.0
            
        # 1. Calculate Minimum Enclosing Ball for active points
        # Returns center and radius_squared
        center, radius_sq = miniball.get_bounding_ball(active_coords)
        center = np.array(center) # Ensure center is numpy array for broadcasting
        
        # 2. Identify points on the boundary (Edge)
        # Calculate squared distances from center to all active points
        dists_sq = np.sum((active_coords - center)**2, axis=1)
        
        # Check tolerance to handle floating point precision issues
        # Points are on the boundary if their distance is close to the radius
        is_on_edge_subset = np.isclose(dists_sq, radius_sq)
        
        # 3. Check Consents of boundary points using oracle
        active_indices = np.nonzero(active_mask)[0]
        active_consents_array = np.zeros(len(active_indices), dtype=bool)
        
        # Only check consent for points on the boundary
        for subset_idx, global_idx in enumerate(active_indices):
            if is_on_edge_subset[subset_idx]:
                # This point is on the boundary, check consent via oracle
                tup = point_tuples[global_idx]
                active_consents_array[subset_idx] = oracle.get_ground_truth(tup)
        
        should_remove_subset = is_on_edge_subset & (~active_consents_array)
        
        # 4. If no non-consenting points are on the boundary, we are done
        if not np.any(should_remove_subset):
            return center.tolist(), np.sqrt(radius_sq)
            
        # 5. Remove non-consenting boundary points and repeat
        indices_to_remove = active_indices[should_remove_subset]
        active_mask[indices_to_remove] = False
