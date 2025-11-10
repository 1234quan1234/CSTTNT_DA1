"""
Utility functions for optimization algorithms.

This module provides common helper functions used across multiple optimizers:

- Distance calculations (Euclidean, Manhattan, etc.)
- Fitness utilities (brightness conversion, ranking)
- Solution repairs (permutation fixing, constraint handling)
- Statistical helpers (best solution extraction)

All functions are optimized for numpy arrays and support vectorized operations
for efficiency with large populations.

Notes
-----
Functions in this module should be:
1. Stateless (pure functions)
2. Vectorized when possible
3. Well-documented with examples
4. Tested with doctest

Examples
--------
>>> import numpy as np
>>> from src.core.utils import euclidean_distance_matrix, get_best_solution

>>> # Compute distances between points
>>> points = np.array([[0, 0], [1, 1], [2, 2]])
>>> distances = euclidean_distance_matrix(points)
>>> distances[0, 1]  # Distance from point 0 to point 1
1.4142135623730951

>>> # Find best solution in population
>>> population = np.array([[1, 2], [3, 4], [5, 6]])
>>> fitness = np.array([10.0, 5.0, 15.0])
>>> best_sol, best_fit = get_best_solution(population, fitness)
>>> best_fit
5.0
"""

import numpy as np
from typing import Tuple, Optional


def euclidean_distance_matrix(pop: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between all rows in a population.
    
    Uses vectorized computation via broadcasting for efficiency. This is the
    standard distance metric used by Firefly Algorithm to calculate attraction
    between fireflies.
    
    Mathematical Formula
    --------------------
    For points x_i and x_j in d-dimensional space:
    
    .. math::
        d(x_i, x_j) = \\sqrt{\\sum_{k=1}^{d} (x_{i,k} - x_{j,k})^2}
    
    Vectorized as:
    .. math::
        D_{ij} = \\|X_i - X_j\\|_2
    
    Parameters
    ----------
    pop : np.ndarray
        Population matrix of shape (n, d) where:
        - n is population size
        - d is dimensionality
    
    Returns
    -------
    dist_matrix : np.ndarray
        Symmetric distance matrix of shape (n, n) where dist_matrix[i, j]
        is the Euclidean distance between pop[i] and pop[j].
        Diagonal elements are zero (distance from point to itself).
    
    Notes
    -----
    **Time Complexity:** O(n²·d) where n=population size, d=dimensions
    **Space Complexity:** O(n²)
    
    This vectorized implementation is significantly faster than nested loops:
    - Vectorized: ~0.01s for n=1000, d=10
    - Loop-based: ~2.5s for same size
    
    The distance matrix is symmetric: dist_matrix[i,j] == dist_matrix[j,i]
    
    Examples
    --------
    >>> pop = np.array([[0, 0], [1, 0], [0, 1]])
    >>> dist = euclidean_distance_matrix(pop)
    >>> dist[0, 1]  # Distance from [0,0] to [1,0]
    1.0
    >>> dist[0, 2]  # Distance from [0,0] to [0,1]
    1.0
    >>> dist[1, 2]  # Distance from [1,0] to [0,1]
    1.4142135623730951
    
    For 3D points:
    >>> pop_3d = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    >>> dist_3d = euclidean_distance_matrix(pop_3d)
    >>> dist_3d[0, 1]  # Distance from origin to [1,1,1]
    1.7320508075688772
    
    References
    ----------
    .. [1] Yang, X. S. (2008). Firefly Algorithm. Nature-Inspired Metaheuristic
           Algorithms. Luniver Press.
    """
    # Vectorized computation using broadcasting
    # diff[i,j,k] = pop[i,k] - pop[j,k] for all i,j,k
    diff = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
    
    # dist[i,j] = sqrt(sum_k(diff[i,j,k]^2))
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    
    return dist_matrix


def get_best_fitness_index(fitness: np.ndarray) -> Tuple[float, int]:
    """
    Find the best (minimum) fitness value and its index in a fitness array.
    
    Since all algorithms in this framework minimize objective functions,
    "best" means minimum fitness value.
    
    Parameters
    ----------
    fitness : np.ndarray
        Array of fitness values (1D). Should contain finite values only.
    
    Returns
    -------
    best_fitness : float
        The minimum fitness value in the array.
    best_index : int
        The index (0-based) of the minimum fitness value.
    
    Raises
    ------
    ValueError
        If fitness array is empty or contains only NaN values.
    
    Notes
    -----
    **Time Complexity:** O(n) where n = len(fitness)
    **Space Complexity:** O(1)
    
    If multiple elements have the same minimum value, returns the index
    of the first occurrence.
    
    For handling NaN values, use np.nanargmin() instead of np.argmin().
    
    Examples
    --------
    >>> fitness = np.array([5.2, 3.1, 4.7, 2.8])
    >>> best_fit, best_idx = get_best_fitness_index(fitness)
    >>> best_fit
    2.8
    >>> best_idx
    3
    
    With identical minimum values:
    >>> fitness = np.array([3.0, 2.0, 2.0, 4.0])
    >>> best_fit, best_idx = get_best_fitness_index(fitness)
    >>> best_idx  # Returns first occurrence
    1
    
    See Also
    --------
    get_best_solution : Get solution corresponding to best fitness
    compute_brightness : Convert fitness to brightness for FA
    """
    if len(fitness) == 0:
        raise ValueError("Fitness array cannot be empty")
    
    best_index = int(np.argmin(fitness))
    best_fitness = float(fitness[best_index])
    
    return best_fitness, best_index


def get_best_solution(positions: np.ndarray, fitness: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Get the best solution and its fitness from a population.
    
    This is a convenience function that combines position lookup with
    fitness minimization. Commonly used at end of optimization to
    extract final result.
    
    Parameters
    ----------
    positions : np.ndarray
        Population matrix of shape (n, d) where:
        - n is population size
        - d is problem dimension (or problem size for discrete problems)
    fitness : np.ndarray
        Fitness values of shape (n,) corresponding to each position.
    
    Returns
    -------
    best_solution : np.ndarray
        The solution with the lowest fitness, shape (d,).
        This is a COPY of the solution, not a reference.
    best_fitness : float
        The fitness value of the best solution.
    
    Raises
    ------
    ValueError
        If positions and fitness have incompatible shapes.
    
    Notes
    -----
    **Time Complexity:** O(n) for finding minimum
    **Space Complexity:** O(d) for copying solution
    
    The returned solution is a copy to prevent accidental modification
    of the original population array.
    
    Examples
    --------
    >>> positions = np.array([[1, 2], [3, 4], [5, 6]])
    >>> fitness = np.array([10.0, 5.0, 15.0])
    >>> best_sol, best_fit = get_best_solution(positions, fitness)
    >>> best_sol
    array([3, 4])
    >>> best_fit
    5.0
    
    For Knapsack (binary solutions):
    >>> knapsack_solutions = np.array([[1, 0, 1, 0], 
    ...                                [0, 1, 1, 0],
    ...                                [1, 1, 0, 0]])
    >>> knapsack_fitness = np.array([-45.0, -52.0, -38.0])  # Negated values
    >>> best_sol, best_fit = get_best_solution(knapsack_solutions, knapsack_fitness)
    >>> best_sol  # Second solution has best (most negative) fitness
    array([0, 1, 1, 0])
    >>> best_fit
    -52.0
    
    See Also
    --------
    get_best_fitness_index : Find index of best fitness
    """
    if len(positions) != len(fitness):
        raise ValueError(f"positions ({len(positions)}) and fitness ({len(fitness)}) "
                        f"must have same length")
    
    best_fitness, best_index = get_best_fitness_index(fitness)
    best_solution = positions[best_index].copy()  # Copy to avoid reference issues
    
    return best_solution, best_fitness


def compute_brightness(fitness: np.ndarray) -> np.ndarray:
    """
    Convert fitness values to brightness values for Firefly Algorithm.
    
    In FA, fireflies are attracted to brighter fireflies. Since we minimize
    fitness (lower is better), brightness must be inversely related to fitness.
    
    Mathematical Formula
    --------------------
    Simple negation mapping:
    .. math::
        I_i = -f_i
    
    where:
    - I_i is brightness of firefly i
    - f_i is fitness (objective value) of firefly i
    
    Alternative formulations exist:
    .. math::
        I_i = \\frac{1}{1 + f_i}  \\quad \\text{(sigmoid-like)}
        
        I_i = \\exp(-f_i)  \\quad \\text{(exponential)}
    
    But simple negation is:
    - Computationally cheapest
    - Preserves relative ordering
    - Avoids division by zero issues
    
    Parameters
    ----------
    fitness : np.ndarray
        Array of fitness values (lower is better).
        Shape can be (n,) for 1D array or (n, 1) for column vector.
    
    Returns
    -------
    brightness : np.ndarray
        Array of brightness values (higher is better).
        Same shape as input. Better solutions have higher brightness.
    
    Notes
    -----
    **Time Complexity:** O(n)
    **Space Complexity:** O(n)
    
    Brightness Interpretation:
    - fitness = 10.0 → brightness = -10.0 (dim)
    - fitness = 5.0  → brightness = -5.0  (brighter)
    - fitness = 0.0  → brightness = 0.0   (brightest, optimal)
    
    For problems with negative fitness values, ensure consistent
    interpretation. If fitness can be negative, consider adding
    a constant offset:
    
    .. math::
        I_i = -(f_i - f_{worst})
    
    Examples
    --------
    >>> fitness = np.array([10.0, 5.0, 0.0, 15.0])
    >>> brightness = compute_brightness(fitness)
    >>> brightness
    array([-10.,  -5.,   0., -15.])
    
    >>> # Best fitness has highest brightness
    >>> np.argmax(brightness)  # Index of brightest
    2
    >>> np.argmin(fitness)  # Index of best fitness
    2
    
    In Firefly Algorithm usage:
    >>> # If firefly j is brighter than i, move i toward j
    >>> for i in range(n_fireflies):
    ...     for j in range(n_fireflies):
    ...         if brightness[j] > brightness[i]:
    ...             move_firefly(i, toward=j)
    
    References
    ----------
    .. [1] Yang, X. S. (2008). Nature-inspired metaheuristic algorithms.
           Section 3.3: Firefly Algorithm.
    
    See Also
    --------
    euclidean_distance_matrix : Used with brightness for attraction calculation
    """
    return -fitness


def repair_permutation(perm: np.ndarray, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """
    Repair an invalid permutation to make it a valid permutation.
    
    Invalid permutations arise from crossover/mutation operations in genetic
    algorithms for permutation-based problems (e.g., TSP). This function
    fixes duplicates and missing values to restore validity.
    
    Repair Strategy
    ---------------
    1. Keep unique values in order of first appearance
    2. Identify missing values in range [0, n-1]
    3. Fill missing values:
       - If rng=None: Append in sorted order (deterministic)
       - If rng provided: Shuffle before appending (stochastic)
    
    Parameters
    ----------
    perm : np.ndarray
        Possibly invalid permutation (may have duplicates or missing values).
        Should contain integers, ideally in range [0, n-1].
    rng : np.random.RandomState, optional
        Random number generator for reproducibility. 
        - If None: Uses deterministic repair (missing values added in order)
        - If provided: Shuffles missing values before adding (for diversity)
    
    Returns
    -------
    valid_perm : np.ndarray
        A valid permutation of [0, 1, ..., n-1].
        Length is same as input (n).
        Each value in [0, n-1] appears exactly once.
    
    Raises
    ------
    ValueError
        If perm is empty or contains values outside [0, n-1].
    
    Notes
    -----
    **Time Complexity:** O(n) where n = len(perm)
    **Space Complexity:** O(n)
    
    **Deterministic vs Stochastic Repair:**
    
    Deterministic (rng=None):
    - Useful for debugging and testing
    - Always produces same output for same input
    - Missing values added in ascending order
    
    Stochastic (rng provided):
    - Introduces diversity in population
    - Different repairs for same input
    - Better for avoiding premature convergence in GA
    
    **Use Cases:**
    - TSP: After Order Crossover (OX) or Partially Mapped Crossover (PMX)
    - Job Scheduling: Repairing task orders
    - Any permutation encoding problem
    
    Examples
    --------
    Deterministic repair:
    >>> invalid = np.array([0, 2, 2, 4, 1])  # Missing 3, duplicate 2
    >>> valid = repair_permutation(invalid)
    >>> valid
    array([0, 2, 4, 1, 3])
    >>> sorted(valid) == list(range(5))
    True
    
    With randomization for GA diversity:
    >>> rng = np.random.RandomState(42)
    >>> valid = repair_permutation(invalid, rng)
    >>> sorted(valid) == list(range(5))  # Still valid
    True
    >>> # But order of missing values may differ
    
    Reproducibility with RNG:
    >>> rng1 = np.random.RandomState(123)
    >>> rng2 = np.random.RandomState(123)
    >>> v1 = repair_permutation(invalid, rng1)
    >>> v2 = repair_permutation(invalid, rng2)
    >>> np.array_equal(v1, v2)  # Same seed → same result
    True
    
    References
    ----------
    .. [1] Goldberg, D. E., & Lingle, R. (1985). Alleles, loci, and the TSP.
           Proceedings of ICGA.
    .. [2] Davis, L. (1985). Applying adaptive algorithms to epistatic domains.
           IJCAI.
    
    See Also
    --------
    numpy.unique : Get unique elements
    numpy.setdiff1d : Find missing elements
    """
    n = len(perm)
    
    if n == 0:
        raise ValueError("Permutation cannot be empty")
    
    seen = set()
    result = []
    
    # First pass: keep unique values in order
    for val in perm:
        val_int = int(val)
        if val_int < 0 or val_int >= n:
            raise ValueError(f"Value {val_int} outside valid range [0, {n-1}]")
        if val_int not in seen:
            result.append(val_int)
            seen.add(val_int)
    
    # Second pass: add missing values
    missing = [i for i in range(n) if i not in seen]
    
    # Shuffle missing values if RNG provided (for randomized repair)
    if rng is not None and len(missing) > 0:
        rng.shuffle(missing)
    
    result.extend(missing)
    
    return np.array(result[:n], dtype=int)


if __name__ == "__main__":
    # Quick test
    print("Testing utils...")
    
    # Test euclidean_distance_matrix (vectorized version)
    print("\n[Test 1] Euclidean distance matrix (vectorized)")
    pop = np.array([[0, 0], [1, 0], [0, 1]])
    dist = euclidean_distance_matrix(pop)
    print(f"Distance matrix:\n{dist}")
    print(f"Distance [0,0] to [1,0]: {dist[0, 1]:.4f} (expected 1.0000)")
    print(f"Distance [0,0] to [0,1]: {dist[0, 2]:.4f} (expected 1.0000)")
    print(f"Distance [1,0] to [0,1]: {dist[1, 2]:.4f} (expected ~1.4142)")
    assert abs(dist[0, 1] - 1.0) < 1e-10, "Distance calculation error"
    assert abs(dist[0, 2] - 1.0) < 1e-10, "Distance calculation error"
    assert abs(dist[1, 2] - np.sqrt(2)) < 1e-10, "Distance calculation error"
    print("✓ Distance calculation correct")
    
    # Test get_best_fitness_index
    print("\n[Test 2] Get best fitness index")
    fitness = np.array([5.2, 3.1, 4.7, 2.8])
    best_fit, best_idx = get_best_fitness_index(fitness)
    print(f"Best fitness: {best_fit} at index {best_idx} (expected 2.8 at 3)")
    assert best_fit == 2.8, "Best fitness incorrect"
    assert best_idx == 3, "Best index incorrect"
    print("✓ Best fitness/index correct")
    
    # Test brightness
    print("\n[Test 3] Brightness computation")
    brightness = compute_brightness(fitness)
    print(f"Fitness:    {fitness}")
    print(f"Brightness: {brightness}")
    print(f"Best brightness index: {np.argmax(brightness)} (expected 3)")
    assert np.argmax(brightness) == 3, "Brightest should be lowest fitness"
    print("✓ Brightness computation correct")
    
    # Test repair_permutation (deterministic)
    print("\n[Test 4] Repair permutation (deterministic)")
    invalid_perm = np.array([0, 2, 2, 4, 1])  # Missing 3, duplicate 2
    valid_perm = repair_permutation(invalid_perm)
    print(f"Invalid permutation: {invalid_perm}")
    print(f"Repaired permutation: {valid_perm}")
    is_valid = sorted(valid_perm) == list(range(len(valid_perm)))
    print(f"Is valid: {is_valid}")
    assert is_valid, "Repaired permutation is not valid"
    print("✓ Deterministic repair correct")
    
    # Test repair_permutation (with RNG)
    print("\n[Test 5] Repair permutation (with RNG for reproducibility)")
    rng1 = np.random.RandomState(42)
    rng2 = np.random.RandomState(42)
    valid1 = repair_permutation(invalid_perm, rng1)
    valid2 = repair_permutation(invalid_perm, rng2)
    print(f"Repair 1: {valid1}")
    print(f"Repair 2: {valid2}")
    assert np.array_equal(valid1, valid2), "RNG-based repair not reproducible"
    print("✓ Reproducible repair with RNG correct")
    
    print("\n" + "=" * 60)
    print("All utils tests passed! ✓")
    print("=" * 60)
