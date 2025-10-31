"""
Utility functions for optimization algorithms.

This module provides helper functions used across multiple optimizers,
such as distance calculations, fitness utilities, and common operations.
"""

import numpy as np
from typing import Tuple, Optional


def euclidean_distance_matrix(pop: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between all rows in a population.
    
    This is used by Firefly Algorithm to calculate distances between fireflies.
    Uses vectorized computation for efficiency.
    
    Parameters
    ----------
    pop : np.ndarray
        Population matrix of shape (n, dim) where n is population size
        and dim is the dimensionality.
    
    Returns
    -------
    dist_matrix : np.ndarray
        Symmetric distance matrix of shape (n, n) where dist_matrix[i, j]
        is the Euclidean distance between pop[i] and pop[j].
    
    Notes
    -----
    This implementation uses vectorized computation via broadcasting,
    which is significantly faster than nested loops for large populations.
    Time complexity: O(n²·d) but with efficient vectorization.
    
    Examples
    --------
    >>> pop = np.array([[0, 0], [1, 1], [2, 2]])
    >>> distances = euclidean_distance_matrix(pop)
    >>> distances[0, 1]  # distance from [0,0] to [1,1]
    1.414...
    """
    # Vectorized computation using broadcasting
    # diff[i,j] = pop[i] - pop[j]
    diff = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
    # dist[i,j] = ||pop[i] - pop[j]||
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    return dist_matrix


def get_best_fitness_index(fitness: np.ndarray) -> Tuple[float, int]:
    """
    Find the best (minimum) fitness value and its index in a fitness array.
    
    Parameters
    ----------
    fitness : np.ndarray
        Array of fitness values (1D).
    
    Returns
    -------
    best_fitness : float
        The minimum fitness value.
    best_index : int
        The index of the minimum fitness value.
    
    Examples
    --------
    >>> fitness = np.array([5.2, 3.1, 4.7, 2.8])
    >>> best_fit, best_idx = get_best_fitness_index(fitness)
    >>> best_fit
    2.8
    >>> best_idx
    3
    """
    best_index = int(np.argmin(fitness))
    best_fitness = float(fitness[best_index])
    return best_fitness, best_index


def get_best_solution(positions: np.ndarray, fitness: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Get the best solution and its fitness from a population.
    
    Parameters
    ----------
    positions : np.ndarray
        Population matrix of shape (n, dim) or (n, problem_size).
    fitness : np.ndarray
        Fitness values of shape (n,).
    
    Returns
    -------
    best_solution : np.ndarray
        The solution with the lowest fitness, shape (dim,) or (problem_size,).
    best_fitness : float
        The fitness value of the best solution.
    
    Examples
    --------
    >>> positions = np.array([[1, 2], [3, 4], [5, 6]])
    >>> fitness = np.array([10.0, 5.0, 15.0])
    >>> best_sol, best_fit = get_best_solution(positions, fitness)
    >>> best_sol
    array([3, 4])
    >>> best_fit
    5.0
    """
    best_fitness, best_index = get_best_fitness_index(fitness)
    best_solution = positions[best_index].copy()
    return best_solution, best_fitness


def compute_brightness(fitness: np.ndarray) -> np.ndarray:
    """
    Convert fitness values to brightness values for Firefly Algorithm.
    
    Since we minimize fitness (lower is better), brightness should be inversely
    related to fitness. We use brightness = -fitness so that better solutions
    (lower fitness) have higher brightness.
    
    Parameters
    ----------
    fitness : np.ndarray
        Array of fitness values.
    
    Returns
    -------
    brightness : np.ndarray
        Array of brightness values (higher = better).
    
    Notes
    -----
    In FA, fireflies are attracted to brighter fireflies. Since we're minimizing,
    the best fitness (lowest value) should correspond to the highest brightness.
    
    Mathematical relationship:
    - fitness = 10.0 → brightness = -10.0
    - fitness = 5.0  → brightness = -5.0 (brighter, better)
    - fitness = 0.0  → brightness = 0.0 (brightest, optimal)
    
    Alternative formulations exist (e.g., brightness = 1/(1+fitness)), but
    simple negation works well for most cases and avoids division issues.
    """
    return -fitness


def repair_permutation(perm: np.ndarray, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """
    Repair an invalid permutation to make it a valid permutation.
    
    This is useful when crossover or mutation operations on TSP solutions
    produce invalid permutations (repeated or missing cities).
    
    Parameters
    ----------
    perm : np.ndarray
        Possibly invalid permutation (may have duplicates or missing values).
    rng : np.random.RandomState, optional
        Random number generator for reproducibility. If None, uses deterministic
        repair (fills missing values in ascending order).
    
    Returns
    -------
    valid_perm : np.ndarray
        A valid permutation of [0, 1, ..., n-1].
    
    Notes
    -----
    This implementation uses a greedy repair strategy:
    1. Keep unique values in order
    2. Fill missing values (in order if rng=None, shuffled if rng provided)
    
    For reproducibility in stochastic algorithms, always pass an RNG instance.
    
    Examples
    --------
    >>> invalid = np.array([0, 2, 2, 4, 1])  # Missing 3, duplicate 2
    >>> valid = repair_permutation(invalid)
    >>> sorted(valid) == list(range(5))
    True
    
    With randomization:
    >>> rng = np.random.RandomState(42)
    >>> valid = repair_permutation(invalid, rng)
    >>> sorted(valid) == list(range(5))
    True
    """
    n = len(perm)
    seen = set()
    result = []
    
    # First pass: keep unique values in order
    for val in perm:
        val_int = int(val)
        if val_int not in seen and 0 <= val_int < n:
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
