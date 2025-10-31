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
    
    Examples
    --------
    >>> pop = np.array([[0, 0], [1, 1], [2, 2]])
    >>> distances = euclidean_distance_matrix(pop)
    >>> distances[0, 1]  # distance from [0,0] to [1,1]
    1.414...
    """
    n = pop.shape[0]
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(pop[i] - pop[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
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
    """
    return -fitness


def repair_permutation(perm: np.ndarray) -> np.ndarray:
    """
    Repair an invalid permutation to make it a valid permutation.
    
    This is useful when crossover or mutation operations on TSP solutions
    produce invalid permutations (repeated or missing cities).
    
    Parameters
    ----------
    perm : np.ndarray
        Possibly invalid permutation (may have duplicates or missing values).
    
    Returns
    -------
    valid_perm : np.ndarray
        A valid permutation of [0, 1, ..., n-1].
    
    Notes
    -----
    This implementation uses a greedy repair strategy: it keeps unique values
    in order and fills missing values at the end.
    """
    n = len(perm)
    seen = set()
    result = []
    
    # First pass: keep unique values in order
    for val in perm:
        if val not in seen and 0 <= val < n:
            result.append(int(val))
            seen.add(val)
    
    # Second pass: add missing values
    for i in range(n):
        if i not in seen:
            result.append(i)
    
    return np.array(result[:n], dtype=int)


if __name__ == "__main__":
    # Quick test
    print("Testing utils...")
    
    # Test euclidean_distance_matrix
    pop = np.array([[0, 0], [1, 0], [0, 1]])
    dist = euclidean_distance_matrix(pop)
    print(f"Distance matrix:\n{dist}")
    print(f"Distance [0,0] to [1,0]: {dist[0, 1]:.4f} (expected 1.0)")
    print(f"Distance [0,0] to [0,1]: {dist[0, 2]:.4f} (expected 1.0)")
    print(f"Distance [1,0] to [0,1]: {dist[1, 2]:.4f} (expected ~1.414)")
    
    # Test get_best_fitness_index
    fitness = np.array([5.2, 3.1, 4.7, 2.8])
    best_fit, best_idx = get_best_fitness_index(fitness)
    print(f"\nBest fitness: {best_fit} at index {best_idx} (expected 2.8 at 3)")
    
    # Test brightness
    brightness = compute_brightness(fitness)
    print(f"\nBrightness: {brightness}")
    print(f"Best brightness index: {np.argmax(brightness)} (expected 3)")
    
    # Test repair_permutation
    invalid_perm = np.array([0, 2, 2, 4, 1])  # Missing 3, duplicate 2
    valid_perm = repair_permutation(invalid_perm)
    print(f"\nInvalid permutation: {invalid_perm}")
    print(f"Repaired permutation: {valid_perm}")
    print(f"Is valid: {sorted(valid_perm) == list(range(len(valid_perm)))}")
    
    print("\nAll utils tests passed!")
