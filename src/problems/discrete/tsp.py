"""
Traveling Salesman Problem (TSP) optimization problem.

The TSP is a classic combinatorial optimization problem where the goal is to find
the shortest route that visits all cities exactly once and returns to the origin.

References
----------
.. [1] https://en.wikipedia.org/wiki/Travelling_salesman_problem
.. [2] Lawler, E. L., et al. (1985). The traveling salesman problem: 
       A guided tour of combinatorial optimization. Wiley.
"""

import numpy as np
from typing import Literal

# Use relative import
from ...core.problem_base import ProblemBase


class TSPProblem(ProblemBase):
    """
    Traveling Salesman Problem (TSP).
    
    Given a set of cities and distances between them, find the shortest tour
    that visits each city exactly once and returns to the starting city.
    
    Solution representation: permutation of city indices.
    For n cities, a solution is a permutation of [0, 1, 2, ..., n-1].
    
    Parameters
    ----------
    coords : np.ndarray
        City coordinates, shape (num_cities, 2) or (num_cities, d) for d-dimensional.
        Each row is the position vector of a city.
    
    Attributes
    ----------
    coords : np.ndarray
        City coordinates, shape (num_cities, spatial_dim).
    num_cities : int
        Number of cities.
    distance_matrix : np.ndarray
        Precomputed pairwise Euclidean distance matrix, shape (num_cities, num_cities).
        distance_matrix[i,j] is the distance from city i to city j.
    
    Notes
    -----
    **Space Complexity:** O(n²) for distance matrix storage
    **Time Complexity (evaluate):** O(n) using precomputed distances
    **Search Space Size:** n! possible tours (factorial growth)
    
    For symmetric TSP, distance_matrix[i,j] == distance_matrix[j,i].
    For asymmetric TSP, distances may differ (not implemented here).
    
    Examples
    --------
    >>> coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    >>> problem = TSPProblem(coords)
    >>> tour = np.array([0, 1, 2, 3])  # Visit in order
    >>> cost = problem.evaluate(tour)
    >>> print(f"Tour cost: {cost:.4f}")  # Perimeter of unit square = 4.0
    """
    
    def __init__(self, coords: np.ndarray):
        """
        Initialize TSP problem.
        
        Parameters
        ----------
        coords : np.ndarray
            City coordinates, shape (num_cities, spatial_dim).
            Typically spatial_dim=2 for 2D Euclidean TSP.
        
        Raises
        ------
        ValueError
            If coords has invalid shape or less than 2 cities.
        """
        self.coords = np.array(coords, dtype=float)
        
        # Validation
        if self.coords.ndim != 2:
            raise ValueError(
                f"coords must be 2D array (num_cities, spatial_dim), "
                f"got shape {self.coords.shape}"
            )
        if len(self.coords) < 2:
            raise ValueError(
                f"TSP requires at least 2 cities, got {len(self.coords)}"
            )
        
        self.num_cities = len(self.coords)
        
        # Precompute distance matrix for efficiency (vectorized)
        self.distance_matrix = self._compute_distance_matrix()
    
    def _compute_distance_matrix(self) -> np.ndarray:
        """
        Compute Euclidean distance matrix between all city pairs (vectorized).
        
        Returns
        -------
        dist_matrix : np.ndarray
            Symmetric distance matrix, shape (num_cities, num_cities).
            dist_matrix[i,j] = ||coords[i] - coords[j]||₂
        
        Notes
        -----
        Uses vectorized computation via broadcasting for efficiency:
        ~100x faster than nested loops for n>100 cities.
        """
        # Vectorized distance computation
        # diff[i,j] = coords[i] - coords[j]
        diff = self.coords[:, np.newaxis, :] - self.coords[np.newaxis, :, :]
        # dist[i,j] = ||coords[i] - coords[j]||
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
        
        return dist_matrix
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the tour length (total distance).
        
        The tour is a closed loop: starts at x[0], visits cities in order,
        and returns to x[0].
        
        Parameters
        ----------
        x : np.ndarray
            Tour permutation, shape (num_cities,), contains integers 0 to num_cities-1.
            Each integer should appear exactly once (valid permutation).
        
        Returns
        -------
        tour_length : float
            Total Euclidean distance of the tour (sum of edge weights).
        
        Notes
        -----
        Time complexity: O(n) where n = num_cities (uses precomputed distances)
        
        Does not validate if x is a valid permutation for performance.
        Invalid tours (e.g., repeated cities) will give incorrect results.
        
        Examples
        --------
        >>> coords = np.array([[0, 0], [1, 0], [0, 1]])
        >>> problem = TSPProblem(coords)
        >>> tour = np.array([0, 1, 2])
        >>> length = problem.evaluate(tour)
        """
        tour = x.astype(int)
        total_distance = 0.0
        
        # Sum distances between consecutive cities in tour
        for i in range(self.num_cities - 1):
            total_distance += self.distance_matrix[tour[i], tour[i + 1]]
        
        # Add return edge to starting city (closed tour)
        total_distance += self.distance_matrix[tour[-1], tour[0]]
        
        return float(total_distance)
    
    def representation_type(self) -> Literal["continuous", "tsp", "knapsack", "graph_coloring"]:
        """Return 'tsp' for this problem type."""
        return "tsp"
    
    def init_solution(self, rng: np.random.RandomState, n: int = 1) -> np.ndarray:
        """
        Generate n random tour permutations.
        
        Parameters
        ----------
        rng : np.random.RandomState
            Random number generator for reproducibility.
        n : int, default=1
            Number of tours to generate.
        
        Returns
        -------
        tours : np.ndarray
            Array of shape (n, num_cities) where each row is a random permutation
            of [0, 1, ..., num_cities-1].
        
        Examples
        --------
        >>> problem = TSPProblem(np.random.rand(5, 2))
        >>> rng = np.random.RandomState(42)
        >>> tours = problem.init_solution(rng, n=3)
        >>> tours.shape
        (3, 5)
        >>> np.all(np.sort(tours, axis=1) == np.arange(5))  # Valid permutations
        True
        """
        tours = np.zeros((n, self.num_cities), dtype=int)
        
        for i in range(n):
            tours[i] = rng.permutation(self.num_cities)
        
        return tours
    
    def clip(self, X: np.ndarray) -> np.ndarray:
        """
        For TSP, clipping is not applicable (discrete permutation problem).
        
        Returns the input unchanged. Validity of permutations should be
        maintained by specialized operators (swap, PMX crossover, 2-opt, etc.).
        
        Parameters
        ----------
        X : np.ndarray
            Tour(s), shape (n, num_cities) or (num_cities,).
        
        Returns
        -------
        X : np.ndarray
            Unchanged input.
        
        Notes
        -----
        For invalid permutations (duplicates/missing cities), use
        repair functions from core.utils (e.g., repair_permutation).
        """
        return X


if __name__ == "__main__":
    # Demo
    print("=" * 70)
    print("TSP PROBLEM COMPREHENSIVE DEMO")
    print("=" * 70)
    
    # Test 1: Simple 4-city square problem
    print("\n[TEST 1] 4-City Square TSP")
    print("-" * 70)
    coords = np.array([
        [0, 0],  # City 0
        [1, 0],  # City 1
        [1, 1],  # City 2
        [0, 1]   # City 3
    ])
    
    problem = TSPProblem(coords)
    print(f"Number of cities: {problem.num_cities}")
    print(f"Coordinates:\n{problem.coords}")
    print(f"\nDistance matrix:\n{problem.distance_matrix}")
    
    # Optimal tour (rectangle perimeter)
    tour_opt = np.array([0, 1, 2, 3])  # 0->1->2->3->0
    cost_opt = problem.evaluate(tour_opt)
    print(f"\nOptimal tour [0,1,2,3]: cost = {cost_opt:.4f} (expected: 4.0)")
    assert abs(cost_opt - 4.0) < 1e-10, "Optimal tour cost incorrect"
    
    # Suboptimal tour (diagonal crossing)
    tour_2 = np.array([0, 2, 1, 3])  # 0->2->1->3->0
    cost_2 = problem.evaluate(tour_2)
    print(f"Crossing tour [0,2,1,3]: cost = {cost_2:.4f} (expected: ~2.83)")
    assert cost_2 > cost_opt, "Crossing tour should be longer"
    
    # Test 2: Random tour generation
    print("\n[TEST 2] Random Tour Generation")
    print("-" * 70)
    rng = np.random.RandomState(42)
    random_tours = problem.init_solution(rng, n=5)
    print(f"Generated 5 random tours, shape: {random_tours.shape}")
    
    costs = []
    for i, tour in enumerate(random_tours):
        # Verify valid permutation
        assert len(np.unique(tour)) == problem.num_cities, f"Tour {i} has duplicates"
        assert set(tour) == set(range(problem.num_cities)), f"Tour {i} missing cities"
        
        cost = problem.evaluate(tour)
        costs.append(cost)
        print(f"  Tour {i}: {tour} -> cost = {cost:.4f}")
    
    print(f"Cost range: [{min(costs):.4f}, {max(costs):.4f}]")
    
    # Test 3: Larger random problem
    print("\n[TEST 3] Larger Random TSP (20 cities)")
    print("-" * 70)
    rng = np.random.RandomState(123)
    coords_20 = rng.rand(20, 2) * 100  # 20 cities in [0, 100]^2
    problem_20 = TSPProblem(coords_20)
    
    # Generate 10 random tours and track best
    tours_20 = problem_20.init_solution(rng, n=10)
    costs_20 = [problem_20.evaluate(tour) for tour in tours_20]
    best_idx = np.argmin(costs_20)
    
    print(f"Number of cities: {problem_20.num_cities}")
    print(f"Search space size: {np.math.factorial(20):,} possible tours")
    print(f"Generated 10 random tours:")
    print(f"  Best random tour cost:  {costs_20[best_idx]:.4f}")
    print(f"  Worst random tour cost: {max(costs_20):.4f}")
    print(f"  Average cost:           {np.mean(costs_20):.4f}")
    print(f"  Std deviation:          {np.std(costs_20):.4f}")
    
    # Test 4: Reproducibility
    print("\n[TEST 4] Reproducibility Test")
    print("-" * 70)
    rng1 = np.random.RandomState(999)
    rng2 = np.random.RandomState(999)
    
    tours1 = problem_20.init_solution(rng1, n=3)
    tours2 = problem_20.init_solution(rng2, n=3)
    
    print(f"Tours identical: {np.array_equal(tours1, tours2)}")
    assert np.array_equal(tours1, tours2), "Reproducibility failed"
    print("✓ Reproducibility test PASSED")
    
    # Test 5: Input Validation
    print("\n[TEST 5] Input Validation")
    print("-" * 70)
    
    # Test with invalid coordinates (too few cities)
    try:
        invalid_coords = np.array([[0, 0]])  # Only 1 city
        problem_invalid = TSPProblem(invalid_coords)
        print("✗ Should have raised ValueError for 1 city")
    except ValueError as e:
        print(f"✓ Correctly rejected 1 city: {e}")
    
    # Test with invalid shape
    try:
        invalid_shape = np.array([0, 1, 2, 3])  # 1D array
        problem_invalid = TSPProblem(invalid_shape)
        print("✗ Should have raised ValueError for 1D array")
    except ValueError as e:
        print(f"✓ Correctly rejected 1D array: {e}")
    
    print("\n" + "=" * 70)
    print("All TSP problem tests PASSED! ✓")
    print("=" * 70)