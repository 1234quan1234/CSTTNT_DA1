"""
Traveling Salesman Problem (TSP) optimization problem.

The TSP is a classic combinatorial optimization problem where the goal is to find
the shortest route that visits all cities exactly once and returns to the origin.

References
----------
.. [1] https://en.wikipedia.org/wiki/Travelling_salesman_problem
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from core.problem_base import ProblemBase
from typing import Literal


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
        City coordinates, shape (num_cities, 2).
        Each row is the (x, y) position of a city.
    
    Attributes
    ----------
    coords : np.ndarray
        City coordinates, shape (num_cities, 2).
    num_cities : int
        Number of cities.
    distance_matrix : np.ndarray
        Precomputed pairwise distance matrix, shape (num_cities, num_cities).
    
    Examples
    --------
    >>> coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    >>> problem = TSPProblem(coords)
    >>> tour = np.array([0, 1, 2, 3])  # Visit in order
    >>> cost = problem.evaluate(tour)
    >>> print(f"Tour cost: {cost}")
    """
    
    def __init__(self, coords: np.ndarray):
        """
        Initialize TSP problem.
        
        Parameters
        ----------
        coords : np.ndarray
            City coordinates, shape (num_cities, 2).
        """
        self.coords = np.array(coords, dtype=float)
        self.num_cities = len(coords)
        
        # Precompute distance matrix for efficiency
        self.distance_matrix = self._compute_distance_matrix()
    
    def _compute_distance_matrix(self) -> np.ndarray:
        """
        Compute Euclidean distance matrix between all city pairs.
        
        Returns
        -------
        dist_matrix : np.ndarray
            Distance matrix, shape (num_cities, num_cities).
        """
        n = self.num_cities
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(self.coords[i] - self.coords[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
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
        
        Returns
        -------
        tour_length : float
            Total Euclidean distance of the tour.
        """
        tour = x.astype(int)
        total_distance = 0.0
        
        # Distance from city to city along the tour
        for i in range(self.num_cities - 1):
            total_distance += self.distance_matrix[tour[i], tour[i + 1]]
        
        # Return to starting city
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
            Random number generator.
        n : int, default=1
            Number of tours to generate.
        
        Returns
        -------
        tours : np.ndarray
            Array of shape (n, num_cities) where each row is a random permutation.
        """
        tours = np.zeros((n, self.num_cities), dtype=int)
        
        for i in range(n):
            tours[i] = rng.permutation(self.num_cities)
        
        return tours
    
    def clip(self, X: np.ndarray) -> np.ndarray:
        """
        For TSP, clipping is not applicable (discrete problem).
        
        Returns the input unchanged. Validity of permutations should be
        maintained by specialized operators (swap, 2-opt, etc.).
        
        Parameters
        ----------
        X : np.ndarray
            Tour(s), shape (n, num_cities) or (num_cities,).
        
        Returns
        -------
        X : np.ndarray
            Unchanged input.
        """
        return X


if __name__ == "__main__":
    # Demo
    print("TSP Problem Demo")
    print("=" * 50)
    
    # Create a simple 4-city problem (square)
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
    
    # Test optimal tour (rectangle perimeter)
    tour_opt = np.array([0, 1, 2, 3])  # 0->1->2->3->0
    cost_opt = problem.evaluate(tour_opt)
    print(f"\nTour [0,1,2,3]: cost = {cost_opt:.4f} (expected: 4.0)")
    
    # Test another tour
    tour_2 = np.array([0, 2, 1, 3])  # 0->2->1->3->0 (crosses)
    cost_2 = problem.evaluate(tour_2)
    print(f"Tour [0,2,1,3]: cost = {cost_2:.4f}")
    
    # Generate random tours
    rng = np.random.RandomState(42)
    random_tours = problem.init_solution(rng, n=3)
    print(f"\nGenerated 3 random tours, shape: {random_tours.shape}")
    for i, tour in enumerate(random_tours):
        cost = problem.evaluate(tour)
        print(f"  Tour {i}: {tour} -> cost = {cost:.4f}")
    
    # Test with larger random problem
    print("\n" + "=" * 50)
    rng = np.random.RandomState(123)
    coords_10 = rng.rand(10, 2) * 10  # 10 cities in [0, 10]^2
    problem_10 = TSPProblem(coords_10)
    tour_10 = problem_10.init_solution(rng, n=1)[0]
    cost_10 = problem_10.evaluate(tour_10)
    print(f"10-city TSP: random tour cost = {cost_10:.4f}")
    
    print("\nTSP problem test passed!")
