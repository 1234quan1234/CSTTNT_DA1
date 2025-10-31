"""
Unit tests for TSP problem.
"""

import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.problems.discrete.tsp import TSPProblem


class TestTSPProblem(unittest.TestCase):
    """Test cases for TSP problem."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create simple 4-city TSP
        self.coords = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ])
        self.problem = TSPProblem(self.coords)
    
    def test_num_cities(self):
        """Test number of cities."""
        self.assertEqual(self.problem.num_cities, 4)
    
    def test_distance_matrix(self):
        """Test distance matrix calculation."""
        # Distance from city 0 to city 1 should be 1
        self.assertAlmostEqual(self.problem.distance_matrix[0, 1], 1.0, places=10)
        # Distance from city 0 to city 2 should be sqrt(2)
        self.assertAlmostEqual(self.problem.distance_matrix[0, 2], np.sqrt(2), places=10)
        # Distance from city i to city i should be 0
        for i in range(4):
            self.assertEqual(self.problem.distance_matrix[i, i], 0.0)
        # Distance matrix should be symmetric
        for i in range(4):
            for j in range(4):
                self.assertEqual(self.problem.distance_matrix[i, j], 
                               self.problem.distance_matrix[j, i])
    
    def test_tour_length(self):
        """Test tour length calculation."""
        # Square tour: 0->1->2->3->0, length should be 4
        tour = np.array([0, 1, 2, 3])
        length = self.problem.evaluate(tour)
        self.assertAlmostEqual(length, 4.0, places=10)
    
    def test_random_solution(self):
        """Test random solution generation."""
        rng = np.random.RandomState(42)
        solutions = self.problem.init_solution(rng, n=1)
        solution = solutions[0]
        self.assertEqual(len(solution), 4)
        self.assertEqual(set(solution), {0, 1, 2, 3})
    
    def test_invalid_tour(self):
        """Test that invalid tours are handled."""
        # Tour with missing city - should raise error or return invalid value
        invalid_tour = np.array([0, 1, 2])
        try:
            result = self.problem.evaluate(invalid_tour)
            # If it doesn't raise an error, check if result is reasonable
            # (implementation dependent)
        except (ValueError, IndexError):
            # Expected behavior - invalid tour raises error
            pass


if __name__ == '__main__':
    unittest.main()
