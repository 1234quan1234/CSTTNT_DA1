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
        self.assertAlmostEqual(self.problem.dist_matrix[0, 1], 1.0, places=10)
        # Distance from city 0 to city 2 should be sqrt(2)
        self.assertAlmostEqual(self.problem.dist_matrix[0, 2], np.sqrt(2), places=10)
        # Distance from city i to city i should be 0
        for i in range(4):
            self.assertEqual(self.problem.dist_matrix[i, i], 0.0)
        # Distance matrix should be symmetric
        for i in range(4):
            for j in range(4):
                self.assertEqual(self.problem.dist_matrix[i, j], 
                               self.problem.dist_matrix[j, i])
    
    def test_tour_length(self):
        """Test tour length calculation."""
        # Square tour: 0->1->2->3->0, length should be 4
        tour = [0, 1, 2, 3]
        length = self.problem.fitness(tour)
        self.assertAlmostEqual(length, 4.0, places=10)
    
    def test_random_solution(self):
        """Test random solution generation."""
        solution = self.problem.random_solution()
        self.assertEqual(len(solution), 4)
        self.assertEqual(set(solution), {0, 1, 2, 3})
    
    def test_invalid_tour(self):
        """Test that invalid tours are handled."""
        # Tour with missing city
        invalid_tour = [0, 1, 2]
        with self.assertRaises((ValueError, IndexError)):
            self.problem.fitness(invalid_tour)


if __name__ == '__main__':
    unittest.main()
