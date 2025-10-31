"""
Unit tests for continuous optimization problems.
"""

import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.problems.continuous.sphere import SphereProblem
from src.problems.continuous.rastrigin import RastriginProblem


class TestSphereProblem(unittest.TestCase):
    """Test cases for Sphere function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.problem = SphereProblem(dim=5)
    
    def test_dimension(self):
        """Test problem dimension."""
        self.assertEqual(self.problem.dim, 5)
    
    def test_bounds(self):
        """Test problem bounds."""
        self.assertEqual(len(self.problem.lower_bound), 5)
        self.assertEqual(len(self.problem.upper_bound), 5)
        self.assertTrue(all(self.problem.lower_bound == -100))
        self.assertTrue(all(self.problem.upper_bound == 100))
    
    def test_optimum_value(self):
        """Test that optimum (origin) gives zero fitness."""
        optimum = np.zeros(5)
        fitness = self.problem.fitness(optimum)
        self.assertAlmostEqual(fitness, 0.0, places=10)
    
    def test_fitness_values(self):
        """Test fitness calculation."""
        x = np.array([1, 2, 3, 4, 5])
        expected = 1 + 4 + 9 + 16 + 25  # sum of squares
        fitness = self.problem.fitness(x)
        self.assertAlmostEqual(fitness, expected, places=10)
    
    def test_random_solution(self):
        """Test random solution generation."""
        solution = self.problem.random_solution()
        self.assertEqual(len(solution), 5)
        self.assertTrue(all(solution >= -100))
        self.assertTrue(all(solution <= 100))


class TestRastriginProblem(unittest.TestCase):
    """Test cases for Rastrigin function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.problem = RastriginProblem(dim=3)
    
    def test_dimension(self):
        """Test problem dimension."""
        self.assertEqual(self.problem.dim, 3)
    
    def test_optimum_value(self):
        """Test that optimum (origin) gives zero fitness."""
        optimum = np.zeros(3)
        fitness = self.problem.fitness(optimum)
        self.assertAlmostEqual(fitness, 0.0, places=10)
    
    def test_fitness_positive(self):
        """Test that fitness is always non-negative."""
        rng = np.random.RandomState(42)
        for _ in range(10):
            x = rng.randn(3) * 5
            fitness = self.problem.fitness(x)
            self.assertGreaterEqual(fitness, 0.0)
    
    def test_random_solution(self):
        """Test random solution generation."""
        solution = self.problem.random_solution()
        self.assertEqual(len(solution), 3)
        self.assertTrue(all(solution >= -5.12))
        self.assertTrue(all(solution <= 5.12))


if __name__ == '__main__':
    unittest.main()
