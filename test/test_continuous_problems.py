"""
Unit tests for continuous optimization problems.
"""

import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.problems.continuous.rastrigin import RastriginProblem


class TestRastriginProblem(unittest.TestCase):
    """Test cases for Rastrigin function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.problem = RastriginProblem(dim=3)
    
    def test_dimension(self):
        """Test problem dimension."""
        self.assertEqual(self.problem.dim, 3)
    
    def test_bounds(self):
        """Test problem bounds."""
        self.assertEqual(len(self.problem.lower), 3)
        self.assertEqual(len(self.problem.upper), 3)
        self.assertTrue(all(self.problem.lower == -5.12))
        self.assertTrue(all(self.problem.upper == 5.12))
    
    def test_optimum_value(self):
        """Test that optimum (origin) gives zero fitness."""
        optimum = np.zeros(3)
        fitness = self.problem.evaluate(optimum)
        self.assertAlmostEqual(fitness, 0.0, places=10)
    
    def test_fitness_positive(self):
        """Test that fitness is always non-negative."""
        rng = np.random.RandomState(42)
        for _ in range(10):
            x = rng.randn(3) * 5
            fitness = self.problem.evaluate(x)
            self.assertGreaterEqual(fitness, 0.0)
    
    def test_random_solution(self):
        """Test random solution generation."""
        rng = np.random.RandomState(42)
        solutions = self.problem.init_solution(rng, n=1)
        solution = solutions[0]
        self.assertEqual(len(solution), 3)
        self.assertTrue(all(solution >= -5.12))
        self.assertTrue(all(solution <= 5.12))
    
    def test_multimodality(self):
        """Test that Rastrigin is multimodal (has local minima)."""
        x_local = np.array([0.99, 0.99, 0.99])
        f_local = self.problem.evaluate(x_local)
        
        self.assertGreater(f_local, 0.0)
        self.assertLess(f_local, 10.0)


if __name__ == '__main__':
    unittest.main()