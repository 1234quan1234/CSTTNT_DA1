"""
Unit tests for Firefly Algorithm.
"""

import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.problems.continuous.sphere import SphereProblem
from src.problems.discrete.tsp import TSPProblem
from src.swarm.fa import FireflyContinuousOptimizer, FireflyDiscreteTSPOptimizer


class TestFireflyContinuousOptimizer(unittest.TestCase):
    """Test cases for continuous Firefly Algorithm."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.problem = SphereProblem(dim=3)
        self.optimizer = FireflyContinuousOptimizer(
            problem=self.problem,
            n_fireflies=10,
            alpha=0.2,
            beta0=1.0,
            gamma=1.0,
            seed=42
        )
    
    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertEqual(self.optimizer.n_fireflies, 10)
        self.assertEqual(self.optimizer.alpha, 0.2)
        self.assertEqual(self.optimizer.beta0, 1.0)
        self.assertEqual(self.optimizer.gamma, 1.0)
    
    def test_run_returns_correct_format(self):
        """Test that run() returns correct output format."""
        best_sol, best_fit, history, trajectory = self.optimizer.run(max_iter=10)
        
        self.assertEqual(len(best_sol), 3)
        self.assertIsInstance(best_fit, (float, np.floating))
        self.assertEqual(len(history), 10)
        self.assertIsInstance(trajectory, list)
    
    def test_convergence(self):
        """Test that algorithm converges (fitness improves)."""
        _, _, history, _ = self.optimizer.run(max_iter=50)
        
        # Final fitness should be better than or equal to initial
        self.assertLessEqual(history[-1], history[0])
    
    def test_deterministic_with_seed(self):
        """Test that results are deterministic with same seed."""
        opt1 = FireflyContinuousOptimizer(self.problem, n_fireflies=10, seed=123)
        opt2 = FireflyContinuousOptimizer(self.problem, n_fireflies=10, seed=123)
        
        _, fit1, _, _ = opt1.run(max_iter=20)
        _, fit2, _, _ = opt2.run(max_iter=20)
        
        self.assertAlmostEqual(fit1, fit2, places=10)


class TestFireflyDiscreteTSPOptimizer(unittest.TestCase):
    """Test cases for discrete TSP Firefly Algorithm."""
    
    def setUp(self):
        """Set up test fixtures."""
        coords = np.random.RandomState(42).rand(5, 2) * 10
        self.problem = TSPProblem(coords)
        self.optimizer = FireflyDiscreteTSPOptimizer(
            problem=self.problem,
            n_fireflies=8,
            alpha_swap=0.2,
            max_swaps_per_move=2,
            seed=42
        )
    
    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertEqual(self.optimizer.n_fireflies, 8)
        self.assertEqual(self.optimizer.alpha_swap, 0.2)
        self.assertEqual(self.optimizer.max_swaps_per_move, 2)
    
    def test_run_returns_valid_tour(self):
        """Test that run() returns a valid tour."""
        best_tour, best_length, history, trajectory = self.optimizer.run(max_iter=10)
        
        # Check tour validity
        self.assertEqual(len(best_tour), 5)
        self.assertEqual(set(best_tour), {0, 1, 2, 3, 4})
        
        # Check outputs
        self.assertIsInstance(best_length, (float, np.floating))
        self.assertEqual(len(history), 10)
        self.assertIsInstance(trajectory, list)
    
    def test_convergence(self):
        """Test that algorithm improves tour length."""
        _, _, history, _ = self.optimizer.run(max_iter=30)
        
        # Final should be better than or equal to initial
        self.assertLessEqual(history[-1], history[0])


if __name__ == '__main__':
    unittest.main()
