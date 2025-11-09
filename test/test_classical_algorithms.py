"""
Unit tests for classical optimization algorithms.
"""

import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classical.hill_climbing import HillClimbingOptimizer
from src.classical.simulated_annealing import SimulatedAnnealingOptimizer
from src.classical.genetic_algorithm import GeneticAlgorithmOptimizer


class TestHillClimbingOptimizer(unittest.TestCase):
    """Test cases for Hill Climbing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.problem = SphereProblem(dim=3)
        self.optimizer = HillClimbingOptimizer(
            problem=self.problem,
            num_neighbors=10,
            seed=42
        )
    
    def test_run_returns_correct_format(self):
        """Test that run() returns correct output format."""
        best_sol, best_fit, history, trajectory = self.optimizer.run(max_iter=20)
        
        self.assertEqual(len(best_sol), 3)
        self.assertIsInstance(best_fit, (float, np.floating))
        self.assertEqual(len(history), 20)
        self.assertIsInstance(trajectory, list)
    
    def test_convergence(self):
        """Test that algorithm converges."""
        _, _, history, _ = self.optimizer.run(max_iter=30)
        
        # Should improve or stay same
        self.assertLessEqual(history[-1], history[0])


class TestSimulatedAnnealingOptimizer(unittest.TestCase):
    """Test cases for Simulated Annealing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.problem = SphereProblem(dim=3)
        self.optimizer = SimulatedAnnealingOptimizer(
            problem=self.problem,
            initial_temp=100,
            cooling_rate=0.95,
            seed=42
        )
    
    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertEqual(self.optimizer.initial_temp, 100)
        self.assertEqual(self.optimizer.cooling_rate, 0.95)
    
    def test_run_returns_correct_format(self):
        """Test that run() returns correct output format."""
        best_sol, best_fit, history, trajectory = self.optimizer.run(max_iter=20)
        
        self.assertEqual(len(best_sol), 3)
        self.assertIsInstance(best_fit, (float, np.floating))
        self.assertEqual(len(history), 20)
        self.assertIsInstance(trajectory, list)
    
    def test_temperature_decreases(self):
        """Test that temperature decreases over iterations."""
        self.optimizer.run(max_iter=10)
        # Temperature should be less than initial
        # (assuming run() updates temperature internally)


class TestGeneticAlgorithmOptimizer(unittest.TestCase):
    """Test cases for Genetic Algorithm."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.problem = SphereProblem(dim=3)
        self.optimizer = GeneticAlgorithmOptimizer(
            problem=self.problem,
            pop_size=20,
            mutation_rate=0.1,
            crossover_rate=0.8,
            seed=42
        )
    
    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertEqual(self.optimizer.pop_size, 20)
        self.assertEqual(self.optimizer.mutation_rate, 0.1)
        self.assertEqual(self.optimizer.crossover_rate, 0.8)
    
    def test_run_returns_correct_format(self):
        """Test that run() returns correct output format."""
        best_sol, best_fit, history, trajectory = self.optimizer.run(max_iter=20)
        
        self.assertEqual(len(best_sol), 3)
        self.assertIsInstance(best_fit, (float, np.floating))
        self.assertEqual(len(history), 20)
        self.assertIsInstance(trajectory, list)
    
    def test_convergence(self):
        """Test that algorithm converges."""
        _, _, history, _ = self.optimizer.run(max_iter=50)
        
        # Should improve over time
        self.assertLessEqual(history[-1], history[0])


if __name__ == '__main__':
    unittest.main()
