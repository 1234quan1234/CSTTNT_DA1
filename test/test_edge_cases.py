"""
Unit tests for edge cases and boundary conditions.

This module tests extreme scenarios, invalid inputs, and boundary values
to ensure algorithms are robust and handle edge cases gracefully.
"""

import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.problems.continuous.rastrigin import RastriginProblem
from src.problems.discrete.knapsack import KnapsackProblem
from src.swarm.fa import FireflyContinuousOptimizer, FireflyKnapsackOptimizer
from src.classical.simulated_annealing import SimulatedAnnealingOptimizer
from src.classical.hill_climbing import HillClimbingOptimizer
from src.classical.genetic_algorithm import GeneticAlgorithmOptimizer


class TestExtremeDimensions(unittest.TestCase):
    """
    Test algorithms with extreme dimensional problems.
    
    Tests cover:
    - Very low dimensions (d=1)
    - High dimensions (d=100)
    - Very high dimensions (d=1000)
    """
    
    def test_1d_problem(self):
        """
        Test with 1-dimensional problem (d=1).
        
        Ensures algorithms work with minimal dimensionality.
        """
        problem = RastriginProblem(dim=1)
        
        # Test FA
        fa = FireflyContinuousOptimizer(problem, n_fireflies=5, seed=42)
        best_sol, best_fit, history, _ = fa.run(max_iter=10)
        
        self.assertEqual(len(best_sol), 1, "Solution should be 1D")
        self.assertIsInstance(best_fit, (float, np.floating))
        self.assertEqual(len(history), 10)
        self.assertTrue(np.isfinite(best_fit), "Fitness should be finite")
    
    def test_high_dimensional_problem(self):
        """
        Test with high-dimensional problem (d=100).
        
        Ensures algorithms scale to higher dimensions without crashing.
        """
        problem = RastriginProblem(dim=100)
        
        # Test with smaller population for speed
        fa = FireflyContinuousOptimizer(problem, n_fireflies=10, seed=42)
        best_sol, best_fit, history, _ = fa.run(max_iter=5)
        
        self.assertEqual(len(best_sol), 100)
        self.assertTrue(np.all(np.isfinite(best_sol)), "Solution should be finite")
        self.assertTrue(np.isfinite(best_fit))
        
        # Check convergence (should improve or stay same)
        self.assertLessEqual(history[-1], history[0] * 1.1,
                           "Should not worsen significantly")
    
    def test_very_high_dimensional_problem(self):
        """
        Test with very high-dimensional problem (d=1000).
        
        Expects slower runtime but should not crash.
        """
        problem = RastriginProblem(dim=1000)
        
        # Use minimal population and iterations
        fa = FireflyContinuousOptimizer(problem, n_fireflies=5, seed=42)
        
        import time
        start = time.time()
        best_sol, best_fit, history, _ = fa.run(max_iter=2)
        elapsed = time.time() - start
        
        self.assertEqual(len(best_sol), 1000)
        self.assertTrue(elapsed < 60, "Should complete within 60 seconds")
        self.assertTrue(np.isfinite(best_fit))


class TestKnapsackEdgeCases(unittest.TestCase):
    """
    Test Knapsack problem with edge cases.
    
    Tests cover:
    - Zero capacity
    - Empty items
    - All items too heavy
    - Single item
    """
    
    def test_zero_capacity(self):
        """
        Test with capacity=0.
        
        Expected: Empty solution (all zeros).
        """
        values = np.array([10, 20, 30])
        weights = np.array([5, 10, 15])
        capacity = 0.0
        
        problem = KnapsackProblem(values, weights, capacity)
        
        fa = FireflyKnapsackOptimizer(problem, n_fireflies=5, seed=42)
        best_sol, best_fit, history, _ = fa.run(max_iter=5)
        
        # Should select no items
        self.assertTrue(np.sum(best_sol) == 0, "Should select no items")
        self.assertEqual(best_fit, 0.0, "Fitness should be 0")
    
    def test_empty_items(self):
        """
        Test with no items.
        
        Expected: Empty solution.
        """
        values = np.array([])
        weights = np.array([])
        capacity = 100.0
        
        problem = KnapsackProblem(values, weights, capacity)
        
        fa = FireflyKnapsackOptimizer(problem, n_fireflies=5, seed=42)
        best_sol, best_fit, history, _ = fa.run(max_iter=5)
        
        self.assertEqual(len(best_sol), 0)
        self.assertEqual(best_fit, 0.0)
    
    def test_all_items_too_heavy(self):
        """
        Test when all items exceed capacity.
        
        Expected: Empty solution or infeasible handling.
        """
        values = np.array([100, 200, 300])
        weights = np.array([50, 60, 70])
        capacity = 10.0  # Much smaller than any weight
        
        problem = KnapsackProblem(values, weights, capacity)
        
        fa = FireflyKnapsackOptimizer(problem, n_fireflies=5, seed=42)
        best_sol, best_fit, history, _ = fa.run(max_iter=10)
        
        # Check feasibility
        total_weight = np.sum(best_sol * weights)
        self.assertTrue(total_weight <= capacity or np.sum(best_sol) == 0,
                       "Solution should be feasible or empty")
    
    def test_single_item(self):
        """
        Test with single item.
        
        Expected: Take it if it fits, otherwise leave it.
        """
        # Case 1: Item fits
        values = np.array([100])
        weights = np.array([50])
        capacity = 60.0
        
        problem = KnapsackProblem(values, weights, capacity)
        fa = FireflyKnapsackOptimizer(problem, n_fireflies=5, seed=42)
        best_sol, best_fit, _, _ = fa.run(max_iter=10)
        
        self.assertEqual(best_sol[0], 1, "Should take item if it fits")
        self.assertEqual(-best_fit, 100, "Should get full value")
        
        # Case 2: Item doesn't fit
        capacity = 40.0
        problem = KnapsackProblem(values, weights, capacity)
        fa = FireflyKnapsackOptimizer(problem, n_fireflies=5, seed=42)
        best_sol, best_fit, _, _ = fa.run(max_iter=10)
        
        self.assertEqual(best_sol[0], 0, "Should not take item if too heavy")


class TestBoundaryValues(unittest.TestCase):
    """
    Test algorithms with boundary parameter values.
    
    Tests cover:
    - max_iter=1
    - n_particles=1
    - Degenerate parameter cases (alpha=0, beta=0)
    """
    
    def test_single_iteration(self):
        """
        Test with max_iter=1.
        
        Should run exactly 1 iteration and return valid results.
        """
        problem = RastriginProblem(dim=3)
        
        fa = FireflyContinuousOptimizer(problem, n_fireflies=5, seed=42)
        best_sol, best_fit, history, trajectory = fa.run(max_iter=1)
        
        self.assertEqual(len(history), 1, "Should have exactly 1 history entry")
        self.assertEqual(len(trajectory), 1, "Should have 1 trajectory entry")
        self.assertTrue(np.isfinite(best_fit))
    
    def test_single_particle(self):
        """
        Test with n_particles=1 (minimal population).
        
        Should still work but may converge poorly.
        """
        problem = RastriginProblem(dim=3)
        
        # Note: FA requires n_fireflies >= 2, so test with HC/SA instead
        hc = HillClimbingOptimizer(problem, num_neighbors=5, seed=42)
        best_sol, best_fit, history, _ = hc.run(max_iter=10)
        
        self.assertEqual(len(best_sol), 3)
        self.assertTrue(np.isfinite(best_fit))
        self.assertEqual(len(history), 10)
    
    def test_degenerate_alpha_zero(self):
        """
        Test FA with alpha=0 (no randomization).
        
        Should still run but may converge prematurely.
        """
        problem = RastriginProblem(dim=3)
        
        fa = FireflyContinuousOptimizer(
            problem,
            n_fireflies=5,
            alpha=0.0,  # No random exploration
            seed=42
        )
        best_sol, best_fit, history, _ = fa.run(max_iter=10)
        
        self.assertTrue(np.isfinite(best_fit))
        self.assertEqual(len(history), 10)
    
    def test_degenerate_beta_zero(self):
        """
        Test FA with beta0=0 (no attraction).
        
        Fireflies won't attract each other, only random walk.
        """
        problem = RastriginProblem(dim=3)
        
        fa = FireflyContinuousOptimizer(
            problem,
            n_fireflies=5,
            beta0=0.0,  # No attraction
            seed=42
        )
        best_sol, best_fit, history, _ = fa.run(max_iter=10)
        
        self.assertTrue(np.isfinite(best_fit))
        # May not converge well, but should not crash


class TestInvalidInputs(unittest.TestCase):
    """
    Test that algorithms properly reject invalid inputs.
    
    Tests cover:
    - Negative dimensions
    - Negative max_iter
    - Invalid parameter ranges
    """
    
    def test_negative_dimension(self):
        """Test that negative dimension raises ValueError."""
        with self.assertRaises(ValueError):
            RastriginProblem(dim=-5)
    
    def test_zero_dimension(self):
        """Test that zero dimension raises ValueError."""
        with self.assertRaises(ValueError):
            RastriginProblem(dim=0)
    
    def test_negative_max_iter(self):
        """Test that negative max_iter is handled."""
        problem = RastriginProblem(dim=3)
        fa = FireflyContinuousOptimizer(problem, seed=42)
        
        # Should either raise error or treat as 0
        with self.assertRaises((ValueError, AssertionError)):
            fa.run(max_iter=-10)
    
    def test_invalid_n_fireflies(self):
        """Test that invalid n_fireflies raises ValueError."""
        problem = RastriginProblem(dim=3)
        
        with self.assertRaises(ValueError):
            FireflyContinuousOptimizer(problem, n_fireflies=0)
        
        with self.assertRaises(ValueError):
            FireflyContinuousOptimizer(problem, n_fireflies=-5)
    
    def test_invalid_alpha_range(self):
        """Test that alpha outside [0,1] raises ValueError."""
        problem = RastriginProblem(dim=3)
        
        with self.assertRaises(ValueError):
            FireflyContinuousOptimizer(problem, alpha=-0.1)
        
        with self.assertRaises(ValueError):
            FireflyContinuousOptimizer(problem, alpha=1.5)
    
    def test_invalid_knapsack_parameters(self):
        """Test that invalid Knapsack parameters raise errors."""
        # Mismatched sizes
        with self.assertRaises((ValueError, AssertionError)):
            KnapsackProblem(
                values=np.array([10, 20]),
                weights=np.array([5, 10, 15]),  # Different length
                capacity=100
            )
        
        # Negative capacity
        with self.assertRaises(ValueError):
            KnapsackProblem(
                values=np.array([10, 20]),
                weights=np.array([5, 10]),
                capacity=-10
            )


class TestNumericalStability(unittest.TestCase):
    """
    Test numerical stability with extreme values.
    
    Tests cover:
    - Very large fitness values
    - Very small fitness values
    - Overflow/underflow scenarios
    """
    
    def test_large_fitness_values(self):
        """Test handling of large fitness values."""
        problem = RastriginProblem(dim=100)  # Will produce large fitness
        
        fa = FireflyContinuousOptimizer(problem, n_fireflies=5, seed=42)
        best_sol, best_fit, history, _ = fa.run(max_iter=5)
        
        # Should handle large values without overflow
        self.assertTrue(np.isfinite(best_fit), "Should not overflow")
        self.assertTrue(np.all(np.isfinite(history)), "History should be finite")
    
    def test_array_operations_stability(self):
        """Test that array operations are numerically stable."""
        problem = RastriginProblem(dim=10)
        
        fa = FireflyContinuousOptimizer(
            problem,
            n_fireflies=20,
            gamma=10.0,  # High gamma might cause numerical issues
            seed=42
        )
        best_sol, best_fit, history, _ = fa.run(max_iter=10)
        
        # Check all values are finite
        self.assertTrue(np.all(np.isfinite(best_sol)))
        self.assertTrue(np.isfinite(best_fit))
        self.assertTrue(np.all(np.isfinite(history)))


if __name__ == '__main__':
    unittest.main()
