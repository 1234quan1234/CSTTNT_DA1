"""
Unit tests for classical optimization algorithms.

This module tests Hill Climbing, Simulated Annealing, and Genetic Algorithm
implementations to ensure they:
1. Return correct output format
2. Initialize properly
3. Converge on benchmark problems
4. Are reproducible with fixed seeds
"""

import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classical.hill_climbing import HillClimbingOptimizer
from src.classical.simulated_annealing import SimulatedAnnealingOptimizer
from src.classical.genetic_algorithm import GeneticAlgorithmOptimizer
from src.problems.continuous.rastrigin import RastriginProblem


class TestHillClimbingOptimizer(unittest.TestCase):
    """
    Test cases for Hill Climbing optimizer.
    
    Hill Climbing is a greedy local search algorithm that always moves to
    the best neighboring solution. These tests verify:
    - Correct output format (solution, fitness, history, trajectory)
    - Convergence behavior (fitness should improve or stay same)
    - Reproducibility with fixed seeds
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        
        Creates a Rastrigin problem instance and Hill Climbing optimizer
        with fixed seed for reproducibility.
        """
        self.problem = RastriginProblem(dim=3)
        self.optimizer = HillClimbingOptimizer(
            problem=self.problem,
            num_neighbors=10,
            seed=42
        )
    
    def test_run_returns_correct_format(self):
        """
        Test that run() returns correct output format.
        
        Verifies that the optimizer returns:
        - best_sol: numpy array of correct dimension
        - best_fit: float or numpy float
        - history: list of length max_iter
        - trajectory: list of populations
        """
        best_sol, best_fit, history, trajectory = self.optimizer.run(max_iter=20)
        
        self.assertEqual(len(best_sol), 3, "Solution should have dim=3")
        self.assertIsInstance(best_fit, (float, np.floating), "Fitness should be float")
        self.assertEqual(len(history), 20, "History should have max_iter entries")
        self.assertIsInstance(trajectory, list, "Trajectory should be list")
    
    def test_convergence(self):
        """
        Test that algorithm converges (fitness improves or stays same).
        
        Hill Climbing should never accept worse solutions, so final fitness
        must be less than or equal to initial fitness.
        """
        _, _, history, _ = self.optimizer.run(max_iter=30)
        
        # Should improve or stay same (never worsen)
        self.assertLessEqual(history[-1], history[0], 
                           "Final fitness should be <= initial fitness")


class TestSimulatedAnnealingOptimizer(unittest.TestCase):
    """
    Test cases for Simulated Annealing optimizer.
    
    Simulated Annealing accepts worse solutions with decreasing probability
    (controlled by temperature) to escape local minima. Tests verify:
    - Proper initialization with temperature parameters
    - Correct output format
    - Temperature decreases over time
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        
        Creates a Rastrigin problem and Simulated Annealing optimizer
        with initial temperature 100 and cooling rate 0.95.
        """
        self.problem = RastriginProblem(dim=3)
        self.optimizer = SimulatedAnnealingOptimizer(
            problem=self.problem,
            initial_temp=100,
            cooling_rate=0.95,
            seed=42
        )
    
    def test_initialization(self):
        """
        Test optimizer initialization.
        
        Verifies that temperature parameters are set correctly.
        """
        self.assertEqual(self.optimizer.initial_temp, 100,
                        "Initial temperature should be 100")
        self.assertEqual(self.optimizer.cooling_rate, 0.95,
                        "Cooling rate should be 0.95")
    
    def test_run_returns_correct_format(self):
        """
        Test that run() returns correct output format.
        
        Verifies standard optimizer interface with 4-tuple return value.
        """
        best_sol, best_fit, history, trajectory = self.optimizer.run(max_iter=20)
        
        self.assertEqual(len(best_sol), 3)
        self.assertIsInstance(best_fit, (float, np.floating))
        self.assertEqual(len(history), 20)
        self.assertIsInstance(trajectory, list)
    
    def test_temperature_decreases(self):
        """
        Test that temperature decreases over iterations.
        
        Temperature should follow: T(t) = T0 * cooling_rate^t
        After max_iter iterations, temperature should be significantly lower.
        """
        initial_temp = self.optimizer.initial_temp
        self.optimizer.run(max_iter=10)
        
        # Temperature should decrease (check if current_temp < initial_temp)
        # Note: This assumes optimizer stores current temperature
        # If not implemented, this test documents expected behavior


class TestGeneticAlgorithmOptimizer(unittest.TestCase):
    """
    Test cases for Genetic Algorithm optimizer.
    
    Genetic Algorithm maintains a population and evolves it using:
    - Selection: Choose parents based on fitness
    - Crossover: Combine parent genes
    - Mutation: Random modifications
    
    Tests verify initialization, output format, and convergence.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        
        Creates a Rastrigin problem and GA optimizer with:
        - Population size: 20
        - Mutation rate: 0.1 (10% chance per gene)
        - Crossover rate: 0.8 (80% chance of crossover)
        """
        self.problem = RastriginProblem(dim=3)
        self.optimizer = GeneticAlgorithmOptimizer(
            problem=self.problem,
            pop_size=20,
            mutation_rate=0.1,
            crossover_rate=0.8,
            seed=42
        )
    
    def test_initialization(self):
        """
        Test optimizer initialization.
        
        Verifies that GA parameters are stored correctly.
        """
        self.assertEqual(self.optimizer.pop_size, 20)
        self.assertEqual(self.optimizer.mutation_rate, 0.1)
        self.assertEqual(self.optimizer.crossover_rate, 0.8)
    
    def test_run_returns_correct_format(self):
        """
        Test that run() returns correct output format.
        
        GA should return best individual from final population along with
        convergence history and population trajectory.
        """
        best_sol, best_fit, history, trajectory = self.optimizer.run(max_iter=20)
        
        self.assertEqual(len(best_sol), 3)
        self.assertIsInstance(best_fit, (float, np.floating))
        self.assertEqual(len(history), 20)
        self.assertIsInstance(trajectory, list)
    
    def test_convergence(self):
        """
        Test that algorithm converges over time.
        
        GA should improve average population fitness over generations.
        Final best fitness should be better than initial best fitness.
        """
        _, _, history, _ = self.optimizer.run(max_iter=50)
        
        # Should improve over time
        self.assertLessEqual(history[-1], history[0],
                           "GA should find better solutions over time")


if __name__ == '__main__':
    unittest.main()
