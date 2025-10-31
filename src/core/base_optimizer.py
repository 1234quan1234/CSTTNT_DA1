"""
Abstract base class for all optimization algorithms.

This module defines the BaseOptimizer interface that all optimization algorithms
(Firefly Algorithm, Hill Climbing, Simulated Annealing, Genetic Algorithm, etc.)
must implement to ensure consistency across the project.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Any, List, Tuple


class BaseOptimizer(ABC):
    """
    Abstract base class for optimization algorithms.
    
    All optimizers in this project (FA, Hill Climbing, SA, GA, etc.) must inherit
    from this class and implement the `run` method with a standardized interface
    for consistency in experiments and benchmarking.
    
    Attributes
    ----------
    problem : ProblemBase
        The optimization problem to solve.
    seed : int or None
        Random seed for reproducibility.
    rng : np.random.RandomState
        Random number generator instance.
    """

    @abstractmethod
    def run(self, max_iter: int) -> Tuple[np.ndarray, float, List[float], List[np.ndarray]]:
        """
        Execute the optimization algorithm for a maximum number of iterations.
        
        This is the main entry point for running the optimization. All algorithms
        must implement this method and return results in the standardized format.
        
        Parameters
        ----------
        max_iter : int
            Maximum number of iterations/generations to run the algorithm.
        
        Returns
        -------
        best_solution : np.ndarray
            The best solution found during optimization.
            - For continuous problems: shape (dim,) real-valued vector
            - For TSP: shape (num_cities,) integer permutation
            - For knapsack: shape (num_items,) binary 0/1 vector
            - For graph coloring: shape (num_nodes,) integer color assignment
        
        best_fitness : float
            The objective function value of the best solution (minimization).
        
        history_best : List[float]
            History of best fitness values at each iteration.
            Length should be max_iter, where history_best[t] is the best
            fitness found up to iteration t.
        
        trajectory : List[np.ndarray]
            Trajectory of the population/solution at each iteration.
            Used for visualization and analysis of convergence behavior.
            - For population-based algorithms (FA, GA): 
              trajectory[t] has shape (population_size, dim) or (population_size, problem_size)
            - For single-solution algorithms (Hill Climbing, SA):
              trajectory[t] has shape (1, dim) or (1, problem_size)
        
        Notes
        -----
        All algorithms are minimizers. If you have a maximization problem,
        negate the objective function value.
        
        Examples
        --------
        >>> from problems.continuous.sphere import SphereProblem
        >>> problem = SphereProblem(dim=10)
        >>> optimizer = SomeOptimizer(problem=problem, seed=42)
        >>> best_sol, best_fit, history, trajectory = optimizer.run(max_iter=100)
        >>> print(f"Best fitness: {best_fit}")
        """
        pass
