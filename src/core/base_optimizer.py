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
        Random number generator instance. All subclasses MUST use self.rng
        instead of np.random to ensure reproducible results.
    
    Notes
    -----
    **Reproducibility Requirement:**
    All concrete optimizer implementations MUST:
    1. Accept a `seed` parameter in __init__
    2. Create self.rng = np.random.RandomState(seed)
    3. Use ONLY self.rng for all random operations (never use np.random directly)
    
    This ensures that running the same optimizer with the same seed produces
    identical results, which is critical for scientific reproducibility and
    fair algorithm comparison.
    
    Examples
    --------
    Correct implementation:
    >>> class MyOptimizer(BaseOptimizer):
    ...     def __init__(self, problem, seed=None):
    ...         self.problem = problem
    ...         self.seed = seed
    ...         self.rng = np.random.RandomState(seed)  # ✓ Correct
    ...     
    ...     def run(self, max_iter):
    ...         x = self.rng.rand(10)  # ✓ Use self.rng
    ...         # ...existing code...
    
    Incorrect implementation:
    >>> class BadOptimizer(BaseOptimizer):
    ...     def run(self, max_iter):
    ...         x = np.random.rand(10)  # ✗ WRONG! Not reproducible
    ...         # ...existing code...
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
            Lower values are better. For maximization problems, the problem's
            evaluate() method should return the negated value.
        
        history_best : List[float]
            History of best fitness values at each iteration.
            Length should be max_iter, where history_best[t] is the best
            fitness found up to and including iteration t.
            This is used for plotting convergence curves.
        
        trajectory : List[np.ndarray]
            Trajectory of the population/solution at each iteration.
            Used for visualization and analysis of convergence behavior.
            - For population-based algorithms (FA, GA): 
              trajectory[t] has shape (population_size, dim) or (population_size, problem_size)
            - For single-solution algorithms (Hill Climbing, SA):
              trajectory[t] has shape (1, dim) or (1, problem_size)
              (wrap solution in a list/array for consistency)
        
        Notes
        -----
        All algorithms are minimizers. If you have a maximization problem,
        the problem's evaluate() method should negate the objective function value.
        
        Reproducibility: When the same seed is used, running this method multiple
        times should produce identical results (same best_solution, best_fitness,
        and history_best values).
        
        Examples
        --------
        >>> from problems.continuous.sphere import SphereProblem
        >>> problem = SphereProblem(dim=10)
        >>> optimizer = SomeOptimizer(problem=problem, seed=42)
        >>> best_sol, best_fit, history, trajectory = optimizer.run(max_iter=100)
        >>> print(f"Best fitness: {best_fit}")
        
        Verify reproducibility:
        >>> optimizer1 = SomeOptimizer(problem=problem, seed=42)
        >>> optimizer2 = SomeOptimizer(problem=problem, seed=42)
        >>> sol1, fit1, _, _ = optimizer1.run(max_iter=100)
        >>> sol2, fit2, _, _ = optimizer2.run(max_iter=100)
        >>> assert fit1 == fit2  # Should be identical
        >>> assert np.allclose(sol1, sol2)  # Should be identical
        """
        pass
