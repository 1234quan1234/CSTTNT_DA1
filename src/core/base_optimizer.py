"""
Abstract base class for all optimization algorithms.

This module defines the BaseOptimizer interface that all optimization algorithms
(Firefly Algorithm, Hill Climbing, Simulated Annealing, Genetic Algorithm, etc.)
must implement to ensure consistency across the project.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, List, Tuple


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
    def run(self, max_iter: int) -> Tuple[np.ndarray, float, List[float], List[Dict[str, float]]]:
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
            - For knapsack: shape (num_items,) binary 0/1 vector
        
        best_fitness : float
            The objective function value of the best solution (minimization).
            Lower values are better. For maximization problems, the problem's
            evaluate() method should return the negated value.
        
        history_best : List[float]
            History of best fitness values at each iteration.
            Length should be max_iter, where history_best[t] is the best
            fitness found up to and including iteration t.
            This is used for plotting convergence curves.
        
        stats_history : List[Dict[str, float]]
            Statistical summary of population/solution at each iteration.
            Each dict contains:
            - 'gen': Generation number
            - 'best_fitness': Best fitness in this generation
            - 'mean_fitness': Mean fitness of population
            - 'std_fitness': Standard deviation of fitness
            - 'diversity': Population diversity (mean distance to centroid)
            For single-solution algorithms (SA, HC), some fields may be None.
        
        Notes
        -----
        **IMPORTANT CHANGE**: The fourth return value has been changed from
        `trajectory` (full population history) to `stats_history` (statistical
        summaries). This dramatically reduces memory usage and file sizes while
        providing richer analytical information.
        
        Old signature (deprecated):
            return best_solution, best_fitness, history_best, trajectory
        
        New signature:
            return best_solution, best_fitness, history_best, stats_history
        
        For population-based algorithms (FA, GA), compute:
        - diversity = mean(distance from each individual to population centroid)
        
        For single-solution algorithms (SA, HC):
        - Set mean_fitness = best_fitness
        - Set std_fitness = 0.0
        - Set diversity = 0.0 (no population)
        """
        pass

