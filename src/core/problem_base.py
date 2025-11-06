"""
Abstract base class for optimization problems.

This module defines the ProblemBase interface that all optimization problems
(continuous and discrete) must implement.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Literal


class ProblemBase(ABC):
    """
    Abstract base class for optimization problems.
    
    All optimization problems (Sphere, Rastrigin, TSP, Knapsack, etc.) must
    inherit from this class and implement the required methods to work with
    the optimization algorithms.
    
    This design allows optimizers to work with any problem type in a uniform way,
    as long as the problem implements the required interface.
    """

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the objective function at a given solution.
        
        Parameters
        ----------
        x : np.ndarray
            The solution to evaluate.
            - For continuous problems: shape (dim,) real-valued vector
            - For knapsack: shape (num_items,) binary 0/1 vector
        
        Returns
        -------
        fitness : float
            The objective function value. Lower is better (minimization).
        
        Notes
        -----
        All problems should be formulated as minimization problems.
        For maximization problems, return the negated objective value.
        """
        pass

    @abstractmethod
    def representation_type(self) -> Literal["continuous", "knapsack"]:
        """
        Return the type of solution representation this problem uses.
        
        Returns
        -------
        repr_type : str
            One of: "continuous", "knapsack"
            This tells optimizers how to handle solution representation
            and what operators to use (e.g., crossover, mutation, move).
        """
        pass

    @abstractmethod
    def init_solution(self, rng: np.random.RandomState, n: int = 1) -> np.ndarray:
        """
        Generate n random valid initial solutions.
        
        Parameters
        ----------
        rng : np.random.RandomState
            Random number generator for reproducibility.
        n : int, default=1
            Number of solutions to generate.
        
        Returns
        -------
        solutions : np.ndarray
            Array of n random solutions.
            - For continuous problems: shape (n, dim)
            - For knapsack: shape (n, num_items) with binary 0/1 values
        
        Notes
        -----
        Solutions should be valid according to problem constraints.
        Knapsack solutions are repaired to satisfy capacity constraints.
        """
        pass

    @abstractmethod
    def clip(self, X: np.ndarray) -> np.ndarray:
        """
        Clip or repair solutions to ensure they are within valid bounds.
        
        Parameters
        ----------
        X : np.ndarray
            Solutions to clip/repair. Same shape as returned by init_solution().
        
        Returns
        -------
        X_clipped : np.ndarray
            Valid solutions within problem constraints.
            - For continuous: clipped to [lower, upper] bounds
            - For discrete: may return X unchanged if discrete operators
              already ensure validity
        
        Notes
        -----
        This method is mainly used for continuous problems to enforce bounds.
        For discrete problems, validity is often handled by specialized operators.
        """
        pass
