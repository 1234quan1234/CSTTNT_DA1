"""
Sphere function optimization problem.

The Sphere function (also known as De Jong's first function) is a simple,
convex, unimodal benchmark function commonly used to test optimization algorithms.

References
----------
.. [1] https://www.sfu.ca/~ssurjano/spheref.html
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from core.problem_base import ProblemBase
from typing import Literal


class SphereProblem(ProblemBase):
    """
    Sphere function: f(x) = sum(x_i^2)
    
    This is one of the simplest optimization benchmark functions. It is:
    - Continuous
    - Convex
    - Unimodal (single global minimum)
    - Separable (can be optimized dimension by dimension)
    
    Global minimum: f(0, 0, ..., 0) = 0
    Typical domain: [-5.12, 5.12]^d
    
    Parameters
    ----------
    dim : int
        Dimensionality of the problem.
    lower : float or np.ndarray, optional
        Lower bound(s) for the search space. Default is -5.12.
    upper : float or np.ndarray, optional
        Upper bound(s) for the search space. Default is 5.12.
    
    Attributes
    ----------
    dim : int
        Problem dimensionality.
    lower : np.ndarray
        Lower bounds, shape (dim,).
    upper : np.ndarray
        Upper bounds, shape (dim,).
    
    Examples
    --------
    >>> problem = SphereProblem(dim=2)
    >>> x = np.array([0.0, 0.0])
    >>> problem.evaluate(x)
    0.0
    >>> x = np.array([1.0, 1.0])
    >>> problem.evaluate(x)
    2.0
    """
    
    def __init__(self, dim: int, lower: float = -5.12, upper: float = 5.12):
        """
        Initialize Sphere problem.
        
        Parameters
        ----------
        dim : int
            Dimensionality of the problem.
        lower : float or np.ndarray, optional
            Lower bound(s). Default is -5.12.
        upper : float or np.ndarray, optional
            Upper bound(s). Default is 5.12.
        """
        self.dim = dim
        
        # Handle scalar or array bounds
        if np.isscalar(lower):
            self.lower = np.full(dim, lower, dtype=float)
        else:
            self.lower = np.array(lower, dtype=float)
        
        if np.isscalar(upper):
            self.upper = np.full(dim, upper, dtype=float)
        else:
            self.upper = np.array(upper, dtype=float)
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Sphere function at point x.
        
        f(x) = sum_{i=1}^{d} x_i^2
        
        Parameters
        ----------
        x : np.ndarray
            Solution vector of shape (dim,).
        
        Returns
        -------
        fitness : float
            Sum of squared components.
        """
        return float(np.sum(x ** 2))
    
    def representation_type(self) -> Literal["continuous", "tsp", "knapsack", "graph_coloring"]:
        """Return 'continuous' for this problem type."""
        return "continuous"
    
    def init_solution(self, rng: np.random.RandomState, n: int = 1) -> np.ndarray:
        """
        Generate n random solutions uniformly within bounds.
        
        Parameters
        ----------
        rng : np.random.RandomState
            Random number generator.
        n : int, default=1
            Number of solutions to generate.
        
        Returns
        -------
        solutions : np.ndarray
            Array of shape (n, dim) with random solutions in [lower, upper].
        """
        solutions = np.zeros((n, self.dim))
        for i in range(self.dim):
            solutions[:, i] = rng.uniform(self.lower[i], self.upper[i], n)
        return solutions
    
    def clip(self, X: np.ndarray) -> np.ndarray:
        """
        Clip solutions to be within [lower, upper] bounds.
        
        Parameters
        ----------
        X : np.ndarray
            Solutions to clip, shape (n, dim) or (dim,).
        
        Returns
        -------
        X_clipped : np.ndarray
            Clipped solutions within bounds.
        """
        return np.clip(X, self.lower, self.upper)


if __name__ == "__main__":
    # Demo
    print("Sphere Function Demo")
    print("=" * 50)
    
    problem = SphereProblem(dim=2)
    
    # Test at global optimum
    x_opt = np.array([0.0, 0.0])
    f_opt = problem.evaluate(x_opt)
    print(f"f([0, 0]) = {f_opt} (expected: 0.0)")
    
    # Test at other points
    x1 = np.array([1.0, 1.0])
    f1 = problem.evaluate(x1)
    print(f"f([1, 1]) = {f1} (expected: 2.0)")
    
    x2 = np.array([3.0, 4.0])
    f2 = problem.evaluate(x2)
    print(f"f([3, 4]) = {f2} (expected: 25.0)")
    
    # Test initialization
    rng = np.random.RandomState(42)
    init_pop = problem.init_solution(rng, n=5)
    print(f"\nGenerated 5 random solutions, shape: {init_pop.shape}")
    print(f"Sample solution: {init_pop[0]}")
    print(f"All within bounds: {np.all(init_pop >= problem.lower) and np.all(init_pop <= problem.upper)}")
    
    # Test clipping
    x_out = np.array([10.0, -10.0])
    x_clipped = problem.clip(x_out)
    print(f"\nClipping [{10.0}, {-10.0}]: {x_clipped}")
    print(f"Within bounds: {np.all(x_clipped >= problem.lower) and np.all(x_clipped <= problem.upper)}")
    
    print("\nSphere problem test passed!")
