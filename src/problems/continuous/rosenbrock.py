"""
Rosenbrock function optimization problem.

The Rosenbrock function (also known as the Banana function or Valley function)
is a non-convex function with a narrow, curved valley that makes it challenging
for optimization algorithms.

References
----------
.. [1] https://en.wikipedia.org/wiki/Rosenbrock_function
.. [2] https://www.sfu.ca/~ssurjano/rosen.html
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from core.problem_base import ProblemBase
from typing import Literal


class RosenbrockProblem(ProblemBase):
    """
    Rosenbrock function: f(x) = sum_{i=1}^{d-1} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
    
    The Rosenbrock function is a classic benchmark for optimization algorithms.
    It has a narrow, curved valley from (1, 1, ..., 1) which is challenging to follow.
    
    Properties:
    - Continuous
    - Non-convex
    - Unimodal
    - The global minimum is inside a long, narrow, parabolic shaped flat valley
    
    Global minimum: f(1, 1, ..., 1) = 0
    Typical domain: [-2.048, 2.048]^d
    
    Parameters
    ----------
    dim : int
        Dimensionality of the problem (must be >= 2).
    lower : float or np.ndarray, optional
        Lower bound(s) for the search space. Default is -2.048.
    upper : float or np.ndarray, optional
        Upper bound(s) for the search space. Default is 2.048.
    
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
    >>> problem = RosenbrockProblem(dim=2)
    >>> x = np.array([1.0, 1.0])
    >>> problem.evaluate(x)
    0.0
    >>> x = np.array([0.0, 0.0])
    >>> problem.evaluate(x)
    1.0
    """
    
    def __init__(self, dim: int, lower: float = -2.048, upper: float = 2.048):
        """
        Initialize Rosenbrock problem.
        
        Parameters
        ----------
        dim : int
            Dimensionality of the problem (must be >= 2).
        lower : float or np.ndarray, optional
            Lower bound(s). Default is -2.048.
        upper : float or np.ndarray, optional
            Upper bound(s). Default is 2.048.
        """
        if dim < 2:
            raise ValueError("Rosenbrock function requires dim >= 2")
        
        self.dim = dim
        
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
        Evaluate the Rosenbrock function at point x.
        
        f(x) = sum_{i=1}^{d-1} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
        
        Parameters
        ----------
        x : np.ndarray
            Solution vector of shape (dim,).
        
        Returns
        -------
        fitness : float
            Rosenbrock function value.
        """
        result = 0.0
        for i in range(self.dim - 1):
            result += 100.0 * (x[i+1] - x[i]**2)**2 + (1.0 - x[i])**2
        return float(result)
    
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
    print("Rosenbrock Function Demo")
    print("=" * 50)
    
    problem = RosenbrockProblem(dim=2)
    
    # Test at global optimum
    x_opt = np.array([1.0, 1.0])
    f_opt = problem.evaluate(x_opt)
    print(f"f([1, 1]) = {f_opt} (expected: 0.0)")
    
    # Test at origin
    x0 = np.array([0.0, 0.0])
    f0 = problem.evaluate(x0)
    print(f"f([0, 0]) = {f0} (expected: 1.0)")
    
    # Test at another point
    x1 = np.array([0.5, 0.5])
    f1 = problem.evaluate(x1)
    print(f"f([0.5, 0.5]) = {f1:.4f}")
    
    # Test 3D
    problem_3d = RosenbrockProblem(dim=3)
    x_3d = np.array([1.0, 1.0, 1.0])
    f_3d = problem_3d.evaluate(x_3d)
    print(f"\n3D: f([1, 1, 1]) = {f_3d} (expected: 0.0)")
    
    # Test initialization
    rng = np.random.RandomState(42)
    init_pop = problem.init_solution(rng, n=5)
    print(f"\nGenerated 5 random solutions, shape: {init_pop.shape}")
    print(f"Sample solution: {init_pop[0]}")
    print(f"All within bounds: {np.all(init_pop >= problem.lower) and np.all(init_pop <= problem.upper)}")
    
    print("\nRosenbrock problem test passed!")
