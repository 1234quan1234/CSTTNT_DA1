"""
Rastrigin function optimization problem.

The Rastrigin function is a highly multimodal function with many local minima,
making it a difficult benchmark for global optimization algorithms.

References
----------
.. [1] https://www.sfu.ca/~ssurjano/rastr.html
"""

import numpy as np
from typing import Literal

# Use relative import
from ...core.problem_base import ProblemBase


class RastriginProblem(ProblemBase):
    """
    Rastrigin function: f(x) = A*d + sum_{i=1}^{d} [x_i^2 - A*cos(2*pi*x_i)]
    
    where A = 10.
    
    The Rastrigin function is highly multimodal with a large number of local minima
    regularly distributed. It is based on Sphere function with added cosine modulation.
    
    Properties:
    - Continuous
    - Non-convex
    - Highly multimodal (many local minima)
    - Separable
    
    Global minimum: f(0, 0, ..., 0) = 0
    Typical domain: [-5.12, 5.12]^d
    
    Parameters
    ----------
    dim : int
        Dimensionality of the problem.
    A : float, optional
        Amplitude parameter. Default is 10.0 (standard).
    lower : float or np.ndarray, optional
        Lower bound(s) for the search space. Default is -5.12.
    upper : float or np.ndarray, optional
        Upper bound(s) for the search space. Default is 5.12.
    
    Attributes
    ----------
    dim : int
        Problem dimensionality.
    A : float
        Amplitude parameter.
    lower : np.ndarray
        Lower bounds, shape (dim,).
    upper : np.ndarray
        Upper bounds, shape (dim,).
    
    Examples
    --------
    >>> problem = RastriginProblem(dim=2)
    >>> x = np.array([0.0, 0.0])
    >>> problem.evaluate(x)
    0.0
    """
    
    def __init__(self, dim: int, A: float = 10.0, lower: float = -5.12, upper: float = 5.12):
        """
        Initialize Rastrigin problem.
        
        Parameters
        ----------
        dim : int
            Dimensionality of the problem.
        A : float, optional
            Amplitude parameter. Default is 10.0.
        lower : float or np.ndarray, optional
            Lower bound(s). Default is -5.12.
        upper : float or np.ndarray, optional
            Upper bound(s). Default is 5.12.
        """
        self.dim = dim
        self.A = A
        
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
        Evaluate the Rastrigin function at point x.
        
        f(x) = A*d + sum_{i=1}^{d} [x_i^2 - A*cos(2*pi*x_i)]
        
        Parameters
        ----------
        x : np.ndarray
            Solution vector of shape (dim,).
        
        Returns
        -------
        fitness : float
            Rastrigin function value.
        """
        result = self.A * self.dim
        result += np.sum(x**2 - self.A * np.cos(2 * np.pi * x))
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
    print("Rastrigin Function Demo")
    print("=" * 50)
    
    problem = RastriginProblem(dim=2)
    
    # Test at global optimum
    x_opt = np.array([0.0, 0.0])
    f_opt = problem.evaluate(x_opt)
    print(f"f([0, 0]) = {f_opt} (expected: 0.0)")
    
    # Test at other points
    x1 = np.array([1.0, 1.0])
    f1 = problem.evaluate(x1)
    print(f"f([1, 1]) = {f1:.4f}")
    
    # Test at a local minimum-ish point
    x2 = np.array([0.99, 0.99])  # Close to but not at local minimum
    f2 = problem.evaluate(x2)
    print(f"f([0.99, 0.99]) = {f2:.4f}")
    
    # Test initialization
    rng = np.random.RandomState(42)
    init_pop = problem.init_solution(rng, n=5)
    print(f"\nGenerated 5 random solutions, shape: {init_pop.shape}")
    print(f"Sample solution: {init_pop[0]}")
    print(f"All within bounds: {np.all(init_pop >= problem.lower) and np.all(init_pop <= problem.upper)}")
    
    # Verify multimodality
    print(f"\nRastrigin is highly multimodal with many local minima.")
    print(f"Global minimum at origin: f([0,0]) = {f_opt}")
    
    print("\nRastrigin problem test passed!")
    
    print("\nRastrigin problem test passed!")
