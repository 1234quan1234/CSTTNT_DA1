"""
Ackley function optimization problem.

The Ackley function is a widely used multimodal benchmark function with a nearly
flat outer region and a large hole at the center, making it challenging for
optimization algorithms.

References
----------
.. [1] https://www.sfu.ca/~ssurjano/ackley.html
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from core.problem_base import ProblemBase
from typing import Literal


class AckleyProblem(ProblemBase):
    """
    Ackley function:
    f(x) = -a*exp(-b*sqrt(1/d * sum(x_i^2))) - exp(1/d * sum(cos(c*x_i))) + a + exp(1)
    
    where typically a = 20, b = 0.2, c = 2*pi.
    
    The Ackley function has a nearly flat outer region with many local minima
    and a large central hole. It is characterized by a nearly flat outer region,
    combined with a large hole at the center.
    
    Properties:
    - Continuous
    - Non-convex
    - Multimodal (many local minima)
    - Separable
    
    Global minimum: f(0, 0, ..., 0) = 0
    Typical domain: [-5, 5]^d or [-32.768, 32.768]^d
    
    Parameters
    ----------
    dim : int
        Dimensionality of the problem.
    a : float, optional
        First parameter. Default is 20.0.
    b : float, optional
        Second parameter. Default is 0.2.
    c : float, optional
        Third parameter. Default is 2*pi.
    lower : float or np.ndarray, optional
        Lower bound(s) for the search space. Default is -5.0.
    upper : float or np.ndarray, optional
        Upper bound(s) for the search space. Default is 5.0.
    
    Attributes
    ----------
    dim : int
        Problem dimensionality.
    a, b, c : float
        Ackley function parameters.
    lower : np.ndarray
        Lower bounds, shape (dim,).
    upper : np.ndarray
        Upper bounds, shape (dim,).
    
    Examples
    --------
    >>> problem = AckleyProblem(dim=2)
    >>> x = np.array([0.0, 0.0])
    >>> abs(problem.evaluate(x)) < 1e-10  # Should be very close to 0
    True
    """
    
    def __init__(
        self, 
        dim: int, 
        a: float = 20.0, 
        b: float = 0.2, 
        c: float = 2 * np.pi,
        lower: float = -5.0, 
        upper: float = 5.0
    ):
        """
        Initialize Ackley problem.
        
        Parameters
        ----------
        dim : int
            Dimensionality of the problem.
        a : float, optional
            First parameter. Default is 20.0.
        b : float, optional
            Second parameter. Default is 0.2.
        c : float, optional
            Third parameter. Default is 2*pi.
        lower : float or np.ndarray, optional
            Lower bound(s). Default is -5.0.
        upper : float or np.ndarray, optional
            Upper bound(s). Default is 5.0.
        """
        self.dim = dim
        self.a = a
        self.b = b
        self.c = c
        
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
        Evaluate the Ackley function at point x.
        
        f(x) = -a*exp(-b*sqrt(1/d * sum(x_i^2))) 
               - exp(1/d * sum(cos(c*x_i))) 
               + a + exp(1)
        
        Parameters
        ----------
        x : np.ndarray
            Solution vector of shape (dim,).
        
        Returns
        -------
        fitness : float
            Ackley function value.
        """
        d = self.dim
        
        # First term: -a * exp(-b * sqrt(mean(x^2)))
        sum_sq = np.sum(x**2)
        term1 = -self.a * np.exp(-self.b * np.sqrt(sum_sq / d))
        
        # Second term: -exp(mean(cos(c*x)))
        sum_cos = np.sum(np.cos(self.c * x))
        term2 = -np.exp(sum_cos / d)
        
        # Complete function
        result = term1 + term2 + self.a + np.exp(1)
        
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
    print("Ackley Function Demo")
    print("=" * 50)
    
    problem = AckleyProblem(dim=2)
    
    # Test at global optimum
    x_opt = np.array([0.0, 0.0])
    f_opt = problem.evaluate(x_opt)
    print(f"f([0, 0]) = {f_opt:.10f} (expected: ~0.0)")
    print(f"Is close to zero: {abs(f_opt) < 1e-10}")
    
    # Test at other points
    x1 = np.array([1.0, 1.0])
    f1 = problem.evaluate(x1)
    print(f"f([1, 1]) = {f1:.4f}")
    
    x2 = np.array([2.0, -2.0])
    f2 = problem.evaluate(x2)
    print(f"f([2, -2]) = {f2:.4f}")
    
    # Test initialization
    rng = np.random.RandomState(42)
    init_pop = problem.init_solution(rng, n=5)
    print(f"\nGenerated 5 random solutions, shape: {init_pop.shape}")
    print(f"Sample solution: {init_pop[0]}")
    print(f"All within bounds: {np.all(init_pop >= problem.lower) and np.all(init_pop <= problem.upper)}")
    
    print("\nAckley problem test passed!")
