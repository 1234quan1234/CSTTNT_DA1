"""
Continuous optimization benchmark problems.

Includes classic test functions: Sphere, Rosenbrock, Rastrigin, and Ackley.
"""

from .sphere import SphereProblem
from .rosenbrock import RosenbrockProblem
from .rastrigin import RastriginProblem
from .ackley import AckleyProblem

__all__ = ['SphereProblem', 'RosenbrockProblem', 'RastriginProblem', 'AckleyProblem']
