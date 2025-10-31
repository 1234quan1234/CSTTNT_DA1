"""
Core package for optimization framework.

This package contains base classes and utilities used across all optimization
algorithms and problems.
"""

from .base_optimizer import BaseOptimizer
from .problem_base import ProblemBase
from . import utils

__all__ = ['BaseOptimizer', 'ProblemBase', 'utils']
