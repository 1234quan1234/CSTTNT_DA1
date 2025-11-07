"""
Optimization problems package.

This package contains benchmark problems for testing and comparing
optimization algorithms.
"""

from . import continuous
from . import discrete

__all__ = ['continuous', 'discrete', 'ProblemBase']
