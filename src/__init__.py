"""
AI Search and Optimization Project.

A comprehensive framework for comparing swarm intelligence algorithms
(Firefly Algorithm) with classical optimization methods on various
continuous and discrete benchmark problems.

Modules
-------
core : Base classes and utilities
problems : Benchmark optimization problems
swarm : Swarm intelligence algorithms
classical : Classical/baseline optimization algorithms
"""

from . import core
from . import problems
from . import swarm
from . import classical

__version__ = "1.0.0"
__all__ = ['core', 'problems', 'swarm', 'classical']
