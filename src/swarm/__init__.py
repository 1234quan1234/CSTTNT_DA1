"""
Swarm intelligence algorithms package.

This package contains nature-inspired swarm intelligence optimization algorithms.
"""

from .fa import FireflyContinuousOptimizer, FireflyDiscreteTSPOptimizer

__all__ = [
    'FireflyContinuousOptimizer',
    'FireflyDiscreteTSPOptimizer'
]
