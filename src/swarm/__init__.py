"""
Swarm intelligence algorithms package.

Currently includes Firefly Algorithm (FA) for both continuous and discrete optimization.
"""

from .fa import FireflyContinuousOptimizer, FireflyDiscreteTSPOptimizer

__all__ = ['FireflyContinuousOptimizer', 'FireflyDiscreteTSPOptimizer']
