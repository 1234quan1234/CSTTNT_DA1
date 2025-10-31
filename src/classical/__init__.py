"""
Classical optimization algorithms package.

This package contains traditional/baseline optimization algorithms for comparison.
"""

from .hill_climbing import HillClimbingOptimizer
from .simulated_annealing import SimulatedAnnealingOptimizer
from .genetic_algorithm import GeneticAlgorithmOptimizer
from .graph_search import bfs, dfs, astar

__all__ = [
    'HillClimbingOptimizer',
    'SimulatedAnnealingOptimizer',
    'GeneticAlgorithmOptimizer',
    'bfs',
    'dfs',
    'astar'
]
