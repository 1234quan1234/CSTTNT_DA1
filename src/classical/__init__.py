"""
Classical optimization algorithms package.

Includes baseline algorithms: Hill Climbing, Simulated Annealing,
Genetic Algorithm, and graph search algorithms (BFS, DFS, A*).
"""

from .hill_climbing import HillClimbingOptimizer
from .simulated_annealing import SimulatedAnnealingOptimizer
from .genetic_algorithm import GeneticAlgorithmOptimizer
from . import graph_search

__all__ = [
    'HillClimbingOptimizer',
    'SimulatedAnnealingOptimizer',
    'GeneticAlgorithmOptimizer',
    'graph_search'
]
