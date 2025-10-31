"""
Discrete optimization problems.

Classic combinatorial optimization problems for evaluating discrete
optimization algorithms.
"""

from .tsp import TSPProblem
from .knapsack import KnapsackProblem
from .graph_coloring import GraphColoringProblem

__all__ = [
    'TSPProblem',
    'KnapsackProblem',
    'GraphColoringProblem'
]
