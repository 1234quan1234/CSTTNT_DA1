"""
Discrete/combinatorial optimization problems.

Includes TSP, Knapsack, and Graph Coloring problems.
"""

from .tsp import TSPProblem
from .knapsack import KnapsackProblem
from .graph_coloring import GraphColoringProblem

__all__ = ['TSPProblem', 'KnapsackProblem', 'GraphColoringProblem']
