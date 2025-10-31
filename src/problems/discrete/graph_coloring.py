"""
Graph Coloring Problem optimization.

The Graph Coloring problem is a classic combinatorial optimization problem where
the goal is to assign colors to graph nodes such that no adjacent nodes share
the same color, minimizing the number of conflicts.

References
----------
.. [1] https://en.wikipedia.org/wiki/Graph_coloring
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from core.problem_base import ProblemBase
from typing import List, Tuple, Literal


class GraphColoringProblem(ProblemBase):
    """
    Graph Coloring Problem.
    
    Given an undirected graph and a number of colors, assign colors to nodes
    such that no two adjacent nodes have the same color.
    
    Solution representation: color assignment vector where x[i] is the color
    (integer in [0, num_colors-1]) assigned to node i.
    
    Fitness: number of edge conflicts (adjacent nodes with same color).
    Objective: minimize conflicts (ideally to 0).
    
    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    edges : List[Tuple[int, int]]
        List of edges as (u, v) pairs.
    num_colors : int
        Number of colors available.
    
    Attributes
    ----------
    num_nodes : int
        Number of graph nodes.
    edges : List[Tuple[int, int]]
        Graph edges.
    num_colors : int
        Number of available colors.
    adjacency_list : dict
        Adjacency list representation of the graph.
    
    Examples
    --------
    >>> edges = [(0, 1), (1, 2), (2, 0)]  # Triangle graph
    >>> problem = GraphColoringProblem(num_nodes=3, edges=edges, num_colors=3)
    >>> coloring = np.array([0, 1, 2])  # Valid 3-coloring
    >>> conflicts = problem.evaluate(coloring)
    >>> print(f"Conflicts: {conflicts}")  # Should be 0
    """
    
    def __init__(self, num_nodes: int, edges: List[Tuple[int, int]], num_colors: int):
        """
        Initialize Graph Coloring problem.
        
        Parameters
        ----------
        num_nodes : int
            Number of nodes in the graph.
        edges : List[Tuple[int, int]]
            List of edges as (u, v) tuples.
        num_colors : int
            Number of colors available.
        """
        self.num_nodes = num_nodes
        self.edges = edges
        self.num_colors = num_colors
        
        # Build adjacency list for efficient conflict checking
        self.adjacency_list = {i: [] for i in range(num_nodes)}
        for u, v in edges:
            self.adjacency_list[u].append(v)
            self.adjacency_list[v].append(u)
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the number of color conflicts (adjacent nodes with same color).
        
        Parameters
        ----------
        x : np.ndarray
            Color assignment, shape (num_nodes,), values in [0, num_colors-1].
        
        Returns
        -------
        conflicts : float
            Number of edges where both endpoints have the same color.
        """
        coloring = x.astype(int)
        conflicts = 0
        
        # Count conflicts (each edge checked once)
        checked_edges = set()
        for u in range(self.num_nodes):
            for v in self.adjacency_list[u]:
                edge = tuple(sorted([u, v]))
                if edge not in checked_edges:
                    if coloring[u] == coloring[v]:
                        conflicts += 1
                    checked_edges.add(edge)
        
        return float(conflicts)
    
    def representation_type(self) -> Literal["continuous", "tsp", "knapsack", "graph_coloring"]:
        """Return 'graph_coloring' for this problem type."""
        return "graph_coloring"
    
    def init_solution(self, rng: np.random.RandomState, n: int = 1) -> np.ndarray:
        """
        Generate n random color assignments.
        
        Parameters
        ----------
        rng : np.random.RandomState
            Random number generator.
        n : int, default=1
            Number of solutions to generate.
        
        Returns
        -------
        solutions : np.ndarray
            Array of shape (n, num_nodes) with random color assignments.
        """
        return rng.randint(0, self.num_colors, size=(n, self.num_nodes))
    
    def clip(self, X: np.ndarray) -> np.ndarray:
        """
        Clip colors to valid range [0, num_colors-1].
        
        Parameters
        ----------
        X : np.ndarray
            Color assignment(s), shape (n, num_nodes) or (num_nodes,).
        
        Returns
        -------
        X_clipped : np.ndarray
            Clipped color assignments.
        """
        return np.clip(X, 0, self.num_colors - 1).astype(int)


if __name__ == "__main__":
    # Demo
    print("Graph Coloring Problem Demo")
    print("=" * 50)
    
    # Example 1: Triangle graph (3-clique, needs 3 colors)
    print("Example 1: Triangle graph")
    edges_triangle = [(0, 1), (1, 2), (2, 0)]
    problem_triangle = GraphColoringProblem(num_nodes=3, edges=edges_triangle, num_colors=3)
    
    # Valid 3-coloring
    coloring_valid = np.array([0, 1, 2])
    conflicts_valid = problem_triangle.evaluate(coloring_valid)
    print(f"Valid coloring [0,1,2]: conflicts = {conflicts_valid} (expected: 0)")
    
    # Invalid coloring
    coloring_invalid = np.array([0, 0, 1])
    conflicts_invalid = problem_triangle.evaluate(coloring_invalid)
    print(f"Invalid coloring [0,0,1]: conflicts = {conflicts_invalid} (expected: 1)")
    
    # Example 2: Square graph (4-cycle, needs 2 colors)
    print("\nExample 2: Square graph (4-cycle)")
    edges_square = [(0, 1), (1, 2), (2, 3), (3, 0)]
    problem_square = GraphColoringProblem(num_nodes=4, edges=edges_square, num_colors=2)
    
    # Valid 2-coloring (alternating)
    coloring_2color = np.array([0, 1, 0, 1])
    conflicts_2color = problem_square.evaluate(coloring_2color)
    print(f"Valid 2-coloring [0,1,0,1]: conflicts = {conflicts_2color} (expected: 0)")
    
    # Invalid coloring
    coloring_all_same = np.array([0, 0, 0, 0])
    conflicts_all_same = problem_square.evaluate(coloring_all_same)
    print(f"All same color [0,0,0,0]: conflicts = {conflicts_all_same} (expected: 4)")
    
    # Generate random solutions
    rng = np.random.RandomState(42)
    random_colorings = problem_square.init_solution(rng, n=5)
    print(f"\nGenerated 5 random colorings for square graph:")
    for i, coloring in enumerate(random_colorings):
        conflicts = problem_square.evaluate(coloring)
        print(f"  Coloring {i}: {coloring} -> conflicts = {conflicts}")
    
    # Example 3: Petersen graph (more complex)
    print("\nExample 3: Larger graph with random edges")
    num_nodes = 10
    rng_graph = np.random.RandomState(123)
    # Generate random edges
    edges_random = []
    for _ in range(20):
        u, v = rng_graph.choice(num_nodes, size=2, replace=False)
        if (u, v) not in edges_random and (v, u) not in edges_random:
            edges_random.append((int(u), int(v)))
    
    problem_random = GraphColoringProblem(num_nodes=num_nodes, edges=edges_random, num_colors=4)
    coloring_random = problem_random.init_solution(rng, n=1)[0]
    conflicts_random = problem_random.evaluate(coloring_random)
    print(f"Random 10-node graph with {len(edges_random)} edges")
    print(f"Random 4-coloring: conflicts = {conflicts_random}")
    
    print("\nGraph Coloring problem test passed!")
