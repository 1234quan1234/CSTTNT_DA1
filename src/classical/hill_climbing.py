"""
Hill Climbing optimization algorithm.

Hill Climbing is a local search algorithm that iteratively moves to better
neighboring solutions until no better neighbors exist (local optimum).

References
----------
.. [1] https://www.geeksforgeeks.org/artificial-intelligence/introduction-hill-climbing-artificial-intelligence/
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from core.base_optimizer import BaseOptimizer
from core.problem_base import ProblemBase
from typing import List, Tuple


class HillClimbingOptimizer(BaseOptimizer):
    """
    Hill Climbing local search optimizer.
    
    Supports both continuous and discrete optimization problems:
    - Continuous: generates neighbors by adding Gaussian noise
    - TSP: generates neighbors by swapping two cities
    - Knapsack: generates neighbors by flipping one bit
    - Graph Coloring: generates neighbors by changing one node's color
    
    The algorithm is greedy: it only accepts improvements (better fitness).
    It terminates when stuck at a local optimum or max_iter is reached.
    
    Parameters
    ----------
    problem : ProblemBase
        The optimization problem.
    num_neighbors : int, default=10
        Number of neighbors to generate and evaluate per iteration.
    step_size : float, default=0.1
        Step size for continuous problems (Gaussian noise std dev).
    seed : int or None, default=None
        Random seed for reproducibility.
    
    Attributes
    ----------
    problem : ProblemBase
        The optimization problem.
    num_neighbors : int
        Number of neighbors per iteration.
    step_size : float
        Perturbation step size for continuous problems.
    rng : np.random.RandomState
        Random number generator.
    current_solution : np.ndarray
        Current solution.
    current_fitness : float
        Current fitness value.
    
    Examples
    --------
    >>> from problems.continuous.sphere import SphereProblem
    >>> problem = SphereProblem(dim=2)
    >>> optimizer = HillClimbingOptimizer(problem, seed=42)
    >>> best_sol, best_fit, history, trajectory = optimizer.run(max_iter=100)
    """
    
    def __init__(
        self,
        problem: ProblemBase,
        num_neighbors: int = 10,
        step_size: float = 0.1,
        seed: int = None
    ):
        """Initialize Hill Climbing optimizer."""
        self.problem = problem
        self.num_neighbors = num_neighbors
        self.step_size = step_size
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        self.current_solution = None
        self.current_fitness = None
    
    def _generate_neighbor_continuous(self) -> np.ndarray:
        """Generate a neighbor for continuous problems by adding Gaussian noise."""
        dim = len(self.current_solution)
        noise = self.rng.randn(dim) * self.step_size
        neighbor = self.current_solution + noise
        return self.problem.clip(neighbor)
    
    def _generate_neighbor_tsp(self) -> np.ndarray:
        """Generate a neighbor for TSP by swapping two cities."""
        neighbor = self.current_solution.copy()
        num_cities = len(neighbor)
        i, j = self.rng.choice(num_cities, size=2, replace=False)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return neighbor
    
    def _generate_neighbor_knapsack(self) -> np.ndarray:
        """Generate a neighbor for Knapsack by flipping one bit."""
        neighbor = self.current_solution.copy()
        flip_idx = self.rng.randint(len(neighbor))
        neighbor[flip_idx] = 1 - neighbor[flip_idx]
        return neighbor
    
    def _generate_neighbor_graph_coloring(self) -> np.ndarray:
        """Generate a neighbor for Graph Coloring by changing one node's color."""
        neighbor = self.current_solution.copy()
        node_idx = self.rng.randint(len(neighbor))
        # Get number of colors from problem
        num_colors = self.problem.num_colors
        # Change to a different random color
        current_color = neighbor[node_idx]
        new_color = self.rng.randint(num_colors)
        while new_color == current_color and num_colors > 1:
            new_color = self.rng.randint(num_colors)
        neighbor[node_idx] = new_color
        return neighbor
    
    def _generate_neighbor(self) -> np.ndarray:
        """Generate a neighbor based on problem type."""
        repr_type = self.problem.representation_type()
        
        if repr_type == "continuous":
            return self._generate_neighbor_continuous()
        elif repr_type == "tsp":
            return self._generate_neighbor_tsp()
        elif repr_type == "knapsack":
            return self._generate_neighbor_knapsack()
        elif repr_type == "graph_coloring":
            return self._generate_neighbor_graph_coloring()
        else:
            raise NotImplementedError(f"Unsupported problem type: {repr_type}")
    
    def run(self, max_iter: int) -> Tuple[np.ndarray, float, List[float], List[np.ndarray]]:
        """
        Run Hill Climbing for max_iter iterations.
        
        Parameters
        ----------
        max_iter : int
            Maximum number of iterations.
        
        Returns
        -------
        best_solution : np.ndarray
            Best solution found.
        best_fitness : float
            Best fitness value (minimum).
        history_best : List[float]
            Best fitness at each iteration.
        trajectory : List[np.ndarray]
            Solution at each iteration, shape (1, problem_size).
        """
        # Initialize with random solution
        self.current_solution = self.problem.init_solution(self.rng, n=1)[0]
        self.current_fitness = self.problem.evaluate(self.current_solution)
        
        history_best = []
        trajectory = []
        
        for iteration in range(max_iter):
            # Generate and evaluate neighbors
            improved = False
            
            for _ in range(self.num_neighbors):
                neighbor = self._generate_neighbor()
                neighbor_fitness = self.problem.evaluate(neighbor)
                
                # Accept if better (greedy)
                if neighbor_fitness < self.current_fitness:
                    self.current_solution = neighbor
                    self.current_fitness = neighbor_fitness
                    improved = True
            
            # Track progress
            history_best.append(self.current_fitness)
            trajectory.append(self.current_solution.reshape(1, -1).copy())
            
            # Early stopping if stuck at local optimum
            # (no improvement in this iteration)
            # Note: We continue anyway to fill max_iter for consistency
        
        return (
            self.current_solution.copy(),
            self.current_fitness,
            history_best,
            trajectory
        )


if __name__ == "__main__":
    print("=" * 60)
    print("HILL CLIMBING OPTIMIZER DEMO")
    print("=" * 60)
    
    # Test 1: Continuous problem (Sphere)
    print("\n1. Hill Climbing on Sphere Function (2D)")
    print("-" * 60)
    from problems.continuous.sphere import SphereProblem
    
    problem_sphere = SphereProblem(dim=2)
    hc_sphere = HillClimbingOptimizer(
        problem=problem_sphere,
        num_neighbors=20,
        step_size=0.5,
        seed=42
    )
    
    best_sol, best_fit, history, trajectory = hc_sphere.run(max_iter=50)
    
    print(f"Initial fitness: {history[0]:.6f}")
    print(f"Final fitness: {history[-1]:.6f}")
    print(f"Best solution: {best_sol}")
    print(f"Improvement: {history[0] - history[-1]:.6f}")
    
    # Test 2: TSP
    print("\n2. Hill Climbing on TSP (10 cities)")
    print("-" * 60)
    from problems.discrete.tsp import TSPProblem
    
    rng_tsp = np.random.RandomState(123)
    coords = rng_tsp.rand(10, 2) * 10
    problem_tsp = TSPProblem(coords)
    
    hc_tsp = HillClimbingOptimizer(
        problem=problem_tsp,
        num_neighbors=20,
        seed=42
    )
    
    best_tour, best_length, history_tsp, _ = hc_tsp.run(max_iter=50)
    
    print(f"Initial tour length: {history_tsp[0]:.4f}")
    print(f"Final tour length: {history_tsp[-1]:.4f}")
    print(f"Best tour: {best_tour}")
    print(f"Improvement: {history_tsp[0] - history_tsp[-1]:.4f}")
    
    # Test 3: Knapsack
    print("\n3. Hill Climbing on Knapsack Problem")
    print("-" * 60)
    from problems.discrete.knapsack import KnapsackProblem
    
    values = np.array([10, 20, 30, 40, 50])
    weights = np.array([1, 2, 3, 4, 5])
    capacity = 7.0
    problem_knapsack = KnapsackProblem(values, weights, capacity)
    
    hc_knapsack = HillClimbingOptimizer(
        problem=problem_knapsack,
        num_neighbors=10,
        seed=42
    )
    
    best_sel, best_fit_k, history_k, _ = hc_knapsack.run(max_iter=30)
    
    print(f"Initial fitness: {history_k[0]:.2f}")
    print(f"Final fitness: {history_k[-1]:.2f}")
    print(f"Best selection: {best_sel}")
    total_weight = np.sum(best_sel * weights)
    total_value = np.sum(best_sel * values)
    print(f"Total weight: {total_weight}, Total value: {total_value}")
    print(f"Feasible: {total_weight <= capacity}")
    
    # Test 4: Graph Coloring
    print("\n4. Hill Climbing on Graph Coloring (Triangle)")
    print("-" * 60)
    from problems.discrete.graph_coloring import GraphColoringProblem
    
    edges = [(0, 1), (1, 2), (2, 0)]  # Triangle
    problem_gc = GraphColoringProblem(num_nodes=3, edges=edges, num_colors=3)
    
    hc_gc = HillClimbingOptimizer(
        problem=problem_gc,
        num_neighbors=15,
        seed=42
    )
    
    best_coloring, best_conflicts, history_gc, _ = hc_gc.run(max_iter=30)
    
    print(f"Initial conflicts: {history_gc[0]:.0f}")
    print(f"Final conflicts: {history_gc[-1]:.0f}")
    print(f"Best coloring: {best_coloring}")
    print(f"Valid coloring: {history_gc[-1] == 0}")
    
    print("\n" + "=" * 60)
    print("All Hill Climbing tests completed!")
    print("=" * 60)
