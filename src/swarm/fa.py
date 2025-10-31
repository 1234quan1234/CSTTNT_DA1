"""
Firefly Algorithm (FA) optimization.

Implements both continuous and discrete (TSP) variants of the Firefly Algorithm,
a nature-inspired metaheuristic based on the flashing behavior of fireflies.

References
----------
.. [1] Yang, X. S. (2008). Nature-inspired metaheuristic algorithms. Luniver press.
.. [2] https://www.alpsconsult.net/post/firefly-algorithm-fa-overview
.. [3] Swap-based discrete FA for TSP
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from core.base_optimizer import BaseOptimizer
from core.problem_base import ProblemBase
from core.utils import get_best_solution, compute_brightness
from typing import List, Tuple


class FireflyContinuousOptimizer(BaseOptimizer):
    """
    Firefly Algorithm for continuous optimization problems.
    
    The algorithm simulates fireflies attracting each other based on brightness
    (fitness). Less bright fireflies move toward brighter ones according to:
    
        x_i = x_i + beta * (x_j - x_i) + alpha * (rand - 0.5)
    
    where:
        - beta = beta0 * exp(-gamma * r_ij^2) is the attractiveness
        - r_ij is the Euclidean distance between fireflies i and j
        - alpha is the randomization parameter
        - beta0 is the attractiveness at r=0
        - gamma is the light absorption coefficient
    
    Parameters
    ----------
    problem : ProblemBase
        The optimization problem (must be continuous).
    n_fireflies : int
        Number of fireflies in the population.
    alpha : float, default=0.2
        Randomization parameter (controls exploration).
    beta0 : float, default=1.0
        Attractiveness at distance r=0.
    gamma : float, default=1.0
        Light absorption coefficient (higher = more local search).
    seed : int or None, default=None
        Random seed for reproducibility.
    
    Attributes
    ----------
    problem : ProblemBase
        The optimization problem.
    n_fireflies : int
        Population size.
    alpha, beta0, gamma : float
        FA parameters.
    rng : np.random.RandomState
        Random number generator.
    positions : np.ndarray
        Current firefly positions, shape (n_fireflies, dim).
    fitness : np.ndarray
        Current fitness values, shape (n_fireflies,).
    
    Examples
    --------
    >>> from problems.continuous.sphere import SphereProblem
    >>> problem = SphereProblem(dim=2)
    >>> optimizer = FireflyContinuousOptimizer(problem, n_fireflies=20, seed=42)
    >>> best_sol, best_fit, history, trajectory = optimizer.run(max_iter=50)
    """
    
    def __init__(
        self,
        problem: ProblemBase,
        n_fireflies: int,
        alpha: float = 0.2,
        beta0: float = 1.0,
        gamma: float = 1.0,
        seed: int = None
    ):
        """Initialize Firefly Algorithm for continuous optimization."""
        # Validate problem type
        if problem.representation_type() != "continuous":
            raise ValueError(
                f"FireflyContinuousOptimizer requires continuous problem, "
                f"got '{problem.representation_type()}'"
            )
        
        self.problem = problem
        self.n_fireflies = n_fireflies
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Will be initialized in run()
        self.positions = None
        self.fitness = None
    
    def _init_population(self):
        """Initialize firefly population randomly."""
        self.positions = self.problem.init_solution(self.rng, self.n_fireflies)
        self.fitness = np.array([self.problem.evaluate(pos) for pos in self.positions])
    
    def _compute_distance(self, i: int, j: int) -> float:
        """Compute Euclidean distance between fireflies i and j."""
        return float(np.linalg.norm(self.positions[i] - self.positions[j]))
    
    def _move_firefly(self, i: int, j: int):
        """
        Move firefly i toward brighter firefly j.
        
        Parameters
        ----------
        i : int
            Index of firefly to move.
        j : int
            Index of brighter firefly (attracting).
        """
        # Compute distance
        r_ij = self._compute_distance(i, j)
        
        # Compute attractiveness (decreases with distance)
        beta = self.beta0 * np.exp(-self.gamma * r_ij**2)
        
        # Random perturbation for exploration
        dim = self.positions.shape[1]
        random_step = self.alpha * (self.rng.rand(dim) - 0.5)
        
        # Update position
        self.positions[i] = (
            self.positions[i]
            + beta * (self.positions[j] - self.positions[i])
            + random_step
        )
        
        # Clip to bounds
        self.positions[i] = self.problem.clip(self.positions[i])
        
        # Update fitness
        self.fitness[i] = self.problem.evaluate(self.positions[i])
    
    def run(self, max_iter: int) -> Tuple[np.ndarray, float, List[float], List[np.ndarray]]:
        """
        Run Firefly Algorithm for max_iter iterations.
        
        Parameters
        ----------
        max_iter : int
            Maximum number of iterations.
        
        Returns
        -------
        best_solution : np.ndarray
            Best solution found, shape (dim,).
        best_fitness : float
            Best fitness value (minimum).
        history_best : List[float]
            Best fitness at each iteration.
        trajectory : List[np.ndarray]
            Population at each iteration, each element has shape (n_fireflies, dim).
        """
        # Initialize
        self._init_population()
        
        history_best = []
        trajectory = []
        
        for iteration in range(max_iter):
            # Compute brightness (higher is better, so negate fitness)
            brightness = compute_brightness(self.fitness)
            
            # Move fireflies
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    # If j is brighter than i, move i toward j
                    if brightness[j] > brightness[i]:
                        self._move_firefly(i, j)
            
            # Track best solution
            best_sol, best_fit = get_best_solution(self.positions, self.fitness)
            history_best.append(best_fit)
            trajectory.append(self.positions.copy())
        
        # Final best solution
        best_solution, best_fitness = get_best_solution(self.positions, self.fitness)
        
        return best_solution, best_fitness, history_best, trajectory


class FireflyDiscreteTSPOptimizer(BaseOptimizer):
    """
    Firefly Algorithm for Traveling Salesman Problem (discrete optimization).
    
    Adapts FA to TSP by replacing continuous movement with swap-based operators.
    Instead of moving in Euclidean space, fireflies "move" by swapping cities
    in their tours to become more similar to better tours.
    
    Movement toward a better tour involves:
    1. Identifying differences between current tour and target (better) tour
    2. Applying swaps to reduce these differences
    3. Adding random swaps for exploration (equivalent to alpha in continuous FA)
    
    Parameters
    ----------
    problem : TSPProblem
        The TSP problem instance.
    n_fireflies : int
        Number of fireflies (tours) in the population.
    alpha_swap : float, default=0.2
        Probability of applying random swap for exploration.
    max_swaps_per_move : int, default=3
        Maximum number of swaps when moving toward another firefly.
    seed : int or None, default=None
        Random seed for reproducibility.
    
    Attributes
    ----------
    problem : TSPProblem
        The TSP problem.
    n_fireflies : int
        Population size.
    alpha_swap : float
        Random swap probability.
    max_swaps_per_move : int
        Maximum swaps per movement.
    rng : np.random.RandomState
        Random number generator.
    tours : np.ndarray
        Current tours, shape (n_fireflies, num_cities).
    fitness : np.ndarray
        Current tour lengths, shape (n_fireflies,).
    
    Examples
    --------
    >>> from problems.discrete.tsp import TSPProblem
    >>> coords = np.random.rand(10, 2)
    >>> problem = TSPProblem(coords)
    >>> optimizer = FireflyDiscreteTSPOptimizer(problem, n_fireflies=20, seed=42)
    >>> best_tour, best_length, history, trajectory = optimizer.run(max_iter=50)
    """
    
    def __init__(
        self,
        problem: ProblemBase,
        n_fireflies: int,
        alpha_swap: float = 0.2,
        max_swaps_per_move: int = 3,
        seed: int = None
    ):
        """Initialize Firefly Algorithm for TSP."""
        # Validate problem type
        if problem.representation_type() != "tsp":
            raise ValueError(
                f"FireflyDiscreteTSPOptimizer requires TSP problem, "
                f"got '{problem.representation_type()}'"
            )
        
        self.problem = problem
        self.n_fireflies = n_fireflies
        self.alpha_swap = alpha_swap
        self.max_swaps_per_move = max_swaps_per_move
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Will be initialized in run()
        self.tours = None
        self.fitness = None
    
    def _init_population(self):
        """Initialize firefly population with random tours."""
        self.tours = self.problem.init_solution(self.rng, self.n_fireflies)
        self.fitness = np.array([self.problem.evaluate(tour) for tour in self.tours])
    
    def _swap_move_towards(self, i_tour: np.ndarray, j_tour: np.ndarray) -> np.ndarray:
        """
        Apply swap operations to make i_tour more similar to j_tour.
        
        Strategy: Find positions where tours differ and swap cities in i_tour
        to match j_tour's structure.
        
        Parameters
        ----------
        i_tour : np.ndarray
            Current tour to modify.
        j_tour : np.ndarray
            Target (better) tour.
        
        Returns
        -------
        new_tour : np.ndarray
            Modified tour.
        """
        new_tour = i_tour.copy()
        num_cities = len(i_tour)
        
        # Apply limited number of swaps to move toward j_tour
        num_swaps = self.rng.randint(1, self.max_swaps_per_move + 1)
        
        for _ in range(num_swaps):
            # Find a position where tours differ
            diff_positions = np.where(new_tour != j_tour)[0]
            
            if len(diff_positions) == 0:
                break  # Tours are identical
            
            # Pick a random differing position
            pos1 = self.rng.choice(diff_positions)
            
            # Find where j_tour's city at pos1 is located in new_tour
            target_city = j_tour[pos1]
            pos2 = np.where(new_tour == target_city)[0][0]
            
            # Swap to bring target_city to pos1
            if pos1 != pos2:
                new_tour[pos1], new_tour[pos2] = new_tour[pos2], new_tour[pos1]
        
        # Apply random swap for exploration (with probability alpha_swap)
        if self.rng.rand() < self.alpha_swap:
            pos_a, pos_b = self.rng.choice(num_cities, size=2, replace=False)
            new_tour[pos_a], new_tour[pos_b] = new_tour[pos_b], new_tour[pos_a]
        
        return new_tour
    
    def run(self, max_iter: int) -> Tuple[np.ndarray, float, List[float], List[np.ndarray]]:
        """
        Run Firefly Algorithm for TSP for max_iter iterations.
        
        Parameters
        ----------
        max_iter : int
            Maximum number of iterations.
        
        Returns
        -------
        best_solution : np.ndarray
            Best tour found, shape (num_cities,).
        best_fitness : float
            Best tour length (minimum).
        history_best : List[float]
            Best tour length at each iteration.
        trajectory : List[np.ndarray]
            Population at each iteration, each element has shape (n_fireflies, num_cities).
        """
        # Initialize
        self._init_population()
        
        history_best = []
        trajectory = []
        
        for iteration in range(max_iter):
            # Compute brightness (lower fitness = higher brightness)
            brightness = compute_brightness(self.fitness)
            
            # Move fireflies
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    # If j is brighter (better tour) than i, move i toward j
                    if brightness[j] > brightness[i]:
                        new_tour = self._swap_move_towards(self.tours[i], self.tours[j])
                        new_fitness = self.problem.evaluate(new_tour)
                        
                        # Accept if better or with some probability
                        # (always accept for simplicity in basic version)
                        if new_fitness <= self.fitness[i]:
                            self.tours[i] = new_tour
                            self.fitness[i] = new_fitness
            
            # Track best solution
            best_sol, best_fit = get_best_solution(self.tours, self.fitness)
            history_best.append(best_fit)
            trajectory.append(self.tours.copy())
        
        # Final best solution
        best_solution, best_fitness = get_best_solution(self.tours, self.fitness)
        
        return best_solution, best_fitness, history_best, trajectory


if __name__ == "__main__":
    # Demo both continuous and discrete FA
    print("=" * 60)
    print("FIREFLY ALGORITHM DEMO")
    print("=" * 60)
    
    # Test 1: Continuous FA on Sphere function
    print("\n1. Continuous FA on Sphere Function (2D)")
    print("-" * 60)
    from problems.continuous.sphere import SphereProblem
    
    problem_sphere = SphereProblem(dim=2)
    fa_continuous = FireflyContinuousOptimizer(
        problem=problem_sphere,
        n_fireflies=10,
        alpha=0.2,
        beta0=1.0,
        gamma=1.0,
        seed=42
    )
    
    best_sol, best_fit, history, trajectory = fa_continuous.run(max_iter=10)
    
    print(f"Initial best fitness: {history[0]:.6f}")
    print(f"Final best fitness: {history[-1]:.6f}")
    print(f"Best solution: {best_sol}")
    print(f"Expected: near [0, 0], fitness near 0")
    print(f"Convergence: {history[0]:.4f} -> {history[-1]:.4f}")
    
    # Test 2: Discrete FA on TSP
    print("\n2. Discrete FA on TSP (10 cities)")
    print("-" * 60)
    from problems.discrete.tsp import TSPProblem
    
    rng_tsp = np.random.RandomState(123)
    coords_tsp = rng_tsp.rand(10, 2) * 10
    problem_tsp = TSPProblem(coords_tsp)
    
    fa_discrete = FireflyDiscreteTSPOptimizer(
        problem=problem_tsp,
        n_fireflies=10,
        alpha_swap=0.2,
        max_swaps_per_move=3,
        seed=42
    )
    
    best_tour, best_length, history_tsp, trajectory_tsp = fa_discrete.run(max_iter=10)
    
    print(f"Initial best tour length: {history_tsp[0]:.4f}")
    print(f"Final best tour length: {history_tsp[-1]:.4f}")
    print(f"Best tour: {best_tour}")
    print(f"Improvement: {history_tsp[0] - history_tsp[-1]:.4f}")
    print(f"Convergence: {history_tsp[0]:.4f} -> {history_tsp[-1]:.4f}")
    
    # Test 3: Continuous FA on Rastrigin (multimodal)
    print("\n3. Continuous FA on Rastrigin Function (2D, multimodal)")
    print("-" * 60)
    from problems.continuous.rastrigin import RastriginProblem
    
    problem_rastrigin = RastriginProblem(dim=2)
    fa_rastrigin = FireflyContinuousOptimizer(
        problem=problem_rastrigin,
        n_fireflies=20,
        alpha=0.25,
        beta0=1.0,
        gamma=0.5,  # Lower gamma for more global search
        seed=42
    )
    
    best_sol_r, best_fit_r, history_r, _ = fa_rastrigin.run(max_iter=30)
    
    print(f"Initial best fitness: {history_r[0]:.6f}")
    print(f"Final best fitness: {history_r[-1]:.6f}")
    print(f"Best solution: {best_sol_r}")
    print(f"Expected: near [0, 0], fitness near 0 (but multimodal, so harder)")
    
    print("\n" + "=" * 60)
    print("All Firefly Algorithm tests completed!")
    print("=" * 60)
