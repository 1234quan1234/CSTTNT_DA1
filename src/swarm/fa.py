"""
Firefly Algorithm (FA) optimization.

Implements both continuous and discrete (TSP) variants of the Firefly Algorithm,
a nature-inspired metaheuristic based on the flashing behavior of fireflies.

References
----------
.. [1] Yang, X. S. (2008). Nature-inspired metaheuristic algorithms. Luniver press.
.. [2] https://www.alpsconsult.net/post/firefly-algorithm-fa-overview
.. [3] https://www.researchgate.net/publication/320480703_Swap-Based_Discrete_Firefly_Algorithm_for_Traveling_Salesman_Problem
"""

import numpy as np
from typing import List, Tuple

# Use relative imports
from ..core.base_optimizer import BaseOptimizer
from ..core.problem_base import ProblemBase
from ..core.utils import get_best_solution, compute_brightness, euclidean_distance_matrix


class FireflyContinuousOptimizer(BaseOptimizer):
    """
    Firefly Algorithm for continuous optimization problems.
    
    The algorithm simulates fireflies attracting each other based on brightness
    (fitness). Less bright fireflies move toward brighter ones according to:
    
        x_i = x_i + β₀·exp(-γ·r²)·(x_j - x_i) + α·(rand - 0.5)
    
    where:
        - β = β₀·exp(-γ·r²) is the attractiveness
        - r is the Euclidean distance between fireflies i and j
        - α is the randomization parameter
        - β₀ is the attractiveness at r=0
        - γ is the light absorption coefficient
    
    Parameters
    ----------
    problem : ProblemBase
        The optimization problem (must have representation_type == "continuous").
    n_fireflies : int, default=30
        Number of fireflies in the population. Larger populations explore better
        but are slower. Typical range: 20-50.
    alpha : float, default=0.2
        Randomization parameter (controls exploration).
        - Lower (0.1): More exploitation, faster convergence
        - Higher (0.5): More exploration, better for multimodal functions
        Typical range: 0.1-0.5
    beta0 : float, default=1.0
        Attractiveness at distance r=0. Controls strength of attraction.
        Typical range: 0.5-2.0
    gamma : float, default=1.0
        Light absorption coefficient (controls attraction decay with distance).
        - Lower (0.1-0.5): More global search, long-range attraction
        - Higher (2.0-10.0): More local search, short-range attraction
        Typical range: 0.1-10.0
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
    
    Notes
    -----
    **Time Complexity:** O(max_iter · n² · d) where n=n_fireflies, d=dimension
    **Space Complexity:** O(n · d)
    
    **Parameter Tuning Guidelines:**
    - Unimodal problems (Sphere, Rosenbrock): gamma=1.0, alpha=0.2
    - Multimodal problems (Rastrigin, Ackley): gamma=0.5, alpha=0.3
    - High-dimensional (d>20): Increase n_fireflies, decrease gamma
    
    Examples
    --------
    >>> from problems.continuous.sphere import SphereProblem
    >>> problem = SphereProblem(dim=2)
    >>> optimizer = FireflyContinuousOptimizer(problem, n_fireflies=20, seed=42)
    >>> best_sol, best_fit, history, trajectory = optimizer.run(max_iter=50)
    >>> print(f"Best fitness: {best_fit:.6f}")
    """
    
    def __init__(
        self,
        problem: ProblemBase,
        n_fireflies: int = 30,
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
                f"got '{problem.representation_type()}'. "
                f"Use FireflyDiscreteTSPOptimizer for TSP problems."
            )
        
        # Validate parameters
        if n_fireflies < 2:
            raise ValueError(f"n_fireflies must be >= 2, got {n_fireflies}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if not 0.0 <= beta0 <= 5.0:
            raise ValueError(f"beta0 must be in [0, 5], got {beta0}")
        if not 0.0 <= gamma <= 20.0:
            raise ValueError(f"gamma must be in [0, 20], got {gamma}")
        
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
        """Initialize firefly population randomly within problem bounds."""
        self.positions = self.problem.init_solution(self.rng, self.n_fireflies)
        self.fitness = np.array([self.problem.evaluate(pos) for pos in self.positions])
    
    def _move_firefly(self, i: int, j: int, dist_matrix: np.ndarray):
        """
        Move firefly i toward brighter firefly j.
        
        Parameters
        ----------
        i : int
            Index of firefly to move (less bright).
        j : int
            Index of brighter firefly (attracting).
        dist_matrix : np.ndarray
            Precomputed distance matrix, shape (n_fireflies, n_fireflies).
        """
        # Get precomputed distance
        r_ij = dist_matrix[i, j]
        
        # Compute attractiveness (decreases exponentially with distance)
        beta = self.beta0 * np.exp(-self.gamma * r_ij**2)
        
        # Random perturbation for exploration
        dim = self.positions.shape[1]
        random_step = self.alpha * (self.rng.rand(dim) - 0.5)
        
        # Update position: x_i = x_i + β(x_j - x_i) + α·ε
        self.positions[i] = (
            self.positions[i]
            + beta * (self.positions[j] - self.positions[i])
            + random_step
        )
        
        # Ensure position stays within bounds
        self.positions[i] = self.problem.clip(self.positions[i].reshape(1, -1)).flatten()
        
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
        
        Notes
        -----
        Convergence typically occurs within 50-200 iterations for most problems.
        Monitor history_best to detect convergence plateau.
        """
        # Initialize
        self._init_population()
        
        history_best = []
        trajectory = []
        
        for iteration in range(max_iter):
            # Compute brightness (higher is better → negate fitness)
            brightness = compute_brightness(self.fitness)
            
            # Precompute distance matrix for efficiency
            dist_matrix = euclidean_distance_matrix(self.positions)
            
            # Move fireflies toward brighter ones
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    # If j is brighter than i, move i toward j
                    if brightness[j] > brightness[i]:
                        self._move_firefly(i, j, dist_matrix)
            
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
    2. Applying directed swaps to reduce these differences
    3. Adding random swaps for exploration (equivalent to α in continuous FA)
    
    Parameters
    ----------
    problem : TSPProblem
        The TSP problem instance (must have representation_type == "tsp").
    n_fireflies : int, default=30
        Number of fireflies (tours) in the population.
        Larger populations help escape local optima but increase runtime.
        Typical range: 20-50.
    alpha_swap : float, default=0.2
        Probability of applying random swap for exploration after directed movement.
        - Lower (0.1): More exploitation
        - Higher (0.4): More exploration
        Typical range: 0.1-0.4
    max_swaps_per_move : int, default=3
        Maximum number of directed swaps when moving toward another firefly.
        - Lower (1-2): More cautious, slower convergence
        - Higher (5-7): More aggressive, faster but may skip good solutions
        Typical range: 2-5
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
    
    Notes
    -----
    **Time Complexity:** O(max_iter · n² · m · k) where:
    - n = n_fireflies
    - m = num_cities
    - k = max_swaps_per_move
    
    **Parameter Tuning:**
    - Small TSP (< 20 cities): n_fireflies=20, max_swaps=2
    - Medium TSP (20-50 cities): n_fireflies=30, max_swaps=3
    - Large TSP (> 50 cities): n_fireflies=50, max_swaps=4-5
    
    Examples
    --------
    >>> from problems.discrete.tsp import TSPProblem
    >>> coords = np.random.rand(10, 2) * 100
    >>> problem = TSPProblem(coords)
    >>> optimizer = FireflyDiscreteTSPOptimizer(problem, n_fireflies=20, seed=42)
    >>> best_tour, best_length, history, trajectory = optimizer.run(max_iter=50)
    >>> print(f"Best tour length: {best_length:.4f}")
    """
    
    def __init__(
        self,
        problem: ProblemBase,
        n_fireflies: int = 30,
        alpha_swap: float = 0.2,
        max_swaps_per_move: int = 3,
        seed: int = None
    ):
        """Initialize Firefly Algorithm for TSP."""
        # Validate problem type
        if problem.representation_type() != "tsp":
            raise ValueError(
                f"FireflyDiscreteTSPOptimizer requires TSP problem, "
                f"got '{problem.representation_type()}'. "
                f"Use FireflyContinuousOptimizer for continuous problems."
            )
        
        # Validate parameters
        if n_fireflies < 2:
            raise ValueError(f"n_fireflies must be >= 2, got {n_fireflies}")
        if not 0.0 <= alpha_swap <= 1.0:
            raise ValueError(f"alpha_swap must be in [0, 1], got {alpha_swap}")
        if max_swaps_per_move < 1:
            raise ValueError(f"max_swaps_per_move must be >= 1, got {max_swaps_per_move}")
        
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
        to match j_tour's structure. This is the discrete equivalent of
        continuous attraction in standard FA.
        
        Parameters
        ----------
        i_tour : np.ndarray
            Current tour to modify (less bright firefly).
        j_tour : np.ndarray
            Target (better) tour (brighter firefly).
        
        Returns
        -------
        new_tour : np.ndarray
            Modified tour after directed swaps.
        
        Notes
        -----
        This implements a simplified version of the swap-based operator
        described in literature for discrete FA on TSP.
        """
        new_tour = i_tour.copy()
        num_cities = len(i_tour)
        
        # Apply limited number of directed swaps to move toward j_tour
        num_swaps = self.rng.randint(1, self.max_swaps_per_move + 1)
        
        for _ in range(num_swaps):
            # Find positions where tours differ
            diff_positions = np.where(new_tour != j_tour)[0]
            
            if len(diff_positions) == 0:
                break  # Tours are identical, no more swaps needed
            
            # Pick a random differing position
            pos1 = self.rng.choice(diff_positions)
            
            # Find where j_tour's city at pos1 is located in new_tour
            target_city = j_tour[pos1]
            pos2 = np.where(new_tour == target_city)[0][0]
            
            # Swap to bring target_city to pos1 (align with j_tour)
            if pos1 != pos2:
                new_tour[pos1], new_tour[pos2] = new_tour[pos2], new_tour[pos1]
        
        # Apply random swap for exploration (equivalent to α random term)
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
        
        Notes
        -----
        Convergence detection: Monitor history_best for plateau.
        Typical improvement seen in first 50-100 iterations for small TSP instances.
        For large instances, may need 200-500 iterations.
        """
        # Initialize
        self._init_population()
        
        history_best = []
        trajectory = []
        
        for iteration in range(max_iter):
            # Compute brightness (lower tour length = higher brightness)
            brightness = compute_brightness(self.fitness)
            
            # Move fireflies toward brighter ones
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    # If j has better tour (brighter) than i, move i toward j
                    if brightness[j] > brightness[i]:
                        new_tour = self._swap_move_towards(self.tours[i], self.tours[j])
                        new_fitness = self.problem.evaluate(new_tour)
                        
                        # Accept if better (greedy acceptance)
                        # Note: Could add probabilistic acceptance for diversity
                        if new_fitness < self.fitness[i]:
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
    print("=" * 70)
    print("FIREFLY ALGORITHM COMPREHENSIVE DEMO")
    print("=" * 70)
    
    # Test 1: Continuous FA on Sphere function
    print("\n[TEST 1] Continuous FA on Sphere Function (2D)")
    print("-" * 70)
    from problems.continuous.sphere import SphereProblem
    
    problem_sphere = SphereProblem(dim=2)
    fa_continuous = FireflyContinuousOptimizer(
        problem=problem_sphere,
        n_fireflies=15,
        alpha=0.2,
        beta0=1.0,
        gamma=1.0,
        seed=42
    )
    
    best_sol, best_fit, history, trajectory = fa_continuous.run(max_iter=30)
    
    print(f"Problem: Minimize f(x,y) = x² + y²")
    print(f"Global minimum: f(0,0) = 0")
    print(f"\nResults:")
    print(f"  Initial best fitness: {history[0]:.6f}")
    print(f"  Final best fitness:   {history[-1]:.6f}")
    print(f"  Best solution:        [{best_sol[0]:.6f}, {best_sol[1]:.6f}]")
    print(f"  Improvement:          {history[0] - history[-1]:.6f}")
    print(f"  Convergence:          {history[0]:.4f} → {history[-1]:.4f}")
    
    # Test 2: Continuous FA on Rastrigin (multimodal)
    print("\n[TEST 2] Continuous FA on Rastrigin Function (2D, multimodal)")
    print("-" * 70)
    from problems.continuous.rastrigin import RastriginProblem
    
    problem_rastrigin = RastriginProblem(dim=2)
    fa_rastrigin = FireflyContinuousOptimizer(
        problem=problem_rastrigin,
        n_fireflies=25,
        alpha=0.3,      # Higher alpha for multimodal
        beta0=1.0,
        gamma=0.5,      # Lower gamma for more global search
        seed=42
    )
    
    best_sol_r, best_fit_r, history_r, _ = fa_rastrigin.run(max_iter=50)
    
    print(f"Problem: Rastrigin (highly multimodal, many local minima)")
    print(f"Global minimum: f(0,0) = 0")
    print(f"\nResults:")
    print(f"  Initial best fitness: {history_r[0]:.6f}")
    print(f"  Final best fitness:   {history_r[-1]:.6f}")
    print(f"  Best solution:        [{best_sol_r[0]:.6f}, {best_sol_r[1]:.6f}]")
    print(f"  Improvement:          {history_r[0] - history_r[-1]:.6f}")
    print(f"Note: Rastrigin is harder due to many local minima")
    
    # Test 3: Discrete FA on TSP
    print("\n[TEST 3] Discrete FA on TSP (15 cities)")
    print("-" * 70)
    from problems.discrete.tsp import TSPProblem
    
    rng_tsp = np.random.RandomState(123)
    coords_tsp = rng_tsp.rand(15, 2) * 100
    problem_tsp = TSPProblem(coords_tsp)
    
    fa_discrete = FireflyDiscreteTSPOptimizer(
        problem=problem_tsp,
        n_fireflies=25,
        alpha_swap=0.2,
        max_swaps_per_move=3,
        seed=42
    )
    
    best_tour, best_length, history_tsp, trajectory_tsp = fa_discrete.run(max_iter=80)
    
    print(f"Problem: TSP with {problem_tsp.num_cities} cities")
    print(f"Search space size: {np.math.factorial(problem_tsp.num_cities):,} possible tours")
    print(f"\nResults:")
    print(f"  Initial best tour length: {history_tsp[0]:.4f}")
    print(f"  Final best tour length:   {history_tsp[-1]:.4f}")
    print(f"  Best tour: {best_tour}")
    print(f"  Improvement:              {history_tsp[0] - history_tsp[-1]:.4f}")
    print(f"  Improvement %:            {100 * (history_tsp[0] - history_tsp[-1]) / history_tsp[0]:.2f}%")
    
    # Test 4: Reproducibility test
    print("\n[TEST 4] Reproducibility Test")
    print("-" * 70)
    
    fa1 = FireflyContinuousOptimizer(problem_sphere, n_fireflies=10, seed=999)
    fa2 = FireflyContinuousOptimizer(problem_sphere, n_fireflies=10, seed=999)
    
    _, fit1, hist1, _ = fa1.run(max_iter=20)
    _, fit2, hist2, _ = fa2.run(max_iter=20)
    
    print(f"Run 1 final fitness: {fit1:.10f}")
    print(f"Run 2 final fitness: {fit2:.10f}")
    print(f"Identical results:   {fit1 == fit2}")
    print(f"History identical:   {np.allclose(hist1, hist2)}")
    
    if fit1 == fit2:
        print("✓ Reproducibility test PASSED")
    else:
        print("✗ Reproducibility test FAILED")
    
    print("\n" + "=" * 70)
    print("All Firefly Algorithm tests completed successfully!")
    print("=" * 70)
    print("All Firefly Algorithm tests completed successfully!")
    print("=" * 70)
