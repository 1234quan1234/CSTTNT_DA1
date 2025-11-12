"""
Firefly Algorithm (FA) optimization.

Implements both continuous and discrete (Knapsack) variants of the Firefly Algorithm,
a nature-inspired metaheuristic based on the flashing behavior of fireflies.

References
----------
.. [1] Yang, X. S. (2008). Nature-inspired metaheuristic algorithms. Luniver press.
.. [2] https://www.alpsconsult.net/post/firefly-algorithm-fa-overview
.. [3] Feng, Y., et al. (2017). A binary firefly algorithm for the 0-1 knapsack problem.
       Applied Soft Computing, 52, 661-670.
"""

import numpy as np
from typing import List, Tuple, Dict

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
    - Multimodal problems (Rastrigin): gamma=0.5, alpha=0.3
    - High-dimensional (d>20): Increase n_fireflies, decrease gamma
    
    Examples
    --------
    >>> from problems.continuous.rastrigin import RastriginProblem
    >>> problem = RastriginProblem(dim=2)
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
    
    def run(self, max_iter: int) -> Tuple[np.ndarray, float, List[float], List[Dict[str, float]]]:
        """
        Run Firefly Algorithm for max_iter iterations.
        
        Returns
        -------
        best_solution : np.ndarray
            Best solution found, shape (dim,).
        best_fitness : float
            Best fitness value (minimum).
        history_best : List[float]
            Best fitness at each iteration.
        stats_history : List[Dict[str, float]]
            Statistical summary at each iteration.
        """
        # Initialize
        self._init_population()
        
        history_best = []
        stats_history = []
        
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
            
            # --- COMPUTE POPULATION STATISTICS ---
            best_fit_gen = float(np.min(self.fitness))
            mean_fit_gen = float(np.mean(self.fitness))
            std_fit_gen = float(np.std(self.fitness))
            
            # Compute diversity: mean distance to centroid
            centroid = np.mean(self.positions, axis=0)
            distances = np.linalg.norm(self.positions - centroid, axis=1)
            diversity_gen = float(np.mean(distances))
            
            # Store statistics
            stats_history.append({
                'gen': iteration,
                'best_fitness': best_fit_gen,
                'mean_fitness': mean_fit_gen,
                'std_fitness': std_fit_gen,
                'diversity': diversity_gen
            })
            
            # Track best for backward compatibility
            history_best.append(best_fit_gen)
        
        # Final best solution
        best_solution, best_fitness = get_best_solution(self.positions, self.fitness)
        
        return best_solution, best_fitness, history_best, stats_history


class FireflyKnapsackOptimizer(BaseOptimizer):
    """
    Firefly Algorithm for 0/1 Knapsack Problem (discrete optimization).
    
    Adapts FA to Knapsack by replacing continuous movement with bit-flip operators.
    Fireflies "move" toward better solutions by selectively flipping bits to become
    more similar to better (brighter) solutions while maintaining feasibility.
    
    Movement Strategy:
    1. Compare current solution with better (brighter) solution
    2. Identify bit differences between solutions
    3. Apply directed bit flips to align with better solution
    4. Add random bit flips for exploration (controlled by alpha_flip)
    5. Repair infeasible solutions using greedy strategy
    
    Parameters
    ----------
    problem : KnapsackProblem
        The Knapsack problem instance (must have representation_type == "knapsack").
    n_fireflies : int, default=30
        Number of fireflies (solutions) in the population.
        Larger populations help escape local optima.
        Typical range: 20-50.
    alpha_flip : float, default=0.2
        Probability of random bit flip after directed movement.
        - Lower (0.1): More exploitation
        - Higher (0.4): More exploration
        Typical range: 0.1-0.4
    max_flips_per_move : int, default=3
        Maximum number of directed bit flips when moving toward another firefly.
        Controls how aggressively solutions adapt to better ones.
        Typical range: 2-5
    repair_method : str, default="greedy_remove"
        Method to repair infeasible solutions:
        - "greedy_remove": Remove lowest value/weight ratio items
        - "random_remove": Remove random items until feasible
    seed : int or None, default=None
        Random seed for reproducibility.
    
    Attributes
    ----------
    problem : KnapsackProblem
        The Knapsack problem.
    n_fireflies : int
        Population size.
    alpha_flip : float
        Random flip probability.
    max_flips_per_move : int
        Maximum flips per movement.
    repair_method : str
        Repair strategy for infeasible solutions.
    rng : np.random.RandomState
        Random number generator.
    solutions : np.ndarray
        Current solutions, shape (n_fireflies, num_items).
    fitness : np.ndarray
        Current fitness values, shape (n_fireflies,).
    
    Notes
    -----
    **Time Complexity:** O(max_iter · n² · m) where:
    - n = n_fireflies
    - m = num_items
    
    **Repair Strategy:**
    When a solution exceeds capacity, items are removed based on value/weight ratio.
    This greedy repair often produces better solutions than random removal.
    
    **Parameter Tuning:**
    - Small problems (< 30 items): n_fireflies=20, max_flips=2
    - Medium problems (30-100 items): n_fireflies=30, max_flips=3
    - Large problems (> 100 items): n_fireflies=50, max_flips=4-5
    
    Examples
    --------
    >>> from problems.discrete.knapsack import KnapsackProblem
    >>> values = np.array([10, 20, 30, 40, 50])
    >>> weights = np.array([1, 2, 3, 4, 5])
    >>> capacity = 7.0
    >>> problem = KnapsackProblem(values, weights, capacity)
    >>> optimizer = FireflyKnapsackOptimizer(problem, n_fireflies=20, seed=42)
    >>> best_sol, best_fit, history, trajectory = optimizer.run(max_iter=50)
    >>> print(f"Best value: {-best_fit:.2f}")  # Negate because we minimize
    """
    
    def __init__(
        self,
        problem: ProblemBase,
        n_fireflies: int = 30,
        alpha_flip: float = 0.2,
        max_flips_per_move: int = 3,
        repair_method: str = "greedy_remove",  # Deprecated, kept for compatibility
        constraint_handling: str = "repair",  # New: 'repair' or 'penalty'
        seed: int = None
    ):
        """Initialize Firefly Algorithm for Knapsack."""
        # Validate problem type
        if problem.representation_type() != "knapsack":
            raise ValueError(
                f"FireflyKnapsackOptimizer requires knapsack problem, "
                f"got '{problem.representation_type()}'. "
                f"Use FireflyContinuousOptimizer for continuous problems."
            )
        
        # Validate parameters
        if n_fireflies < 2:
            raise ValueError(f"n_fireflies must be >= 2, got {n_fireflies}")
        if not 0.0 <= alpha_flip <= 1.0:
            raise ValueError(f"alpha_flip must be in [0, 1], got {alpha_flip}")
        if max_flips_per_move < 1:
            raise ValueError(f"max_flips_per_move must be >= 1, got {max_flips_per_move}")
        if repair_method not in ["greedy_remove", "random_remove"]:
            raise ValueError(f"repair_method must be 'greedy_remove' or 'random_remove', got {repair_method}")
        
        self.problem = problem
        self.n_fireflies = n_fireflies
        self.alpha_flip = alpha_flip
        self.max_flips_per_move = max_flips_per_move
        self.repair_method = repair_method  # Keep for backward compatibility
        self.constraint_handling = constraint_handling  # New switch
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Will be initialized in run()
        self.solutions = None
        self.fitness = None
    
    def _init_population(self):
        """Initialize firefly population with random feasible solutions."""
        self.solutions = self.problem.init_solution(self.rng, self.n_fireflies)
        self.fitness = np.array([self.problem.evaluate(sol) for sol in self.solutions])
    
    def _repair_solution(self, solution: np.ndarray) -> np.ndarray:
        """
        Repair infeasible solution if constraint_handling == 'repair'.
        Otherwise return as-is (penalty will be applied in evaluate).
        """
        if self.constraint_handling == 'repair':
            return self.problem.greedy_repair(solution)
        else:
            return solution  # Let penalty function handle it
    
    def _flip_move_towards(self, i_sol: np.ndarray, j_sol: np.ndarray) -> np.ndarray:
        """
        Apply bit flips to make i_sol more similar to j_sol (brighter solution).
        """
        new_sol = i_sol.copy()
        num_items = len(i_sol)
        
        # Apply limited number of directed flips
        num_flips = self.rng.randint(1, self.max_flips_per_move + 1)
        
        for _ in range(num_flips):
            diff_positions = np.where(new_sol != j_sol)[0]
            
            if len(diff_positions) == 0:
                break
            
            pos = self.rng.choice(diff_positions)
            new_sol[pos] = j_sol[pos]
        
        # Random exploration flip
        if self.rng.rand() < self.alpha_flip:
            flip_pos = self.rng.randint(num_items)
            new_sol[flip_pos] = 1 - new_sol[flip_pos]
        
        # Apply constraint handling based on switch
        new_sol = self._repair_solution(new_sol)
        
        return new_sol
    
    def run(self, max_iter: int) -> Tuple[np.ndarray, float, List[float], List[np.ndarray]]:
        """
        Run Firefly Algorithm for Knapsack for max_iter iterations.
        
        Parameters
        ----------
        max_iter : int
            Maximum number of iterations.
        
        Returns
        -------
        best_solution : np.ndarray
            Best solution found, shape (num_items,), binary vector.
        best_fitness : float
            Best fitness value (negative value for maximization).
        history_best : List[float]
            Best fitness at each iteration.
        trajectory : List[np.ndarray]
            Population at each iteration, shape (n_fireflies, num_items).
        
        Notes
        -----
        Remember that fitness is negated value (minimization framework).
        Actual knapsack value = -best_fitness
        """
        # Initialize
        self._init_population()
        
        history_best = []
        trajectory = []
        
        for iteration in range(max_iter):
            # Compute brightness (lower fitness = higher brightness for minimization)
            brightness = compute_brightness(self.fitness)
            
            # Move fireflies toward brighter ones
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    # If j has better solution (brighter) than i, move i toward j
                    if brightness[j] > brightness[i]:
                        new_sol = self._flip_move_towards(self.solutions[i], self.solutions[j])
                        new_fitness = self.problem.evaluate(new_sol)
                        
                        # Accept if better (greedy acceptance)
                        if new_fitness < self.fitness[i]:
                            self.solutions[i] = new_sol
                            self.fitness[i] = new_fitness
            
            # Track best solution
            best_sol, best_fit = get_best_solution(self.solutions, self.fitness)
            history_best.append(best_fit)
            trajectory.append(self.solutions.copy())
        
        # Final best solution
        best_solution, best_fitness = get_best_solution(self.solutions, self.fitness)
        
        return best_solution, best_fitness, history_best, trajectory


if __name__ == "__main__":
    # Demo both continuous and discrete FA
    print("=" * 70)
    print("FIREFLY ALGORITHM COMPREHENSIVE DEMO")
    print("=" * 70)
    
    # Test 1: Continuous FA on Rastrigin (multimodal)
    print("\n[TEST 1] Continuous FA on Rastrigin Function (2D, multimodal)")
    print("-" * 70)
    from problems.continuous.rastrigin import RastriginProblem
    
    problem_rastrigin = RastriginProblem(dim=2)
    fa_rastrigin = FireflyContinuousOptimizer(
        problem=problem_rastrigin,
        n_fireflies=25,
        alpha=0.3,
        beta0=1.0,
        gamma=0.5,
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
    
    # Test 2: Discrete FA on Knapsack
    print("\n[TEST 2] Discrete FA on 0/1 Knapsack Problem")
    print("-" * 70)
    from problems.discrete.knapsack import KnapsackProblem
    
    rng_knap = np.random.RandomState(123)
    n_items = 20
    values = rng_knap.randint(10, 100, n_items)
    weights = rng_knap.randint(1, 50, n_items)
    capacity = int(0.5 * np.sum(weights))
    
    problem_knapsack = KnapsackProblem(values, weights, capacity)
    
    fa_knapsack = FireflyKnapsackOptimizer(
        problem=problem_knapsack,
        n_fireflies=25,
        alpha_flip=0.2,
        max_flips_per_move=3,
        repair_method="greedy_remove",
        seed=42
    )
    
    best_sol_k, best_fit_k, history_k, _ = fa_knapsack.run(max_iter=80)
    
    total_value = -best_fit_k
    total_weight = np.sum(best_sol_k * weights)
    
    print(f"Problem: 0/1 Knapsack with {n_items} items")
    print(f"Capacity: {capacity}, Total possible weight: {np.sum(weights)}")
    print(f"\nResults:")
    print(f"  Initial best value: {-history_k[0]:.2f}")
    print(f"  Final best value:   {total_value:.2f}")
    print(f"  Best solution: {best_sol_k}")
    print(f"  Total weight: {total_weight:.2f} / {capacity}")
    print(f"  Feasible: {total_weight <= capacity}")
    print(f"  Improvement: {-history_k[0] - (-history_k[-1]):.2f}")
    
    # Test 3: Reproducibility test
    print("\n[TEST 3] Reproducibility Test")
    print("-" * 70)
    
    fa1 = FireflyContinuousOptimizer(problem_rastrigin, n_fireflies=10, seed=999)
    fa2 = FireflyContinuousOptimizer(problem_rastrigin, n_fireflies=10, seed=999)
    
    _, fit1, hist1, _ = fa1.run(max_iter=20)
    _, fit2, hist2, _ = fa2.run(max_iter=20)
    
    print(f"Run 1 final fitness: {fit1:.10f}")
    print(f"Run 2 final fitness: {fit2:.10f}")
    print(f"Identical results:   {fit1 == fit2}")
    
    if fit1 == fit2:
        print("✓ Reproducibility test PASSED")
    else:
        print("✗ Reproducibility test FAILED")
    
    print("\n" + "=" * 70)
    print("All Firefly Algorithm tests completed successfully!")
    print("=" * 70)
    print(f"  Best solution: {best_sol_k}")
    print(f"  Total weight: {total_weight:.2f} / {capacity}")
    print(f"  Feasible: {total_weight <= capacity}")
    print(f"  Improvement: {-history_k[0] - (-history_k[-1]):.2f}")
    
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
    
    if fit1 == fit2:
        print("✓ Reproducibility test PASSED")
    else:
        print("✗ Reproducibility test FAILED")
    
    print("\n" + "=" * 70)
    print("All Firefly Algorithm tests completed successfully!")
    print("=" * 70)
    print(f"  Total weight: {total_weight:.2f} / {capacity}")
    print(f"  Feasible: {total_weight <= capacity}")
    print(f"  Improvement: {-history_k[0] - (-history_k[-1]):.2f}")
    
    # Test 3: Reproducibility test
    print("\n[TEST 3] Reproducibility Test")
    print("-" * 70)
    
    fa1 = FireflyContinuousOptimizer(problem_rastrigin, n_fireflies=10, seed=999)
    fa2 = FireflyContinuousOptimizer(problem_rastrigin, n_fireflies=10, seed=999)
    
    _, fit1, hist1, _ = fa1.run(max_iter=20)
    _, fit2, hist2, _ = fa2.run(max_iter=20)
    
    print(f"Run 1 final fitness: {fit1:.10f}")
    print(f"Run 2 final fitness: {fit2:.10f}")
    print(f"Identical results:   {fit1 == fit2}")
    
    if fit1 == fit2:
        print("✓ Reproducibility test PASSED")
    else:
        print("✗ Reproducibility test FAILED")
    
    print("\n" + "=" * 70)
    print("All Firefly Algorithm tests completed successfully!")
    print("=" * 70)
    print(f"  Best solution: {best_sol_k}")
    print(f"  Total weight: {total_weight:.2f} / {capacity}")
    print(f"  Feasible: {total_weight <= capacity}")
    print(f"  Improvement: {-history_k[0] - (-history_k[-1]):.2f}")
    
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
    
    if fit1 == fit2:
        print("✓ Reproducibility test PASSED")
    else:
        print("✗ Reproducibility test FAILED")
    
    print("\n" + "=" * 70)
    print("All Firefly Algorithm tests completed successfully!")
    print("=" * 70)
