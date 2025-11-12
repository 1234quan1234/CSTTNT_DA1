"""
Simulated Annealing optimization algorithm.

Simulated Annealing is a probabilistic technique that approximates the global
optimum by accepting worse solutions with probability that decreases over time,
allowing escape from local optima.

References
----------
.. [1] https://en.wikipedia.org/wiki/Simulated_annealing
"""

import numpy as np
from typing import List, Tuple, Dict
import logging
import math  # Add math import for exp

from ..core.base_optimizer import BaseOptimizer
from ..core.problem_base import ProblemBase

logger = logging.getLogger(__name__)

class SimulatedAnnealingOptimizer(BaseOptimizer):
    """
    Simulated Annealing optimizer.
    
    Inspired by annealing in metallurgy, SA accepts worse solutions with
    probability exp(-ΔE/T) where ΔE is the fitness increase and T is temperature.
    Temperature decreases over iterations, making the algorithm more greedy over time.
    
    Supports both continuous and discrete problems:
    - Continuous: generates neighbors by adding Gaussian noise
    - Knapsack: generates neighbors by flipping one bit
    
    Parameters
    ----------
    problem : ProblemBase
        The optimization problem.
    initial_temp : float, default=100.0
        Initial temperature T0.
    cooling_rate : float, default=0.95
        Cooling schedule parameter (multiplicative decay).
        T(t) = T0 * cooling_rate^t
    step_size : float, default=0.1
        Step size for continuous problems (Gaussian noise std dev).
    seed : int or None, default=None
        Random seed for reproducibility.
    
    Attributes
    ----------
    problem : ProblemBase
        The optimization problem.
    initial_temp : float
        Initial temperature.
    cooling_rate : float
        Temperature decay rate.
    step_size : float
        Perturbation step size for continuous problems.
    rng : np.random.RandomState
        Random number generator.
    current_solution : np.ndarray
        Current solution.
    current_fitness : float
        Current fitness value.
    best_solution : np.ndarray
        Best solution found so far.
    best_fitness : float
        Best fitness found so far.
    
    Examples
    --------
    >>> from problems.continuous.rastrigin import RastriginProblem
    >>> problem = RastriginProblem(dim=2)
    >>> optimizer = SimulatedAnnealingOptimizer(problem, seed=42)
    >>> best_sol, best_fit, history, trajectory = optimizer.run(max_iter=100)
    """
    
    def __init__(
        self,
        problem: ProblemBase,
        initial_temp: float = 100.0,
        cooling_rate: float = 0.95,
        step_size: float = 0.1,
        min_temp: float = 1e-8,
        repair_method: str = None,  # Deprecated
        constraint_handling: str = "penalty",  # New: 'repair' or 'penalty'
        seed: int = None
    ):
        """Initialize Simulated Annealing optimizer."""
        # Validate parameters
        if initial_temp <= 0:
            raise ValueError(f"initial_temp must be > 0, got {initial_temp}")
        if not 0 < cooling_rate < 1:
            raise ValueError(f"cooling_rate must be in (0, 1), got {cooling_rate}")
        if min_temp <= 0:
            raise ValueError(f"min_temp must be > 0, got {min_temp}")
        if initial_temp <= min_temp:
            raise ValueError(f"initial_temp ({initial_temp}) must be > min_temp ({min_temp})")
        if step_size <= 0:
            raise ValueError(f"step_size must be > 0, got {step_size}")
        
        self.problem = problem
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.step_size = step_size
        self.min_temp = min_temp
        self.repair_method = repair_method  # Deprecated
        self.constraint_handling = constraint_handling  # New switch
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        self.current_solution = None
        self.current_fitness = None
        self.best_solution = None
        self.best_fitness = None
    
    def _generate_neighbor_continuous(self) -> np.ndarray:
        """Generate a neighbor for continuous problems by adding Gaussian noise."""
        dim = len(self.current_solution)
        noise = self.rng.randn(dim) * self.step_size
        neighbor = self.current_solution + noise
        return self.problem.clip(neighbor)
    
    def _generate_neighbor_knapsack(self) -> np.ndarray:
        """Generate a neighbor for Knapsack by flipping one bit."""
        neighbor = self.current_solution.copy()
        flip_idx = self.rng.randint(len(neighbor))
        neighbor[flip_idx] = 1 - neighbor[flip_idx]
        return neighbor
    
    def _generate_neighbor(self) -> np.ndarray:
        """Generate a neighbor based on problem type."""
        repr_type = self.problem.representation_type()
        
        if repr_type == "continuous":
            return self._generate_neighbor_continuous()
        elif repr_type == "knapsack":
            return self._generate_neighbor_knapsack()
        else:
            raise NotImplementedError(f"Unsupported problem type: {repr_type}")
    
    def _repair_knapsack(self, solution: np.ndarray) -> np.ndarray:
        """Repair infeasible Knapsack solution based on constraint_handling."""
        if self.constraint_handling != 'repair':
            return solution  # Let penalty handle it
        
        if self.problem.representation_type() != 'knapsack':
            return solution
        
        return self.problem.greedy_repair(solution)
    
    def _acceptance_probability(self, delta_e: float, temperature: float) -> float:
        """Compute acceptance probability for a worse solution with numerical stability."""
        if delta_e < 0:
            return 1.0
        else:
            # Clamp to avoid overflow: exp(-delta/T)
            # When T → 0, delta/T → ∞, so we clamp the exponent
            z = -delta_e / max(temperature, 1e-300)
            # Clamp z to avoid exp overflow (exp(700) ≈ 1e304)
            z_clamped = max(z, -700.0)
            return math.exp(z_clamped)
    
    def run(self, max_iter: int) -> Tuple[np.ndarray, float, List[float], List[Dict[str, float]]]:
        """
        Run Simulated Annealing for max_iter iterations.
        
        Returns
        -------
        best_solution : np.ndarray
        best_fitness : float
        history_best : List[float]
        stats_history : List[Dict[str, float]]
            For SA (single solution), diversity is always 0.
        """
        # Initialize
        self.current_solution = self.problem.init_solution(self.rng, n=1)[0]
        self.current_solution = self._repair_knapsack(self.current_solution)
        self.current_fitness = self.problem.evaluate(self.current_solution)
        
        self.best_solution = self.current_solution.copy()
        self.best_fitness = self.current_fitness
        
        history_best = []
        stats_history = []
        
        for iteration in range(max_iter):
            # Calculate temperature with clamping (no break)
            temperature = self.initial_temp * (self.cooling_rate ** iteration)
            if temperature < self.min_temp:
                temperature = self.min_temp  # Clamp instead of breaking
            
            # Generate neighbor - this is 1 evaluation
            neighbor = self._generate_neighbor()
            neighbor = self._repair_knapsack(neighbor)
            neighbor_fitness = self.problem.evaluate(neighbor)
            
            # Acceptance decision
            delta_e = neighbor_fitness - self.current_fitness
            accept_prob = self._acceptance_probability(delta_e, temperature)
            
            if self.rng.rand() < accept_prob:
                self.current_solution = neighbor
                self.current_fitness = neighbor_fitness
                
                if self.current_fitness < self.best_fitness:
                    self.best_solution = self.current_solution.copy()
                    self.best_fitness = self.current_fitness
            
            # For single-solution: no population diversity
            stats_history.append({
                'gen': iteration,
                'best_fitness': float(self.best_fitness),
                'mean_fitness': float(self.best_fitness),  # Same as best
                'std_fitness': 0.0,  # No population
                'diversity': 0.0  # No population
            })
            
            # Track progress
            history_best.append(self.best_fitness)
        
        return self.best_solution, self.best_fitness, history_best, stats_history


if __name__ == "__main__":
    print("=" * 60)
    print("SIMULATED ANNEALING OPTIMIZER DEMO")
    print("=" * 60)
    
    # Test on Rastrigin
    print("\n1. Simulated Annealing on Rastrigin Function (2D)")
    print("-" * 60)
    from problems.continuous.rastrigin import RastriginProblem
    
    problem_rastrigin = RastriginProblem(dim=2)
    sa_rastrigin = SimulatedAnnealingOptimizer(
        problem=problem_rastrigin,
        initial_temp=100,
        cooling_rate=0.95,
        seed=42
    )
    
    best_sol, best_fit, history, _ = sa_rastrigin.run(max_iter=100)
    
    print(f"Initial fitness: {history[0]:.6f}")
    print(f"Final fitness: {history[-1]:.6f}")
    print(f"Best solution: {best_sol}")
    print(f"Improvement: {history[0] - history[-1]:.6f}")
    
        # Test on Knapsack
    print("\n2. Simulated Annealing on Knapsack Problem")
    print("-" * 60)
    from problems.discrete.knapsack import KnapsackProblem
    
    values = np.array([10, 20, 30, 40, 50, 60])
    weights = np.array([1, 2, 3, 4, 5, 6])
    capacity = 10.0
    problem_knapsack = KnapsackProblem(values, weights, capacity)
    
    sa_knapsack = SimulatedAnnealingOptimizer(
        problem=problem_knapsack,
        initial_temp=10.0,
        cooling_rate=0.95,
        seed=42
    )
    
    best_sel, best_fit_k, history_k, _ = sa_knapsack.run(max_iter=100)
    
    print(f"Initial fitness: {history_k[0]:.2f}")
    print(f"Final fitness: {history_k[-1]:.2f}")
    print(f"Best selection: {best_sel}")
    total_weight = np.sum(best_sel * weights)
    total_value = np.sum(best_sel * values)
    print(f"Total weight: {total_weight}, Total value: {total_value}")
    print(f"Feasible: {total_weight <= capacity}")
    
    # Test 3: Rastrigin (multimodal - SA should handle better than HC)
    print("\n3. Simulated Annealing on Rastrigin (2D, multimodal)")
    print("-" * 60)
    from problems.continuous.rastrigin import RastriginProblem
    
    problem_rastrigin = RastriginProblem(dim=2)
    sa_rastrigin = SimulatedAnnealingOptimizer(
        problem=problem_rastrigin,
        initial_temp=200.0,
        cooling_rate=0.97,
        step_size=0.3,
        seed=42
    )
    
    best_sol_r, best_fit_r, history_r, _ = sa_rastrigin.run(max_iter=200)
    
    print(f"Initial fitness: {history_r[0]:.6f}")
    print(f"Final fitness: {history_r[-1]:.6f}")
    print(f"Best solution: {best_sol_r}")
    print(f"Improvement: {history_r[0] - history_r[-1]:.6f}")
    print(f"(Note: Rastrigin is highly multimodal, SA helps escape local minima)")
    
    print("\n" + "=" * 60)
    print("All Simulated Annealing tests completed!")
    print("=" * 60)