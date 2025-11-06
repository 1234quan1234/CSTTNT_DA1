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
from typing import List, Tuple

# Use relative imports
from ..core.base_optimizer import BaseOptimizer
from ..core.problem_base import ProblemBase


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
        seed: int = None
    ):
        """Initialize Simulated Annealing optimizer."""
        self.problem = problem
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.step_size = step_size
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
    
    def _acceptance_probability(self, delta_e: float, temperature: float) -> float:
        """
        Compute acceptance probability for a worse solution.
        
        Parameters
        ----------
        delta_e : float
            Change in fitness (new_fitness - current_fitness).
        temperature : float
            Current temperature.
        
        Returns
        -------
        probability : float
            Acceptance probability.
        """
        if delta_e < 0:
            # Better solution, always accept
            return 1.0
        else:
            # Worse solution, accept with probability exp(-ΔE/T)
            return np.exp(-delta_e / temperature)
    
    def run(self, max_iter: int) -> Tuple[np.ndarray, float, List[float], List[np.ndarray]]:
        """
        Run Simulated Annealing for max_iter iterations.
        
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
            Current solution at each iteration, shape (1, problem_size).
        """
        # Initialize with random solution
        self.current_solution = self.problem.init_solution(self.rng, n=1)[0]
        self.current_fitness = self.problem.evaluate(self.current_solution)
        
        # Track best solution
        self.best_solution = self.current_solution.copy()
        self.best_fitness = self.current_fitness
        
        history_best = []
        trajectory = []
        
        for iteration in range(max_iter):
            # Calculate current temperature
            temperature = self.initial_temp * (self.cooling_rate ** iteration)
            
            # Generate neighbor
            neighbor = self._generate_neighbor()
            neighbor_fitness = self.problem.evaluate(neighbor)
            
            # Compute fitness change
            delta_e = neighbor_fitness - self.current_fitness
            
            # Accept or reject
            accept_prob = self._acceptance_probability(delta_e, temperature)
            
            if self.rng.rand() < accept_prob:
                # Accept the neighbor
                self.current_solution = neighbor
                self.current_fitness = neighbor_fitness
                
                # Update best if needed
                if self.current_fitness < self.best_fitness:
                    self.best_solution = self.current_solution.copy()
                    self.best_fitness = self.current_fitness
            
            # Track progress (always track best found so far)
            history_best.append(self.best_fitness)
            trajectory.append(self.current_solution.reshape(1, -1).copy())
        
        return (
            self.best_solution.copy(),
            self.best_fitness,
            history_best,
            trajectory
        )


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