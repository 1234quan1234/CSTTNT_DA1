"""
Genetic Algorithm (GA) optimization.

Genetic Algorithm is a population-based metaheuristic inspired by natural evolution,
using selection, crossover, and mutation to evolve solutions over generations.

References
----------
.. [1] https://en.wikipedia.org/wiki/Genetic_algorithm
"""

import numpy as np
from typing import List, Tuple
import logging

from ..core.base_optimizer import BaseOptimizer
from ..core.problem_base import ProblemBase
from ..core.utils import get_best_solution

logger = logging.getLogger(__name__)

class GeneticAlgorithmOptimizer(BaseOptimizer):
    """
    Genetic Algorithm optimizer.
    
    Supports discrete problems (TSP, Knapsack, Graph Coloring) and can be
    extended to continuous problems. Uses:
    - Selection: Tournament or fitness-proportionate
    - Crossover: Problem-specific (PMX for TSP, 1-point for binary)
    - Mutation: Problem-specific (swap for TSP, bit flip for binary)
    
    Parameters
    ----------
    problem : ProblemBase
        The optimization problem.
    pop_size : int, default=50
        Population size.
    crossover_rate : float, default=0.8
        Probability of applying crossover.
    mutation_rate : float, default=0.1
        Probability of applying mutation.
    selection_type : str, default="tournament"
        Selection method: "tournament" or "roulette".
    tournament_size : int, default=3
        Size of tournament for tournament selection.
    elitism : int, default=1
        Number of best individuals to preserve across generations.
    seed : int or None, default=None
        Random seed for reproducibility.
    
    Attributes
    ----------
    problem : ProblemBase
        The optimization problem.
    pop_size : int
        Population size.
    crossover_rate, mutation_rate : float
        GA operator probabilities.
    selection_type : str
        Selection method.
    tournament_size : int
        Tournament size.
    elitism : int
        Number of elite individuals.
    rng : np.random.RandomState
        Random number generator.
    population : np.ndarray
        Current population.
    fitness : np.ndarray
        Fitness values.
    
    Examples
    --------
    >>> from problems.discrete.tsp import TSPProblem
    >>> coords = np.random.rand(10, 2)
    >>> problem = TSPProblem(coords)
    >>> ga = GeneticAlgorithmOptimizer(problem, pop_size=30, seed=42)
    >>> best_sol, best_fit, history, trajectory = ga.run(max_iter=50)
    """
    
    def __init__(
        self,
        problem: ProblemBase,
        pop_size: int = 50,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elitism: int | bool = True,
        tournament_size: int = 3,
        selection_type: str = "tournament",
        repair_method: str = None,
        seed: int = None
    ):
        """Initialize Genetic Algorithm optimizer."""
        # Validate selection_type
        if selection_type not in ["tournament", "roulette"]:
            raise ValueError(f"selection_type must be 'tournament' or 'roulette', got '{selection_type}'")
        
        self.problem = problem
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        
        # Handle elitism (bool or int)
        if isinstance(elitism, bool):
            self.elitism = 1 if elitism else 0
            logger.debug(f"GA: Converted elitism {elitism} → {self.elitism}")
        else:
            self.elitism = int(elitism)
        
        self.tournament_size = tournament_size
        self.selection_type = selection_type
        self.repair_method = repair_method
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        self.population = None
        self.fitness = None
    
    def _init_population(self):
        """Initialize population randomly."""
        self.population = self.problem.init_solution(self.rng, self.pop_size)
        self.fitness = np.array([self.problem.evaluate(ind) for ind in self.population])
    
    def _tournament_selection(self) -> int:
        """Select an individual using tournament selection."""
        tournament_indices = self.rng.choice(self.pop_size, self.tournament_size, replace=False)
        tournament_fitness = self.fitness[tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return int(winner_idx)
    
    def _roulette_selection(self) -> int:
        """Select an individual using fitness-proportionate (roulette wheel) selection."""
        # Convert fitness to selection probabilities (lower fitness = higher probability)
        # Use rank-based selection to avoid issues with negative fitness
        ranks = np.argsort(np.argsort(self.fitness))  # 0 = best, pop_size-1 = worst
        weights = 1.0 / (ranks + 1)  # Higher weight for better (lower rank)
        probabilities = weights / np.sum(weights)
        
        selected_idx = self.rng.choice(self.pop_size, p=probabilities)
        return int(selected_idx)
    
    def _select_parent(self) -> int:
        """Select a parent based on selection type."""
        if self.selection_type == "tournament":
            return self._tournament_selection()
        elif self.selection_type == "roulette":
            return self._roulette_selection()
        else:
            raise ValueError(f"Unknown selection type: {self.selection_type}")
    
    # ========== Crossover Operators ==========
    
    def _crossover_binary_onepoint(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """One-point crossover for binary strings (Knapsack)."""
        n = len(parent1)
        cx_point = self.rng.randint(1, n)
        
        offspring1 = np.concatenate([parent1[:cx_point], parent2[cx_point:]])
        offspring2 = np.concatenate([parent2[:cx_point], parent1[cx_point:]])
        
        return offspring1, offspring2
    
    def _crossover_discrete(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Discrete crossover (one-point for Knapsack) with repair."""
        point = self.rng.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        
        child1 = self._repair_knapsack(child1)
        child2 = self._repair_knapsack(child2)
        
        return child1, child2
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply crossover based on problem type."""
        repr_type = self.problem.representation_type()
        
        if repr_type == "knapsack":
            return self._crossover_discrete(parent1, parent2)
        elif repr_type == "continuous":
            alpha = self.rng.rand()
            offspring1 = alpha * parent1 + (1 - alpha) * parent2
            offspring2 = (1 - alpha) * parent1 + alpha * parent2
            return self.problem.clip(offspring1), self.problem.clip(offspring2)
        else:
            raise NotImplementedError(f"Crossover not implemented for {repr_type}")
    
    # ========== Mutation Operators ==========
    
    def _mutate_knapsack(self, individual: np.ndarray) -> np.ndarray:
        """Mutation for Knapsack: per-gene bit flip with repair."""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if self.rng.rand() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]
        return self._repair_knapsack(mutated)
    
    def _mutate_continuous(self, individual: np.ndarray) -> np.ndarray:
        """Mutation for continuous: add Gaussian noise."""
        mutated = individual + self.rng.randn(len(individual)) * 0.1
        return self.problem.clip(mutated)
    
    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Apply mutation based on problem type."""
        repr_type = self.problem.representation_type()
        
        if repr_type == "knapsack":
            return self._mutate_knapsack(individual)
        elif repr_type == "continuous":
            return self.mutate_continuous(individual)
        else:
            raise NotImplementedError(f"Mutation not implemented for {repr_type}")
    
    def run(self, max_iter: int) -> Tuple[np.ndarray, float, List[float], List[np.ndarray]]:
        """
        Run Genetic Algorithm for max_iter generations.
        
        Parameters
        ----------
        max_iter : int
            Maximum number of generations.
        
        Returns
        -------
        best_solution : np.ndarray
            Best solution found.
        best_fitness : float
            Best fitness value (minimum).
        history_best : List[float]
            Best fitness at each generation.
        trajectory : List[np.ndarray]
            Population at each generation, shape (pop_size, problem_size).
        """
        # Initialize
        self._init_population()
        
        history_best = []
        trajectory = []
        
        for generation in range(max_iter):
            # Sort population by fitness for elitism
            sorted_indices = np.argsort(self.fitness)
            
            # Create new population
            new_population = []
            
            # Elitism: keep best individuals
            for i in range(self.elitism):
                new_population.append(self.population[sorted_indices[i]].copy())
            
            # Generate offspring to fill the rest
            while len(new_population) < self.pop_size:
                # Select parents
                parent1_idx = self._select_parent()
                parent2_idx = self._select_parent()
                
                parent1 = self.population[parent1_idx]
                parent2 = self.population[parent2_idx]
                
                # Crossover
                if self.rng.rand() < self.crossover_rate:
                    offspring1, offspring2 = self._crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1.copy(), parent2.copy()
                
                # Mutation
                if self.rng.rand() < self.mutation_rate:
                    offspring1 = self._mutate(offspring1)
                if self.rng.rand() < self.mutation_rate:
                    offspring2 = self._mutate(offspring2)
                
                new_population.append(offspring1)
                if len(new_population) < self.pop_size:
                    new_population.append(offspring2)
            
            # Update population
            self.population = np.array(new_population[:self.pop_size])
            self.fitness = np.array([self.problem.evaluate(ind) for ind in self.population])
            
            # Track best
            best_sol, best_fit = get_best_solution(self.population, self.fitness)
            history_best.append(best_fit)
            trajectory.append(self.population.copy())
        
        # Final best solution
        best_solution, best_fitness = get_best_solution(self.population, self.fitness)
        
        return best_solution, best_fitness, history_best, trajectory


if __name__ == "__main__":
    print("=" * 60)
    print("GENETIC ALGORITHM OPTIMIZER DEMO")
    print("=" * 60)
    
    # Test on Knapsack
    print("\n1. Genetic Algorithm on Knapsack Problem")
    print("-" * 60)
    from problems.discrete.knapsack import KnapsackProblem
    
    values = np.array([10, 20, 30, 40, 50, 60])
    weights = np.array([1, 2, 3, 4, 5, 6])
    capacity = 10.0
    problem_knapsack = KnapsackProblem(values, weights, capacity)
    
    ga_knapsack = GeneticAlgorithmOptimizer(
        problem=problem_knapsack,
        pop_size=20,
        crossover_rate=0.7,
        mutation_rate=0.15,
        seed=42
    )
    
    best_sel, best_fit, history_k, _ = ga_knapsack.run(max_iter=40)
    
    print(f"Initial best fitness: {history_k[0]:.2f}")
    print(f"Final best fitness: {history_k[-1]:.2f}")
    print(f"Best selection: {best_sel}")
    total_weight = np.sum(best_sel * weights)
    total_value = np.sum(best_sel * values)
    print(f"Total weight: {total_weight}, Total value: {total_value}")
    print(f"Feasible: {total_weight <= capacity}")
    
    # Test on Rastrigin
    print("\n2. Genetic Algorithm on Rastrigin Function (2D)")
    print("-" * 60)
    from problems.continuous.rastrigin import RastriginProblem
    
    problem_rastrigin = RastriginProblem(dim=2)
    ga_rastrigin = GeneticAlgorithmOptimizer(
        problem=problem_rastrigin,
        pop_size=20,
        crossover_rate=0.8,
        mutation_rate=0.15,
        seed=42
    )
    
    best_sol, best_fit, history_s, _ = ga_rastrigin.run(max_iter=50)
    
    print(f"Initial best fitness: {history_s[0]:.6f}")
    print(f"Final best fitness: {history_s[-1]:.6f}")
    print(f"Best solution: {best_sol}")
    print(f"Improvement: {history_s[0] - history_s[-1]:.6f}")
    
    print("\n" + "=" * 60)
    print("All Genetic Algorithm tests completed!")
    print("=" * 60)
    print("=" * 60)
    values = np.array([10, 20, 30, 40, 50, 60])
    weights = np.array([1, 2, 3, 4, 5, 6])
    capacity = 10.0
    problem_knapsack = KnapsackProblem(values, weights, capacity)
    
    ga_knapsack = GeneticAlgorithmOptimizer(
        problem=problem_knapsack,
        pop_size=20,
        crossover_rate=0.7,
        mutation_rate=0.15,
        seed=42
    )
    
    best_sel, best_fit, history_k, _ = ga_knapsack.run(max_iter=40)
    
    print(f"Initial best fitness: {history_k[0]:.2f}")
    print(f"Final best fitness: {history_k[-1]:.2f}")
    print(f"Best selection: {best_sel}")
    total_weight = np.sum(best_sel * weights)
    total_value = np.sum(best_sel * values)
    print(f"Total weight: {total_weight}, Total value: {total_value}")
    print(f"Feasible: {total_weight <= capacity}")
    
    # Test 3: Graph Coloring
    print("\n3. Genetic Algorithm on Graph Coloring")
    print("-" * 60)
    from problems.discrete.graph_coloring import GraphColoringProblem
    
    # Create a cycle graph (needs 2 colors if even, 3 if odd)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]  # 4-cycle
    problem_gc = GraphColoringProblem(num_nodes=4, edges=edges, num_colors=2)
    
    ga_gc = GeneticAlgorithmOptimizer(
        problem=problem_gc,
        pop_size=20,
        crossover_rate=0.8,
        mutation_rate=0.2,
        seed=42
    )
    
    best_coloring, best_conflicts, history_gc, _ = ga_gc.run(max_iter=30)
    
    print(f"Initial best conflicts: {history_gc[0]:.0f}")
    print(f"Final best conflicts: {history_gc[-1]:.0f}")
    print(f"Best coloring: {best_coloring}")
    print(f"Valid 2-coloring found: {history_gc[-1] == 0}")
    
    # Test 4: Continuous (Sphere)
    print("\n4. Genetic Algorithm on Sphere Function (2D)")
    print("-" * 60)
    from problems.continuous.sphere import SphereProblem
    
    problem_sphere = SphereProblem(dim=2)
    ga_sphere = GeneticAlgorithmOptimizer(
        problem=problem_sphere,
        pop_size=20,
        crossover_rate=0.8,
        mutation_rate=0.15,
        seed=42
    )
    
    best_sol, best_fit, history_s, _ = ga_sphere.run(max_iter=50)
    
    print(f"Initial best fitness: {history_s[0]:.6f}")
    print(f"Final best fitness: {history_s[-1]:.6f}")
    print(f"Best solution: {best_sol}")
    print(f"Improvement: {history_s[0] - history_s[-1]:.6f}")
    
    print("\n" + "=" * 60)
    print("All Genetic Algorithm tests completed!")
    print("=" * 60)
    print("=" * 60)
    print(f"Improvement: {history_s[0] - history_s[-1]:.6f}")
    
    print("\n" + "=" * 60)
    print("All Genetic Algorithm tests completed!")
    print("=" * 60)
    print("=" * 60)
    values = np.array([10, 20, 30, 40, 50, 60])
    weights = np.array([1, 2, 3, 4, 5, 6])
    capacity = 10.0
    problem_knapsack = KnapsackProblem(values, weights, capacity)
    
    ga_knapsack = GeneticAlgorithmOptimizer(
        problem=problem_knapsack,
        pop_size=20,
        crossover_rate=0.7,
        mutation_rate=0.15,
        seed=42
    )
    
    best_sel, best_fit, history_k, _ = ga_knapsack.run(max_iter=40)
    
    print(f"Initial best fitness: {history_k[0]:.2f}")
    print(f"Final best fitness: {history_k[-1]:.2f}")
    print(f"Best selection: {best_sel}")
    total_weight = np.sum(best_sel * weights)
    total_value = np.sum(best_sel * values)
    print(f"Total weight: {total_weight}, Total value: {total_value}")
    print(f"Feasible: {total_weight <= capacity}")
    
    # Test 3: Graph Coloring
    print("\n3. Genetic Algorithm on Graph Coloring")
    print("-" * 60)
    from problems.discrete.graph_coloring import GraphColoringProblem
    
    # Create a cycle graph (needs 2 colors if even, 3 if odd)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]  # 4-cycle
    problem_gc = GraphColoringProblem(num_nodes=4, edges=edges, num_colors=2)
    
    ga_gc = GeneticAlgorithmOptimizer(
        problem=problem_gc,
        pop_size=20,
        crossover_rate=0.8,
        mutation_rate=0.2,
        seed=42
    )
    
    best_coloring, best_conflicts, history_gc, _ = ga_gc.run(max_iter=30)
    
    print(f"Initial best conflicts: {history_gc[0]:.0f}")
    print(f"Final best conflicts: {history_gc[-1]:.0f}")
    print(f"Best coloring: {best_coloring}")
    print(f"Valid 2-coloring found: {history_gc[-1] == 0}")
    
    # Test 4: Continuous (Sphere)
    print("\n4. Genetic Algorithm on Sphere Function (2D)")
    print("-" * 60)
    from problems.continuous.sphere import SphereProblem
    
    problem_sphere = SphereProblem(dim=2)
    ga_sphere = GeneticAlgorithmOptimizer(
        problem=problem_sphere,
        pop_size=20,
        crossover_rate=0.8,
        mutation_rate=0.15,
        seed=42
    )
    
    best_sol, best_fit, history_s, _ = ga_sphere.run(max_iter=50)
    
    print(f"Initial best fitness: {history_s[0]:.6f}")
    print(f"Final best fitness: {history_s[-1]:.6f}")
    print(f"Best solution: {best_sol}")
    print(f"Improvement: {history_s[0] - history_s[-1]:.6f}")
    
    print("\n" + "=" * 60)
    print("All Genetic Algorithm tests completed!")
    print("=" * 60)
    print("=" * 60)
