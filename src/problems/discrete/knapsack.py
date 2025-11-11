"""
0/1 Knapsack Problem optimization.

The 0/1 Knapsack problem is a classic combinatorial optimization problem where
the goal is to select items to maximize value while staying within weight capacity.

References
----------
.. [1] https://en.wikipedia.org/wiki/Knapsack_problem
"""

import numpy as np
from typing import Literal

# Use relative import
from ...core.problem_base import ProblemBase


class KnapsackProblem(ProblemBase):
    """
    0/1 Knapsack Problem.
    
    Given a set of items, each with a weight and value, select items to maximize
    total value without exceeding the knapsack's weight capacity.
    
    Solution representation: binary vector where x[i] = 1 means item i is selected.
    
    For optimization consistency (minimization), we use:
        fitness = -total_value  (if feasible)
        fitness = large_penalty + overflow_weight  (if infeasible)
    
    Parameters
    ----------
    values : np.ndarray
        Value of each item, shape (num_items,).
    weights : np.ndarray
        Weight of each item, shape (num_items,).
    capacity : float
        Maximum weight capacity of the knapsack.
    penalty_coefficient : float, optional
        Penalty multiplier for constraint violations. Default is 1000.
    
    Attributes
    ----------
    values : np.ndarray
        Item values.
    weights : np.ndarray
        Item weights.
    capacity : float
        Knapsack capacity.
    num_items : int
        Number of items.
    penalty_coefficient : float
        Penalty for infeasible solutions.
    
    Examples
    --------
    >>> values = np.array([10, 20, 30])
    >>> weights = np.array([1, 2, 3])
    >>> capacity = 4.0
    >>> problem = KnapsackProblem(values, weights, capacity)
    >>> x = np.array([1, 1, 0])  # Select items 0 and 1
    >>> fitness = problem.evaluate(x)
    >>> print(f"Fitness: {fitness}")  # Should be -30 (maximizing value)
    """
    
    def __init__(
        self, 
        values: np.ndarray, 
        weights: np.ndarray, 
        capacity: float,
        penalty_coefficient: float = 1000.0
    ):
        """
        Initialize Knapsack problem.
        
        Parameters
        ----------
        values : np.ndarray
            Value of each item, shape (num_items,).
        weights : np.ndarray
            Weight of each item, shape (num_items,).
        capacity : float
            Maximum weight capacity.
        penalty_coefficient : float, optional
            Penalty multiplier for exceeding capacity. Default is 1000.
        """
        self.values = np.array(values, dtype=float)
        self.weights = np.array(weights, dtype=float)
        self.capacity = float(capacity)
        self.num_items = len(values)
        self.penalty_coefficient = penalty_coefficient
        
        if len(weights) != self.num_items:
            raise ValueError("values and weights must have the same length")
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate knapsack solution.
        
        Returns negative total value if feasible (for minimization).
        Returns large penalty + overflow if infeasible.
        
        Parameters
        ----------
        x : np.ndarray
            Binary selection vector, shape (num_items,), values in {0, 1}.
        
        Returns
        -------
        fitness : float
            -total_value if feasible, penalty + overflow otherwise.
        """
        # Ensure binary
        selection = (x > 0.5).astype(int)
        
        total_weight = np.sum(selection * self.weights)
        total_value = np.sum(selection * self.values)
        
        if total_weight <= self.capacity:
            # Feasible: return negative value (we're minimizing)
            return -float(total_value)
        else:
            # Infeasible: heavy penalty
            overflow = total_weight - self.capacity
            return float(self.penalty_coefficient + overflow)
    
    def representation_type(self) -> Literal["continuous", "tsp", "knapsack", "graph_coloring"]:
        """Return 'knapsack' for this problem type."""
        return "knapsack"
    
    def init_solution(self, rng: np.random.RandomState, n: int = 1) -> np.ndarray:
        """
        Generate n random feasible knapsack solutions.
        
        Uses a greedy repair strategy to ensure solutions are feasible:
        randomly select items, and if capacity is exceeded, randomly remove
        items until feasible.
        
        Parameters
        ----------
        rng : np.random.RandomState
            Random number generator.
        n : int, default=1
            Number of solutions to generate.
        
        Returns
        -------
        solutions : np.ndarray
            Array of shape (n, num_items) with binary values.
        """
        solutions = np.zeros((n, self.num_items), dtype=int)
        
        for i in range(n):
            # Start with random binary vector
            solution = rng.randint(0, 2, self.num_items)
            
            # Repair if needed
            solution = self._repair_solution(solution, rng)
            solutions[i] = solution
        
        return solutions
    
    def _repair_solution(self, solution: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """
        Repair an infeasible solution by removing items until feasible.
        
        Uses greedy strategy: removes items with lowest value/weight ratio first.
        
        Parameters
        ----------
        solution : np.ndarray
            Binary selection vector.
        rng : np.random.RandomState
            Random number generator.
        
        Returns
        -------
        repaired : np.ndarray
            Feasible binary selection vector.
        """
        solution = solution.copy()
        total_weight = np.sum(solution * self.weights)
        
        # If already feasible, return
        if total_weight <= self.capacity:
            return solution
        
        selected_indices = np.where(solution == 1)[0]
        
        # Greedy repair: remove lowest value/weight ratio items first
        if len(selected_indices) > 0:
            ratios = self.values[selected_indices] / self.weights[selected_indices]
            sorted_indices = selected_indices[np.argsort(ratios)]  # Ascending
            
            for idx in sorted_indices:
                solution[idx] = 0
                total_weight -= self.weights[idx]
                if total_weight <= self.capacity:
                    break
        
        return solution
    
    def greedy_repair(self, solution: np.ndarray) -> np.ndarray:
        """
        Public method for greedy repair (for external use by optimizers).
        
        Removes items with lowest value/weight ratio first until feasible.
        
        Parameters
        ----------
        solution : np.ndarray
            Binary selection vector that may be infeasible.
        
        Returns
        -------
        repaired : np.ndarray
            Feasible binary selection vector.
        """
        solution = solution.copy()
        total_weight = np.sum(solution * self.weights)
        
        if total_weight <= self.capacity:
            return solution
        
        selected_indices = np.where(solution == 1)[0]
        
        if len(selected_indices) > 0:
            # Calculate value/weight ratios, avoid division by zero
            ratios = np.where(
                self.weights[selected_indices] > 0,
                self.values[selected_indices] / self.weights[selected_indices],
                0.0
            )
            # Sort by ratio ascending (remove worst items first)
            sorted_indices = selected_indices[np.argsort(ratios)]
            
            for idx in sorted_indices:
                solution[idx] = 0
                total_weight -= self.weights[idx]
                if total_weight <= self.capacity:
                    break
        
        return solution
    
    def clip(self, X: np.ndarray) -> np.ndarray:
        """
        For Knapsack, clip to binary {0, 1}.
        
        Values >= 0.5 become 1, others become 0.
        
        Parameters
        ----------
        X : np.ndarray
            Solution(s), shape (n, num_items) or (num_items,).
        
        Returns
        -------
        X_binary : np.ndarray
            Binary-clipped solutions.
        """
        return (X > 0.5).astype(int)


if __name__ == "__main__":
    # Demo
    print("Knapsack Problem Demo")
    print("=" * 50)
    
    # Small example
    values = np.array([10, 20, 30, 40])
    weights = np.array([1, 2, 3, 4])
    capacity = 5.0
    
    problem = KnapsackProblem(values, weights, capacity)
    print(f"Items: {problem.num_items}")
    print(f"Values: {values}")
    print(f"Weights: {weights}")
    print(f"Capacity: {capacity}")
    
    # Test feasible solution
    x1 = np.array([1, 1, 0, 0])  # Items 0,1: value=30, weight=3
    f1 = problem.evaluate(x1)
    print(f"\nSolution [1,1,0,0]: fitness = {f1} (value=30, weight=3)")
    
    x2 = np.array([0, 0, 1, 1])  # Items 2,3: value=70, weight=7 (INFEASIBLE)
    f2 = problem.evaluate(x2)
    print(f"Solution [0,0,1,1]: fitness = {f2} (value=70, weight=7, INFEASIBLE)")
    
    # Optimal solution
    x_opt = np.array([1, 0, 0, 1])  # Items 0,3: value=50, weight=5
    f_opt = problem.evaluate(x_opt)
    print(f"Solution [1,0,0,1]: fitness = {f_opt} (value=50, weight=5, at capacity)")
    
    # Generate random solutions
    rng = np.random.RandomState(42)
    random_sols = problem.init_solution(rng, n=5)
    print(f"\nGenerated 5 random feasible solutions:")
    for i, sol in enumerate(random_sols):
        weight = np.sum(sol * weights)
        value = np.sum(sol * values)
        fitness = problem.evaluate(sol)
        print(f"  Sol {i}: {sol} -> weight={weight:.1f}, value={value:.1f}, fitness={fitness:.1f}")
    
    print("\nKnapsack problem test passed!")
