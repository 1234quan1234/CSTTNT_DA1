# Quick Start Guide - AI Optimization Framework

## Installation & Setup

```bash
# Navigate to project directory
cd /home/bui-anh-quan/CSTTNT_DA1

# Install dependencies
pip install -r requirements.txt
```

## Testing Your Implementation

### 1. Test Individual Modules

Each module has built-in tests. Run them to verify everything works:

```bash
# Test core utilities
python src/core/utils.py

# Test continuous problems
python src/problems/continuous/sphere.py
python src/problems/continuous/rosenbrock.py
python src/problems/continuous/rastrigin.py
python src/problems/continuous/ackley.py

# Test discrete problems
python src/problems/discrete/tsp.py
python src/problems/discrete/knapsack.py
python src/problems/discrete/graph_coloring.py

# Test Firefly Algorithm
python src/swarm/fa.py

# Test classical algorithms
python src/classical/hill_climbing.py
python src/classical/simulated_annealing.py
python src/classical/genetic_algorithm.py
python src/classical/graph_search.py
```

### 2. Quick Examples

#### Example 1: Run FA on Sphere Function

```python
import sys
sys.path.append('/home/bui-anh-quan/CSTTNT_DA1')

from src.problems.continuous.sphere import SphereProblem
from src.swarm.fa import FireflyContinuousOptimizer

# Setup
problem = SphereProblem(dim=5)
optimizer = FireflyContinuousOptimizer(
    problem=problem,
    n_fireflies=20,
    alpha=0.2,
    beta0=1.0,
    gamma=1.0,
    seed=42
)

# Run
best_sol, best_fit, history, trajectory = optimizer.run(max_iter=50)

# Results
print(f"Best fitness: {best_fit:.6f}")
print(f"Convergence: {history[0]:.4f} -> {history[-1]:.4f}")
```

#### Example 2: Compare Algorithms on Rastrigin

```python
import sys
sys.path.append('/home/bui-anh-quan/CSTTNT_DA1')

from src.problems.continuous.rastrigin import RastriginProblem
from src.swarm.fa import FireflyContinuousOptimizer
from src.classical.simulated_annealing import SimulatedAnnealingOptimizer
from src.classical.hill_climbing import HillClimbingOptimizer

problem = RastriginProblem(dim=5)

# Firefly Algorithm
fa = FireflyContinuousOptimizer(problem, n_fireflies=20, seed=42)
_, fa_fit, fa_hist, _ = fa.run(max_iter=100)

# Simulated Annealing
sa = SimulatedAnnealingOptimizer(problem, initial_temp=100, seed=42)
_, sa_fit, sa_hist, _ = sa.run(max_iter=100)

# Hill Climbing
hc = HillClimbingOptimizer(problem, num_neighbors=20, seed=42)
_, hc_fit, hc_hist, _ = hc.run(max_iter=100)

print("Algorithm Comparison on Rastrigin Function:")
print(f"FA:  {fa_fit:.6f} (improvement: {fa_hist[0] - fa_hist[-1]:.4f})")
print(f"SA:  {sa_fit:.6f} (improvement: {sa_hist[0] - sa_hist[-1]:.4f})")
print(f"HC:  {hc_fit:.6f} (improvement: {hc_hist[0] - hc_hist[-1]:.4f})")
```

#### Example 3: TSP with Multiple Algorithms

```python
import sys
import numpy as np
sys.path.append('/home/bui-anh-quan/CSTTNT_DA1')

from src.problems.discrete.tsp import TSPProblem
from src.swarm.fa import FireflyDiscreteTSPOptimizer
from src.classical.simulated_annealing import SimulatedAnnealingOptimizer
from src.classical.genetic_algorithm import GeneticAlgorithmOptimizer

# Create TSP instance
rng = np.random.RandomState(123)
coords = rng.rand(15, 2) * 100  # 15 cities
problem = TSPProblem(coords)

# Discrete Firefly Algorithm
fa = FireflyDiscreteTSPOptimizer(problem, n_fireflies=20, seed=42)
_, fa_length, fa_hist, _ = fa.run(max_iter=100)

# Simulated Annealing
sa = SimulatedAnnealingOptimizer(problem, initial_temp=50, seed=42)
_, sa_length, sa_hist, _ = sa.run(max_iter=200)

# Genetic Algorithm
ga = GeneticAlgorithmOptimizer(problem, pop_size=30, seed=42)
_, ga_length, ga_hist, _ = ga.run(max_iter=100)

print("TSP Results (15 cities):")
print(f"Discrete FA: {fa_length:.4f}")
print(f"SA:          {sa_length:.4f}")
print(f"GA:          {ga_length:.4f}")
```

## Parameter Tuning Guide

### Firefly Algorithm (Continuous)

- **n_fireflies** (10-50): Population size
  - Smaller: Faster, may converge prematurely
  - Larger: Better exploration, slower

- **alpha** (0.1-0.5): Randomization
  - Smaller: More exploitation (local search)
  - Larger: More exploration (avoid local minima)

- **gamma** (0.1-2.0): Light absorption
  - Smaller: More global search (long-range attraction)
  - Larger: More local search (short-range attraction)

- **beta0** (0.5-2.0): Base attractiveness
  - Higher: Stronger attraction between fireflies

**Recommended for different problems:**
- **Unimodal (Sphere, Rosenbrock)**: gamma=1.0, alpha=0.2
- **Multimodal (Rastrigin, Ackley)**: gamma=0.5, alpha=0.3

### Simulated Annealing

- **initial_temp** (10-200): Starting temperature
- **cooling_rate** (0.90-0.99): How fast temperature decreases
  - Higher (0.99): Slower cooling, more exploration
  - Lower (0.90): Faster cooling, quicker convergence

### Genetic Algorithm

- **pop_size** (20-100): Population size
- **crossover_rate** (0.6-0.9): Probability of crossover
- **mutation_rate** (0.05-0.2): Probability of mutation
- **elitism** (1-5): Number of best individuals to preserve

## Common Issues & Solutions

### Issue 1: Import errors
```python
# Solution: Add src to path
import sys
sys.path.append('/home/bui-anh-quan/CSTTNT_DA1')
```

### Issue 2: NumPy not found
```bash
# Solution: Install numpy
pip install numpy
```

### Issue 3: Poor convergence on multimodal functions
```python
# Solution: Adjust FA parameters
optimizer = FireflyContinuousOptimizer(
    problem=problem,
    n_fireflies=30,      # Increase population
    alpha=0.3,           # Increase randomization
    gamma=0.5,           # Decrease gamma for global search
    seed=42
)
```

## Next Steps

1. **Create visualizations**: Use `history_best` and `trajectory` to plot convergence
2. **Run benchmarks**: Compare algorithms across multiple runs
3. **Tune parameters**: Experiment with different parameter settings
4. **Add new problems**: Extend the framework with custom problems
5. **Add new algorithms**: Implement PSO, ACO, ABC, etc.

## File Structure Summary

```
src/
├── core/                   # Base classes
│   ├── base_optimizer.py   # All optimizers inherit from BaseOptimizer
│   ├── problem_base.py     # All problems inherit from ProblemBase
│   └── utils.py            # Helper functions
│
├── problems/
│   ├── continuous/         # Continuous benchmark functions
│   └── discrete/           # Discrete optimization problems
│
├── swarm/
│   └── fa.py              # Firefly Algorithm (continuous & TSP)
│
└── classical/
    ├── hill_climbing.py
    ├── simulated_annealing.py
    ├── genetic_algorithm.py
    └── graph_search.py     # BFS, DFS, A*
```

## Key Concepts

### All algorithms return the same format:
```python
best_solution, best_fitness, history_best, trajectory = optimizer.run(max_iter)
```

- `best_solution`: Best solution found
- `best_fitness`: Best objective value (lower is better - minimization)
- `history_best`: List of best fitness per iteration (for plotting convergence)
- `trajectory`: List of populations/solutions per iteration (for animation)

### All problems implement:
- `evaluate(x)`: Returns fitness value (minimize)
- `init_solution(rng, n)`: Generates n random solutions
- `clip(X)`: Ensures solutions are within valid bounds
- `representation_type()`: Returns problem type ("continuous", "tsp", etc.)

This allows any optimizer to work with any compatible problem!
