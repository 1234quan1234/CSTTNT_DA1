# Quick Start Guide - AI Optimization Framework

## Installation & Setup

```bash
# Navigate to project directory
cd /home/bui-anh-quan/CSTTNT_DA1

# Install dependencies
pip install -r requirements.txt

# Or use conda environment
conda env create -f environment.yml
conda activate aisearch
```

## Quick Demo

Run benchmarks faster with parallel execution:

```bash
# Quick demo with 4 parallel workers (much faster!)
python demo.py --parallel --jobs 4

# Or run specific benchmarks
python benchmark/run_rastrigin.py --config quick_convergence --jobs 4
python benchmark/run_knapsack.py --size 50 --jobs 4
```

This generates plots in `results/` folder demonstrating:
- Convergence curves on Rastrigin function
- Knapsack optimization results
- Algorithm comparisons (FA, SA, HC, GA)
- Parameter sensitivity analysis
- Swarm trajectory plots

## ⏱️ Estimated Runtime (with 4 cores)

| Benchmark | Sequential | Parallel (4 cores) |
|-----------|------------|-------------------|
| Rastrigin quick | ~5 min | ~2 min |
| Rastrigin all | ~45 min | ~15 min |
| Knapsack n=50 | ~30 min | ~10 min |
| Knapsack n=100 | ~1 hour | ~20 min |
| **Total** | ~7 hours | **~2-3 hours** |

## Testing Your Implementation

### 1. Test Individual Modules

Each module has built-in tests. Run them to verify everything works:

```bash
# Test core utilities
python src/core/utils.py

# Test Rastrigin problem
python src/problems/continuous/rastrigin.py

# Test Knapsack problem
python src/problems/discrete/knapsack.py

# Test Firefly Algorithm
python src/swarm/fa.py

# Test classical algorithms
python src/classical/hill_climbing.py
python src/classical/simulated_annealing.py
python src/classical/genetic_algorithm.py
```

### 2. Quick Examples

#### Example 1: Run FA on Rastrigin Function

```python
import sys
sys.path.append('/home/bui-anh-quan/CSTTNT_DA1')

from src.problems.continuous.rastrigin import RastriginProblem
from src.swarm.fa import FireflyContinuousOptimizer

# Setup
problem = RastriginProblem(dim=5)
optimizer = FireflyContinuousOptimizer(
    problem=problem,
    n_fireflies=20,
    alpha=0.3,      # Higher for multimodal
    beta0=1.0,
    gamma=0.5,      # Lower for global search
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

#### Example 3: Knapsack with Firefly Algorithm

```python
import sys
import numpy as np
sys.path.append('/home/bui-anh-quan/CSTTNT_DA1')

from src.problems.discrete.knapsack import KnapsackProblem
from src.swarm.fa import FireflyKnapsackOptimizer
from src.classical.genetic_algorithm import GeneticAlgorithmOptimizer

# Create Knapsack instance
rng = np.random.RandomState(123)
n_items = 30
values = rng.randint(10, 100, n_items)
weights = rng.randint(1, 50, n_items)
capacity = int(0.5 * np.sum(weights))

problem = KnapsackProblem(values, weights, capacity)

# Firefly Algorithm
fa = FireflyKnapsackOptimizer(problem, n_fireflies=25, seed=42)
_, fa_value, fa_hist, _ = fa.run(max_iter=100)

# Genetic Algorithm
ga = GeneticAlgorithmOptimizer(problem, pop_size=30, seed=42)
_, ga_value, ga_hist, _ = ga.run(max_iter=100)

print("Knapsack Results (30 items):")
print(f"FA: {-fa_value:.2f}")  # Negate for actual value
print(f"GA: {-ga_value:.2f}")
```

## Creating Visualizations

The framework includes ready-to-use visualization utilities:

```python
import sys
sys.path.append('/home/bui-anh-quan/CSTTNT_DA1')

from src.problems.continuous.rastrigin import RastriginProblem
from src.swarm.fa import FireflyContinuousOptimizer
from src.utils.visualization import plot_convergence, plot_comparison

# Run optimization
problem = RastriginProblem(dim=5)
optimizer = FireflyContinuousOptimizer(problem, n_fireflies=20, seed=42)
best_sol, best_fit, history, trajectory = optimizer.run(max_iter=100)

# Plot convergence
plot_convergence(
    history,
    title="FA Convergence on Rastrigin",
    save_path="my_convergence.png",
    show=True
)

# Compare multiple algorithms
from src.classical.simulated_annealing import SimulatedAnnealingOptimizer

sa = SimulatedAnnealingOptimizer(problem, seed=42)
_, _, sa_hist, _ = sa.run(max_iter=100)

plot_comparison(
    {'FA': history, 'SA': sa_hist},
    title="FA vs SA",
    save_path="comparison.png",
    show=True
)
```

### Available Visualization Functions

From `src.utils.visualization`:

- `plot_convergence()` - Single algorithm convergence curve
- `plot_comparison()` - Multiple algorithms comparison (linear/log scale)
- `plot_trajectory_2d()` - Swarm movement on 2D landscape
- `plot_parameter_sensitivity()` - Parameter tuning results

## Interactive Notebooks

For interactive exploration, use Jupyter notebooks:

```bash
# Start JupyterLab
jupyter lab

# Open notebook
# notebooks/fa_visualization.ipynb
```

The notebook includes:
- ✓ FA on 2D Rastrigin with contour plots
- ✓ FA on multimodal landscapes
- ✓ Algorithm comparison charts
- ✓ Parameter sensitivity analysis
- ✓ Optional: animated swarm movement

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
- **Multimodal (Rastrigin)**: gamma=0.5, alpha=0.3, n_fireflies=30

### Firefly Algorithm (Knapsack)

- **n_fireflies** (20-50): Population size
- **alpha_flip** (0.1-0.4): Random bit flip probability
  - Lower: More exploitation
  - Higher: More exploration
- **max_flips_per_move** (2-5): Directed flips per movement
  - Controls adaptation speed to better solutions

**Recommended:** alpha_flip=0.2, max_flips_per_move=3

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

1. **Run the demo**: `python demo.py` to see all features
2. **Create visualizations**: Use `history` and `trajectory` to plot convergence
3. **Run benchmarks**: Compare algorithms across multiple runs
4. **Tune parameters**: Experiment with different parameter settings
5. **Extend the framework**: Add custom problems or algorithms

## File Structure Summary

```
src/
├── core/                   # Base classes
│   ├── base_optimizer.py   # All optimizers inherit from BaseOptimizer
│   ├── problem_base.py     # All problems inherit from ProblemBase
│   └── utils.py            # Helper functions (distances, brightness, etc.)
│
├── problems/
│   ├── continuous/         # Continuous benchmark functions
│   │   └── rastrigin.py    # Rastrigin function (multimodal)
│   └── discrete/           # Discrete optimization problems
│       └── knapsack.py     # 0/1 Knapsack problem
│
├── swarm/
│   └── fa.py              # Firefly Algorithm (continuous & Knapsack)
│
├── classical/
│   ├── hill_climbing.py
│   ├── simulated_annealing.py
│   └── genetic_algorithm.py
│
└── utils/
    └── visualization.py    # Plotting utilities

test/                       # Unit tests
notebooks/                  # Interactive demos
results/                    # Generated plots (auto-created)
demo.py                     # Comprehensive demo script
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
- `representation_type()`: Returns problem type ("continuous" or "knapsack")

This allows any optimizer to work with any compatible problem!
- `representation_type()`: Returns problem type ("continuous", "tsp", etc.)

This allows any optimizer to work with any compatible problem!
