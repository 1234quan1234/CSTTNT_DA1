# AI Search and Optimization Project

A comprehensive Python framework for comparing **Firefly Algorithm** (FA) with classical optimization methods on continuous and discrete benchmark problems.

## 🎯 Project Overview

This project implements and benchmarks multiple optimization algorithms:

### Swarm Intelligence
- **Firefly Algorithm (FA)** - Continuous optimization
- **Firefly Algorithm (FA)** - Discrete Knapsack variant

### Classical Baselines
- **Hill Climbing** - Greedy local search
- **Simulated Annealing** - Probabilistic local search with temperature scheduling
- **Genetic Algorithm** - Evolutionary optimization

### Benchmark Problems

#### Continuous Functions
- **Rastrigin** - Highly multimodal with many local minima

#### Discrete Problems
- **0/1 Knapsack** - Maximize value within capacity constraint

## 📁 Project Structure

```
CSTTNT_DA1/
├── src/
│   ├── core/                      # Base classes and utilities
│   │   ├── base_optimizer.py      # Abstract optimizer interface
│   │   ├── problem_base.py        # Abstract problem interface
│   │   └── utils.py               # Helper functions
│   │
│   ├── problems/
│   │   ├── continuous/            # Continuous benchmark functions
│   │   │   └── rastrigin.py       # Rastrigin function
│   │   └── discrete/              # Discrete optimization problems
│   │       └── knapsack.py        # 0/1 Knapsack problem
│   │
│   ├── swarm/                     # Swarm intelligence algorithms
│   │   └── fa.py                  # Firefly Algorithm (continuous & Knapsack)
│   │
│   ├── classical/                 # Classical baseline algorithms
│   │   ├── hill_climbing.py
│   │   ├── simulated_annealing.py
│   │   └── genetic_algorithm.py
│   │
│   └── utils/                     # Utility modules
│       └── visualization.py       # Plotting and visualization functions
│
├── test/                          # Unit tests
│   ├── test_continuous_problems.py
│   ├── test_knapsack_problem.py
│   ├── test_firefly_algorithm.py
│   ├── test_classical_algorithms.py
│   ├── run_all_tests.py
│   └── README.md
│
├── notebooks/                     # Jupyter notebooks
│   └── fa_visualization.ipynb     # Interactive visualization demos
│
├── benchmark/                     # Benchmark suite
│   ├── config.py
│   ├── run_rastrigin.py
│   ├── run_knapsack.py
│   ├── analyze_results.py
│   ├── visualize.py
│   └── README.md
│
├── results/                       # Output directory (auto-created)
│   ├── *.png                      # Generated plots
│   └── *.csv                      # Benchmark results
│
├── demo.py                        # Comprehensive demo script
├── environment.yml                # Conda environment
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── QUICKSTART.md                  # Quick start guide
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- NumPy
- Matplotlib (for visualization)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/1234quan1234/CSTTNT_DA1.git
cd CSTTNT_DA1
```

2. Install dependencies using conda (recommended):
```bash
conda env create -f environment.yml
conda activate aisearch
```

Or using pip:
```bash
pip install -r requirements.txt
```

### Quick Test

Run benchmarks with parallel execution (faster):

```bash
# Use all available CPU cores
python benchmark/run_rastrigin.py --config quick_convergence --jobs -1

# Use 4 parallel workers
python benchmark/run_rastrigin.py --config quick_convergence --jobs 4

# Run all benchmarks in parallel
python benchmark/run_all.py --quick --jobs 4
```

This will:
- Run FA on Rastrigin function (continuous multimodal)
- Run FA on Knapsack problem (discrete)
- Compare FA with SA, HC, and GA
- Perform parameter sensitivity analysis
- Generate visualization plots in `results/` folder

Or test individual modules:

```bash
# Test Firefly Algorithm
python src/swarm/fa.py

# Test problems
python src/problems/continuous/rastrigin.py
python src/problems/discrete/knapsack.py

# Run all unit tests
python test/run_all_tests.py
```

## 💡 Usage Examples

### Example 1: Rastrigin Function with Firefly Algorithm

```python
import numpy as np
from src.problems.continuous.rastrigin import RastriginProblem
from src.swarm.fa import FireflyContinuousOptimizer

# Create problem
problem = RastriginProblem(dim=5)

# Create optimizer
optimizer = FireflyContinuousOptimizer(
    problem=problem,
    n_fireflies=20,
    alpha=0.3,      # Higher for multimodal problems
    beta0=1.0,      # Base attractiveness
    gamma=0.5,      # Lower for global search
    seed=42
)

# Run optimization
best_solution, best_fitness, history, trajectory = optimizer.run(max_iter=50)

print(f"Best fitness: {best_fitness:.6f}")
print(f"Convergence: {history[0]:.4f} -> {history[-1]:.4f}")
```

### Example 2: Knapsack with Discrete Firefly Algorithm

```python
import numpy as np
from src.problems.discrete.knapsack import KnapsackProblem
from src.swarm.fa import FireflyKnapsackOptimizer

# Create Knapsack instance
rng = np.random.RandomState(42)
n_items = 30
values = rng.randint(10, 100, n_items)
weights = rng.randint(1, 50, n_items)
capacity = int(0.5 * np.sum(weights))

problem = KnapsackProblem(values, weights, capacity)

# Create optimizer
optimizer = FireflyKnapsackOptimizer(
    problem=problem,
    n_fireflies=25,
    alpha_flip=0.2,
    max_flips_per_move=3,
    repair_method="greedy_remove",
    seed=42
)

# Run optimization
best_sol, best_fit, history, trajectory = optimizer.run(max_iter=100)

total_value = -best_fit  # Negate for actual value
total_weight = np.sum(best_sol * weights)

print(f"Best value: {total_value:.2f}")
print(f"Weight: {total_weight:.2f}/{capacity}")
print(f"Items selected: {np.sum(best_sol)}/{n_items}")
```

### Example 3: Compare Multiple Algorithms

```python
from src.problems.continuous.rastrigin import RastriginProblem
from src.swarm.fa import FireflyContinuousOptimizer
from src.classical.simulated_annealing import SimulatedAnnealingOptimizer
from src.classical.hill_climbing import HillClimbingOptimizer
from src.classical.genetic_algorithm import GeneticAlgorithmOptimizer

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

# Genetic Algorithm
ga = GeneticAlgorithmOptimizer(problem, pop_size=20, seed=42)
_, ga_fit, ga_hist, _ = ga.run(max_iter=100)

print(f"FA: {fa_fit:.6f}")
print(f"SA: {sa_fit:.6f}")
print(f"HC: {hc_fit:.6f}")
print(f"GA: {ga_fit:.6f}")
```

## 📊 Return Format

All optimizers follow the same interface (inherited from `BaseOptimizer`):

```python
best_solution, best_fitness, history_best, trajectory = optimizer.run(max_iter)
```

**Returns:**
- `best_solution` (np.ndarray): Best solution found
  - Continuous: shape (dim,) - real values
  - Knapsack: shape (n_items,) - binary 0/1 array
- `best_fitness` (float): Best objective value (minimization)
- `history_best` (List[float]): Best fitness value at each iteration
- `trajectory` (List[np.ndarray]): Population/solution snapshots

**Reproducibility:**
All algorithms accept a `seed` parameter for reproducible results.

## 🔬 Algorithm Details

### Firefly Algorithm (Continuous)

Movement equation:
```
x_i = x_i + β₀·exp(-γ·r²)·(x_j - x_i) + α·(rand - 0.5)
```

Parameters:
- `β₀=1.0`: Attractiveness at distance r=0
- `γ=1.0`: Light absorption coefficient
- `α=0.2`: Randomization parameter
- For Rastrigin: `α=0.3, γ=0.5` (more exploration)

### Firefly Algorithm (Knapsack)

Uses bit-flip operators instead of continuous movement:
1. Identify differences between solutions
2. Apply directed bit flips toward better solutions
3. Random exploration flips (controlled by `alpha_flip`)
4. Greedy repair for constraint violations

Parameters:
- `alpha_flip=0.2`: Random bit flip probability
- `max_flips_per_move=3`: Directed flips per step

### Optimization Problems

| Problem | Type | Dimension | Global Optimum | Domain |
|---------|------|-----------|----------------|--------|
| Rastrigin | Continuous | Any | f(0,...,0) = 0 | [-5.12, 5.12]^d |
| Knapsack | Discrete | N items | Maximize value ≤ capacity | Binary {0,1}^N |

## 🧪 Testing

Run tests for individual modules:

```bash
# Test problems
python src/problems/continuous/rastrigin.py
python src/problems/discrete/knapsack.py

# Test Firefly Algorithm
python src/swarm/fa.py

# Test classical algorithms
python src/classical/hill_climbing.py

# Run all unit tests
python test/run_all_tests.py
```

## 📈 Visualization

```python
from src.utils.visualization import plot_convergence, plot_comparison

# Convergence plot
plot_convergence(history, "FA on Rastrigin", save_path="results/conv.png")

# Algorithm comparison
plot_comparison(
    {"FA": fa_hist, "SA": sa_hist, "HC": hc_hist},
    "Algorithm Comparison",
    save_path="results/comparison.png"
)
```

## 📖 References

1. Yang, X. S. (2008). *Nature-inspired metaheuristic algorithms*. Luniver press.
2. [Rastrigin Function](https://www.sfu.ca/~ssurjano/rastr.html)
3. [0/1 Knapsack Problem](https://en.wikipedia.org/wiki/Knapsack_problem)

## 👥 Contributors

- Bùi Anh Quân (@1234quan1234)

## 📝 License

This project is for educational purposes as part of the CSTTNT course at HCMUS.