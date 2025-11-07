# AI Search and Optimization Project

A comprehensive Python framework for comparing **Firefly Algorithm** (FA) with classical optimization methods on continuous and discrete benchmark problems.

## 🎯 Project Overview

This project implements and benchmarks multiple optimization algorithms on two main problems:

### Swarm Intelligence
- **Firefly Algorithm (FA)** - Continuous optimization
- **Firefly Algorithm (FA)** - Discrete Knapsack variant

### Classical Baselines
- **Hill Climbing** - Greedy local search
- **Simulated Annealing** - Probabilistic local search with temperature scheduling
- **Genetic Algorithm** - Evolutionary optimization

### Benchmark Problems

#### Continuous Functions
- **Rastrigin** - Highly multimodal with many local minima (main test function)

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
│   │   └── fa.py                  # Firefly Algorithm (continuous & discrete)
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

Run the comprehensive demo script:

```bash
python demo.py
```

This will:
- Run FA on Rastrigin function (continuous multimodal)
- Run FA on Knapsack problem (discrete)
- Compare FA with SA, HC, and GA on both problems
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

## Running Benchmarks

### Quick Test
```bash
# Test all benchmarks quickly
python benchmark/test_benchmarks.py --quick

# Run quick benchmark suite
python benchmark/run_all.py --quick
```

### Full Benchmark Suite
```bash
# Run all benchmarks with all configurations
python benchmark/run_all.py --full

# Run specific problem benchmark
python benchmark/run_rastrigin.py --config quick_convergence
python benchmark/run_tsp.py --config small
python benchmark/run_knapsack.py --config medium
```

### Analyze Results
```bash
# Analyze all results
python benchmark/analyze_results.py

# Analyze specific benchmark
python benchmark/analyze_results.py --problem rastrigin --config quick_convergence
```

### Using pytest
```bash
# Run all tests
pytest benchmark/test_benchmarks.py -v

# Run specific test class
pytest benchmark/test_benchmarks.py::TestRastriginBenchmark -v
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

print(f"FA best: {fa_fit:.6f}")
print(f"SA best: {sa_fit:.6f}")
print(f"HC best: {hc_fit:.6f}")
print(f"GA best: {ga_fit:.6f}")
```

## 📊 Return Format

All optimizers follow the same interface (inherited from `BaseOptimizer`):

```python
best_solution, best_fitness, history_best, trajectory = optimizer.run(max_iter)
```

**Returns:**
- `best_solution` (np.ndarray): Best solution found
  - Rastrigin: shape (dim,) - real values in [-5.12, 5.12]
  - Knapsack: shape (n_items,) - binary 0/1 array
- `best_fitness` (float): Best objective value (minimization)
- `history_best` (List[float]): Best fitness value at each iteration
- `trajectory` (List): Population/solution snapshots at each iteration
  - For population methods (FA, GA): List of population arrays
  - For single-solution methods (HC, SA): List of solution arrays

**Reproducibility:**
All algorithms accept a `seed` parameter (int or None) for reproducible results.

```python
# Same seed = same results
optimizer1 = FireflyContinuousOptimizer(problem, seed=42)
optimizer2 = FireflyContinuousOptimizer(problem, seed=42)

sol1, fit1, _, _ = optimizer1.run(max_iter=100)
sol2, fit2, _, _ = optimizer2.run(max_iter=100)

assert fit1 == fit2  # ✓ Guaranteed
assert np.allclose(sol1, sol2)  # ✓ Guaranteed
```

## 🔬 Algorithm Details

### Firefly Algorithm (Continuous)

The continuous FA moves fireflies based on attractiveness and brightness:

```
x_i = x_i + β₀·exp(-γ·r²)·(x_j - x_i) + α·(rand - 0.5)
```

Where:
- `β₀`: Attractiveness at distance r=0 (default: 1.0)
- `γ`: Light absorption coefficient (default: 1.0)
- `α`: Randomization parameter (default: 0.2)
- `r`: Euclidean distance between fireflies i and j

**Parameters for Rastrigin:**
- `alpha=0.3`: Higher for multimodal exploration
- `gamma=0.5`: Lower for global search
- `beta0=1.0`: Standard attractiveness

### Firefly Algorithm (Discrete/Knapsack)

For Knapsack, FA uses bit-flip operators:

**Movement Strategy:**
1. Compare current solution with better (brighter) solution
2. Identify bit differences between solutions
3. Apply directed bit flips to align with better solution
4. Add random bit flips (controlled by `alpha_flip`) for exploration
5. Repair infeasible solutions using greedy value/weight ratio

**Parameters:**
- `alpha_flip=0.2`: Probability of random bit flip
- `max_flips_per_move=3`: Maximum directed flips per step
- `repair_method="greedy_remove"`: Greedy repair for constraint violation

### Optimization Problems

| Problem | Type | Dimension | Global Minimum | Domain |
|---------|------|-----------|----------------|--------|
| Rastrigin | Continuous | Any | f(0,...,0) = 0 | [-5.12, 5.12]^d |
| Knapsack | Discrete | N items | Max value ≤ capacity | Binary {0,1}^N |

**Rastrigin Function:**
- Highly multimodal with many local minima
- Tests global optimization capability
- Formula: f(x) = 10d + Σ[x_i² - 10cos(2πx_i)]

**0/1 Knapsack:**
- NP-hard combinatorial optimization
- Binary decision variables
- Constraint: Σ(w_i · x_i) ≤ capacity
- Objective: maximize Σ(v_i · x_i)

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
python src/classical/simulated_annealing.py
python src/classical/genetic_algorithm.py

# Run all unit tests
python test/run_all_tests.py
```

## 📈 Visualization

The project includes comprehensive visualization utilities:

```python
from src.utils.visualization import plot_convergence, plot_comparison, plot_trajectory_2d

# Convergence plot
plot_convergence(history, "FA on Rastrigin", save_path="results/convergence.png")

# Algorithm comparison
plot_comparison(
    {"FA": fa_hist, "SA": sa_hist, "HC": hc_hist, "GA": ga_hist},
    "Algorithm Comparison",
    save_path="results/comparison.png"
)

# 2D trajectory (for 2D problems)
plot_trajectory_2d(trajectory, problem, save_path="results/trajectory.png")
```

## 📖 References

1. Yang, X. S. (2008). *Nature-inspired metaheuristic algorithms*. Luniver press.
2. [Firefly Algorithm Overview](https://www.alpsconsult.net/post/firefly-algorithm-fa-overview)
3. [Virtual Library of Simulation Experiments - Rastrigin Function](https://www.sfu.ca/~ssurjano/rastr.html)
4. [0/1 Knapsack Problem](https://en.wikipedia.org/wiki/Knapsack_problem)

## 👥 Contributors

- Bùi Anh Quân (@1234quan1234)

## 📝 License

This project is for educational purposes as part of the CSTTNT course at HCMUS.

---

**Note:** This is a research and educational project implementing classical metaheuristic algorithms.
The implementation focuses on clarity and educational value rather than production optimization.