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
- **Graph Search** - BFS, DFS, A* for graph problems

### Benchmark Problems

#### Continuous Functions
- **Sphere** - Simple convex unimodal function
- **Rosenbrock** - Narrow curved valley (Banana function)
- **Rastrigin** - Highly multimodal with many local minima
- **Ackley** - Multimodal with nearly flat outer region

#### Discrete Problems
- **Traveling Salesman Problem (TSP)** - Find shortest tour
- **0/1 Knapsack** - Maximize value within capacity constraint
- **Graph Coloring** - Minimize color conflicts

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
│   │   │   ├── sphere.py
│   │   │   ├── rosenbrock.py
│   │   │   ├── rastrigin.py
│   │   │   └── ackley.py
│   │   └── discrete/              # Discrete optimization problems
│   │       ├── tsp.py
│   │       ├── knapsack.py
│   │       └── graph_coloring.py
│   │
│   ├── swarm/                     # Swarm intelligence algorithms
│   │   └── fa.py                  # Firefly Algorithm (continuous & discrete)
│   │
│   ├── classical/                 # Classical baseline algorithms
│   │   ├── hill_climbing.py
│   │   ├── simulated_annealing.py
│   │   ├── genetic_algorithm.py
│   │   └── graph_search.py        # BFS, DFS, A*
│   │
│   └── utils/                     # Utility modules
│       └── visualization.py       # Plotting and visualization functions
│
├── test/                          # Unit tests
│   ├── test_continuous_problems.py
│   ├── test_tsp_problem.py
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

- Python 3.10
- NumPy
- SciPy (for optimization baselines)
- Matplotlib (for visualization)
- NetworkX (for graph problems)

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
- Run FA on Sphere and Rastrigin functions
- Run FA on TSP
- Compare FA with SA, HC, and GA
- Perform parameter sensitivity analysis
- Generate visualization plots in `results/` folder

Or test individual modules:

```bash
# Test Firefly Algorithm
python src/swarm/fa.py

# Test problems
python src/problems/continuous/sphere.py
python src/problems/discrete/tsp.py

# Test visualization utilities
python src/utils/visualization.py

# Run all unit tests
python test/run_all_tests.py
```

## 💡 Usage Examples

### Example 1: Continuous Optimization with Firefly Algorithm

```python
import numpy as np
from src.problems.continuous.sphere import SphereProblem
from src.swarm.fa import FireflyContinuousOptimizer

# Create problem
problem = SphereProblem(dim=10)

# Create optimizer
optimizer = FireflyContinuousOptimizer(
    problem=problem,
    n_fireflies=30,
    alpha=0.2,      # Randomization parameter
    beta0=1.0,      # Base attractiveness
    gamma=1.0,      # Light absorption coefficient
    seed=42
)

# Run optimization
best_solution, best_fitness, history, trajectory = optimizer.run(max_iter=100)

print(f"Best fitness: {best_fitness}")
print(f"Best solution: {best_solution}")
```

### Example 2: Knapsack with Discrete Firefly Algorithm

```python
import numpy as np
from src.problems.discrete.knapsack import KnapsackProblem
from src.swarm.fa import FireflyKnapsackOptimizer

# Create Knapsack instance
rng = np.random.RandomState(42)
n_items = 20
values = rng.randint(10, 100, n_items)
weights = rng.randint(1, 50, n_items)
capacity = int(0.5 * np.sum(weights))

problem = KnapsackProblem(values, weights, capacity)

# Create optimizer
optimizer = FireflyKnapsackOptimizer(
    problem=problem,
    n_fireflies=30,
    alpha_flip=0.2,
    max_flips_per_move=3,
    seed=42
)

# Run optimization
best_sol, best_fit, history, trajectory = optimizer.run(max_iter=100)

print(f"Best value: {-best_fit:.2f}")  # Negate because minimization
print(f"Best solution: {best_sol}")
```

### Example 3: Compare Multiple Algorithms

```python
from src.problems.continuous.rastrigin import RastriginProblem
from src.swarm.fa import FireflyContinuousOptimizer
from src.classical.simulated_annealing import SimulatedAnnealingOptimizer
from src.classical.genetic_algorithm import GeneticAlgorithmOptimizer

problem = RastriginProblem(dim=5)

# Firefly Algorithm
fa = FireflyContinuousOptimizer(problem, n_fireflies=20, seed=42)
fa_sol, fa_fit, fa_hist, _ = fa.run(max_iter=100)

# Simulated Annealing
sa = SimulatedAnnealingOptimizer(problem, initial_temp=100, seed=42)
sa_sol, sa_fit, sa_hist, _ = sa.run(max_iter=100)

# Genetic Algorithm
ga = GeneticAlgorithmOptimizer(problem, pop_size=20, seed=42)
ga_sol, ga_fit, ga_hist, _ = ga.run(max_iter=100)

print(f"FA best: {fa_fit:.6f}")
print(f"SA best: {sa_fit:.6f}")
print(f"GA best: {ga_fit:.6f}")
```

## 📊 Return Format

All optimizers follow the same interface (inherited from `BaseOptimizer`):

```python
best_solution, best_fitness, history_best, trajectory = optimizer.run(max_iter)
```

**Returns:**
- `best_solution` (np.ndarray): Best solution found
  - Continuous: shape (dim,) - real values
  - TSP: shape (n_cities,) - permutation of city indices
  - Knapsack: shape (n_items,) - binary 0/1 array
  - Coloring: shape (n_nodes,) - color assignments
- `best_fitness` (float): Best objective value (minimization)
- `history_best` (List[float]): Best fitness value at each iteration
- `trajectory` (List): Population/solution snapshots at each iteration
  - For population methods: List of population arrays
  - For single-solution methods: List of solution arrays

**Reproducibility:**
All algorithms accept a `seed` parameter (int or None) for reproducible results.
When seed is set, results will be identical across runs **on the same machine**.

```python
# Same seed = same results
optimizer1 = FireflyContinuousOptimizer(problem, seed=42)
optimizer2 = FireflyContinuousOptimizer(problem, seed=42)

sol1, fit1, _, _ = optimizer1.run(max_iter=100)
sol2, fit2, _, _ = optimizer2.run(max_iter=100)

assert fit1 == fit2  # ✓ Guaranteed
assert np.allclose(sol1, sol2)  # ✓ Guaranteed
```

**Important:** All optimizers use `np.random.RandomState(seed)` internally
instead of global `np.random` to ensure isolation and reproducibility.
This follows scikit-learn best practices.

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

**Key Properties:**
- Brightness inversely proportional to fitness (minimize problems)
- Less bright fireflies move toward brighter ones
- Movement strength decreases exponentially with distance

**Parameters:**
- `alpha` (0.0-1.0): Controls exploration. Higher = more random movement
- `gamma` (0.0-10.0): Controls attraction decay. Higher = more local search
- `beta0` (0.0-2.0): Base attractiveness at r=0

### Firefly Algorithm (Discrete/Knapsack)

For Knapsack, FA uses bit-flip operators instead of continuous movement:

**Movement Strategy:**
1. Compare current solution with better (brighter) solution
2. Identify bit differences between solutions
3. Apply directed bit flips to align with better solution
4. Add random bit flips (controlled by `alpha_flip`) for exploration
5. Repair infeasible solutions using greedy value/weight ratio

**Parameters:**
- `alpha_flip` (0.0-1.0): Probability of random bit flip after directed movement
- `max_flips_per_move`: Maximum number of directed flips per attraction step
- `repair_method`: "greedy_remove" (recommended) or "random_remove"

### Optimization Problems

| Problem | Type | Dimension | Global Minimum | Domain |
|---------|------|-----------|----------------|--------|
| Sphere | Continuous | Any | f(0,...,0) = 0 | [-5.12, 5.12]^d |
| Rosenbrock | Continuous | ≥2 | f(1,...,1) = 0 | [-2.048, 2.048]^d |
| Rastrigin | Continuous | Any | f(0,...,0) = 0 | [-5.12, 5.12]^d |
| Ackley | Continuous | Any | f(0,...,0) = 0 | [-5, 5]^d |
| TSP | Discrete | N cities | Shortest tour | Permutation |
| Knapsack | Discrete | N items | Max value ≤ capacity | Binary |
| Graph Coloring | Discrete | N nodes | 0 conflicts | Integer colors |

## 🧪 Testing

Each module includes a `__main__` block with tests. Run tests:

```bash
# Test all continuous problems
python src/problems/continuous/sphere.py
python src/problems/continuous/rosenbrock.py
python src/problems/continuous/rastrigin.py
python src/problems/continuous/ackley.py

# Test all discrete problems
python src/problems/discrete/tsp.py
python src/problems/discrete/knapsack.py
python src/problems/discrete/graph_coloring.py

# Test all optimizers
python src/swarm/fa.py
python src/classical/hill_climbing.py
python src/classical/simulated_annealing.py
python src/classical/genetic_algorithm.py
python src/classical/graph_search.py
```

## 📖 References

1. Yang, X. S. (2008). *Nature-inspired metaheuristic algorithms*. Luniver press.
2. [Firefly Algorithm Overview](https://www.alpsconsult.net/post/firefly-algorithm-fa-overview)
3. [Virtual Library of Simulation Experiments - Test Functions](https://www.sfu.ca/~ssurjano/optimization.html)
4. [Swap-Based Discrete Firefly Algorithm for TSP](https://www.researchgate.net/publication/320480703_Swap-Based_Discrete_Firefly_Algorithm_for_Traveling_Salesman_Problem)

## 👥 Contributors

- Bùi Anh Quân (@1234quan1234)

## 📝 License

This project is for educational purposes as part of the CSTTNT course at HCMUS.

---

**Note:** This is a research and educational project implementing classical metaheuristic algorithms.
For production use, consider:
- More optimized implementations (vectorization, JIT compilation)
- Advanced variants (adaptive parameters, hybrid methods)
- Parallel/distributed execution
- Comprehensive benchmarking on standard test suites
