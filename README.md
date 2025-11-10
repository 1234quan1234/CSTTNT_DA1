# AI Search and Optimization Project

A comprehensive, production-ready Python framework for comparing **Firefly Algorithm** (FA) with classical optimization methods on continuous and discrete benchmark problems.

## 🎯 Project Overview

This project implements and benchmarks multiple optimization algorithms with:

- ✅ **Full type hints** for all functions and classes
- ✅ **Comprehensive error handling** with actionable error messages
- ✅ **>80% test coverage** with edge case testing
- ✅ **Parallel execution** support for faster benchmarking
- ✅ **Academic-grade visualizations** following metaheuristic best practices
- ✅ **Reproducible results** with fixed seeds
- ✅ **Statistical analysis** with Wilcoxon and Friedman tests

### Algorithms Implemented

#### Swarm Intelligence
- **Firefly Algorithm (FA)** - Bio-inspired optimization
  - Continuous optimization variant
  - Discrete Knapsack variant with repair strategies

#### Classical Baselines
- **Hill Climbing (HC)** - Greedy local search with restart
- **Simulated Annealing (SA)** - Probabilistic local search with temperature scheduling
- **Genetic Algorithm (GA)** - Evolutionary optimization with elitism

### Benchmark Problems

#### Continuous Functions
- **Rastrigin** - Highly multimodal with many local minima
  - Dimensions: d=10, 30, 50
  - Global optimum: f(0,...,0) = 0
  - Domain: [-5.12, 5.12]^d
  - Three test configurations: quick convergence, multimodal escape, scalability

#### Discrete Problems
- **0/1 Knapsack** - Maximize value within capacity constraint
  - Sizes: n=50, 100, 200 items
  - 4 instance types: uncorrelated, weakly correlated, strongly correlated, subset-sum
  - DP optimal solution available for n ≤ 100
  - Multiple random seeds for statistical robustness

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
│   │       └── knapsack.py        # 0/1 Knapsack problem with DP solver
│   │
│   ├── swarm/                     # Swarm intelligence algorithms
│   │   └── fa.py                  # Firefly Algorithm (continuous & discrete)
│   │
│   ├── classical/                 # Classical baseline algorithms
│   │   ├── hill_climbing.py       # Hill Climbing with random restart
│   │   ├── simulated_annealing.py # Simulated Annealing
│   │   └── genetic_algorithm.py   # Genetic Algorithm
│   │
│   └── utils/                     # Utility modules
│       └── visualization.py       # Academic-grade plotting functions
│
├── test/                          # Unit tests (>80% coverage)
│   ├── test_continuous_problems.py
│   ├── test_knapsack_problem.py
│   ├── test_firefly_algorithm.py
│   ├── test_classical_algorithms.py
│   ├── test_edge_cases.py
│   ├── test_parallel_execution.py
│   ├── run_all_tests.py
│   └── README.md
│
├── benchmark/                     # Comprehensive benchmark suite
│   ├── config.py                  # Centralized benchmark configurations
│   ├── instance_generator.py      # Knapsack instance generation
│   ├── run_rastrigin.py          # Rastrigin benchmark runner
│   ├── run_knapsack.py           # Knapsack benchmark runner
│   ├── analyze_results.py        # Statistical analysis (Wilcoxon, Friedman)
│   ├── visualize.py              # Generate all plots
│   ├── run_all.py                # Master script (parallel execution)
│   ├── run_all.sh                # Shell script wrapper
│   ├── test_benchmarks.py        # Benchmark integration tests
│   ├── results/                  # Auto-generated results
│   │   ├── rastrigin/           # Rastrigin results by config
│   │   ├── knapsack/            # Knapsack results by instance
│   │   ├── plots/               # All visualizations
│   │   ├── logs/                # Execution logs
│   │   ├── summaries/           # Statistical summaries
│   │   ├── rastrigin_summary.csv
│   │   └── knapsack_summary.csv
│   └── README.md                 # Detailed benchmark documentation
│
├── results/                       # Legacy results directory (deprecated)
├── demo.py                        # Quick demonstration script
├── environment.yml                # Conda environment
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── QUICKSTART.md                  # Quick start guide
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- NumPy
- SciPy (for statistical tests)
- Matplotlib (for visualization)
- Pandas (for data analysis)
- pytest (for testing)

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

### Quick Start

#### Option 1: Run Complete Benchmark Suite (Recommended)

```bash
# Fast mode with parallel execution (use all CPU cores)
python benchmark/run_all.py --quick --jobs -1

# Full benchmark with 4 parallel workers
python benchmark/run_all.py --full --jobs 4
```

This will:
- Run all Rastrigin configurations (quick_convergence, multimodal_escape, scalability)
- Run all Knapsack instances (n=50, 100, 200 with 4 instance types)
- Perform 30 independent runs per configuration
- Generate statistical analysis (mean, std, median, Wilcoxon tests, Friedman tests)
- Create all visualizations in `benchmark/results/plots/`

#### Option 2: Run Individual Benchmarks

**Rastrigin Benchmark:**
```bash
# Quick convergence test (d=10, ~2 minutes with 4 cores)
python benchmark/run_rastrigin.py --config quick_convergence --jobs 4

# Multimodal escape test (d=30, ~5 minutes)
python benchmark/run_rastrigin.py --config multimodal_escape --jobs -1

# Scalability test (d=50, ~10 minutes)
python benchmark/run_rastrigin.py --config scalability --jobs -1
```

**Knapsack Benchmark:**
```bash
# Small instances (n=50, ~5 minutes with 4 cores)
python benchmark/run_knapsack.py --size 50 --jobs 4

# Medium instances with DP optimal (n=100, ~15 minutes)
python benchmark/run_knapsack.py --size 100 --jobs -1

# Large instances (n=200, ~30 minutes)
python benchmark/run_knapsack.py --size 200 --jobs 4
```

#### Option 3: Analysis and Visualization Only

If you already have benchmark results:

```bash
# Generate statistical analysis
python benchmark/analyze_results.py --problem all

# Generate all plots
python benchmark/visualize.py
```

### Quick Test

Test individual algorithm implementations:

```bash
# Test Firefly Algorithm
python src/swarm/fa.py

# Test problem definitions
python src/problems/continuous/rastrigin.py
python src/problems/discrete/knapsack.py

# Test classical algorithms
python src/classical/hill_climbing.py
python src/classical/simulated_annealing.py
python src/classical/genetic_algorithm.py
```

## 💡 Usage Examples

### Example 1: Rastrigin Function with Firefly Algorithm

```python
import numpy as np
from src.problems.continuous.rastrigin import RastriginProblem
from src.swarm.fa import FireflyContinuousOptimizer

# Create problem instance
problem = RastriginProblem(dim=10)

# Create optimizer with parameters tuned for multimodal problems
optimizer = FireflyContinuousOptimizer(
    problem=problem,
    n_fireflies=40,
    alpha=0.3,      # Higher randomization for exploration
    beta0=1.0,      # Base attractiveness
    gamma=1.0,      # Light absorption coefficient
    seed=42         # For reproducibility
)

# Run optimization
best_solution, best_fitness, history, trajectory = optimizer.run(max_iter=100)

print(f"Best fitness: {best_fitness:.6f}")
print(f"Error to optimum: {abs(best_fitness):.6f}")
print(f"Convergence: {history[0]:.4f} -> {history[-1]:.4f}")
```

### Example 2: Knapsack with Discrete Firefly Algorithm

```python
import numpy as np
from src.problems.discrete.knapsack import KnapsackProblem
from src.swarm.fa import FireflyKnapsackOptimizer

# Create Knapsack instance (uncorrelated type)
rng = np.random.RandomState(42)
n_items = 50
values = rng.randint(10, 100, n_items)
weights = rng.randint(1, 50, n_items)
capacity = int(0.5 * np.sum(weights))

problem = KnapsackProblem(values, weights, capacity)

# Compute DP optimal for comparison (feasible for n ≤ 100)
dp_optimal = problem.solve_dp()
print(f"DP Optimal: {dp_optimal}")

# Create optimizer with discrete-specific parameters
optimizer = FireflyKnapsackOptimizer(
    problem=problem,
    n_fireflies=60,
    alpha_flip=0.2,           # Random bit flip probability
    max_flips_per_move=3,     # Directed flips per attraction
    repair_method="greedy_remove",  # Constraint repair strategy
    seed=42
)

# Run optimization
best_sol, best_fit, history, trajectory = optimizer.run(max_iter=166)

total_value = -best_fit  # Negate for actual value
total_weight = np.sum(best_sol * weights)
optimality_gap = (dp_optimal - total_value) / dp_optimal * 100

print(f"Best value: {total_value:.0f} (DP: {dp_optimal:.0f})")
print(f"Optimality gap: {optimality_gap:.2f}%")
print(f"Weight: {total_weight:.0f}/{capacity} ({total_weight/capacity*100:.1f}%)")
print(f"Items selected: {np.sum(best_sol)}/{n_items}")
```

### Example 3: Compare All Algorithms

```python
from src.problems.continuous.rastrigin import RastriginProblem
from src.swarm.fa import FireflyContinuousOptimizer
from src.classical.simulated_annealing import SimulatedAnnealingOptimizer
from src.classical.hill_climbing import HillClimbingOptimizer
from src.classical.genetic_algorithm import GeneticAlgorithmOptimizer

problem = RastriginProblem(dim=10)
max_iter = 125  # Same budget for fair comparison

# Firefly Algorithm
fa = FireflyContinuousOptimizer(problem, n_fireflies=40, alpha=0.3, seed=42)
_, fa_fit, fa_hist, _ = fa.run(max_iter=max_iter)

# Simulated Annealing
sa = SimulatedAnnealingOptimizer(problem, initial_temp=100, cooling_rate=0.95, seed=42)
_, sa_fit, sa_hist, _ = sa.run(max_iter=max_iter)

# Hill Climbing with restart
hc = HillClimbingOptimizer(problem, num_neighbors=20, restart_interval=50, seed=42)
_, hc_fit, hc_hist, _ = hc.run(max_iter=max_iter)

# Genetic Algorithm
ga = GeneticAlgorithmOptimizer(problem, pop_size=40, crossover_rate=0.8, seed=42)
_, ga_fit, ga_hist, _ = ga.run(max_iter=max_iter)

print(f"FA: {abs(fa_fit):.6f}")
print(f"SA: {abs(sa_fit):.6f}")
print(f"HC: {abs(hc_fit):.6f}")
print(f"GA: {abs(ga_fit):.6f}")
```

## 📊 Benchmark Configurations

### Rastrigin Configurations

| Config Name | Dimension | Budget (evals) | Max Iter | Target Error | Purpose |
|-------------|-----------|----------------|----------|--------------|---------|
| `quick_convergence` | 10 | 5,000 | 125 | 10.0 | Fast convergence test |
| `multimodal_escape` | 30 | 20,000 | 500 | 50.0 | Escape local minima |
| `scalability` | 50 | 40,000 | 1,000 | 100.0 | High-dimensional scaling |

**Algorithm Parameters:**
- **FA**: n_fireflies=40, α=0.3, β₀=1.0, γ=1.0
- **SA**: T₀=100, cooling=0.95, step=0.5
- **HC**: neighbors=20, step=0.5, restart=50
- **GA**: pop=40, crossover=0.8, mutation=0.1

### Knapsack Configurations

| n Items | Instance Types | Seeds | Budget (evals) | Max Iter (FA/GA) | Max Iter (SA/HC) | DP Optimal? |
|---------|----------------|-------|----------------|------------------|------------------|-------------|
| 50 | All 4 types | 42, 123, 999 | 10,000 | 166 | 10,000 | ✓ Yes |
| 100 | All 4 types | 42, 123, 999 | 15,000 | 250 | 15,000 | ✓ Yes |
| 200 | Uncorr, Weak | 42, 123, 999 | 30,000 | 500 | 30,000 | ✗ No |

**Instance Types:**
1. **Uncorrelated**: Random values and weights
2. **Weakly Correlated**: values ≈ weights ± noise
3. **Strongly Correlated**: values = weights + 100
4. **Subset-Sum**: values = weights (hardest)

**Algorithm Parameters:**
- **FA**: n_fireflies=60, α_flip=0.2, max_flips=3, repair="greedy_remove"
- **SA**: T₀=1000, cooling=0.95
- **HC**: neighbors=20, restart=100
- **GA**: pop=60, crossover=0.8, mutation=1/n, elitism=0.1

## 📈 Output Format

All benchmark results are saved in JSON format for reproducibility.

### Rastrigin Results

**File naming:** `benchmark/results/rastrigin/{config}/{algorithm}_seed{i}.json`

```json
{
  "algorithm": "FA",
  "config": "quick_convergence",
  "seed": 0,
  "best_fitness": 8.4567,
  "history": [45.6, 34.2, 23.1, 15.8, 8.4567],
  "elapsed_time": 2.15,
  "evaluations": 5000,
  "final_error": 8.4567
}
```

### Knapsack Results

**File naming:** `benchmark/results/knapsack/n{size}_{type}_seed{instance_seed}_{algo}.json`

```json
{
  "config": {
    "n_items": 50,
    "instance_type": "uncorrelated",
    "instance_seed": 42,
    "capacity": 500.0,
    "budget": 10000,
    "dp_optimal": 2450.0
  },
  "results": [
    {
      "algorithm": "FA",
      "seed": 0,
      "best_value": 2387.0,
      "best_fitness": -2387.0,
      "total_weight": 487.5,
      "is_feasible": true,
      "history": [-1200.0, -1500.0, ..., -2387.0],
      "elapsed_time": 3.45,
      "items_selected": 18,
      "capacity_utilization": 0.975,
      "optimality_gap": 2.57
    }
  ]
}
```

### Summary CSV Files

Auto-generated by `analyze_results.py`:

**rastrigin_summary.csv:**
```csv
Config,Algorithm,Mean,Std,Median,Best,Worst,Q1,Q3,Success_Rate
quick_convergence,FA,8.45,2.31,7.89,3.21,15.67,6.12,10.23,0.83
quick_convergence,SA,15.23,4.56,14.12,7.89,28.45,11.34,18.67,0.43
...
```

**knapsack_summary.csv:**
```csv
Size,Type,InstanceSeed,Algorithm,Mean_Gap,Std_Gap,Median_Gap,Best_Gap,Worst_Gap,Feasibility_Rate
50,uncorrelated,42,FA,2.34,1.12,2.15,0.45,5.67,1.00
50,uncorrelated,42,SA,8.76,3.45,8.12,3.21,15.34,0.97
...
```

## 📊 Visualizations (Academic Standards)

All plots follow metaheuristic benchmarking best practices and are saved in `benchmark/results/plots/`.

### Rastrigin Visualizations

1. **Convergence Curves** (`rastrigin_{config}_convergence.png`)
   - X-axis: Function evaluations (not iterations)
   - Y-axis: Error to optimum |f(x) - 0| (log scale)
   - Median trajectory with IQR (25-75%) shaded bands
   - Shows convergence speed fairly across algorithms

2. **Final Error Boxplots** (`rastrigin_{config}_boxplot.png`)
   - Distribution of final errors across 30 runs
   - Log scale for better visualization
   - Mean markers (red diamonds)
   - Shows robustness and outlier behavior

3. **ECDF Plots** (`rastrigin_{config}_ecdf.png`)
   - Empirical Cumulative Distribution Function
   - Shows P(error ≤ x) for each algorithm
   - Better than mean/median for tail behavior analysis

4. **Scalability Plot** (`rastrigin_scalability.png`)
   - Mean error vs dimension (d=10/30/50)
   - Log scale with error bars (±1 std)
   - Shows which algorithms scale well to higher dimensions

### Knapsack Visualizations

**Per-Instance Plots:**

1. **Convergence Curves** (`knapsack_n{size}_{type}_seed{seed}_convergence.png`)
   - X-axis: Function evaluations
   - Y-axis: Best value found (higher is better)
   - Median with IQR bands
   - **DP optimal reference line** (red dashed) when available

2. **Optimality Gap Boxplots** (`knapsack_n{size}_{type}_seed{seed}_gap_boxplot.png`)
   - Distribution of (DP_opt - best_value) / DP_opt × 100%
   - Only for n=50, n=100 where DP is feasible
   - Lower is better (0% = optimal)

**Aggregate Plots:**

3. **Feasibility Rate** (`knapsack_feasibility.png`)
   - Bar chart showing % of feasible solutions
   - Grouped by n_items (50/100/200)
   - Sub-grouped by instance type
   - **Critical metric**: Algorithms violating constraints are penalized

4. **Capacity Utilization** (`knapsack_capacity_utilization.png`)
   - Boxplots of weight_used / capacity
   - Grouped by n_items
   - Green line at 1.0 = perfect utilization
   - Values >1.0 indicate constraint violations

5. **Runtime vs Quality** (`knapsack_runtime_quality.png`)
   - Scatter plot of elapsed_time vs optimality_gap
   - Shows Pareto front of fast-but-good algorithms
   - Color-coded by algorithm
   - Helps identify practical trade-offs

6. **Scalability Plots** (`knapsack_{type}_seed{seed}_scalability.png`)
   - Mean optimality gap vs n_items (50/100/200)
   - Error bars show ±1 std
   - Generated for uncorrelated and weakly_correlated types

## 📈 Performance Metrics

### Rastrigin Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Error to Optimum** | \|f(x) - 0\| | Lower is better (0 = perfect) |
| **Convergence Speed** | Evals to reach target | Faster is better |
| **Success Rate** | % runs achieving target error | Higher is better |
| **ECDF** | P(error ≤ x) | Shows distribution tail |

### Knapsack Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Optimality Gap** | (DP_opt - value) / DP_opt × 100% | Lower is better (0% = optimal) |
| **Feasibility Rate** | % feasible solutions | **Must be 100%** |
| **Capacity Utilization** | weight_used / capacity | Higher is better (≤1.0) |
| **Runtime** | Elapsed time (seconds) | Lower is better |

### Statistical Tests

The benchmark suite performs rigorous statistical analysis:

1. **Friedman Test** (non-parametric ANOVA)
   - Tests if algorithms have significantly different performance
   - Reports average ranks (lower is better)
   - p-value < 0.05 indicates significant differences

2. **Wilcoxon Signed-Rank Test** (pairwise)
   - Compares each pair of algorithms
   - Reports p-values in matrix form
   - p-value < 0.05 indicates significant difference
   - Bonferroni correction for multiple comparisons

Example output:
```
Average Ranks (lower is better):
  FA: 1.47
  GA: 2.13
  SA: 2.87
  HC: 3.53

Pairwise Wilcoxon (p-values):
          FA      GA      SA      HC
FA        —   0.0234  0.0001  0.0000
GA   0.0234       —   0.0012  0.0000
SA   0.0001  0.0012       —   0.0345
HC   0.0000  0.0000  0.0345       —
```

## 🎯 Expected Results

### Rastrigin Performance

| Algorithm | d=10 | d=30 | d=50 | Scaling | Strengths |
|-----------|------|------|------|---------|-----------|
| **FA** | ✓✓✓ | ✓✓ | ✓ | Good | Fast early convergence, swarm cooperation |
| **GA** | ✓✓ | ✓✓ | ✓✓ | Excellent | Stable across dimensions, genetic diversity |
| **SA** | ✓ | ✗ | ✗ | Poor | Struggles with high-dimensional multimodal |
| **HC** | ✗ | ✗ | ✗ | Poor | Gets trapped in local minima |

**Key Findings:**
- FA achieves best performance on d=10 due to effective swarm search
- GA maintains consistent quality across all dimensions
- SA and HC struggle with multimodal landscapes

### Knapsack Performance

| Algorithm | Uncorr | Weakly | Strongly | Subset | Strengths |
|-----------|--------|--------|----------|--------|-----------|
| **FA** | ✓✓ | ✓✓ | ✓✓✓ | ✓ | Good balance, effective repair |
| **GA** | ✓✓✓ | ✓✓✓ | ✓✓ | ✓✓ | Best overall, strong crossover |
| **SA** | ✓ | ✓ | ✓ | ✗ | Decent for easy instances |
| **HC** | ✗ | ✗ | ✗ | ✗ | Poor exploration |

**Key Findings:**
- FA/GA achieve <5% optimality gap for n≤100
- Strongly correlated instances favor swarm intelligence
- Subset-sum is hardest for all algorithms (exact value=weight matching)
- Repair strategies are critical for maintaining feasibility

## 🧪 Testing

### Run Complete Test Suite

```bash
# Run all tests with coverage report
pytest test/ --cov=src --cov=benchmark --cov-report=html --cov-report=term

# View HTML coverage report
xdg-open htmlcov/index.html  # Linux
open htmlcov/index.html      # macOS
```

### Test Coverage Status

| Module | Coverage Target | Status |
|--------|-----------------|--------|
| `src/core/*.py` | 90%+ | ✓ Achieved |
| `src/swarm/*.py` | 80%+ | ✓ Achieved |
| `src/classical/*.py` | 80%+ | ✓ Achieved |
| `src/problems/*.py` | 85%+ | ✓ Achieved |
| `benchmark/*.py` | 70%+ | ✓ Achieved |

### Test Categories

**Unit Tests (`test/`):**
- `test_continuous_problems.py` - Continuous benchmark functions
- `test_knapsack_problem.py` - Knapsack problem and DP solver
- `test_firefly_algorithm.py` - FA continuous and discrete variants
- `test_classical_algorithms.py` - HC, SA, GA implementations
- `test_edge_cases.py` - Boundary conditions, extreme inputs
- `test_parallel_execution.py` - Concurrency, reproducibility
- `test_utils.py` - Utility and visualization functions

**Integration Tests:**
```bash
# Quick integration test (5 runs per config)
python test/run_all_tests.py --quick

# Full integration test (30 runs per config)
python test/run_all_tests.py --full
```

**Benchmark Tests:**
```bash
# Test benchmark infrastructure
pytest benchmark/test_benchmarks.py -v
```

## 📚 References

1. Yang, X. S. (2008). *Nature-inspired metaheuristic algorithms*. Luniver press.
2. Yang, X. S. (2010). "Firefly algorithm, stochastic test functions and design optimisation". *International Journal of Bio-Inspired Computation*, 2(2), 78-84.
3. [Rastrigin Function - Virtual Library of Simulation Experiments](https://www.sfu.ca/~ssurjano/rastr.html)
4. [Knapsack Problem - Wikipedia](https://en.wikipedia.org/wiki/Knapsack_problem)
5. Pisinger, D. (1995). "An expanding-core algorithm for the exact 0-1 knapsack problem". *European Journal of Operational Research*, 87(1), 175-187.
6. Wilcoxon, F. (1945). "Individual comparisons by ranking methods". *Biometrics Bulletin*, 1(6), 80-83.
7. Friedman, M. (1937). "The use of ranks to avoid the assumption of normality implicit in the analysis of variance". *Journal of the American Statistical Association*, 32(200), 675-701.

## 👥 Contributors

- Bùi Anh Quân (@1234quan1234)

## 📝 License

This project is for educational purposes as part of the CSTTNT (Cơ Sở Trí Tuệ Nhân Tạo) course at HCMUS (Ho Chi Minh City University of Science).

---

**For detailed benchmark documentation, see:** [`benchmark/README.md`](benchmark/README.md)

**For quick start guide, see:** [`QUICKSTART.md`](QUICKSTART.md)

**For testing guide, see:** [`test/README.md`](test/README.md)