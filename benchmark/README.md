# Benchmark Suite - Complete Testing Framework

Comprehensive benchmark comparing **Firefly Algorithm (FA)** with classical baselines:
- Hill Climbing (HC)
- Simulated Annealing (SA)
- Genetic Algorithm (GA)

## 📁 Structure

```
benchmark/
├── config.py                  # Centralized configuration
├── instance_generator.py      # Knapsack instance generation
├── run_rastrigin.py          # Rastrigin benchmark script
├── run_knapsack.py           # Knapsack benchmark script
├── analyze_results.py        # Statistical analysis (Wilcoxon, Friedman)
├── visualize.py              # Generate all plots
├── run_all.py                # Master script with parallel execution
├── run_all.sh                # Shell wrapper for run_all.py
├── test_benchmarks.py        # Integration tests for benchmark suite
├── results/                  # Output directory (auto-generated)
│   ├── rastrigin/
│   │   ├── quick_convergence/
│   │   ├── multimodal_escape/
│   │   └── scalability/
│   ├── knapsack/
│   │   └── n{size}_{type}_seed{seed}_{algo}.json
│   ├── plots/               # All generated visualizations
│   ├── logs/                # Execution logs
│   ├── summaries/           # Statistical summary reports
│   ├── rastrigin_summary.csv
│   └── knapsack_summary.csv
└── README.md                 # This file
```

## 🚀 Quick Start

### Option 1: Run Everything (Parallel - Recommended)

```bash
# Quick mode: Use all CPU cores, reduced runs for testing
python benchmark/run_all.py --quick --jobs -1

# Full mode: 30 runs per config, use 4 parallel workers
python benchmark/run_all.py --full --jobs 4

# Shell wrapper (same as above)
./benchmark/run_all.sh --quick --jobs -1
```

**What happens:**
- Runs all Rastrigin configs (quick_convergence, multimodal_escape, scalability)
- Runs all Knapsack instances (n=50,100,200 × 4 types × 3 seeds)
- Generates statistical analysis with Wilcoxon and Friedman tests
- Creates all visualizations in `results/plots/`
- Saves summary tables in CSV format

**Execution time:**
- Quick mode (5 runs): ~15-30 minutes with 4 cores
- Full mode (30 runs): ~2-4 hours with 4 cores

### Option 2: Run Individual Benchmarks (Parallel)

#### Rastrigin Benchmark

```bash
# Quick convergence test (d=10, ~2 minutes with 4 cores)
python benchmark/run_rastrigin.py --config quick_convergence --jobs 4

# Multimodal escape test (d=30, ~5 minutes with auto-detect cores)
python benchmark/run_rastrigin.py --config multimodal_escape --jobs -1

# Scalability test (d=50, ~10 minutes)
python benchmark/run_rastrigin.py --config scalability --jobs -1

# Run all Rastrigin configs sequentially
python benchmark/run_rastrigin.py --config all --jobs 4
```

#### Knapsack Benchmark

```bash
# Small instances (n=50, all types, ~5 minutes with 4 cores)
python benchmark/run_knapsack.py --size 50 --jobs 4

# Medium instances with DP optimal (n=100, ~15 minutes)
python benchmark/run_knapsack.py --size 100 --jobs -1

# Large instances (n=200, only uncorrelated & weakly, ~30 minutes)
python benchmark/run_knapsack.py --size 200 --jobs 4

# Run all sizes sequentially
python benchmark/run_knapsack.py --size all --jobs -1
```

### Option 3: Analysis and Visualization Only

If you already have benchmark results:

```bash
# Generate statistical analysis for both problems
python benchmark/analyze_results.py --problem all

# Analyze only Rastrigin results
python benchmark/analyze_results.py --problem rastrigin

# Analyze only Knapsack results
python benchmark/analyze_results.py --problem knapsack

# Generate all visualizations
python benchmark/visualize.py

# Generate specific plots
python benchmark/visualize.py --problem rastrigin
python benchmark/visualize.py --problem knapsack
```

## 📊 Configurations

### Rastrigin Configurations

| Config Name | Dimension | Budget (evals) | Max Iter | Target Error | Purpose |
|-------------|-----------|----------------|----------|--------------|---------|
| `quick_convergence` | 10 | 5,000 | 125 | 10.0 | Fast convergence test |
| `multimodal_escape` | 30 | 20,000 | 500 | 50.0 | Escape local minima |
| `scalability` | 50 | 40,000 | 1,000 | 100.0 | High-dimensional scaling |

**Algorithm Parameters (from `config.py`):**
- **FA**: n_fireflies=40, α=0.3, β₀=1.0, γ=1.0
- **SA**: T₀=100, cooling=0.95, step_size=0.5
- **HC**: num_neighbors=20, step_size=0.5, restart_interval=50
- **GA**: pop_size=40, crossover_rate=0.8, mutation_rate=0.1

**Number of Independent Runs:**
- Quick mode: 5 runs per configuration
- Full mode: 30 runs per configuration (for statistical significance)

### Knapsack Configurations

| n Items | Instance Types | Seeds | Budget (evals) | Max Iter (FA/GA) | Max Iter (SA/HC) | DP Optimal? |
|---------|----------------|-------|----------------|------------------|------------------|-------------|
| 50 | All 4 types | 42, 123, 999 | 10,000 | 166 | 10,000 | ✓ Yes |
| 100 | All 4 types | 42, 123, 999 | 15,000 | 250 | 15,000 | ✓ Yes |
| 200 | Uncorr, Weak | 42, 123, 999 | 30,000 | 500 | 30,000 | ✗ No (too large) |

**Instance Types (from `instance_generator.py`):**
1. **Uncorrelated**: `values ~ U[10,100]`, `weights ~ U[1,50]`
2. **Weakly Correlated**: `values = weights + U[-10,10]`
3. **Strongly Correlated**: `values = weights + 100`
4. **Subset-Sum**: `values = weights` (hardest variant)

**Algorithm Parameters (from `config.py`):**
- **FA**: n_fireflies=60, α_flip=0.2, max_flips_per_move=3, repair="greedy_remove"
- **SA**: T₀=1000, cooling_rate=0.95
- **HC**: num_neighbors=20, restart_interval=100
- **GA**: pop_size=60, crossover_rate=0.8, mutation_rate=1/n, elitism_rate=0.1

**Number of Independent Runs:**
- Quick mode: 5 runs per instance
- Full mode: 30 runs per instance

**Note:** For n=200, strongly correlated and subset-sum are skipped due to extreme computational cost.

## 📦 Output Format

### Rastrigin JSON Output

**File naming:** `results/rastrigin/{config}/{algorithm}_seed{i}.json`

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

**Fields:**
- `algorithm`: Algorithm name (FA, SA, HC, GA)
- `config`: Configuration name
- `seed`: Random seed for this run
- `best_fitness`: Final best fitness value
- `history`: Best-so-far trajectory (every 40 iterations)
- `elapsed_time`: Wall-clock time in seconds
- `evaluations`: Total function evaluations
- `final_error`: |f(x) - 0| (same as best_fitness for Rastrigin)

### Knapsack JSON Output

**File naming:** `results/knapsack/n{size}_{type}_seed{instance_seed}_{algo}.json`

Example: `n100_uncorrelated_seed42_FA.json`

```json
{
  "config": {
    "n_items": 100,
    "instance_type": "uncorrelated",
    "instance_seed": 42,
    "capacity": 1250.0,
    "budget": 15000,
    "dp_optimal": 2450.0
  },
  "results": [
    {
      "algorithm": "FA",
      "seed": 0,
      "best_value": 2387.0,
      "best_fitness": -2387.0,
      "total_weight": 1215.5,
      "is_feasible": true,
      "history": [-1200.0, -1500.0, -1800.0, ..., -2387.0],
      "elapsed_time": 3.45,
      "items_selected": 45,
      "capacity_utilization": 0.972,
      "optimality_gap": 2.57
    },
    {
      "algorithm": "FA",
      "seed": 1,
      "best_value": 2401.0,
      ...
    }
  ]
}
```

**Fields:**
- `config`: Instance configuration
  - `n_items`: Number of items
  - `instance_type`: Type of instance
  - `instance_seed`: Seed used to generate this instance
  - `capacity`: Knapsack capacity
  - `budget`: Function evaluation budget
  - `dp_optimal`: Optimal value from DP (null if n>100)
- `results`: Array of 30 runs (or 5 in quick mode)
  - `best_value`: Total value of selected items (negative of fitness)
  - `best_fitness`: Objective value (negated for minimization)
  - `total_weight`: Sum of weights of selected items
  - `is_feasible`: Whether solution satisfies capacity constraint
  - `history`: Best-so-far value trajectory
  - `elapsed_time`: Wall-clock time in seconds
  - `items_selected`: Number of items in knapsack
  - `capacity_utilization`: total_weight / capacity
  - `optimality_gap`: (dp_optimal - best_value) / dp_optimal × 100%

### Summary CSV Files

Auto-generated by `analyze_results.py`:

**`results/rastrigin_summary.csv`:**
```csv
Config,Algorithm,Mean,Std,Median,Best,Worst,Q1,Q3,Success_Rate
quick_convergence,FA,8.45,2.31,7.89,3.21,15.67,6.12,10.23,0.83
quick_convergence,GA,10.12,3.45,9.56,4.23,18.90,7.34,12.67,0.70
quick_convergence,SA,15.23,4.56,14.12,7.89,28.45,11.34,18.67,0.43
quick_convergence,HC,32.45,8.91,30.12,18.34,52.67,25.67,38.90,0.10
multimodal_escape,FA,45.67,12.34,43.21,28.90,78.45,36.78,54.32,0.60
...
```

**Fields:**
- `Mean`: Average final error across runs
- `Std`: Standard deviation
- `Median`: Median final error
- `Best`: Minimum error achieved
- `Worst`: Maximum error
- `Q1, Q3`: First and third quartiles
- `Success_Rate`: Fraction of runs achieving target threshold

**`results/knapsack_summary.csv`:**
```csv
Size,Type,InstanceSeed,Algorithm,Mean_Value,Std_Value,Median_Value,Best_Value,Worst_Value,Mean_Gap,Std_Gap,Median_Gap,Feasibility_Rate,Mean_Utilization
50,uncorrelated,42,FA,2387.5,45.6,2390.0,2430.0,2280.0,2.34,1.12,2.15,1.00,0.975
50,uncorrelated,42,GA,2410.3,38.9,2415.0,2445.0,2320.0,1.56,0.98,1.43,1.00,0.982
50,uncorrelated,42,SA,2245.8,67.3,2250.0,2380.0,2100.0,8.33,2.74,8.16,0.97,0.895
50,uncorrelated,42,HC,2120.5,89.2,2130.0,2290.0,1950.0,13.47,3.64,13.06,0.93,0.848
...
```

**Fields:**
- `Mean_Value`: Average best value found
- `Std_Value`: Standard deviation of values
- `Mean_Gap`: Average optimality gap (%)
- `Median_Gap`: Median optimality gap (%)
- `Feasibility_Rate`: Fraction of runs producing feasible solutions
- `Mean_Utilization`: Average capacity utilization

## 📊 Visualizations

Generated in `results/plots/` following academic metaheuristic benchmarking best practices.

### Rastrigin Visualizations

1. **Convergence Curves** (`rastrigin_{config}_convergence.png`)
   - **X-axis**: Function evaluations (not iterations)
   - **Y-axis**: Error to optimum |f(x) - 0| (log scale)
   - **Lines**: Median trajectory across 30 runs
   - **Shaded bands**: Interquartile range (IQR, 25th-75th percentile)
   - Shows convergence speed fairly across algorithms with different population sizes

2. **Final Error Boxplots** (`rastrigin_{config}_boxplot.png`)
   - Distribution of final errors across all runs
   - Log scale on Y-axis
   - Red diamonds show mean values
   - Shows robustness and presence of outliers

3. **ECDF Plots** (`rastrigin_{config}_ecdf.png`)
   - **NEW**: Empirical Cumulative Distribution Function
   - X-axis: Error value (log scale)
   - Y-axis: P(error ≤ x)
   - Shows what fraction of runs achieved error ≤ x
   - Better than mean/median for understanding tail behavior

4. **Scalability Plot** (`rastrigin_scalability.png`)
   - Mean error vs dimension (d=10, 30, 50)
   - Log scale with error bars (±1 std)
   - Shows which algorithms scale well to higher dimensions

### Knapsack Visualizations

**Per-Instance Plots:**

1. **Convergence Curves** (`knapsack_n{size}_{type}_seed{seed}_convergence.png`)
   - **X-axis**: Function evaluations
   - **Y-axis**: Best value found (higher is better)
   - Median trajectory with IQR bands
   - **Red dashed line**: DP optimal (when available, n≤100)
   - Shows if algorithms can reach optimal or how close they get

2. **Optimality Gap Boxplots** (`knapsack_n{size}_{type}_seed{seed}_gap_boxplot.png`)
   - Distribution of (DP_opt - best_value) / DP_opt × 100%
   - Only generated for n=50, n=100 where DP optimal is available
   - Lower is better (0% = optimal solution)
   - Zero line shows optimal performance

**Aggregate Plots:**

3. **Feasibility Rate Bar Chart** (`knapsack_feasibility.png`)
   - **NEW**: Shows % of runs producing feasible solutions
   - Grouped by n_items (50, 100, 200)
   - Sub-grouped by instance type
   - **Critical metric**: Must be 100% for practical use
   - Algorithms violating constraints are heavily penalized

4. **Capacity Utilization Boxplots** (`knapsack_capacity_utilization.png`)
   - **NEW**: Distribution of weight_used / capacity
   - Grouped by n_items
   - Green line at 1.0 = perfect capacity usage
   - Values > 1.0 indicate constraint violations (infeasible)
   - Shows packing efficiency

5. **Runtime vs Quality Scatter** (`knapsack_runtime_quality.png`)
   - **NEW**: Scatter plot comparing speed and solution quality
   - X-axis: Elapsed time (seconds, log scale)
   - Y-axis: Optimality gap (%, lower is better)
   - Color-coded by algorithm
   - Shows Pareto frontier of fast-and-good algorithms

6. **Scalability Line Plots** (`knapsack_{type}_seed{seed}_scalability.png`)
   - Mean optimality gap vs n_items (50, 100, 200)
   - Generated for uncorrelated and weakly_correlated types
   - Error bars show ±1 std
   - Shows which algorithms scale well to larger instances

## 📈 Performance Metrics (Academic Standards)

### Rastrigin Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Error to Optimum** | \|f(x) - 0\| | Lower is better (0 = perfect) |
| **Convergence Speed** | Evaluations to reach target threshold | Faster is better |
| **Success Rate** | % runs achieving target error | Higher is better (robustness) |
| **ECDF** | P(error ≤ x) | Higher curve = better distribution |

**Target Thresholds:**
- quick_convergence (d=10): 10.0
- multimodal_escape (d=30): 50.0
- scalability (d=50): 100.0

### Knapsack Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Optimality Gap** | (DP_opt - value) / DP_opt × 100% | Lower is better (0% = optimal) |
| **Feasibility Rate** | % feasible solutions | **Must be 100%** |
| **Capacity Utilization** | weight_used / capacity | Higher is better (≤1.0) |
| **Runtime** | Elapsed time (seconds) | Lower is better |
| **Items Selected** | Number of items in knapsack | Context-dependent |

**Key Points:**
- **Infeasible solutions are rejected** (have zero value)
- Optimality gap only meaningful for n≤100 (where DP is feasible)
- For n=200, we compare algorithms relative to each other

### Statistical Tests

The `analyze_results.py` script performs rigorous statistical analysis:

1. **Friedman Test** (non-parametric ANOVA)
   - Tests null hypothesis: all algorithms have equal performance
   - Reports average ranks (lower rank = better performance)
   - p-value < 0.05 indicates significant differences exist

2. **Wilcoxon Signed-Rank Test** (pairwise comparison)
   - Compares each pair of algorithms on matched samples
   - Reports p-values in matrix form
   - p-value < 0.05 indicates significant difference
   - Bonferroni correction applied for multiple comparisons

**Example Output:**
```
Friedman Test Results:
  Chi-square statistic: 45.67
  p-value: 1.23e-09
  Significant differences detected!

Average Ranks (lower is better):
  FA: 1.47
  GA: 2.13
  SA: 2.87
  HC: 3.53

Pairwise Wilcoxon (p-values with Bonferroni correction):
          FA      GA      SA      HC
FA        —   0.0234  0.0001  0.0000
GA   0.0234       —   0.0012  0.0000
SA   0.0001  0.0012       —   0.0345
HC   0.0000  0.0000  0.0345       —
```

## 🎯 Expected Results

### Rastrigin Performance Hypothesis

| Algorithm | d=10 | d=30 | d=50 | Scaling | Key Strengths |
|-----------|------|------|------|---------|---------------|
| **FA** | ✓✓✓ | ✓✓ | ✓ | Good | Fast early convergence, swarm cooperation |
| **GA** | ✓✓ | ✓✓ | ✓✓ | Excellent | Stable across dimensions, genetic diversity |
| **SA** | ✓ | ✗ | ✗ | Poor | Struggles with high-dimensional multimodal |
| **HC** | ✗ | ✗ | ✗ | Poor | Gets trapped in nearest local minimum |

**Legend:** ✓✓✓ = Excellent, ✓✓ = Good, ✓ = Moderate, ✗ = Poor

**Key Findings:**
- **FA** excels in early-stage exploration due to swarm-based search
- **GA** maintains consistent performance across dimensions due to population diversity
- **SA** performance degrades rapidly with dimension (curse of dimensionality)
- **HC** gets trapped in local minima, no global search capability

### Knapsack Performance Hypothesis

| Algorithm | Uncorrelated | Weakly Corr | Strongly Corr | Subset-Sum |
|-----------|--------------|-------------|---------------|------------|
| **FA** | ✓✓ | ✓✓ | ✓✓✓ | ✓ |
| **GA** | ✓✓✓ | ✓✓✓ | ✓✓ | ✓✓ |
| **SA** | ✓ | ✓ | ✓ | ✗ |
| **HC** | ✗ | ✗ | ✗ | ✗ |

**Legend:** ✓✓✓ = <3% gap, ✓✓ = <5% gap, ✓ = <10% gap, ✗ = >10% gap

**Key Findings:**
- **FA/GA** achieve <5% optimality gap for n≤100
- **Strongly correlated** instances favor FA due to better constraint handling
- **Subset-sum** (values=weights) is hardest for all algorithms
- **Repair strategies** are critical for maintaining feasibility
- HC/SA struggle with discrete combinatorial structure

## 🔧 Command-Line Options

### `run_rastrigin.py`

```bash
python benchmark/run_rastrigin.py [OPTIONS]

Options:
  --config {quick_convergence,multimodal_escape,scalability,all}
      Which Rastrigin configuration to run (default: all)
  
  --jobs N
      Number of parallel workers (default: 1)
      Use -1 for all CPU cores, 0 for sequential
  
  --runs N
      Number of independent runs per algorithm (default: 30)
      Use 5 for quick testing
```

### `run_knapsack.py`

```bash
python benchmark/run_knapsack.py [OPTIONS]

Options:
  --size {50,100,200,all}
      Knapsack size to benchmark (default: all)
  
  --jobs N
      Number of parallel workers (default: 1)
      Use -1 for all CPU cores
  
  --runs N
      Number of independent runs per instance (default: 30)
```

### `run_all.py`

```bash
python benchmark/run_all.py [OPTIONS]

Options:
  --quick
      Quick mode: 5 runs per config (for testing)
  
  --full
      Full mode: 30 runs per config (for publication)
  
  --jobs N
      Number of parallel workers (default: 1)
  
  --skip-rastrigin
      Skip Rastrigin benchmarks
  
  --skip-knapsack
      Skip Knapsack benchmarks
```

### `analyze_results.py`

```bash
python benchmark/analyze_results.py [OPTIONS]

Options:
  --problem {rastrigin,knapsack,all}
      Which problem to analyze (default: all)
  
  --output-dir PATH
      Where to save summaries (default: results/summaries/)
```

### `visualize.py`

```bash
python benchmark/visualize.py [OPTIONS]

Options:
  --problem {rastrigin,knapsack,all}
      Which problem to visualize (default: all)
  
  --output-dir PATH
      Where to save plots (default: results/plots/)
  
  --format {png,pdf,svg}
      Plot file format (default: png)
  
  --dpi N
      Plot resolution (default: 300)
```

## 🔧 Troubleshooting

### Import Errors

```bash
# Ensure you're in project root
cd /home/bui-anh-quan/CSTTNT_DA1

# Test imports
python -c "from src.swarm.fa import FireflyContinuousOptimizer"
python -c "from benchmark.config import BenchmarkConfig"
```

### Memory Issues

For large instances (n=200), reduce population size:

```python
# Edit benchmark/config.py
KNAPSACK_CONFIG = {
    'fa_params': {
        'n_fireflies': 40,  # Reduce from 60
        # ...
    },
    'ga_params': {
        'pop_size': 40,  # Reduce from 60
        # ...
    }
}
```

### Slow Execution

**Enable parallel processing:**
```bash
# Use all CPU cores minus 1
python benchmark/run_all.py --full --jobs -1

# Or specify exact number
python benchmark/run_all.py --full --jobs 4
```

**Run specific configs only:**
```bash
# Only test quick convergence
python benchmark/run_rastrigin.py --config quick_convergence --jobs 4

# Only small Knapsack
python benchmark/run_knapsack.py --size 50 --jobs 4
```

### Missing Results

If plots or analysis fail:

```bash
# Check if results exist
ls -lh benchmark/results/rastrigin/
ls -lh benchmark/results/knapsack/

# Re-run analysis
python benchmark/analyze_results.py --problem all

# Re-generate plots
python benchmark/visualize.py --problem all
```

### Plotting Errors

```bash
# Install/update matplotlib
pip install --upgrade matplotlib seaborn

# Check if results have enough data (need ≥2 algorithms)
python -c "import json; print(json.load(open('benchmark/results/rastrigin/quick_convergence/FA_seed0.json')))"
```

## 🧪 Testing

### Integration Tests

Test the entire benchmark pipeline:

```bash
# Run benchmark integration tests
pytest benchmark/test_benchmarks.py -v

# Test with coverage
pytest benchmark/test_benchmarks.py --cov=benchmark --cov-report=term
```

**Test categories:**
- Configuration loading and validation
- Instance generation (all 4 types)
- Parallel execution correctness
- Result file I/O
- Analysis and visualization pipelines

### Smoke Tests

Quick sanity checks:

```bash
# Test Rastrigin with minimal runs
python benchmark/run_rastrigin.py --config quick_convergence --runs 2 --jobs 1

# Test Knapsack with minimal runs
python benchmark/run_knapsack.py --size 50 --runs 2 --jobs 1

# Verify outputs exist
ls benchmark/results/rastrigin/quick_convergence/
ls benchmark/results/knapsack/
```

### Expected Test Execution Time

| Test Type | Duration | Purpose |
|-----------|----------|---------|
| Unit tests (`pytest test/`) | ~5 seconds | Code correctness |
| Integration tests | ~30 seconds | Pipeline correctness |
| Smoke test (2 runs) | ~1 minute | Quick validation |
| Quick mode (5 runs) | ~15-30 minutes | Pre-publication check |
| Full mode (30 runs) | ~2-4 hours | Final results |

## 📝 Deliverables Checklist

**Core Infrastructure:**
- [x] Configuration management (`config.py`)
- [x] Instance generation (`instance_generator.py`)
- [x] Rastrigin benchmark runner (`run_rastrigin.py`)
- [x] Knapsack benchmark runner (`run_knapsack.py`)
- [x] Master execution script (`run_all.py`)
- [x] Shell wrapper (`run_all.sh`)

**Analysis Pipeline:**
- [x] Statistical analysis (`analyze_results.py`)
  - [x] Friedman test implementation
  - [x] Wilcoxon pairwise tests
  - [x] Bonferroni correction
  - [x] Summary CSV generation
- [x] Visualization suite (`visualize.py`)
  - [x] Convergence curves with IQR bands
  - [x] Boxplots with outliers
  - [x] ECDF plots
  - [x] Scalability plots
  - [x] Feasibility rate charts
  - [x] Capacity utilization plots
  - [x] Runtime vs quality scatter plots

**Testing:**
- [x] Unit tests (>80% coverage)
- [x] Integration tests (`test_benchmarks.py`)
- [x] Edge case testing
- [x] Parallel execution tests

**Documentation:**
- [x] Main README (`/README.md`)
- [x] Benchmark README (this file)
- [x] Quick start guide (`/QUICKSTART.md`)
- [x] Test documentation (`/test/README.md`)

**Results Archive:**
- [x] Directory structure (`results/`)
- [x] JSON result format
- [x] CSV summary format
- [x] Plot generation

## 🚦 Workflow Example

**Step 1: Run initial test**
```bash
# Quick sanity check (2 runs, 1 minute)
python benchmark/run_rastrigin.py --config quick_convergence --runs 2
```

**Step 2: Run quick mode**
```bash
# Full pipeline with reduced runs (5 runs, ~30 minutes)
python benchmark/run_all.py --quick --jobs -1
```

**Step 3: Review preliminary results**
```bash
# Generate analysis
python benchmark/analyze_results.py --problem all

# Generate plots
python benchmark/visualize.py --problem all

# Open plots
xdg-open benchmark/results/plots/rastrigin_quick_convergence_convergence.png
xdg-open benchmark/results/plots/knapsack_n50_uncorrelated_seed42_convergence.png
```

**Step 4: Run full benchmark**
```bash
# Publication-ready results (30 runs, ~4 hours)
python benchmark/run_all.py --full --jobs 4
```

**Step 5: Generate final deliverables**
```bash
# Regenerate all analysis and plots
python benchmark/analyze_results.py --problem all
python benchmark/visualize.py --problem all

# Check summary tables
cat benchmark/results/rastrigin_summary.csv
cat benchmark/results/knapsack_summary.csv
```

## 📚 References

1. Yang, X. S. (2008). *Nature-inspired metaheuristic algorithms*. Luniver press.
2. Yang, X. S. (2010). "Firefly algorithm, stochastic test functions and design optimisation". *International Journal of Bio-Inspired Computation*, 2(2), 78-84.
3. [Rastrigin Function - Virtual Library of Simulation Experiments](https://www.sfu.ca/~ssurjano/rastr.html)
4. [Knapsack Problem - Wikipedia](https://en.wikipedia.org/wiki/Knapsack_problem)
5. Pisinger, D. (1995). "An expanding-core algorithm for the exact 0-1 knapsack problem". *European Journal of Operational Research*, 87(1), 175-187.
6. Wilcoxon, F. (1945). "Individual comparisons by ranking methods". *Biometrics Bulletin*, 1(6), 80-83.
7. Friedman, M. (1937). "The use of ranks to avoid the assumption of normality implicit in the analysis of variance". *Journal of the American Statistical Association*, 32(200), 675-701.

## 👥 Contact

For issues or questions, contact: @1234quan1234

---

**Note**: This is an academic benchmark suite for the CSTTNT (Cơ Sở Trí Tuệ Nhân Tạo) course at HCMUS. Execution times and results may vary based on hardware. All random seeds are fixed for reproducibility.