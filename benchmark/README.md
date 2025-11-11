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

**File naming:** `results/rastrigin/rastrigin_{config}_{algo}_{timestamp}.json`

**Structure (NEW):**
```json
{
  "metadata": {
    "problem": "rastrigin",
    "config_name": "quick_convergence",
    "algorithm": "FA",
    "timestamp": "20251110T200402",
    "dimension": 10,
    "budget": 10000,
    "max_iter": 250,
    "pop_size": 40,
    "problem_seed": 42,
    "n_runs": 30,
    "n_successful": 28,
    "n_failed": 2,
    "status_breakdown": {"ok": 28, "timeout": 1, "nan": 1},
    "threshold": 10.0,
    "avg_budget_utilization": 0.998
  },
  "all_results": [
    {
      "algorithm": "FA",
      "seed": 0,
      "algo_seed": 0,
      "problem_seed": 42,
      "best_fitness": 8.4567,
      "history": [45.6, 34.2, 23.1, 15.8, 8.4567],
      "elapsed_time": 2.15,
      "evaluations": 10000,
      "budget": 10000,
      "budget_utilization": 1.0,
      "success": true,
      "hit_evaluations": 8200,
      "status": "ok",
      "error_type": null,
      "error_msg": null
    }
  ]
}
```

**New fields:**
- ⭐ `metadata`: Centralized configuration + status tracking
- ⭐ `status_breakdown`: Count of ok/timeout/nan/numerical_error runs
- ⭐ `hit_evaluations`: When threshold was first achieved (null if never)
- ⭐ `budget_utilization`: Fraction of budget actually used (should be ~1.0)
- ⭐ `all_results`: ALL runs (including failed ones) for reproducibility

### Knapsack JSON Output

**File naming:** `results/knapsack/knapsack_n{size}_{type}_seed{seed}_{algo}_{timestamp}.json`

**Structure (NEW):**
```json
{
  "metadata": {
    "problem": "knapsack",
    "n_items": 50,
    "instance_type": "uncorrelated",
    "instance_seed": 42,
    "algorithm": "FA",
    "timestamp": "20251110T202419",
    "capacity": 500.0,
    "budget": 10000,
    "n_runs": 30,
    "n_successful": 30,
    "n_failed": 0,
    "status_breakdown": {"ok": 30},
    "dp_optimal": 2450.0,
    "avg_gap_%": 2.34,
    "avg_feasibility_rate": 1.0
  },
  "all_results": [
    {
      "algorithm": "FA",
      "seed": 0,
      "algo_seed": 0,
      "best_value": 2387.0,
      "best_fitness": -2387.0,
      "total_weight": 487.5,
      "is_feasible": true,
      "history": [-1200.0, -1500.0, ..., -2387.0],
      "elapsed_time": 3.45,
      "items_selected": 18,
      "capacity_utilization": 0.975,
      "optimality_gap": 2.57,
      "status": "ok",
      "error_type": null,
      "error_msg": null
    }
  ]
}
```

### Summary CSV Files

⭐ **NEW COLUMNS** for advanced statistical analysis:

**`results/rastrigin_summary.csv`:**

Columns added:
- `SR_<=0.1`, `SR_<=0.001`, `SR_<=1e-05`: **Success rates** at tolerance levels
- `HT_med_<=0.1`, `HT_med_<=0.001`, `HT_med_<=1e-05`: **Hitting times** (median evals to reach tolerance)
- `AUC_median`, `AUC_mean`: **Anytime performance** (log integral of convergence trajectory)

Example:
```csv
Configuration,Algorithm,Mean,Std,Median,Best,Worst,Mean_Time,SR_<=0.1,SR_<=0.001,HT_med_<=0.1,AUC_median
quick_convergence,FA,8.45,2.31,7.89,3.21,15.67,2.15,0.83,0.45,245,156.34
quick_convergence,GA,10.12,3.45,9.56,4.23,18.90,2.34,0.70,0.30,412,198.76
```

**`results/knapsack_summary.csv`:**

Columns added:
- `Mean_Norm_Value`, `Std_Norm_Value`: **Normalized** by DP optimal
- `SR_Gap_<=1.0%`, `SR_Gap_<=5.0%`, `SR_Gap_<=10.0%`: **Success rates** at gap thresholds
- `HT_med_<=1%_gap`: **Hitting time** to 1% gap target

Example:
```csv
n_items,type,seed,Algorithm,Mean_Gap_%,Mean_Norm_Value,SR_Gap_<=1.0%,SR_Gap_<=5.0%,HT_med_<=1%_gap
50,uncorrelated,42,FA,2.34,0.9744,0.67,0.93,1250
50,uncorrelated,42,GA,1.56,0.9837,0.87,0.97,892
```

### Global Ranks CSV Files

⭐ **NEW FILES** generated by `generate_*_global_ranks()`:

**`results/rastrigin_global_ranks.csv`:**
```csv
Algorithm,Avg_Rank,N_Configs
FA,1.47,3
GA,2.13,3
SA,2.87,3
HC,3.53,3
```

**`results/knapsack_global_ranks.csv`:**
```csv
Algorithm,Avg_Rank,N_Configs
GA,1.34,36
FA,1.89,36
SA,3.12,36
HC,4.65,36
```

**Purpose:** Aggregate ranking across all configurations — shows which algorithm consistently outperforms others.

## 📊 Visualizations

### NEW: Performance Profile Plots

⭐ **`rastrigin_perf_profile_auc.png`** — Dolan–Moré performance profile
- X-axis: τ (performance ratio factor, 1 to 5)
- Y-axis: Fraction of instances within τ × best
- Shows robustness across all configurations
- Algorithm reaching 100% at smallest τ is most efficient

⭐ **`knapsack_perf_profile_gap.png`** — Similar, using optimality gap

### NEW: Success Rate Plots

⭐ **`rastrigin_success_rates.png`** — Grouped bar chart
- X-axis: Algorithm
- Y-axis: Success rate (%)
- Grouped by tolerance level (1e-1, 1e-3, 1e-5)
- Shows robustness at different accuracy requirements

⭐ **`knapsack_success_rates.png`** — Similar for gap thresholds (1%, 5%, 10%)

### NEW: Hitting Time Plots

⭐ **`rastrigin_hitting_times.png`** — Grouped bar chart
- X-axis: Algorithm
- Y-axis: Median evaluations to reach tolerance (log scale)
- Grouped by configuration (quick_convergence, multimodal_escape, scalability)
- Shows convergence speed

### NEW: Global Rank Plots

⭐ **`rastrigin_global_ranks.png`** — Bar chart
- X-axis: Algorithm
- Y-axis: Average rank (lower is better)
- Shows best algorithm overall (smallest rank)

⭐ **`knapsack_global_ranks.png`** — Similar for Knapsack

### IMPROVED: Runtime-Quality Scatter

⭐ **`knapsack_runtime_quality_improved.png`** — Enhanced bubble scatter
- X-axis: Mean runtime (seconds, log scale)
- Y-axis: Quality gap (%)
- Bubble size: Feasibility rate (larger = more feasible)
- Now includes feasibility visualization

## 📈 Advanced Metrics (NEW)

### Success Rate & Hitting Time

**Definition:**
- **Success Rate (%)**: Fraction of runs achieving error ≤ tolerance
  - For Rastrigin: Tests at tol = 0.1, 0.001, 1e-5
  - For Knapsack: Tests at gap = 1%, 5%, 10%
- **Hitting Time**: Evaluations until first success
  - Reports median across successful runs
  - If no run succeeds, reports NaN

**Interpretation:**
- High success rate = robust algorithm (works consistently)
- Low hitting time = fast convergence
- Combination = ideal: robust AND fast

### Anytime Performance (AUC)

⭐ **NEW for Rastrigin**: Integrated area under log convergence curve
- Formula: AUC = ∫ log(1 + |f(x)|) dx (normalized by budget)
- Lower is better (steeper convergence)
- Captures **entire trajectory**, not just final error
- More informative than simple mean/median