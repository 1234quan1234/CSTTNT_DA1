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
├── analyze_results.py        # Statistical analysis
├── visualize.py              # Generate plots
├── run_all.sh               # Master script to run everything
├── results/                  # Output directory
│   ├── rastrigin/
│   │   ├── quick_convergence/
│   │   ├── multimodal_escape/
│   │   └── scalability/
│   ├── knapsack/
│   ├── plots/               # Generated visualizations
│   ├── rastrigin_summary.csv
│   └── knapsack_summary.csv
└── README.md                 # This file
```

## 🚀 Quick Start

### Option 1: Run Everything (Full Benchmark)

```bash
# This will run all experiments (4-7 hours)
chmod +x benchmark/run_all.sh
./benchmark/run_all.sh
```

### Option 2: Run Individual Benchmarks

#### Rastrigin Benchmark

```bash
# Quick test (d=10, ~5 minutes)
python benchmark/run_rastrigin.py --config quick_convergence

# Multimodal escape (d=30, ~15 minutes)
python benchmark/run_rastrigin.py --config multimodal_escape

# Scalability (d=50, ~30 minutes)
python benchmark/run_rastrigin.py --config scalability
```

#### Knapsack Benchmark

```bash
# Small scale (n=50, ~30 minutes)
python benchmark/run_knapsack.py --size 50

# Medium scale (n=100, ~1 hour)
python benchmark/run_knapsack.py --size 100

# Large scale (n=200, ~2 hours)
python benchmark/run_knapsack.py --size 200

# Stress test (n=500, ~3 hours)
python benchmark/run_knapsack.py --size 500

# Run all
python benchmark/run_knapsack.py --size all
```

### Option 3: Analysis Only

If you already have results:

```bash
# Analyze results and generate summary tables
python benchmark/analyze_results.py --problem all

# Generate visualizations
python benchmark/visualize.py
```

## 📊 Configurations

### Rastrigin Configurations

| Config Name | Dimension | Budget (evals) | Max Iter | Threshold | Purpose |
|-------------|-----------|----------------|----------|-----------|---------|
| quick_convergence | 10 | 5,000 | 125 | 10.0 | Fast convergence test |
| multimodal_escape | 30 | 20,000 | 500 | 50.0 | Escape local minima |
| scalability | 50 | 40,000 | 1,000 | 100.0 | High-dimensional scaling |

**Algorithm Parameters:**
- **FA**: n_fireflies=40, α=0.3, β₀=1.0, γ=1.0
- **SA**: T₀=100, cooling=0.95, step=0.5
- **HC**: neighbors=20, step=0.5, restart=50
- **GA**: pop=40, crossover=0.8, mutation=0.1

### Knapsack Configurations

| n Items | Types | Seeds | Budget (evals) | Max Iter FA/GA | Max Iter SA/HC | DP Optimal? |
|---------|-------|-------|----------------|----------------|----------------|-------------|
| 50 | All 4 | 42, 43 | 10,000 | 166 | 10,000 | ✓ Yes |
| 100 | All 4 | 42, 43 | 15,000 | 250 | 15,000 | ✓ Yes |
| 200 | Uncorr, Weak | 42, 43 | 30,000 | 500 | 30,000 | ✗ No |
| 500 | Uncorr only | 42, 43 | 50,000 | 833 | 50,000 | ✗ No |

**Instance Types:**
1. **Uncorrelated**: Random values and weights
2. **Weakly Correlated**: values ≈ weights ± random noise
3. **Strongly Correlated**: values = weights + 100
4. **Subset-Sum**: values = weights (hardest)

**Total Instances:** 22 (8×n=50 + 8×n=100 + 4×n=200 + 2×n=500)

**Algorithm Parameters:**
- **FA**: n_fireflies=60, α_flip=0.2, max_flips=3
- **SA**: T₀=1000, cooling=0.95
- **HC**: neighbors=20, restart=100
- **GA**: pop=60, crossover=0.8, mutation=1/n

## 📈 Metrics

### Primary Metrics (All Problems)

| Metric | Description | Purpose |
|--------|-------------|---------|
| **Best Fitness** | Final best solution quality | Main performance indicator |
| **Mean ± Std** | Average over 30 runs | Robustness measure |
| **Median** | Median performance | Robust central tendency |
| **Success Rate** | % achieving threshold | Reliability measure |
| **Convergence Curve** | Best-so-far trajectory | Convergence speed |
| **Time per Eval** | Execution time | Computational cost |

### Rastrigin-Specific Metrics

- **Error to Optimum**: |f(x) - 0|
- **ERT**: Expected running time to threshold
- **Population Diversity**: Average pairwise distance (FA/GA)

### Knapsack-Specific Metrics

- **Optimality Gap**: (DP_opt - best_value) / DP_opt × 100% (for n≤100)
- **Feasibility Rate**: % valid solutions (weight ≤ capacity)
- **Capacity Utilization**: Average Σ(w×x) / C
- **Items Selected**: Average Σ(x)
- **Constraint Violations**: % infeasible before repair

## 🔬 Statistical Tests

### Pairwise Comparisons
- **Wilcoxon Signed-Rank Test**: Compare two algorithms
- **Significance Level**: α = 0.05
- **Null Hypothesis**: No difference in median performance

### Multiple Comparisons
- **Friedman Test**: Omnibus test for k algorithms
- **Nemenyi Post-hoc**: Pairwise comparisons after Friedman
- **Average Ranks**: Lower rank = better performance

### Example Output

```
Friedman Test:
  Statistic: 45.67
  P-value: 1.23e-10
  Significant: Yes

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

## 📦 Output Format

### Per-Run JSON (Rastrigin)

```json
{
  "algorithm": "FA",
  "seed": 0,
  "best_fitness": 12.3456,
  "history": [45.6, 34.2, 23.1, ...],
  "elapsed_time": 2.15,
  "evaluations": 5000
}
```

### Per-Run JSON (Knapsack)

```json
{
  "config": {
    "n_items": 50,
    "instance_type": "uncorrelated",
    "instance_seed": 42,
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
      "capacity": 500.0,
      "is_feasible": true,
      "history": [1200.0, 1500.0, ...],
      "elapsed_time": 3.45,
      "items_selected": 18,
      "capacity_utilization": 0.975,
      "optimality_gap": 2.57
    }
  ]
}
```

### Summary CSV (Auto-generated)

```csv
Config,Algorithm,Mean,Std,Median,Best,Worst,Q1,Q3
quick_convergence,FA,8.45,2.31,7.89,3.21,15.67,6.12,10.23
quick_convergence,SA,15.23,4.56,14.12,7.89,28.45,11.34,18.67
...
```

## 📊 Visualizations

Generated in `benchmark/results/plots/`:

### Rastrigin Plots

1. **Convergence Curves** (`*_convergence.png`)
   - Median best-so-far with IQR bands
   - All 4 algorithms on same plot
   - Shows convergence speed

2. **Boxplots** (`*_boxplot.png`)
   - Final fitness distributions
   - Shows robustness and outliers
   - Includes mean markers

3. **Bar Charts** (`*_bars.png`)
   - Mean ± std comparisons
   - Error bars for uncertainty
   - Value labels on bars

4. **Heatmaps** (`*_heatmap.png`)
   - Win-loss matrix (row vs column)
   - Color-coded performance
   - Pairwise comparisons

5. **Scalability Plot** (`scalability.png`)
   - Performance vs dimension (d=10/30/50)
   - Log scale for better visualization
   - Error bars for uncertainty

### Knapsack Plots (TODO)

Similar visualizations for Knapsack results.

## ⏱️ Estimated Runtime

### Sequential Execution

| Benchmark | Time |
|-----------|------|
| Rastrigin (all configs) | ~30-45 min |
| Knapsack (n=50) | ~30 min |
| Knapsack (n=100) | ~1 hour |
| Knapsack (n=200) | ~2 hours |
| Knapsack (n=500) | ~3 hours |
| **Total** | **~6-7 hours** |

### Parallel Execution (4 cores)

| Benchmark | Time |
|-----------|------|
| Rastrigin | ~10-15 min |
| Knapsack | ~2-3 hours |
| **Total** | **~3-4 hours** |

## 🎯 Expected Results

### Rastrigin Hypothesis

| Algorithm | d=10 | d=30 | d=50 | Scaling |
|-----------|------|------|------|---------|
| **FA** | ✓✓✓ Best | ✓✓ Good | ✓ Moderate | Good |
| **GA** | ✓✓ Good | ✓✓ Good | ✓✓ Good | Excellent |
| **SA** | ✓ Decent | ✗ Poor | ✗ Poor | Poor |
| **HC** | ✗ Worst | ✗ Worst | ✗ Worst | Poor |

**Key Findings:**
- FA excels in early convergence due to swarm cooperation
- GA maintains stable performance across dimensions
- SA struggles with high-dimensional multimodal landscapes
- HC gets trapped in nearest local minimum

### Knapsack Hypothesis

| Algorithm | Uncorrelated | Weakly Corr | Strongly Corr | Subset-Sum |
|-----------|--------------|-------------|---------------|------------|
| **FA** | ✓✓ Good | ✓✓ Good | ✓✓✓ Best | ✓ Moderate |
| **GA** | ✓✓✓ Best | ✓✓✓ Best | ✓✓ Good | ✓✓ Good |
| **SA** | ✓ Decent | ✓ Decent | ✓ Decent | ✗ Poor |
| **HC** | ✗ Poor | ✗ Poor | ✗ Poor | ✗ Worst |

**Key Findings:**
- FA/GA achieve <5% optimality gap for n≤100
- Strongly correlated instances favor swarm intelligence
- Subset-sum is hardest for all algorithms
- Repair strategy is critical for constraint satisfaction

## 🔧 Troubleshooting

### Import Errors

```bash
# Make sure you're in project root
cd /home/bui-anh-quan/CSTTNT_DA1

# Test imports
python -c "from src.swarm.fa import FireflyContinuousOptimizer"
```

### Memory Issues

For large instances (n=500), reduce population:

```python
# In config.py
config.fa_params['n_fireflies'] = 40  # Instead of 60
config.ga_params['pop_size'] = 40
```

### Slow Execution

Enable parallel processing:

```bash
# Run multiple configs in parallel
python benchmark/run_rastrigin.py --config quick_convergence &
python benchmark/run_rastrigin.py --config multimodal_escape &
python benchmark/run_rastrigin.py --config scalability &
wait
```

### Missing Dependencies

```bash
pip install numpy scipy pandas matplotlib seaborn
```

## 📝 Deliverables Checklist

- [x] Benchmark infrastructure
- [x] Rastrigin benchmark script
- [x] Knapsack benchmark script
- [x] Instance generators
- [x] Statistical analysis tools
- [x] Visualization suite
- [ ] Master execution script (`run_all.sh`)
- [ ] Technical report template
- [ ] Results archive structure

## 🚦 Next Steps

1. **Run small-scale test** (5 runs instead of 30):
   ```bash
   # Modify config.py: change seeds to range(5)
   python benchmark/run_rastrigin.py --config quick_convergence
   ```

2. **Verify outputs**:
   - Check JSON files in `benchmark/results/`
   - Run analysis: `python benchmark/analyze_results.py`
   - Generate plots: `python benchmark/visualize.py`

3. **Run full benchmark**:
   ```bash
   ./benchmark/run_all.sh
   ```

4. **Write technical report** using generated tables and plots

5. **Archive results** for reproducibility

## 📚 References

1. Yang, X. S. (2008). *Nature-inspired metaheuristic algorithms*. Luniver press.
2. [Rastrigin Function - Virtual Library](https://www.sfu.ca/~ssurjano/rastr.html)
3. [Knapsack Problem - Wikipedia](https://en.wikipedia.org/wiki/Knapsack_problem)
4. Wilcoxon, F. (1945). "Individual comparisons by ranking methods". *Biometrics Bulletin*.
5. Friedman, M. (1937). "The use of ranks to avoid the assumption of normality". *JASA*.

## 👥 Contact

For issues or questions, contact: @1234quan1234

---

**Note**: This is an academic benchmark suite. Execution times and results may vary based on hardware. All random seeds are fixed for reproducibility.

