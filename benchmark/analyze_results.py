"""
Analyze and compare benchmark results.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats


def load_results(result_dir, algorithm_names=None):
    """Load results from JSON files."""
    if algorithm_names is None:
        algorithm_names = ['FA', 'SA', 'HC', 'GA']
    
    results = {}
    for algo in algorithm_names:
        result_file = Path(result_dir) / f'{algo}_results.json'
        if result_file.exists():
            with open(result_file, 'r') as f:
                results[algo] = json.load(f)
    
    return results


def compute_statistics(results):
    """Compute statistics for each algorithm."""
    stats_dict = {}
    
    for algo, runs in results.items():
        best_fits = [r['best_fitness'] for r in runs]
        times = [r['elapsed_time'] for r in runs]
        
        stats_dict[algo] = {
            'mean': np.mean(best_fits),
            'std': np.std(best_fits),
            'median': np.median(best_fits),
            'min': np.min(best_fits),
            'max': np.max(best_fits),
            'q25': np.percentile(best_fits, 25),
            'q75': np.percentile(best_fits, 75),
            'mean_time': np.mean(times),
            'std_time': np.std(times)
        }
    
    return stats_dict


def statistical_comparison(results, alpha=0.05):
    """Perform statistical tests comparing algorithms."""
    algorithms = list(results.keys())
    n_algos = len(algorithms)
    
    # Extract best fitness values
    fitness_data = {algo: [r['best_fitness'] for r in runs] 
                    for algo, runs in results.items()}
    
    # Pairwise Mann-Whitney U tests
    comparison_matrix = np.zeros((n_algos, n_algos))
    p_values = {}
    
    for i, algo1 in enumerate(algorithms):
        for j, algo2 in enumerate(algorithms):
            if i < j:
                statistic, p_value = stats.mannwhitneyu(
                    fitness_data[algo1], 
                    fitness_data[algo2],
                    alternative='two-sided'
                )
                p_values[f'{algo1}_vs_{algo2}'] = p_value
                comparison_matrix[i, j] = p_value
                comparison_matrix[j, i] = p_value
    
    return comparison_matrix, p_values


def print_summary_table(stats_dict):
    """Print a summary table of results."""
    df = pd.DataFrame(stats_dict).T
    
    print("\n" + "=" * 80)
    print("STATISTICAL SUMMARY")
    print("=" * 80)
    print(df.to_string())
    print("=" * 80)


def print_comparison_results(p_values, alpha=0.05):
    """Print statistical comparison results."""
    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISON (Mann-Whitney U test)")
    print("=" * 80)
    
    for comparison, p_value in p_values.items():
        significance = "SIGNIFICANT" if p_value < alpha else "NOT significant"
        print(f"{comparison}: p-value = {p_value:.4f} ({significance})")
    
    print("=" * 80)


def analyze_benchmark(result_dir, config_name):
    """Analyze results from a benchmark run."""
    print(f"\n{'=' * 80}")
    print(f"ANALYZING: {config_name}")
    print(f"{'=' * 80}")
    
    result_path = Path(result_dir) / config_name
    
    if not result_path.exists():
        print(f"Error: Results not found at {result_path}")
        return
    
    # Load results
    results = load_results(result_path)
    
    if not results:
        print("Error: No results found")
        return
    
    # Compute statistics
    stats_dict = compute_statistics(results)
    print_summary_table(stats_dict)
    
    # Statistical comparison
    if len(results) > 1:
        _, p_values = statistical_comparison(results)
        print_comparison_results(p_values)
    
    # Find best algorithm
    best_algo = min(stats_dict.items(), key=lambda x: x[1]['mean'])[0]
    print(f"\nBest algorithm (by mean): {best_algo}")
    print(f"Mean fitness: {stats_dict[best_algo]['mean']:.6f}")


def analyze_all_benchmarks(base_dir='benchmark/results'):
    """Analyze all benchmark results."""
    base_path = Path(base_dir)
    
    print("=" * 80)
    print("ANALYZING ALL BENCHMARKS")
    print("=" * 80)
    
    # Analyze Rastrigin
    rastrigin_dir = base_path / 'rastrigin'
    if rastrigin_dir.exists():
        for config_dir in rastrigin_dir.iterdir():
            if config_dir.is_dir():
                analyze_benchmark(rastrigin_dir, config_dir.name)
    
    # Analyze TSP
    tsp_dir = base_path / 'tsp'
    if tsp_dir.exists():
        for config_dir in tsp_dir.iterdir():
            if config_dir.is_dir():
                analyze_benchmark(tsp_dir, config_dir.name)
    
    # Analyze Knapsack
    knapsack_dir = base_path / 'knapsack'
    if knapsack_dir.exists():
        for config_dir in knapsack_dir.iterdir():
            if config_dir.is_dir():
                analyze_benchmark(knapsack_dir, config_dir.name)


def load_rastrigin_results(results_dir: str, config_name: str):
    """Load Rastrigin results for a specific configuration."""
    config_path = Path(results_dir) / config_name
    
    results = {}
    for algo_file in config_path.glob('*_results.json'):
        algo_name = algo_file.stem.replace('_results', '')
        with open(algo_file, 'r') as f:
            data = json.load(f)
            results[algo_name] = [r['best_fitness'] for r in data]
    
    return results


def friedman_test(results):
    """Perform Friedman test on results."""
    data = [results[algo] for algo in sorted(results.keys())]
    statistic, p_value = stats.friedmanchisquare(*data)
    return statistic, p_value


def wilcoxon_test(data1, data2):
    """Perform Wilcoxon signed-rank test."""
    statistic, p_value = stats.wilcoxon(data1, data2)
    return statistic, p_value


def compute_ranks(results):
    """Compute average ranks across runs."""
    algorithms = sorted(results.keys())
    n_runs = len(results[algorithms[0]])
    
    ranks = {algo: 0 for algo in algorithms}
    
    for run_idx in range(n_runs):
        run_values = [(algo, results[algo][run_idx]) for algo in algorithms]
        run_values.sort(key=lambda x: x[1])
        
        for rank, (algo, _) in enumerate(run_values, 1):
            ranks[algo] += rank
    
    # Average ranks
    for algo in algorithms:
        ranks[algo] /= n_runs
    
    return ranks


def generate_rastrigin_summary(results_dir: str, output_file: str):
    """Generate summary CSV for Rastrigin results."""
    
    configs = ['quick_convergence', 'multimodal_escape', 'scalability']
    summary_data = []
    
    for config_name in configs:
        results = load_rastrigin_results(results_dir, config_name)
        
        for algo, fitness_values in results.items():
            summary_data.append({
                'Configuration': config_name,
                'Algorithm': algo,
                'Mean': np.mean(fitness_values),
                'Std': np.std(fitness_values),
                'Min': np.min(fitness_values),
                'Max': np.max(fitness_values),
                'Median': np.median(fitness_values)
            })
    
    df = pd.DataFrame(summary_data)
    df = df.sort_values(['Configuration', 'Algorithm'])
    df.to_csv(output_file, index=False)
    
    print(f"\nRastrigin summary saved to: {output_file}")
    print("\n" + "=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    
    return df


def generate_knapsack_summary(results_dir: str, output_file: str):
    """Generate summary CSV for Knapsack results."""
    
    summary_data = []
    
    for config_dir in Path(results_dir).iterdir():
        if not config_dir.is_dir():
            continue
        
        # Parse config name (e.g., "n50_uncorrelated_seed42")
        parts = config_dir.name.split('_')
        n_items = int(parts[0][1:])
        knapsack_type = parts[1]
        seed = int(parts[2].replace('seed', ''))
        
        for algo_file in config_dir.glob('*_results.json'):
            algo_name = algo_file.stem.replace('_results', '')
            
            with open(algo_file, 'r') as f:
                data = json.load(f)
            
            best_values = [r['best_fitness'] for r in data]
            times = [r['elapsed_time'] for r in data]
            
            row = {
                'n_items': n_items,
                'type': knapsack_type,
                'seed': seed,
                'Algorithm': algo_name,
                'Mean_Value': np.mean(best_values),
                'Std_Value': np.std(best_values),
                'Max_Value': np.max(best_values),
                'Min_Value': np.min(best_values),
                'Mean_Time': np.mean(times),
                'Std_Time': np.std(times)
            }
            
            # Calculate gap if optimal is available
            gaps = []
            for r in data:
                if 'gap_percent' in r:
                    gaps.append(r['gap_percent'])
            
            if gaps:
                row['Avg_Gap_%'] = np.mean(gaps)
                row['Std_Gap_%'] = np.std(gaps)
            
            summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    df = df.sort_values(['n_items', 'type', 'seed', 'Algorithm'])
    df.to_csv(output_file, index=False)
    
    print(f"\nKnapsack summary saved to: {output_file}")
    print("\n" + "=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    
    return df


def perform_statistical_tests(results_dir: str, problem: str = 'rastrigin'):
    """Perform statistical tests and print results."""
    
    print(f"\n{'=' * 80}")
    print(f"STATISTICAL ANALYSIS: {problem.upper()}")
    print(f"{'=' * 80}")
    
    if problem == 'rastrigin':
        configs = ['quick_convergence', 'multimodal_escape', 'scalability']
        
        for config_name in configs:
            print(f"\n{'-' * 80}")
            print(f"Configuration: {config_name}")
            print(f"{'-' * 80}")
            
            results = load_rastrigin_results(results_dir, config_name)
            
            if len(results) < 2:
                print("Not enough algorithms for comparison")
                continue
            
            # Friedman test
            friedman_stat, friedman_p = friedman_test(results)
            print(f"\nFriedman Test:")
            print(f"  Statistic: {friedman_stat:.4f}")
            print(f"  P-value: {friedman_p:.4e}")
            print(f"  Significant: {'Yes' if friedman_p < 0.05 else 'No'}")
            
            # Average ranks
            ranks = compute_ranks(results)
            print(f"\nAverage Ranks (lower is better):")
            for algo, rank in sorted(ranks.items(), key=lambda x: x[1]):
                print(f"  {algo}: {rank:.2f}")
            
            # Pairwise Wilcoxon tests
            print(f"\nPairwise Wilcoxon Tests (p-values):")
            algorithms = sorted(results.keys())
            
            # Create matrix
            print(f"{'':>8}", end='')
            for algo in algorithms:
                print(f"{algo:>8}", end='')
            print()
            
            for algo1 in algorithms:
                print(f"{algo1:>8}", end='')
                for algo2 in algorithms:
                    if algo1 == algo2:
                        print(f"{'—':>8}", end='')
                    else:
                        _, p_val = wilcoxon_test(results[algo1], results[algo2])
                        print(f"{p_val:>8.4f}", end='')
                print()


def main():
    """Main analysis pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze benchmark results')
    parser.add_argument('--problem', type=str, choices=['rastrigin', 'knapsack', 'all'],
                        default='all', help='Problem to analyze')
    parser.add_argument('--rastrigin-dir', type=str,
                        default='benchmark/results/rastrigin',
                        help='Rastrigin results directory')
    parser.add_argument('--knapsack-dir', type=str,
                        default='benchmark/results/knapsack',
                        help='Knapsack results directory')
    parser.add_argument('--output-dir', type=str,
                        default='benchmark/results',
                        help='Output directory for summaries')
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if args.problem in ['rastrigin', 'all']:
        if Path(args.rastrigin_dir).exists():
            # Generate summary
            generate_rastrigin_summary(
                args.rastrigin_dir,
                str(output_path / 'rastrigin_summary.csv')
            )
            
            # Statistical tests
            perform_statistical_tests(args.rastrigin_dir, 'rastrigin')
        else:
            print(f"Rastrigin results not found at: {args.rastrigin_dir}")
    
    if args.problem in ['knapsack', 'all']:
        if Path(args.knapsack_dir).exists():
            # Generate summary
            generate_knapsack_summary(
                args.knapsack_dir,
                str(output_path / 'knapsack_summary.csv')
            )
        else:
            print(f"Knapsack results not found at: {args.knapsack_dir}")
    
    print(f"\n{'=' * 80}")
    print("Analysis complete!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()