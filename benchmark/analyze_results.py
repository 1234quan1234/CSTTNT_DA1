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
from typing import Dict, List, Tuple

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


def get_rastrigin_raw_data(results_dir: str, config_name: str) -> Dict:
    """Extract raw per-run data for Rastrigin."""
    config_path = Path(results_dir) / config_name
    
    if not config_path.exists():
        return {}
    
    raw_data = {}
    for algo in ['FA', 'SA', 'HC', 'GA']:
        algo_file = config_path / f'{algo}_results.json'
        if not algo_file.exists():
            continue
        
        with open(algo_file, 'r') as f:
            data = json.load(f)
        
        # Data is a list of runs
        if not isinstance(data, list) or len(data) == 0:
            continue
        
        raw_data[algo] = {
            'runs': data,
            'best_fitness': np.array([r['best_fitness'] for r in data]),
            'error_to_optimum': np.array([abs(r['best_fitness'] - 0.0) for r in data]),
            'histories': [r['history'] for r in data],
            'elapsed_times': np.array([r['elapsed_time'] for r in data]),
            'n_evals': np.array([r.get('evaluations', len(r['history'])) for r in data])
        }
    
    return raw_data


def get_knapsack_raw_data(results_dir: str, n_items: int, instance_type: str,
                         instance_seed: int) -> Dict:
    """Extract raw per-run data for Knapsack."""
    results_path = Path(results_dir)
    
    # Pattern để tìm tất cả file của config này
    pattern = f"n{n_items}_{instance_type}_seed{instance_seed}_*.json"
    json_files = list(results_path.glob(pattern))
    
    if not json_files:
        print(f"Warning: No files found matching pattern: {pattern}")
        return {}
    
    raw_data = {
        'dp_optimal': None,
        'config': {}
    }
    
    # Load từng file algorithm
    for json_file in json_files:
        # Parse algorithm name từ filename: n50_uncorrelated_seed42_FA.json -> FA
        algo = json_file.stem.rsplit('_', 1)[-1]  # Lấy phần cuối sau dấu _ cuối cùng
        
        if algo not in ['FA', 'SA', 'HC', 'GA']:
            continue
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Validate structure
            if not isinstance(data, dict):
                print(f"Warning: Unexpected format in {json_file.name}, expected dict")
                continue
            
            # Lấy config và dp_optimal từ file đầu tiên
            if not raw_data['config']:
                raw_data['config'] = data.get('config', {})
                raw_data['dp_optimal'] = data.get('config', {}).get('dp_optimal', None)
            
            results_list = data.get('results', [])
            
            if not isinstance(results_list, list):
                print(f"Warning: 'results' is not a list in {json_file.name}")
                continue
            
            # Initialize algo data structure
            raw_data[algo] = {
                'runs': [],
                'best_values': [],
                'optimality_gaps': [],
                'histories': [],
                'feasibility': [],
                'capacity_utilization': [],
                'elapsed_times': [],
                'n_evals': []
            }
            
            # Process all runs for this algorithm
            for result in results_list:
                raw_data[algo]['runs'].append(result)
                raw_data[algo]['best_values'].append(result['best_value'])
                raw_data[algo]['histories'].append(result['history'])
                raw_data[algo]['feasibility'].append(1.0 if result['is_feasible'] else 0.0)
                raw_data[algo]['capacity_utilization'].append(result.get('capacity_utilization', 0.0))
                raw_data[algo]['elapsed_times'].append(result['elapsed_time'])
                raw_data[algo]['n_evals'].append(len(result['history']))
                
                # Calculate optimality gap only for feasible solutions
                if raw_data['dp_optimal'] is not None and result['is_feasible']:
                    gap = (raw_data['dp_optimal'] - result['best_value']) / raw_data['dp_optimal'] * 100
                    raw_data[algo]['optimality_gaps'].append(gap)
            
            # Convert lists to numpy arrays
            raw_data[algo]['best_values'] = np.array(raw_data[algo]['best_values'])
            raw_data[algo]['feasibility'] = np.array(raw_data[algo]['feasibility'])
            raw_data[algo]['capacity_utilization'] = np.array(raw_data[algo]['capacity_utilization'])
            raw_data[algo]['elapsed_times'] = np.array(raw_data[algo]['elapsed_times'])
            raw_data[algo]['n_evals'] = np.array(raw_data[algo]['n_evals'])
            
            if raw_data[algo]['optimality_gaps']:
                raw_data[algo]['optimality_gaps'] = np.array(raw_data[algo]['optimality_gaps'])
            else:
                raw_data[algo]['optimality_gaps'] = np.array([])
                
        except Exception as e:
            print(f"Warning: Error loading {json_file.name}: {e}")
            continue
    
    return raw_datac


def generate_rastrigin_summary(results_dir: str, output_file: str):
    """Generate summary CSV for Rastrigin results."""
    
    configs = ['quick_convergence', 'multimodal_escape', 'scalability']
    summary_data = []
    
    for config_name in configs:
        try:
            raw_data = get_rastrigin_raw_data(results_dir, config_name)
            
            for algo, data in raw_data.items():
                errors = data['error_to_optimum']
                evals = data['n_evals']
                times = data['elapsed_times']
                
                summary_data.append({
                    'Configuration': config_name,
                    'Algorithm': algo,
                    'Mean': np.mean(errors),
                    'Std': np.std(errors),
                    'Median': np.median(errors),
                    'Best': np.min(errors),
                    'Worst': np.max(errors),
                    'Q1': np.percentile(errors, 25),
                    'Q3': np.percentile(errors, 75),
                    'Mean_Evals': np.mean(evals),
                    'Mean_Time': np.mean(times)
                })
        except FileNotFoundError:
            print(f"Warning: Results not found for configuration '{config_name}'")
            continue
    
    if not summary_data:
        print("Warning: No data found for summary")
        return None
    
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
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Warning: Results directory not found: {results_dir}")
        return None
    
    # Find all JSON result files
    json_files = list(results_path.glob('n*_*.json'))
    
    if not json_files:
        print(f"Warning: No JSON files found in {results_dir}")
        return None
    
    # Group files by configuration (n_items, instance_type, seed)
    configs = {}
    for json_file in json_files:
        # Parse filename: n50_uncorrelated_seed42_FA.json
        filename = json_file.stem
        
        try:
            # Remove algorithm suffix
            parts = filename.rsplit('_', 1)  # Split from right
            config_part = parts[0]  # n50_uncorrelated_seed42
            algo = parts[1]  # FA, GA, HC, SA
            
            # Parse config: n50_uncorrelated_seed42
            config_parts = config_part.split('_')
            n_items = int(config_parts[0][1:])  # n50 -> 50
            
            # Find seed (last part starting with 'seed')
            seed_idx = next(i for i, p in enumerate(config_parts) if p.startswith('seed'))
            seed = int(config_parts[seed_idx].replace('seed', ''))
            
            # Instance type is everything between n_items and seed
            instance_type = '_'.join(config_parts[1:seed_idx])
            
            config_key = (n_items, instance_type, seed)
            if config_key not in configs:
                configs[config_key] = {}
            configs[config_key][algo] = json_file
            
        except (ValueError, IndexError, StopIteration) as e:
            print(f"Warning: Could not parse filename {filename}: {e}")
            continue
    
    # Process each configuration
    for (n_items, instance_type, seed), algo_files in configs.items():
        # Load one file to get config and dp_optimal
        first_file = next(iter(algo_files.values()))
        
        with open(first_file, 'r') as f:
            first_data = json.load(f)
        
        dp_optimal = first_data.get('dp_optimal')
        
        # Process each algorithm
        for algo in ['FA', 'SA', 'HC', 'GA']:
            if algo not in algo_files:
                continue
            
            algo_file = algo_files[algo]
            
            try:
                with open(algo_file, 'r') as f:
                    data = json.load(f)
                
                if not isinstance(data, dict):
                    continue
                
                best_value = data.get('best_value')
                best_values_history = data.get('history', [])
                is_feasible = data.get('is_feasible', False)
                capacity_util = data.get('capacity_utilization', 0.0)
                elapsed_time = data.get('elapsed_time', 0.0)
                
                row = {
                    'n_items': n_items,
                    'type': instance_type,
                    'seed': seed,
                    'Algorithm': algo,
                    'Best_Value': best_value,
                    'Feasible': 'Yes' if is_feasible else 'No',
                    'Capacity_Util': capacity_util,
                    'Elapsed_Time': elapsed_time,
                    'N_Evals': len(best_values_history)
                }
                
                # Add optimality gap if available
                if dp_optimal is not None and is_feasible:
                    gap = (dp_optimal - best_value) / dp_optimal * 100
                    row['DP_Optimal'] = dp_optimal
                    row['Gap_%'] = gap
                
                summary_data.append(row)
                
            except Exception as e:
                print(f"Warning: Error loading {algo_file}: {e}")
                continue
    
    if not summary_data:
        print("Warning: No valid data found for summary")
        return None
    
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
            try:
                # Generate summary
                generate_rastrigin_summary(
                    args.rastrigin_dir,
                    str(output_path / 'rastrigin_summary.csv')
                )
                
                # Statistical tests
                perform_statistical_tests(args.rastrigin_dir, 'rastrigin')
            except Exception as e:
                print(f"Error analyzing Rastrigin results: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Rastrigin results not found at: {args.rastrigin_dir}")
    
    if args.problem in ['knapsack', 'all']:
        if Path(args.knapsack_dir).exists():
            try:
                # Generate summary
                df = generate_knapsack_summary(
                    args.knapsack_dir,
                    str(output_path / 'knapsack_summary.csv')
                )
                
                if df is not None:
                    print("\nKnapsack summary generated successfully")
                else:
                    print("\nWarning: Could not generate Knapsack summary")
            except Exception as e:
                print(f"Error analyzing Knapsack results: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Knapsack results not found at: {args.knapsack_dir}")
    
    print(f"\n{'=' * 80}")
    print("Analysis complete!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()