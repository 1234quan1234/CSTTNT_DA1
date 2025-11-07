"""
Master script to run all benchmarks.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
from pathlib import Path

from benchmark.run_rastrigin import run_rastrigin_benchmark
from benchmark.run_tsp import run_tsp_benchmark
from benchmark.run_knapsack import run_knapsack_benchmark


def run_all_benchmarks(output_dir='benchmark/results', configs=None):
    """Run all benchmarks with specified configurations."""
    start_time = time.time()
    
    print("=" * 80)
    print("RUNNING ALL BENCHMARKS")
    print("=" * 80)
    
    # Default configs if none specified
    if configs is None:
        configs = {
            'rastrigin': ['quick_convergence'],
            'tsp': ['small'],
            'knapsack': ['small']
        }
    
    results_summary = {}
    
    # Run Rastrigin benchmarks
    if 'rastrigin' in configs:
        print("\n" + "=" * 80)
        print("RASTRIGIN BENCHMARKS")
        print("=" * 80)
        for config in configs['rastrigin']:
            try:
                run_rastrigin_benchmark(
                    config_name=config,
                    output_dir=f"{output_dir}/rastrigin"
                )
                results_summary[f'rastrigin_{config}'] = 'SUCCESS'
            except Exception as e:
                print(f"ERROR in Rastrigin {config}: {e}")
                results_summary[f'rastrigin_{config}'] = f'FAILED: {e}'
    
    # Run TSP benchmarks
    if 'tsp' in configs:
        print("\n" + "=" * 80)
        print("TSP BENCHMARKS")
        print("=" * 80)
        for config in configs['tsp']:
            try:
                run_tsp_benchmark(
                    config_name=config,
                    output_dir=f"{output_dir}/tsp"
                )
                results_summary[f'tsp_{config}'] = 'SUCCESS'
            except Exception as e:
                print(f"ERROR in TSP {config}: {e}")
                results_summary[f'tsp_{config}'] = f'FAILED: {e}'
    
    # Run Knapsack benchmarks
    if 'knapsack' in configs:
        print("\n" + "=" * 80)
        print("KNAPSACK BENCHMARKS")
        print("=" * 80)
        for config in configs['knapsack']:
            try:
                run_knapsack_benchmark(
                    config_name=config,
                    output_dir=f"{output_dir}/knapsack"
                )
                results_summary[f'knapsack_{config}'] = 'SUCCESS'
            except Exception as e:
                print(f"ERROR in Knapsack {config}: {e}")
                results_summary[f'knapsack_{config}'] = f'FAILED: {e}'
    
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    for benchmark, status in results_summary.items():
        print(f"  {benchmark}: {status}")
    print(f"\nTotal time: {elapsed_time/60:.2f} minutes")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run all benchmarks')
    parser.add_argument('--output', type=str, default='benchmark/results',
                        help='Output directory')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick tests only')
    parser.add_argument('--full', action='store_true',
                        help='Run full benchmark suite')
    
    args = parser.parse_args()
    
    if args.quick:
        configs = {
            'rastrigin': ['quick_convergence'],
            'tsp': ['small'],
            'knapsack': ['small']
        }
    elif args.full:
        configs = {
            'rastrigin': ['quick_convergence', 'multimodal_escape', 'scalability'],
            'tsp': ['small', 'medium', 'large'],
            'knapsack': ['small', 'medium', 'large']
        }
    else:
        # Default: medium suite
        configs = {
            'rastrigin': ['quick_convergence', 'multimodal_escape'],
            'tsp': ['small', 'medium'],
            'knapsack': ['small', 'medium']
        }
    
    run_all_benchmarks(output_dir=args.output, configs=configs)
