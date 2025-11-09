"""
Run comprehensive Knapsack benchmark comparing FA, SA, HC, and GA.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import time
from pathlib import Path
import multiprocessing as mp

from src.problems.discrete.knapsack import KnapsackProblem
from src.swarm.fa import FireflyKnapsackOptimizer
from src.classical.simulated_annealing import SimulatedAnnealingOptimizer
from src.classical.hill_climbing import HillClimbingOptimizer
from src.classical.genetic_algorithm import GeneticAlgorithmOptimizer

from benchmark.config import get_knapsack_configs
from benchmark.instance_generator import generate_knapsack_instance


def solve_knapsack_dp(values, weights, capacity):
    """
    Solve 0/1 Knapsack using dynamic programming.
    Only for n <= 100 (memory constraint).
    
    Returns
    -------
    optimal_value : float
        Optimal total value.
    optimal_selection : np.ndarray
        Binary selection vector.
    """
    n = len(values)
    C = int(capacity)
    
    # DP table: dp[i][w] = max value using items 0..i-1 with capacity w
    dp = np.zeros((n + 1, C + 1), dtype=float)
    
    # Fill DP table
    for i in range(1, n + 1):
        for w in range(C + 1):
            # Don't take item i-1
            dp[i][w] = dp[i-1][w]
            
            # Take item i-1 if possible
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w], dp[i-1][w - int(weights[i-1])] + values[i-1])
    
    # Backtrack to find selection
    optimal_value = dp[n][C]
    selection = np.zeros(n, dtype=int)
    
    w = C
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selection[i-1] = 1
            w -= int(weights[i-1])
    
    return optimal_value, selection


def run_single_knapsack_experiment(algo_name, problem, params, seed, max_iter):
    """Run single Knapsack experiment (for parallel execution)."""
    import time
    import numpy as np
    from src.swarm.fa import FireflyKnapsackOptimizer
    from src.classical.simulated_annealing import SimulatedAnnealingOptimizer
    from src.classical.hill_climbing import HillClimbingOptimizer
    from src.classical.genetic_algorithm import GeneticAlgorithmOptimizer
    
    algo_map = {
        'FA': FireflyKnapsackOptimizer,
        'SA': SimulatedAnnealingOptimizer,
        'HC': HillClimbingOptimizer,
        'GA': GeneticAlgorithmOptimizer
    }
    
    optimizer = algo_map[algo_name](problem=problem, seed=seed, **params)
    
    start_time = time.time()
    best_sol, best_fitness, history, _ = optimizer.run(max_iter=max_iter)
    elapsed = time.time() - start_time
    
    total_value = -best_fitness
    total_weight = np.sum(best_sol * problem.weights)
    is_feasible = bool(total_weight <= problem.capacity)
    
    return {
        'algorithm': algo_name,
        'seed': int(seed),
        'best_value': float(total_value),
        'best_fitness': float(best_fitness),
        'total_weight': float(total_weight),
        'capacity': float(problem.capacity),
        'is_feasible': is_feasible,
        'history': [float(h) for h in history],
        'elapsed_time': float(elapsed),
        'items_selected': int(np.sum(best_sol)),
        'capacity_utilization': float(total_weight / problem.capacity)
    }


def run_knapsack_benchmark(size=50, instance_type='uncorrelated', output_dir='benchmark/results/knapsack', n_jobs=None, config_name=None):
    """
    Run Knapsack benchmark with parallel execution.
    
    Parameters
    ----------
    size : int or str
        Number of items (50, 100, 200, 500, or 'all')
    instance_type : str
        Instance type
    output_dir : str
        Output directory
    n_jobs : int, optional
        Number of parallel jobs
    config_name : str, optional
        Config name ('small', 'medium', 'large'). If provided, overrides size/instance_type.
    """
    
    # Map config_name to size/instance_type if provided
    if config_name is not None:
        config_map = {
            'small': (50, 'all'),
            'medium': (100, 'all'),
            'large': (200, 'all')
        }
        if config_name in config_map:
            size, instance_type = config_map[config_name]
    
    # Get all configs or filter by size/type
    all_configs = get_knapsack_configs()
    if size != 'all':
        all_configs = [c for c in all_configs if c.n_items == size]
    
    if instance_type != 'all':
        all_configs = [c for c in all_configs if c.instance_type == instance_type]
    
    print(f"=" * 70)
    print(f"Knapsack Benchmark")
    print(f"=" * 70)
    print(f"Total configurations: {len(all_configs)}")
    print(f"Runs per config: 30")
    print(f"Total experiments: {len(all_configs) * 4 * 30}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)
    
    for config_idx, config in enumerate(all_configs, 1):
        print(f"\n{'-' * 70}")
        print(f"Configuration {config_idx}/{len(all_configs)}")
        print(f"  n_items: {config.n_items}")
        print(f"  type: {config.instance_type}")
        print(f"  seed: {config.seed}")
        print(f"  budget: {config.budget}")
        print(f"{'-' * 70}")
        
        # Generate instance
        values, weights, capacity = generate_knapsack_instance(
            config.n_items, config.instance_type, config.seed
        )
        
        problem = KnapsackProblem(values, weights, capacity)
        
        # Compute DP optimal if applicable
        dp_optimal_value = None
        if config.has_dp_optimal:
            print("  Computing DP optimal solution...")
            dp_optimal_value, dp_selection = solve_knapsack_dp(values, weights, capacity)
            print(f"  DP optimal value: {dp_optimal_value:.2f}")
        
        # Calculate budget in iterations
        max_iter_fa = config.budget // config.fa_params['n_fireflies']
        max_iter_ga = config.budget // config.ga_params['pop_size']
        max_iter_single = config.budget  # HC and SA evaluate 1 solution per iter
        
        # Setup algorithms
        algorithms = {
            'FA': (config.fa_params, max_iter_fa),
            'SA': (config.sa_params, max_iter_single),
            'HC': (config.hc_params, max_iter_single),
            'GA': (config.ga_params, max_iter_ga)
        }
        
        seeds = list(range(30))
        
        # Run experiments for each algorithm IN PARALLEL
        for algo_name, (algo_params, max_iter) in algorithms.items():
            print(f"\nRunning {algo_name} ({len(seeds)} runs in parallel)...")
            
            # Prepare arguments for parallel execution
            args_list = [
                (algo_name, problem, algo_params, seed, max_iter)
                for seed in seeds
            ]
            
            # Run in parallel
            with mp.Pool(processes=n_jobs) as pool:
                results = pool.starmap(run_single_knapsack_experiment, args_list)
            
            # Add DP optimal if available
            if dp_optimal_value is not None:
                for result in results:
                    result['dp_optimal_value'] = float(dp_optimal_value)
                    result['optimality_gap'] = float((dp_optimal_value - result['best_value']) / dp_optimal_value * 100)
            
            # Save results
            filename = f"n{config.n_items}_{config.instance_type}_seed{config.seed}_{algo_name}.json"
            result_file = output_path / filename
            
            with open(result_file, 'w') as f:
                json.dump({
                    'config': {
                        'n_items': int(config.n_items),
                        'instance_type': str(config.instance_type),
                        'instance_seed': int(config.seed),
                        'budget': int(config.budget),
                        'dp_optimal': float(dp_optimal_value) if dp_optimal_value is not None else None
                    },
                    'results': results
                }, f, indent=2)
            
            # Print summary
            values_list = [r['best_value'] for r in results]
            feasible_count = sum(1 for r in results if r['is_feasible'])
            
            print(f"\n    Summary for {algo_name}:")
            print(f"      Mean ± Std: {np.mean(values_list):.2f} ± {np.std(values_list):.2f}")
            print(f"      Median: {np.median(values_list):.2f}")
            print(f"      Best: {np.max(values_list):.2f}")
            print(f"      Worst: {np.min(values_list):.2f}")
            print(f"      Feasibility: {feasible_count}/30 ({feasible_count/30*100:.1f}%)")
            
            if dp_optimal_value is not None:
                gaps = [r['optimality_gap'] for r in results]
                print(f"      Avg gap: {np.mean(gaps):.2f}%")
            
            print(f"      Avg time: {np.mean([r['elapsed_time'] for r in results]):.2f}s")
    
    print(f"\n{'=' * 70}")
    print(f"Knapsack benchmark complete! Results saved to: {output_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Knapsack benchmark')
    parser.add_argument('--size', type=str, default='50',
                        help='Instance size: 50, 100, 200, 500, or "all"')
    parser.add_argument('--type', type=str, default='all',
                        choices=['all', 'uncorrelated', 'weakly', 'strongly', 'subset'],
                        help='Instance type')
    parser.add_argument('--output', type=str, default='benchmark/results/knapsack',
                        help='Output directory')
    
    args = parser.parse_args()
    
    size = 'all' if args.size == 'all' else int(args.size)
    
    run_knapsack_benchmark(size=size, instance_type=args.type, output_dir=args.output)