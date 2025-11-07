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


def run_single_experiment(algorithm_name, optimizer, max_iter, seed, problem):
    """Run a single Knapsack experiment."""
    start_time = time.time()
    best_sol, best_fit, history, trajectory = optimizer.run(max_iter=max_iter)
    elapsed_time = time.time() - start_time
    
    # Calculate actual value (fitness is negative value)
    total_value = -best_fit
    total_weight = np.sum(best_sol * problem.weights)
    is_feasible = total_weight <= problem.capacity
    
    return {
        'algorithm': algorithm_name,
        'seed': seed,
        'best_value': float(total_value),
        'best_fitness': float(best_fit),
        'total_weight': float(total_weight),
        'capacity': float(problem.capacity),
        'is_feasible': bool(is_feasible),
        'history': [-h for h in history],  # Convert to values
        'elapsed_time': elapsed_time,
        'items_selected': int(np.sum(best_sol)),
        'capacity_utilization': float(total_weight / problem.capacity)
    }


def run_knapsack_benchmark(size=50, instance_type='all', output_dir='benchmark/results/knapsack'):
    """Run Knapsack benchmark for specified configurations."""
    
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
            'FA': {
                'factory': lambda s: FireflyKnapsackOptimizer(
                    problem=problem,
                    n_fireflies=config.fa_params['n_fireflies'],
                    alpha_flip=config.fa_params['alpha_flip'],
                    max_flips_per_move=config.fa_params['max_flips_per_move'],
                    repair_method=config.fa_params['repair_method'],
                    seed=s
                ),
                'max_iter': max_iter_fa
            },
            'SA': {
                'factory': lambda s: SimulatedAnnealingOptimizer(
                    problem=problem,
                    initial_temp=config.sa_params['initial_temp'],
                    cooling_rate=config.sa_params['cooling_rate'],
                    seed=s
                ),
                'max_iter': max_iter_single
            },
            'HC': {
                'factory': lambda s: HillClimbingOptimizer(
                    problem=problem,
                    num_neighbors=config.hc_params['num_neighbors'],
                    restart_interval=config.hc_params['restart_interval'],
                    seed=s
                ),
                'max_iter': max_iter_single
            },
            'GA': {
                'factory': lambda s: GeneticAlgorithmOptimizer(
                    problem=problem,
                    pop_size=config.ga_params['pop_size'],
                    crossover_rate=config.ga_params['crossover_rate'],
                    mutation_rate=1.0 / config.n_items,  # 1/n
                    tournament_size=config.ga_params['tournament_size'],
                    elitism=config.ga_params['elitism'],
                    seed=s
                ),
                'max_iter': max_iter_ga
            }
        }
        
        # Run experiments
        for algo_name, algo_config in algorithms.items():
            print(f"\n  Running {algo_name}...")
            
            results = []
            seeds = list(range(30))
            
            for i, seed in enumerate(seeds):
                optimizer = algo_config['factory'](seed)
                result = run_single_experiment(
                    algo_name, optimizer, algo_config['max_iter'], seed, problem
                )
                
                # Add DP optimal if available
                if dp_optimal_value is not None:
                    result['dp_optimal_value'] = dp_optimal_value
                    result['optimality_gap'] = (dp_optimal_value - result['best_value']) / dp_optimal_value * 100
                
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"    Completed {i + 1}/30 runs")
            
            # Save results
            filename = f"n{config.n_items}_{config.instance_type}_seed{config.seed}_{algo_name}.json"
            result_file = output_path / filename
            
            with open(result_file, 'w') as f:
                json.dump({
                    'config': {
                        'n_items': config.n_items,
                        'instance_type': config.instance_type,
                        'instance_seed': config.seed,
                        'budget': config.budget,
                        'dp_optimal': dp_optimal_value
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
