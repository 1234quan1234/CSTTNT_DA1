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
import logging

from src.problems.discrete.knapsack import KnapsackProblem
from src.swarm.fa import FireflyKnapsackOptimizer
from src.classical.simulated_annealing import SimulatedAnnealingOptimizer
from src.classical.hill_climbing import HillClimbingOptimizer
from src.classical.genetic_algorithm import GeneticAlgorithmOptimizer

from benchmark.config import get_knapsack_configs
from benchmark.instance_generator import generate_knapsack_instance

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


def run_single_knapsack_experiment(algo_name, problem, params, seed, max_iter, instance_seed, constraint_handling='penalty'):
    """
    Run single Knapsack experiment with status tracking.
    
    Parameters
    ----------
    instance_seed : int
        Seed used for instance generation (for tracking)
    constraint_handling : str
        'repair' or 'penalty' - controls constraint handling strategy
        
    Returns
    -------
    dict
        Result dict with status tracking (never None)
    """
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
    
    # Base result structure
    base_result = {
        'algorithm': algo_name,
        'seed': seed,
        'algo_seed': seed,
        'instance_seed': instance_seed,
        'best_value': None,
        'best_fitness': None,
        'total_weight': None,
        'capacity': float(problem.capacity),
        'is_feasible': False,
        'history': [],
        'elapsed_time': 0.0,
        'items_selected': 0,
        'capacity_utilization': 0.0,
        'evaluations': 0,
        'budget': 0,
        'budget_utilization': 0.0,
        'status': 'error',
        'error_type': None,
        'error_msg': None
    }
    
    if algo_name not in algo_map:
        logger.error(f"Unknown algorithm: {algo_name}")
        base_result.update({
            'status': 'error',
            'error_type': 'UnknownAlgorithm',
            'error_msg': f'Unknown algorithm: {algo_name}'
        })
        return base_result
    
    try:
        # Explicit per-worker RNG seeding
        rng = np.random.default_rng(seed)
        np.random.seed(seed)  # Fallback for code using global np.random
        
        optimizer = algo_map[algo_name](
            problem=problem, 
            seed=seed, 
            constraint_handling=constraint_handling,  # Pass the switch
            **params
        )
        
        start_time = time.time()
        best_sol, best_fitness, history, _ = optimizer.run(max_iter=max_iter)
        elapsed = time.time() - start_time
        
        # Validate results
        if not isinstance(history, list) or len(history) == 0:
            logger.warning(f"{algo_name} seed={seed}: Empty history")
            base_result.update({
                'status': 'invalid_history',
                'error_type': 'EmptyHistory',
                'error_msg': 'History is empty or invalid',
                'elapsed_time': float(elapsed)
            })
            return base_result
        
        # Check for invalid fitness
        if np.isnan(best_fitness) or np.isinf(best_fitness):
            logger.warning(f"{algo_name} seed={seed}: Invalid fitness value: {best_fitness}")
            base_result.update({
                'status': 'nan',
                'error_type': 'InvalidFitness',
                'error_msg': f'NaN or Inf fitness: {best_fitness}',
                'elapsed_time': float(elapsed),
                'history': [float(h) if not (np.isnan(h) or np.isinf(h)) else None for h in history]
            })
            return base_result
        
        total_value = -best_fitness
        total_weight = np.sum(best_sol * problem.weights)
        is_feasible = bool(total_weight <= problem.capacity)
        
        # Calculate actual evaluations
        if algo_name in ['FA', 'GA']:  # Population-based
            pop_size = params.get('n_fireflies') or params.get('pop_size', 1)
            actual_evaluations = len(history) * pop_size
            budget = max_iter * pop_size
        else:  # Single-solution (SA, HC)
            actual_evaluations = len(history)
            budget = max_iter
        
        # Success case
        base_result.update({
            'status': 'ok',
            'best_value': float(total_value),
            'best_fitness': float(best_fitness),
            'total_weight': float(total_weight),
            'is_feasible': is_feasible,
            'history': [float(h) for h in history],
            'elapsed_time': float(elapsed),
            'items_selected': int(np.sum(best_sol)),
            'capacity_utilization': float(total_weight / problem.capacity),
            'evaluations': int(actual_evaluations),
            'budget': int(budget),
            'budget_utilization': float(actual_evaluations / budget),
            'error_type': None,
            'error_msg': None
        })
        return base_result
        
    except (FloatingPointError, OverflowError) as e:
        logger.error(f"{algo_name} seed={seed}: Numerical error: {e}")
        base_result.update({
            'status': 'numerical_error',
            'error_type': type(e).__name__,
            'error_msg': str(e)
        })
        return base_result
    except MemoryError as e:
        logger.error(f"{algo_name} seed={seed}: Out of memory")
        base_result.update({
            'status': 'memory',
            'error_type': 'MemoryError',
            'error_msg': 'Out of memory'
        })
        return base_result
    except Exception as e:
        logger.error(f"{algo_name} seed={seed}: {type(e).__name__}: {e}")
        base_result.update({
            'status': 'error',
            'error_type': type(e).__name__,
            'error_msg': str(e)
        })
        return base_result


def run_knapsack_benchmark(size=50, instance_type='uncorrelated', output_dir='benchmark/results', 
                          n_jobs=None, config_name=None, constraint_handling='penalty'):
    """
    Run Knapsack benchmark with parallel execution.
    
    Parameters
    ----------
    size : int or str
        Number of items (50, 100, 200, or 'all')
    instance_type : str
        Instance type
    output_dir : str
        Output directory (default: benchmark/results)
    n_jobs : int, optional
        Number of parallel jobs
    config_name : str, optional
        Config name ('small', 'medium', 'large'). If provided, overrides size/instance_type.
    constraint_handling : str, optional
        'repair' or 'penalty' or 'both' - controls constraint handling strategy
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
    
    # Generate timestamp for this run (ISO 8601 format)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    
    print(f"=" * 70)
    print(f"Knapsack Benchmark")
    print(f"=" * 70)
    print(f"Total configurations: {len(all_configs)}")
    print(f"Runs per config: 30")
    print(f"Total experiments: {len(all_configs) * 4 * 30}")
    print(f"Timestamp: {timestamp}")
    print(f"Constraint handling: {constraint_handling}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)
    
    # Determine which strategies to run
    if constraint_handling == 'both':
        strategies = ['repair', 'penalty']
    else:
        strategies = [constraint_handling]
    
    for config_idx, config in enumerate(all_configs, 1):
        print(f"\n{'-' * 70}")
        print(f"Configuration {config_idx}/{len(all_configs)}")
        print(f"  n_items: {config.n_items}")
        print(f"  type: {config.instance_type}")
        print(f"  instance_seed: {config.seed}")
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
        
        # Setup algorithms with max_iter
        algorithms = {
            'FA': (config.fa_params, max_iter_fa),
            'SA': (config.sa_params, max_iter_single),
            'HC': (config.hc_params, max_iter_single),
            'GA': (config.ga_params, max_iter_ga)
        }
        
        seeds = list(range(30))
        
        # Run experiments for each strategy
        for strategy in strategies:
            print(f"\nStrategy: {strategy.upper()}")
            
            # Run experiments for each algorithm IN PARALLEL
            for algo_name, (algo_params, max_iter) in algorithms.items():
                print(f"\nRunning {algo_name} with {strategy} strategy ({len(seeds)} runs in parallel)...")
                
                # Extract pop_size for metadata
                if algo_name == 'FA':
                    pop_size = algo_params['n_fireflies']
                elif algo_name == 'GA':
                    pop_size = algo_params['pop_size']
                else:
                    pop_size = 1
                
                # Prepare arguments for parallel execution (NOW includes instance_seed)
                args_list = [
                    (algo_name, problem, algo_params, seed, max_iter, config.seed, strategy)
                    for seed in seeds
                ]
                
                # Run in parallel
                try:
                    with mp.Pool(processes=n_jobs) as pool:
                        all_results = pool.starmap(run_single_knapsack_experiment, args_list)
                except Exception as e:
                    logger.error(f"Parallel execution failed for {algo_name}: {e}")
                    continue
                
                # Separate successful and failed results
                successful_results = [r for r in all_results if r['status'] == 'ok']
                failed_results = [r for r in all_results if r['status'] != 'ok']
                
                # Status breakdown
                status_counts = {}
                for r in all_results:
                    status = r['status']
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                # Add DP optimal gap if available (only for successful results)
                gap_results = all_results.copy()
                if dp_optimal_value is not None:
                    for r in gap_results:
                        if r['status'] == 'ok' and r['best_value'] is not None:
                            r['optimality_gap'] = float((dp_optimal_value - r['best_value']) / dp_optimal_value * 100)
                        else:
                            r['optimality_gap'] = None
                
                # Calculate average budget utilization (only for successful runs)
                avg_budget_util = np.mean([r['budget_utilization'] for r in successful_results]) if successful_results else 0.0
                
                if len(failed_results) > 0:
                    logger.warning(f"{algo_name}: {len(failed_results)}/{len(seeds)} runs failed")
                
                if len(successful_results) == 0:
                    logger.error(f"{algo_name}: All runs failed, but still saving results")
                
                # New naming: knapsack_n{size}_{type}_seed{seed}_{algo}_{timestamp}.json
                filename = f"knapsack_n{config.n_items}_{config.instance_type}_seed{config.seed}_{algo_name}_{strategy}_{timestamp}.json"
                result_file = output_path / filename
                
                # Add metadata (includes instance_seed and status breakdown)
                output_data = {
                    'metadata': {
                        'problem': 'knapsack',
                        'n_items': int(config.n_items),
                        'instance_type': str(config.instance_type),
                        'instance_seed': int(config.seed),
                        'algorithm': algo_name,
                        'timestamp': timestamp,
                        'budget': int(config.budget),
                        'max_iter': int(max_iter),
                        'pop_size': int(pop_size),
                        'dp_optimal': float(dp_optimal_value) if dp_optimal_value is not None else None,
                        'has_dp_optimal': config.has_dp_optimal,
                        'n_runs': len(seeds),
                        'n_successful': len(successful_results),
                        'n_failed': len(failed_results),
                        'status_breakdown': status_counts,
                        'avg_budget_utilization': float(avg_budget_util)
                    },
                    'all_results': gap_results  # All results including failed ones
                }
                
                with open(result_file, 'w') as f:
                    json.dump(output_data, f, indent=2)
                
                print(f"  Saved: {filename}")
                
                # Print summary (includes status breakdown)
                if successful_results:
                    values_list = [r['best_value'] for r in successful_results]
                    feasible_count = sum(1 for r in successful_results if r['is_feasible'])
                    
                    print(f"\n    Summary for {algo_name}:")
                    print(f"      Mean ± Std: {np.mean(values_list):.2f} ± {np.std(values_list):.2f}")
                    print(f"      Median: {np.median(values_list):.2f}")
                    print(f"      Best: {np.max(values_list):.2f}")
                    print(f"      Worst: {np.min(values_list):.2f}")
                    print(f"      Feasibility: {feasible_count}/{len(successful_results)} ({feasible_count/len(successful_results)*100:.1f}%)")
                    
                    if dp_optimal_value is not None:
                        gaps = [r['optimality_gap'] for r in gap_results if r['status'] == 'ok' and r['optimality_gap'] is not None]
                        if gaps:
                            print(f"      Avg gap: {np.mean(gaps):.2f}%")
                    
                    print(f"      Avg time: {np.mean([r['elapsed_time'] for r in successful_results]):.2f}s")
                    print(f"      Budget util: {avg_budget_util:.2%}")
                
                # Status breakdown
                status_str = ", ".join([f"{count} {status}" for status, count in status_counts.items()])
                print(f"    Status breakdown: {status_str}")
    
    print(f"\n{'=' * 70}")
    print(f"Knapsack benchmark complete! Results saved to: {output_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Knapsack benchmark')
    parser.add_argument('--size', type=str, default='50',
                        help='Instance size: 50, 100, 200 or "all"')
    parser.add_argument('--type', type=str, default='all',
                        choices=['all', 'uncorrelated', 'weakly', 'strongly', 'subset'],
                        help='Instance type')
    parser.add_argument('--output', type=str, default='benchmark/results/knapsack',
                        help='Output directory')
    parser.add_argument('--jobs', type=int, default=None,
                        help='Number of parallel jobs (default: CPU count - 1)')
    parser.add_argument('--constraint', type=str, default='penalty',
                        choices=['repair', 'penalty', 'both'],
                        help='Constraint handling: repair, penalty, or both')
    
    args = parser.parse_args()
    
    size = 'all' if args.size == 'all' else int(args.size)
    
    run_knapsack_benchmark(
        size=size, 
        instance_type=args.type, 
        output_dir=args.output, 
        n_jobs=args.jobs,
        constraint_handling=args.constraint
    )