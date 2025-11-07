"""
Run comprehensive Rastrigin benchmark comparing FA, SA, HC, and GA.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import time
from pathlib import Path
import multiprocessing as mp
from functools import partial

from src.problems.continuous.rastrigin import RastriginProblem
from src.swarm.fa import FireflyContinuousOptimizer
from src.classical.simulated_annealing import SimulatedAnnealingOptimizer
from src.classical.hill_climbing import HillClimbingOptimizer
from src.classical.genetic_algorithm import GeneticAlgorithmOptimizer

from benchmark.config import RASTRIGIN_CONFIGS


def run_single_experiment(algorithm_name, optimizer, max_iter, seed, problem):
    """Run a single experiment and return results."""
    start_time = time.time()
    best_sol, best_fit, history, trajectory = optimizer.run(max_iter=max_iter)
    elapsed_time = time.time() - start_time
    
    return {
        'algorithm': algorithm_name,
        'seed': seed,
        'best_fitness': float(best_fit),
        'history': [float(h) for h in history],
        'elapsed_time': elapsed_time,
        'evaluations': len(history) * (optimizer.pop_size if hasattr(optimizer, 'pop_size') 
                                        else optimizer.n_fireflies if hasattr(optimizer, 'n_fireflies') 
                                        else 1)
    }


def run_single_experiment(algo_name, problem, params, seed, max_iter):
    """Run single experiment (for parallel execution)."""
    import time
    from src.swarm.fa import FireflyContinuousOptimizer
    from src.classical.simulated_annealing import SimulatedAnnealingOptimizer
    from src.classical.hill_climbing import HillClimbingOptimizer
    from src.classical.genetic_algorithm import GeneticAlgorithmOptimizer
    
    algo_map = {
        'FA': FireflyContinuousOptimizer,
        'SA': SimulatedAnnealingOptimizer,
        'HC': HillClimbingOptimizer,
        'GA': GeneticAlgorithmOptimizer
    }
    
    optimizer = algo_map[algo_name](problem=problem, seed=seed, **params)
    
    start_time = time.time()
    _, best_fitness, history, _ = optimizer.run(max_iter=max_iter)
    elapsed = time.time() - start_time
    
    return {
        'algorithm': algo_name,
        'seed': seed,
        'best_fitness': best_fitness,
        'history': history,
        'elapsed_time': elapsed,
        'evaluations': len(history) * (params.get('n_fireflies', 1) or params.get('pop_size', 1))
    }


def run_rastrigin_benchmark(config_name='quick_convergence', output_dir='benchmark/results/rastrigin', n_jobs=None):
    """
    Run Rastrigin benchmark with parallel execution.
    
    Parameters
    ----------
    config_name : str
        Configuration name from config.py
    output_dir : str
        Output directory
    n_jobs : int, optional
        Number of parallel jobs. If None, uses CPU count - 1
    """
    config = RASTRIGIN_CONFIGS[config_name]
    output_path = Path(output_dir) / config_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"=" * 70)
    print(f"Rastrigin Benchmark: {config_name}")
    print(f"=" * 70)
    print(f"Dimension: {config.dim}")
    print(f"Budget: {config.budget} evaluations")
    print(f"Max iterations: {config.max_iter}")
    print(f"Number of runs: {len(config.seeds)}")
    print(f"Success threshold: {config.threshold}")
    
    problem = RastriginProblem(dim=config.dim)
    
    # Extract algorithm parameters
    algo_params = {
        'FA': config.fa_params,
        'SA': config.sa_params,
        'HC': config.hc_params,
        'GA': config.ga_params
    }
    
    seeds = config.seeds
    n_runs = len(seeds)
    
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)
    
    print(f"Using {n_jobs} parallel workers")
    
    # Run experiments for each algorithm IN PARALLEL
    for algo_name in algo_params:
        print(f"\nRunning {algo_name} ({n_runs} runs in parallel)...")
        
        # Prepare arguments for parallel execution
        args_list = [
            (algo_name, problem, algo_params[algo_name], seed, config.max_iter)
            for seed in seeds
        ]
        
        # Run in parallel
        with mp.Pool(processes=n_jobs) as pool:
            results = pool.starmap(run_single_experiment, args_list)
        
        # Save individual results
        result_file = output_path / f"{algo_name}_results.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        best_fits = [r['best_fitness'] for r in results]
        print(f"\n  Summary for {algo_name}:")
        print(f"    Mean ± Std: {np.mean(best_fits):.4f} ± {np.std(best_fits):.4f}")
        print(f"    Median: {np.median(best_fits):.4f}")
        print(f"    Best: {np.min(best_fits):.4f}")
        print(f"    Worst: {np.max(best_fits):.4f}")
        print(f"    Success rate: {np.mean([f < config.threshold for f in best_fits]):.2%}")
        print(f"    Avg time: {np.mean([r['elapsed_time'] for r in results]):.2f}s")
    
    print(f"\n{'=' * 70}")
    print(f"Benchmark complete! Results saved to: {output_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Rastrigin benchmark')
    parser.add_argument('--config', type=str, default='quick_convergence',
                        choices=['quick_convergence', 'multimodal_escape', 'scalability'],
                        help='Benchmark configuration')
    parser.add_argument('--output', type=str, default='benchmark/results/rastrigin',
                        help='Output directory')
    
    args = parser.parse_args()
    
    run_rastrigin_benchmark(config_name=args.config, output_dir=args.output)
    print(f"    Worst: {np.max(best_fits):.4f}")
    print(f"    Success rate: {np.mean([f < config.threshold for f in best_fits]):.2%}")
    print(f"    Avg time: {np.mean([r['elapsed_time'] for r in results]):.2f}s")
    
    print(f"\n{'=' * 70}")
    print(f"Benchmark complete! Results saved to: {output_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Rastrigin benchmark')
    parser.add_argument('--config', type=str, default='quick_convergence',
                        choices=['quick_convergence', 'multimodal_escape', 'scalability'],
                        help='Benchmark configuration')
    parser.add_argument('--output', type=str, default='benchmark/results/rastrigin',
                        help='Output directory')
    
    args = parser.parse_args()
    
    run_rastrigin_benchmark(config_name=args.config, output_dir=args.output)
