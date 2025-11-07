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


def run_rastrigin_benchmark(config_name='quick_convergence', output_dir='benchmark/results/rastrigin'):
    """Run Rastrigin benchmark for all algorithms."""
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
    
    algorithms = {
        'FA': lambda seed: FireflyContinuousOptimizer(
            problem=problem,
            n_fireflies=config.fa_params['n_fireflies'],
            alpha=config.fa_params['alpha'],
            beta0=config.fa_params['beta0'],
            gamma=config.fa_params['gamma'],
            seed=seed
        ),
        'SA': lambda seed: SimulatedAnnealingOptimizer(
            problem=problem,
            initial_temp=config.sa_params['initial_temp'],
            cooling_rate=config.sa_params['cooling_rate'],
            step_size=config.sa_params['step_size'],
            seed=seed
        ),
        'HC': lambda seed: HillClimbingOptimizer(
            problem=problem,
            step_size=config.hc_params['step_size'],
            num_neighbors=config.hc_params['num_neighbors'],
            seed=seed
        ),
        'GA': lambda seed: GeneticAlgorithmOptimizer(
            problem=problem,
            pop_size=config.ga_params['pop_size'],
            crossover_rate=config.ga_params['crossover_rate'],
            mutation_rate=config.ga_params['mutation_rate'],
            tournament_size=config.ga_params['tournament_size'],
            elitism=config.ga_params['elitism'],
            seed=seed
        )
    }
    
    for algo_name, algo_factory in algorithms.items():
        print(f"\n{'-' * 70}")
        print(f"Running {algo_name}...")
        print(f"{'-' * 70}")
        
        results = []
        
        for i, seed in enumerate(config.seeds):
            optimizer = algo_factory(seed)
            result = run_single_experiment(algo_name, optimizer, config.max_iter, seed, problem)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{len(config.seeds)} runs")
        
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
