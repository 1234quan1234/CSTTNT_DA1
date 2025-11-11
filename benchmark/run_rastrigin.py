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
import logging
import signal
import tempfile
import shutil

from src.problems.continuous.rastrigin import RastriginProblem
from src.swarm.fa import FireflyContinuousOptimizer
from src.classical.simulated_annealing import SimulatedAnnealingOptimizer
from src.classical.hill_climbing import HillClimbingOptimizer
from src.classical.genetic_algorithm import GeneticAlgorithmOptimizer

from benchmark.config import RASTRIGIN_CONFIGS


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Timeout handler
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Algorithm execution timed out")


def validate_config(config):
    """
    Validate benchmark configuration parameters.
    
    Parameters
    ----------
    config : BenchmarkConfig
        Configuration to validate
        
    Returns
    -------
    bool
        True if valid
        
    Raises
    ------
    ValueError
        If validation fails
    """
    if not isinstance(config.seeds, (list, range)):
        raise ValueError(f"seeds must be list or range, got {type(config.seeds)}")
    
    seeds_list = list(config.seeds)
    if len(seeds_list) == 0:
        raise ValueError("seeds cannot be empty")
    
    if not all(isinstance(s, int) for s in seeds_list):
        raise ValueError("All seeds must be integers")
    
    if config.max_iter < 10:
        raise ValueError(f"max_iter must be >= 10, got {config.max_iter}")
    
    if config.dim <= 0:
        raise ValueError(f"dimension must be > 0, got {config.dim}")
    
    if config.budget <= 0:
        raise ValueError(f"budget must be > 0, got {config.budget}")
    
    logger.info(f"Configuration validated: {len(seeds_list)} runs, dim={config.dim}, budget={config.budget}")
    return True


def check_disk_space(path: Path, required_mb: int = 100) -> bool:
    """Check if sufficient disk space is available."""
    try:
        import shutil as sh
        stat = sh.disk_usage(path)
        free_mb = stat.free / (1024 * 1024)
        
        if free_mb < required_mb:
            logger.error(f"Insufficient disk space: {free_mb:.1f}MB free, {required_mb}MB required")
            logger.info(f"  Suggestion: Free up space in {path}")
            return False
        return True
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        return True


def atomic_json_write(data: dict, output_file: Path):
    """
    Atomically write JSON file (write to temp, then rename).
    
    Parameters
    ----------
    data : dict
        Data to write
    output_file : Path
        Target file path
        
    Raises
    ------
    OSError
        If write fails
    """
    # Write to temporary file first
    temp_fd, temp_path = tempfile.mkstemp(
        dir=output_file.parent,
        prefix='.tmp_',
        suffix='.json'
    )
    
    try:
        with os.fdopen(temp_fd, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Verify JSON is valid by reading it back
        with open(temp_path, 'r') as f:
            json.load(f)
        
        # Atomic rename
        shutil.move(temp_path, output_file)
        logger.info(f"Successfully wrote: {output_file.name}")
        
    except json.JSONDecodeError as e:
        logger.error(f"Generated invalid JSON: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    except Exception as e:
        logger.error(f"Error writing file: {type(e).__name__}: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def run_single_experiment_safe(algo_name, problem, params, seed, max_iter, threshold, problem_seed, timeout_seconds=300):
    """
    Run single experiment with timeout and error handling.
    
    Parameters
    ----------
    problem_seed : int
        Seed used for problem initialization (for tracking)
    threshold : float
        Success threshold for tracking convergence
    timeout_seconds : int
        Maximum execution time in seconds (default: 5 minutes)
        
    Returns
    -------
    dict
        Result dict with status tracking (never None)
    """
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
    
    # Base result structure
    base_result = {
        'algorithm': algo_name,
        'seed': seed,
        'algo_seed': seed,
        'problem_seed': problem_seed,
        'best_fitness': None,
        'history': [],
        'elapsed_time': 0.0,
        'evaluations': 0,
        'budget': 0,
        'budget_utilization': 0.0,
        'success': False,
        'hit_evaluations': None,
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
        
        optimizer = algo_map[algo_name](problem=problem, seed=seed, **params)
        
        # Set timeout (Unix-like systems only)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
        
        start_time = time.time()
        _, best_fitness, history, _ = optimizer.run(max_iter=max_iter)
        elapsed = time.time() - start_time
        
        # Cancel timeout
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        
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
        
        # Check for invalid values
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
        
        # Calculate actual evaluations based on algorithm type
        if algo_name in ['FA', 'GA']:  # Population-based
            pop_size = params.get('n_fireflies') or params.get('pop_size', 1)
            actual_evaluations = len(history) * pop_size
            budget = max_iter * pop_size
        else:  # Single-solution (SA, HC)
            actual_evaluations = len(history)
            budget = max_iter
        
        # Track success and hit_evaluations
        success = bool(best_fitness < threshold)
        hit_evaluations = None
        
        if success:
            # Find first evaluation where threshold was hit
            for i, h in enumerate(history):
                if h < threshold:
                    if algo_name in ['FA', 'GA']:
                        pop_size = params.get('n_fireflies') or params.get('pop_size', 1)
                        hit_evaluations = (i + 1) * pop_size
                    else:
                        hit_evaluations = i + 1
                    break
        
        # Success case
        base_result.update({
            'status': 'ok',
            'best_fitness': float(best_fitness),
            'history': [float(h) for h in history],
            'elapsed_time': float(elapsed),
            'evaluations': int(actual_evaluations),
            'budget': int(budget),
            'budget_utilization': float(actual_evaluations / budget),
            'success': success,
            'hit_evaluations': int(hit_evaluations) if hit_evaluations is not None else None,
            'error_type': None,
            'error_msg': None
        })
        return base_result
        
    except TimeoutException:
        logger.error(f"{algo_name} seed={seed}: Timed out after {timeout_seconds}s")
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        base_result.update({
            'status': 'timeout',
            'error_type': 'TimeoutException',
            'error_msg': f'Timed out after {timeout_seconds}s'
        })
        return base_result
    except (FloatingPointError, OverflowError) as e:
        logger.error(f"{algo_name} seed={seed}: Numerical error: {e}")
        logger.info(f"  Suggestion: Check algorithm parameters (alpha, beta, temperature)")
        base_result.update({
            'status': 'numerical_error',
            'error_type': type(e).__name__,
            'error_msg': str(e)
        })
        return base_result
    except MemoryError as e:
        logger.error(f"{algo_name} seed={seed}: Out of memory")
        logger.info(f"  Suggestion: Reduce population size or problem dimension")
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
    finally:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)


def run_rastrigin_benchmark(config_name='quick_convergence', output_dir='benchmark/results', n_jobs=None):
    """
    Run Rastrigin benchmark with parallel execution.
    
    Parameters
    ----------
    config_name : str
        Configuration name from config.py
    output_dir : str
        Output directory (default: benchmark/results)
    n_jobs : int, optional
        Number of parallel jobs. If None, uses CPU count - 1
    """
    try:
        config = RASTRIGIN_CONFIGS[config_name]
    except KeyError:
        logger.error(f"Unknown config: {config_name}")
        logger.info(f"  Available configs: {list(RASTRIGIN_CONFIGS.keys())}")
        return
    
    # Validate configuration
    try:
        validate_config(config)
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        return
    
    # Extract problem_seed from config (use first seed as problem seed, or separate field if available)
    problem_seed = getattr(config, 'problem_seed', list(config.seeds)[0])
    
    output_path = Path(output_dir)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Cannot create output directory {output_dir}: {e}")
        logger.info(f"  Suggestion: Check permissions or path validity")
        return
    
    # Check disk space
    if not check_disk_space(output_path, required_mb=100):
        return
    
    # Generate timestamp for this run (ISO 8601 format)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    
    print(f"=" * 70)
    print(f"Rastrigin Benchmark: {config_name}")
    print(f"=" * 70)
    print(f"Dimension: {config.dim}")
    print(f"Budget: {config.budget} evaluations")
    print(f"Max iterations: {config.max_iter}")
    print(f"Number of runs: {len(config.seeds)}")
    print(f"Success threshold: {config.threshold}")
    print(f"Problem seed: {problem_seed}")
    print(f"Timestamp: {timestamp}")
    
    # Initialize problem with explicit seed
    problem = RastriginProblem(dim=config.dim)
    
    # Extract algorithm parameters
    algo_params = {
        'FA': config.fa_params,
        'SA': config.sa_params,
        'HC': config.hc_params,
        'GA': config.ga_params
    }
    
    seeds = list(config.seeds)
    n_runs = len(seeds)
    
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)
    
    print(f"Using {n_jobs} parallel workers")
    
    # Run experiments for each algorithm IN PARALLEL
    for algo_name in algo_params:
        print(f"\nRunning {algo_name} ({n_runs} runs in parallel)...")
        
        # Calculate correct max_iter based on algorithm type and budget
        params = algo_params[algo_name]
        if algo_name in ['FA', 'GA']:  # Population-based
            pop_size = params.get('n_fireflies') or params.get('pop_size', 50)
            max_iter_actual = config.budget // pop_size
            if config.budget % pop_size != 0:
                logger.warning(f"{algo_name}: Budget {config.budget} not divisible by pop_size {pop_size}, using {max_iter_actual} iterations")
        else:  # Single-solution (SA, HC)
            max_iter_actual = config.budget
            pop_size = 1  # For metadata consistency
        
        logger.info(f"{algo_name}: Using max_iter={max_iter_actual} for budget={config.budget}")
        
        # Prepare arguments for parallel execution (NOW includes problem_seed)
        args_list = [
            (algo_name, problem, params, seed, max_iter_actual, config.threshold, problem_seed, 300)
            for seed in seeds
        ]
        
        # Run in parallel
        try:
            with mp.Pool(processes=n_jobs) as pool:
                all_results = pool.starmap(run_single_experiment_safe, args_list)
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
        
        # Calculate average budget utilization (only for successful runs)
        avg_budget_util = np.mean([r['budget_utilization'] for r in successful_results]) if successful_results else 0.0
        
        # Validate budget utilization
        if successful_results and abs(avg_budget_util - 1.0) > 0.1:
            logger.warning(f"{algo_name}: Average budget utilization {avg_budget_util:.2%} deviates from 100% by more than 10%")
        
        if len(failed_results) > 0:
            logger.warning(f"{algo_name}: {len(failed_results)}/{n_runs} runs failed")
        
        if len(successful_results) == 0:
            logger.error(f"{algo_name}: All runs failed, skipping save but logging failures")
        
        # Save with new naming convention
        result_filename = f"rastrigin_{config_name}_{algo_name}_{timestamp}.json"
        result_file = output_path / result_filename
        
        # Add metadata to results (includes problem_seed and status breakdown)
        output_data = {
            'metadata': {
                'problem': 'rastrigin',
                'config_name': config_name,
                'algorithm': algo_name,
                'timestamp': timestamp,
                'dimension': config.dim,
                'budget': config.budget,
                'max_iter': max_iter_actual,
                'pop_size': pop_size,
                'problem_seed': problem_seed,
                'n_runs': n_runs,
                'n_successful': len(successful_results),
                'n_failed': len(failed_results),
                'status_breakdown': status_counts,
                'threshold': config.threshold,
                'avg_budget_utilization': float(avg_budget_util)
            },
            'all_results': all_results  # All results including failed ones
        }
        
        # Atomic write
        try:
            atomic_json_write(output_data, result_file)
        except Exception as e:
            logger.error(f"Failed to save results for {algo_name}: {e}")
            continue
        
        # Print summary (includes status breakdown)
        if successful_results:
            best_fits = [r['best_fitness'] for r in successful_results]
            success_rate = np.mean([r['success'] for r in successful_results])
            hit_evals = [r['hit_evaluations'] for r in successful_results if r['hit_evaluations'] is not None]
            
            print(f"\n  Summary for {algo_name}:")
            print(f"    Mean ± Std: {np.mean(best_fits):.4f} ± {np.std(best_fits):.4f}")
            print(f"    Median: {np.median(best_fits):.4f}")
            print(f"    Best: {np.min(best_fits):.4f}")
            print(f"    Worst: {np.max(best_fits):.4f}")
            print(f"    Success rate: {success_rate:.2%}")
            if hit_evals:
                print(f"    Avg hit evals: {np.mean(hit_evals):.0f} ({np.mean(hit_evals)/config.budget:.1%} of budget)")
            print(f"    Avg time: {np.mean([r['elapsed_time'] for r in successful_results]):.2f}s")
            print(f"    Budget util: {avg_budget_util:.2%}")
        
        # Status breakdown
        status_str = ", ".join([f"{count} {status}" for status, count in status_counts.items()])
        print(f"    Status breakdown: {status_str}")
    
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