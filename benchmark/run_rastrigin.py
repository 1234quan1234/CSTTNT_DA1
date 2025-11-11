"""
Run comprehensive Rastrigin benchmark comparing FA, SA, HC, and GA.
Supports two scenarios: 'out_of_the_box' and 'specialist' (with auto-tuning).
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
import itertools
import pandas as pd
from typing import Dict, List, Tuple, Any

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


def run_single_experiment_safe(algo_name, problem, params, seed, max_iter, thresholds, problem_seed, timeout_seconds=300):
    """
    Run single experiment with timeout and error handling.
    
    Parameters
    ----------
    thresholds : Dict[str, float]
        Dictionary of success levels (e.g., {'gold': 1.0, 'silver': 10.0, 'bronze': 50.0})
    problem_seed : int
        Seed used for problem initialization (for tracking)
    timeout_seconds : int
        Maximum execution time in seconds (default: 5 minutes)
        
    Returns
    -------
    dict
        Result dict with status tracking and multi-level success metrics
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
    
    # Base result structure with NEW multi-level success tracking
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
        'success_levels': {},  # NEW: Replaces single 'success' field
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
        np.random.seed(seed)
        
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
        if algo_name in ['FA', 'GA']:
            pop_size = params.get('n_fireflies') or params.get('pop_size', 1)
            actual_evaluations = len(history) * pop_size
            budget = max_iter * pop_size
        else:
            actual_evaluations = len(history)
            budget = max_iter
        
        # Validate budget utilization for SA
        if algo_name == 'SA':
            budget_util = actual_evaluations / budget
            if budget_util < 0.995:
                logger.warning(f"SA seed={seed}: Low budget utilization {budget_util:.2%}")
        
        # NEW: Calculate success for each threshold level
        success_levels = {}
        
        # Sort thresholds from lowest (best/hardest) to highest (easier)
        sorted_thresholds = sorted(thresholds.items(), key=lambda item: item[1])
        
        for level_name, threshold_value in sorted_thresholds:
            is_successful = bool(abs(best_fitness) < threshold_value)
            hit_evals = None
            
            if is_successful:
                # Find first evaluation where threshold was achieved
                for i, h in enumerate(history):
                    if abs(h) < threshold_value:
                        if algo_name in ['FA', 'GA']:
                            pop_size = params.get('n_fireflies') or params.get('pop_size', 1)
                            hit_evals = (i + 1) * pop_size
                        else:
                            hit_evals = i + 1
                        break
            
            success_levels[level_name] = {
                'success': is_successful,
                'threshold': float(threshold_value),
                'hit_evaluations': int(hit_evals) if hit_evals is not None else None
            }
        
        # Success case with NEW multi-level tracking
        base_result.update({
            'status': 'ok',
            'best_fitness': float(best_fitness),
            'history': [float(h) for h in history],
            'elapsed_time': float(elapsed),
            'evaluations': int(actual_evaluations),
            'budget': int(budget),
            'budget_utilization': float(actual_evaluations / budget),
            'success_levels': success_levels,  # NEW: Multi-level success
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
    finally:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)


# ============================================================================
# PARAMETER TUNING FUNCTIONS
# ============================================================================

def tune_algorithm_parameters(
    algo_name: str,
    base_params: Dict[str, Any],
    tuning_grid: Dict[str, List],
    problem,
    budget: int,
    max_iter: int,
    thresholds: Dict[str, float],  # UPDATED: Pass thresholds dict
    problem_seed: int,
    n_seeds_for_tuning: int = 5,
    n_jobs: int = 4
) -> Dict[str, Any]:
    """
    Light-tune algorithm parameters using grid search.
    
    Parameters
    ----------
    thresholds : Dict[str, float]
        Multi-level success thresholds
    algo_name : str
        Algorithm name ('FA', 'SA', 'HC', 'GA')
    base_params : dict
        Base parameters (will be updated with grid values)
    tuning_grid : dict
        Parameter grid, e.g., {'alpha': [0.2, 0.3], 'gamma': [0.5, 1.0]}
    problem : Problem
        Problem instance
    budget : int
        Evaluation budget
    max_iter : int
        Maximum iterations
    problem_seed : int
        Problem seed for reproducibility
    n_seeds_for_tuning : int
        Number of random seeds per parameter combination
    n_jobs : int
        Number of parallel workers
        
    Returns
    -------
    dict
        Best parameters found
    """
    print(f"\n{'='*70}")
    print(f"TUNING PARAMETERS FOR {algo_name}")
    print(f"{'='*70}")
    print(f"Parameter grid: {tuning_grid}")
    print(f"Thresholds: {thresholds}")
    print(f"Number of seeds per config: {n_seeds_for_tuning}")
    
    # Generate all parameter combinations
    param_names = list(tuning_grid.keys())
    param_values = list(tuning_grid.values())
    combinations = list(itertools.product(*param_values))
    
    print(f"Total combinations to test: {len(combinations)}")
    
    # Generate random seeds for tuning
    tuning_seeds = list(range(1000, 1000 + n_seeds_for_tuning))
    
    # Prepare all tasks
    tasks = []
    for combo in combinations:
        params = base_params.copy()
        for param_name, param_value in zip(param_names, combo):
            params[param_name] = param_value
        
        for seed in tuning_seeds:
            tasks.append((algo_name, problem, params, seed, max_iter, thresholds, problem_seed, 300))
    
    print(f"Running {len(tasks)} experiments in parallel...")
    
    # Run in parallel
    try:
        with mp.Pool(processes=n_jobs) as pool:
            all_results = pool.starmap(run_single_experiment_safe, tasks)
    except Exception as e:
        logger.error(f"Parallel tuning failed for {algo_name}: {e}")
        return base_params
    
    # Convert to DataFrame for analysis
    records = []
    task_idx = 0
    for combo in combinations:
        combo_dict = dict(zip(param_names, combo))
        for seed in tuning_seeds:
            result = all_results[task_idx]
            if result['status'] == 'ok':
                record = combo_dict.copy()
                record['seed'] = seed
                record['best_fitness'] = result['best_fitness']
                # Use bronze level success for tuning (most lenient)
                record['success'] = result['success_levels'].get('bronze', {}).get('success', False)
                records.append(record)
            task_idx += 1
    
    if not records:
        logger.warning(f"{algo_name}: All tuning runs failed, using default parameters")
        return base_params
    
    df = pd.DataFrame(records)
    
    # Group by parameter combination and calculate median fitness
    group_cols = param_names
    summary = df.groupby(group_cols).agg({
        'best_fitness': ['median', 'mean', 'std'],
        'success': 'mean'
    }).reset_index()
    
    summary.columns = param_names + ['median_fitness', 'mean_fitness', 'std_fitness', 'success_rate']
    summary = summary.sort_values('median_fitness')
    
    # Print tuning results
    print(f"\nTuning Results (sorted by median fitness):")
    print(summary.to_string(index=False))
    
    # Select best parameters
    best_row = summary.iloc[0]
    best_params = base_params.copy()
    for param_name in param_names:
        best_params[param_name] = best_row[param_name]
    
    print(f"\n✓ BEST PARAMETERS FOR {algo_name}:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"  Median fitness: {best_row['median_fitness']:.6f}")
    print(f"  Success rate: {best_row['success_rate']:.2%}")
    
    return best_params


def run_tuning_phase(
    config,
    problem,
    n_seeds_for_tuning: int = 5,
    n_jobs: int = 4
) -> Dict[str, Dict[str, Any]]:
    """
    Run tuning phase for all algorithms with tuning grids.
    
    Returns
    -------
    dict
        Mapping from algo_name to best parameters
    """
    print("\n" + "="*70)
    print("PHASE 1: PARAMETER TUNING")
    print("="*70)
    
    tuned_params = {}
    
    for algo_name in ['FA', 'SA', 'HC', 'GA']:
        if algo_name not in config.tuning_grids:
            logger.info(f"No tuning grid for {algo_name}, using default parameters")
            continue
        
        tuning_grid = config.tuning_grids[algo_name]
        if not tuning_grid:
            logger.info(f"Empty tuning grid for {algo_name}, using default parameters")
            continue
        
        # Get base parameters
        if algo_name == 'FA':
            base_params = config.fa_params.copy()
        elif algo_name == 'SA':
            base_params = config.sa_params.copy()
        elif algo_name == 'HC':
            base_params = config.hc_params.copy()
        elif algo_name == 'GA':
            base_params = config.ga_params.copy()
        else:
            continue
        
        # Calculate max_iter for tuning
        if algo_name in ['FA', 'GA']:
            pop_size = base_params.get('n_fireflies') or base_params.get('pop_size', 50)
            max_iter_tuning = config.budget // pop_size
        else:
            max_iter_tuning = config.budget
        
        problem_seed = list(config.seeds)[0]
        
        # Tune parameters with NEW thresholds dict
        best_params = tune_algorithm_parameters(
            algo_name=algo_name,
            base_params=base_params,
            tuning_grid=tuning_grid,
            problem=problem,
            budget=config.budget,
            max_iter=max_iter_tuning,
            thresholds=config.thresholds,  # UPDATED
            problem_seed=problem_seed,
            n_seeds_for_tuning=n_seeds_for_tuning,
            n_jobs=n_jobs
        )
        
        tuned_params[algo_name] = best_params
    
    return tuned_params


# ============================================================================
# MAIN BENCHMARK FUNCTION (UPDATED)
# ============================================================================

def run_rastrigin_benchmark(
    config_name='quick_convergence',
    output_dir='benchmark/results',
    scenario='out_of_the_box',
    n_jobs=None
):
    """
    Run Rastrigin benchmark with parallel execution.
    
    Parameters
    ----------
    config_name : str
        Configuration name from config.py
    output_dir : str
        Output directory
    scenario : str
        'out_of_the_box' (default params) or 'specialist' (tuned params)
    n_jobs : int, optional
        Number of parallel jobs
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
    
    problem_seed = getattr(config, 'problem_seed', list(config.seeds)[0])
    
    output_path = Path(output_dir)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Cannot create output directory {output_dir}: {e}")
        return
    
    if not check_disk_space(output_path, required_mb=100):
        return
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    
    print(f"=" * 80)
    print(f"Rastrigin Benchmark: {config_name}")
    print(f"Scenario: {scenario.upper()}")
    print(f"=" * 80)
    print(f"Dimension: {config.dim}")
    print(f"Budget: {config.budget} evaluations")
    print(f"Max iterations: {config.max_iter}")
    print(f"Number of runs: {len(config.seeds)}")
    print(f"Success thresholds: {config.thresholds}")  # UPDATED
    print(f"Problem seed: {problem_seed}")
    print(f"Timestamp: {timestamp}")
    
    problem = RastriginProblem(dim=config.dim)
    
    # Prepare algorithm parameters based on scenario
    if scenario == 'specialist':
        print(f"\n{'='*80}")
        print("SCENARIO: SPECIALIST (TUNING PARAMETERS)")
        print(f"{'='*80}")
        
        if n_jobs is None:
            n_jobs = max(1, mp.cpu_count() - 1)
        
        # Run tuning phase
        tuned_params = run_tuning_phase(
            config=config,
            problem=problem,
            n_seeds_for_tuning=5,
            n_jobs=n_jobs
        )
        
        # Update algo_params with tuned values
        algo_params = {
            'FA': tuned_params.get('FA', config.fa_params),
            'SA': tuned_params.get('SA', config.sa_params),
            'HC': tuned_params.get('HC', config.hc_params),
            'GA': tuned_params.get('GA', config.ga_params)
        }
        
        print(f"\n{'='*80}")
        print("TUNING COMPLETE - FINAL PARAMETERS:")
        print(f"{'='*80}")
        for algo_name, params in algo_params.items():
            print(f"\n{algo_name}:")
            for k, v in params.items():
                print(f"  {k}: {v}")
        
    else:  # out_of_the_box
        print(f"\n{'='*80}")
        print("SCENARIO: OUT-OF-THE-BOX (USING DEFAULT PARAMETERS)")
        print(f"{'='*80}")
        
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
    
    print(f"\n{'='*80}")
    print("PHASE 2: OFFICIAL BENCHMARK")
    print(f"{'='*80}")
    print(f"Using {n_jobs} parallel workers")
    
    # Run experiments for each algorithm
    for algo_name in algo_params:
        print(f"\nRunning {algo_name} ({n_runs} runs in parallel)...")
        
        params = algo_params[algo_name]
        
        # Calculate correct max_iter
        if algo_name in ['FA', 'GA']:
            pop_size = params.get('n_fireflies') or params.get('pop_size', 50)
            max_iter_actual = config.budget // pop_size
            if config.budget % pop_size != 0:
                logger.warning(f"{algo_name}: Budget {config.budget} not divisible by pop_size {pop_size}")
        else:
            max_iter_actual = config.budget
            pop_size = 1
        
        logger.info(f"{algo_name}: Using max_iter={max_iter_actual} for budget={config.budget}")
        
        # UPDATED: Prepare arguments with thresholds dict
        args_list = [
            (algo_name, problem, params, seed, max_iter_actual, config.thresholds, problem_seed, 300)
            for seed in seeds
        ]
        
        # Run in parallel
        try:
            with mp.Pool(processes=n_jobs) as pool:
                all_results = pool.starmap(run_single_experiment_safe, args_list)
        except Exception as e:
            logger.error(f"Parallel execution failed for {algo_name}: {e}")
            continue
        
        # Separate results
        successful_results = [r for r in all_results if r['status'] == 'ok']
        failed_results = [r for r in all_results if r['status'] != 'ok']
        
        status_counts = {}
        for r in all_results:
            status = r['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        avg_budget_util = np.mean([r['budget_utilization'] for r in successful_results]) if successful_results else 0.0
        
        if len(failed_results) > 0:
            logger.warning(f"{algo_name}: {len(failed_results)}/{n_runs} runs failed")
        
        # Save results with scenario in filename
        result_filename = f"rastrigin_{config_name}_{algo_name}_{scenario}_{timestamp}.json"
        result_file = output_path / result_filename
        
        # UPDATED: Save with thresholds metadata
        output_data = {
            'metadata': {
                'problem': 'rastrigin',
                'config_name': config_name,
                'algorithm': algo_name,
                'scenario': scenario,
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
                'thresholds_used': config.thresholds,  # NEW: Store thresholds
                'avg_budget_utilization': float(avg_budget_util),
                'final_params': params
            },
            'results': successful_results,
            'all_results': all_results
        }
        
        try:
            atomic_json_write(output_data, result_file)
        except Exception as e:
            logger.error(f"Failed to save results for {algo_name}: {e}")
            continue
        
        # UPDATED: Print summary with multi-level success rates
        if successful_results:
            best_fits = [r['best_fitness'] for r in successful_results]
            
            print(f"\n  Summary for {algo_name}:")
            print(f"    Mean ± Std: {np.mean(best_fits):.4f} ± {np.std(best_fits):.4f}")
            print(f"    Median: {np.median(best_fits):.4f}")
            print(f"    Best: {np.min(best_fits):.4f}")
            print(f"    Worst: {np.max(best_fits):.4f}")
            
            # NEW: Multi-level success rate breakdown
            print(f"\n    Success Rate Breakdown:")
            for level in sorted(config.thresholds.keys()):
                threshold_val = config.thresholds[level]
                count = sum(1 for r in successful_results 
                           if r['success_levels'].get(level, {}).get('success', False))
                rate = count / len(successful_results)
                
                # Calculate average hitting time for successful runs
                hit_times = [r['success_levels'].get(level, {}).get('hit_evaluations')
                            for r in successful_results
                            if r['success_levels'].get(level, {}).get('success', False)]
                hit_times = [t for t in hit_times if t is not None]
                avg_hit = np.mean(hit_times) if hit_times else 0
                
                print(f"      {level.capitalize()} (<{threshold_val:.1f}): {rate:.2%} " +
                     (f"(avg hit: {avg_hit:.0f} evals)" if hit_times else ""))
            
            print(f"\n    Avg time: {np.mean([r['elapsed_time'] for r in successful_results]):.2f}s")
            print(f"    Budget util: {avg_budget_util:.2%}")
        
        status_str = ", ".join([f"{count} {status}" for status, count in status_counts.items()])
        print(f"    Status breakdown: {status_str}")
    
    print(f"\n{'=' * 80}")
    print(f"Benchmark complete! Results saved to: {output_path}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Rastrigin benchmark')
    parser.add_argument('--config', type=str, default='quick_convergence',
                        choices=['quick_convergence', 'multimodal_escape', 'scalability', 'all'],
                        help='Benchmark configuration')
    parser.add_argument('--output', type=str, default='benchmark/results/rastrigin',
                        help='Output directory')
    parser.add_argument('--scenario', type=str, default='all',
                        choices=['out_of_the_box', 'specialist', 'all'],
                        help="Benchmark scenario: 'out_of_the_box' (default params), 'specialist' (tuned params), or 'all'")
    parser.add_argument('--jobs', type=int, default=None,
                        help='Number of parallel jobs (default: CPU count - 1)')
    
    args = parser.parse_args()
    
    # Determine which scenarios to run
    if args.scenario == 'all':
        scenarios_to_run = ['out_of_the_box', 'specialist']
    else:
        scenarios_to_run = [args.scenario]
    
    # Determine which configs to run
    if args.config == 'all':
        configs_to_run = ['quick_convergence', 'multimodal_escape', 'scalability']
    else:
        configs_to_run = [args.config]
    
    # Run all combinations
    for config_name in configs_to_run:
        for scenario in scenarios_to_run:
            print(f"\n{'#'*80}")
            print(f"# Config: {config_name} | Scenario: {scenario}")
            print(f"{'#'*80}\n")
            
            run_rastrigin_benchmark(
                config_name=config_name,
                output_dir=args.output,
                scenario=scenario,
                n_jobs=args.jobs
            )