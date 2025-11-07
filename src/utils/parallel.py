"""
Parallel execution utilities for benchmarks.
"""

import multiprocessing as mp
from typing import Callable, List, Any, Optional
import numpy as np


def run_parallel(func: Callable, args_list: List[tuple], n_jobs: Optional[int] = None) -> List[Any]:
    """
    Run function in parallel with different arguments.
    
    Parameters
    ----------
    func : Callable
        Function to execute in parallel.
    args_list : List[tuple]
        List of argument tuples for each function call.
    n_jobs : int, optional
        Number of parallel jobs. If None, uses CPU count.
    
    Returns
    -------
    results : List[Any]
        List of results from each function call.
    """
    if n_jobs is None:
        n_jobs = mp.cpu_count()
    
    with mp.Pool(processes=n_jobs) as pool:
        results = pool.starmap(func, args_list)
    
    return results


def run_single_experiment(optimizer_class, problem, params, seed, max_iter):
    """
    Run a single optimization experiment (for parallel execution).
    
    Parameters
    ----------
    optimizer_class : class
        Optimizer class to instantiate.
    problem : ProblemBase
        Problem instance.
    params : dict
        Parameters for optimizer.
    seed : int
        Random seed.
    max_iter : int
        Maximum iterations.
    
    Returns
    -------
    result : dict
        Dictionary containing seed, best_fitness, and history.
    """
    import time
    
    optimizer = optimizer_class(problem=problem, seed=seed, **params)
    
    start_time = time.time()
    _, best_fitness, history, _ = optimizer.run(max_iter=max_iter)
    elapsed = time.time() - start_time
    
    return {
        'seed': seed,
        'best_fitness': best_fitness,
        'history': history,
        'elapsed_time': elapsed
    }
