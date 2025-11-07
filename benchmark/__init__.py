"""
Benchmark module for comparing optimization algorithms.
"""

from .run_rastrigin import run_rastrigin_benchmark
from .run_knapsack import run_knapsack_benchmark
from .analyze_results import analyze_benchmark, analyze_all_benchmarks

__all__ = [
    'run_rastrigin_benchmark',
    'run_knapsack_benchmark',
    'analyze_benchmark',
    'analyze_all_benchmarks'
]
