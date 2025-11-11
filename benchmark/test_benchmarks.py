"""
Test suite for benchmarks - verify all components work correctly.
Includes validation for new metadata/status tracking features.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from pathlib import Path
import shutil
from multiprocessing import Pool
import glob
import json

from benchmark.run_rastrigin import run_rastrigin_benchmark
from benchmark.run_knapsack import run_knapsack_benchmark
from benchmark.config import RASTRIGIN_CONFIGS, KNAPSACK_CONFIGS


def get_rastrigin_configs():
    """Get Rastrigin configurations."""
    return list(RASTRIGIN_CONFIGS.values())


def get_knapsack_configs():
    """Get Knapsack configurations."""
    return list(KNAPSACK_CONFIGS.values())


def find_result_files(output_dir, config_name, algo_name):
    """Find all result files matching the pattern."""
    pattern = str(Path(output_dir) / f"rastrigin_{config_name}_{algo_name}_*.json")
    return glob.glob(pattern)


def validate_result_file(filepath, expected_config, expected_algo, expected_dim, expected_budget):
    """
    Validate a single result file.
    
    Checks:
    - JSON structure (metadata + all_results)
    - Required metadata fields
    - Status tracking (ok/timeout/nan/numerical_error)
    - Budget utilization (should be ±5% of 1.0)
    - All required fields in each run
    - No NaN/Inf in convergence history
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Validate structure
    assert 'metadata' in data, f"Missing metadata in {filepath}"
    assert 'results' in data, f"Missing results in {filepath}"
    
    # Validate metadata
    metadata = data['metadata']
    
    # Validate metadata
    assert metadata['config_name'] == expected_config, \
        f"Config mismatch: expected {expected_config}, got {metadata['config_name']}"
    assert metadata['algorithm'] == expected_algo, \
        f"Algorithm mismatch: expected {expected_algo}, got {metadata['algorithm']}"
    assert metadata['dimension'] == expected_dim, \
        f"Dimension mismatch: expected {expected_dim}, got {metadata['dimension']}"
    
    # Validate status breakdown
    assert 'status_breakdown' in metadata, "Missing status_breakdown in metadata"
    status_counts = metadata['status_breakdown']
    assert isinstance(status_counts, dict), "status_breakdown must be dict"
    assert sum(status_counts.values()) == len(data['all_results']), \
        "status_breakdown counts don't match results"
    
    # Validate results
    results = data['results']
    assert len(results) > 0, f"No results in {filepath}"
    
    for i, result in enumerate(data['all_results']):
        # Check required fields including NEW ones
        required_fields = [
            'algorithm', 'seed', 'best_fitness', 'history', 
            'evaluations', 'budget', 'budget_utilization',
            'status', 'error_type', 'error_msg',  # NEW tracking fields
            'hit_evaluations'  # NEW hitting time field
        ]
        for field in required_fields:
            assert field in result, f"Missing field '{field}' in result {i}"
        
        # Validate budget utilization (±5% tolerance)
        budget_util = result['budget_utilization']
        assert 0.95 <= budget_util <= 1.05, \
            f"Run {i}: Budget utilization {budget_util:.2%} outside ±5% tolerance"
        
        # Validate actual evaluations
        actual_evals = result['actual_evaluations']
        expected_evals = result['budget']
        diff_pct = abs(actual_evals - expected_evals) / expected_evals * 100
        assert diff_pct <= 5, \
            f"Run {i}: Actual evaluations {actual_evals} differs from budget {expected_evals} by {diff_pct:.2f}%"
        
        # Validate history is non-empty and has no NaN/Inf
        history = result['history']
        assert len(history) > 0, f"Run {i}: Empty convergence history"
        assert all(not (h != h or h == float('inf') or h == float('-inf')) for h in history), \
            f"Run {i}: NaN or Inf in convergence history"
        
        # Validate best_fitness is finite
        fitness = result['best_fitness']
        assert fitness == fitness and fitness != float('inf') and fitness != float('-inf'), \
            f"Run {i}: Invalid best_fitness {fitness}"
        
        # Validate status field (NEW)
        valid_statuses = ['ok', 'timeout', 'nan', 'numerical_error', 'memory', 'invalid_history']
        assert result['status'] in valid_statuses, \
            f"Run {i}: Invalid status '{result['status']}'"
        
        # If status is 'ok', error_type and error_msg should be null
        if result['status'] == 'ok':
            assert result['error_type'] is None, \
                f"Run {i}: Status is 'ok' but error_type is '{result['error_type']}'"
            assert result['error_msg'] is None, \
                f"Run {i}: Status is 'ok' but error_msg is present"
        else:
            # If status is not 'ok', should have error info
            assert result['error_type'] is not None, \
                f"Run {i}: Status is '{result['status']}' but error_type is null"
            assert isinstance(result['error_msg'], str), \
                f"Run {i}: error_msg should be string, got {type(result['error_msg'])}"


class TestRastriginBenchmark:
    """Test Rastrigin benchmark."""
    
    def test_quick_convergence(self, tmp_path):
        """Test quick convergence config runs without errors."""
        run_rastrigin_benchmark(config_name='quick_convergence', output_dir=str(tmp_path), n_jobs=2)
        
        # Find result files with new naming convention
        for algo in ['FA', 'SA', 'HC', 'GA']:
            result_files = find_result_files(tmp_path, 'quick_convergence', algo)
            assert len(result_files) > 0, f"No result files found for {algo}"
    
    def test_all_algorithms_produce_results(self, tmp_path):
        """Test that all algorithms produce results."""
        run_rastrigin_benchmark(config_name='quick_convergence', output_dir=str(tmp_path), n_jobs=2)
        
        for algo in ['FA', 'SA', 'HC', 'GA']:
            result_files = find_result_files(tmp_path, 'quick_convergence', algo)
            assert len(result_files) > 0, f"No result files found for {algo}"
            
            # Validate the most recent file
            latest_file = max(result_files, key=lambda x: Path(x).stat().st_mtime)
            validate_result_file(
                latest_file,
                expected_config='quick_convergence',
                expected_algo=algo,
                expected_dim=2,
                expected_budget=3000
            )
    
    @pytest.mark.parametrize("config_name,expected_dim,expected_budget", [
        ("quick_convergence", 2, 3000),
        ("multimodal_escape", 5, 10000),
        ("scalability", 10, 30000),
    ])
    def test_rastrigin_configs(self, config_name, expected_dim, expected_budget, tmp_path):
        """Test different Rastrigin configurations."""
        run_rastrigin_benchmark(config_name=config_name, output_dir=str(tmp_path), n_jobs=2)
        
        for algo in ['FA', 'SA', 'HC', 'GA']:
            result_files = find_result_files(tmp_path, config_name, algo)
            assert len(result_files) > 0, f"No result files found for {algo} in {config_name}"
            
            # Validate latest file
            latest_file = max(result_files, key=lambda x: Path(x).stat().st_mtime)
            validate_result_file(
                latest_file,
                expected_config=config_name,
                expected_algo=algo,
                expected_dim=expected_dim,
                expected_budget=expected_budget
            )
    
    def test_failed_runs_handling(self, tmp_path):
        """Test that failed runs are logged properly."""
        run_rastrigin_benchmark(config_name='quick_convergence', output_dir=str(tmp_path), n_jobs=2)
        
        # Check if failed runs log exists
        failed_logs = list(tmp_path.glob('failed_runs_*.json'))
        if failed_logs:
            with open(failed_logs[0], 'r') as f:
                data = json.load(f)
                assert 'failed_runs' in data
                for failed_run in data['failed_runs']:
                    assert 'algorithm' in failed_run
                    assert 'seed' in failed_run
                    assert 'error' in failed_run


def run_quick_tests(parallel=False, num_workers=4):
    """Run quick sanity tests without pytest."""
    test_dir = Path('benchmark/results/quick_test')
    test_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("=" * 70)
        print("RUNNING QUICK BENCHMARK TESTS")
        print("=" * 70)
        
        if parallel:
            # Parallel execution
            print("\n1. Testing Rastrigin benchmark (parallel)...")
            run_rastrigin_benchmark(
                config_name='quick_convergence',
                output_dir=str(test_dir / 'rastrigin'),
                n_jobs=num_workers
            )
            print("   ✓ Rastrigin benchmark passed")
            
            print("\n2. Testing Knapsack benchmark (parallel)...")
            run_knapsack_benchmark(
                size=50,
                instance_type='uncorrelated',
                output_dir=str(test_dir / 'knapsack'),
                n_jobs=num_workers
            )
            print("   ✓ Knapsack benchmark passed")
        else:
            # Sequential execution
            print("\n1. Testing Rastrigin benchmark...")
            run_rastrigin_benchmark(
                config_name='quick_convergence',
                output_dir=str(test_dir / 'rastrigin'),
                n_jobs=2
            )
            print("   ✓ Rastrigin benchmark passed")
            
            print("\n2. Testing Knapsack benchmark...")
            run_knapsack_benchmark(
                size=50,
                instance_type='uncorrelated',
                output_dir=str(test_dir / 'knapsack'),
                n_jobs=2
            )
            print("   ✓ Knapsack benchmark passed")
        
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED!")
        print("=" * 70)
        
    finally:
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test benchmarks')
    parser.add_argument('--pytest', action='store_true',
                        help='Run with pytest')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick sanity tests')
    parser.add_argument('--parallel', action='store_true',
                        help='Run benchmarks in parallel')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    
    args = parser.parse_args()
    
    if args.pytest:
        pytest.main([__file__, '-v'])
    else:
        run_quick_tests(parallel=args.parallel, num_workers=args.workers)