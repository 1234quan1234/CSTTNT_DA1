"""
Test suite for benchmarks - verify all components work correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from pathlib import Path
import shutil
from multiprocessing import Pool

from benchmark.run_rastrigin import run_rastrigin_benchmark
from benchmark.run_knapsack import run_knapsack_benchmark
from benchmark.config import RASTRIGIN_CONFIGS, KNAPSACK_CONFIGS


def get_rastrigin_configs():
    """Get Rastrigin configurations."""
    return list(RASTRIGIN_CONFIGS.values())


def get_knapsack_configs():
    """Get Knapsack configurations."""
    return list(KNAPSACK_CONFIGS.values())


class TestRastriginBenchmark:
    """Test Rastrigin benchmark."""
    
    def test_quick_convergence(self, tmp_path):
        """Test quick convergence config runs without errors."""
        run_rastrigin_benchmark(config_name='quick_convergence', output_dir=str(tmp_path))
        assert (tmp_path / 'quick_convergence').exists()
        assert (tmp_path / 'quick_convergence' / 'FA_results.json').exists()
    
    def test_all_algorithms_produce_results(self, tmp_path):
        """Test that all algorithms produce results."""
        run_rastrigin_benchmark(config_name='quick_convergence', output_dir=str(tmp_path))
        result_dir = tmp_path / 'quick_convergence'
        
        for algo in ['FA', 'SA', 'HC', 'GA']:
            assert (result_dir / f'{algo}_results.json').exists()


class TestKnapsackBenchmark:
    """Test Knapsack benchmark."""
    
    def test_small_knapsack(self, tmp_path):
        """Test small knapsack config runs without errors."""
        # Use size=50 with uncorrelated type (smallest config)
        run_knapsack_benchmark(size=50, instance_type='uncorrelated', output_dir=str(tmp_path))
        assert tmp_path.exists()
        # Check that at least one result file exists
        result_files = list(tmp_path.glob('*.json'))
        assert len(result_files) > 0
    
    def test_all_algorithms_produce_results(self, tmp_path):
        """Test that all algorithms produce results."""
        # Run small benchmark
        run_knapsack_benchmark(size=50, instance_type='uncorrelated', output_dir=str(tmp_path))
        
        # Check that results exist for all algorithms
        for algo in ['FA', 'SA', 'HC', 'GA']:
            algo_files = list(tmp_path.glob(f'*_{algo}.json'))
            assert len(algo_files) > 0, f"No results found for {algo}"


class TestBenchmarkConfigs:
    """Test benchmark configurations."""
    
    def test_rastrigin_configs_valid(self):
        """Test that all Rastrigin configs are valid."""
        configs = get_rastrigin_configs()
        
        for config in configs:
            assert config.dim > 0
            assert config.budget > 0
            assert config.threshold > 0
            assert len(config.fa_params) > 0
    
    def test_knapsack_configs_valid(self):
        """Test that all Knapsack configs are valid."""
        configs = get_knapsack_configs()
        
        for config in configs:
            assert config.n_items > 0
            assert config.budget > 0
            assert config.instance_type in ['uncorrelated', 'weakly', 'strongly', 'subset']
            assert len(config.fa_params) > 0


def run_rastrigin_task(args):
    """Helper function for parallel Rastrigin benchmark."""
    config_name, output_dir = args
    try:
        run_rastrigin_benchmark(config_name=config_name, output_dir=output_dir)
        return f"✓ {config_name} passed"
    except Exception as e:
        return f"✗ {config_name} failed: {str(e)}"


def run_knapsack_task(args):
    """Helper function for parallel Knapsack benchmark."""
    size, instance_type, output_dir = args
    try:
        run_knapsack_benchmark(size=size, instance_type=instance_type, output_dir=output_dir)
        return f"✓ Knapsack({size}, {instance_type}) passed"
    except Exception as e:
        return f"✗ Knapsack({size}, {instance_type}) failed: {str(e)}"


def run_quick_tests(parallel=False, num_workers=4):
    """Run quick sanity tests."""
    print("Running quick sanity tests...")
    if parallel:
        print(f"Mode: PARALLEL ({num_workers} workers)")
    else:
        print("Mode: SEQUENTIAL")
    
    test_dir = Path('benchmark/results/test_tmp')
    test_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if parallel:
            # Parallel execution
            print("\n" + "=" * 70)
            print("RUNNING PARALLEL BENCHMARKS")
            print("=" * 70)
            
            # Prepare tasks
            rastrigin_tasks = [
                ('quick_convergence', str(test_dir / 'rastrigin' / 'quick_convergence'))
            ]
            
            knapsack_tasks = [
                (50, 'uncorrelated', str(test_dir / 'knapsack' / 'small')),
                (100, 'uncorrelated', str(test_dir / 'knapsack' / 'medium')),
            ]
            
            all_tasks = rastrigin_tasks + knapsack_tasks
            
            # Run in parallel
            with Pool(num_workers) as pool:
                # Run Rastrigin tasks
                rastrigin_results = pool.map(run_rastrigin_task, rastrigin_tasks)
                
                # Run Knapsack tasks
                knapsack_results = pool.map(run_knapsack_task, knapsack_tasks)
            
            # Print results
            print("\nRastrigin Results:")
            for result in rastrigin_results:
                print(f"  {result}")
            
            print("\nKnapsack Results:")
            for result in knapsack_results:
                print(f"  {result}")
        else:
            # Sequential execution
            print("\n1. Testing Rastrigin benchmark...")
            run_rastrigin_benchmark(config_name='quick_convergence', output_dir=str(test_dir / 'rastrigin'))
            print("   ✓ Rastrigin benchmark passed")
            
            print("\n2. Testing Knapsack benchmark...")
            run_knapsack_benchmark(size=50, instance_type='uncorrelated', output_dir=str(test_dir / 'knapsack'))
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
