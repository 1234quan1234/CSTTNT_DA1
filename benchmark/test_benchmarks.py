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

from benchmark.run_rastrigin import run_rastrigin_benchmark
from benchmark.run_knapsack import run_knapsack_benchmark
from benchmark.config import RASTRIGIN_CONFIGS, KNAPSACK_CONFIGS


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
        run_knapsack_benchmark(config_name='small', output_dir=str(tmp_path))
        assert (tmp_path / 'small').exists()
    
    def test_all_algorithms_produce_results(self, tmp_path):
        """Test that all algorithms produce results."""
        run_knapsack_benchmark(config_name='small', output_dir=str(tmp_path))
        result_dir = tmp_path / 'small'
        
        for algo in ['FA', 'SA', 'HC', 'GA']:
            assert (result_dir / f'{algo}_results.json').exists()


class TestBenchmarkConfigs:
    """Test benchmark configurations."""
    
    def test_rastrigin_configs_valid(self):
        """Test that all Rastrigin configs are valid."""
        for name, config in RASTRIGIN_CONFIGS.items():
            assert config.dim > 0
            assert config.budget > 0
            assert config.max_iter > 0
            assert len(config.seeds) > 0
    
    
    def test_knapsack_configs_valid(self):
        """Test that all Knapsack configs are valid."""
        for name, config in KNAPSACK_CONFIGS.items():
            assert config.n_items > 0
            assert config.capacity > 0
            assert config.budget > 0
            assert config.max_iter > 0
            assert len(config.seeds) > 0


def run_quick_tests():
    """Run quick sanity tests."""
    print("Running quick sanity tests...")
    
    test_dir = Path('benchmark/results/test_tmp')
    test_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Test Rastrigin
        print("\n1. Testing Rastrigin benchmark...")
        run_rastrigin_benchmark(config_name='quick_convergence', output_dir=str(test_dir / 'rastrigin'))
        print("   ✓ Rastrigin benchmark passed")
        
        # Test Knapsack
        print("\n2. Testing Knapsack benchmark...")
        run_knapsack_benchmark(config_name='small', output_dir=str(test_dir / 'knapsack'))
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
    
    args = parser.parse_args()
    
    if args.pytest:
        pytest.main([__file__, '-v'])
    else:
        run_quick_tests()
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test benchmarks')
    parser.add_argument('--pytest', action='store_true',
                        help='Run with pytest')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick sanity tests')
    
    args = parser.parse_args()
    
    if args.pytest:
        pytest.main([__file__, '-v'])
    else:
        run_quick_tests()
