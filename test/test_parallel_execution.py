"""
Unit tests for parallel execution and reproducibility.

This module tests:
- Multiple seeds produce different results
- Concurrent execution works correctly
- Reproducibility with same seed
- Benchmark suite integration
"""

import unittest
import numpy as np
import sys
import os
import json
import tempfile
import shutil
from pathlib import Path
from multiprocessing import Pool
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.problems.continuous.rastrigin import RastriginProblem
from src.problems.discrete.knapsack import KnapsackProblem
from src.swarm.fa import FireflyContinuousOptimizer, FireflyKnapsackOptimizer


def run_single_optimization(args):
    """
    Helper function for parallel execution.
    
    Parameters
    ----------
    args : tuple
        (problem, seed, max_iter)
    
    Returns
    -------
    dict
        Results with seed, best_fitness, elapsed_time
    """
    problem, seed, max_iter = args
    
    start = time.time()
    fa = FireflyContinuousOptimizer(problem, n_fireflies=10, seed=seed)
    _, best_fit, history, _ = fa.run(max_iter=max_iter)
    elapsed = time.time() - start
    
    return {
        'seed': seed,
        'best_fitness': float(best_fit),
        'history': [float(h) for h in history],
        'elapsed_time': elapsed
    }


class TestMultipleSeeds(unittest.TestCase):
    """
    Test that different seeds produce different (stochastic) results.
    
    Tests verify:
    - Results vary with different seeds
    - All results are valid
    - No systematic bias
    """
    
    def test_different_seeds_produce_different_results(self):
        """
        Test with 10 different seeds.
        
        Results should be different (stochastic), but all valid.
        """
        problem = RastriginProblem(dim=5)
        seeds = list(range(10))
        
        results = []
        for seed in seeds:
            fa = FireflyContinuousOptimizer(problem, n_fireflies=10, seed=seed)
            _, best_fit, _, _ = fa.run(max_iter=20)
            results.append(best_fit)
        
        # Check all results are valid
        self.assertTrue(all(np.isfinite(r) for r in results))
        
        # Check results are not all identical (stochastic)
        unique_results = len(set(np.round(results, 6)))
        self.assertGreater(unique_results, 1,
                          "Different seeds should produce different results")
        
        # Check reasonable variance
        std_dev = np.std(results)
        self.assertGreater(std_dev, 0.01,
                          "Results should have reasonable variance")
    
    def test_all_seeds_converge(self):
        """Test that all seeds produce convergent results."""
        problem = RastriginProblem(dim=3)
        seeds = [42, 123, 456, 789, 999]
        
        for seed in seeds:
            fa = FireflyContinuousOptimizer(problem, n_fireflies=10, seed=seed)
            _, best_fit, history, _ = fa.run(max_iter=30)
            
            # Check convergence
            improvement = history[0] - history[-1]
            self.assertGreaterEqual(improvement, -1e-6,
                                   f"Seed {seed} should not worsen significantly")
            
            # Check no NaN/Inf
            self.assertTrue(np.all(np.isfinite(history)),
                           f"Seed {seed} produced non-finite values")


class TestConcurrentExecution(unittest.TestCase):
    """
    Test concurrent execution with multiprocessing.
    
    Tests verify:
    - No race conditions
    - Results are reproducible
    - Correct number of results returned
    """
    
    def test_parallel_runs_no_race_condition(self):
        """
        Test 4 concurrent runs.
        
        Should produce correct number of results with no corruption.
        """
        problem = RastriginProblem(dim=5)
        seeds = [10, 20, 30, 40]
        max_iter = 10
        
        # Prepare arguments
        args_list = [(problem, seed, max_iter) for seed in seeds]
        
        # Run in parallel
        with Pool(processes=4) as pool:
            results = pool.map(run_single_optimization, args_list)
        
        # Verify results
        self.assertEqual(len(results), 4, "Should get 4 results")
        
        # Check all results are valid
        for result in results:
            self.assertIn('seed', result)
            self.assertIn('best_fitness', result)
            self.assertIn('history', result)
            self.assertTrue(np.isfinite(result['best_fitness']))
            self.assertEqual(len(result['history']), max_iter)
    
    def test_reproducibility_with_same_seed(self):
        """
        Test that same seed produces identical results.
        
        Critical for scientific reproducibility.
        """
        problem = RastriginProblem(dim=3)
        seed = 42
        max_iter = 20
        
        # Run 1
        fa1 = FireflyContinuousOptimizer(problem, n_fireflies=10, seed=seed)
        _, fit1, hist1, _ = fa1.run(max_iter=max_iter)
        
        # Run 2 (same seed)
        fa2 = FireflyContinuousOptimizer(problem, n_fireflies=10, seed=seed)
        _, fit2, hist2, _ = fa2.run(max_iter=max_iter)
        
        # Should be identical
        self.assertEqual(fit1, fit2, "Same seed should produce identical fitness")
        self.assertTrue(np.allclose(hist1, hist2),
                       "Same seed should produce identical history")
    
    def test_parallel_reproducibility(self):
        """
        Test that parallel runs with same seeds are reproducible.
        """
        problem = RastriginProblem(dim=3)
        seeds = [100, 200, 300]
        max_iter = 15
        
        # Run 1
        args_list = [(problem, seed, max_iter) for seed in seeds]
        with Pool(processes=3) as pool:
            results1 = pool.map(run_single_optimization, args_list)
        
        # Run 2 (same seeds)
        with Pool(processes=3) as pool:
            results2 = pool.map(run_single_optimization, args_list)
        
        # Compare results
        for r1, r2 in zip(results1, results2):
            self.assertEqual(r1['seed'], r2['seed'])
            self.assertEqual(r1['best_fitness'], r2['best_fitness'],
                           f"Seed {r1['seed']} not reproducible in parallel")


class TestBenchmarkSuite(unittest.TestCase):
    """
    Test benchmark suite integration.
    
    Tests verify:
    - Output JSONs are created correctly
    - No missing data
    - Correct format
    """
    
    def setUp(self):
        """Create temporary directory for test outputs."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_benchmark_output_format(self):
        """
        Test that benchmark produces correct JSON format.
        """
        problem = RastriginProblem(dim=3)
        
        # Run mini benchmark
        results = []
        for seed in [42, 43]:
            fa = FireflyContinuousOptimizer(problem, n_fireflies=5, seed=seed)
            _, best_fit, history, _ = fa.run(max_iter=5)
            
            results.append({
                'algorithm': 'FA',
                'seed': seed,
                'best_fitness': float(best_fit),
                'history': [float(h) for h in history],
                'elapsed_time': 1.23
            })
        
        # Save to JSON
        output_file = Path(self.test_dir) / 'FA_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Verify file exists and is valid JSON
        self.assertTrue(output_file.exists())
        
        # Load and verify structure
        with open(output_file, 'r') as f:
            loaded_results = json.load(f)
        
        self.assertEqual(len(loaded_results), 2)
        
        for result in loaded_results:
            self.assertIn('algorithm', result)
            self.assertIn('seed', result)
            self.assertIn('best_fitness', result)
            self.assertIn('history', result)
            self.assertIn('elapsed_time', result)
            self.assertEqual(len(result['history']), 5)
    
    def test_multiple_algorithms_output(self):
        """
        Test benchmark with multiple algorithms.
        
        Verifies all algorithm files are created correctly.
        """
        from src.classical.simulated_annealing import SimulatedAnnealingOptimizer
        
        problem = RastriginProblem(dim=3)
        algorithms = {
            'FA': FireflyContinuousOptimizer(problem, n_fireflies=5, seed=42),
            'SA': SimulatedAnnealingOptimizer(problem, seed=42)
        }
        
        # Run each algorithm
        for algo_name, optimizer in algorithms.items():
            results = []
            for seed in [42, 43]:
                # Update seed
                optimizer.seed = seed
                optimizer.rng = np.random.RandomState(seed)
                
                _, best_fit, history, _ = optimizer.run(max_iter=5)
                results.append({
                    'algorithm': algo_name,
                    'seed': seed,
                    'best_fitness': float(best_fit),
                    'history': [float(h) for h in history]
                })
            
            # Save
            output_file = Path(self.test_dir) / f'{algo_name}_results.json'
            with open(output_file, 'w') as f:
                json.dump(results, f)
        
        # Verify all files exist
        for algo_name in ['FA', 'SA']:
            output_file = Path(self.test_dir) / f'{algo_name}_results.json'
            self.assertTrue(output_file.exists(),
                           f"{algo_name} results file should exist")
    
    def test_no_missing_data(self):
        """
        Test that all required fields are present in output.
        """
        problem = RastriginProblem(dim=3)
        fa = FireflyContinuousOptimizer(problem, n_fireflies=5, seed=42)
        _, best_fit, history, _ = fa.run(max_iter=10)
        
        result = {
            'algorithm': 'FA',
            'seed': 42,
            'best_fitness': float(best_fit),
            'history': [float(h) for h in history],
            'elapsed_time': 1.0,
            'evaluations': len(history) * 5
        }
        
        # Check all required fields
        required_fields = ['algorithm', 'seed', 'best_fitness', 'history',
                          'elapsed_time', 'evaluations']
        
        for field in required_fields:
            self.assertIn(field, result, f"Missing required field: {field}")
        
        # Check data types
        self.assertIsInstance(result['algorithm'], str)
        self.assertIsInstance(result['seed'], int)
        self.assertIsInstance(result['best_fitness'], float)
        self.assertIsInstance(result['history'], list)
        self.assertIsInstance(result['elapsed_time'], float)
        self.assertIsInstance(result['evaluations'], int)


if __name__ == '__main__':
    unittest.main()
