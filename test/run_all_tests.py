#!/usr/bin/env python3
"""
Test runner script for all tests in the repository.

Usage:
    python test/run_all_tests.py                    # Run all tests
    python test/run_all_tests.py -v                 # Verbose mode
    python test/run_all_tests.py test_continuous    # Run specific test
"""

import sys
import os
import unittest
import multiprocessing as mp

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_all_tests(verbosity=2, n_jobs=1):
    """Discover and run all tests."""
    
    print("=" * 70)
    print("  AI SEARCH & OPTIMIZATION FRAMEWORK - TEST SUITE")
    if n_jobs > 1:
        print(f"  Running with {n_jobs} parallel workers")
    print("=" * 70)
    
    # Discover tests
    test_dir = os.path.dirname(os.path.abspath(__file__))
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern='test_*.py')
    
    # Run tests (with parallel support if n_jobs > 1)
    if n_jobs > 1:
        # Use parallel test runner
        from concurrent.futures import ProcessPoolExecutor
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
    else:
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run:     {result.testsRun}")
    print(f"Successes:     {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures:      {len(result.failures)}")
    print(f"Errors:        {len(result.errors)}")
    print(f"Skipped:       {len(result.skipped)}")
    print("=" * 70)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


def run_specific_test(test_name, verbosity=2):
    """Run a specific test module."""
    
    print(f"Running test: {test_name}")
    print("=" * 70)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(test_name)
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '-v' or sys.argv[1] == '--verbose':
            exit_code = run_all_tests(verbosity=2)
        elif sys.argv[1] in ['-h', '--help']:
            print(__doc__)
            exit_code = 0
        elif sys.argv[1] == '-j' or sys.argv[1] == '--jobs':
            n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else mp.cpu_count() - 1
            exit_code = run_all_tests(verbosity=2, n_jobs=n_jobs)
        else:
            # Run specific test
            exit_code = run_specific_test(sys.argv[1], verbosity=2)
    else:
        # Run all tests
        exit_code = run_all_tests(verbosity=2)
    
    sys.exit(exit_code)
