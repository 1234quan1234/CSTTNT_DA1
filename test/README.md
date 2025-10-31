# Test Suite

This folder contains comprehensive unit tests for the AI Search & Optimization Framework.

## Test Structure

```
test/
├── __init__.py                      # Package initialization
├── test_continuous_problems.py      # Tests for continuous problems (Sphere, Rastrigin)
├── test_tsp_problem.py             # Tests for TSP problem
├── test_firefly_algorithm.py       # Tests for Firefly Algorithm (continuous & discrete)
├── test_classical_algorithms.py    # Tests for classical algorithms (HC, SA, GA)
├── run_all_tests.py                # Main test runner
└── README.md                       # This file
```

## Running Tests

### Run All Tests
```bash
# From project root
python test/run_all_tests.py

# Or with verbose output
python test/run_all_tests.py -v
```

### Run Specific Test Module
```bash
python test/run_all_tests.py test_continuous_problems
python test/run_all_tests.py test_firefly_algorithm
```

### Run Individual Test File
```bash
python -m unittest test.test_continuous_problems
python -m unittest test.test_tsp_problem
```

### Run Specific Test Class or Method
```bash
python -m unittest test.test_continuous_problems.TestSphereProblem
python -m unittest test.test_continuous_problems.TestSphereProblem.test_optimum_value
```

## Test Coverage

### Continuous Problems (`test_continuous_problems.py`)
- ✓ Sphere function dimension and bounds
- ✓ Optimum value verification
- ✓ Fitness calculation correctness
- ✓ Random solution generation
- ✓ Rastrigin function properties

### TSP Problem (`test_tsp_problem.py`)
- ✓ City count and coordinates
- ✓ Distance matrix calculation and symmetry
- ✓ Tour length calculation
- ✓ Random tour generation
- ✓ Invalid tour handling

### Firefly Algorithm (`test_firefly_algorithm.py`)
- ✓ Continuous optimizer initialization
- ✓ Output format validation
- ✓ Convergence behavior
- ✓ Deterministic results with seed
- ✓ Discrete TSP optimizer
- ✓ Valid tour generation

### Classical Algorithms (`test_classical_algorithms.py`)
- ✓ Hill Climbing functionality
- ✓ Simulated Annealing with temperature schedule
- ✓ Genetic Algorithm with crossover and mutation
- ✓ Convergence for all algorithms

## Adding New Tests

To add tests for a new component:

1. Create a new test file: `test_your_component.py`
2. Import unittest and your component
3. Create test class inheriting from `unittest.TestCase`
4. Write test methods (must start with `test_`)
5. Run tests to verify

Example:
```python
import unittest
from src.your_module import YourClass

class TestYourClass(unittest.TestCase):
    def setUp(self):
        self.obj = YourClass()
    
    def test_something(self):
        result = self.obj.method()
        self.assertEqual(result, expected_value)
```

## Test Best Practices

1. **Isolation**: Each test should be independent
2. **Determinism**: Use seeds for reproducible results
3. **Coverage**: Test normal cases, edge cases, and error cases
4. **Clarity**: Use descriptive test names
5. **Speed**: Keep tests fast (use small problem sizes)

## Continuous Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: python test/run_all_tests.py
```

## Troubleshooting

**Import errors**: Make sure you're running from the project root or have the correct PYTHONPATH set.

**Missing dependencies**: Install required packages:
```bash
pip install numpy
```

**Failed tests**: Check the error message and traceback for details. Ensure all source files are present and correct.
