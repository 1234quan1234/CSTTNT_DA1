"""
Unit tests for utility functions.

Tests cover:
- Euclidean distance calculations
- Brightness computation
- Solution extraction
- Permutation repair
- Edge cases and numerical stability
"""

import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.utils import (
    euclidean_distance_matrix,
    get_best_fitness_index,
    get_best_solution,
    compute_brightness,
    repair_permutation
)


class TestEuclideanDistance(unittest.TestCase):
    """
    Test Euclidean distance matrix computation.
    
    Tests cover:
    - 2D, 3D, nD points
    - Numpy arrays vs lists
    - Distance to itself = 0
    - Symmetry
    """
    
    def test_2d_points(self):
        """Test with 2D points."""
        points = np.array([[0, 0], [1, 0], [0, 1]])
        dist = euclidean_distance_matrix(points)
        
        # Check shape
        self.assertEqual(dist.shape, (3, 3))
        
        # Check specific distances
        self.assertAlmostEqual(dist[0, 1], 1.0, places=10)
        self.assertAlmostEqual(dist[0, 2], 1.0, places=10)
        self.assertAlmostEqual(dist[1, 2], np.sqrt(2), places=10)
        
        # Check diagonal (distance to self)
        self.assertTrue(np.allclose(np.diag(dist), 0.0))
        
        # Check symmetry
        self.assertTrue(np.allclose(dist, dist.T))
    
    def test_3d_points(self):
        """Test with 3D points."""
        points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        dist = euclidean_distance_matrix(points)
        
        # Distance from [0,0,0] to [1,1,1]
        expected = np.sqrt(3)
        self.assertAlmostEqual(dist[0, 1], expected, places=10)
        
        # Distance from [1,1,1] to [2,2,2]
        self.assertAlmostEqual(dist[1, 2], expected, places=10)
        
        # Diagonal should be zero
        self.assertTrue(np.allclose(np.diag(dist), 0.0))
    
    def test_high_dimensional(self):
        """Test with high-dimensional points."""
        n_points = 10
        dim = 50
        points = np.random.rand(n_points, dim)
        
        dist = euclidean_distance_matrix(points)
        
        # Check shape
        self.assertEqual(dist.shape, (n_points, n_points))
        
        # Check properties
        self.assertTrue(np.all(dist >= 0), "Distances should be non-negative")
        self.assertTrue(np.allclose(np.diag(dist), 0.0))
        self.assertTrue(np.allclose(dist, dist.T))
    
    def test_single_point(self):
        """Test with single point."""
        points = np.array([[1, 2, 3]])
        dist = euclidean_distance_matrix(points)
        
        self.assertEqual(dist.shape, (1, 1))
        self.assertEqual(dist[0, 0], 0.0)
    
    def test_identical_points(self):
        """Test with identical points."""
        points = np.array([[1, 1], [1, 1], [1, 1]])
        dist = euclidean_distance_matrix(points)
        
        # All distances should be zero
        self.assertTrue(np.allclose(dist, 0.0))


class TestBestSolution(unittest.TestCase):
    """
    Test best solution extraction functions.
    
    Tests cover:
    - get_best_fitness_index
    - get_best_solution
    - Edge cases
    """
    
    def test_get_best_fitness_index(self):
        """Test finding best fitness index."""
        fitness = np.array([5.2, 3.1, 4.7, 2.8])
        
        best_fit, best_idx = get_best_fitness_index(fitness)
        
        self.assertEqual(best_fit, 2.8)
        self.assertEqual(best_idx, 3)
    
    def test_get_best_solution(self):
        """Test extracting best solution."""
        positions = np.array([[1, 2], [3, 4], [5, 6]])
        fitness = np.array([10.0, 5.0, 15.0])
        
        best_sol, best_fit = get_best_solution(positions, fitness)
        
        self.assertTrue(np.array_equal(best_sol, [3, 4]))
        self.assertEqual(best_fit, 5.0)
    
    def test_first_occurrence_with_ties(self):
        """Test that first occurrence is returned when there are ties."""
        fitness = np.array([3.0, 2.0, 2.0, 4.0])
        
        best_fit, best_idx = get_best_fitness_index(fitness)
        
        self.assertEqual(best_fit, 2.0)
        self.assertEqual(best_idx, 1, "Should return first occurrence")
    
    def test_single_solution(self):
        """Test with single solution."""
        positions = np.array([[1, 2, 3]])
        fitness = np.array([10.0])
        
        best_sol, best_fit = get_best_solution(positions, fitness)
        
        self.assertTrue(np.array_equal(best_sol, [1, 2, 3]))
        self.assertEqual(best_fit, 10.0)
    
    def test_solution_is_copy(self):
        """Test that returned solution is a copy, not reference."""
        positions = np.array([[1, 2], [3, 4]])
        fitness = np.array([10.0, 5.0])
        
        best_sol, _ = get_best_solution(positions, fitness)
        
        # Modify returned solution
        best_sol[0] = 999
        
        # Original should not be modified
        self.assertEqual(positions[1, 0], 3)


class TestBrightness(unittest.TestCase):
    """
    Test brightness computation.
    
    Tests cover:
    - Correct negation
    - Best fitness = highest brightness
    - Array operations
    """
    
    def test_basic_brightness(self):
        """Test basic brightness computation."""
        fitness = np.array([10.0, 5.0, 0.0, 15.0])
        brightness = compute_brightness(fitness)
        
        expected = np.array([-10.0, -5.0, 0.0, -15.0])
        self.assertTrue(np.array_equal(brightness, expected))
    
    def test_best_fitness_is_brightest(self):
        """Test that best fitness corresponds to highest brightness."""
        fitness = np.array([10.0, 5.0, 0.0, 15.0])
        brightness = compute_brightness(fitness)
        
        best_fit_idx = np.argmin(fitness)
        brightest_idx = np.argmax(brightness)
        
        self.assertEqual(best_fit_idx, brightest_idx)
    
    def test_negative_fitness(self):
        """Test with negative fitness values."""
        fitness = np.array([-10.0, -5.0, 0.0, 5.0])
        brightness = compute_brightness(fitness)
        
        expected = np.array([10.0, 5.0, 0.0, -5.0])
        self.assertTrue(np.array_equal(brightness, expected))
    
    def test_single_value(self):
        """Test with single value."""
        fitness = np.array([5.0])
        brightness = compute_brightness(fitness)
        
        self.assertEqual(brightness[0], -5.0)


class TestPermutationRepair(unittest.TestCase):
    """
    Test permutation repair function.
    
    Tests cover:
    - Fixing duplicates
    - Adding missing values
    - Deterministic vs stochastic repair
    - Reproducibility with RNG
    """
    
    def test_repair_with_duplicates(self):
        """Test repairing permutation with duplicates."""
        invalid = np.array([0, 2, 2, 4, 1])  # Missing 3, duplicate 2
        
        valid = repair_permutation(invalid)
        
        # Check validity
        self.assertEqual(len(valid), 5)
        self.assertTrue(sorted(valid) == list(range(5)))
        self.assertEqual(len(set(valid)), 5)
    
    def test_deterministic_repair(self):
        """Test deterministic repair (no RNG)."""
        invalid = np.array([1, 3, 3, 0])  # Missing 2, duplicate 3
        
        valid1 = repair_permutation(invalid)
        valid2 = repair_permutation(invalid)
        
        # Should be identical (deterministic)
        self.assertTrue(np.array_equal(valid1, valid2))
    
    def test_stochastic_repair_with_rng(self):
        """Test stochastic repair with RNG."""
        invalid = np.array([0, 2, 2, 4, 1])
        
        rng = np.random.RandomState(42)
        valid = repair_permutation(invalid, rng)
        
        # Check validity
        self.assertTrue(sorted(valid) == list(range(5)))
    
    def test_reproducibility_with_rng(self):
        """Test that same RNG seed produces same repair."""
        invalid = np.array([0, 2, 2, 4, 1])
        
        rng1 = np.random.RandomState(123)
        valid1 = repair_permutation(invalid, rng1)
        
        rng2 = np.random.RandomState(123)
        valid2 = repair_permutation(invalid, rng2)
        
        # Should be identical
        self.assertTrue(np.array_equal(valid1, valid2))
    
    def test_already_valid_permutation(self):
        """Test with already valid permutation."""
        valid_perm = np.array([2, 0, 3, 1, 4])
        
        result = repair_permutation(valid_perm)
        
        # Should return valid permutation
        self.assertTrue(sorted(result) == list(range(5)))
    
    def test_all_same_value(self):
        """Test with all same values."""
        invalid = np.array([1, 1, 1, 1, 1])
        
        valid = repair_permutation(invalid)
        
        # Should produce valid permutation
        self.assertTrue(sorted(valid) == list(range(5)))
    
    def test_empty_permutation(self):
        """Test with empty permutation."""
        # Should either raise error or return empty
        with self.assertRaises((ValueError, IndexError)):
            repair_permutation(np.array([]))


class TestNumericalStability(unittest.TestCase):
    """
    Test numerical stability of utils functions.
    
    Tests cover:
    - Very large values
    - Very small values
    - Mixed scales
    """
    
    def test_large_distances(self):
        """Test distance computation with large coordinates."""
        points = np.array([[0, 0], [1e10, 1e10]])
        dist = euclidean_distance_matrix(points)
        
        # Should not overflow
        self.assertTrue(np.all(np.isfinite(dist)))
        self.assertGreater(dist[0, 1], 0)
    
    def test_small_distances(self):
        """Test distance computation with small coordinates."""
        points = np.array([[0, 0], [1e-10, 1e-10]])
        dist = euclidean_distance_matrix(points)
        
        # Should not underflow
        self.assertTrue(np.all(np.isfinite(dist)))
        self.assertGreater(dist[0, 1], 0)
    
    def test_mixed_scale_distances(self):
        """Test with mixed scale coordinates."""
        points = np.array([[0, 0], [1e10, 1e-10]])
        dist = euclidean_distance_matrix(points)
        
        self.assertTrue(np.all(np.isfinite(dist)))


if __name__ == '__main__':
    unittest.main()
