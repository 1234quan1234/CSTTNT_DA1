"""
Generate Knapsack problem instances for benchmarking.
Supports 4 instance types: uncorrelated, weakly, strongly, subset-sum.
"""

import numpy as np
from typing import Tuple


def generate_knapsack_instance(
    n_items: int,
    instance_type: str,
    seed: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Generate a Knapsack instance.
    
    Parameters
    ----------
    ...
    
    Returns
    -------
    values : np.ndarray
        Item values, shape (n_items,), dtype=int.
    weights : np.ndarray
        Item weights, shape (n_items,), dtype=int.
    capacity : int
        Knapsack capacity (50% of total weight), dtype=int.
        This ensures DP algorithms can use it as array index.
    """
    rng = np.random.RandomState(seed)
    
    if instance_type == 'uncorrelated':
        values = rng.randint(1, 1001, n_items)
        weights = rng.randint(1, 1001, n_items)
    
    elif instance_type == 'weakly':
        weights = rng.randint(1, 1001, n_items)
        values = weights + rng.randint(-100, 101, n_items)
        values = np.maximum(values, 1)
    
    elif instance_type == 'strongly':
        weights = rng.randint(1, 1001, n_items)
        values = weights + 100
    
    elif instance_type == 'subset':
        weights = rng.randint(1, 1001, n_items)
        values = weights.copy()
    
    else:
        raise ValueError(f"Unknown instance_type: {instance_type}")
    
    # Ensure integer types for DP compatibility
    values = values.astype(np.int64)
    weights = weights.astype(np.int64)
    capacity = int(0.5 * np.sum(weights))  # Convert to int for DP
    
    return values, weights, capacity


if __name__ == "__main__":
    # Test instance generation
    print("Testing Knapsack Instance Generation")
    print("=" * 60)
    
    for inst_type in ['uncorrelated', 'weakly', 'strongly', 'subset']:
        values, weights, capacity = generate_knapsack_instance(50, inst_type, 42)
        print(f"\n{inst_type.upper()}:")
        print(f"  Items: {len(values)}")
        print(f"  Capacity: {capacity:.1f}")
        print(f"  Total weight: {np.sum(weights):.1f}")
        print(f"  Value range: [{np.min(values)}, {np.max(values)}]")
        print(f"  Weight range: [{np.min(weights)}, {np.max(weights)}]")
        
        if inst_type == 'subset':
            print(f"  Values == Weights: {np.allclose(values, weights)}")
