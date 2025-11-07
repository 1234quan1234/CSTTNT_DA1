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
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Generate a Knapsack instance.
    
    Parameters
    ----------
    n_items : int
        Number of items.
    instance_type : str
        One of: 'uncorrelated', 'weakly', 'strongly', 'subset'.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    values : np.ndarray
        Item values, shape (n_items,).
    weights : np.ndarray
        Item weights, shape (n_items,).
    capacity : float
        Knapsack capacity (50% of total weight).
    """
    rng = np.random.RandomState(seed)
    
    if instance_type == 'uncorrelated':
        # Random values and weights
        values = rng.randint(1, 1001, n_items)
        weights = rng.randint(1, 1001, n_items)
    
    elif instance_type == 'weakly':
        # Values weakly correlated with weights
        weights = rng.randint(1, 1001, n_items)
        values = weights + rng.randint(-100, 101, n_items)
        values = np.maximum(values, 1)  # Ensure positive
    
    elif instance_type == 'strongly':
        # Values strongly correlated with weights
        weights = rng.randint(1, 1001, n_items)
        values = weights + 100
    
    elif instance_type == 'subset':
        # Subset-sum: values equal weights
        weights = rng.randint(1, 1001, n_items)
        values = weights.copy()
    
    else:
        raise ValueError(f"Unknown instance_type: {instance_type}")
    
    capacity = 0.5 * np.sum(weights)
    
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
