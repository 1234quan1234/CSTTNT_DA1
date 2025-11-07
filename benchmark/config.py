"""
Benchmark configurations for Rastrigin and Knapsack problems.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np


@dataclass
class RastriginConfig:
    """Configuration for Rastrigin benchmark."""
    name: str
    dim: int
    budget: int
    max_iter: int
    threshold: float
    seeds: List[int]
    fa_params: Dict[str, Any]
    sa_params: Dict[str, Any]
    hc_params: Dict[str, Any]
    ga_params: Dict[str, Any]


@dataclass
class KnapsackConfig:
    """Configuration for Knapsack benchmark."""
    n_items: int
    instance_type: str
    seed: int
    budget: int
    fa_params: Dict[str, Any]
    sa_params: Dict[str, Any]
    hc_params: Dict[str, Any]
    ga_params: Dict[str, Any]
    has_dp_optimal: bool = True


# Rastrigin Benchmark Configurations
RASTRIGIN_CONFIGS = {
    'quick_convergence': RastriginConfig(
        name='quick_convergence',
        dim=2,
        budget=3000,
        max_iter=100,
        threshold=1.0,
        seeds=list(range(30)),
        fa_params={
            'n_fireflies': 30,
            'alpha': 0.2,
            'beta0': 1.0,
            'gamma': 1.0
        },
        sa_params={
            'initial_temp': 100.0,
            'cooling_rate': 0.95
        },
        hc_params={
            'num_neighbors': 20,
            'restart_interval': 50
        },
        ga_params={
            'pop_size': 30,
            'crossover_rate': 0.8,
            'mutation_rate': 0.1,
            'tournament_size': 3,
            'elitism': True
        }
    ),
    
    'multimodal_escape': RastriginConfig(
        name='multimodal_escape',
        dim=5,
        budget=10000,
        max_iter=200,
        threshold=5.0,
        seeds=list(range(30)),
        fa_params={
            'n_fireflies': 50,
            'alpha': 0.3,
            'beta0': 1.0,
            'gamma': 0.5
        },
        sa_params={
            'initial_temp': 200.0,
            'cooling_rate': 0.98
        },
        hc_params={
            'num_neighbors': 30,
            'restart_interval': 30
        },
        ga_params={
            'pop_size': 50,
            'crossover_rate': 0.8,
            'mutation_rate': 0.1,
            'tournament_size': 5,
            'elitism': True
        }
    ),
    
    'scalability': RastriginConfig(
        name='scalability',
        dim=10,
        budget=30000,
        max_iter=300,
        threshold=20.0,
        seeds=list(range(30)),
        fa_params={
            'n_fireflies': 100,
            'alpha': 0.25,
            'beta0': 1.0,
            'gamma': 0.3
        },
        sa_params={
            'initial_temp': 300.0,
            'cooling_rate': 0.99
        },
        hc_params={
            'num_neighbors': 50,
            'restart_interval': 50
        },
        ga_params={
            'pop_size': 100,
            'crossover_rate': 0.85,
            'mutation_rate': 0.05,
            'tournament_size': 7,
            'elitism': True
        }
    )
}


def get_knapsack_configs() -> List[KnapsackConfig]:
    """
    Generate Knapsack benchmark configurations.
    
    Returns
    -------
    configs : List[KnapsackConfig]
        All Knapsack configurations.
    """
    configs = []
    
    # Size variations with different instance types
    sizes = [50, 100, 200, 500]
    instance_types = ['uncorrelated', 'weakly', 'strongly', 'subset']
    seeds = [42, 123, 999]
    
    for size in sizes:
        for inst_type in instance_types:
            for seed in seeds:
                # DP optimal only for n <= 100
                has_dp = (size <= 100)
                
                # Budget scales with problem size
                if size <= 100:
                    budget = 5000
                    n_fireflies = 30
                    pop_size = 30
                elif size <= 200:
                    budget = 10000
                    n_fireflies = 40
                    pop_size = 40
                else:
                    budget = 20000
                    n_fireflies = 50
                    pop_size = 50
                
                config = KnapsackConfig(
                    n_items=size,
                    instance_type=inst_type,
                    seed=seed,
                    budget=budget,
                    fa_params={
                        'n_fireflies': n_fireflies,
                        'alpha_flip': 0.2,
                        'max_flips_per_move': 3,
                        'repair_method': 'greedy_remove'
                    },
                    sa_params={
                        'initial_temp': 100.0,
                        'cooling_rate': 0.95
                    },
                    hc_params={
                        'num_neighbors': 20,
                        'restart_interval': 50
                    },
                    ga_params={
                        'pop_size': pop_size,
                        'crossover_rate': 0.8,
                        'mutation_rate': 1.0 / size,
                        'tournament_size': 3,
                        'elitism': True
                    },
                    has_dp_optimal=has_dp
                )
                
                configs.append(config)
    
    return configs


# Export for convenience
KNAPSACK_CONFIGS = {
    f"n{c.n_items}_{c.instance_type}_seed{c.seed}": c
    for c in get_knapsack_configs()
}