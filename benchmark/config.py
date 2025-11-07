"""
Configuration management for benchmark experiments.
Ensures fairness and reproducibility across all algorithms.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class RastriginConfig:
    """Configuration for Rastrigin benchmark."""
    dim: int
    budget: int  # Number of evaluations
    max_iter: int  # Derived from budget and population size
    seeds: List[int]
    threshold: float  # Success threshold
    
    # Algorithm-specific parameters
    fa_params: Dict = None
    sa_params: Dict = None
    hc_params: Dict = None
    ga_params: Dict = None
    
    def __post_init__(self):
        if self.fa_params is None:
            self.fa_params = {
                'n_fireflies': 40,
                'alpha': 0.3,
                'beta0': 1.0,
                'gamma': 1.0,
                'alpha_decay': 0.97
            }
        if self.sa_params is None:
            self.sa_params = {
                'initial_temp': 100.0,
                'cooling_rate': 0.95,
                'step_size': 0.5,
                'min_temp': 0.01
            }
        if self.hc_params is None:
            self.hc_params = {
                'step_size': 0.5,
                'num_neighbors': 20,
                'restart_interval': 50
            }
        if self.ga_params is None:
            self.ga_params = {
                'pop_size': 40,
                'crossover_rate': 0.8,
                'mutation_rate': 0.1,
                'tournament_size': 3,
                'elitism': 2
            }


@dataclass
class KnapsackConfig:
    """Configuration for Knapsack benchmark."""
    n_items: int
    instance_type: str  # 'uncorrelated', 'weakly', 'strongly', 'subset'
    seed: int
    budget: int
    max_iter: int
    seeds: List[int]
    capacity: int = 0
    has_dp_optimal: bool = True
    
    # Algorithm-specific parameters
    fa_params: Dict = None
    sa_params: Dict = None
    hc_params: Dict = None
    ga_params: Dict = None
    
    def __post_init__(self):
        if self.fa_params is None:
            self.fa_params = {
                'n_fireflies': 60,
                'alpha_flip': 0.2,
                'beta0': 1.0,
                'gamma': 1.0,
                'max_flips_per_move': 3,
                'repair_method': 'greedy_remove'
            }
        if self.sa_params is None:
            self.sa_params = {
                'initial_temp': 1000.0,
                'cooling_rate': 0.95,
                'min_temp': 0.01
            }
        if self.hc_params is None:
            self.hc_params = {
                'num_neighbors': 20,
                'restart_interval': 100
            }
        if self.ga_params is None:
            self.ga_params = {
                'pop_size': 60,
                'crossover_rate': 0.8,
                'mutation_rate': None,  # Will be 1/n
                'tournament_size': 3,
                'elitism': 2
            }


# Predefined configurations for Rastrigin
RASTRIGIN_CONFIGS = {
    'quick_convergence': RastriginConfig(
        dim=10,
        budget=5000,
        max_iter=125,  # 5000 / 40
        seeds=list(range(30)),
        threshold=10.0
    ),
    'multimodal_escape': RastriginConfig(
        dim=30,
        budget=20000,
        max_iter=500,  # 20000 / 40
        seeds=list(range(30)),
        threshold=50.0
    ),
    'scalability': RastriginConfig(
        dim=50,
        budget=40000,
        max_iter=1000,  # 40000 / 40
        seeds=list(range(30)),
        threshold=100.0
    )
}


# Predefined configurations for Knapsack
KNAPSACK_CONFIGS = {
    'small': KnapsackConfig(
        n_items=50,
        instance_type='uncorrelated',
        seed=42,
        budget=5000,
        max_iter=83,  # 5000 / 60
        seeds=[42, 43],
        capacity=1000,
        has_dp_optimal=True
    ),
    'medium': KnapsackConfig(
        n_items=100,
        instance_type='uncorrelated',
        seed=42,
        budget=10000,
        max_iter=166,  # 10000 / 60
        seeds=[42, 43],
        capacity=2000,
        has_dp_optimal=True
    ),
    'large': KnapsackConfig(
        n_items=200,
        instance_type='uncorrelated',
        seed=42,
        budget=20000,
        max_iter=333,  # 20000 / 60
        seeds=[42, 43],
        capacity=4000,
        has_dp_optimal=False
    )
}


def get_knapsack_configs() -> List[KnapsackConfig]:
    """Generate all Knapsack benchmark configurations."""
    configs = []
    
    # n=50: All 4 types, 2 seeds
    for inst_type in ['uncorrelated', 'weakly', 'strongly', 'subset']:
        for seed in [42, 43]:
            configs.append(KnapsackConfig(
                n_items=50,
                instance_type=inst_type,
                seed=seed,
                budget=10000,
                max_iter=166,
                seeds=[seed],
                has_dp_optimal=True
            ))
    
    # n=100: All 4 types, 2 seeds
    for inst_type in ['uncorrelated', 'weakly', 'strongly', 'subset']:
        for seed in [42, 43]:
            configs.append(KnapsackConfig(
                n_items=100,
                instance_type=inst_type,
                seed=seed,
                budget=15000,
                max_iter=250,
                seeds=[seed],
                has_dp_optimal=True
            ))
    
    # n=200: Uncorr, Weak only, 2 seeds
    for inst_type in ['uncorrelated', 'weakly']:
        for seed in [42, 43]:
            configs.append(KnapsackConfig(
                n_items=200,
                instance_type=inst_type,
                seed=seed,
                budget=30000,
                max_iter=500,
                seeds=[seed],
                has_dp_optimal=False
            ))
    
    # n=500: Uncorr only, 2 seeds
    for seed in [42, 43]:
        configs.append(KnapsackConfig(
            n_items=500,
            instance_type='uncorrelated',
            seed=seed,
            budget=50000,
            max_iter=833,
            seeds=[seed],
            has_dp_optimal=False
        ))
    
    return configs
