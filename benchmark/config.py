"""
Benchmark configurations for Rastrigin and Knapsack problems.
"""

from dataclasses import dataclass
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def validate_ranges(params: Dict, param_name: str) -> bool:
    """
    Validate parameter ranges.
    
    Parameters
    ----------
    params : dict
        Parameter dictionary
    param_name : str
        Parameter name for logging
        
    Returns
    -------
    bool
        True if valid
        
    Raises
    ------
    ValueError
        If validation fails
    """
    # Check positive values
    positive_params = ['n_fireflies', 'pop_size', 'n_neighbors', 'restart_after']
    for key in positive_params:
        if key in params and params[key] <= 0:
            raise ValueError(f"{param_name}.{key} must be > 0, got {params[key]}")
    
    # Check probability ranges [0, 1]
    prob_params = ['alpha', 'beta0', 'gamma', 'alpha_flip', 'crossover_rate', 'mutation_rate']
    for key in prob_params:
        if key in params:
            if not (0 <= params[key] <= 1):
                raise ValueError(f"{param_name}.{key} must be in [0, 1], got {params[key]}")
    
    # Check temperature
    if 'T0' in params and params['T0'] <= 0:
        raise ValueError(f"{param_name}.T0 must be > 0, got {params['T0']}")
    
    if 'cooling_rate' in params:
        if not (0 < params['cooling_rate'] < 1):
            raise ValueError(f"{param_name}.cooling_rate must be in (0, 1), got {params['cooling_rate']}")
    
    logger.debug(f"Parameters validated: {param_name}")
    return True


def validate_paths(output_dir: str = 'benchmark/results') -> bool:
    """
    Validate that output directories are writable.
    
    Parameters
    ----------
    output_dir : str
        Output directory path
        
    Returns
    -------
    bool
        True if writable
        
    Raises
    ------
    PermissionError
        If directory is not writable
    """
    from pathlib import Path
    import os
    
    path = Path(output_dir)
    
    # Try to create directory
    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        logger.error(f"Cannot create directory {output_dir}: {e}")
        raise PermissionError(f"Directory not writable: {output_dir}")
    
    # Check if writable
    if not os.access(path, os.W_OK):
        logger.error(f"Directory not writable: {output_dir}")
        logger.info(f"  Suggestion: Run 'chmod u+w {output_dir}'")
        raise PermissionError(f"Directory not writable: {output_dir}")
    
    logger.debug(f"Output directory validated: {output_dir}")
    return True


@dataclass
class RastriginConfig:
    """Configuration for Rastrigin benchmark."""
    dim: int
    budget: int
    max_iter: int
    threshold: float
    seeds: range
    fa_params: Dict
    sa_params: Dict
    hc_params: Dict
    ga_params: Dict
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        try:
            # Validate dimensions
            if self.dim <= 0:
                raise ValueError(f"dim must be > 0, got {self.dim}")
            
            if self.budget <= 0:
                raise ValueError(f"budget must be > 0, got {self.budget}")
            
            if self.max_iter < 10:
                raise ValueError(f"max_iter must be >= 10, got {self.max_iter}")
            
            if self.threshold < 0:
                raise ValueError(f"threshold must be >= 0, got {self.threshold}")
            
            if len(self.seeds) == 0:
                raise ValueError("seeds cannot be empty")
            
            # Validate algorithm parameters
            validate_ranges(self.fa_params, 'fa_params')
            validate_ranges(self.sa_params, 'sa_params')
            validate_ranges(self.hc_params, 'hc_params')
            validate_ranges(self.ga_params, 'ga_params')
            
            logger.info(f"RastriginConfig validated: dim={self.dim}, budget={self.budget}")
            
        except ValueError as e:
            logger.error(f"Invalid RastriginConfig: {e}")
            raise


@dataclass
class KnapsackConfig:
    """Configuration for Knapsack benchmark."""
    n_items: int
    instance_type: str
    seed: int
    budget: int
    has_dp_optimal: bool
    fa_params: Dict
    sa_params: Dict
    hc_params: Dict
    ga_params: Dict
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        try:
            # Validate parameters
            if self.n_items <= 0:
                raise ValueError(f"n_items must be > 0, got {self.n_items}")
            
            valid_types = ['uncorrelated', 'weakly', 'strongly', 'subset']
            if self.instance_type not in valid_types:
                raise ValueError(f"instance_type must be one of {valid_types}, got {self.instance_type}")
            
            if self.budget <= 0:
                raise ValueError(f"budget must be > 0, got {self.budget}")
            
            # Validate algorithm parameters
            validate_ranges(self.fa_params, 'fa_params')
            validate_ranges(self.sa_params, 'sa_params')
            validate_ranges(self.hc_params, 'hc_params')
            validate_ranges(self.ga_params, 'ga_params')
            
            logger.info(f"KnapsackConfig validated: n={self.n_items}, type={self.instance_type}")
            
        except ValueError as e:
            logger.error(f"Invalid KnapsackConfig: {e}")
            raise


# Rastrigin Benchmark Configurations
RASTRIGIN_CONFIGS = {
    'quick_convergence': RastriginConfig(
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