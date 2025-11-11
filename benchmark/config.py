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
    # Firefly Algorithm (FA) validations
    if 'n_fireflies' in params:
        if params['n_fireflies'] < 2:
            raise ValueError(f"{param_name}.n_fireflies must be >= 2, got {params['n_fireflies']}")
    
    if 'alpha' in params:
        if not 0 <= params['alpha'] <= 1:
            raise ValueError(f"{param_name}.alpha must be in [0, 1], got {params['alpha']}")
    
    if 'beta0' in params:
        if not 0 <= params['beta0'] <= 1:
            raise ValueError(f"{param_name}.beta0 must be in [0, 1], got {params['beta0']}")
    
    if 'gamma' in params:
        if not 0 <= params['gamma'] <= 1:
            raise ValueError(f"{param_name}.gamma must be in [0, 1], got {params['gamma']}")
    
    # Simulated Annealing (SA) validations
    if 'initial_temp' in params:  # Correct field name
        if params['initial_temp'] <= 0:
            raise ValueError(f"{param_name}.initial_temp must be > 0, got {params['initial_temp']}")
    
    if 'T0' in params:  # Also check old field name for backward compatibility
        if params['T0'] <= 0:
            raise ValueError(f"{param_name}.T0 must be > 0, got {params['T0']}")
    
    if 'cooling_rate' in params:
        if not 0 < params['cooling_rate'] < 1:
            raise ValueError(f"{param_name}.cooling_rate must be in (0, 1), got {params['cooling_rate']}")
    
    if 'min_temp' in params:
        if params['min_temp'] <= 0:
            raise ValueError(f"{param_name}.min_temp must be > 0, got {params['min_temp']}")
    
    # Hill Climbing (HC) validations
    if 'num_neighbors' in params:  # Correct field name
        if params['num_neighbors'] < 1:
            raise ValueError(f"{param_name}.num_neighbors must be >= 1, got {params['num_neighbors']}")
    
    if 'n_neighbors' in params:  # Old field name
        logger.warning(f"{param_name}: Use 'num_neighbors' instead of 'n_neighbors'")
        if params['n_neighbors'] < 1:
            raise ValueError(f"{param_name}.n_neighbors must be >= 1, got {params['n_neighbors']}")
    
    if 'restart_interval' in params:  # Correct field name
        if params['restart_interval'] < 1:
            raise ValueError(f"{param_name}.restart_interval must be >= 1, got {params['restart_interval']}")
    
    if 'restart_after' in params:  # Old field name
        logger.warning(f"{param_name}: Use 'restart_interval' instead of 'restart_after'")
        if params['restart_after'] < 1:
            raise ValueError(f"{param_name}.restart_after must be >= 1, got {params['restart_after']}")
    
    # Genetic Algorithm (GA) validations
    if 'pop_size' in params:
        if params['pop_size'] < 2:
            raise ValueError(f"{param_name}.pop_size must be >= 2, got {params['pop_size']}")
    
    if 'crossover_rate' in params:
        if not 0 <= params['crossover_rate'] <= 1:
            raise ValueError(f"{param_name}.crossover_rate must be in [0, 1], got {params['crossover_rate']}")
    
    if 'mutation_rate' in params:
        if not 0 <= params['mutation_rate'] <= 1:
            raise ValueError(f"{param_name}.mutation_rate must be in [0, 1], got {params['mutation_rate']}")
    
    # Knapsack-specific FA validations
    if 'alpha_flip' in params:
        if not 0 <= params['alpha_flip'] <= 1:
            raise ValueError(f"{param_name}.alpha_flip must be in [0, 1], got {params['alpha_flip']}")
    
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
    thresholds: Dict[str, float]  # NEW: Multi-level success thresholds (replaced 'threshold')
    seeds: range
    fa_params: Dict
    sa_params: Dict
    hc_params: Dict
    ga_params: Dict
    tuning_grids: Dict  # NEW: Parameter grids for tuning
    
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
            
            # Validate thresholds
            if not isinstance(self.thresholds, dict):
                raise ValueError(f"thresholds must be dict, got {type(self.thresholds)}")
            
            if len(self.thresholds) == 0:
                raise ValueError("thresholds cannot be empty")
            
            for level, value in self.thresholds.items():
                if not isinstance(level, str):
                    raise ValueError(f"threshold level must be string, got {type(level)}")
                if not isinstance(value, (int, float)) or value < 0:
                    raise ValueError(f"threshold value must be non-negative number, got {value}")
            
            if len(self.seeds) == 0:
                raise ValueError("seeds cannot be empty")
            
            # Validate algorithm parameters
            validate_ranges(self.fa_params, 'fa_params')
            validate_ranges(self.sa_params, 'sa_params')
            validate_ranges(self.hc_params, 'hc_params')
            validate_ranges(self.ga_params, 'ga_params')
            
            # Validate tuning_grids
            if hasattr(self, 'tuning_grids') and self.tuning_grids:
                for algo, grid in self.tuning_grids.items():
                    if not isinstance(grid, dict):
                        raise ValueError(f"tuning_grids[{algo}] must be dict, got {type(grid)}")
                    logger.debug(f"Tuning grid for {algo}: {grid}")
            
            logger.info(f"RastriginConfig validated: dim={self.dim}, budget={self.budget}, thresholds={self.thresholds}")
            
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
        dim=10,
        budget=10000,
        max_iter=250,
        thresholds={
            'gold': 1.0,      # Very close to optimum
            'silver': 10.0,    # Escaped bad regions, found good local minimum
            'bronze': 30.0    # Escaped worst regions
        },
        seeds=list(range(30)),
        fa_params={
            'n_fireflies': 40,
            'alpha': 0.2,
            'beta0': 1.0,
            'gamma': 1.0
        },
        sa_params={
            'initial_temp': 100.0,
            'cooling_rate': 0.95,
            'min_temp': 1.0
        },
        hc_params={
            'num_neighbors': 20,
            'restart_interval': 50
        },
        ga_params={
            'pop_size': 40,
            'crossover_rate': 0.8,
            'mutation_rate': 0.1,
            'tournament_size': 3,
            'elitism': True
        },
        tuning_grids={
            'FA': {
                'alpha': [0.15, 0.2, 0.25],
                'gamma': [0.8, 1.0, 1.2]
            },
            'SA': {
                'cooling_rate': [0.93, 0.95, 0.97]
            },
            'HC': {
                'num_neighbors': [15, 20, 25]
            },
            'GA': {
                'mutation_rate': [0.05, 0.1, 0.15],
                'crossover_rate': [0.75, 0.8, 0.85]
            }
        }
    ),
    
    'multimodal_escape': RastriginConfig(
        dim=30,
        budget=30000,
        max_iter=500,
        thresholds={
            'gold': 5.0,      # Very close to optimum
            'silver': 25.0,   # Escaped bad regions, found good local minimum
            'bronze': 50.0    # Escaped worst regions
        },
        seeds=list(range(30)),
        fa_params={
            'n_fireflies': 60,
            'alpha': 0.3,
            'beta0': 1.0,
            'gamma': 0.5
        },
        sa_params={
            'initial_temp': 200.0,
            'cooling_rate': 0.98,
            'min_temp': 1.0
        },
        hc_params={
            'num_neighbors': 30,
            'restart_interval': 30
        },
        ga_params={
            'pop_size': 60,
            'crossover_rate': 0.8,
            'mutation_rate': 0.1,
            'tournament_size': 5,
            'elitism': True
        },
        tuning_grids={
            'FA': {
                'alpha': [0.25, 0.3, 0.35],
                'gamma': [0.4, 0.5, 0.6]
            },
            'SA': {
                'cooling_rate': [0.96, 0.98, 0.99]
            },
            'HC': {
                'num_neighbors': [25, 30, 35]
            },
            'GA': {
                'mutation_rate': [0.08, 0.1, 0.12],
                'crossover_rate': [0.75, 0.8, 0.85]
            }
        }
    ),
    
    'scalability': RastriginConfig(
        dim=50,
        budget=50000,
        max_iter=625,
        thresholds={
            'gold': 10.0,     # Very close to optimum (higher threshold for harder problem)
            'silver': 50.0,   # Escaped bad regions
            'bronze': 80.0   # Escaped worst regions
        },
        seeds=list(range(30)),
        fa_params={
            'n_fireflies': 80,
            'alpha': 0.25,
            'beta0': 1.0,
            'gamma': 0.3
        },
        sa_params={
            'initial_temp': 300.0,
            'cooling_rate': 0.99,
            'min_temp': 1.0
        },
        hc_params={
            'num_neighbors': 50,
            'restart_interval': 50
        },
        ga_params={
            'pop_size': 80,
            'crossover_rate': 0.85,
            'mutation_rate': 0.05,
            'tournament_size': 7,
            'elitism': True
        },
        tuning_grids={
            'FA': {
                'alpha': [0.2, 0.25, 0.3],
                'gamma': [0.25, 0.3, 0.35]
            },
            'SA': {
                'cooling_rate': [0.98, 0.99, 0.995]
            },
            'HC': {
                'num_neighbors': [40, 50, 60]
            },
            'GA': {
                'mutation_rate': [0.03, 0.05, 0.07],
                'crossover_rate': [0.8, 0.85, 0.9]
            }
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
    sizes = [50, 100, 200]
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