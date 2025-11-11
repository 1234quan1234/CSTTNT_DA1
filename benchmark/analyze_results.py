"""
Analyze and compare benchmark results.
Supports new file naming structure with timestamps: 
  - rastrigin_{config}_{algo}_{timestamp}.json
  - knapsack_n{size}_{type}_seed{seed}_{algo}_{timestamp}.json
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import re
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

RASTRIGIN_TOLS = [1e-1, 1e-3, 1e-5]
KNAPSACK_GAP_TOLS = [1.0, 5.0, 10.0]  # %


def load_json_safe(file_path: Path) -> Optional[Dict[str, Any]]:
    """Safely load JSON file with error handling."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path.name}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading {file_path.name}: {type(e).__name__}: {e}")
        return None


def extract_algo_from_filename(filename: str) -> Optional[str]:
    """Extract algorithm name (FA, GA, SA, HC) from filename with timestamp."""
    # Pattern: ..._{ALGO}_{timestamp}.json where timestamp is 15 digits like 20251110T200402
    match = re.search(r'_(FA|GA|SA|HC)_\d{8}T\d{6}\.json$', filename)
    if match:
        return match.group(1)
    # Fallback for filenames without timestamp
    if '_FA_' in filename or filename.endswith('_FA.json'):
        return 'FA'
    elif '_GA_' in filename or filename.endswith('_GA.json'):
        return 'GA'
    elif '_SA_' in filename or filename.endswith('_SA.json'):
        return 'SA'
    elif '_HC_' in filename or filename.endswith('_HC.json'):
        return 'HC'
    return None


# ============================================================================
# RASTRIGIN DATA LOADING
# ============================================================================

def get_rastrigin_raw_data(
    results_dir: Union[str, Path],
    config_name: str
) -> Dict[str, Dict[str, Any]]:
    """
    Extract raw per-run data for Rastrigin from timestamped files.
    Pattern: rastrigin_{config_name}_{algo}_{timestamp}.json
    
    Returns: {algo: {'runs', 'best_fitness', 'error_to_optimum', 'histories', 
                     'elapsed_times', 'n_evals'}}
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return {}
    
    # Find files: rastrigin_quick_convergence_FA_20251110T200402.json
    pattern = f"rastrigin_{config_name}_*_*.json"
    json_files = list(results_path.glob(pattern))
    
    if not json_files:
        logger.warning(f"No result files found for config '{config_name}'")
        logger.info(f"  Searched pattern: {pattern}")
        return {}
    
    logger.info(f"Found {len(json_files)} result file(s) for {config_name}")
    
    raw_data: Dict[str, Dict[str, Any]] = {}
    
    for json_file in json_files:
        algo = extract_algo_from_filename(json_file.name)
        if not algo:
            logger.warning(f"Cannot infer algorithm from {json_file.name}")
            continue
        
        data = load_json_safe(json_file)
        if not data:
            continue
        
        # Handle both metadata+results and flat structures
        if 'metadata' in data and 'results' in data:
            runs = data['results']
        elif isinstance(data, list):
            runs = data
        else:
            logger.warning(f"Unknown structure in {json_file.name}")
            continue
        
        if not runs or not isinstance(runs, list):
            logger.warning(f"Empty or invalid runs in {json_file.name}")
            continue
        
        # Validate and fix missing fields
        for run in runs:
            if 'best_fitness' not in run:
                run['best_fitness'] = float('inf')
            if 'history' not in run:
                run['history'] = []
            if 'elapsed_time' not in run:
                run['elapsed_time'] = 0.0
        
        # Aggregate by algorithm (handle multiple timestamps)
        if algo not in raw_data:
            raw_data[algo] = {
                'runs': [],
                'best_fitness': [],
                'error_to_optimum': [],
                'histories': [],
                'elapsed_times': [],
                'n_evals': []
            }
        
        raw_data[algo]['runs'].extend(runs)
    
    # Convert to numpy arrays
    for algo, data in raw_data.items():
        runs = data['runs']
        try:
            data['best_fitness'] = np.array([r['best_fitness'] for r in runs])
            data['error_to_optimum'] = np.abs(data['best_fitness'])
            data['histories'] = [r['history'] for r in runs]
            data['elapsed_times'] = np.array([r['elapsed_time'] for r in runs])
            data['n_evals'] = np.array([len(r['history']) for r in runs])
            logger.info(f"  Loaded {len(runs)} runs for {algo}")
        except Exception as e:
            logger.error(f"Error converting data for {algo}: {e}")
    
    return raw_data


def discover_rastrigin_configs(results_dir: Union[str, Path]) -> List[str]:
    """Discover all Rastrigin configurations from result files."""
    results_path = Path(results_dir)
    configs = set()
    
    for json_file in results_path.glob('rastrigin_*_*_*.json'):
        # Pattern: rastrigin_{config}_{algo}_{timestamp}.json
        match = re.match(r'rastrigin_([a-z_]+)_[A-Z]{2}_\d{8}T\d{6}\.json', json_file.name)
        if match:
            configs.add(match.group(1))
    
    return sorted(configs)


# ============================================================================
# KNAPSACK DATA LOADING
# ============================================================================

def get_knapsack_raw_data(
    results_dir: Union[str, Path],
    n_items: int,
    instance_type: str,
    instance_seed: int
) -> Dict[str, Any]:
    """
    Extract raw per-run data for Knapsack from timestamped files.
    Pattern: knapsack_n{size}_{type}_seed{seed}_{algo}_{timestamp}.json
    
    Returns: {
        'dp_optimal': float,
        'config': dict,
        algo: {'runs', 'best_values', 'optimality_gaps', 'histories',
               'feasibility', 'capacity_utilization', 'elapsed_times', 'n_evals'}
    }
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return {}
    
    # Find files: knapsack_n100_uncorrelated_seed42_FA_20251110T202419.json
    pattern = f"knapsack_n{n_items}_{instance_type}_seed{instance_seed}_*_*.json"
    json_files = list(results_path.glob(pattern))
    
    if not json_files:
        logger.warning(f"No result files for n={n_items}, type={instance_type}, seed={instance_seed}")
        return {}
    
    logger.info(f"Found {len(json_files)} result file(s)")
    
    raw_data = {'dp_optimal': None, 'config': {}}
    
    for json_file in json_files:
        algo = extract_algo_from_filename(json_file.name)
        if not algo:
            logger.warning(f"Cannot infer algorithm from {json_file.name}")
            continue
        
        data = load_json_safe(json_file)
        if not data:
            continue
        
        # Handle both metadata+results and flat structures
        if 'metadata' in data and 'results' in data:
            config = data['metadata']
            runs = data['results']
        elif 'config' in data and 'results' in data:
            config = data['config']
            runs = data['results']
        else:
            logger.warning(f"Unknown structure in {json_file.name}")
            continue
        
        if not runs or not isinstance(runs, list):
            logger.warning(f"Empty or invalid runs in {json_file.name}")
            continue
        
        # Store config and DP optimal once
        if not raw_data['config']:
            raw_data['config'] = {
                'n_items': config.get('n_items', n_items),
                'instance_type': config.get('instance_type', instance_type),
                'instance_seed': config.get('instance_seed', instance_seed)
            }
        if raw_data['dp_optimal'] is None:
            raw_data['dp_optimal'] = config.get('dp_optimal')
        
        # Validate and fix missing fields
        for run in runs:
            if 'best_value' not in run:
                run['best_value'] = 0.0
            if 'is_feasible' not in run:
                run['is_feasible'] = False
            if 'history' not in run:
                run['history'] = []
            if 'elapsed_time' not in run:
                run['elapsed_time'] = 0.0
            if 'capacity_utilization' not in run:
                run['capacity_utilization'] = 0.0
        
        # Initialize algo entry
        if algo not in raw_data:
            raw_data[algo] = {
                'runs': [],
                'best_values': [],
                'optimality_gaps': [],
                'histories': [],
                'feasibility': [],
                'capacity_utilization': [],
                'elapsed_times': [],
                'n_evals': []
            }
        
        # Collect data
        for run in runs:
            raw_data[algo]['runs'].append(run)
            raw_data[algo]['best_values'].append(run['best_value'])
            raw_data[algo]['histories'].append(run['history'])
            raw_data[algo]['feasibility'].append(1.0 if run['is_feasible'] else 0.0)
            raw_data[algo]['capacity_utilization'].append(run['capacity_utilization'])
            raw_data[algo]['elapsed_times'].append(run['elapsed_time'])
            raw_data[algo]['n_evals'].append(len(run['history']))
            
            # Calculate optimality gap if feasible and DP optimal known
            if raw_data['dp_optimal'] and run['is_feasible']:
                gap = (raw_data['dp_optimal'] - run['best_value']) / raw_data['dp_optimal'] * 100
                raw_data[algo]['optimality_gaps'].append(gap)
    
    # Convert to numpy arrays
    for algo in list(raw_data.keys()):
        if algo not in ['dp_optimal', 'config']:
            try:
                raw_data[algo]['best_values'] = np.array(raw_data[algo]['best_values'])
                raw_data[algo]['feasibility'] = np.array(raw_data[algo]['feasibility'])
                raw_data[algo]['capacity_utilization'] = np.array(raw_data[algo]['capacity_utilization'])
                raw_data[algo]['elapsed_times'] = np.array(raw_data[algo]['elapsed_times'])
                raw_data[algo]['n_evals'] = np.array(raw_data[algo]['n_evals'])
                if raw_data[algo]['optimality_gaps']:
                    raw_data[algo]['optimality_gaps'] = np.array(raw_data[algo]['optimality_gaps'])
                else:
                    raw_data[algo]['optimality_gaps'] = np.array([])
                
                logger.info(f"  Loaded {len(raw_data[algo]['runs'])} runs for {algo}")
            except Exception as e:
                logger.error(f"Error converting data for {algo}: {e}")
    
    return raw_data


def discover_knapsack_configs(results_dir: Union[str, Path]) -> List[Tuple[int, str, int]]:
    """Discover all Knapsack configurations from result files."""
    results_path = Path(results_dir)
    configs = set()
    
    for json_file in results_path.glob('knapsack_n*_*_seed*_*_*.json'):
        # Pattern: knapsack_n50_uncorrelated_seed42_FA_20251110T202419.json
        match = re.match(r'knapsack_n(\d+)_([a-z_]+)_seed(\d+)_[A-Z]{2}_\d{8}T\d{6}\.json', json_file.name)
        if match:
            n_items = int(match.group(1))
            instance_type = match.group(2)
            seed = int(match.group(3))
            configs.add((n_items, instance_type, seed))
    
    return sorted(configs)


# ============================================================================
# HELPER FUNCTIONS FOR ANYTIME PERFORMANCE
# ============================================================================

def compute_hitting_time(history: List[float], tolerance: float) -> int:
    """
    Find first evaluation index where |f(x)| <= tolerance.
    Returns len(history) if never hit tolerance.
    """
    if not history:
        return 0
    for i, val in enumerate(history, start=1):
        if val is not None and abs(val) <= tolerance:
            return i
    return len(history)


def compute_auc(history: List[float]) -> float:
    """
    Anytime performance metric: normalized AUC under log(1 + |f(x)|) curve.
    Used in Dolan-Moré performance profile methodology.
    Lower is better (optimizer converges faster/better).
    """
    if not history:
        return float('inf')
    
    h = np.array([abs(v) if v is not None else np.nan for v in history], dtype=float)
    # Replace NaN with max finite value or 1.0
    h = np.nan_to_num(h, nan=(np.nanmax(h) if np.isfinite(np.nanmax(h)) else 1.0))
    
    x = np.arange(1, len(h) + 1, dtype=float)
    x = x / x[-1]  # Normalize to [0, 1]
    y = np.log1p(h)
    
    return float(np.trapz(y, x))


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def friedman_test(results: Dict[str, List[float]]) -> Tuple[float, float]:
    """Perform Friedman test."""
    data = [results[algo] for algo in sorted(results.keys())]
    statistic, p_value = stats.friedmanchisquare(*data)
    return float(statistic), float(p_value)


def wilcoxon_test(data1: List[float], data2: List[float]) -> Tuple[float, float]:
    """Perform Wilcoxon signed-rank test."""
    statistic, p_value = stats.wilcoxon(data1, data2)
    return float(statistic), float(p_value)


def compute_ranks(results: Dict[str, List[float]]) -> Dict[str, float]:
    """Compute average ranks across runs."""
    algorithms = sorted(results.keys())
    n_runs = len(results[algorithms[0]])
    
    ranks: Dict[str, float] = {algo: 0.0 for algo in algorithms}
    
    for run_idx in range(n_runs):
        run_values = [(algo, results[algo][run_idx]) for algo in algorithms]
        run_values.sort(key=lambda x: x[1])
        
        for rank, (algo, _) in enumerate(run_values, 1):
            ranks[algo] += rank
    
    for algo in algorithms:
        ranks[algo] /= n_runs
    
    return ranks


def perform_rastrigin_statistical_tests(
    raw_data_by_config: Dict[str, Dict[str, Dict[str, Any]]]
) -> None:
    """Perform statistical tests on Rastrigin results."""
    
    for config_name in sorted(raw_data_by_config.keys()):
        print(f"\n{'-' * 80}")
        print(f"Configuration: {config_name}")
        print(f"{'-' * 80}")
        
        raw_data = raw_data_by_config[config_name]
        
        # Convert to Dict[algo, List[float]] for statistical tests
        results = {algo: data['error_to_optimum'].tolist() 
                  for algo, data in raw_data.items()}
        
        if len(results) < 2:
            print("Not enough algorithms for comparison")
            continue
        
        # Friedman test
        try:
            friedman_stat, friedman_p = friedman_test(results)
            print(f"\nFriedman Test:")
            print(f"  Statistic: {friedman_stat:.4f}")
            print(f"  P-value: {friedman_p:.4e}")
            print(f"  Significant: {'Yes' if friedman_p < 0.05 else 'No'}")
        except Exception as e:
            logger.warning(f"Friedman test failed: {e}")
        
        # Average ranks
        try:
            ranks = compute_ranks(results)
            print(f"\nAverage Ranks (lower is better):")
            for algo, rank in sorted(ranks.items(), key=lambda x: x[1]):
                print(f"  {algo}: {rank:.2f}")
        except Exception as e:
            logger.warning(f"Rank computation failed: {e}")
        
        # Pairwise Wilcoxon tests
        try:
            print(f"\nPairwise Wilcoxon Tests (p-values):")
            algorithms = sorted(results.keys())
            
            print(f"{'':>8}", end='')
            for algo in algorithms:
                print(f"{algo:>8}", end='')
            print()
            
            for algo1 in algorithms:
                print(f"{algo1:>8}", end='')
                for algo2 in algorithms:
                    if algo1 == algo2:
                        print(f"{'—':>8}", end='')
                    else:
                        _, p_val = wilcoxon_test(results[algo1], results[algo2])
                        sig = "*" if p_val < 0.05 else ""
                        print(f"{p_val:>7.4f}{sig}", end='')
                print()
        except Exception as e:
            logger.warning(f"Wilcoxon tests failed: {e}")


# ============================================================================
# SUMMARY GENERATION
# ============================================================================

def generate_rastrigin_summary(
    results_dir: Union[str, Path],
    output_file: Union[str, Path]
) -> Optional[pd.DataFrame]:
    """Generate Rastrigin summary CSV with success rates, hitting times, AUC."""
    
    output_path = Path(output_file)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Cannot create directory: {e}")
        return None
    
    configs = discover_rastrigin_configs(results_dir)
    if not configs:
        logger.error("No Rastrigin configurations found")
        return None
    
    summary_data = []
    
    for config_name in configs:
        raw_data = get_rastrigin_raw_data(results_dir, config_name)
        
        for algo, data in raw_data.items():
            errors = data['error_to_optimum']
            times = data['elapsed_times']
            histories = data['histories']
            
            if len(errors) == 0:
                continue
            
            row = {
                'Configuration': config_name,
                'Algorithm': algo,
                'Mean': np.mean(errors),
                'Std': np.std(errors),
                'Median': np.median(errors),
                'Best': np.min(errors),
                'Worst': np.max(errors),
                'Q1': np.percentile(errors, 25),
                'Q3': np.percentile(errors, 75),
                'Mean_Time': np.mean(times)
            }
            
            # Success rates at multiple tolerances
            for tol in RASTRIGIN_TOLS:
                sr = np.mean(errors <= tol) * 100.0
                row[f'SR_<={tol}'] = sr
            
            # Hitting times (median across runs)
            for tol in RASTRIGIN_TOLS:
                hts = [compute_hitting_time(h, tol) for h in histories if h]
                row[f'HT_med_<={tol}'] = float(np.median(hts)) if hts else np.nan
            
            # AUC (anytime performance)
            aucs = [compute_auc(h) for h in histories if h]
            row['AUC_median'] = float(np.median(aucs)) if aucs else np.nan
            row['AUC_mean'] = float(np.mean(aucs)) if aucs else np.nan
            
            summary_data.append(row)
    
    if not summary_data:
        logger.error("No data found for summary")
        return None
    
    try:
        df = pd.DataFrame(summary_data)
        df = df.sort_values(['Configuration', 'Algorithm'])
        df.to_csv(output_file, index=False)
        
        logger.info(f"Rastrigin summary saved to: {output_file}")
        print("\n" + "=" * 100)
        print(df.to_string(index=False))
        print("=" * 100)
        
        return df
    except Exception as e:
        logger.error(f"Error saving summary: {e}")
        return None


def generate_rastrigin_global_ranks(
    summary_df: pd.DataFrame,
    metric: str = 'AUC_median',
    output_file: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Compute average rank of each algorithm across all configurations.
    Metric: column to optimize (lower is better).
    """
    grouped = summary_df.groupby('Configuration')
    algo_metrics: Dict[str, List[float]] = {}
    
    for config, group in grouped:
        valid = group.dropna(subset=[metric])
        if len(valid) < 2:
            continue
        sorted_algos = valid.sort_values(metric)['Algorithm'].tolist()
        ranks = {algo: rank + 1 for rank, algo in enumerate(sorted_algos)}
        for algo, rank in ranks.items():
            algo_metrics.setdefault(algo, []).append(rank)
    
    avg_ranks = {algo: float(np.mean(rs)) for algo, rs in algo_metrics.items()}
    
    rank_df = pd.DataFrame(
        [{'Algorithm': a, 'Avg_Rank': r, 'N_Configs': len(algo_metrics[a])}
         for a, r in sorted(avg_ranks.items(), key=lambda x: x[1])]
    )
    
    if output_file:
        rank_df.to_csv(output_file, index=False)
        logger.info(f"Global ranks saved to: {output_file}")
    
    print(f"\nGlobal Rastrigin Ranks (by {metric}, lower is better):")
    print(rank_df.to_string(index=False))
    
    return rank_df


# ============================================================================
# KNAPSACK DATA LOADING
# ============================================================================

def get_knapsack_raw_data(
    results_dir: Union[str, Path],
    n_items: int,
    instance_type: str,
    instance_seed: int
) -> Dict[str, Any]:
    """
    Extract raw per-run data for Knapsack from timestamped files.
    Pattern: knapsack_n{size}_{type}_seed{seed}_{algo}_{timestamp}.json
    
    Returns: {
        'dp_optimal': float,
        'config': dict,
        algo: {'runs', 'best_values', 'optimality_gaps', 'histories',
               'feasibility', 'capacity_utilization', 'elapsed_times', 'n_evals'}
    }
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return {}
    
    # Find files: knapsack_n100_uncorrelated_seed42_FA_20251110T202419.json
    pattern = f"knapsack_n{n_items}_{instance_type}_seed{instance_seed}_*_*.json"
    json_files = list(results_path.glob(pattern))
    
    if not json_files:
        logger.warning(f"No result files for n={n_items}, type={instance_type}, seed={instance_seed}")
        return {}
    
    logger.info(f"Found {len(json_files)} result file(s)")
    
    raw_data = {'dp_optimal': None, 'config': {}}
    
    for json_file in json_files:
        algo = extract_algo_from_filename(json_file.name)
        if not algo:
            logger.warning(f"Cannot infer algorithm from {json_file.name}")
            continue
        
        data = load_json_safe(json_file)
        if not data:
            continue
        
        # Handle both metadata+results and flat structures
        if 'metadata' in data and 'results' in data:
            config = data['metadata']
            runs = data['results']
        elif 'config' in data and 'results' in data:
            config = data['config']
            runs = data['results']
        else:
            logger.warning(f"Unknown structure in {json_file.name}")
            continue
        
        if not runs or not isinstance(runs, list):
            logger.warning(f"Empty or invalid runs in {json_file.name}")
            continue
        
        # Store config and DP optimal once
        if not raw_data['config']:
            raw_data['config'] = {
                'n_items': config.get('n_items', n_items),
                'instance_type': config.get('instance_type', instance_type),
                'instance_seed': config.get('instance_seed', instance_seed)
            }
        if raw_data['dp_optimal'] is None:
            raw_data['dp_optimal'] = config.get('dp_optimal')
        
        # Validate and fix missing fields
        for run in runs:
            if 'best_value' not in run:
                run['best_value'] = 0.0
            if 'is_feasible' not in run:
                run['is_feasible'] = False
            if 'history' not in run:
                run['history'] = []
            if 'elapsed_time' not in run:
                run['elapsed_time'] = 0.0
            if 'capacity_utilization' not in run:
                run['capacity_utilization'] = 0.0
        
        # Initialize algo entry
        if algo not in raw_data:
            raw_data[algo] = {
                'runs': [],
                'best_values': [],
                'optimality_gaps': [],
                'histories': [],
                'feasibility': [],
                'capacity_utilization': [],
                'elapsed_times': [],
                'n_evals': []
            }
        
        # Collect data
        for run in runs:
            raw_data[algo]['runs'].append(run)
            raw_data[algo]['best_values'].append(run['best_value'])
            raw_data[algo]['histories'].append(run['history'])
            raw_data[algo]['feasibility'].append(1.0 if run['is_feasible'] else 0.0)
            raw_data[algo]['capacity_utilization'].append(run['capacity_utilization'])
            raw_data[algo]['elapsed_times'].append(run['elapsed_time'])
            raw_data[algo]['n_evals'].append(len(run['history']))
            
            # Calculate optimality gap if feasible and DP optimal known
            if raw_data['dp_optimal'] and run['is_feasible']:
                gap = (raw_data['dp_optimal'] - run['best_value']) / raw_data['dp_optimal'] * 100
                raw_data[algo]['optimality_gaps'].append(gap)
    
    # Convert to numpy arrays
    for algo in list(raw_data.keys()):
        if algo not in ['dp_optimal', 'config']:
            try:
                raw_data[algo]['best_values'] = np.array(raw_data[algo]['best_values'])
                raw_data[algo]['feasibility'] = np.array(raw_data[algo]['feasibility'])
                raw_data[algo]['capacity_utilization'] = np.array(raw_data[algo]['capacity_utilization'])
                raw_data[algo]['elapsed_times'] = np.array(raw_data[algo]['elapsed_times'])
                raw_data[algo]['n_evals'] = np.array(raw_data[algo]['n_evals'])
                if raw_data[algo]['optimality_gaps']:
                    raw_data[algo]['optimality_gaps'] = np.array(raw_data[algo]['optimality_gaps'])
                else:
                    raw_data[algo]['optimality_gaps'] = np.array([])
                
                logger.info(f"  Loaded {len(raw_data[algo]['runs'])} runs for {algo}")
            except Exception as e:
                logger.error(f"Error converting data for {algo}: {e}")
    
    return raw_data


def discover_knapsack_configs(results_dir: Union[str, Path]) -> List[Tuple[int, str, int]]:
    """Discover all Knapsack configurations from result files."""
    results_path = Path(results_dir)
    configs = set()
    
    for json_file in results_path.glob('knapsack_n*_*_seed*_*_*.json'):
        # Pattern: knapsack_n50_uncorrelated_seed42_FA_20251110T202419.json
        match = re.match(r'knapsack_n(\d+)_([a-z_]+)_seed(\d+)_[A-Z]{2}_\d{8}T\d{6}\.json', json_file.name)
        if match:
            n_items = int(match.group(1))
            instance_type = match.group(2)
            seed = int(match.group(3))
            configs.add((n_items, instance_type, seed))
    
    return sorted(configs)


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def friedman_test(results: Dict[str, List[float]]) -> Tuple[float, float]:
    """Perform Friedman test."""
    data = [results[algo] for algo in sorted(results.keys())]
    statistic, p_value = stats.friedmanchisquare(*data)
    return float(statistic), float(p_value)


def wilcoxon_test(data1: List[float], data2: List[float]) -> Tuple[float, float]:
    """Perform Wilcoxon signed-rank test."""
    statistic, p_value = stats.wilcoxon(data1, data2)
    return float(statistic), float(p_value)


def compute_ranks(results: Dict[str, List[float]]) -> Dict[str, float]:
    """Compute average ranks across runs."""
    algorithms = sorted(results.keys())
    n_runs = len(results[algorithms[0]])
    
    ranks: Dict[str, float] = {algo: 0.0 for algo in algorithms}
    
    for run_idx in range(n_runs):
        run_values = [(algo, results[algo][run_idx]) for algo in algorithms]
        run_values.sort(key=lambda x: x[1])
        
        for rank, (algo, _) in enumerate(run_values, 1):
            ranks[algo] += rank
    
    for algo in algorithms:
        ranks[algo] /= n_runs
    
    return ranks


def perform_rastrigin_statistical_tests(
    raw_data_by_config: Dict[str, Dict[str, Dict[str, Any]]]
) -> None:
    """Perform statistical tests on Rastrigin results."""
    
    for config_name in sorted(raw_data_by_config.keys()):
        print(f"\n{'-' * 80}")
        print(f"Configuration: {config_name}")
        print(f"{'-' * 80}")
        
        raw_data = raw_data_by_config[config_name]
        
        # Convert to Dict[algo, List[float]] for statistical tests
        results = {algo: data['error_to_optimum'].tolist() 
                  for algo, data in raw_data.items()}
        
        if len(results) < 2:
            print("Not enough algorithms for comparison")
            continue
        
        # Friedman test
        try:
            friedman_stat, friedman_p = friedman_test(results)
            print(f"\nFriedman Test:")
            print(f"  Statistic: {friedman_stat:.4f}")
            print(f"  P-value: {friedman_p:.4e}")
            print(f"  Significant: {'Yes' if friedman_p < 0.05 else 'No'}")
        except Exception as e:
            logger.warning(f"Friedman test failed: {e}")
        
        # Average ranks
        try:
            ranks = compute_ranks(results)
            print(f"\nAverage Ranks (lower is better):")
            for algo, rank in sorted(ranks.items(), key=lambda x: x[1]):
                print(f"  {algo}: {rank:.2f}")
        except Exception as e:
            logger.warning(f"Rank computation failed: {e}")
        
        # Pairwise Wilcoxon tests
        try:
            print(f"\nPairwise Wilcoxon Tests (p-values):")
            algorithms = sorted(results.keys())
            
            print(f"{'':>8}", end='')
            for algo in algorithms:
                print(f"{algo:>8}", end='')
            print()
            
            for algo1 in algorithms:
                print(f"{algo1:>8}", end='')
                for algo2 in algorithms:
                    if algo1 == algo2:
                        print(f"{'—':>8}", end='')
                    else:
                        _, p_val = wilcoxon_test(results[algo1], results[algo2])
                        sig = "*" if p_val < 0.05 else ""
                        print(f"{p_val:>7.4f}{sig}", end='')
                print()
        except Exception as e:
            logger.warning(f"Wilcoxon tests failed: {e}")


# ============================================================================
# SUMMARY GENERATION
# ============================================================================

def generate_rastrigin_summary(
    results_dir: Union[str, Path],
    output_file: Union[str, Path]
) -> Optional[pd.DataFrame]:
    """Generate Rastrigin summary CSV with success rates, hitting times, AUC."""
    
    output_path = Path(output_file)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Cannot create directory: {e}")
        return None
    
    configs = discover_rastrigin_configs(results_dir)
    if not configs:
        logger.error("No Rastrigin configurations found")
        return None
    
    summary_data = []
    
    for config_name in configs:
        raw_data = get_rastrigin_raw_data(results_dir, config_name)
        
        for algo, data in raw_data.items():
            errors = data['error_to_optimum']
            times = data['elapsed_times']
            histories = data['histories']
            
            if len(errors) == 0:
                continue
            
            row = {
                'Configuration': config_name,
                'Algorithm': algo,
                'Mean': np.mean(errors),
                'Std': np.std(errors),
                'Median': np.median(errors),
                'Best': np.min(errors),
                'Worst': np.max(errors),
                'Q1': np.percentile(errors, 25),
                'Q3': np.percentile(errors, 75),
                'Mean_Time': np.mean(times)
            }
            
            # Success rates at multiple tolerances
            for tol in RASTRIGIN_TOLS:
                sr = np.mean(errors <= tol) * 100.0
                row[f'SR_<={tol}'] = sr
            
            # Hitting times (median across runs)
            for tol in RASTRIGIN_TOLS:
                hts = [compute_hitting_time(h, tol) for h in histories if h]
                row[f'HT_med_<={tol}'] = float(np.median(hts)) if hts else np.nan
            
            # AUC (anytime performance)
            aucs = [compute_auc(h) for h in histories if h]
            row['AUC_median'] = float(np.median(aucs)) if aucs else np.nan
            row['AUC_mean'] = float(np.mean(aucs)) if aucs else np.nan
            
            summary_data.append(row)
    
    if not summary_data:
        logger.error("No data found for summary")
        return None
    
    try:
        df = pd.DataFrame(summary_data)
        df = df.sort_values(['Configuration', 'Algorithm'])
        df.to_csv(output_file, index=False)
        
        logger.info(f"Rastrigin summary saved to: {output_file}")
        print("\n" + "=" * 100)
        print(df.to_string(index=False))
        print("=" * 100)
        
        return df
    except Exception as e:
        logger.error(f"Error saving summary: {e}")
        return None


def generate_rastrigin_global_ranks(
    summary_df: pd.DataFrame,
    metric: str = 'AUC_median',
    output_file: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Compute average rank of each algorithm across all configurations.
    Metric: column to optimize (lower is better).
    """
    grouped = summary_df.groupby('Configuration')
    algo_metrics: Dict[str, List[float]] = {}
    
    for config, group in grouped:
        valid = group.dropna(subset=[metric])
        if len(valid) < 2:
            continue
        sorted_algos = valid.sort_values(metric)['Algorithm'].tolist()
        ranks = {algo: rank + 1 for rank, algo in enumerate(sorted_algos)}
        for algo, rank in ranks.items():
            algo_metrics.setdefault(algo, []).append(rank)
    
    avg_ranks = {algo: float(np.mean(rs)) for algo, rs in algo_metrics.items()}
    
    rank_df = pd.DataFrame(
        [{'Algorithm': a, 'Avg_Rank': r, 'N_Configs': len(algo_metrics[a])}
         for a, r in sorted(avg_ranks.items(), key=lambda x: x[1])]
    )
    
    if output_file:
        rank_df.to_csv(output_file, index=False)
        logger.info(f"Global ranks saved to: {output_file}")
    
    print(f"\nGlobal Rastrigin Ranks (by {metric}, lower is better):")
    print(rank_df.to_string(index=False))
    
    return rank_df


def generate_knapsack_summary(
    results_dir: Union[str, Path],
    output_file: Union[str, Path]
) -> Optional[pd.DataFrame]:
    """Generate Knapsack summary CSV with normalized values, success rates, hitting times."""
    
    output_path = Path(output_file)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Cannot create directory: {e}")
        return None
    
    configs = discover_knapsack_configs(results_dir)
    if not configs:
        logger.error("No Knapsack configurations found")
        return None
    
    summary_data = []
    
    for n_items, instance_type, seed in configs:
        raw_data = get_knapsack_raw_data(results_dir, n_items, instance_type, seed)
        
        dp_optimal = raw_data.get('dp_optimal')
        
        for algo in sorted(raw_data.keys()):
            if algo in ['dp_optimal', 'config']:
                continue
            
            data = raw_data[algo]
            best_values = data['best_values']
            gaps = data['optimality_gaps']
            feasibility = data['feasibility']
            times = data['elapsed_times']
            histories = data['histories']
            
            if len(best_values) == 0:
                continue
            
            row = {
                'n_items': n_items,
                'type': instance_type,
                'seed': seed,
                'Algorithm': algo,
                'Mean_Value': np.mean(best_values),
                'Std_Value': np.std(best_values),
                'Feasibility_Rate': np.mean(feasibility) * 100.0,
                'Mean_Time': np.mean(times),
                'DP_Optimal': dp_optimal
            }
            
            # Normalized value and gap
            if dp_optimal and dp_optimal > 0:
                norm_values = best_values / dp_optimal
                row['Mean_Norm_Value'] = float(np.mean(norm_values))
                row['Std_Norm_Value'] = float(np.std(norm_values))
                
                if len(gaps) > 0:
                    row['Mean_Gap_%'] = float(np.mean(gaps))
                    row['Std_Gap_%'] = float(np.std(gaps))
                    
                    # Success rates at gap thresholds
                    for tol in KNAPSACK_GAP_TOLS:
                        sr = np.mean(np.array(gaps) <= tol) * 100.0
                        row[f'SR_Gap_<={tol}%'] = sr
                    
                    # Hitting time to 1% gap target
                    target = dp_optimal * (1 - 0.01)  # 1% gap
                    hts = []
                    for h in histories:
                        if not h:
                            continue
                        t = None
                        for i, val in enumerate(h, start=1):
                            if val is not None and val >= target:
                                t = i
                                break
                        hts.append(t if t is not None else len(h))
                    row['HT_med_<=1%_gap'] = float(np.median(hts)) if hts else np.nan
            else:
                row['Mean_Norm_Value'] = np.nan
                row['Std_Norm_Value'] = np.nan
            
            summary_data.append(row)
    
    if not summary_data:
        logger.error("No data found for summary")
        return None
    
    try:
        df = pd.DataFrame(summary_data)
        df = df.sort_values(['n_items', 'type', 'seed', 'Algorithm'])
        df.to_csv(output_file, index=False)
        
        logger.info(f"Knapsack summary saved to: {output_file}")
        print("\n" + "=" * 100)
        print(df.to_string(index=False))
        print("=" * 100)
        
        return df
    except Exception as e:
        logger.error(f"Error saving summary: {e}")
        return None


def generate_knapsack_global_ranks(
    summary_df: pd.DataFrame,
    metric: str = 'Mean_Gap_%',
    output_file: Optional[Union[str, Path]] = None,
    minimize: bool = True
) -> pd.DataFrame:
    """
    Compute average rank of each algorithm across all configurations.
    Metric: column to optimize.
    minimize: if True, lower metric is better; if False, higher is better.
    """
    # Group by configuration (n_items, type, seed)
    group_cols = ['n_items', 'type', 'seed']
    grouped = summary_df.groupby(group_cols)
    algo_metrics: Dict[str, List[float]] = {}
    
    for (n, t, s), group in grouped:
        valid = group.dropna(subset=[metric])
        if len(valid) < 2:
            continue
        
        if minimize:
            sorted_algos = valid.sort_values(metric)['Algorithm'].tolist()
        else:
            sorted_algos = valid.sort_values(metric, ascending=False)['Algorithm'].tolist()
        
        ranks = {algo: rank + 1 for rank, algo in enumerate(sorted_algos)}
        for algo, rank in ranks.items():
            algo_metrics.setdefault(algo, []).append(rank)
    
    avg_ranks = {algo: float(np.mean(rs)) for algo, rs in algo_metrics.items()}
    
    rank_df = pd.DataFrame(
        [{'Algorithm': a, 'Avg_Rank': r, 'N_Configs': len(algo_metrics[a])}
         for a, r in sorted(avg_ranks.items(), key=lambda x: x[1])]
    )
    
    if output_file:
        rank_df.to_csv(output_file, index=False)
        logger.info(f"Global ranks saved to: {output_file}")
    
    direction = "lower better" if minimize else "higher better"
    print(f"\nGlobal Knapsack Ranks (by {metric}, {direction}):")
    print(rank_df.to_string(index=False))
    
    return rank_df


def main() -> None:
    """Main analysis pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze benchmark results')
    parser.add_argument('--problem', type=str, choices=['rastrigin', 'knapsack', 'all'],
                        default='all', help='Problem to analyze')
    parser.add_argument('--results-dir', type=str,
                        default='benchmark/results',
                        help='Results directory')
    parser.add_argument('--output-dir', type=str,
                        default='benchmark/results/summaries',
                        help='Output directory for summaries')
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 100)
    print("BENCHMARK ANALYSIS")
    print("=" * 100)
    
    if args.problem in ['rastrigin', 'all']:
        rastrigin_dir = Path(args.results_dir) / 'rastrigin'
        if rastrigin_dir.exists():
            logger.info("\n[RASTRIGIN ANALYSIS]")
            
            # Generate summary
            df = generate_rastrigin_summary(
                rastrigin_dir,
                str(output_path / 'rastrigin_summary.csv')
            )
            
            # Generate global ranks
            if df is not None:
                generate_rastrigin_global_ranks(
                    df,
                    metric='AUC_median',
                    output_file=str(output_path / 'rastrigin_global_ranks.csv')
                )
                
                # Statistical tests
                configs = discover_rastrigin_configs(rastrigin_dir)
                raw_data_by_config = {
                    config: get_rastrigin_raw_data(rastrigin_dir, config)
                    for config in configs
                }
                perform_rastrigin_statistical_tests(raw_data_by_config)
        else:
            logger.error(f"Rastrigin results not found at: {rastrigin_dir}")
    
    if args.problem in ['knapsack', 'all']:
        knapsack_dir = Path(args.results_dir) / 'knapsack'
        if knapsack_dir.exists():
            logger.info("\n[KNAPSACK ANALYSIS]")
            
            # Generate summary
            df = generate_knapsack_summary(
                knapsack_dir,
                str(output_path / 'knapsack_summary.csv')
            )
            
            # Generate global ranks
            if df is not None:
                generate_knapsack_global_ranks(
                    df,
                    metric='Mean_Gap_%',
                    output_file=str(output_path / 'knapsack_global_ranks.csv'),
                    minimize=True
                )
                
                logger.info("Knapsack summary and ranks generated successfully")
        else:
            logger.error(f"Knapsack results not found at: {knapsack_dir}")
    
    print("\n" + "=" * 100)
    print("Analysis complete!")
    print("=" * 100)


if __name__ == "__main__":
    main()