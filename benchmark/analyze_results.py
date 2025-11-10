"""
Analyze and compare benchmark results.
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
import numpy.typing as npt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_results(
    result_dir: Union[str, Path],
    algorithm_names: Optional[List[str]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load results from JSON files.
    
    Parameters
    ----------
    result_dir : str or Path
        Directory containing result JSON files
    algorithm_names : list of str, optional
        List of algorithm names to load. Default: ['FA', 'SA', 'HC', 'GA']
    
    Returns
    -------
    dict
        Dictionary mapping algorithm name to list of run results
    """
    if algorithm_names is None:
        algorithm_names = ['FA', 'SA', 'HC', 'GA']
    
    results: Dict[str, List[Dict[str, Any]]] = {}
    for algo in algorithm_names:
        result_file = Path(result_dir) / f'{algo}_results.json'
        if result_file.exists():
            with open(result_file, 'r') as f:
                results[algo] = json.load(f)
    
    return results


def compute_statistics(
    results: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Dict[str, float]]:
    """
    Compute statistics for each algorithm.
    
    Parameters
    ----------
    results : dict
        Dictionary mapping algorithm name to list of run results
    
    Returns
    -------
    dict
        Dictionary mapping algorithm name to statistics dict with keys:
        'mean', 'std', 'median', 'min', 'max', 'q25', 'q75', 
        'mean_time', 'std_time'
    """
    stats_dict: Dict[str, Dict[str, float]] = {}
    
    for algo, runs in results.items():
        best_fits = [r['best_fitness'] for r in runs]
        times = [r['elapsed_time'] for r in runs]
        
        stats_dict[algo] = {
            'mean': float(np.mean(best_fits)),
            'std': float(np.std(best_fits)),
            'median': float(np.median(best_fits)),
            'min': float(np.min(best_fits)),
            'max': float(np.max(best_fits)),
            'q25': float(np.percentile(best_fits, 25)),
            'q75': float(np.percentile(best_fits, 75)),
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times))
        }
    
    return stats_dict


def statistical_comparison(
    results: Dict[str, List[Dict[str, Any]]],
    alpha: float = 0.05
) -> Tuple[npt.NDArray[np.float64], Dict[str, float]]:
    """
    Perform statistical tests comparing algorithms.
    
    Parameters
    ----------
    results : dict
        Dictionary mapping algorithm name to list of run results
    alpha : float, default=0.05
        Significance level for hypothesis testing
    
    Returns
    -------
    comparison_matrix : ndarray
        Pairwise p-value matrix
    p_values : dict
        Dictionary of comparison names to p-values
    """
    algorithms = list(results.keys())
    n_algos = len(algorithms)
    
    # Extract best fitness values
    fitness_data: Dict[str, List[float]] = {
        algo: [r['best_fitness'] for r in runs] 
        for algo, runs in results.items()
    }
    
    # Pairwise Mann-Whitney U tests
    comparison_matrix = np.zeros((n_algos, n_algos))
    p_values: Dict[str, float] = {}
    
    for i, algo1 in enumerate(algorithms):
        for j, algo2 in enumerate(algorithms):
            if i < j:
                statistic, p_value = stats.mannwhitneyu(
                    fitness_data[algo1], 
                    fitness_data[algo2],
                    alternative='two-sided'
                )
                p_values[f'{algo1}_vs_{algo2}'] = float(p_value)
                comparison_matrix[i, j] = p_value
                comparison_matrix[j, i] = p_value
    
    return comparison_matrix, p_values


def print_summary_table(stats_dict: Dict[str, Dict[str, float]]) -> None:
    """
    Print a summary table of results.
    
    Parameters
    ----------
    stats_dict : dict
        Dictionary of statistics from compute_statistics()
    """
    df = pd.DataFrame(stats_dict).T
    
    print("\n" + "=" * 80)
    print("STATISTICAL SUMMARY")
    print("=" * 80)
    print(df.to_string())
    print("=" * 80)


def print_comparison_results(
    p_values: Dict[str, float],
    alpha: float = 0.05
) -> None:
    """
    Print statistical comparison results.
    
    Parameters
    ----------
    p_values : dict
        Dictionary of comparison names to p-values
    alpha : float, default=0.05
        Significance threshold
    """
    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISON (Mann-Whitney U test)")
    print("=" * 80)
    
    for comparison, p_value in p_values.items():
        significance = "SIGNIFICANT" if p_value < alpha else "NOT significant"
        print(f"{comparison}: p-value = {p_value:.4f} ({significance})")
    
    print("=" * 80)


def analyze_benchmark(
    result_dir: Union[str, Path],
    config_name: str
) -> None:
    """
    Analyze results from a benchmark run.
    
    Parameters
    ----------
    result_dir : str or Path
        Base results directory
    config_name : str
        Name of configuration to analyze
    """
    print(f"\n{'=' * 80}")
    print(f"ANALYZING: {config_name}")
    print(f"{'=' * 80}")
    
    result_path = Path(result_dir) / config_name
    
    if not result_path.exists():
        print(f"Error: Results not found at {result_path}")
        return
    
    # Load results
    results = load_results(result_path)
    
    if not results:
        print("Error: No results found")
        return
    
    # Compute statistics
    stats_dict = compute_statistics(results)
    print_summary_table(stats_dict)
    
    # Statistical comparison
    if len(results) > 1:
        _, p_values = statistical_comparison(results)
        print_comparison_results(p_values)
    
    # Find best algorithm
    best_algo = min(stats_dict.items(), key=lambda x: x[1]['mean'])[0]
    print(f"\nBest algorithm (by mean): {best_algo}")
    print(f"Mean fitness: {stats_dict[best_algo]['mean']:.6f}")


def analyze_all_benchmarks(base_dir: Union[str, Path] = 'benchmark/results') -> None:
    """
    Analyze all benchmark results.
    
    Parameters
    ----------
    base_dir : str or Path, default='benchmark/results'
        Base directory containing all benchmark results
    """
    base_path = Path(base_dir)
    
    print("=" * 80)
    print("ANALYZING ALL BENCHMARKS")
    print("=" * 80)
    
    # Analyze Rastrigin
    rastrigin_dir = base_path / 'rastrigin'
    if rastrigin_dir.exists():
        for config_dir in rastrigin_dir.iterdir():
            if config_dir.is_dir():
                analyze_benchmark(rastrigin_dir, config_dir.name)
    
    # Analyze Knapsack
    knapsack_dir = base_path / 'knapsack'
    if knapsack_dir.exists():
        for config_dir in knapsack_dir.iterdir():
            if config_dir.is_dir():
                analyze_benchmark(knapsack_dir, config_dir.name)


def load_json_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Safely load JSON file with error handling.
    
    Parameters
    ----------
    file_path : Path
        Path to JSON file
    
    Returns
    -------
    dict or None
        Loaded JSON data, or None if error occurred
    """
    try:
        with open(file_path, 'r') as f:
            data: Dict[str, Any] = json.load(f)
        return data
    except FileNotFoundError:
        logger.warning(f"File not found: {file_path}")
        logger.info(f"  Suggestion: Check if benchmark has been run for this configuration")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path.name}")
        logger.error(f"  Error at line {e.lineno}, column {e.colno}: {e.msg}")
        logger.info(f"  Suggestion: File may be corrupted. Try re-running the benchmark.")
        return None
    except PermissionError:
        logger.error(f"Permission denied: {file_path}")
        logger.info(f"  Suggestion: Run 'chmod 644 {file_path}' to fix permissions")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading {file_path.name}: {type(e).__name__}: {e}")
        return None


def check_disk_space(path: Path, required_mb: int = 10) -> bool:
    """
    Check if sufficient disk space is available.
    
    Parameters
    ----------
    path : Path
        Directory to check
    required_mb : int, default=10
        Required free space in MB
        
    Returns
    -------
    bool
        True if sufficient space available
    """
    try:
        import shutil
        stat = shutil.disk_usage(path)
        free_mb = stat.free / (1024 * 1024)
        
        if free_mb < required_mb:
            logger.error(f"Insufficient disk space: {free_mb:.1f}MB free, {required_mb}MB required")
            logger.info(f"  Suggestion: Free up disk space in {path}")
            return False
        return True
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        return True  # Assume OK if check fails


def safe_save_plot(fig: Any, output_file: Union[str, Path]) -> None:
    """
    Safely save matplotlib figure with error handling.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    output_file : str or Path
        Output file path
    """
    output_path = Path(output_file)
    
    # Check directory exists
    if not output_path.parent.exists():
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {output_path.parent}")
        except OSError as e:
            logger.error(f"Cannot create directory {output_path.parent}: {e}")
            logger.info(f"  Suggestion: Check permissions or create manually: mkdir -p {output_path.parent}")
            return
    
    # Check disk space
    if not check_disk_space(output_path.parent, required_mb=10):
        return
    
    try:
        fig.savefig(str(output_file), dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot: {output_file}")
    except RuntimeError as e:
        logger.error(f"Failed to save plot {output_path.name}: {e}")
        if "out of memory" in str(e).lower():
            logger.info(f"  Suggestion: Reduce DPI or figure size")
        else:
            logger.info(f"  Suggestion: Check matplotlib backend and display settings")
    except OSError as e:
        logger.error(f"OS error saving plot {output_path.name}: {e}")
        if not check_disk_space(output_path.parent, required_mb=50):
            logger.info(f"  Suggestion: Free up disk space")
        else:
            logger.info(f"  Suggestion: Check file permissions and path validity")
    except Exception as e:
        logger.error(f"Unexpected error saving plot {output_path.name}: {type(e).__name__}: {e}")


def get_rastrigin_raw_data(
    results_dir: Union[str, Path],
    config_name: str
) -> Dict[str, Dict[str, Any]]:
    """
    Extract raw per-run data for Rastrigin.
    Supports both old structure (subdirectories) and new structure (flat with timestamps).
    
    Parameters
    ----------
    results_dir : str or Path
        Directory containing results
    config_name : str
        Configuration name (e.g., 'quick_convergence')
    
    Returns
    -------
    dict
        Dictionary mapping algorithm name to data dict with keys:
        'runs', 'best_fitness', 'error_to_optimum', 'histories', 
        'elapsed_times', 'n_evals'
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        logger.error(f"Results directory not found: {results_dir}")
        logger.info(f"  Suggestion: Run benchmark first: python benchmark/run_rastrigin.py --config {config_name}")
        return {}
    
    raw_data: Dict[str, Dict[str, Any]] = {}
    
    # Try new structure first: rastrigin_{config}_{algo}_{timestamp}.json
    pattern = f"rastrigin_{config_name}_*_*.json"
    json_files: List[Path] = list(results_path.glob(pattern))
    
    # If no files found, try old structure
    if not json_files:
        config_path = results_path / config_name
        if config_path.exists():
            for algo in ['FA', 'SA', 'HC', 'GA']:
                algo_file = config_path / f'{algo}_results.json'
                if algo_file.exists():
                    json_files.append(algo_file)
    
    if not json_files:
        logger.warning(f"No result files found for config '{config_name}' in {results_dir}")
        logger.info(f"  Searched pattern: {pattern}")
        logger.info(f"  Suggestion: Check config name or run benchmark")
        return {}
    
    logger.info(f"Found {len(json_files)} result file(s) for {config_name}")
    
    # Group by algorithm (handle multiple timestamps)
    algo_data: Dict[str, List[Dict[str, Any]]] = {}
    
    for json_file in json_files:
        data = load_json_file(json_file)
        if data is None:
            continue
        
        try:
            # Detect structure
            if 'metadata' in data and 'results' in data:
                # New structure
                algo: str = data['metadata']['algorithm']
                runs: List[Dict[str, Any]] = data['results']
            elif isinstance(data, list):
                # Old structure (list of runs)
                # Try to infer algorithm from filename
                if '_FA_' in json_file.name or json_file.name.startswith('FA'):
                    algo = 'FA'
                elif '_SA_' in json_file.name or json_file.name.startswith('SA'):
                    algo = 'SA'
                elif '_HC_' in json_file.name or json_file.name.startswith('HC'):
                    algo = 'HC'
                elif '_GA_' in json_file.name or json_file.name.startswith('GA'):
                    algo = 'GA'
                else:
                    logger.warning(f"Cannot infer algorithm from filename: {json_file.name}")
                    continue
                runs = data
            else:
                logger.warning(f"Unknown JSON structure in {json_file.name}")
                continue
            
            # Validate runs data
            if not isinstance(runs, list) or len(runs) == 0:
                logger.warning(f"Empty or invalid runs data in {json_file.name}")
                continue
            
            # Validate required fields
            required_fields = ['best_fitness', 'history', 'elapsed_time']
            for run in runs:
                missing = [f for f in required_fields if f not in run]
                if missing:
                    logger.warning(f"Missing fields in {json_file.name}: {missing}")
                    logger.info(f"  Using default values for missing fields")
                    if 'best_fitness' not in run:
                        run['best_fitness'] = float('inf')
                    if 'history' not in run:
                        run['history'] = []
                    if 'elapsed_time' not in run:
                        run['elapsed_time'] = 0.0
            
            if algo not in algo_data:
                algo_data[algo] = []
            algo_data[algo].extend(runs)
            
        except KeyError as e:
            logger.error(f"Missing key in {json_file.name}: {e}")
            logger.info(f"  Suggestion: File format may be outdated, try re-running benchmark")
            continue
        except Exception as e:
            logger.error(f"Error processing {json_file.name}: {type(e).__name__}: {e}")
            continue
    
    # Convert to expected format
    for algo, runs in algo_data.items():
        try:
            raw_data[algo] = {
                'runs': runs,
                'best_fitness': np.array([r['best_fitness'] for r in runs]),
                'error_to_optimum': np.array([abs(r['best_fitness'] - 0.0) for r in runs]),
                'histories': [r['history'] for r in runs],
                'elapsed_times': np.array([r['elapsed_time'] for r in runs]),
                'n_evals': np.array([r.get('evaluations', len(r['history'])) for r in runs])
            }
            logger.info(f"  Loaded {len(runs)} runs for {algo}")
        except Exception as e:
            logger.error(f"Error converting data for {algo}: {type(e).__name__}: {e}")
            continue
    
    return raw_data


def get_knapsack_raw_data(
    results_dir: Union[str, Path],
    n_items: int,
    instance_type: str,
    instance_seed: int
) -> Dict[str, Any]:
    """
    Extract raw per-run data for Knapsack.
    Supports both old structure and new structure with timestamps.
    
    Parameters
    ----------
    results_dir : str or Path
        Directory containing results
    n_items : int
        Number of items in knapsack
    instance_type : str
        Instance type (e.g., 'uncorrelated', 'weakly', 'strongly', 'subset')
    instance_seed : int
        Instance generation seed
    
    Returns
    -------
    dict
        Dictionary with keys 'dp_optimal', 'config', and algorithm names.
        Each algorithm maps to dict with keys: 'runs', 'best_values',
        'optimality_gaps', 'histories', 'feasibility', 
        'capacity_utilization', 'elapsed_times', 'n_evals'
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        logger.error(f"Results directory not found: {results_dir}")
        logger.info(f"  Suggestion: Run benchmark first: python benchmark/run_knapsack.py --size {n_items}")
        return {}
    
    # Try new structure: knapsack_n{size}_{type}_seed{seed}_{algo}_{timestamp}.json
    pattern = f"knapsack_n{n_items}_{instance_type}_seed{instance_seed}_*_*.json"
    json_files = list(results_path.glob(pattern))
    
    # If no files, try old structure without timestamp
    if not json_files:
        old_pattern = f"n{n_items}_{instance_type}_seed{instance_seed}_*.json"
        json_files = list(results_path.glob(old_pattern))
    
    if not json_files:
        logger.warning(f"No result files found for n={n_items}, type={instance_type}, seed={instance_seed}")
        logger.info(f"  Searched patterns: {pattern}")
        logger.info(f"  Suggestion: Run benchmark or check parameters")
        return {}
    
    logger.info(f"Found {len(json_files)} result file(s) for knapsack n={n_items}")
    
    raw_data = {
        'dp_optimal': None,
        'config': {}
    }
    
    # Load each algorithm's data
    for json_file in json_files:
        # Extract algorithm from filename
        filename = json_file.stem
        
        try:
            # Try new format: knapsack_n50_uncorrelated_seed42_FA_20240101T120000
            parts = filename.split('_')
            algo_idx = -2 if len(parts) > 5 and parts[-1].startswith('2') else -1
            algo = parts[algo_idx]
            
            if algo not in ['FA', 'SA', 'HC', 'GA']:
                logger.warning(f"Unknown algorithm in filename: {filename}")
                continue
            
            data = load_json_file(json_file)
            if data is None:
                continue
            
            # Detect structure
            if 'metadata' in data and 'results' in data:
                # New structure
                if not raw_data['config']:
                    raw_data['config'] = {
                        'n_items': data['metadata']['n_items'],
                        'instance_type': data['metadata']['instance_type'],
                        'instance_seed': data['metadata']['instance_seed']
                    }
                    raw_data['dp_optimal'] = data['metadata'].get('dp_optimal')
                
                results_list = data['results']
            elif 'config' in data and 'results' in data:
                # Old structure
                if not raw_data['config']:
                    raw_data['config'] = data['config']
                    raw_data['dp_optimal'] = data['config'].get('dp_optimal')
                
                results_list = data['results']
            else:
                logger.warning(f"Unknown structure in {json_file.name}")
                continue
            
            # Validate results
            if not isinstance(results_list, list) or len(results_list) == 0:
                logger.warning(f"Empty results in {json_file.name}")
                continue
            
            # Initialize algo data structure
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
            
            # Process all runs
            for result in results_list:
                # Validate required fields
                required_fields = ['best_value', 'is_feasible', 'history', 'elapsed_time']
                missing = [f for f in required_fields if f not in result]
                if missing:
                    logger.warning(f"Missing fields in result: {missing}")
                    if 'best_value' not in result:
                        result['best_value'] = 0.0
                    if 'is_feasible' not in result:
                        result['is_feasible'] = False
                    if 'history' not in result:
                        result['history'] = []
                    if 'elapsed_time' not in result:
                        result['elapsed_time'] = 0.0
                
                raw_data[algo]['runs'].append(result)
                raw_data[algo]['best_values'].append(result['best_value'])
                raw_data[algo]['histories'].append(result['history'])
                raw_data[algo]['feasibility'].append(1.0 if result['is_feasible'] else 0.0)
                raw_data[algo]['capacity_utilization'].append(result.get('capacity_utilization', 0.0))
                raw_data[algo]['elapsed_times'].append(result['elapsed_time'])
                raw_data[algo]['n_evals'].append(len(result['history']))
                
                # Calculate optimality gap
                if raw_data['dp_optimal'] is not None and result['is_feasible']:
                    gap = (raw_data['dp_optimal'] - result['best_value']) / raw_data['dp_optimal'] * 100
                    raw_data[algo]['optimality_gaps'].append(gap)
            
            # Convert to numpy arrays
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
                
                logger.info(f"  Loaded {len(results_list)} runs for {algo}")
            except Exception as e:
                logger.error(f"Error converting arrays for {algo}: {e}")
                continue
                
        except Exception as e:
            logger.error(f"Error loading {json_file.name}: {type(e).__name__}: {e}")
            continue
    
    return raw_data


def generate_rastrigin_summary(
    results_dir: Union[str, Path],
    output_file: Union[str, Path]
) -> Optional[pd.DataFrame]:
    """
    Generate summary CSV for Rastrigin results.
    
    Parameters
    ----------
    results_dir : str or Path
        Directory containing Rastrigin results
    output_file : str or Path
        Output CSV file path
    
    Returns
    -------
    DataFrame or None
        Summary DataFrame, or None if error occurred
    """
    
    configs = ['quick_convergence', 'multimodal_escape', 'scalability']
    summary_data = []
    
    # Create summaries directory
    output_path = Path(output_file)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Cannot create summaries directory: {e}")
        logger.info(f"  Suggestion: Check permissions: ls -ld {output_path.parent.parent}")
        return None
    
    for config_name in configs:
        try:
            raw_data = get_rastrigin_raw_data(results_dir, config_name)
            
            if not raw_data:
                logger.warning(f"No data for configuration '{config_name}', skipping")
                continue
            
            for algo, data in raw_data.items():
                try:
                    errors = data['error_to_optimum']
                    evals = data['n_evals']
                    times = data['elapsed_times']
                    
                    # Validate arrays
                    if len(errors) == 0:
                        logger.warning(f"No error data for {algo} in {config_name}")
                        continue
                    
                    summary_data.append({
                        'Configuration': config_name,
                        'Algorithm': algo,
                        'Mean': np.mean(errors),
                        'Std': np.std(errors),
                        'Median': np.median(errors),
                        'Best': np.min(errors),
                        'Worst': np.max(errors),
                        'Q1': np.percentile(errors, 25),
                        'Q3': np.percentile(errors, 75),
                        'Mean_Evals': np.mean(evals),
                        'Mean_Time': np.mean(times)
                    })
                except KeyError as e:
                    logger.error(f"Missing key for {algo} in {config_name}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing {algo} in {config_name}: {type(e).__name__}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error processing configuration '{config_name}': {type(e).__name__}: {e}")
            continue
    
    if not summary_data:
        logger.error("No data found for summary")
        logger.info("  Suggestion: Run benchmarks first")
        return None
    
    try:
        df = pd.DataFrame(summary_data)
        df = df.sort_values(['Configuration', 'Algorithm'])
        
        # Check disk space before writing
        if not check_disk_space(output_path.parent, required_mb=10):
            return None
        
        df.to_csv(output_file, index=False)
        
        logger.info(f"Rastrigin summary saved to: {output_file}")
        print("\n" + "=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)
        
        return df
    except pd.errors.EmptyDataError:
        logger.error("DataFrame is empty, cannot save CSV")
        return None
    except OSError as e:
        logger.error(f"Cannot write CSV file: {e}")
        logger.info(f"  Suggestion: Check permissions: ls -l {output_path.parent}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error saving summary: {type(e).__name__}: {e}")
        return None


def generate_knapsack_summary(
    results_dir: Union[str, Path],
    output_file: Union[str, Path]
) -> Optional[pd.DataFrame]:
    """
    Generate summary CSV for Knapsack results.
    
    Parameters
    ----------
    results_dir : str or Path
        Directory containing Knapsack results
    output_file : str or Path
        Output CSV file path
    
    Returns
    -------
    DataFrame or None
        Summary DataFrame, or None if error occurred
    """
    
    summary_data = []
    results_path = Path(results_dir)
    
    # Create summaries directory
    output_path = Path(output_file)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Cannot create summaries directory: {e}")
        return None
    
    if not results_path.exists():
        logger.error(f"Results directory not found: {results_dir}")
        logger.info(f"  Suggestion: Run knapsack benchmark first")
        return None
    
    # Find all JSON result files (both old and new patterns)
    json_files = list(results_path.glob('knapsack_*.json')) + list(results_path.glob('n*_*.json'))
    
    if not json_files:
        logger.warning(f"No JSON files found in {results_dir}")
        logger.info(f"  Suggestion: Run benchmark: python benchmark/run_knapsack.py")
        return None
    
    logger.info(f"Found {len(json_files)} knapsack result file(s)")
    
    # Group files by configuration (n_items, instance_type, seed)
    configs = {}
    for json_file in json_files:
        # Parse filename: n50_uncorrelated_seed42_FA.json
        filename = json_file.stem
        
        try:
            # Remove algorithm suffix
            parts = filename.rsplit('_', 1)  # Split from right
            config_part = parts[0]  # n50_uncorrelated_seed42
            algo = parts[1]  # FA, GA, HC, SA
            
            # Parse config: n50_uncorrelated_seed42
            config_parts = config_part.split('_')
            n_items = int(config_parts[0][1:])  # n50 -> 50
            
            # Find seed (last part starting with 'seed')
            seed_idx = next(i for i, p in enumerate(config_parts) if p.startswith('seed'))
            seed = int(config_parts[seed_idx].replace('seed', ''))
            
            # Instance type is everything between n_items and seed
            instance_type = '_'.join(config_parts[1:seed_idx])
            
            config_key = (n_items, instance_type, seed)
            if config_key not in configs:
                configs[config_key] = {}
            configs[config_key][algo] = json_file
            
        except (ValueError, IndexError, StopIteration) as e:
            logger.warning(f"Warning: Could not parse filename {filename}: {e}")
            continue
    
    # Process each configuration
    for (n_items, instance_type, seed), algo_files in configs.items():
        # Load one file to get config and dp_optimal
        first_file = next(iter(algo_files.values()))
        
        with open(first_file, 'r') as f:
            first_data = json.load(f)
        
        dp_optimal = first_data.get('dp_optimal')
        
        # Process each algorithm
        for algo in ['FA', 'SA', 'HC', 'GA']:
            if algo not in algo_files:
                continue
            
            algo_file = algo_files[algo]
            
            try:
                with open(algo_file, 'r') as f:
                    data = json.load(f)
                
                if not isinstance(data, dict):
                    continue
                
                best_value = data.get('best_value')
                best_values_history = data.get('history', [])
                is_feasible = data.get('is_feasible', False)
                capacity_util = data.get('capacity_utilization', 0.0)
                elapsed_time = data.get('elapsed_time', 0.0)
                
                row = {
                    'n_items': n_items,
                    'type': instance_type,
                    'seed': seed,
                    'Algorithm': algo,
                    'Best_Value': best_value,
                    'Feasible': 'Yes' if is_feasible else 'No',
                    'Capacity_Util': capacity_util,
                    'Elapsed_Time': elapsed_time,
                    'N_Evals': len(best_values_history)
                }
                
                # Add optimality gap if available
                if dp_optimal is not None and is_feasible:
                    gap = (dp_optimal - best_value) / dp_optimal * 100
                    row['DP_Optimal'] = dp_optimal
                    row['Gap_%'] = gap
                
                summary_data.append(row)
                
            except Exception as e:
                logger.warning(f"Warning: Error loading {algo_file}: {e}")
                continue
    
    if not summary_data:
        logger.error("No valid data found for summary")
        return None
    
    try:
        df = pd.DataFrame(summary_data)
        
        if df.empty:
            logger.error("DataFrame is empty after processing")
            return None
        
        df = df.sort_values(['n_items', 'type', 'seed', 'Algorithm'])
        
        if not check_disk_space(output_path.parent, required_mb=10):
            return None
        
        df.to_csv(output_file, index=False)
        
        logger.info(f"Knapsack summary saved to: {output_file}")
        print("\n" + "=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)
        
        return df
    except Exception as e:
        logger.error(f"Error saving knapsack summary: {type(e).__name__}: {e}")
        return None


def perform_statistical_tests(
    results_dir: Union[str, Path],
    problem: str = 'rastrigin'
) -> None:
    """
    Perform statistical tests and print results.
    
    Parameters
    ----------
    results_dir : str or Path
        Directory containing results
    problem : str, default='rastrigin'
        Problem type ('rastrigin' or 'knapsack')
    """
    
    print(f"\n{'=' * 80}")
    print(f"STATISTICAL ANALYSIS: {problem.upper()}")
    print(f"{'=' * 80}")
    
    if problem == 'rastrigin':
        configs = ['quick_convergence', 'multimodal_escape', 'scalability']
        
        for config_name in configs:
            print(f"\n{'-' * 80}")
            print(f"Configuration: {config_name}")
            print(f"{'-' * 80}")
            
            results = load_rastrigin_results(results_dir, config_name)
            
            if len(results) < 2:
                print("Not enough algorithms for comparison")
                continue
            
            # Friedman test
            friedman_stat, friedman_p = friedman_test(results)
            print(f"\nFriedman Test:")
            print(f"  Statistic: {friedman_stat:.4f}")
            print(f"  P-value: {friedman_p:.4e}")
            print(f"  Significant: {'Yes' if friedman_p < 0.05 else 'No'}")
            
            # Average ranks
            ranks = compute_ranks(results)
            print(f"\nAverage Ranks (lower is better):")
            for algo, rank in sorted(ranks.items(), key=lambda x: x[1]):
                print(f"  {algo}: {rank:.2f}")
            
            # Pairwise Wilcoxon tests
            print(f"\nPairwise Wilcoxon Tests (p-values):")
            algorithms = sorted(results.keys())
            
            # Create matrix
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
                        print(f"{p_val:>8.4f}", end='')
                print()


def friedman_test(
    results: Dict[str, List[float]]
) -> Tuple[float, float]:
    """
    Perform Friedman test on results.
    
    Parameters
    ----------
    results : dict
        Dictionary mapping algorithm name to list of fitness values
    
    Returns
    -------
    statistic : float
        Test statistic
    p_value : float
        P-value
    """
    data = [results[algo] for algo in sorted(results.keys())]
    statistic, p_value = stats.friedmanchisquare(*data)
    return float(statistic), float(p_value)


def wilcoxon_test(
    data1: List[float],
    data2: List[float]
) -> Tuple[float, float]:
    """
    Perform Wilcoxon signed-rank test.
    
    Parameters
    ----------
    data1 : list of float
        First dataset
    data2 : list of float
        Second dataset
    
    Returns
    -------
    statistic : float
        Test statistic
    p_value : float
        P-value
    """
    statistic, p_value = stats.wilcoxon(data1, data2)
    return float(statistic), float(p_value)


def compute_ranks(
    results: Dict[str, List[float]]
) -> Dict[str, float]:
    """
    Compute average ranks across runs.
    
    Parameters
    ----------
    results : dict
        Dictionary mapping algorithm name to list of fitness values
    
    Returns
    -------
    dict
        Dictionary mapping algorithm name to average rank
    """
    algorithms = sorted(results.keys())
    n_runs = len(results[algorithms[0]])
    
    ranks: Dict[str, float] = {algo: 0.0 for algo in algorithms}
    
    for run_idx in range(n_runs):
        run_values = [(algo, results[algo][run_idx]) for algo in algorithms]
        run_values.sort(key=lambda x: x[1])
        
        for rank, (algo, _) in enumerate(run_values, 1):
            ranks[algo] += rank
    
    # Average ranks
    for algo in algorithms:
        ranks[algo] /= n_runs
    
    return ranks


def load_rastrigin_results(
    results_dir: Union[str, Path],
    config_name: str
) -> Dict[str, List[float]]:
    """
    Load Rastrigin results for a specific configuration.
    
    Parameters
    ----------
    results_dir : str or Path
        Directory containing results
    config_name : str
        Configuration name
    
    Returns
    -------
    dict
        Dictionary mapping algorithm name to list of fitness values
    """
    config_path = Path(results_dir) / config_name
    
    results: Dict[str, List[float]] = {}
    for algo_file in config_path.glob('*_results.json'):
        algo_name = algo_file.stem.replace('_results', '')
        with open(algo_file, 'r') as f:
            data: List[Dict[str, Any]] = json.load(f)
            results[algo_name] = [r['best_fitness'] for r in data]
    
    return results


def main() -> None:
    """Main analysis pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze benchmark results')
    parser.add_argument('--problem', type=str, choices=['rastrigin', 'knapsack', 'all'],
                        default='all', help='Problem to analyze')
    parser.add_argument('--rastrigin-dir', type=str,
                        default='benchmark/results',
                        help='Rastrigin results directory')
    parser.add_argument('--knapsack-dir', type=str,
                        default='benchmark/results',
                        help='Knapsack results directory')
    parser.add_argument('--output-dir', type=str,
                        default='benchmark/results/summaries',
                        help='Output directory for summaries')
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if args.problem in ['rastrigin', 'all']:
        if Path(args.rastrigin_dir).exists():
            try:
                # Generate summary
                generate_rastrigin_summary(
                    args.rastrigin_dir,
                    str(output_path / 'rastrigin_summary.csv')
                )
                
                # Statistical tests
                perform_statistical_tests(args.rastrigin_dir, 'rastrigin')
            except Exception as e:
                logger.error(f"Error analyzing Rastrigin results: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.error(f"Rastrigin results not found at: {args.rastrigin_dir}")
    
    if args.problem in ['knapsack', 'all']:
        if Path(args.knapsack_dir).exists():
            try:
                # Generate summary
                df = generate_knapsack_summary(
                    args.knapsack_dir,
                    str(output_path / 'knapsack_summary.csv')
                )
                
                if df is not None:
                    logger.info("\nKnapsack summary generated successfully")
                else:
                    logger.warning("\nWarning: Could not generate Knapsack summary")
            except Exception as e:
                logger.error(f"Error analyzing Knapsack results: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.error(f"Knapsack results not found at: {args.knapsack_dir}")
    
    print(f"\n{'=' * 80}")
    print("Analysis complete!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()