"""
Comprehensive benchmark results analysis with unified data loading architecture.
Supports multi-level analysis for both Rastrigin and Knapsack problems.
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
# UNIFIED DATA LOADING - CORE FUNCTION
# ============================================================================

def load_all_results_to_dataframe(results_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Universal data loader: scan all JSON files and create unified DataFrame.
    
    This is the CORE function that replaces all get_*_raw_data() functions.
    
    Returns
    -------
    pd.DataFrame
        Unified dataframe with columns:
        - Problem: 'rastrigin' or 'knapsack'
        - Scenario: e.g., 'out_of_the_box', 'specialist', 'repair', 'penalty'
        - Algorithm: 'FA', 'GA', 'SA', 'HC'
        - Configuration: for Rastrigin (e.g., 'quick_convergence')
        - N_Items, Instance_Type, Instance_Seed: for Knapsack
        - Algo_Seed: random seed for this run
        - Timestamp: file generation timestamp
        - Metrics: best_fitness, elapsed_time, etc.
        - Success_Levels: flattened (success_gold, hit_evals_gold, etc.)
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return pd.DataFrame()
    
    all_runs = []
    
    # Scan all JSON files
    json_files = list(results_path.rglob('*.json'))
    logger.info(f"Found {len(json_files)} JSON files in {results_dir}")
    
    for json_file in json_files:
        # Parse filename using regex
        filename = json_file.name
        
        # Try Rastrigin pattern: rastrigin_{config}_{algo}_{scenario}_{timestamp}.json
        match_rast = re.match(
            r'rastrigin_([a-z_]+)_([A-Z]{2})_([a-z_]+)_(\d{8}T\d{6})\.json',
            filename
        )
        
        # Try Knapsack pattern: knapsack_n{size}_{type}_seed{seed}_{algo}_{scenario}_{timestamp}.json
        match_knap = re.match(
            r'knapsack_n(\d+)_([a-z_]+)_seed(\d+)_([A-Z]{2})_([a-z_]+)_(\d{8}T\d{6})\.json',
            filename
        )
        
        # Load JSON safely
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {filename}: {e}")
            continue
        
        # Extract metadata and results
        if 'metadata' in data and 'all_results' in data:
            metadata = data['metadata']
            runs = data['all_results']
        elif 'metadata' in data and 'results' in data:
            metadata = data['metadata']
            runs = data['results']
        elif 'results' in data:
            metadata = {}
            runs = data['results']
        else:
            logger.warning(f"Unknown structure in {filename}")
            continue
        
        # Process based on problem type
        if match_rast:
            problem = 'rastrigin'
            config_name = match_rast.group(1)
            algorithm = match_rast.group(2)
            scenario = match_rast.group(3)
            timestamp = match_rast.group(4)
            
            # Process each run
            for run in runs:
                flat_run = {
                    'Problem': problem,
                    'Scenario': scenario,
                    'Algorithm': algorithm,
                    'Configuration': config_name,
                    'Algo_Seed': run.get('algo_seed') or run.get('seed'),
                    'Problem_Seed': run.get('problem_seed'),
                    'Timestamp': timestamp,
                    
                    # Core metrics
                    'Best_Fitness': run.get('best_fitness'),
                    'Elapsed_Time': run.get('elapsed_time'),
                    'Evaluations': run.get('evaluations'),
                    'Budget_Utilization': run.get('budget_utilization'),
                    'Status': run.get('status', 'ok'),
                    
                    # History
                    'History': run.get('history'),
                }
                
                # Flatten success_levels (multi-threshold)
                if 'success_levels' in run:
                    for level, level_data in run['success_levels'].items():
                        flat_run[f'Success_{level.capitalize()}'] = level_data.get('success', False)
                        flat_run[f'HitEvals_{level.capitalize()}'] = level_data.get('hit_evaluations')
                        flat_run[f'Threshold_{level.capitalize()}'] = level_data.get('threshold')
                
                all_runs.append(flat_run)
        
        elif match_knap:
            problem = 'knapsack'
            n_items = int(match_knap.group(1))
            instance_type = match_knap.group(2)
            instance_seed = int(match_knap.group(3))
            algorithm = match_knap.group(4)
            scenario = match_knap.group(5)
            timestamp = match_knap.group(6)
            
            dp_optimal = metadata.get('dp_optimal')
            
            # Process each run
            for run in runs:
                flat_run = {
                    'Problem': problem,
                    'Scenario': scenario,
                    'Algorithm': algorithm,
                    'N_Items': n_items,
                    'Instance_Type': instance_type,
                    'Instance_Seed': instance_seed,
                    'Algo_Seed': run.get('algo_seed') or run.get('seed'),
                    'Timestamp': timestamp,
                    
                    # Core metrics
                    'Best_Value': run.get('best_value'),
                    'Best_Fitness': run.get('best_fitness'),
                    'Total_Weight': run.get('total_weight'),
                    'Capacity': run.get('capacity'),
                    'Is_Feasible': run.get('is_feasible', False),
                    'Elapsed_Time': run.get('elapsed_time'),
                    'Items_Selected': run.get('items_selected'),
                    'Capacity_Utilization': run.get('capacity_utilization'),
                    'Evaluations': run.get('evaluations'),
                    'Budget_Utilization': run.get('budget_utilization'),
                    'Status': run.get('status', 'ok'),
                    
                    # DP optimal and gap
                    'DP_Optimal': dp_optimal,
                    'Optimality_Gap': run.get('optimality_gap'),
                    
                    # History
                    'History': run.get('history'),
                }
                
                all_runs.append(flat_run)
        else:
            logger.warning(f"Filename does not match known patterns: {filename}")
            continue
    
    if not all_runs:
        logger.error("No valid data found")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_runs)
    logger.info(f"Loaded {len(df)} runs across {df['Problem'].nunique()} problems")
    logger.info(f"  Rastrigin: {len(df[df['Problem']=='rastrigin'])} runs")
    logger.info(f"  Knapsack: {len(df[df['Problem']=='knapsack'])} runs")
    
    return df


# ============================================================================
# RASTRIGIN ANALYSIS
# ============================================================================

def analyze_rastrigin(df: pd.DataFrame, output_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """
    Comprehensive Rastrigin analysis with multi-level success rates.
    
    Returns
    -------
    dict
        Dictionary of summary DataFrames:
        - 'summary': Main summary table
        - 'ranks': Global ranking table
        - 'statistics': Statistical test results
    """
    df_rast = df[df['Problem'] == 'rastrigin'].copy()
    
    if df_rast.empty:
        logger.warning("No Rastrigin data found")
        return {}
    
    # Filter only successful runs for analysis
    df_rast = df_rast[df_rast['Status'] == 'ok']
    
    # Compute error to optimum (Rastrigin optimum = 0)
    df_rast['Error_To_Optimum'] = df_rast['Best_Fitness'].abs()
    
    print("\n" + "=" * 100)
    print("RASTRIGIN ANALYSIS")
    print("=" * 100)
    
    # Group by Configuration, Scenario, Algorithm
    group_cols = ['Configuration', 'Scenario', 'Algorithm']
    
    summary_data = []
    
    for (config, scenario, algo), group in df_rast.groupby(group_cols):
        errors = group['Error_To_Optimum'].values
        times = group['Elapsed_Time'].values
        
        row = {
            'Configuration': config,
            'Scenario': scenario,
            'Algorithm': algo,
            'N_Runs': len(errors),
            'Mean_Error': np.mean(errors),
            'Std_Error': np.std(errors),
            'Median_Error': np.median(errors),
            'Best_Error': np.min(errors),
            'Worst_Error': np.max(errors),
            'Q1_Error': np.percentile(errors, 25),
            'Q3_Error': np.percentile(errors, 75),
            'Mean_Time': np.mean(times),
        }
        
        # Multi-level success rates (gold, silver, bronze)
        for level in ['Gold', 'Silver', 'Bronze']:
            success_col = f'Success_{level}'
            hit_col = f'HitEvals_{level}'
            
            if success_col in group.columns:
                successes = group[success_col].values
                row[f'SR_{level}_%'] = np.mean(successes) * 100.0
                
                # Hitting times for successful runs
                hit_times = group[group[success_col] == True][hit_col].dropna().values
                row[f'HT_Med_{level}'] = float(np.median(hit_times)) if len(hit_times) > 0 else np.nan
        
        # AUC (anytime performance)
        aucs = []
        for hist in group['History'].values:
            if hist and isinstance(hist, list):
                auc = compute_auc(hist)
                if np.isfinite(auc):
                    aucs.append(auc)
        
        row['AUC_Median'] = float(np.median(aucs)) if aucs else np.nan
        row['AUC_Mean'] = float(np.mean(aucs)) if aucs else np.nan
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values(['Configuration', 'Scenario', 'Algorithm'])
    
    print("\n" + "-" * 100)
    print("SUMMARY TABLE")
    print("-" * 100)
    print(summary_df.to_string(index=False))
    
    # Statistical tests per (config, scenario)
    print("\n" + "-" * 100)
    print("STATISTICAL TESTS")
    print("-" * 100)
    
    stat_results = []
    
    for (config, scenario), group_df in df_rast.groupby(['Configuration', 'Scenario']):
        print(f"\nConfiguration: {config}, Scenario: {scenario}")
        
        algos = sorted(group_df['Algorithm'].unique())
        if len(algos) < 2:
            continue
        
        # Prepare data for tests
        results = {}
        for algo in algos:
            algo_errors = group_df[group_df['Algorithm'] == algo]['Error_To_Optimum'].values
            results[algo] = algo_errors.tolist()
        
        # Friedman test
        try:
            friedman_stat, friedman_p = friedman_test(results)
            print(f"  Friedman Test: χ²={friedman_stat:.4f}, p={friedman_p:.4e}, " +
                  f"Significant={'Yes' if friedman_p < 0.05 else 'No'}")
            
            stat_results.append({
                'Configuration': config,
                'Scenario': scenario,
                'Test': 'Friedman',
                'Statistic': friedman_stat,
                'P_Value': friedman_p,
                'Significant': friedman_p < 0.05
            })
        except Exception as e:
            logger.warning(f"Friedman test failed: {e}")
        
        # Average ranks
        try:
            ranks = compute_ranks(results)
            print(f"  Average Ranks (lower=better):")
            for algo, rank in sorted(ranks.items(), key=lambda x: x[1]):
                print(f"    {algo}: {rank:.2f}")
        except Exception as e:
            logger.warning(f"Rank computation failed: {e}")
    
    stat_df = pd.DataFrame(stat_results) if stat_results else pd.DataFrame()
    
    # Global ranks
    rank_df = generate_rastrigin_global_ranks(summary_df, metric='AUC_Median')
    
    # Save outputs
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary_df.to_csv(output_dir / 'rastrigin_summary.csv', index=False)
        rank_df.to_csv(output_dir / 'rastrigin_global_ranks.csv', index=False)
        if not stat_df.empty:
            stat_df.to_csv(output_dir / 'rastrigin_statistics.csv', index=False)
        
        logger.info(f"Rastrigin results saved to {output_dir}")
    
    return {
        'summary': summary_df,
        'ranks': rank_df,
        'statistics': stat_df
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_summary(
    df: pd.DataFrame,
    group_by_cols: List[str],
    value_cols: List[str],
    agg_funcs: Dict[str, str] = None
) -> pd.DataFrame:
    """
    Flexible summary creation function.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw data from load_all_results_to_dataframe()
    group_by_cols : List[str]
        Columns to group by
    value_cols : List[str]
        Columns to aggregate
    agg_funcs : Dict[str, str], optional
        Custom aggregation functions. Default: {'mean', 'std', 'median', 'min', 'max'}
    
    Returns
    -------
    pd.DataFrame
        Summary dataframe
    """
    if agg_funcs is None:
        # Default aggregations
        agg_funcs = {
            'mean': 'mean',
            'std': 'std',
            'median': 'median',
            'min': 'min',
            'max': 'max'
        }
    
    summary_data = []
    
    for group_key, group in df.groupby(group_by_cols):
        row = dict(zip(group_by_cols, group_key if isinstance(group_key, tuple) else [group_key]))
        row['N_Runs'] = len(group)
        
        for col in value_cols:
            if col not in group.columns:
                continue
            
            values = group[col].dropna().values
            if len(values) == 0:
                continue
            
            for agg_name, agg_func in agg_funcs.items():
                if agg_func == 'mean':
                    row[f'{col}_Mean'] = np.mean(values)
                elif agg_func == 'std':
                    row[f'{col}_Std'] = np.std(values)
                elif agg_func == 'median':
                    row[f'{col}_Median'] = np.median(values)
                elif agg_func == 'min':
                    row[f'{col}_Min'] = np.min(values)
                elif agg_func == 'max':
                    row[f'{col}_Max'] = np.max(values)
        
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)


# ============================================================================
# KNAPSACK ANALYSIS (UPDATED FOR MULTI-LEVEL SUMMARIES)
# ============================================================================

def analyze_knapsack(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None
) -> Dict[str, pd.DataFrame]:
    """
    Multi-level Knapsack analysis with flexible grouping.
    Creates BOTH instance-level and type-level summaries.
    
    Parameters
    ----------
    df : pd.DataFrame
        Unified dataframe from load_all_results_to_dataframe()
    output_dir : Path, optional
        Directory to save CSV outputs
    
    Returns
    -------
    dict
        Dictionary of summary DataFrames:
        - 'summary_by_instance': Instance-level detail
        - 'summary_by_type': Type-level aggregation
        - 'ranks': Global ranking
        - 'statistics': Statistical tests
    """
    df_knap = df[df['Problem'] == 'knapsack'].copy()
    
    if df_knap.empty:
        logger.warning("No Knapsack data found")
        return {}
    
    # Filter only successful runs
    df_knap = df_knap[df_knap['Status'] == 'ok']
    
    print("\n" + "=" * 100)
    print("KNAPSACK ANALYSIS (Multi-Level Summaries)")
    print("=" * 100)
    
    # ========================================================================
    # LEVEL 1: BY INSTANCE (Most detailed)
    # ========================================================================
    print("\n[LEVEL 1: Instance-Level Summary]")
    
    group_by_instance = ['N_Items', 'Instance_Type', 'Instance_Seed', 'Scenario', 'Algorithm']
    
    summary_by_instance = []
    
    for group_key, group in df_knap.groupby(group_by_instance):
        group_dict = dict(zip(group_by_instance, group_key))
        
        values = group['Best_Value'].values
        gaps = group['Optimality_Gap'].dropna().values
        times = group['Elapsed_Time'].values
        feasible = group['Is_Feasible'].values
        
        row = group_dict.copy()
        row.update({
            'N_Runs': len(values),
            'Mean_Value': np.mean(values),
            'Std_Value': np.std(values),
            'Median_Value': np.median(values),
            'Best_Value': np.max(values),
            'Worst_Value': np.min(values),
            'Mean_Time': np.mean(times),
            'Feasibility_%': np.mean(feasible) * 100.0,
        })
        
        # Optimality gap (if DP optimal available)
        if len(gaps) > 0:
            row['Mean_Gap_%'] = np.mean(gaps)
            row['Std_Gap_%'] = np.std(gaps)
            row['Median_Gap_%'] = np.median(gaps)
            row['Best_Gap_%'] = np.min(gaps)
            row['Worst_Gap_%'] = np.max(gaps)
            
            # Normalized value
            dp_opt = group['DP_Optimal'].iloc[0]
            if dp_opt and dp_opt > 0:
                row['Mean_Norm_Value'] = np.mean(values) / dp_opt
                row['Std_Norm_Value'] = np.std(values) / dp_opt
        
        # Success rates at gap thresholds
        for gap_tol in [1.0, 5.0, 10.0]:
            if len(gaps) > 0:
                sr = np.mean(gaps <= gap_tol) * 100.0
                row[f'SR_Gap<={gap_tol}%'] = sr
        
        summary_by_instance.append(row)
    
    df_by_instance = pd.DataFrame(summary_by_instance)
    df_by_instance = df_by_instance.sort_values(group_by_instance)
    
    print(f"  Created {len(df_by_instance)} instance-level records")
    
    # ========================================================================
    # LEVEL 2: BY TYPE (Aggregated across seeds)
    # ========================================================================
    print("\n[LEVEL 2: Type-Level Summary (averaged across instance seeds)]")
    
    if df_by_instance.empty:
        logger.warning("Cannot create type-level summary: instance-level is empty")
        df_by_type = pd.DataFrame()
    else:
        group_by_type = ['N_Items', 'Instance_Type', 'Scenario', 'Algorithm']
        
        # Aggregate from instance-level
        summary_by_type = []
        
        for group_key, group in df_by_instance.groupby(group_by_type):
            group_dict = dict(zip(group_by_type, group_key))
            
            row = group_dict.copy()
            row['N_Instances'] = len(group)  # Number of instance seeds
            row['Total_Runs'] = group['N_Runs'].sum()
            
            # Aggregate metrics
            for col in ['Mean_Value', 'Mean_Gap_%', 'Feasibility_%', 'Mean_Time']:
                if col in group.columns:
                    row[col] = group[col].mean()
                    row[f'{col}_Std'] = group[col].std()
            
            # Success rates
            for gap_tol in [1.0, 5.0, 10.0]:
                sr_col = f'SR_Gap<={gap_tol}%'
                if sr_col in group.columns:
                    row[sr_col] = group[sr_col].mean()
            
            summary_by_type.append(row)
        
        df_by_type = pd.DataFrame(summary_by_type)
        df_by_type = df_by_type.sort_values(group_by_type)
        
        print(f"  Created {len(df_by_type)} type-level records")
    
    # ========================================================================
    # PRINT SAMPLE OF INSTANCE-LEVEL SUMMARY
    # ========================================================================
    print("\n" + "-" * 100)
    print("INSTANCE-LEVEL SUMMARY (first 10 rows)")
    print("-" * 100)
    print(df_by_instance.head(10).to_string(index=False))
    
    # ========================================================================
    # PRINT SAMPLE OF TYPE-LEVEL SUMMARY
    # ========================================================================
    if not df_by_type.empty:
        print("\n" + "-" * 100)
        print("TYPE-LEVEL SUMMARY (all rows)")
        print("-" * 100)
        print(df_by_type.to_string(index=False))
    
    # ========================================================================
    # STATISTICAL TESTS (Use instance-level data)
    # ========================================================================
    print("\n" + "-" * 100)
    print("STATISTICAL TESTS (using instance-level data)")
    print("-" * 100)
    
    stat_results = []
    
    for group_key, group_df in df_knap.groupby(['N_Items', 'Instance_Type', 'Instance_Seed', 'Scenario']):
        group_dict = dict(zip(['N_Items', 'Instance_Type', 'Instance_Seed', 'Scenario'], group_key))
        print(f"\nGroup: {group_dict}")
        
        algos = sorted(group_df['Algorithm'].unique())
        if len(algos) < 2:
            continue
        
        # Use optimality gap if available, else use best value
        if 'Optimality_Gap' in group_df.columns and group_df['Optimality_Gap'].notna().sum() > 0:
            metric_col = 'Optimality_Gap'
            minimize = True
        else:
            metric_col = 'Best_Value'
            minimize = False
        
        results = {}
        for algo in algos:
            algo_data = group_df[group_df['Algorithm'] == algo][metric_col].dropna().values
            results[algo] = algo_data.tolist()
        
        # Friedman test
        try:
            friedman_stat, friedman_p = friedman_test(results)
            print(f"  Friedman Test: χ²={friedman_stat:.4f}, p={friedman_p:.4e}")
            
            stat_results.append({
                **group_dict,
                'Test': 'Friedman',
                'Metric': metric_col,
                'Statistic': friedman_stat,
                'P_Value': friedman_p,
                'Significant': friedman_p < 0.05
            })
        except Exception as e:
            logger.warning(f"Friedman test failed: {e}")
        
        # Average ranks
        try:
            ranks = compute_ranks(results)
            print(f"  Average Ranks:")
            for algo, rank in sorted(ranks.items(), key=lambda x: x[1]):
                print(f"    {algo}: {rank:.2f}")
        except Exception as e:
            logger.warning(f"Rank computation failed: {e}")
    
    stat_df = pd.DataFrame(stat_results) if stat_results else pd.DataFrame()
    
    # ========================================================================
    # GLOBAL RANKS (Computed from instance-level data)
    # ========================================================================
    rank_df = generate_knapsack_global_ranks(
        df_by_instance, 
        metric='Mean_Gap_%' if 'Mean_Gap_%' in df_by_instance.columns else 'Mean_Value'
    )
    
    # ========================================================================
    # SAVE ALL OUTPUTS
    # ========================================================================
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save instance-level summary
        df_by_instance.to_csv(output_dir / 'knapsack_summary_by_instance.csv', index=False)
        logger.info(f"✓ Saved: knapsack_summary_by_instance.csv ({len(df_by_instance)} rows)")
        
        # Save type-level summary
        if not df_by_type.empty:
            df_by_type.to_csv(output_dir / 'knapsack_summary_by_type.csv', index=False)
            logger.info(f"✓ Saved: knapsack_summary_by_type.csv ({len(df_by_type)} rows)")
        
        # Save ranks
        rank_df.to_csv(output_dir / 'knapsack_global_ranks.csv', index=False)
        logger.info(f"✓ Saved: knapsack_global_ranks.csv")
        
        # Save statistics
        if not stat_df.empty:
            stat_df.to_csv(output_dir / 'knapsack_statistics.csv', index=False)
            logger.info(f"✓ Saved: knapsack_statistics.csv")
        
        logger.info(f"All Knapsack results saved to {output_dir}")
    
    return {
        'summary_by_instance': df_by_instance,
        'summary_by_type': df_by_type,
        'ranks': rank_df,
        'statistics': stat_df
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_auc(history: List[float]) -> float:
    """Anytime performance: AUC under log(1+|f(x)|) curve."""
    if not history:
        return float('inf')
    
    h = np.array([abs(v) if v is not None else np.nan for v in history], dtype=float)
    h = np.nan_to_num(h, nan=(np.nanmax(h) if np.isfinite(np.nanmax(h)) else 1.0))
    
    x = np.arange(1, len(h) + 1, dtype=float)
    x = x / x[-1]
    y = np.log1p(h)
    
    return float(np.trapz(y, x))


def friedman_test(results: Dict[str, List[float]]) -> Tuple[float, float]:
    """Friedman test for multiple algorithms."""
    data = [results[algo] for algo in sorted(results.keys())]
    statistic, p_value = stats.friedmanchisquare(*data)
    return float(statistic), float(p_value)


def compute_ranks(results: Dict[str, List[float]]) -> Dict[str, float]:
    """Compute average ranks across runs."""
    algorithms = sorted(results.keys())
    n_runs = len(results[algorithms[0]])
    
    ranks = {algo: 0.0 for algo in algorithms}
    
    for run_idx in range(n_runs):
        run_values = [(algo, results[algo][run_idx]) for algo in algorithms]
        run_values.sort(key=lambda x: x[1])
        
        for rank, (algo, _) in enumerate(run_values, 1):
            ranks[algo] += rank
    
    for algo in algorithms:
        ranks[algo] /= n_runs
    
    return ranks


def generate_rastrigin_global_ranks(
    summary_df: pd.DataFrame,
    metric: str = 'AUC_Median'
) -> pd.DataFrame:
    """Compute global average ranks for Rastrigin."""
    grouped = summary_df.groupby(['Configuration', 'Scenario'])
    algo_metrics = defaultdict(list)
    
    for (config, scenario), group in grouped:
        valid = group.dropna(subset=[metric])
        if len(valid) < 2:
            continue
        
        sorted_algos = valid.sort_values(metric)['Algorithm'].tolist()
        ranks = {algo: rank + 1 for rank, algo in enumerate(sorted_algos)}
        
        for algo, rank in ranks.items():
            algo_metrics[algo].append(rank)
    
    avg_ranks = {algo: float(np.mean(rs)) for algo, rs in algo_metrics.items()}
    
    rank_df = pd.DataFrame([
        {'Algorithm': a, 'Avg_Rank': r, 'N_Configs': len(algo_metrics[a])}
        for a, r in sorted(avg_ranks.items(), key=lambda x: x[1])
    ])
    
    print(f"\nGlobal Rastrigin Ranks (by {metric}):")
    print(rank_df.to_string(index=False))
    
    return rank_df


def generate_knapsack_global_ranks(
    summary_df: pd.DataFrame,
    metric: str = 'Mean_Gap_%'
) -> pd.DataFrame:
    """Compute global average ranks for Knapsack."""
    # Group by all instance columns + scenario
    instance_cols = [col for col in ['N_Items', 'Instance_Type', 'Instance_Seed'] if col in summary_df.columns]
    group_cols = instance_cols + ['Scenario']
    
    grouped = summary_df.groupby(group_cols)
    algo_metrics = defaultdict(list)
    
    for group_key, group in grouped:
        valid = group.dropna(subset=[metric])
        if len(valid) < 2:
            continue
        
        sorted_algos = valid.sort_values(metric)['Algorithm'].tolist()
        ranks = {algo: rank + 1 for rank, algo in enumerate(sorted_algos)}
        
        for algo, rank in ranks.items():
            algo_metrics[algo].append(rank)
    
    avg_ranks = {algo: float(np.mean(rs)) for algo, rs in algo_metrics.items()}
    
    rank_df = pd.DataFrame([
        {'Algorithm': a, 'Avg_Rank': r, 'N_Configs': len(algo_metrics[a])}
        for a, r in sorted(avg_ranks.items(), key=lambda x: x[1])
    ])
    
    print(f"\nGlobal Knapsack Ranks (by {metric}):")
    print(rank_df.to_string(index=False))
    
    return rank_df


# ============================================================================
# MAIN FUNCTION WITH CLI (UPDATED)
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Comprehensive benchmark analysis with unified data loading',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze Rastrigin results
  python benchmark/analyze_results.py --problem rastrigin
  
  # Analyze Knapsack (creates both by_instance and by_type summaries)
  python benchmark/analyze_results.py --problem knapsack
  
  # Analyze both problems
  python benchmark/analyze_results.py --problem all --output-dir benchmark/results/summaries
        """
    )
    
    parser.add_argument(
        '--problem', type=str, required=True,
        choices=['rastrigin', 'knapsack', 'all'],
        help='Which problem to analyze'
    )
    parser.add_argument(
        '--results-dir', type=str,
        default='benchmark/results',
        help='Directory containing result JSON files'
    )
    parser.add_argument(
        '--output-dir', type=str,
        default='benchmark/results/summaries',
        help='Directory to save summary CSV files'
    )
    
    args = parser.parse_args()
    
    print("=" * 100)
    print("UNIFIED BENCHMARK ANALYSIS")
    print("=" * 100)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Problem: {args.problem}")
    
    # CORE: Load all data into unified DataFrame
    print("\nLoading all results...")
    df = load_all_results_to_dataframe(args.results_dir)
    
    if df.empty:
        logger.error("No data loaded, exiting")
        return
    
    print(f"\nLoaded {len(df)} runs:")
    print(df.groupby(['Problem', 'Scenario', 'Algorithm']).size())
    
    output_dir = Path(args.output_dir)
    
    # Run analyses
    if args.problem in ['rastrigin', 'all']:
        analyze_rastrigin(df, output_dir=output_dir)
    
    if args.problem in ['knapsack', 'all']:
        analyze_knapsack(df, output_dir=output_dir)
    
    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()