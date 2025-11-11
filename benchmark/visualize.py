"""
Generate comprehensive visualizations for benchmark results.
Academic-grade benchmark visualization following metaheuristic best practices.
Updated to work with latest JSON structure and file naming conventions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
from collections import defaultdict
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# Algorithm colors and markers
COLORS = {'FA': '#FF6B6B', 'SA': '#4ECDC4', 'HC': '#45B7D1', 'GA': '#FFA07A'}
MARKERS = {'FA': 'o', 'SA': 's', 'HC': '^', 'GA': 'D'}


# ============================================================================
# UNIFIED DATA LOADING (Robust to file structure changes)
# ============================================================================

def load_json_safe(filepath: Path) -> Optional[dict]:
    """Safely load JSON file with error handling."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return None


def get_rastrigin_raw_data(results_dir: str, config_name: str) -> Dict:
    """
    Load Rastrigin results from files matching pattern:
    rastrigin_{config_name}_{algo}_{scenario}_{timestamp}.json
    
    Returns: {algo: {'histories': [...], 'error_to_optimum': [...], 'success_levels': {...}}}
    """
    results_path = Path(results_dir) / 'rastrigin'
    if not results_path.exists():
        results_path = Path(results_dir)
    
    pattern = f'rastrigin_{config_name}_*.json'
    json_files = list(results_path.glob(pattern))
    
    if not json_files:
        print(f"No results found for {config_name} in {results_path}")
        return {}
    
    raw_data = defaultdict(lambda: {'histories': [], 'error_to_optimum': []})
    
    for json_file in json_files:
        data = load_json_safe(json_file)
        if not data:
            continue
        
        # robust algo extraction (with or without strict timestamp)
        match = re.search(r'_(FA|GA|SA|HC)_\d{8}T\d{6}\.json$', json_file.name)
        if not match:
            match = re.search(r'_(FA|GA|SA|HC)\.json$', json_file.name)
        if not match:
            # can't infer algorithm -> skip
            continue
        algo = match.group(1)
        
        # normalize runs: support aggregated 'results', single-run dict, or list-of-runs
        runs = []
        if isinstance(data, dict) and 'results' in data and isinstance(data['results'], list):
            runs = data['results']
        elif isinstance(data, list):
            runs = data
        elif isinstance(data, dict):
            runs = [data]
        else:
            continue
        
        for run in runs:
            # tolerate multiple possible key names
            history = run.get('history') or run.get('hist') or run.get('trajectory') or []
            final_error = run.get('final_error')
            
            # many files store best fitness as 'best_fitness' -> convert
            if final_error is None:
                bf = run.get('best_fitness')
                if bf is not None:
                    # Rastrigin optimum = 0 -> error = abs(best_fitness)
                    final_error = abs(bf)
            
            # if still missing but history present, take last history value
            if final_error is None and history:
                try:
                    last = history[-1]
                    if isinstance(last, (int, float)):
                        final_error = abs(last)
                except Exception:
                    final_error = None
            
            if history:
                raw_data[algo]['histories'].append(list(history))
            if final_error is not None:
                raw_data[algo]['error_to_optimum'].append(float(final_error))
            
            # NEW: Extract success_levels if present
            if 'success_levels' in run:
                if 'success_levels' not in raw_data[algo]:
                    raw_data[algo]['success_levels'] = {}
                for level, level_data in run['success_levels'].items():
                    if level not in raw_data[algo]['success_levels']:
                        raw_data[algo]['success_levels'][level] = []
                    raw_data[algo]['success_levels'][level].append(level_data)
    
    return dict(raw_data)


def get_knapsack_raw_data(results_dir: str, n_items: int, 
                          instance_type: str, instance_seed: int) -> Dict:
    """
    Load Knapsack results from files matching pattern:
    knapsack_n{size}_{type}_seed{seed}_{algo}_{timestamp}.json
    
    Returns: {
        'dp_optimal': float,
        algo: {
            'histories': [...],
            'optimality_gaps': [...],
            'elapsed_times': [...],
            'capacity_utilization': [...],
            'feasibility': [...]
        }
    }
    """
    results_path = Path(results_dir) / 'knapsack'
    if not results_path.exists():
        results_path = Path(results_dir)
    
    pattern = f'knapsack_n{n_items}_{instance_type}_seed{instance_seed}_*.json'
    json_files = list(results_path.glob(pattern))
    
    if not json_files:
        return {}
    
    raw_data = {'dp_optimal': None}
    
    for json_file in json_files:
        data = load_json_safe(json_file)
        if not data:
            continue
        
        # Extract algorithm from filename
        match = re.search(r'_(FA|GA|SA|HC)_\d+T\d+\.json$', json_file.name)
        if not match:
            continue
        algo = match.group(1)
        
        if algo not in raw_data:
            raw_data[algo] = {
                'histories': [],
                'optimality_gaps': [],
                'elapsed_times': [],
                'capacity_utilization': [],
                'feasibility': []
            }
        
        # Extract DP optimal if present
        if 'config' in data and 'dp_optimal' in data['config']:
            raw_data['dp_optimal'] = data['config']['dp_optimal']
        
        # Extract run data
        if 'results' in data:
            # Aggregated format
            for run in data['results']:
                raw_data[algo]['histories'].append(run.get('history', []))
                if 'optimality_gap' in run:
                    raw_data[algo]['optimality_gaps'].append(run['optimality_gap'])
                if 'elapsed_time' in run:
                    raw_data[algo]['elapsed_times'].append(run['elapsed_time'])
                if 'capacity_utilization' in run:
                    raw_data[algo]['capacity_utilization'].append(run['capacity_utilization'])
                if 'is_feasible' in run:
                    raw_data[algo]['feasibility'].append(bool(run['is_feasible']))
        else:
            # Single run format
            raw_data[algo]['histories'].append(data.get('history', []))
            if 'optimality_gap' in data:
                raw_data[algo]['optimality_gaps'].append(data['optimality_gap'])
            if 'elapsed_time' in data:
                raw_data[algo]['elapsed_times'].append(data['elapsed_time'])
            if 'capacity_utilization' in data:
                raw_data[algo]['capacity_utilization'].append(data['capacity_utilization'])
            if 'is_feasible' in data:
                raw_data[algo]['feasibility'].append(bool(data['is_feasible']))
    
    return raw_data


def discover_knapsack_configs(results_dir: str) -> List[Tuple[int, str, int]]:
    """
    Discover all knapsack configurations from result files.
    Returns: [(n_items, instance_type, seed), ...]
    """
    results_path = Path(results_dir) / 'knapsack'
    if not results_path.exists():
        results_path = Path(results_dir)
    
    configs = set()
    
    for json_file in results_path.glob('knapsack_n*_*.json'):
        # Parse: knapsack_n100_uncorrelated_seed42_FA_20251110T202419.json
        match = re.match(r'knapsack_n(\d+)_([a-z_]+)_seed(\d+)_', json_file.name)
        if match:
            n_items = int(match.group(1))
            instance_type = match.group(2)
            seed = int(match.group(3))
            configs.add((n_items, instance_type, seed))
    
    return sorted(configs)


# ============================================================================
# RASTRIGIN VISUALIZATIONS (Continuous Optimization)
# ============================================================================

def plot_rastrigin_convergence(results_dir: str, config_name: str, output_file: str):
    """
    Plot convergence curves: median error-to-optimum vs evaluations with IQR bands.
    """
    raw_data = get_rastrigin_raw_data(results_dir, config_name)
    
    if not raw_data:
        print(f"No results found for {config_name}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for algo in sorted(raw_data.keys()):
        data = raw_data[algo]
        histories = data['histories']
        
        if not histories:
            continue
        
        # Convert to error-to-optimum (Rastrigin optimum = 0)
        error_histories = [np.abs(np.array(h)) for h in histories]
        
        # Align to same evaluation budget
        min_len = min(len(h) for h in error_histories)
        error_histories = [h[:min_len] for h in error_histories]
        error_histories = np.array(error_histories)
        
        # Calculate median and IQR
        median = np.median(error_histories, axis=0)
        q25 = np.percentile(error_histories, 25, axis=0)
        q75 = np.percentile(error_histories, 75, axis=0)
        
        evals = np.arange(1, len(median) + 1)
        
        # Plot median line with IQR band
        ax.plot(evals, median, label=algo, color=COLORS[algo],
                linewidth=2, alpha=0.9)
        ax.fill_between(evals, q25, q75, color=COLORS[algo], alpha=0.2)
    
    ax.set_xlabel('Function Evaluations', fontsize=12)
    ax.set_ylabel('Error to Optimum |f(x) - 0|', fontsize=12)
    ax.set_title(f'Convergence - Rastrigin {config_name.replace("_", " ").title()}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


def plot_rastrigin_boxplots(results_dir: str, config_name: str, output_file: str):
    """Plot boxplots of final error-to-optimum distributions."""
    raw_data = get_rastrigin_raw_data(results_dir, config_name)
    
    if not raw_data:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    algorithms = sorted(raw_data.keys())
    data_to_plot = [raw_data[algo]['error_to_optimum'] for algo in algorithms]
    
    bp = ax.boxplot(data_to_plot, labels=algorithms, patch_artist=True,
                    showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    
    colors = [COLORS[algo] for algo in algorithms]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Error to Optimum |f(x) - 0|', fontsize=12)
    ax.set_title(f'Final Error Distribution - Rastrigin {config_name.replace("_", " ").title()}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


def plot_rastrigin_ecdf(results_dir: str, config_name: str, output_file: str):
    """Plot Empirical Cumulative Distribution Function of final errors."""
    raw_data = get_rastrigin_raw_data(results_dir, config_name)
    
    if not raw_data:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for algo in sorted(raw_data.keys()):
        errors = np.sort(raw_data[algo]['error_to_optimum'])
        n = len(errors)
        if n == 0:
            continue
        ecdf = np.arange(1, n + 1) / n
        
        ax.plot(errors, ecdf, label=algo, color=COLORS[algo],
                linewidth=2.5, alpha=0.9, marker=MARKERS[algo], 
                markersize=4, markevery=max(1, n // 10))
    
    ax.set_xlabel('Error to Optimum |f(x) - 0|', fontsize=12)
    ax.set_ylabel('Cumulative Probability P(Error ≤ x)', fontsize=12)
    ax.set_title(f'ECDF - Rastrigin {config_name.replace("_", " ").title()}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


def plot_rastrigin_scalability(results_dir: str, output_file: str):
    """Plot scalability: mean error vs dimension with error bars."""
    configs = ['quick_convergence', 'multimodal_escape', 'scalability']
    dims = [10, 30, 50]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for algo in ['FA', 'GA', 'SA', 'HC']:
        means = []
        stds = []
        
        for config_name in configs:
            raw_data = get_rastrigin_raw_data(results_dir, config_name)
            
            if algo in raw_data and len(raw_data[algo]['error_to_optimum']) > 0:
                errors = raw_data[algo]['error_to_optimum']
                means.append(np.mean(errors))
                stds.append(np.std(errors))
            else:
                means.append(np.nan)
                stds.append(np.nan)
        
        ax.errorbar(dims, means, yerr=stds, label=algo, marker=MARKERS[algo],
                   markersize=10, linewidth=2.5, capsize=5, color=COLORS[algo])
    
    ax.set_xlabel('Dimension', fontsize=12)
    ax.set_ylabel('Mean Error to Optimum', fontsize=12)
    ax.set_title('Scalability Analysis - Rastrigin Function',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


# ============================================================================
# KNAPSACK VISUALIZATIONS (Constrained Discrete Optimization)
# ============================================================================

def plot_knapsack_convergence(results_dir: str, n_items: int, instance_type: str,
                              instance_seed: int, output_file: str):
    """Plot convergence: best value vs evaluations with DP optimal reference."""
    raw_data = get_knapsack_raw_data(results_dir, n_items, instance_type, instance_seed)
    
    if not raw_data or len(raw_data) <= 1:
        print(f"No results for n={n_items}, {instance_type}, seed={instance_seed}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    dp_optimal = raw_data.get('dp_optimal')
    
    for algo in ['FA', 'GA', 'SA', 'HC']:
        if algo not in raw_data:
            continue
        
        data = raw_data[algo]
        histories = data['histories']
        
        if not histories:
            continue
        
        # Align to same evaluation budget
        min_len = min(len(h) for h in histories if len(h) > 0)
        if min_len == 0:
            continue
        histories = [h[:min_len] for h in histories if len(h) > 0]
        histories = np.array(histories)
        
        # Calculate median and IQR
        median = np.median(histories, axis=0)
        q25 = np.percentile(histories, 25, axis=0)
        q75 = np.percentile(histories, 75, axis=0)
        
        evals = np.arange(1, len(median) + 1)
        
        ax.plot(evals, median, label=algo, color=COLORS[algo],
                linewidth=2, alpha=0.9)
        ax.fill_between(evals, q25, q75, color=COLORS[algo], alpha=0.2)
    
    # DP optimal reference line
    if dp_optimal:
        ax.axhline(y=dp_optimal, color='red', linestyle='--', linewidth=2, 
                   label=f'DP Optimal ({dp_optimal:.0f})', alpha=0.7)
    
    ax.set_xlabel('Function Evaluations', fontsize=12)
    ax.set_ylabel('Best Value Found', fontsize=12)
    ax.set_title(f'Knapsack Convergence - n={n_items}, {instance_type.title()}, seed={instance_seed}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


def plot_knapsack_gap(results_dir: str, n_items: int, instance_type: str,
                     instance_seed: int, output_file: str):
    """Plot optimality gap distribution (only when DP optimal available)."""
    raw_data = get_knapsack_raw_data(results_dir, n_items, instance_type, instance_seed)
    
    if not raw_data or not raw_data.get('dp_optimal'):
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    algorithms = []
    data_to_plot = []
    
    for algo in ['FA', 'GA', 'SA', 'HC']:
        if algo in raw_data and len(raw_data[algo]['optimality_gaps']) > 0:
            algorithms.append(algo)
            data_to_plot.append(raw_data[algo]['optimality_gaps'])
    
    if not algorithms:
        return
    
    bp = ax.boxplot(data_to_plot, tick_labels=algorithms, patch_artist=True,
                    showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    
    colors = [COLORS[algo] for algo in algorithms]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Optimality Gap (%)', fontsize=12)
    ax.set_title(f'Knapsack Optimality Gap - n={n_items}, {instance_type.title()}, seed={instance_seed}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


def plot_knapsack_aggregate_feasibility(results_dir: str, output_file: str):
    """Plot feasibility rate across all configurations."""
    configs = discover_knapsack_configs(results_dir)
    
    if not configs:
        print("No knapsack results found")
        return
    
    feasibility_data = []
    
    for n_items, instance_type, seed in configs:
        raw_data = get_knapsack_raw_data(results_dir, n_items, instance_type, seed)
        
        for algo in ['FA', 'GA', 'SA', 'HC']:
            if algo in raw_data and len(raw_data[algo]['feasibility']) > 0:
                feas_rate = np.mean(raw_data[algo]['feasibility']) * 100
                feasibility_data.append({
                    'n_items': n_items,
                    'type': instance_type,
                    'Algorithm': algo,
                    'Feasibility': feas_rate
                })
    
    if not feasibility_data:
        return
    
    df = pd.DataFrame(feasibility_data)
    
    # Get unique sizes
    sizes = sorted(df['n_items'].unique())
    n_sizes = len(sizes)
    
    fig, axes = plt.subplots(1, n_sizes, figsize=(5 * n_sizes, 5))
    if n_sizes == 1:
        axes = [axes]
    
    for idx, n in enumerate(sizes):
        ax = axes[idx]
        df_n = df[df['n_items'] == n]
        
        if df_n.empty:
            continue
        
        # Pivot for grouped bar chart
        pivot = df_n.pivot_table(values='Feasibility', index='type', 
                                 columns='Algorithm', aggfunc='mean')
        
        pivot.plot(kind='bar', ax=ax, color=[COLORS.get(a, 'gray') for a in pivot.columns])
        ax.set_title(f'n={n}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feasibility Rate (%)', fontsize=11)
        ax.set_xlabel('')
        ax.axhline(y=100, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylim([0, 105])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Knapsack Feasibility Rate', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


def plot_knapsack_aggregate_capacity(results_dir: str, output_file: str):
    """Plot capacity utilization distribution across all configs."""
    configs = discover_knapsack_configs(results_dir)
    
    if not configs:
        return
    
    import pandas as pd
    
    util_data = []
    
    for n_items, instance_type, seed in configs:
        raw_data = get_knapsack_raw_data(results_dir, n_items, instance_type, seed)
        
        for algo in ['FA', 'GA', 'SA', 'HC']:
            if algo in raw_data and len(raw_data[algo]['capacity_utilization']) > 0:
                for u in raw_data[algo]['capacity_utilization']:
                    util_data.append({
                        'n_items': n_items,
                        'Algorithm': algo,
                        'Utilization': u
                    })
    
    if not util_data:
        return
    
    df = pd.DataFrame(util_data)
    
    sizes = sorted(df['n_items'].unique())
    n_sizes = len(sizes)
    
    fig, axes = plt.subplots(1, n_sizes, figsize=(5 * n_sizes, 5))
    if n_sizes == 1:
        axes = [axes]
    
    for idx, n in enumerate(sizes):
        ax = axes[idx]
        df_n = df[df['n_items'] == n]
        
        if df_n.empty:
            continue
        
        data_to_plot = [df_n[df_n['Algorithm'] == algo]['Utilization'].values 
                       for algo in ['FA', 'GA', 'SA', 'HC'] 
                       if algo in df_n['Algorithm'].values]
        labels = [algo for algo in ['FA', 'GA', 'SA', 'HC'] 
                 if algo in df_n['Algorithm'].values]
        
        bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
        
        for patch, label in zip(bp['boxes'], labels):
            patch.set_facecolor(COLORS[label])
            patch.set_alpha(0.7)
        
        ax.set_title(f'n={n}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Capacity Utilization', fontsize=11)
        ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Knapsack Capacity Utilization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


def plot_knapsack_scalability(results_dir: str, instance_type: str,
                              instance_seed: int, output_file: str):
    """Plot scalability: gap vs problem size."""
    configs = discover_knapsack_configs(results_dir)
    
    # Filter by type and seed
    configs = [(n, t, s) for n, t, s in configs if t == instance_type and s == instance_seed]
    
    if not configs:
        return
    
    sizes = sorted(set(n for n, _, _ in configs))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for algo in ['FA', 'GA', 'SA', 'HC']:
        gaps = []
        stds = []
        
        for n_items in sizes:
            raw_data = get_knapsack_raw_data(results_dir, n_items, instance_type, instance_seed)
            
            if algo in raw_data and len(raw_data[algo]['optimality_gaps']) > 0:
                gaps.append(np.mean(raw_data[algo]['optimality_gaps']))
                stds.append(np.std(raw_data[algo]['optimality_gaps']))
            else:
                gaps.append(np.nan)
                stds.append(np.nan)
        
        ax.errorbar(sizes, gaps, yerr=stds, label=algo, marker=MARKERS[algo],
                   markersize=10, linewidth=2.5, capsize=5, color=COLORS[algo])
    
    ax.set_xlabel('Number of Items', fontsize=12)
    ax.set_ylabel('Mean Optimality Gap (%)', fontsize=12)
    ax.set_title(f'Knapsack Scalability - {instance_type.title()}, seed={instance_seed}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


def plot_knapsack_runtime_quality(results_dir: str, output_file: str):
    """Scatter plot: runtime vs quality trade-off."""
    configs = discover_knapsack_configs(results_dir)
    
    if not configs:
        return
    
    import pandas as pd
    
    scatter_data = []
    
    for n_items, instance_type, seed in configs:
        raw_data = get_knapsack_raw_data(results_dir, n_items, instance_type, seed)
        
        for algo in ['FA', 'GA', 'SA', 'HC']:
            if algo in raw_data and len(raw_data[algo]['optimality_gaps']) > 0:
                times = raw_data[algo]['elapsed_times']
                gaps = raw_data[algo]['optimality_gaps']
                
                for t, g in zip(times, gaps):
                    scatter_data.append({
                        'Algorithm': algo,
                        'Time': t,
                        'Gap': g,
                        'n_items': n_items
                    })
    
    if not scatter_data:
        return
    
    df = pd.DataFrame(scatter_data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for algo in ['FA', 'GA', 'SA', 'HC']:
        df_algo = df[df['Algorithm'] == algo]
        if not df_algo.empty:
            ax.scatter(df_algo['Time'], df_algo['Gap'], 
                      label=algo, color=COLORS[algo], marker=MARKERS[algo],
                      s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Elapsed Time (seconds)', fontsize=12)
    ax.set_ylabel('Optimality Gap (%)', fontsize=12)
    ax.set_title('Knapsack: Runtime vs Quality Trade-off',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


# ============================================================================
# ADVANCED PLOTS (Performance Profile, Success Rate, Global Ranks)
# ============================================================================

def plot_performance_profile(
    summary_df: pd.DataFrame,
    metric_col: str,
    output_file: str,
    minimize: bool = True,
    title: str = ''
):
    """
    Plot Dolan–Moré performance profile.
    
    metric_col: column to compare (AUC_median, Mean_Gap_%, etc.).
    minimize: if True, lower metric is better; if False, higher is better.
    Builds cumulative distribution: fraction of instances solved within tau * best.
    """
    if summary_df.empty or metric_col not in summary_df.columns:
        print(f"Warning: Cannot plot performance profile, metric '{metric_col}' not found")
        return
    
    algos = sorted(summary_df['Algorithm'].unique())
    
    # Identify instance grouping column(s)
    if 'Configuration' in summary_df.columns:
        instance_col = 'Configuration'
        group_cols = ['Configuration']
    elif all(col in summary_df.columns for col in ['n_items', 'type', 'seed']):
        group_cols = ['n_items', 'type', 'seed']
    else:
        print("Warning: Cannot identify instance grouping columns")
        return
    
    # Group by instances
    grouped = summary_df.groupby(group_cols)
    
    # Build performance ratios for each algo
    perf_ratios = {algo: [] for algo in algos}
    
    for _, group in grouped:
        valid = group.dropna(subset=[metric_col])
        if len(valid) < 1:
            continue
        
        # Find best performance on this instance
        if minimize:
            best = valid[metric_col].min()
        else:
            best = valid[metric_col].max()
        
        # Compute ratio for each algo
        for _, row in valid.iterrows():
            algo = row['Algorithm']
            metric_val = row[metric_col]
            
            if minimize:
                ratio = metric_val / best if best > 0 else 1.0
            else:
                ratio = best / metric_val if metric_val > 0 else 1.0
            
            perf_ratios[algo].append(ratio)
    
    # Plot CDF
    fig, ax = plt.subplots(figsize=(10, 6))
    
    taus = np.linspace(1.0, 5.0, 500)
    
    for algo in algos:
        ratios = np.array(perf_ratios[algo])
        if len(ratios) == 0:
            continue
        
        # Fraction of instances where ratio <= tau
        ys = [np.mean(ratios <= tau) for tau in taus]
        
        ax.plot(taus, ys, label=algo, color=COLORS.get(algo, 'gray'),
                linewidth=2.5, marker=MARKERS.get(algo, 'o'), markersize=3, markevery=50)
    
    ax.set_xlabel(r'$\tau$ (performance ratio factor)', fontsize=12)
    ax.set_ylabel('Fraction of instances solved within $\\tau \\cdot$ best', fontsize=12)
    ax.set_title(title or f'Performance Profile ({metric_col})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


def plot_global_ranks(
    rank_df: pd.DataFrame,
    output_file: str,
    title: str = ''
):
    """Plot global average ranks as bar chart."""
    if rank_df.empty:
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    algos = rank_df['Algorithm'].values
    ranks = rank_df['Avg_Rank'].values
    colors = [COLORS.get(a, 'gray') for a in algos]
    
    bars = ax.bar(algos, ranks, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, rank in zip(bars, ranks):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{rank:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Average Rank (lower is better)', fontsize=12)
    ax.set_title(title or 'Global Ranks', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, max(ranks) * 1.15])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


def plot_rastrigin_success_rates(
    summary_df: pd.DataFrame,
    output_file: str
):
    """Plot success rates at different tolerance levels for Rastrigin."""
    if summary_df.empty:
        return
    
    # Find all success rate columns
    sr_cols = [col for col in summary_df.columns if col.startswith('SR_<=')]
    if not sr_cols:
        print("Warning: No success rate columns found in Rastrigin summary")
        return
    
    algos = sorted(summary_df['Algorithm'].unique())
    n_tols = len(sr_cols)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(algos))
    width = 0.8 / n_tols
    
    for i, sr_col in enumerate(sorted(sr_cols)):
        # Mean success rate across all configurations
        sr_mean = summary_df.groupby('Algorithm')[sr_col].mean()
        values = [sr_mean.get(algo, 0) for algo in algos]
        
        # Extract tolerance from column name
        tol = sr_col.replace('SR_<=', '')
        
        ax.bar(x + i * width, values, width, label=f'tol={tol}',
               color=plt.cm.Blues(0.4 + 0.4 * i / n_tols), alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Rastrigin: Success Rates at Different Tolerances', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (n_tols - 1) / 2)
    ax.set_xticklabels(algos)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


def plot_knapsack_success_rates(
    summary_df: pd.DataFrame,
    output_file: str
):
    """Plot success rates at different gap levels for Knapsack."""
    if summary_df.empty:
        return
    
    sr_cols = [col for col in summary_df.columns if col.startswith('SR_Gap_<=')]
    if not sr_cols:
        print("Warning: No gap success rate columns found in Knapsack summary")
        return
    
    algos = sorted(summary_df['Algorithm'].unique())
    n_gaps = len(sr_cols)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(algos))
    width = 0.8 / n_gaps
    
    for i, sr_col in enumerate(sorted(sr_cols)):
        sr_mean = summary_df.groupby('Algorithm')[sr_col].mean()
        values = [sr_mean.get(algo, 0) for algo in algos]
        
        gap = sr_col.replace('SR_Gap_<=', '')
        
        ax.bar(x + i * width, values, width, label=f'gap ≤ {gap}',
               color=plt.cm.Greens(0.4 + 0.4 * i / n_gaps), alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Knapsack: Success Rates at Different Gap Thresholds', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (n_gaps - 1) / 2)
    ax.set_xticklabels(algos)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


def plot_rastrigin_hitting_times(
    summary_df: pd.DataFrame,
    output_file: str
):
    """Plot hitting times (median) for Rastrigin across algorithms and configs."""
    if summary_df.empty:
        return
    
    ht_cols = [col for col in summary_df.columns if col.startswith('HT_med_<=')]
    if not ht_cols:
        print("Warning: No hitting time columns found in Rastrigin summary")
        return
    
    algos = sorted(summary_df['Algorithm'].unique())
    configs = sorted(summary_df['Configuration'].unique())
    n_configs = len(configs)
    
    fig, ax = plt.subplots(figsize=(11, 6))
    
    x = np.arange(len(algos))
    width = 0.8 / n_configs
    
    for i, config in enumerate(configs):
        df_config = summary_df[summary_df['Configuration'] == config]
        
        # Use first hitting time column as representative
        ht_col = ht_cols[0]
        ht_values = []
        for algo in algos:
            row = df_config[df_config['Algorithm'] == algo]
            if not row.empty and pd.notna(row[ht_col].values[0]):
                ht_values.append(row[ht_col].values[0])
            else:
                ht_values.append(0)
        
        ax.bar(x + i * width, ht_values, width, label=config,
               alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel(f'Median Hitting Time (evals)', fontsize=12)
    ax.set_title('Rastrigin: Hitting Times to Target Tolerance', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (n_configs - 1) / 2)
    ax.set_xticklabels(algos)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


def plot_knapsack_runtime_quality_improved(
    summary_df: pd.DataFrame,
    output_file: str
):
    """
    Improved scatter plot: runtime vs quality (gap or normalized value).
    Bubble size represents feasibility rate.
    """
    if summary_df.empty or 'Mean_Time' not in summary_df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for algo in sorted(summary_df['Algorithm'].unique()):
        df_algo = summary_df[summary_df['Algorithm'] == algo]
        
        # Prefer gap if available, else use 1 - norm_value
        if 'Mean_Gap_%' in summary_df.columns:
            y = df_algo['Mean_Gap_%'].values
        elif 'Mean_Norm_Value' in summary_df.columns:
            y = (1.0 - df_algo['Mean_Norm_Value'].values) * 100.0
        else:
            continue
        
        x = df_algo['Mean_Time'].values
        
        # Size based on feasibility rate (if available)
        if 'Feasibility_Rate' in summary_df.columns:
            sizes = df_algo['Feasibility_Rate'].values * 2 + 50  # scale 50-250
        else:
            sizes = 100
        
        ax.scatter(x, y, s=sizes, label=algo, color=COLORS.get(algo, 'gray'),
                   marker=MARKERS.get(algo, 'o'), alpha=0.6,
                   edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Mean Runtime (seconds)', fontsize=12)
    ax.set_ylabel('Quality Gap (%)', fontsize=12)
    ax.set_title('Knapsack: Runtime–Quality Trade-off\n(bubble size = feasibility rate)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


# ============================================================================
# MASTER GENERATION FUNCTION
# ============================================================================

def generate_all_plots(results_dir: str = 'benchmark/results',
                      output_dir: str = 'benchmark/results/plots'):
    """Generate all academic-grade visualization plots."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 100)
    print("GENERATING ACADEMIC-GRADE BENCHMARK VISUALIZATIONS")
    print("=" * 100)
    
    # RASTRIGIN PLOTS
    print("\n[RASTRIGIN BENCHMARKS]")
    configs = ['quick_convergence', 'multimodal_escape', 'scalability']
    
    for config_name in configs:
        print(f"\nProcessing {config_name}...")
        
        plot_rastrigin_convergence(
            results_dir, config_name,
            str(output_path / f'rastrigin_{config_name}_convergence.png')
        )
        
        plot_rastrigin_boxplots(
            results_dir, config_name,
            str(output_path / f'rastrigin_{config_name}_boxplot.png')
        )
        
        plot_rastrigin_ecdf(
            results_dir, config_name,
            str(output_path / f'rastrigin_{config_name}_ecdf.png')
        )
    
    print("\nGenerating Rastrigin scalability plot...")
    plot_rastrigin_scalability(
        results_dir,
        str(output_path / 'rastrigin_scalability.png')
    )
    
    # KNAPSACK PLOTS
    print("\n[KNAPSACK BENCHMARKS]")
    
    configs = discover_knapsack_configs(results_dir)
    
    if not configs:
        print("Warning: No knapsack results found")
    else:
        print(f"Found {len(configs)} knapsack configurations")
        
        # Per-instance plots
        for n_items, instance_type, seed in configs:
            print(f"\nProcessing Knapsack n={n_items}, {instance_type}, seed={seed}...")
            
            prefix = f'knapsack_n{n_items}_{instance_type}_seed{seed}'
            
            plot_knapsack_convergence(
                results_dir, n_items, instance_type, seed,
                str(output_path / f'{prefix}_convergence.png')
            )
            
            plot_knapsack_gap(
                results_dir, n_items, instance_type, seed,
                str(output_path / f'{prefix}_gap_boxplot.png')
            )
        
        # Aggregate plots
        print("\nGenerating Knapsack aggregate plots...")
        
        plot_knapsack_aggregate_feasibility(
            results_dir,
            str(output_path / 'knapsack_feasibility.png')
        )
        
        plot_knapsack_aggregate_capacity(
            results_dir,
            str(output_path / 'knapsack_capacity_utilization.png')
        )
        
        plot_knapsack_runtime_quality(
            results_dir,
            str(output_path / 'knapsack_runtime_quality.png')
        )
        
        # Scalability plots (for instance types with multiple sizes)
        instance_types_seeds = set((t, s) for _, t, s in configs)
        for instance_type, seed in instance_types_seeds:
            sizes_for_type = [n for n, t, s in configs if t == instance_type and s == seed]
            if len(sizes_for_type) > 1:  # Only if multiple sizes exist
                plot_knapsack_scalability(
                    results_dir, instance_type, seed,
                    str(output_path / f'knapsack_{instance_type}_seed{seed}_scalability.png')
                )
    
    # ADVANCED PLOTS FROM SUMMARY FILES
    print("\n[ADVANCED PLOTS FROM SUMMARIES]")
    
    # Rastrigin advanced plots
    rastrigin_summary_path = Path(results_dir) / 'rastrigin_summary.csv'
    if rastrigin_summary_path.exists():
        print("\nGenerating Rastrigin advanced plots...")
        df_r = pd.read_csv(rastrigin_summary_path)
        
        plot_performance_profile(
            df_r, 'AUC_median',
            str(output_path / 'rastrigin_perf_profile_auc.png'),
            minimize=True,
            title='Rastrigin Performance Profile (AUC)'
        )
        
        plot_rastrigin_success_rates(
            df_r,
            str(output_path / 'rastrigin_success_rates.png')
        )
        
        plot_rastrigin_hitting_times(
            df_r,
            str(output_path / 'rastrigin_hitting_times.png')
        )
    
    # Rastrigin global ranks
    rastrigin_ranks_path = Path(results_dir) / 'rastrigin_global_ranks.csv'
    if rastrigin_ranks_path.exists():
        print("Plotting Rastrigin global ranks...")
        df_ranks = pd.read_csv(rastrigin_ranks_path)
        
        plot_global_ranks(
            df_ranks,
            str(output_path / 'rastrigin_global_ranks.png'),
            title='Rastrigin: Global Average Ranks (by AUC)'
        )
    
    # Knapsack advanced plots
    knapsack_summary_path = Path(results_dir) / 'knapsack_summary.csv'
    if knapsack_summary_path.exists():
        print("\nGenerating Knapsack advanced plots...")
        df_k = pd.read_csv(knapsack_summary_path)
        
        plot_performance_profile(
            df_k, 'Mean_Gap_%',
            str(output_path / 'knapsack_perf_profile_gap.png'),
            minimize=True,
            title='Knapsack Performance Profile (Gap%)'
        )
        
        plot_knapsack_success_rates(
            df_k,
            str(output_path / 'knapsack_success_rates.png')
        )
        
        plot_knapsack_runtime_quality_improved(
            df_k,
            str(output_path / 'knapsack_runtime_quality_improved.png')
        )
    
    # Knapsack global ranks
    knapsack_ranks_path = Path(results_dir) / 'knapsack_global_ranks.csv'
    if knapsack_ranks_path.exists():
        print("Plotting Knapsack global ranks...")
        df_ranks = pd.read_csv(knapsack_ranks_path)
        
        plot_global_ranks(
            df_ranks,
            str(output_path / 'knapsack_global_ranks.png'),
            title='Knapsack: Global Average Ranks (by Gap%)'
        )
    
    print("\n" + "=" * 100)
    print(f"All plots saved to: {output_path}")
    print("=" * 100)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate academic-grade benchmark visualizations')
    parser.add_argument('--results-dir', type=str,
                        default='benchmark/results',
                        help='Results directory (contains rastrigin/ and knapsack/)')
    parser.add_argument('--output-dir', type=str,
                        default='benchmark/results/plots',
                        help='Output directory for plots')
    parser.add_argument('--problem', type=str, choices=['all', 'rastrigin', 'knapsack'],
                        default='all',
                        help='Which problem to visualize')
    
    args = parser.parse_args()
    
    generate_all_plots(args.results_dir, args.output_dir)