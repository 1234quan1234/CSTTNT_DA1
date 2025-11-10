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
    rastrigin_{config_name}_{algo}_{timestamp}.json
    
    Returns: {algo: {'histories': [...], 'error_to_optimum': [...]}}
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
    
    import pandas as pd
    
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
# MASTER GENERATION FUNCTION
# ============================================================================

def generate_all_plots(results_dir: str = 'benchmark/results',
                      output_dir: str = 'benchmark/results/plots'):
    """Generate all academic-grade visualization plots."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("GENERATING ACADEMIC-GRADE BENCHMARK VISUALIZATIONS")
    print("=" * 80)
    
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
    
    print("\n" + "=" * 80)
    print(f"All plots saved to: {output_path}")
    print("=" * 80)


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
    
    if args.problem in ['all', 'rastrigin', 'knapsack']:
        generate_all_plots(args.results_dir, args.output_dir)
    else:
        generate_all_plots(args.results_dir, args.output_dir)