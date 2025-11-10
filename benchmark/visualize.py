"""
Generate comprehensive visualizations for benchmark results.
Academic-grade benchmark visualization following metaheuristic best practices.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

# Import helpers from analyze_results
from benchmark.analyze_results import get_rastrigin_raw_data, get_knapsack_raw_data

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# Algorithm colors and markers
COLORS = {'FA': '#FF6B6B', 'SA': '#4ECDC4', 'HC': '#45B7D1', 'GA': '#FFA07A'}
MARKERS = {'FA': 'o', 'SA': 's', 'HC': '^', 'GA': 'D'}


# ============================================================================
# RASTRIGIN VISUALIZATIONS (Continuous Optimization)
# ============================================================================

def plot_rastrigin_convergence(results_dir: str, config_name: str, output_file: str):
    """
    Plot convergence curves: median error-to-optimum vs evaluations with IQR bands.
    This is the standard way to visualize metaheuristic convergence.
    """
    raw_data = get_rastrigin_raw_data(results_dir, config_name)
    
    if not raw_data:
        print(f"No results found for {config_name}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for algo, data in raw_data.items():
        histories = data['histories']
        
        # Convert to error-to-optimum
        error_histories = [np.abs(np.array(h) - 0.0) for h in histories]
        
        # Align to same evaluation budget (use shortest)
        min_len = min(len(h) for h in error_histories)
        error_histories = [h[:min_len] for h in error_histories]
        error_histories = np.array(error_histories)
        
        # Calculate median and IQR
        median = np.median(error_histories, axis=0)
        q25 = np.percentile(error_histories, 25, axis=0)
        q75 = np.percentile(error_histories, 75, axis=0)
        
        evals = np.arange(1, len(median) + 1)
        
        # Plot median line
        ax.plot(evals, median, label=algo, color=COLORS[algo],
                linewidth=2, alpha=0.9)
        
        # Plot IQR band
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
    
    print(f"Rastrigin convergence plot saved: {output_file}")


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
    
    # Color boxes
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
    
    print(f"Rastrigin boxplot saved: {output_file}")


def plot_rastrigin_ecdf(results_dir: str, config_name: str, output_file: str):
    """
    Plot Empirical Cumulative Distribution Function (ECDF) of final errors.
    Shows probability of achieving certain quality level.
    """
    raw_data = get_rastrigin_raw_data(results_dir, config_name)
    
    if not raw_data:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for algo in sorted(raw_data.keys()):
        errors = np.sort(raw_data[algo]['error_to_optimum'])
        n = len(errors)
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
    
    print(f"Rastrigin ECDF plot saved: {output_file}")


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
            
            if algo in raw_data:
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
    
    print(f"Rastrigin scalability plot saved: {output_file}")


# ============================================================================
# KNAPSACK VISUALIZATIONS (Constrained Discrete Optimization)
# ============================================================================

def normalize_instance_type(instance_type: str) -> str:
    """Map config instance type to filename format."""
    type_map = {
        'uncorrelated': 'uncorrelated',
        'weakly_correlated': 'weakly',
        'strongly_correlated': 'strongly',
        'subset_sum': 'subset'
    }
    return type_map.get(instance_type, instance_type)


def plot_knapsack_convergence(results_dir: str, n_items: int, instance_type: str,
                              instance_seed: int, output_file: str):
    """Plot convergence: best value vs evaluations with DP optimal reference."""
    # Normalize instance type for filename matching
    file_type = normalize_instance_type(instance_type)
    raw_data = get_knapsack_raw_data(results_dir, n_items, file_type, instance_seed)
    
    if not raw_data or len(raw_data) <= 2:
        print(f"No results for n={n_items}, {instance_type}, seed={instance_seed}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    dp_optimal = raw_data.get('dp_optimal')
    
    for algo in ['FA', 'GA', 'SA', 'HC']:
        if algo not in raw_data:
            continue
        
        data = raw_data[algo]
        histories = data['histories']
        
        # Align to same evaluation budget
        min_len = min(len(h) for h in histories)
        histories = [h[:min_len] for h in histories]
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
    ax.set_title(f'Knapsack Convergence - n={n_items}, {instance_type.replace("_", " ").title()}, seed={instance_seed}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Knapsack convergence saved: {output_file}")


def plot_knapsack_gap(results_dir: str, n_items: int, instance_type: str,
                     instance_seed: int, output_file: str):
    """Plot optimality gap distribution (only when DP optimal available)."""
    file_type = normalize_instance_type(instance_type)
    raw_data = get_knapsack_raw_data(results_dir, n_items, file_type, instance_seed)
    
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
    ax.set_title(f'Knapsack Optimality Gap - n={n_items}, {instance_type.replace("_", " ").title()}, seed={instance_seed}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Knapsack gap plot saved: {output_file}")


def plot_knapsack_feasibility(results_dir: str, output_file: str):
    """Plot feasibility rate across all configurations."""
    results_path = Path(results_dir)
    json_files = list(results_path.glob('n*_*.json'))  # Match actual format
    
    if not json_files:
        return
    
    # Collect data
    feasibility_data = []
    
    for json_file in json_files:
        filename = json_file.stem
        try:
            # Parse: n50_uncorrelated_seed42_FA.json
            parts = filename.rsplit('_', 1)
            algo = parts[1]
            if algo not in ['FA', 'GA', 'HC', 'SA']:
                continue
            
            config_part = parts[0]
            config_parts = config_part.split('_')
            n_items = int(config_parts[0][1:])
            seed_idx = next(i for i, p in enumerate(config_parts) if p.startswith('seed'))
            seed = int(config_parts[seed_idx].replace('seed', ''))
            instance_type = '_'.join(config_parts[1:seed_idx])
        except:
            continue
        
        raw_data = get_knapsack_raw_data(results_path, n_items, instance_type, seed)
        
        for algo in ['FA', 'GA', 'SA', 'HC']:
            if algo in raw_data:
                feas_rate = np.mean(raw_data[algo]['feasibility']) * 100
                feasibility_data.append({
                    'n_items': n_items,
                    'type': instance_type,
                    'Algorithm': algo,
                    'Feasibility': feas_rate
                })
    
    if not feasibility_data:
        return
    
    import pandas as pd
    df = pd.DataFrame(feasibility_data)
    
    # Group and plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, n in enumerate([50, 100, 200]):
        ax = axes[idx]
        df_n = df[df['n_items'] == n]
        
        if df_n.empty:
            continue
        
        # Pivot for grouped bar chart
        pivot = df_n.pivot_table(values='Feasibility', index='type', 
                                 columns='Algorithm', aggfunc='mean')
        
        pivot.plot(kind='bar', ax=ax, color=[COLORS[a] for a in pivot.columns])
        ax.set_title(f'n={n}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feasibility Rate (%)', fontsize=11)
        ax.set_xlabel('')
        ax.axhline(y=100, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylim([0, 105])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Knapsack Feasibility Rate', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Knapsack feasibility plot saved: {output_file}")


def plot_knapsack_capacity_utilization(results_dir: str, output_file: str):
    """Plot capacity utilization distribution."""
    results_path = Path(results_dir)
    json_files = list(results_path.glob('n*_*.json'))
    
    if not json_files:
        return
    
    # Collect data
    util_data = []
    
    for json_file in json_files:
        filename = json_file.stem
        try:
            parts = filename.rsplit('_', 1)
            algo = parts[1]
            if algo not in ['FA', 'GA', 'HC', 'SA']:
                continue
            
            config_part = parts[0]
            config_parts = config_part.split('_')
            n_items = int(config_parts[0][1:])
            seed_idx = next(i for i, p in enumerate(config_parts) if p.startswith('seed'))
            seed = int(config_parts[seed_idx].replace('seed', ''))
            instance_type = '_'.join(config_parts[1:seed_idx])
        except:
            continue
        
        raw_data = get_knapsack_raw_data(results_path, n_items, instance_type, seed)
        
        for algo in ['FA', 'GA', 'SA', 'HC']:
            if algo in raw_data:
                utils = raw_data[algo]['capacity_utilization']
                for u in utils:
                    util_data.append({
                        'n_items': n_items,
                        'Algorithm': algo,
                        'Utilization': u
                    })
    
    if not util_data:
        return
    
    import pandas as pd
    df = pd.DataFrame(util_data)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, n in enumerate([50, 100, 200]):
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
        ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Full')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Knapsack Capacity Utilization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Knapsack capacity utilization plot saved: {output_file}")


def plot_knapsack_scalability(results_dir: str, instance_type: str,
                              instance_seed: int, output_file: str):
    """Plot scalability: gap vs problem size."""
    sizes = [50, 100, 200]
    file_type = normalize_instance_type(instance_type)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for algo in ['FA', 'GA', 'SA', 'HC']:
        gaps = []
        stds = []
        
        for n_items in sizes:
            raw_data = get_knapsack_raw_data(results_dir, n_items, file_type, instance_seed)
            
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
    ax.set_title(f'Knapsack Scalability - {instance_type.replace("_", " ").title()}, seed={instance_seed}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Knapsack scalability saved: {output_file}")


def plot_knapsack_runtime_quality(results_dir: str, output_file: str):
    """Scatter plot: runtime vs quality trade-off."""
    results_path = Path(results_dir)
    json_files = list(results_path.glob('n*_*.json'))
    
    if not json_files:
        return
    
    # Collect data
    scatter_data = []
    
    for json_file in json_files:
        filename = json_file.stem
        try:
            parts = filename.rsplit('_', 1)
            algo = parts[1]
            if algo not in ['FA', 'GA', 'HC', 'SA']:
                continue
            
            config_part = parts[0]
            config_parts = config_part.split('_')
            n_items = int(config_parts[0][1:])
            seed_idx = next(i for i, p in enumerate(config_parts) if p.startswith('seed'))
            seed = int(config_parts[seed_idx].replace('seed', ''))
            instance_type = '_'.join(config_parts[1:seed_idx])
        except:
            continue
        
        raw_data = get_knapsack_raw_data(results_path, n_items, instance_type, seed)
        
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
    
    import pandas as pd
    df = pd.DataFrame(scatter_data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for algo in ['FA', 'GA', 'SA', 'HC']:
        df_algo = df[df['Algorithm'] == algo]
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
    
    print(f"Knapsack runtime-quality plot saved: {output_file}")


# ============================================================================
# MASTER GENERATION FUNCTION
# ============================================================================

def generate_all_plots(results_dir: str = 'benchmark/results',
                      knapsack_dir: str = 'benchmark/results',
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
        print(f"\n  Processing {config_name}...")
        
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
    
    print("\n  Generating Rastrigin scalability plot...")
    plot_rastrigin_scalability(
        results_dir,
        str(output_path / 'rastrigin_scalability.png')
    )
    
    # KNAPSACK PLOTS
    print("\n[KNAPSACK BENCHMARKS]")
    
    if not Path(knapsack_dir).exists():
        print(f"  Warning: Knapsack directory not found: {knapsack_dir}")
    else:
        sizes = [50, 100, 200]
        instance_types = ['uncorrelated', 'weakly_correlated', 
                         'strongly_correlated', 'subset_sum']
        seeds = [42, 43]
        
        # Per-instance plots
        for n_items in sizes:
            for inst_type in instance_types:
                if n_items == 200 and inst_type in ['strongly_correlated', 'subset_sum']:
                    continue
                
                for seed in seeds:
                    print(f"\n  Processing Knapsack n={n_items}, {inst_type}, seed={seed}...")
                    
                    prefix = f'knapsack_n{n_items}_{inst_type}_seed{seed}'
                    
                    plot_knapsack_convergence(
                        knapsack_dir, n_items, inst_type, seed,
                        str(output_path / f'{prefix}_convergence.png')
                    )
                    
                    plot_knapsack_gap(
                        knapsack_dir, n_items, inst_type, seed,
                        str(output_path / f'{prefix}_gap_boxplot.png')
                    )
        
        # Aggregate plots
        print("\n  Generating Knapsack aggregate plots...")
        
        plot_knapsack_feasibility(
            knapsack_dir,
            str(output_path / 'knapsack_feasibility.png')
        )
        
        plot_knapsack_capacity_utilization(
            knapsack_dir,
            str(output_path / 'knapsack_capacity_utilization.png')
        )
        
        plot_knapsack_runtime_quality(
            knapsack_dir,
            str(output_path / 'knapsack_runtime_quality.png')
        )
        
        # Scalability plots
        for inst_type in ['uncorrelated', 'weakly_correlated']:
            for seed in seeds:
                plot_knapsack_scalability(
                    knapsack_dir, inst_type, seed,
                    str(output_path / f'knapsack_{inst_type}_seed{seed}_scalability.png')
                )
    
    print("\n" + "=" * 80)
    print(f"All plots saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate academic-grade benchmark visualizations')
    parser.add_argument('--rastrigin-dir', type=str,
                        default='benchmark/results',
                        help='Rastrigin results directory')
    parser.add_argument('--knapsack-dir', type=str,
                        default='benchmark/results',
                        help='Knapsack results directory')
    parser.add_argument('--output-dir', type=str,
                        default='benchmark/results/plots',
                        help='Output directory for plots')
    
    args = parser.parse_args()
    
    generate_all_plots(args.rastrigin_dir, args.knapsack_dir, args.output_dir)
