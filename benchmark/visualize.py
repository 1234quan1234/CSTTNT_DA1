"""
Generate comprehensive visualizations for benchmark results.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def load_rastrigin_results(results_dir: str, config_name: str) -> Dict:
    """Load Rastrigin results including histories."""
    results = {}
    results_path = Path(results_dir) / config_name
    
    for algo in ['FA', 'SA', 'HC', 'GA']:
        file_path = results_path / f"{algo}_results.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Extract best fitness and histories
                results[algo] = {
                    'best_fitness': [r['best_fitness'] for r in data],
                    'histories': [r['history'] for r in data]
                }
    
    return results


def plot_convergence_curves(results_dir: str, config_name: str, output_file: str):
    """Plot median convergence curves with IQR bands."""
    results = load_rastrigin_results(results_dir, config_name)
    
    if not results:
        print(f"No results found for {config_name}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = {'FA': '#FF6B6B', 'SA': '#4ECDC4', 'HC': '#45B7D1', 'GA': '#FFA07A'}
    
    for algo, data in results.items():
        histories = np.array(data['histories'])
        
        # Calculate median and IQR
        median = np.median(histories, axis=0)
        q25 = np.percentile(histories, 25, axis=0)
        q75 = np.percentile(histories, 75, axis=0)
        
        iterations = np.arange(len(median))
        
        # Plot median line
        ax.plot(iterations, median, label=algo, color=colors.get(algo, 'gray'),
                linewidth=2, alpha=0.9)
        
        # Plot IQR band
        ax.fill_between(iterations, q25, q75, color=colors.get(algo, 'gray'),
                        alpha=0.2)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Best Fitness', fontsize=12)
    ax.set_title(f'Convergence Curves - {config_name.replace("_", " ").title()}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Convergence plot saved to: {output_file}")


def plot_boxplots(results_dir: str, config_name: str, output_file: str):
    """Plot boxplots comparing final fitness distributions."""
    results = load_rastrigin_results(results_dir, config_name)
    
    if not results:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    algorithms = sorted(results.keys())
    data_to_plot = [results[algo]['best_fitness'] for algo in algorithms]
    
    bp = ax.boxplot(data_to_plot, labels=algorithms, patch_artist=True,
                    showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    
    # Color boxes
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Best Fitness', fontsize=12)
    ax.set_title(f'Final Fitness Distribution - {config_name.replace("_", " ").title()}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Boxplot saved to: {output_file}")


def plot_performance_bars(results_dir: str, config_name: str, output_file: str):
    """Plot bar chart with mean ± std error bars."""
    results = load_rastrigin_results(results_dir, config_name)
    
    if not results:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    algorithms = sorted(results.keys())
    means = [np.mean(results[algo]['best_fitness']) for algo in algorithms]
    stds = [np.std(results[algo]['best_fitness']) for algo in algorithms]
    
    x_pos = np.arange(len(algorithms))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.8,
                  error_kw={'linewidth': 2, 'ecolor': 'black'})
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algorithms, fontsize=11)
    ax.set_ylabel('Mean Best Fitness', fontsize=12)
    ax.set_title(f'Algorithm Performance - {config_name.replace("_", " ").title()}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mean:.2f}±{std:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Bar chart saved to: {output_file}")


def plot_scalability(results_dir: str, output_file: str):
    """Plot scalability: performance vs dimension."""
    configs = ['quick_convergence', 'multimodal_escape', 'scalability']
    dims = [10, 30, 50]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'FA': '#FF6B6B', 'SA': '#4ECDC4', 'HC': '#45B7D1', 'GA': '#FFA07A'}
    markers = {'FA': 'o', 'SA': 's', 'HC': '^', 'GA': 'D'}
    
    for algo in ['FA', 'SA', 'HC', 'GA']:
        means = []
        stds = []
        
        for config_name in configs:
            results = load_rastrigin_results(results_dir, config_name)
            
            if algo in results:
                fitness_values = results[algo]['best_fitness']
                means.append(np.mean(fitness_values))
                stds.append(np.std(fitness_values))
            else:
                means.append(np.nan)
                stds.append(np.nan)
        
        ax.errorbar(dims, means, yerr=stds, label=algo, marker=markers[algo],
                   markersize=8, linewidth=2, capsize=5, color=colors[algo])
    
    ax.set_xlabel('Dimension', fontsize=12)
    ax.set_ylabel('Mean Best Fitness', fontsize=12)
    ax.set_title('Scalability Analysis - Rastrigin Function',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale for better visualization
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Scalability plot saved to: {output_file}")


def plot_heatmap_winloss(results_dir: str, config_name: str, output_file: str):
    """Plot win-loss heatmap comparing algorithms."""
    results = load_rastrigin_results(results_dir, config_name)
    
    if not results:
        return
    
    algorithms = sorted(results.keys())
    n_algos = len(algorithms)
    
    # Create win-loss matrix
    win_matrix = np.zeros((n_algos, n_algos))
    
    for i, algo1 in enumerate(algorithms):
        for j, algo2 in enumerate(algorithms):
            if i == j:
                continue
            
            fitness1 = np.array(results[algo1]['best_fitness'])
            fitness2 = np.array(results[algo2]['best_fitness'])
            
            # Count wins (lower fitness is better)
            wins = np.sum(fitness1 < fitness2)
            win_matrix[i, j] = wins
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    
    im = ax.imshow(win_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=30)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Wins (out of 30)', fontsize=11)
    
    # Set ticks
    ax.set_xticks(np.arange(n_algos))
    ax.set_yticks(np.arange(n_algos))
    ax.set_xticklabels(algorithms)
    ax.set_yticklabels(algorithms)
    
    # Add text annotations
    for i in range(n_algos):
        for j in range(n_algos):
            if i != j:
                text = ax.text(j, i, int(win_matrix[i, j]),
                              ha="center", va="center", color="black", fontsize=10)
    
    ax.set_title(f'Win-Loss Matrix - {config_name.replace("_", " ").title()}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Algorithm (Column)', fontsize=11)
    ax.set_ylabel('Algorithm (Row)', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap saved to: {output_file}")


def generate_all_plots(results_dir: str = 'benchmark/results/rastrigin',
                      output_dir: str = 'benchmark/results/plots'):
    """Generate all visualization plots."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Generating Visualizations")
    print("=" * 70)
    
    configs = ['quick_convergence', 'multimodal_escape', 'scalability']
    
    for config_name in configs:
        print(f"\nProcessing {config_name}...")
        
        # Convergence curves
        plot_convergence_curves(
            results_dir, config_name,
            str(output_path / f'{config_name}_convergence.png')
        )
        
        # Boxplots
        plot_boxplots(
            results_dir, config_name,
            str(output_path / f'{config_name}_boxplot.png')
        )
        
        # Bar charts
        plot_performance_bars(
            results_dir, config_name,
            str(output_path / f'{config_name}_bars.png')
        )
        
        # Heatmaps
        plot_heatmap_winloss(
            results_dir, config_name,
            str(output_path / f'{config_name}_heatmap.png')
        )
    
    # Scalability plot
    print("\nGenerating scalability plot...")
    plot_scalability(results_dir, str(output_path / 'scalability.png'))
    
    print("\n" + "=" * 70)
    print(f"All plots saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate benchmark visualizations')
    parser.add_argument('--results-dir', type=str,
                        default='benchmark/results/rastrigin',
                        help='Results directory')
    parser.add_argument('--output-dir', type=str,
                        default='benchmark/results/plots',
                        help='Output directory for plots')
    
    args = parser.parse_args()
    
    generate_all_plots(args.results_dir, args.output_dir)
