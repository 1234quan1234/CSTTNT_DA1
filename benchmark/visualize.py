"""
Modern academic-grade visualization using CSV summaries.
All plots are generated from pre-computed summary files, not raw JSON.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# Algorithm colors and markers
COLORS = {'FA': '#FF6B6B', 'SA': '#4ECDC4', 'HC': '#45B7D1', 'GA': '#FFA07A'}
MARKERS = {'FA': 'o', 'SA': 's', 'HC': '^', 'GA': 'D'}
SCENARIO_MARKERS = {
    'out_of_the_box': 'o',
    'specialist': 's',
    'repair': '^',
    'penalty': 'D'
}


# ============================================================================
# DATA LOADING (UPDATED: Load Multiple Summaries)
# ============================================================================

def load_summary_data(summary_path: Path) -> Optional[pd.DataFrame]:
    """
    Load summary CSV file.
    
    Parameters
    ----------
    summary_path : Path
        Path to CSV summary file
    
    Returns
    -------
    pd.DataFrame or None
        Loaded dataframe, or None if file not found
    """
    if not summary_path.exists():
        logger.warning(f"Summary file not found: {summary_path}")
        return None
    
    try:
        df = pd.read_csv(summary_path)
        logger.info(f"Loaded {len(df)} rows from {summary_path.name}")
        return df
    except Exception as e:
        logger.error(f"Error loading {summary_path}: {e}")
        return None

def load_available_summaries(summary_dir: Path, prefix: str) -> Dict[str, pd.DataFrame]:
    """
    Load all available summary files with given prefix.
    
    Parameters
    ----------
    summary_dir : Path
        Directory containing summary CSVs
    prefix : str
        Prefix of summary files (e.g., 'knapsack_summary')
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping summary type to DataFrame:
        - 'by_instance': Instance-level summary
        - 'by_type': Type-level summary
    """
    summaries = {}
    
    # Try to load by_instance
    instance_path = summary_dir / f'{prefix}_by_instance.csv'
    if instance_path.exists():
        try:
            df = pd.read_csv(instance_path)
            summaries['by_instance'] = df
            logger.info(f"Loaded {len(df)} rows from {instance_path.name}")
        except Exception as e:
            logger.error(f"Error loading {instance_path}: {e}")
    
    # Try to load by_type
    type_path = summary_dir / f'{prefix}_by_type.csv'
    if type_path.exists():
        try:
            df = pd.read_csv(type_path)
            summaries['by_type'] = df
            logger.info(f"Loaded {len(df)} rows from {type_path.name}")
        except Exception as e:
            logger.error(f"Error loading {type_path}: {e}")
    
    return summaries


# ============================================================================
# RASTRIGIN VISUALIZATIONS
# ============================================================================

def plot_rastrigin_boxplots_by_scenario(df: pd.DataFrame, output_dir: str):
    """
    Boxplots of final error for each configuration and scenario.
    Separate subplot for each (config, scenario) pair.
    """
    if df.empty:
        return
    
    configs = sorted(df['Configuration'].unique())
    scenarios = sorted(df['Scenario'].unique())
    
    n_configs = len(configs)
    n_scenarios = len(scenarios)
    
    fig, axes = plt.subplots(n_scenarios, n_configs, 
                             figsize=(5 * n_configs, 4 * n_scenarios),
                             squeeze=False)
    
    for i, scenario in enumerate(scenarios):
        for j, config in enumerate(configs):
            ax = axes[i, j]
            
            subset = df[(df['Configuration'] == config) & (df['Scenario'] == scenario)]
            
            if subset.empty:
                ax.axis('off')
                continue
            
            # Pivot for boxplot
            algos = sorted(subset['Algorithm'].unique())
            data_to_plot = [subset[subset['Algorithm'] == algo]['Median_Error'].values 
                           for algo in algos]
            
            bp = ax.boxplot(data_to_plot, labels=algos, patch_artist=True,
                           showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
            
            for patch, algo in zip(bp['boxes'], algos):
                patch.set_facecolor(COLORS.get(algo, 'gray'))
                patch.set_alpha(0.7)
            
            ax.set_title(f'{config}\n{scenario}', fontsize=10, fontweight='bold')
            ax.set_ylabel('Error to Optimum' if j == 0 else '')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Rastrigin: Final Error Distribution by Configuration & Scenario',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'rastrigin_boxplots_by_scenario.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved: {output_path}")


def plot_rastrigin_success_breakdown(df: pd.DataFrame, output_dir: str):
    """
    Multi-level success rate visualization (Gold, Silver, Bronze).
    Grouped bar chart for each configuration x scenario.
    """
    if df.empty:
        return
    
    # Check if multi-level columns exist
    level_cols = [col for col in df.columns if col.startswith('SR_')]
    if not level_cols:
        logger.warning("No success rate columns found")
        return
    
    configs = sorted(df['Configuration'].unique())
    
    for config in configs:
        df_config = df[df['Configuration'] == config]
        scenarios = sorted(df_config['Scenario'].unique())
        
        fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 5),
                                squeeze=False)
        axes = axes.flatten()
        
        for idx, scenario in enumerate(scenarios):
            ax = axes[idx]
            subset = df_config[df_config['Scenario'] == scenario]
            
            if subset.empty:
                ax.axis('off')
                continue
            
            algos = sorted(subset['Algorithm'].unique())
            
            # Find all SR columns (e.g., SR_Gold_%, SR_Silver_%, SR_Bronze_%)
            sr_cols = [col for col in subset.columns if col.startswith('SR_') and col.endswith('_%')]
            
            if not sr_cols:
                ax.axis('off')
                continue
            
            x = np.arange(len(algos))
            width = 0.8 / len(sr_cols)
            
            for i, sr_col in enumerate(sorted(sr_cols)):
                level_name = sr_col.replace('SR_', '').replace('_%', '')
                values = [subset[subset['Algorithm'] == algo][sr_col].values[0] 
                         if len(subset[subset['Algorithm'] == algo]) > 0 else 0
                         for algo in algos]
                
                color_intensity = 0.4 + 0.4 * i / len(sr_cols)
                ax.bar(x + i * width, values, width, label=level_name,
                      color=plt.cm.Blues(color_intensity), alpha=0.8,
                      edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Algorithm', fontsize=11)
            ax.set_ylabel('Success Rate (%)' if idx == 0 else '', fontsize=11)
            ax.set_title(f'{scenario.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.set_xticks(x + width * (len(sr_cols) - 1) / 2)
            ax.set_xticklabels(algos)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([0, 105])
        
        plt.suptitle(f'Rastrigin Multi-Level Success Rates - {config.replace("_", " ").title()}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = Path(output_dir) / f'rastrigin_success_breakdown_{config}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved: {output_path}")


def plot_rastrigin_cross_scenario_comparison(df: pd.DataFrame, output_dir: str):
    """
    Cross-scenario comparison: grouped bar chart showing performance of each
    algorithm across different scenarios.
    """
    if df.empty or 'Scenario' not in df.columns:
        return
    
    configs = sorted(df['Configuration'].unique())
    
    for config in configs:
        df_config = df[df['Configuration'] == config]
        
        # Pivot table: Algorithm x Scenario
        pivot = df_config.pivot_table(
            values='Median_Error',
            index='Algorithm',
            columns='Scenario',
            aggfunc='mean'
        )
        
        if pivot.empty:
            continue
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        pivot.plot(kind='bar', ax=ax, color=plt.cm.Set2.colors, alpha=0.8,
                  edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylabel('Median Error to Optimum', fontsize=12)
        ax.set_title(f'Rastrigin Cross-Scenario Comparison - {config.replace("_", " ").title()}',
                    fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(title='Scenario', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        
        output_path = Path(output_dir) / f'rastrigin_cross_scenario_{config}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved: {output_path}")


def plot_rastrigin_hitting_times(df: pd.DataFrame, output_dir: str):
    """
    Hitting time comparison across algorithms, configs, and scenarios.
    Uses first available hitting time column.
    """
    if df.empty:
        return
    
    # Find hitting time columns
    ht_cols = [col for col in df.columns if col.startswith('HT_Med_')]
    if not ht_cols:
        logger.warning("No hitting time columns found")
        return
    
    # Use first HT column
    ht_col = ht_cols[0]
    
    configs = sorted(df['Configuration'].unique())
    
    for config in configs:
        df_config = df[df['Configuration'] == config]
        scenarios = sorted(df_config['Scenario'].unique())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        algos = sorted(df_config['Algorithm'].unique())
        x = np.arange(len(algos))
        width = 0.8 / len(scenarios)
        
        for i, scenario in enumerate(scenarios):
            subset = df_config[df_config['Scenario'] == scenario]
            
            values = []
            for algo in algos:
                row = subset[subset['Algorithm'] == algo]
                if not row.empty and pd.notna(row[ht_col].values[0]):
                    values.append(row[ht_col].values[0])
                else:
                    values.append(0)
            
            ax.bar(x + i * width, values, width, label=scenario.replace('_', ' ').title(),
                  alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylabel('Median Hitting Time (evaluations)', fontsize=12)
        ax.set_title(f'Rastrigin Hitting Times - {config.replace("_", " ").title()}',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(scenarios) - 1) / 2)
        ax.set_xticklabels(algos)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')
        
        plt.tight_layout()
        
        output_path = Path(output_dir) / f'rastrigin_hitting_times_{config}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved: {output_path}")


# ============================================================================
# ADVANCED PLOTS (UPDATED: Fix Instance Column Detection)
# ============================================================================

def plot_performance_profile(
    df: pd.DataFrame,
    metric_col: str,
    output_file: str,
    minimize: bool = True,
    title: str = ''
):
    """Dolan–Moré performance profile from summary data."""
    if df.empty or metric_col not in df.columns:
        logger.warning(f"Cannot plot performance profile: {metric_col} not found")
        return
    
    algos = sorted(df['Algorithm'].unique())
    
    # Identify instance columns (FIXED FOR KNAPSACK)
    instance_cols = []
    
    # For Rastrigin
    if 'Configuration' in df.columns:
        instance_cols.append('Configuration')
    
    # For Knapsack (instance-level)
    if 'N_Items' in df.columns:
        instance_cols.append('N_Items')
    if 'Instance_Type' in df.columns:
        instance_cols.append('Instance_Type')
    if 'Instance_Seed' in df.columns:
        instance_cols.append('Instance_Seed')
    
    # Scenario is always included if present
    if 'Scenario' in df.columns:
        instance_cols.append('Scenario')
    
    if not instance_cols:
        logger.warning("Cannot identify instance grouping columns for Performance Profile")
        return
    
    logger.info(f"Performance Profile grouping by: {instance_cols}")
    
    # Group by instances
    perf_ratios = {algo: [] for algo in algos}
    
    for _, group in df.groupby(instance_cols):
        valid = group.dropna(subset=[metric_col])
        if len(valid) < 1:
            continue
        
        if minimize:
            best = valid[metric_col].min()
        else:
            best = valid[metric_col].max()
        
        if best == 0:
            continue  # Skip if best is zero (would cause division by zero)
        
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
            logger.warning(f"No performance ratios for {algo}")
            continue
        
        ys = [np.mean(ratios <= tau) for tau in taus]
        
        ax.plot(taus, ys, label=algo, color=COLORS.get(algo, 'gray'),
                linewidth=2.5, marker=MARKERS.get(algo, 'o'), markersize=3, markevery=50)
    
    ax.set_xlabel(r'$\tau$ (performance ratio)', fontsize=12)
    ax.set_ylabel('Fraction of instances solved within $\\tau \\times$ best', fontsize=12)
    ax.set_title(title or f'Performance Profile ({metric_col})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved: {output_file}")


# ============================================================================
# KNAPSACK VISUALIZATIONS (UPDATED: Smart Data Selection)
# ============================================================================

def plot_global_ranks(df: pd.DataFrame, output_file: str, title: str = ''):
    """Plot global average ranks from ranks CSV."""
    if df.empty:
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    algos = df['Algorithm'].values
    ranks = df['Avg_Rank'].values
    colors = [COLORS.get(a, 'gray') for a in algos]
    
    bars = ax.bar(algos, ranks, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    for bar, rank in zip(bars, ranks):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{rank:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Average Rank (lower is better)', fontsize=12)
    ax.set_title(title or 'Global Algorithm Ranks', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, max(ranks) * 1.15])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved: {output_file}")

def plot_knapsack_gap_comparison(df: pd.DataFrame, output_dir: str):
    """
    Comprehensive gap comparison: Grid layout showing all combinations.
    Rows = Scenarios, Columns = N_Items, X-axis = Instance_Type, Bars = Algorithms.
    """
    if df.empty or 'Mean_Gap_%' not in df.columns:
        logger.warning("No gap data available for Knapsack")
        return
    
    scenarios = sorted(df['Scenario'].unique())
    sizes = sorted(df['N_Items'].unique())
    
    n_scenarios = len(scenarios)
    n_sizes = len(sizes)
    
    fig, axes = plt.subplots(n_scenarios, n_sizes,
                             figsize=(6 * n_sizes, 5 * n_scenarios),
                             squeeze=False)
    
    for i, scenario in enumerate(scenarios):
        for j, n_items in enumerate(sizes):
            ax = axes[i, j]
            
            subset = df[(df['Scenario'] == scenario) & (df['N_Items'] == n_items)]
            
            if subset.empty:
                ax.axis('off')
                continue
            
            # Pivot for grouped bar chart: Instance_Type x Algorithm
            pivot = subset.pivot_table(
                values='Mean_Gap_%',
                index='Instance_Type',
                columns='Algorithm',
                aggfunc='mean'
            )
            
            if pivot.empty:
                ax.axis('off')
                continue
            
            # Plot grouped bars
            pivot.plot(kind='bar', ax=ax, color=[COLORS.get(a, 'gray') for a in pivot.columns],
                      alpha=0.8, edgecolor='black', linewidth=1)
            
            ax.set_title(f'{scenario.replace("_", " ").title()}\nn={n_items}',
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('Instance Type' if i == n_scenarios - 1 else '', fontsize=10)
            ax.set_ylabel('Mean Gap (%)' if j == 0 else '', fontsize=10)
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Knapsack: Optimality Gap Comparison Across All Configurations',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'knapsack_gap_comparison_grid.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved: {output_path}")


def plot_knapsack_cross_scenario_performance(df: pd.DataFrame, output_dir: str):
    """
    Dumbbell plot showing performance difference between repair and penalty.
    Each subplot = (N_Items, Instance_Type), Y-axis = Algorithms, X-axis = Gap%.
    """
    if df.empty or 'Scenario' not in df.columns:
        return
    
    # Check if we have both scenarios
    scenarios = sorted(df['Scenario'].unique())
    if len(scenarios) < 2:
        logger.warning("Need at least 2 scenarios for cross-scenario comparison")
        return
    
    sizes = sorted(df['N_Items'].unique())
    
    for n_items in sizes:
        df_n = df[df['N_Items'] == n_items]
        inst_types = sorted(df_n['Instance_Type'].unique())
        
        n_types = len(inst_types)
        
        fig, axes = plt.subplots(1, n_types, figsize=(6 * n_types, 6),
                                squeeze=False)
        axes = axes.flatten()
        
        for idx, inst_type in enumerate(inst_types):
            ax = axes[idx]
            
            subset = df_n[df_n['Instance_Type'] == inst_type]
            
            if subset.empty:
                ax.axis('off')
                continue
            
            # Pivot: Algorithm x Scenario
            pivot = subset.pivot_table(
                values='Mean_Gap_%',
                index='Algorithm',
                columns='Scenario',
                aggfunc='mean'
            )
            
            if pivot.empty or len(pivot.columns) < 2:
                ax.axis('off')
                continue
            
            algos = pivot.index.tolist()
            y_pos = np.arange(len(algos))
            
            # Plot dumbbells
            for i, algo in enumerate(algos):
                vals = pivot.loc[algo].values
                
                # Draw line connecting two scenarios
                ax.plot(vals, [i, i], color='gray', linewidth=2, alpha=0.5, zorder=1)
                
                # Draw points
                for j, (scenario, val) in enumerate(zip(pivot.columns, vals)):
                    marker = SCENARIO_MARKERS.get(scenario, 'o')
                    ax.scatter(val, i, s=150, marker=marker, 
                             color=COLORS.get(algo, 'gray'),
                             edgecolor='black', linewidth=1.5, 
                             label=f'{scenario}' if i == 0 else '', zorder=2)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(algos)
            ax.set_xlabel('Mean Optimality Gap (%)', fontsize=11)
            ax.set_title(f'{inst_type.replace("_", " ").title()}',
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            if idx == 0:
                handles, labels = ax.get_legend_handles_labels()
                # Remove duplicates
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), fontsize=9, loc='upper right')
        
        plt.suptitle(f'Knapsack Cross-Scenario Performance - n={n_items}\n(lines connect same algorithm across scenarios)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = Path(output_dir) / f'knapsack_cross_scenario_n{n_items}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved: {output_path}")


def plot_knapsack_feasibility_heatmap(df: pd.DataFrame, output_dir: str):
    """
    Heatmap showing feasibility rates across all configurations.
    Perfect for quickly validating that all solutions are valid.
    """
    if df.empty or 'Feasibility_%' not in df.columns:
        return
    
    # Create multi-index for rows
    df_copy = df.copy()
    df_copy['Config'] = df_copy['N_Items'].astype(str) + '_' + \
                        df_copy['Instance_Type'] + '_' + \
                        df_copy['Scenario']
    
    # Pivot: Config x Algorithm
    pivot = df_copy.pivot_table(
        values='Feasibility_%',
        index='Config',
        columns='Algorithm',
        aggfunc='mean'
    )
    
    if pivot.empty:
        return
    
    # Sort by N_Items and Scenario
    pivot = pivot.sort_index()
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.5)))
    
    # Create heatmap with custom colormap (green = 100%, yellow/red = below 100%)
    from matplotlib.colors import LinearSegmentedColormap
    colors_map = ['#d73027', '#fee08b', '#d9ef8b', '#1a9850']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('feasibility', colors_map, N=n_bins)
    
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap=cmap, vmin=90, vmax=100,
                cbar_kws={'label': 'Feasibility Rate (%)'}, ax=ax,
                linewidths=0.5, linecolor='gray')
    
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Configuration (Size_Type_Scenario)', fontsize=12)
    ax.set_title('Knapsack: Feasibility Rate Heatmap\n(Green = 100% valid solutions)',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'knapsack_feasibility_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved: {output_path}")


def plot_knapsack_scenario_impact(df: pd.DataFrame, output_dir: str):
    """
    Bar chart showing percentage improvement from penalty to repair for each algorithm.
    Aggregated across all instances.
    """
    if df.empty or 'Scenario' not in df.columns:
        return
    
    scenarios = sorted(df['Scenario'].unique())
    if len(scenarios) < 2:
        logger.warning("Need both repair and penalty scenarios")
        return
    
    # Assume scenarios are 'penalty' and 'repair'
    penalty_scenario = 'penalty'
    repair_scenario = 'repair'
    
    if penalty_scenario not in scenarios or repair_scenario not in scenarios:
        logger.warning(f"Expected scenarios 'repair' and 'penalty', found: {scenarios}")
        return
    
    algos = sorted(df['Algorithm'].unique())
    
    improvements = []
    
    for algo in algos:
        penalty_gap = df[(df['Algorithm'] == algo) & (df['Scenario'] == penalty_scenario)]['Mean_Gap_%'].mean()
        repair_gap = df[(df['Algorithm'] == algo) & (df['Scenario'] == repair_scenario)]['Mean_Gap_%'].mean()
        
        if pd.notna(penalty_gap) and pd.notna(repair_gap) and penalty_gap > 0:
            improvement = ((penalty_gap - repair_gap) / penalty_gap) * 100.0
            improvements.append(improvement)
        else:
            improvements.append(0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = [COLORS.get(a, 'gray') for a in algos]
    bars = ax.bar(algos, improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, val in zip(bars, improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{val:+.1f}%', ha='center', va='bottom' if val >= 0 else 'top',
                fontsize=11, fontweight='bold')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Gap Improvement from Penalty to Repair (%)', fontsize=12)
    ax.set_title('Knapsack: Impact of Repair Strategy\n(Positive = Repair is better)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'knapsack_scenario_impact.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved: {output_path}")


def plot_knapsack_algorithm_ranking_by_size(df: pd.DataFrame, output_dir: str):
    """
    Show how algorithm rankings change with problem size.
    Line plot: X = N_Items, Y = Average Rank, Lines = Algorithms.
    """
    if df.empty:
        return
    
    sizes = sorted(df['N_Items'].unique())
    algos = sorted(df['Algorithm'].unique())
    
    # Calculate average gap for each (size, algo) combination
    ranks_by_size = {}
    
    for n_items in sizes:
        df_n = df[df['N_Items'] == n_items]
        
        # Calculate mean gap for each algorithm
        algo_gaps = {}
        for algo in algos:
            mean_gap = df_n[df_n['Algorithm'] == algo]['Mean_Gap_%'].mean()
            if pd.notna(mean_gap):
                algo_gaps[algo] = mean_gap
        
        # Rank algorithms (1 = best)
        sorted_algos = sorted(algo_gaps.items(), key=lambda x: x[1])
        ranks = {algo: rank + 1 for rank, (algo, _) in enumerate(sorted_algos)}
        
        ranks_by_size[n_items] = ranks
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for algo in algos:
        ranks = [ranks_by_size[n].get(algo, np.nan) for n in sizes]
        
        ax.plot(sizes, ranks, label=algo, marker=MARKERS.get(algo, 'o'),
                color=COLORS.get(algo, 'gray'), linewidth=2.5, markersize=10)
    
    ax.set_xlabel('Number of Items', fontsize=12)
    ax.set_ylabel('Average Rank (lower = better)', fontsize=12)
    ax.set_title('Knapsack: Algorithm Ranking by Problem Size\n(Based on Mean Optimality Gap)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(sizes)
    ax.set_yticks(range(1, len(algos) + 1))
    ax.invert_yaxis()  # Best rank (1) at top
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'knapsack_ranking_by_size.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved: {output_path}")


def generate_knapsack_visualizations(
    summaries: Dict[str, pd.DataFrame],
    ranks_df: Optional[pd.DataFrame],
    output_dir: str
):
    """
    Master function to generate all Knapsack visualizations.
    Intelligently selects appropriate summary level for each plot type.
    
    Parameters
    ----------
    summaries : Dict[str, pd.DataFrame]
        Dictionary with keys 'by_instance' and/or 'by_type'
    ranks_df : pd.DataFrame, optional
        Global ranks data
    output_dir : str
        Output directory for plots
    """
    logger.info("--- Generating Knapsack Visualizations ---")
    
    df_by_type = summaries.get('by_type')
    df_by_instance = summaries.get('by_instance')
    
    # ========================================================================
    # AGGREGATE PLOTS (Use by_type if available, else aggregate from by_instance)
    # ========================================================================
    
    df_for_aggregate = df_by_type if df_by_type is not None else df_by_instance
    
    if df_for_aggregate is not None:
        logger.info(f"Using '{'by_type' if df_by_type is not None else 'by_instance'}' summary for aggregate plots...")
        
        # 1. Gap comparison grid
        plot_knapsack_gap_comparison(df_for_aggregate, output_dir)
        
        # 2. Cross-scenario performance
        plot_knapsack_cross_scenario_performance(df_for_aggregate, output_dir)
        
        # 3. Feasibility heatmap
        plot_knapsack_feasibility_heatmap(df_for_aggregate, output_dir)
        
        # 4. Scenario impact
        plot_knapsack_scenario_impact(df_for_aggregate, output_dir)
        
        # 5. Algorithm ranking by size
        plot_knapsack_algorithm_ranking_by_size(df_for_aggregate, output_dir)
    else:
        logger.warning("No summary data available for aggregate plots")
    
    # ========================================================================
    # DETAILED PLOTS (REQUIRE by_instance)
    # ========================================================================
    
    if df_by_instance is not None:
        logger.info("Using 'by_instance' summary for detailed plots...")
        
        # Performance profile (MUST use instance-level data)
        if 'Mean_Gap_%' in df_by_instance.columns:
            plot_performance_profile(
                df_by_instance, 
                'Mean_Gap_%',
                str(Path(output_dir) / 'knapsack_perf_profile.png'),
                minimize=True,
                title='Knapsack Performance Profile (Gap %)'
            )
        else:
            logger.warning("'Mean_Gap_%' column not found in by_instance summary")
    else:
        logger.warning("'by_instance' summary not found. Skipping Performance Profile plot.")
    
    # ========================================================================
    # GLOBAL RANKS PLOT
    # ========================================================================
    
    if ranks_df is not None:
        plot_global_ranks(
            ranks_df,
            str(Path(output_dir) / 'knapsack_global_ranks.png'),
            title='Knapsack: Global Algorithm Ranks (by Gap %)'
        )
    
    logger.info("✓ Knapsack visualizations complete")


# ============================================================================
# MASTER GENERATION FUNCTION (UPDATED)
# ============================================================================

def generate_all_plots(
    summary_dir: str = 'benchmark/results/summaries',
    output_dir: str = 'benchmark/results/plots'
):
    """
    Generate all visualizations from CSV summaries.
    
    This is the ONLY entry point - reads CSV, not JSON.
    """
    summary_path = Path(summary_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 100)
    print("GENERATING VISUALIZATIONS FROM CSV SUMMARIES")
    print("=" * 100)
    
    # Load Rastrigin summaries
    rastrigin_df = load_summary_data(summary_path / 'rastrigin_summary.csv')
    rastrigin_ranks_df = load_summary_data(summary_path / 'rastrigin_global_ranks.csv')
    
    # Load Knapsack summaries (UPDATED: Load multiple levels)
    knapsack_summaries = load_available_summaries(summary_path, 'knapsack_summary')
    knapsack_ranks_df = load_summary_data(summary_path / 'knapsack_global_ranks.csv')
    
    # RASTRIGIN PLOTS
    if rastrigin_df is not None:
        print("\n[RASTRIGIN VISUALIZATIONS]")
        
        plot_rastrigin_boxplots_by_scenario(rastrigin_df, output_dir)
        plot_rastrigin_success_breakdown(rastrigin_df, output_dir)
        plot_rastrigin_cross_scenario_comparison(rastrigin_df, output_dir)
        plot_rastrigin_hitting_times(rastrigin_df, output_dir)
        
        # Performance profile
        if 'AUC_Median' in rastrigin_df.columns:
            plot_performance_profile(
                rastrigin_df, 'AUC_Median',
                str(output_path / 'rastrigin_perf_profile.png'),
                minimize=True,
                title='Rastrigin Performance Profile (AUC)'
            )
        
        # Global ranks
        if rastrigin_ranks_df is not None:
            plot_global_ranks(
                rastrigin_ranks_df,
                str(output_path / 'rastrigin_global_ranks.png'),
                title='Rastrigin: Global Algorithm Ranks'
            )
    
    # KNAPSACK PLOTS (UPDATED: Use smart data selection)
    if knapsack_summaries:
        print("\n[KNAPSACK VISUALIZATIONS]")
        print(f"  Available summaries: {list(knapsack_summaries.keys())}")
        generate_knapsack_visualizations(knapsack_summaries, knapsack_ranks_df, output_dir)
    else:
        logger.warning("No Knapsack summary data found")
    
    print("\n" + "=" * 100)
    print(f"All plots saved to: {output_path}")
    print("=" * 100)


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate visualizations from CSV summaries',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all plots from default location
  python benchmark/visualize.py
  
  # Custom summary and output directories
  python benchmark/visualize.py --summary-dir analysis/results --output-dir plots
        """
    )
    
    parser.add_argument(
        '--summary-dir', type=str,
        default='benchmark/results/summaries',
        help='Directory containing CSV summary files'
    )
    parser.add_argument(
        '--output-dir', type=str,
        default='benchmark/results/plots',
        help='Output directory for plots'
    )
    
    args = parser.parse_args()
    
    generate_all_plots(args.summary_dir, args.output_dir)


if __name__ == "__main__":
    main()