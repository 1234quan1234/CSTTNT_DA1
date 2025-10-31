"""
Visualization utilities for optimization algorithms.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Optional


def plot_convergence(
    history: List[float],
    title: str = "Convergence Curve",
    xlabel: str = "Iteration",
    ylabel: str = "Best Fitness",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot convergence curve for a single algorithm.
    
    Args:
        history: List of best fitness values over iterations
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: If provided, save plot to this path
        show: Whether to display the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history, linewidth=2, color='#2E86AB')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison(
    histories: Dict[str, List[float]],
    title: str = "Algorithm Comparison",
    xlabel: str = "Iteration",
    ylabel: str = "Best Fitness",
    save_path: Optional[str] = None,
    show: bool = True,
    log_scale: bool = False
):
    """
    Plot convergence curves for multiple algorithms.
    
    Args:
        histories: Dict mapping algorithm names to their fitness histories
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: If provided, save plot to this path
        show: Whether to display the plot
        log_scale: Use logarithmic scale for y-axis
    """
    plt.figure(figsize=(12, 7))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    for idx, (name, history) in enumerate(histories.items()):
        color = colors[idx % len(colors)]
        plt.plot(history, label=name, linewidth=2, color=color, alpha=0.8)
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    
    if log_scale:
        plt.yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_trajectory_2d(
    trajectory: List[np.ndarray],
    title: str = "Firefly Trajectory (2D)",
    save_path: Optional[str] = None,
    show: bool = True,
    sample_rate: int = 1
):
    """
    Plot 2D trajectory of fireflies over iterations.
    
    Args:
        trajectory: List of population positions [iteration][firefly][dimension]
        title: Plot title
        save_path: If provided, save plot to this path
        show: Whether to display the plot
        sample_rate: Plot every Nth iteration to reduce clutter
    """
    trajectory = trajectory[::sample_rate]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot trajectories
    for firefly_idx in range(trajectory[0].shape[0]):
        path = np.array([pop[firefly_idx, :2] for pop in trajectory])
        ax.plot(path[:, 0], path[:, 1], alpha=0.3, linewidth=1, color='gray')
        
        # Mark start and end
        ax.scatter(path[0, 0], path[0, 1], c='green', s=50, marker='o', 
                  alpha=0.6, edgecolors='black', linewidth=1)
        ax.scatter(path[-1, 0], path[-1, 1], c='red', s=50, marker='*', 
                  alpha=0.8, edgecolors='black', linewidth=1)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
               markersize=8, label='Start'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
               markersize=10, label='End')
    ]
    ax.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_tsp_tour(
    coords: np.ndarray,
    tour: List[int],
    title: str = "TSP Tour",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot TSP tour on 2D coordinates.
    
    Args:
        coords: City coordinates [num_cities, 2]
        tour: Tour as list of city indices
        title: Plot title
        save_path: If provided, save plot to this path
        show: Whether to display the plot
    """
    plt.figure(figsize=(10, 10))
    
    # Plot cities
    plt.scatter(coords[:, 0], coords[:, 1], c='red', s=200, 
               zorder=3, edgecolors='black', linewidth=2)
    
    # Plot tour
    for i in range(len(tour)):
        city_a = tour[i]
        city_b = tour[(i + 1) % len(tour)]
        plt.plot([coords[city_a, 0], coords[city_b, 0]],
                [coords[city_a, 1], coords[city_b, 1]],
                'b-', linewidth=2, alpha=0.6)
    
    # Label cities
    for i, (x, y) in enumerate(coords):
        plt.text(x, y, str(i), fontsize=10, ha='center', va='center',
                color='white', fontweight='bold')
    
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_parameter_sensitivity(
    param_values: List[float],
    final_fitness: List[float],
    param_name: str = "Parameter",
    title: str = "Parameter Sensitivity Analysis",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot parameter sensitivity analysis.
    
    Args:
        param_values: List of parameter values tested
        final_fitness: List of final fitness values for each parameter
        param_name: Name of the parameter being varied
        title: Plot title
        save_path: If provided, save plot to this path
        show: Whether to display the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, final_fitness, 'o-', linewidth=2, 
            markersize=8, color='#2E86AB')
    plt.xlabel(param_name, fontsize=12)
    plt.ylabel('Final Fitness', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
