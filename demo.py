#!/usr/bin/env python3
"""
Comprehensive demo script for Firefly Algorithm.

This script demonstrates:
1. Running FA on Rastrigin function (continuous multimodal problem)
2. Running FA on Knapsack problem (discrete problem)
3. Comparing FA with classical algorithms (SA, HC, GA)
4. Parameter sensitivity analysis
5. Comprehensive visualization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# Error handling for imports
try:
    from src.problems.continuous.rastrigin import RastriginProblem
    from src.problems.discrete.knapsack import KnapsackProblem
    from src.swarm.fa import FireflyContinuousOptimizer, FireflyKnapsackOptimizer
    from src.classical.hill_climbing import HillClimbingOptimizer
    from src.classical.simulated_annealing import SimulatedAnnealingOptimizer
    from src.classical.genetic_algorithm import GeneticAlgorithmOptimizer
    from src.utils.visualization import (
        plot_convergence, plot_comparison, plot_trajectory_2d,
        plot_parameter_sensitivity
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("\nMake sure all required files exist:")
    print("  - src/problems/continuous/rastrigin.py")
    print("  - src/problems/discrete/knapsack.py")
    print("  - src/swarm/fa.py")
    print("  - src/classical/*.py")
    print("  - src/utils/visualization.py")
    sys.exit(1)


def demo_fa_rastrigin():
    """Demo 1: Firefly Algorithm on Rastrigin function."""
    print("=" * 70)
    print("DEMO 1: Firefly Algorithm on Rastrigin Function")
    print("=" * 70)
    
    dim = 10  # Test dimension
    max_iter = 100
    seed = 42
    
    print(f"\nProblem: Rastrigin function (dim={dim})")
    print("Properties: Highly multimodal, many local minima")
    print("Global minimum: f(0,...,0) = 0")
    print("-" * 70)
    
    problem = RastriginProblem(dim=dim)
    
    optimizer = FireflyContinuousOptimizer(
        problem=problem,
        n_fireflies=30,
        alpha=0.3,      # Higher alpha for multimodal
        beta0=1.0,
        gamma=0.5,      # Lower gamma for global search
        seed=seed
    )
    
    best_sol, best_fit, history, stats_history = optimizer.run(max_iter=max_iter)  # UPDATED
    
    print(f"\nResults:")
    print(f"  Initial fitness: {history[0]:.6f}")
    print(f"  Final fitness:   {history[-1]:.6f}")
    print(f"  Improvement:     {history[0] - history[-1]:.6f}")
    print(f"  Improvement %:   {100 * (history[0] - history[-1]) / history[0]:.2f}%")
    print(f"  Best solution:   {best_sol[:3]}... (showing first 3 dims)")
    
    # NEW: Show diversity statistics
    if stats_history:
        initial_div = stats_history[0]['diversity']
        final_div = stats_history[-1]['diversity']
        print(f"\n  Population Diversity:")
        print(f"    Initial: {initial_div:.4f}")
        print(f"    Final:   {final_div:.4f}")
        print(f"    Loss:    {100*(initial_div-final_div)/initial_div:.1f}%")
    
    # Visualize convergence
    plot_convergence(
        history,
        title=f"FA on Rastrigin Function (dim={dim})",
        save_path="results/fa_rastrigin_convergence.png",
        show=False
    )
    
    # Visualize trajectory
    plot_trajectory_2d(
        trajectory,
        title="FA Swarm Trajectory on Rastrigin (First 2 Dims)",
        save_path="results/fa_rastrigin_trajectory.png",
        show=False,
        sample_rate=10
    )
    
    return {'problem': 'Rastrigin', 'initial': history[0], 'final': history[-1], 'history': history}


def demo_fa_knapsack():
    """Demo 2: Firefly Algorithm on Knapsack problem."""
    print("\n" + "=" * 70)
    print("DEMO 2: Firefly Algorithm on 0/1 Knapsack Problem")
    print("=" * 70)
    
    seed = 42
    rng = np.random.RandomState(seed)
    
    # Create Knapsack instance
    n_items = 30
    weights = rng.randint(1, 50, n_items)
    values = rng.randint(10, 100, n_items)
    capacity = int(0.5 * np.sum(weights))
    
    problem = KnapsackProblem(values, weights, capacity)
    
    print(f"\nProblem setup:")
    print(f"  Items: {n_items}")
    print(f"  Capacity: {capacity}")
    print(f"  Total weight: {np.sum(weights)}")
    print(f"  Total value: {np.sum(values)}")
    print("-" * 70)
    
    # Test both constraint handling strategies
    for strategy in ['repair', 'penalty']:
        print(f"\n--- Strategy: {strategy.upper()} ---")
        
        optimizer = FireflyKnapsackOptimizer(
            problem=problem,
            n_fireflies=30,
            alpha_flip=0.2,
            max_flips_per_move=3,
            constraint_handling=strategy,  # Use the switch
            seed=seed
        )
        
        best_sol, best_fit, history, _ = optimizer.run(max_iter=100)
        
        total_value = -best_fit
        total_weight = np.sum(best_sol * weights)
        
        print(f"\nResults ({strategy}):")
        print(f"  Final value:   {total_value:.2f}")
        print(f"  Best weight:   {total_weight:.2f} / {capacity}")
        print(f"  Feasible:      {total_weight <= capacity}")
    
    # Visualize convergence
    plot_convergence(
        [-h for h in history],  # Negate for actual values
        title=f"FA on Knapsack Problem ({n_items} items)",
        ylabel="Total Value",
        save_path="results/fa_knapsack_convergence.png",
        show=False
    )
    
    return {'problem': 'Knapsack', 'initial': history[0], 'final': history[-1], 'history': history}


def demo_algorithm_comparison():
    """Demo 3: Compare FA with classical algorithms on Rastrigin."""
    print("\n" + "=" * 70)
    print("DEMO 3: Algorithm Comparison on Rastrigin Function")
    print("=" * 70)
    
    problem = RastriginProblem(dim=5)
    max_iter = 100
    
    print(f"\nProblem: Rastrigin function (dim=5)")
    print(f"Iterations: {max_iter}")
    print(f"Global optimum: 0.0 at origin")
    print("\nRunning algorithms...")
    
    results = []
    histories_dict = {}
    
    # Firefly Algorithm
    try:
        print("  - Firefly Algorithm...")
        fa = FireflyContinuousOptimizer(problem, n_fireflies=20, alpha=0.3, gamma=0.5, seed=42)
        _, fa_fit, fa_hist, _ = fa.run(max_iter=max_iter)
        results.append(('Firefly Algorithm', fa_fit, fa_hist))
        histories_dict['Firefly Algorithm'] = fa_hist
    except Exception as e:
        print(f"    ✗ Error: {e}")
    
    # Simulated Annealing
    try:
        print("  - Simulated Annealing...")
        sa = SimulatedAnnealingOptimizer(problem, initial_temp=100, seed=42)
        _, sa_fit, sa_hist, _ = sa.run(max_iter=max_iter)
        results.append(('Simulated Annealing', sa_fit, sa_hist))
        histories_dict['Simulated Annealing'] = sa_hist
    except Exception as e:
        print(f"    ✗ Error: {e}")
    
    # Hill Climbing
    try:
        print("  - Hill Climbing...")
        hc = HillClimbingOptimizer(problem, num_neighbors=20, seed=42)
        _, hc_fit, hc_hist, _ = hc.run(max_iter=max_iter)
        results.append(('Hill Climbing', hc_fit, hc_hist))
        histories_dict['Hill Climbing'] = hc_hist
    except Exception as e:
        print(f"    ✗ Error: {e}")
    
    # Genetic Algorithm
    try:
        print("  - Genetic Algorithm...")
        ga = GeneticAlgorithmOptimizer(problem, pop_size=20, seed=42)
        _, ga_fit, ga_hist, _ = ga.run(max_iter=max_iter)
        results.append(('Genetic Algorithm', ga_fit, ga_hist))
        histories_dict['Genetic Algorithm'] = ga_hist
    except Exception as e:
        print(f"    ✗ Error: {e}")
    
    # Print results
    if results:
        print("\n" + "-" * 70)
        print("RESULTS:")
        print("-" * 70)
        print(f"{'Algorithm':<25} {'Initial':<12} {'Final':<12} {'Improvement':<12}")
        print("-" * 70)
        for name, final_fit, history in results:
            print(f"{name:<25} {history[0]:>11.6f} {final_fit:>11.6f} {history[0]-final_fit:>11.6f}")
        print("-" * 70)
        
        # Find best
        best_algo, best_fitness, _ = min(results, key=lambda x: x[1])
        print(f"\n🏆 Best: {best_algo} with fitness {best_fitness:.6f}")
        
        # Visualize comparison
        plot_comparison(
            histories_dict,
            title="Algorithm Comparison on Rastrigin Function",
            save_path="results/algorithm_comparison.png",
            show=False
        )
        
        plot_comparison(
            histories_dict,
            title="Algorithm Comparison (Log Scale)",
            save_path="results/algorithm_comparison_log.png",
            show=False,
            log_scale=True
        )
    else:
        print("\n✗ No algorithms completed successfully")


def demo_parameter_sensitivity():
    """Demo 4: FA parameter sensitivity analysis."""
    print("\n" + "=" * 70)
    print("DEMO 4: Firefly Algorithm Parameter Sensitivity")
    print("=" * 70)
    
    problem = RastriginProblem(dim=5)
    max_iter = 100
    
    print("\nTesting different gamma values (light absorption coefficient)")
    print("Lower gamma = more global search, Higher gamma = more local search")
    print("\n" + "-" * 70)
    print(f"{'Gamma':<10} {'Final Fitness':<15} {'Convergence Speed':<20}")
    print("-" * 70)
    
    gamma_values = [0.3, 0.5, 1.0, 2.0, 5.0]
    fitness_values = []
    
    for gamma in gamma_values:
        try:
            optimizer = FireflyContinuousOptimizer(
                problem=problem,
                n_fireflies=20,
                alpha=0.2,
                beta0=1.0,
                gamma=gamma,
                seed=42
            )
            _, fitness, history, _ = optimizer.run(max_iter=max_iter)
            fitness_values.append(fitness)
            
            # Measure convergence speed (iteration where 90% of improvement achieved)
            total_improvement = history[0] - history[-1]
            if total_improvement > 0:
                target = history[0] - 0.9 * total_improvement
                conv_iter = next((i for i, h in enumerate(history) if h <= target), max_iter)
            else:
                conv_iter = max_iter
            
            print(f"{gamma:<10.1f} {fitness:<15.6f} {conv_iter}/{max_iter} iterations")
        except Exception as e:
            print(f"{gamma:<10.1f} Error: {e}")
            fitness_values.append(None)
    
    print("-" * 70)
    print("\nObservation: Optimal gamma depends on problem landscape.")
    print("For Rastrigin (multimodal), lower gamma often performs better.")
    
    # Visualize parameter sensitivity
    valid_gammas = [g for g, f in zip(gamma_values, fitness_values) if f is not None]
    valid_fitness = [f for f in fitness_values if f is not None]
    
    if valid_gammas:
        plot_parameter_sensitivity(
            valid_gammas,
            valid_fitness,
            param_name="Gamma (Light Absorption)",
            title="FA Parameter Sensitivity: Gamma on Rastrigin",
            save_path="results/parameter_sensitivity_gamma.png",
            show=False
        )


def main():
    """Run all comprehensive demos."""
    print("\n" + "=" * 70)
    print("  FIREFLY ALGORITHM - COMPREHENSIVE TESTING")
    print("=" * 70)
    
    os.makedirs("results", exist_ok=True)
    
    try:
        rastrigin_result = demo_fa_rastrigin()
        knapsack_result = demo_fa_knapsack()
        demo_algorithm_comparison()
        demo_parameter_sensitivity()
        
        print("\n" + "=" * 70)
        print("COMPREHENSIVE DEMO COMPLETE")
        print("=" * 70)
        print("\nWhat you've seen:")
        print("  ✓ FA on Rastrigin function (continuous multimodal)")
        print("  ✓ FA on 0/1 Knapsack problem (discrete with repair/penalty strategies)")
        print("  ✓ Algorithm comparison (FA, SA, HC, GA)")
        print("  ✓ Parameter sensitivity analysis")
        print("\nVisualization files saved to results/:")
        print("  • fa_rastrigin_convergence.png")
        print("  • fa_rastrigin_trajectory.png")
        print("  • fa_knapsack_convergence.png")
        print("  • algorithm_comparison.png")
        print("  • algorithm_comparison_log.png")
        print("  • parameter_sensitivity_gamma.png")
        print("\n📊 Summary Statistics:")
        print(f"  • Rastrigin: {rastrigin_result['initial']:.6f} → {rastrigin_result['final']:.6f}")
        print(f"  • Knapsack: {-knapsack_result['initial']:.2f} → {-knapsack_result['final']:.2f} (value)")
        print(f"  • Total visualizations: 6 plots")
        print("\n✨ Next steps:")
        print("  • Run full benchmarks: python benchmark/run_all.py --quick")
        print("  • Analyze results: python benchmark/analyze_results.py --problem all")
        print("  • Generate plots: python benchmark/visualize.py")
        print("  • Explore notebooks: jupyter notebook notebooks/fa_visualization.ipynb")
        print("\nSee README.md and benchmark/README.md for more details!")
        print("=" * 70 + "\n")
    
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
