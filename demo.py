#!/usr/bin/env python3
"""
Simple demo script to showcase the optimization framework.

This script demonstrates:
1. Running FA on continuous problems
2. Running FA on TSP
3. Comparing multiple algorithms
4. Parameter sensitivity analysis
5. Visualizing results
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import math

# Error handling for imports
try:
    from src.problems.continuous.sphere import SphereProblem
    from src.problems.continuous.rastrigin import RastriginProblem
    from src.problems.discrete.tsp import TSPProblem
    from src.swarm.fa import FireflyContinuousOptimizer, FireflyDiscreteTSPOptimizer
    from src.classical.hill_climbing import HillClimbingOptimizer
    from src.classical.simulated_annealing import SimulatedAnnealingOptimizer
    from src.classical.genetic_algorithm import GeneticAlgorithmOptimizer
    from src.utils.visualization import (
        plot_convergence, plot_comparison, plot_trajectory_2d,
        plot_tsp_tour, plot_parameter_sensitivity
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("\nMake sure all required files exist:")
    print("  - src/problems/continuous/*.py")
    print("  - src/problems/discrete/*.py")
    print("  - src/swarm/fa.py")
    print("  - src/classical/*.py")
    print("  - src/utils/visualization.py")
    sys.exit(1)


def demo_fa_continuous():
    """Demo 1: Firefly Algorithm on continuous problems."""
    print("=" * 70)
    print("DEMO 1: Firefly Algorithm on Continuous Problems")
    print("=" * 70)
    
    # Sphere function (easy, unimodal)
    print("\n[1.1] Sphere Function (dim=5)")
    print("-" * 70)
    problem = SphereProblem(dim=5)
    optimizer = FireflyContinuousOptimizer(
        problem=problem,
        n_fireflies=20,
        alpha=0.2,
        beta0=1.0,
        gamma=1.0,
        seed=42
    )
    best_sol, best_fit, history, _ = optimizer.run(max_iter=50)
    print(f"Initial fitness: {history[0]:.6f}")
    print(f"Final fitness:   {history[-1]:.6f}")
    print(f"Improvement:     {history[0] - history[-1]:.6f}")
    print(f"Best solution:   {best_sol}")
    
    # Visualize convergence
    plot_convergence(
        history,
        title="FA on Sphere Function (dim=5)",
        save_path="results/fa_sphere_convergence.png",
        show=False
    )
    
    # Rastrigin function (hard, multimodal)
    print("\n[1.2] Rastrigin Function (dim=5)")
    print("-" * 70)
    problem = RastriginProblem(dim=5)
    optimizer = FireflyContinuousOptimizer(
        problem=problem,
        n_fireflies=30,
        alpha=0.3,  # Higher for multimodal
        beta0=1.0,
        gamma=0.5,  # Lower for more global search
        seed=42
    )
    best_sol, best_fit, history, trajectory = optimizer.run(max_iter=100)
    print(f"Initial fitness: {history[0]:.6f}")
    print(f"Final fitness:   {history[-1]:.6f}")
    print(f"Improvement:     {history[0] - history[-1]:.6f}")
    print(f"Global optimum:  0.0 at origin")
    
    # Visualize convergence and trajectory
    plot_convergence(
        history,
        title="FA on Rastrigin Function (dim=5)",
        save_path="results/fa_rastrigin_convergence.png",
        show=False
    )
    
    if len(trajectory) > 0:
        plot_trajectory_2d(
            trajectory,
            title="Firefly Swarm Trajectory on Rastrigin (First 2 Dims)",
            save_path="results/fa_rastrigin_trajectory.png",
            show=False,
            sample_rate=5
        )


def demo_fa_discrete():
    """Demo 2: Firefly Algorithm on TSP."""
    print("\n" + "=" * 70)
    print("DEMO 2: Discrete Firefly Algorithm on TSP")
    print("=" * 70)
    
    # Create random TSP instance
    rng = np.random.RandomState(123)
    coords = rng.rand(15, 2) * 100  # 15 cities in [0,100]^2
    problem = TSPProblem(coords)
    
    print(f"\nTSP Instance: {problem.num_cities} cities")
    print(f"Search space: {math.factorial(problem.num_cities):,} possible tours")
    
    optimizer = FireflyDiscreteTSPOptimizer(
        problem=problem,
        n_fireflies=25,
        alpha_swap=0.2,
        max_swaps_per_move=3,
        seed=42
    )
    
    best_tour, best_length, history, _ = optimizer.run(max_iter=100)
    
    print(f"\nInitial best tour length: {history[0]:.4f}")
    print(f"Final best tour length:   {history[-1]:.4f}")
    print(f"Improvement:              {history[0] - history[-1]:.4f}")
    print(f"Improvement %:            {100 * (history[0] - history[-1]) / history[0]:.2f}%")
    print(f"Best tour: {best_tour}")
    
    # Visualize TSP tour and convergence
    plot_tsp_tour(
        coords,
        best_tour,
        title=f"Best TSP Tour (Length: {best_length:.2f})",
        save_path="results/tsp_tour.png",
        show=False
    )
    
    plot_convergence(
        history,
        title="FA on TSP - Convergence",
        ylabel="Tour Length",
        save_path="results/tsp_convergence.png",
        show=False
    )


def demo_algorithm_comparison():
    """Demo 3: Compare multiple algorithms on same problem."""
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
        fa = FireflyContinuousOptimizer(problem, n_fireflies=20, seed=42)
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
    """Run all demos."""
    print("\n" + "=" * 70)
    print("  AI SEARCH & OPTIMIZATION FRAMEWORK - COMPREHENSIVE DEMO")
    print("=" * 70)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    try:
        # Run demos
        demo_fa_continuous()
        demo_fa_discrete()
        demo_algorithm_comparison()
        demo_parameter_sensitivity()
        
        # Final summary
        print("\n" + "=" * 70)
        print("DEMO COMPLETE")
        print("=" * 70)
        print("\nWhat you've seen:")
        print("  ✓ Firefly Algorithm on continuous problems (Sphere, Rastrigin)")
        print("  ✓ Discrete FA on Traveling Salesman Problem")
        print("  ✓ Comparison of multiple algorithms (FA, SA, HC, GA)")
        print("  ✓ Parameter sensitivity analysis (gamma)")
        print("  ✓ Visualization of convergence, trajectories, and comparisons")
        print("\nVisualization files saved to:")
        print("  • results/fa_sphere_convergence.png")
        print("  • results/fa_rastrigin_convergence.png")
        print("  • results/fa_rastrigin_trajectory.png")
        print("  • results/tsp_tour.png")
        print("  • results/tsp_convergence.png")
        print("  • results/algorithm_comparison.png")
        print("  • results/algorithm_comparison_log.png")
        print("  • results/parameter_sensitivity_gamma.png")
        print("\nNext steps:")
        print("  • Customize visualizations in src/utils/visualization.py")
        print("  • Create animations using matplotlib.animation")
        print("  • Run statistical benchmarks over multiple seeds")
        print("  • Create notebooks for interactive demonstrations")
        print("\nSee QUICKSTART.md for more usage examples!")
        print("=" * 70 + "\n")
    
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
