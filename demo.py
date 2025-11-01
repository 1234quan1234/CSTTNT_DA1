#!/usr/bin/env python3
"""
Comprehensive demo script for Firefly Algorithm on all problems.

This script demonstrates:
1. Running FA on ALL continuous problems (Sphere, Rosenbrock, Rastrigin, Ackley)
2. Running FA on ALL discrete problems (TSP, Knapsack, Graph Coloring)
3. Comparing FA with classical algorithms
4. Parameter sensitivity analysis
5. Comprehensive visualization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import math

# Error handling for imports
try:
    from src.problems.continuous.sphere import SphereProblem
    from src.problems.continuous.rosenbrock import RosenbrockProblem
    from src.problems.continuous.rastrigin import RastriginProblem
    from src.problems.continuous.ackley import AckleyProblem
    from src.problems.discrete.tsp import TSPProblem
    from src.problems.discrete.knapsack import KnapsackProblem
    from src.problems.discrete.graph_coloring import GraphColoringProblem
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


def demo_fa_all_continuous():
    """Demo 1: Firefly Algorithm on ALL continuous problems."""
    print("=" * 70)
    print("DEMO 1: Firefly Algorithm on ALL Continuous Problems")
    print("=" * 70)
    
    from src.problems.continuous.sphere import SphereProblem
    from src.problems.continuous.rosenbrock import RosenbrockProblem
    from src.problems.continuous.rastrigin import RastriginProblem
    from src.problems.continuous.ackley import AckleyProblem
    
    dim = 10  # Test dimension
    max_iter = 100
    seed = 42
    
    problems = [
        ("Sphere", SphereProblem(dim=dim), {"n_fireflies": 20, "alpha": 0.2, "gamma": 1.0}),
        ("Rosenbrock", RosenbrockProblem(dim=dim), {"n_fireflies": 30, "alpha": 0.25, "gamma": 0.8}),
        ("Rastrigin", RastriginProblem(dim=dim), {"n_fireflies": 30, "alpha": 0.3, "gamma": 0.5}),
        ("Ackley", AckleyProblem(dim=dim), {"n_fireflies": 25, "alpha": 0.3, "gamma": 0.6}),
    ]
    
    results = []
    
    for name, problem, params in problems:
        print(f"\n[{name}]")
        print("-" * 70)
        
        optimizer = FireflyContinuousOptimizer(
            problem=problem,
            n_fireflies=params["n_fireflies"],
            alpha=params["alpha"],
            beta0=1.0,
            gamma=params["gamma"],
            seed=seed
        )
        
        best_sol, best_fit, history, trajectory = optimizer.run(max_iter=max_iter)
        
        print(f"  Initial fitness: {history[0]:.6f}")
        print(f"  Final fitness:   {history[-1]:.6f}")
        print(f"  Improvement:     {history[0] - history[-1]:.6f}")
        print(f"  Best solution:   {best_sol[:3]}... (showing first 3 dims)")
        
        results.append({
            'Problem': name,
            'Initial': history[0],
            'Final': history[-1],
            'Improvement': history[0] - history[-1],
            'History': history
        })
        
        # Visualize convergence
        plot_convergence(
            history,
            title=f"FA on {name} Function (dim={dim})",
            save_path=f"results/fa_{name.lower()}_convergence.png",
            show=False
        )
        
        # Visualize trajectory for 2D projection
        if len(trajectory) > 0 and name in ["Rastrigin", "Ackley"]:
            plot_trajectory_2d(
                trajectory,
                title=f"FA Swarm Trajectory on {name} (First 2 Dims)",
                save_path=f"results/fa_{name.lower()}_trajectory.png",
                show=False,
                sample_rate=10
            )
    
    # Print summary table
    print("\n" + "=" * 70)
    print("CONTINUOUS PROBLEMS SUMMARY")
    print("=" * 70)
    print(f"{'Problem':<15} {'Initial':<12} {'Final':<12} {'Improvement':<12} {'Improv %':<10}")
    print("-" * 70)
    for r in results:
        improv_pct = 100 * r['Improvement'] / r['Initial'] if r['Initial'] > 0 else 0
        print(f"{r['Problem']:<15} {r['Initial']:>11.6f} {r['Final']:>11.6f} {r['Improvement']:>11.6f} {improv_pct:>9.2f}%")
    print("-" * 70)
    
    # Plot all convergences together
    histories_dict = {r['Problem']: r['History'] for r in results}
    plot_comparison(
        histories_dict,
        title="FA Convergence on All Continuous Problems",
        save_path="results/fa_all_continuous_comparison.png",
        show=False
    )
    
    return results


def demo_fa_all_discrete():
    """Demo 2: Firefly Algorithm on ALL discrete problems."""
    print("\n" + "=" * 70)
    print("DEMO 2: Firefly Algorithm on ALL Discrete Problems")
    print("=" * 70)
    
    from src.problems.discrete.tsp import TSPProblem
    from src.problems.discrete.knapsack import KnapsackProblem
    from src.problems.discrete.graph_coloring import GraphColoringProblem
    from src.swarm.fa import FireflyDiscreteTSPOptimizer
    
    results = []
    seed = 42
    rng = np.random.RandomState(seed)
    
    # 1. TSP Problem
    print("\n[1] Traveling Salesman Problem")
    print("-" * 70)
    coords = rng.rand(20, 2) * 100  # 20 cities
    tsp_problem = TSPProblem(coords)
    
    print(f"  Cities: {tsp_problem.num_cities}")
    print(f"  Search space: {math.factorial(tsp_problem.num_cities):,} possible tours")
    
    tsp_optimizer = FireflyDiscreteTSPOptimizer(
        problem=tsp_problem,
        n_fireflies=30,
        alpha_swap=0.2,
        max_swaps_per_move=3,
        seed=seed
    )
    
    best_tour, best_length, tsp_history, _ = tsp_optimizer.run(max_iter=150)
    
    print(f"  Initial tour length: {tsp_history[0]:.4f}")
    print(f"  Final tour length:   {tsp_history[-1]:.4f}")
    print(f"  Improvement:         {tsp_history[0] - tsp_history[-1]:.4f}")
    print(f"  Best tour: {best_tour}")
    
    results.append({
        'Problem': 'TSP',
        'Size': f"{tsp_problem.num_cities} cities",
        'Initial': tsp_history[0],
        'Final': tsp_history[-1],
        'Improvement': tsp_history[0] - tsp_history[-1],
        'History': tsp_history
    })
    
    plot_tsp_tour(
        coords,
        best_tour,
        title=f"Best TSP Tour (Length: {best_length:.2f})",
        save_path="results/fa_tsp_tour.png",
        show=False
    )
    
    plot_convergence(
        tsp_history,
        title="FA on TSP - Convergence",
        ylabel="Tour Length",
        save_path="results/fa_tsp_convergence.png",
        show=False
    )
    
    # 2. Knapsack Problem
    print("\n[2] 0/1 Knapsack Problem")
    print("-" * 70)
    n_items = 30
    weights = rng.randint(1, 50, n_items)
    values = rng.randint(10, 100, n_items)
    capacity = int(0.5 * np.sum(weights))
    
    knapsack_problem = KnapsackProblem(weights, values, capacity)
    
    print(f"  Items: {n_items}")
    print(f"  Capacity: {capacity}")
    print(f"  Total weight: {np.sum(weights)}")
    print(f"  Total value: {np.sum(values)}")
    
    # Use Simulated Annealing for Knapsack (FA doesn't have specific knapsack variant)
    from src.classical.simulated_annealing import SimulatedAnnealingOptimizer
    from src.classical.genetic_algorithm import GeneticAlgorithmOptimizer
    
    print("  Note: Using SA and GA for Knapsack (no dedicated FA variant)")
    
    sa_knapsack = SimulatedAnnealingOptimizer(knapsack_problem, initial_temp=100, seed=seed)
    _, sa_value, sa_history, _ = sa_knapsack.run(max_iter=200)
    
    ga_knapsack = GeneticAlgorithmOptimizer(knapsack_problem, pop_size=30, seed=seed)
    _, ga_value, ga_history, _ = ga_knapsack.run(max_iter=100)
    
    # Note: Knapsack is maximization, so we negate for comparison
    print(f"  SA Final value: {-sa_value:.2f}")
    print(f"  GA Final value: {-ga_value:.2f}")
    
    results.append({
        'Problem': 'Knapsack (SA)',
        'Size': f"{n_items} items",
        'Initial': sa_history[0],
        'Final': sa_history[-1],
        'Improvement': sa_history[0] - sa_history[-1],
        'History': sa_history
    })
    
    plot_convergence(
        sa_history,
        title=f"SA on Knapsack Problem ({n_items} items)",
        ylabel="Negative Value (minimize)",
        save_path="results/sa_knapsack_convergence.png",
        show=False
    )
    
    # 3. Graph Coloring Problem
    print("\n[3] Graph Coloring Problem")
    print("-" * 70)
    n_nodes = 20
    edge_prob = 0.3
    num_colors = 5  # Allow up to 5 colors
    
    # Generate random graph
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.rand() < edge_prob:
                edges.append((i, j))
    
    coloring_problem = GraphColoringProblem(n_nodes, edges, num_colors)
    
    print(f"  Nodes: {n_nodes}")
    print(f"  Edges: {len(edges)}")
    print(f"  Colors available: {num_colors}")
    print(f"  Density: {len(edges) / (n_nodes * (n_nodes - 1) / 2):.2%}")
    
    print("  Note: Using SA for Graph Coloring (no dedicated FA variant)")
    
    sa_coloring = SimulatedAnnealingOptimizer(coloring_problem, initial_temp=50, seed=seed)
    _, sa_conflicts, sa_col_history, _ = sa_coloring.run(max_iter=200)
    
    print(f"  Final conflicts: {sa_conflicts:.0f}")
    print(f"  Improvement: {sa_col_history[0] - sa_col_history[-1]:.0f}")
    
    results.append({
        'Problem': 'Graph Coloring (SA)',
        'Size': f"{n_nodes} nodes, {len(edges)} edges",
        'Initial': sa_col_history[0],
        'Final': sa_col_history[-1],
        'Improvement': sa_col_history[0] - sa_col_history[-1],
        'History': sa_col_history
    })
    
    plot_convergence(
        sa_col_history,
        title=f"SA on Graph Coloring ({n_nodes} nodes)",
        ylabel="Number of Conflicts",
        save_path="results/sa_coloring_convergence.png",
        show=False
    )
    
    # Print summary table
    print("\n" + "=" * 70)
    print("DISCRETE PROBLEMS SUMMARY")
    print("=" * 70)
    print(f"{'Problem':<25} {'Size':<20} {'Initial':<12} {'Final':<12} {'Improvement':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r['Problem']:<25} {r['Size']:<20} {r['Initial']:>11.4f} {r['Final']:>11.4f} {r['Improvement']:>11.4f}")
    print("-" * 70)
    
    return results


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
    """Run all comprehensive demos."""
    print("\n" + "=" * 70)
    print("  FIREFLY ALGORITHM - COMPREHENSIVE TESTING ON ALL PROBLEMS")
    print("=" * 70)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    try:
        # Run comprehensive demos
        continuous_results = demo_fa_all_continuous()
        discrete_results = demo_fa_all_discrete()
        demo_algorithm_comparison()
        demo_parameter_sensitivity()
        
        # Final summary
        print("\n" + "=" * 70)
        print("COMPREHENSIVE DEMO COMPLETE")
        print("=" * 70)
        print("\nWhat you've seen:")
        print("  ✓ FA on ALL continuous problems:")
        print("    • Sphere (unimodal, convex)")
        print("    • Rosenbrock (unimodal, narrow valley)")
        print("    • Rastrigin (multimodal, many local minima)")
        print("    • Ackley (multimodal, nearly flat outer region)")
        print("  ✓ FA/SA/GA on ALL discrete problems:")
        print("    • Traveling Salesman Problem (TSP)")
        print("    • 0/1 Knapsack Problem")
        print("    • Graph Coloring Problem")
        print("  ✓ Algorithm comparison (FA, SA, HC, GA)")
        print("  ✓ Parameter sensitivity analysis")
        print("\nVisualization files saved to results/:")
        print("  Continuous problems:")
        print("    • fa_sphere_convergence.png")
        print("    • fa_rosenbrock_convergence.png")
        print("    • fa_rastrigin_convergence.png & trajectory.png")
        print("    • fa_ackley_convergence.png & trajectory.png")
        print("    • fa_all_continuous_comparison.png")
        print("  Discrete problems:")
        print("    • fa_tsp_tour.png & convergence.png")
        print("    • sa_knapsack_convergence.png")
        print("    • sa_coloring_convergence.png")
        print("  Comparisons:")
        print("    • algorithm_comparison.png & log.png")
        print("    • parameter_sensitivity_gamma.png")
        print("\n📊 Summary Statistics:")
        print(f"  • Continuous problems tested: {len(continuous_results)}")
        print(f"  • Discrete problems tested: {len(discrete_results)}")
        print(f"  • Total visualizations: ~15+ plots")
        print("\nNext steps:")
        print("  • Run notebooks/fa_visualization.ipynb for interactive demos")
        print("  • Customize parameters in each demo function")
        print("  • Add statistical analysis with multiple seeds")
        print("  • Create animations for swarm movement")
        print("\nSee README.md and QUICKSTART.md for more details!")
        print("=" * 70 + "\n")
    
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
