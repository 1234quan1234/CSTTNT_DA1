#!/usr/bin/env python3
"""
Simple demo script to showcase the optimization framework.

This script demonstrates:
1. Running FA on continuous problems
2. Running FA on TSP
3. Comparing multiple algorithms
4. Parameter sensitivity analysis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.problems.continuous.sphere import SphereProblem
from src.problems.continuous.rastrigin import RastriginProblem
from src.problems.discrete.tsp import TSPProblem
from src.swarm.fa import FireflyContinuousOptimizer, FireflyDiscreteTSPOptimizer
from src.classical.hill_climbing import HillClimbingOptimizer
from src.classical.simulated_annealing import SimulatedAnnealingOptimizer
from src.classical.genetic_algorithm import GeneticAlgorithmOptimizer


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
    best_sol, best_fit, history, _ = optimizer.run(max_iter=100)
    print(f"Initial fitness: {history[0]:.6f}")
    print(f"Final fitness:   {history[-1]:.6f}")
    print(f"Improvement:     {history[0] - history[-1]:.6f}")
    print(f"Global optimum:  0.0 at origin")


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
    print(f"Search space: {np.math.factorial(problem.num_cities):,} possible tours")
    
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
    
    # Firefly Algorithm
    print("  - Firefly Algorithm...")
    fa = FireflyContinuousOptimizer(problem, n_fireflies=20, seed=42)
    _, fa_fit, fa_hist, _ = fa.run(max_iter=max_iter)
    
    # Simulated Annealing
    print("  - Simulated Annealing...")
    sa = SimulatedAnnealingOptimizer(problem, initial_temp=100, seed=42)
    _, sa_fit, sa_hist, _ = sa.run(max_iter=max_iter)
    
    # Hill Climbing
    print("  - Hill Climbing...")
    hc = HillClimbingOptimizer(problem, num_neighbors=20, seed=42)
    _, hc_fit, hc_hist, _ = hc.run(max_iter=max_iter)
    
    # Genetic Algorithm
    print("  - Genetic Algorithm...")
    ga = GeneticAlgorithmOptimizer(problem, pop_size=20, seed=42)
    _, ga_fit, ga_hist, _ = ga.run(max_iter=max_iter)
    
    # Print results
    print("\n" + "-" * 70)
    print("RESULTS:")
    print("-" * 70)
    print(f"{'Algorithm':<25} {'Initial':<12} {'Final':<12} {'Improvement':<12}")
    print("-" * 70)
    print(f"{'Firefly Algorithm':<25} {fa_hist[0]:>11.6f} {fa_fit:>11.6f} {fa_hist[0]-fa_fit:>11.6f}")
    print(f"{'Simulated Annealing':<25} {sa_hist[0]:>11.6f} {sa_fit:>11.6f} {sa_hist[0]-sa_fit:>11.6f}")
    print(f"{'Hill Climbing':<25} {hc_hist[0]:>11.6f} {hc_fit:>11.6f} {hc_hist[0]-hc_fit:>11.6f}")
    print(f"{'Genetic Algorithm':<25} {ga_hist[0]:>11.6f} {ga_fit:>11.6f} {ga_hist[0]-ga_fit:>11.6f}")
    print("-" * 70)
    
    # Find best
    results = [
        ('Firefly Algorithm', fa_fit),
        ('Simulated Annealing', sa_fit),
        ('Hill Climbing', hc_fit),
        ('Genetic Algorithm', ga_fit)
    ]
    best_algo, best_fitness = min(results, key=lambda x: x[1])
    print(f"\n🏆 Best: {best_algo} with fitness {best_fitness:.6f}")


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
    
    for gamma in gamma_values:
        optimizer = FireflyContinuousOptimizer(
            problem=problem,
            n_fireflies=20,
            alpha=0.2,
            beta0=1.0,
            gamma=gamma,
            seed=42
        )
        _, fitness, history, _ = optimizer.run(max_iter=max_iter)
        
        # Measure convergence speed (iteration where 90% of improvement achieved)
        total_improvement = history[0] - history[-1]
        target = history[0] - 0.9 * total_improvement
        conv_iter = next((i for i, h in enumerate(history) if h <= target), max_iter)
        
        print(f"{gamma:<10.1f} {fitness:<15.6f} {conv_iter}/{max_iter} iterations")
    
    print("-" * 70)
    print("\nObservation: Optimal gamma depends on problem landscape.")
    print("For Rastrigin (multimodal), lower gamma often performs better.")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("  AI SEARCH & OPTIMIZATION FRAMEWORK - COMPREHENSIVE DEMO")
    print("=" * 70)
    
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
    print("  ✓ Comparison of 4 algorithms (FA, SA, HC, GA)")
    print("  ✓ Parameter sensitivity analysis (gamma)")
    print("\nNext steps:")
    print("  • Visualize convergence curves using history_best")
    print("  • Animate swarm movement using trajectory")
    print("  • Run statistical benchmarks over multiple seeds")
    print("  • Create notebooks for video demonstrations")
    print("\nSee QUICKSTART.md for more usage examples!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
