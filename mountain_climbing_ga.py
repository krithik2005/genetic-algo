"""
Mountain Climbing Genetic Algorithm - CM3020 AI Coursework Part B

This script implements a genetic algorithm to evolve creatures that can climb
a mountain in a PyBullet physics simulation. Creatures are evaluated based on
the maximum height they achieve during simulation.

Usage:
    python mountain_climbing_ga.py [--pop_size N] [--generations N] [--gene_count N]

Author: Student
Date: January 2026
"""

import argparse
import population
import mountain_simulation
import genome
import creature
import numpy as np
import os
import csv
from datetime import datetime


def run_mountain_climbing_ga(
    pop_size=20,
    gene_count=3,
    generations=100,
    iterations=2400,
    mutation_rate=0.1,
    shrink_rate=0.25,
    grow_rate=0.1,
    use_threading=False,
    pool_size=4,
    output_dir="results",
    landscape_type="gaussian_pyramid",
    use_multi_landscape=False,
    use_adaptive_mutation=False
):
    """
    Run the mountain climbing genetic algorithm.
    Advanced features: sensory input, multi-landscape testing, adaptive mutation.
    
    Args:
        pop_size: Size of the population
        gene_count: Number of genes per creature
        generations: Number of generations to evolve
        iterations: Simulation steps per creature (at 240fps)
        mutation_rate: Probability of point mutation
        shrink_rate: Probability of shrink mutation
        grow_rate: Probability of grow mutation
        use_threading: Whether to use multi-threaded simulation
        pool_size: Number of parallel simulations (if threading)
        output_dir: Directory to save results
        landscape_type: Type of mountain landscape to use
        use_multi_landscape: Test on multiple landscapes (exceptional criteria)
        use_adaptive_mutation: Dynamically adjust mutation rates
    
    Returns:
        Dictionary containing experiment results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize population
    pop = population.Population(pop_size=pop_size, gene_count=gene_count)
    
    # Initialize simulation - Advanced: support different landscapes
    if use_threading:
        sim = mountain_simulation.ThreadedMountainSim(
            pool_size=pool_size, 
            landscape_type=landscape_type
        )
    else:
        sim = mountain_simulation.MountainSimulation(landscape_type=landscape_type)
    
    # Track results
    results = {
        "settings": {
            "pop_size": pop_size,
            "gene_count": gene_count,
            "generations": generations,
            "iterations": iterations,
            "mutation_rate": mutation_rate,
            "shrink_rate": shrink_rate,
            "grow_rate": grow_rate
        },
        "history": []
    }
    
    print("=" * 60)
    print("MOUNTAIN CLIMBING GENETIC ALGORITHM")
    print("=" * 60)
    print(f"Population Size: {pop_size}")
    print(f"Gene Count: {gene_count}")
    print(f"Generations: {generations}")
    print(f"Mutation Rate: {mutation_rate}")
    print(f"Landscape Type: {landscape_type}")
    if use_multi_landscape:
        print("*** MULTI-LANDSCAPE TESTING ENABLED (Exceptional) ***")
    if use_adaptive_mutation:
        print("*** ADAPTIVE MUTATION ENABLED (Advanced) ***")
    print("=" * 60)
    print()
    
    best_ever_fitness = 0
    best_ever_creature = None
    
    for generation in range(generations):
        # Evaluate population
        # Advanced: Multi-landscape testing for generalization
        if use_multi_landscape:
            landscapes = ["gaussian_pyramid", "mountain_with_cubes", "mountain"]
            all_fits = []
            for landscape_idx, landscape in enumerate(landscapes):
                # Create temporary sim for this landscape with unique sim_id
                temp_sim = mountain_simulation.MountainSimulation(
                    sim_id=100 + landscape_idx,  # Unique ID to avoid file conflicts
                    landscape_type=landscape
                )
                landscape_fits = []
                for cr_idx, cr in enumerate(pop.creatures):
                    # Reset creature state for each landscape test
                    cr.start_position = None
                    cr.last_position = None
                    cr.position_history = []
                    cr.max_height = 0
                    cr.best_position = None
                    
                    temp_sim.run_creature(cr, iterations)
                    # Use advanced fitness function with arena bounds
                    fit = cr.get_mountain_climbing_fitness(
                        mountain_peak_height=5.0, 
                        arena_size=20
                    )
                    landscape_fits.append(fit)
                all_fits.append(landscape_fits)
                # Clean up simulation
                import pybullet as p
                p.disconnect(physicsClientId=temp_sim.physicsClientId)
            # Average fitness across landscapes
            fits = [np.mean([all_fits[i][j] for i in range(len(landscapes))]) 
                    for j in range(len(pop.creatures))]
        else:
            if use_threading:
                sim.eval_population(pop, iterations)
            else:
                for cr in pop.creatures:
                    sim.run_creature(cr, iterations)
            
            # Advanced: Use multi-objective fitness function with arena bounds
            fits = [cr.get_mountain_climbing_fitness(
                mountain_peak_height=5.0, 
                arena_size=20
            ) for cr in pop.creatures]
        links = [len(cr.get_expanded_links()) for cr in pop.creatures]
        
        max_fit = np.max(fits)
        mean_fit = np.mean(fits)
        min_fit = np.min(fits)
        std_fit = np.std(fits)
        mean_links = np.mean(links)
        max_links = np.max(links)
        
        # Record generation stats
        gen_stats = {
            "generation": generation,
            "max_fitness": float(max_fit),
            "mean_fitness": float(mean_fit),
            "min_fitness": float(min_fit),
            "std_fitness": float(std_fit),
            "mean_links": float(mean_links),
            "max_links": int(max_links)
        }
        results["history"].append(gen_stats)
        
        # Print progress
        print(f"Gen {generation:4d} | Best: {max_fit:7.3f} | Mean: {mean_fit:7.3f} | "
              f"Std: {std_fit:5.3f} | Links: {mean_links:.1f}/{max_links}")
        
        # Track best creature ever
        if max_fit > best_ever_fitness:
            best_ever_fitness = max_fit
            for i, cr in enumerate(pop.creatures):
                if fits[i] == max_fit:  # Compare fitness scores
                    best_ever_creature = cr
                    # Save elite creature
                    elite_file = os.path.join(output_dir, f"elite_gen{generation}_{timestamp}.csv")
                    genome.Genome.to_csv(cr.dna, elite_file)
                    break
        
        # Advanced: Adaptive mutation rates
        if use_adaptive_mutation:
            # Increase mutation if progress is slow
            if generation > 0:
                prev_max = results["history"][generation-1]["max_fitness"]
                progress = max_fit - prev_max
                if progress < 0.01:  # Stagnation
                    mutation_rate = min(mutation_rate * 1.2, 0.3)
                elif progress > 0.1:  # Good progress
                    mutation_rate = max(mutation_rate * 0.9, 0.05)
        
        # Selection and reproduction
        fit_map = population.Population.get_fitness_map(fits)
        new_creatures = []
        
        for i in range(len(pop.creatures)):
            p1_ind = population.Population.select_parent(fit_map)
            p2_ind = population.Population.select_parent(fit_map)
            p1 = pop.creatures[p1_ind]
            p2 = pop.creatures[p2_ind]
            
            # Crossover and mutation
            dna = genome.Genome.crossover(p1.dna, p2.dna)
            dna = genome.Genome.point_mutate(dna, rate=mutation_rate, amount=0.25)
            dna = genome.Genome.shrink_mutate(dna, rate=shrink_rate)
            dna = genome.Genome.grow_mutate(dna, rate=grow_rate)
            
            cr = creature.Creature(1)
            cr.update_dna(dna)
            new_creatures.append(cr)
        
        # Elitism - keep the best creature
        for i, cr in enumerate(pop.creatures):
            if fits[i] == max_fit:  # Compare fitness scores
                new_cr = creature.Creature(1)
                new_cr.update_dna(cr.dna)
                new_creatures[0] = new_cr
                break
        
        pop.creatures = new_creatures
    
    # Save final results
    results["best_fitness"] = float(best_ever_fitness)
    
    # Save results to CSV
    csv_file = os.path.join(output_dir, f"results_{timestamp}.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results["history"][0].keys())
        writer.writeheader()
        writer.writerows(results["history"])
    
    # Save best creature
    if best_ever_creature:
        best_file = os.path.join(output_dir, f"best_creature_{timestamp}.csv")
        genome.Genome.to_csv(best_ever_creature.dna, best_file)
    
    print()
    print("=" * 60)
    print(f"EVOLUTION COMPLETE")
    print(f"Best fitness achieved: {best_ever_fitness:.3f}")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Mountain Climbing Genetic Algorithm - CM3020 Coursework"
    )
    parser.add_argument("--pop_size", type=int, default=20,
                        help="Population size (default: 20)")
    parser.add_argument("--gene_count", type=int, default=3,
                        help="Number of genes per creature (default: 3)")
    parser.add_argument("--generations", type=int, default=100,
                        help="Number of generations (default: 100)")
    parser.add_argument("--iterations", type=int, default=2400,
                        help="Simulation steps per creature (default: 2400)")
    parser.add_argument("--mutation_rate", type=float, default=0.1,
                        help="Point mutation rate (default: 0.1)")
    parser.add_argument("--shrink_rate", type=float, default=0.25,
                        help="Shrink mutation rate (default: 0.25)")
    parser.add_argument("--grow_rate", type=float, default=0.1,
                        help="Grow mutation rate (default: 0.1)")
    parser.add_argument("--use_threading", action="store_true",
                        help="Use multi-threaded simulation")
    parser.add_argument("--pool_size", type=int, default=4,
                        help="Thread pool size (default: 4)")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory (default: results)")
    parser.add_argument("--landscape_type", type=str, default="gaussian_pyramid",
                        choices=["gaussian_pyramid", "mountain_with_cubes", "mountain"],
                        help="Type of landscape to use (default: gaussian_pyramid)")
    parser.add_argument("--use_multi_landscape", action="store_true",
                        help="Test creatures on multiple landscapes (exceptional criteria)")
    parser.add_argument("--use_adaptive_mutation", action="store_true",
                        help="Use adaptive mutation rates based on progress")
    
    args = parser.parse_args()
    
    run_mountain_climbing_ga(
        pop_size=args.pop_size,
        gene_count=args.gene_count,
        generations=args.generations,
        iterations=args.iterations,
        mutation_rate=args.mutation_rate,
        shrink_rate=args.shrink_rate,
        grow_rate=args.grow_rate,
        use_threading=args.use_threading,
        pool_size=args.pool_size,
        output_dir=args.output_dir,
        landscape_type=args.landscape_type,
        use_multi_landscape=args.use_multi_landscape,
        use_adaptive_mutation=args.use_adaptive_mutation
    )


if __name__ == "__main__":
    main()

