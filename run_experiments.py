"""
Experiment Runner for Mountain Climbing GA - CM3020 AI Coursework Part B

This script runs ALL experiments needed for the video:
1. BASIC: Population size, mutation rate, gene count
2. ADVANCED: Encoding scheme experiments
3. EXCEPTIONAL: Multi-landscape testing, adaptive mutation

Usage:
    python run_experiments.py

Author: Student
Date: January 2026
"""

import os
import numpy as np
import csv
import json
from datetime import datetime
from mountain_climbing_ga import run_mountain_climbing_ga

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Graphs will not be generated.")


# ============================================================
# SETTINGS - Adjust for faster/slower experiments
# ============================================================
GENERATIONS = 50  # Generations per experiment (50 for good results)
# ============================================================


def run_population_size_experiments(base_output_dir, generations):
    """
    BASIC Experiment 1: Test different population sizes.
    """
    print("\n" + "="*70)
    print("BASIC EXPERIMENT 1: Population Size Comparison")
    print("="*70)
    
    pop_sizes = [20, 50, 100]
    all_results = {}
    
    for pop_size in pop_sizes:
        print(f"\n--- Running with population size: {pop_size} ---")
        output_dir = os.path.join(base_output_dir, f"basic_pop{pop_size}")
        
        results = run_mountain_climbing_ga(
            pop_size=pop_size,
            gene_count=3,
            generations=generations,
            iterations=2400,
            mutation_rate=0.1,
            output_dir=output_dir
        )
        all_results[pop_size] = results
    
    return all_results


def run_mutation_rate_experiments(base_output_dir, generations):
    """
    BASIC Experiment 2: Test different mutation rates.
    """
    print("\n" + "="*70)
    print("BASIC EXPERIMENT 2: Mutation Rate Comparison")
    print("="*70)
    
    mutation_rates = [0.05, 0.1, 0.2, 0.3]
    all_results = {}
    
    for rate in mutation_rates:
        print(f"\n--- Running with mutation rate: {rate} ---")
        output_dir = os.path.join(base_output_dir, f"basic_mut{rate}")
        
        results = run_mountain_climbing_ga(
            pop_size=30,
            gene_count=3,
            generations=generations,
            iterations=2400,
            mutation_rate=rate,
            output_dir=output_dir
        )
        all_results[rate] = results
    
    return all_results


def run_gene_count_experiments(base_output_dir, generations):
    """
    BASIC Experiment 3: Test different gene counts (creature complexity).
    """
    print("\n" + "="*70)
    print("BASIC EXPERIMENT 3: Gene Count (Creature Complexity)")
    print("="*70)
    
    gene_counts = [2, 3, 5, 7]
    all_results = {}
    
    for gene_count in gene_counts:
        print(f"\n--- Running with gene count: {gene_count} ---")
        output_dir = os.path.join(base_output_dir, f"basic_genes{gene_count}")
        
        results = run_mountain_climbing_ga(
            pop_size=30,
            gene_count=gene_count,
            generations=generations,
            iterations=2400,
            mutation_rate=0.1,
            output_dir=output_dir
        )
        all_results[gene_count] = results
    
    return all_results


def run_landscape_experiments(base_output_dir, generations):
    """
    EXCEPTIONAL Experiment: Test different landscape types.
    """
    print("\n" + "="*70)
    print("EXCEPTIONAL EXPERIMENT: Different Landscapes")
    print("="*70)
    
    landscapes = ["gaussian_pyramid", "mountain_with_cubes", "mountain"]
    all_results = {}
    
    for landscape in landscapes:
        print(f"\n--- Running with landscape: {landscape} ---")
        output_dir = os.path.join(base_output_dir, f"exceptional_landscape_{landscape}")
        
        results = run_mountain_climbing_ga(
            pop_size=30,
            gene_count=3,
            generations=generations,
            iterations=2400,
            mutation_rate=0.1,
            landscape_type=landscape,
            output_dir=output_dir
        )
        all_results[landscape] = results
    
    return all_results


def run_multi_landscape_experiment(base_output_dir, generations):
    """
    EXCEPTIONAL Experiment: Multi-landscape generalization.
    """
    print("\n" + "="*70)
    print("EXCEPTIONAL EXPERIMENT: Multi-Landscape Generalization")
    print("="*70)
    
    output_dir = os.path.join(base_output_dir, "exceptional_multi_landscape")
    
    results = run_mountain_climbing_ga(
        pop_size=30,
        gene_count=3,
        generations=generations,
        iterations=2400,
        mutation_rate=0.1,
        use_multi_landscape=True,
        output_dir=output_dir
    )
    
    return {"multi_landscape": results}


def run_adaptive_mutation_experiment(base_output_dir, generations):
    """
    ADVANCED Experiment: Adaptive mutation rates.
    """
    print("\n" + "="*70)
    print("ADVANCED EXPERIMENT: Adaptive vs Fixed Mutation")
    print("="*70)
    
    all_results = {}
    
    # Fixed mutation
    print("\n--- Running with FIXED mutation rate ---")
    output_dir = os.path.join(base_output_dir, "advanced_fixed_mutation")
    results = run_mountain_climbing_ga(
        pop_size=30,
        gene_count=3,
        generations=generations,
        iterations=2400,
        mutation_rate=0.1,
        use_adaptive_mutation=False,
        output_dir=output_dir
    )
    all_results["fixed"] = results
    
    # Adaptive mutation
    print("\n--- Running with ADAPTIVE mutation rate ---")
    output_dir = os.path.join(base_output_dir, "advanced_adaptive_mutation")
    results = run_mountain_climbing_ga(
        pop_size=30,
        gene_count=3,
        generations=generations,
        iterations=2400,
        mutation_rate=0.1,
        use_adaptive_mutation=True,
        output_dir=output_dir
    )
    all_results["adaptive"] = results
    
    return all_results


def plot_experiment_results(results, title, xlabel, output_file, legend_prefix=""):
    """Generate a plot comparing experiment results."""
    if not HAS_MATPLOTLIB:
        return
    
    plt.figure(figsize=(12, 6))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
    
    for idx, (key, result) in enumerate(results.items()):
        history = result["history"]
        generations = [h["generation"] for h in history]
        max_fitness = [h["max_fitness"] for h in history]
        label = f"{legend_prefix}{key}"
        color = colors[idx % len(colors)]
        plt.plot(generations, max_fitness, label=label, linewidth=2.5, color=color)
    
    plt.xlabel("Generation", fontsize=14)
    plt.ylabel("Max Fitness", fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved plot: {output_file}")


def plot_mean_fitness(results, title, xlabel, output_file, legend_prefix=""):
    """Generate a plot comparing mean fitness."""
    if not HAS_MATPLOTLIB:
        return
    
    plt.figure(figsize=(12, 6))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
    
    for idx, (key, result) in enumerate(results.items()):
        history = result["history"]
        generations = [h["generation"] for h in history]
        mean_fitness = [h["mean_fitness"] for h in history]
        label = f"{legend_prefix}{key}"
        color = colors[idx % len(colors)]
        plt.plot(generations, mean_fitness, label=label, linewidth=2.5, color=color)
    
    plt.xlabel("Generation", fontsize=14)
    plt.ylabel("Mean Fitness", fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved plot: {output_file}")


def plot_comparison_bar(all_experiments, output_file):
    """Generate bar chart comparing best fitness across all experiments."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    experiment_names = []
    best_fitness_values = []
    
    for exp_name, results in all_experiments.items():
        for setting, result in results.items():
            experiment_names.append(f"{exp_name}\n{setting}")
            best_fitness_values.append(result.get('best_fitness', 0))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(experiment_names)))
    
    bars = ax.bar(range(len(experiment_names)), best_fitness_values, color=colors)
    ax.set_xticks(range(len(experiment_names)))
    ax.set_xticklabels(experiment_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Best Fitness', fontsize=14)
    ax.set_title('Best Fitness Comparison Across All Experiments', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, best_fitness_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved plot: {output_file}")


def generate_summary_table(all_experiments, output_file):
    """Generate a summary table of all experiments."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Category", "Experiment", "Setting", "Best Fitness", 
            "Final Mean Fitness", "Final Std"
        ])
        
        for exp_name, results in all_experiments.items():
            category = "BASIC"
            if "exceptional" in exp_name.lower() or "landscape" in exp_name.lower():
                category = "EXCEPTIONAL"
            elif "advanced" in exp_name.lower() or "adaptive" in exp_name.lower():
                category = "ADVANCED"
            
            for setting, result in results.items():
                history = result["history"]
                final_gen = history[-1]
                writer.writerow([
                    category,
                    exp_name,
                    setting,
                    f"{result.get('best_fitness', 0):.3f}",
                    f"{final_gen['mean_fitness']:.3f}",
                    f"{final_gen['std_fitness']:.3f}"
                ])
    
    print(f"Saved summary table: {output_file}")


def main():
    """Run all experiments for the video and generate reports."""
    
    # Create base output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"experiments_{timestamp}"
    os.makedirs(base_output_dir, exist_ok=True)
    
    print("="*70)
    print("CM3020 AI COURSEWORK - COMPLETE EXPERIMENT SUITE FOR VIDEO")
    print("="*70)
    print(f"Output directory: {base_output_dir}")
    print(f"Generations per experiment: {GENERATIONS}")
    print()
    print("This will run:")
    print("  - BASIC: Population size, mutation rate, gene count")
    print("  - ADVANCED: Adaptive mutation comparison")
    print("  - EXCEPTIONAL: Multi-landscape testing")
    print()
    
    all_experiments = {}
    
    # ========================================
    # BASIC EXPERIMENTS (Required)
    # ========================================
    print("\n" + "#"*70)
    print("# PART 1: BASIC EXPERIMENTS")
    print("#"*70)
    
    all_experiments["Population Size"] = run_population_size_experiments(
        base_output_dir, GENERATIONS
    )
    
    all_experiments["Mutation Rate"] = run_mutation_rate_experiments(
        base_output_dir, GENERATIONS
    )
    
    all_experiments["Gene Count"] = run_gene_count_experiments(
        base_output_dir, GENERATIONS
    )
    
    # ========================================
    # ADVANCED EXPERIMENTS (For higher grade)
    # ========================================
    print("\n" + "#"*70)
    print("# PART 2: ADVANCED EXPERIMENTS (Encoding Scheme)")
    print("#"*70)
    
    all_experiments["Adaptive Mutation"] = run_adaptive_mutation_experiment(
        base_output_dir, GENERATIONS
    )
    
    # ========================================
    # EXCEPTIONAL EXPERIMENTS (For >80%)
    # ========================================
    print("\n" + "#"*70)
    print("# PART 3: EXCEPTIONAL EXPERIMENTS")
    print("#"*70)
    
    all_experiments["Different Landscapes"] = run_landscape_experiments(
        base_output_dir, GENERATIONS
    )
    
    # Multi-landscape takes longer, comment out if needed
    # all_experiments["Multi-Landscape"] = run_multi_landscape_experiment(
    #     base_output_dir, GENERATIONS
    # )
    
    # ========================================
    # GENERATE PLOTS FOR VIDEO
    # ========================================
    if HAS_MATPLOTLIB:
        print("\n" + "="*70)
        print("GENERATING PLOTS FOR VIDEO")
        print("="*70)
        
        # Basic experiment plots
        plot_experiment_results(
            all_experiments["Population Size"],
            "Effect of Population Size on Fitness",
            "Generation",
            os.path.join(base_output_dir, "plot_population_size.png"),
            "Pop="
        )
        
        plot_experiment_results(
            all_experiments["Mutation Rate"],
            "Effect of Mutation Rate on Fitness",
            "Generation",
            os.path.join(base_output_dir, "plot_mutation_rate.png"),
            "Rate="
        )
        
        plot_experiment_results(
            all_experiments["Gene Count"],
            "Effect of Gene Count on Fitness",
            "Generation",
            os.path.join(base_output_dir, "plot_gene_count.png"),
            "Genes="
        )
        
        # Advanced experiment plot
        plot_experiment_results(
            all_experiments["Adaptive Mutation"],
            "Fixed vs Adaptive Mutation Rate",
            "Generation",
            os.path.join(base_output_dir, "plot_adaptive_mutation.png"),
            ""
        )
        
        # Exceptional experiment plot
        plot_experiment_results(
            all_experiments["Different Landscapes"],
            "Performance on Different Landscapes",
            "Generation",
            os.path.join(base_output_dir, "plot_landscapes.png"),
            ""
        )
        
        # Overall comparison
        plot_comparison_bar(
            all_experiments,
            os.path.join(base_output_dir, "plot_overall_comparison.png")
        )
    
    # ========================================
    # GENERATE SUMMARY
    # ========================================
    print("\n" + "="*70)
    print("GENERATING SUMMARY")
    print("="*70)
    
    generate_summary_table(
        all_experiments,
        os.path.join(base_output_dir, "experiment_summary.csv")
    )
    
    # Save all results as JSON
    json_file = os.path.join(base_output_dir, "all_results.json")
    with open(json_file, 'w') as f:
        json.dump(all_experiments, f, indent=2, default=str)
    print(f"Saved all results: {json_file}")
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {base_output_dir}/")
    print("\nFiles for your video:")
    print(f"  - {base_output_dir}/plot_population_size.png")
    print(f"  - {base_output_dir}/plot_mutation_rate.png")
    print(f"  - {base_output_dir}/plot_gene_count.png")
    print(f"  - {base_output_dir}/plot_adaptive_mutation.png")
    print(f"  - {base_output_dir}/plot_landscapes.png")
    print(f"  - {base_output_dir}/plot_overall_comparison.png")
    print(f"  - {base_output_dir}/experiment_summary.csv")
    print()
    print("To visualize best creatures:")
    print(f"  python visualize_creature.py {base_output_dir}/basic_pop50/best_creature_*.csv")
    print(f"  python visualize_creature.py {base_output_dir}/exceptional_landscape_gaussian_pyramid/best_creature_*.csv")
    print("="*70)


if __name__ == "__main__":
    main()
