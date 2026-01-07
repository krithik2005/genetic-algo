#!/usr/bin/env python3
"""
Generate plots from existing experiment results.
Run this after experiments are complete.
"""

import os
import sys
import csv
import glob
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not available. Plots will not be generated.")

BASE_DIR = "experiments_20260107_005117"

def load_experiment_results(exp_dir):
    """Load fitness history from results CSV."""
    results_file = os.path.join(exp_dir, "results_*.csv")
    files = glob.glob(results_file)
    if not files:
        return None
    
    history = []
    with open(files[0], 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            history.append({
                'generation': int(row['generation']),
                'best_fitness': float(row['max_fitness']),  # CSV uses 'max_fitness'
                'mean_fitness': float(row['mean_fitness']),
                'std_fitness': float(row['std_fitness'])
            })
    return history

def plot_experiment_comparison(experiments, title, output_file, label_prefix=""):
    """Plot multiple experiments on same graph."""
    if not HAS_MATPLOTLIB:
        return
    
    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    
    for (name, history), color in zip(experiments.items(), colors):
        if history:
            gens = [h['generation'] for h in history]
            best = [h['best_fitness'] for h in history]
            mean = [h['mean_fitness'] for h in history]
            
            plt.plot(gens, best, label=f"{label_prefix}{name} (Best)", 
                    color=color, linewidth=2, linestyle='-')
            plt.plot(gens, mean, label=f"{label_prefix}{name} (Mean)", 
                    color=color, linewidth=1, linestyle='--', alpha=0.6)
    
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved: {output_file}")

def main():
    if not HAS_MATPLOTLIB:
        print("ERROR: matplotlib required for plotting. Install with: pip install matplotlib")
        return
    
    print("="*70)
    print("GENERATING PLOTS FROM EXISTING EXPERIMENTS")
    print("="*70)
    
    # 1. Population Size
    print("\n[1/5] Population Size Comparison...")
    pop_experiments = {}
    for pop in [20, 50, 100]:
        exp_dir = os.path.join(BASE_DIR, f"basic_pop{pop}")
        history = load_experiment_results(exp_dir)
        if history:
            pop_experiments[str(pop)] = history
    
    if pop_experiments:
        plot_experiment_comparison(
            pop_experiments,
            "Effect of Population Size on Evolution",
            os.path.join(BASE_DIR, "plot_population_size.png"),
            "Pop="
        )
    
    # 2. Mutation Rate
    print("\n[2/5] Mutation Rate Comparison...")
    mut_experiments = {}
    for rate in [0.05, 0.1, 0.2, 0.3]:
        exp_dir = os.path.join(BASE_DIR, f"basic_mut{rate}")
        history = load_experiment_results(exp_dir)
        if history:
            mut_experiments[str(rate)] = history
    
    if mut_experiments:
        plot_experiment_comparison(
            mut_experiments,
            "Effect of Mutation Rate on Evolution",
            os.path.join(BASE_DIR, "plot_mutation_rate.png"),
            "Mut="
        )
    
    # 3. Gene Count
    print("\n[3/5] Gene Count Comparison...")
    gene_experiments = {}
    for count in [2, 3, 5, 7]:
        exp_dir = os.path.join(BASE_DIR, f"basic_genes{count}")
        history = load_experiment_results(exp_dir)
        if history:
            gene_experiments[str(count)] = history
    
    if gene_experiments:
        plot_experiment_comparison(
            gene_experiments,
            "Effect of Gene Count (Creature Complexity) on Evolution",
            os.path.join(BASE_DIR, "plot_gene_count.png"),
            "Genes="
        )
    
    # 4. Landscapes
    print("\n[4/5] Landscape Comparison...")
    landscape_experiments = {}
    landscapes = {
        "gaussian_pyramid": "Gaussian Pyramid",
        "mountain_with_cubes": "Mountain with Cubes",
        "mountain": "Standard Mountain"
    }
    for key, name in landscapes.items():
        exp_dir = os.path.join(BASE_DIR, f"exceptional_landscape_{key}")
        history = load_experiment_results(exp_dir)
        if history:
            landscape_experiments[name] = history
    
    if landscape_experiments:
        plot_experiment_comparison(
            landscape_experiments,
            "Performance on Different Landscapes",
            os.path.join(BASE_DIR, "plot_landscapes.png")
        )
    
    # 5. Adaptive Mutation
    print("\n[5/5] Adaptive Mutation Comparison...")
    adaptive_experiments = {}
    for name, key in [("Fixed", "fixed"), ("Adaptive", "adaptive")]:
        exp_dir = os.path.join(BASE_DIR, f"advanced_{key}_mutation")
        history = load_experiment_results(exp_dir)
        if history:
            adaptive_experiments[name] = history
    
    if adaptive_experiments:
        plot_experiment_comparison(
            adaptive_experiments,
            "Fixed vs Adaptive Mutation Rate",
            os.path.join(BASE_DIR, "plot_adaptive_mutation.png")
        )
    
    # Overall comparison bar chart
    print("\n[Bonus] Overall Best Fitness Comparison...")
    if HAS_MATPLOTLIB:
        plt.figure(figsize=(14, 6))
        experiment_names = []
        best_fitness_values = []
        
        all_exps = {
            "Pop20": "basic_pop20",
            "Pop50": "basic_pop50",
            "Pop100": "basic_pop100",
            "Mut0.05": "basic_mut0.05",
            "Mut0.1": "basic_mut0.1",
            "Mut0.2": "basic_mut0.2",
            "Mut0.3": "basic_mut0.3",
            "Genes2": "basic_genes2",
            "Genes3": "basic_genes3",
            "Genes5": "basic_genes5",
            "Genes7": "basic_genes7",
            "GaussPyramid": "exceptional_landscape_gaussian_pyramid",
            "MtnCubes": "exceptional_landscape_mountain_with_cubes",
            "Mountain": "exceptional_landscape_mountain",
            "FixedMut": "advanced_fixed_mutation",
            "AdaptMut": "advanced_adaptive_mutation"
        }
        
        for name, dir_key in all_exps.items():
            exp_dir = os.path.join(BASE_DIR, dir_key)
            history = load_experiment_results(exp_dir)
            if history:
                best = max([h['best_fitness'] for h in history])
                experiment_names.append(name)
                best_fitness_values.append(best)
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(experiment_names)))
        bars = plt.bar(range(len(experiment_names)), best_fitness_values, color=colors)
        plt.xticks(range(len(experiment_names)), experiment_names, rotation=45, ha='right')
        plt.ylabel('Best Fitness', fontsize=12)
        plt.title('Best Fitness Across All Experiments', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, best_fitness_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, "plot_overall_comparison.png"), dpi=150)
        plt.close()
        print(f"Saved: {os.path.join(BASE_DIR, 'plot_overall_comparison.png')}")
    
    print("\n" + "="*70)
    print("ALL PLOTS GENERATED!")
    print("="*70)
    print(f"\nPlots saved to: {BASE_DIR}/")
    print("\nGenerated files:")
    for plot_file in glob.glob(os.path.join(BASE_DIR, "plot_*.png")):
        print(f"  - {plot_file}")

if __name__ == "__main__":
    main()