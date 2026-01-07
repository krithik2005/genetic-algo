#!/usr/bin/env python3
"""
Continue experiments from where they stopped.
Run this to complete the remaining experiments.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mountain_climbing_ga import run_mountain_climbing_ga

# Use the same output directory from the interrupted run
BASE_DIR = "experiments_20260107_005117"
GENERATIONS = 50

def main():
    print("="*70)
    print("RUNNING REMAINING EXPERIMENTS (Landscape + Adaptive)")
    print("="*70)
    
    # 1. Landscape: gaussian_pyramid
    print("\n[1/5] Landscape: gaussian_pyramid...")
    try:
        run_mountain_climbing_ga(
            pop_size=30,
            gene_count=3,
            generations=GENERATIONS,
            iterations=2400,
            mutation_rate=0.1,
            landscape_type="gaussian_pyramid",
            output_dir=os.path.join(BASE_DIR, "exceptional_landscape_gaussian_pyramid"),
            use_threading=True,
            pool_size=8
        )
    except Exception as e:
        print(f"Landscape gaussian_pyramid failed: {e}")
    
    # 2. Landscape: mountain_with_cubes
    print("\n[2/5] Landscape: mountain_with_cubes...")
    try:
        run_mountain_climbing_ga(
            pop_size=30,
            gene_count=3,
            generations=GENERATIONS,
            iterations=2400,
            mutation_rate=0.1,
            landscape_type="mountain_with_cubes",
            output_dir=os.path.join(BASE_DIR, "exceptional_landscape_mountain_with_cubes"),
            use_threading=True,
            pool_size=8
        )
    except Exception as e:
        print(f"Landscape mountain_with_cubes failed: {e}")
    
    # 3. Landscape: mountain
    print("\n[3/5] Landscape: mountain...")
    try:
        run_mountain_climbing_ga(
            pop_size=30,
            gene_count=3,
            generations=GENERATIONS,
            iterations=2400,
            mutation_rate=0.1,
            landscape_type="mountain",
            output_dir=os.path.join(BASE_DIR, "exceptional_landscape_mountain"),
            use_threading=True,
            pool_size=8
        )
    except Exception as e:
        print(f"Landscape mountain failed: {e}")
    
    # 4. Fixed mutation
    print("\n[4/5] Fixed mutation rate...")
    try:
        run_mountain_climbing_ga(
            pop_size=30,
            gene_count=3,
            generations=GENERATIONS,
            iterations=2400,
            mutation_rate=0.1,
            use_adaptive_mutation=False,
            output_dir=os.path.join(BASE_DIR, "advanced_fixed_mutation"),
            use_threading=True,
            pool_size=8
        )
    except Exception as e:
        print(f"Fixed mutation failed: {e}")
    
    # 5. Adaptive mutation
    print("\n[5/5] Adaptive mutation rate...")
    try:
        run_mountain_climbing_ga(
            pop_size=30,
            gene_count=3,
            generations=GENERATIONS,
            iterations=2400,
            mutation_rate=0.1,
            use_adaptive_mutation=True,
            output_dir=os.path.join(BASE_DIR, "advanced_adaptive_mutation"),
            use_threading=True,
            pool_size=8
        )
    except Exception as e:
        print(f"Adaptive mutation failed: {e}")
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {BASE_DIR}/")
    print("\nTo visualize best creatures:")
    print(f"  python visualize_creature.py {BASE_DIR}/exceptional_landscape_gaussian_pyramid/best_creature_*.csv")
    print(f"  python visualize_creature.py {BASE_DIR}/exceptional_landscape_mountain/best_creature_*.csv")

if __name__ == "__main__":
    main()

