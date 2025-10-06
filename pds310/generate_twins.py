#!/usr/bin/env python
"""
Generate Digital Twin Cohort for PDS310.

This script creates synthetic patients using the digital twin generation framework.
"""

import sys
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pds310.profile_database import load_profile_database
from pds310.twin_generator import generate_twin_cohort, get_twin_statistics
from pds310.mutation import batch_mutate
from pds310.twin_validator import validate_twin_cohort
from pds310.twin_metrics import (
    calculate_twin_diversity,
    plot_twin_vs_real_comparison,
    export_diversity_report
)


def main():
    parser = argparse.ArgumentParser(description="Generate digital twin cohort")
    parser.add_argument(
        "--profiles",
        type=str,
        default="outputs/pds310/patient_profiles.csv",
        help="Path to patient profiles database"
    )
    parser.add_argument(
        "--n_twins",
        type=int,
        default=1000,
        help="Number of twins to generate"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="random",
        choices=["random", "cluster", "arm_specific", "subgroup"],
        help="Generation strategy"
    )
    parser.add_argument(
        "--mutation_rate",
        type=float,
        default=0.05,
        help="Mutation rate (0-1)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/pds310",
        help="Output directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--ras_wt_only",
        action="store_true",
        help="Generate only RAS wild-type twins"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PDS310 DIGITAL TWIN GENERATION")
    print("=" * 80)
    print()
    
    # Step 1: Load real patient profiles
    print("STEP 1: Loading patient profiles...")
    print("-" * 80)
    real_profiles = load_profile_database(args.profiles)
    print(f"Loaded {len(real_profiles)} real patient profiles")
    print()
    
    # Step 2: Generate twin cohort
    print("STEP 2: Generating digital twins...")
    print("-" * 80)
    
    constraints = None
    if args.ras_wt_only:
        constraints = {"RAS_status": "WILD-TYPE"}
        print("Constraint: RAS wild-type only")
    
    twins = generate_twin_cohort(
        profile_db=real_profiles,
        n_twins=args.n_twins,
        strategy=args.strategy,
        constraints=constraints,
        validation_threshold=0.0,  # No validation during generation for speed
        seed=args.seed,
        verbose=True
    )
    print()
    
    # Step 3: Apply mutations
    print("STEP 3: Applying mutations...")
    print("-" * 80)
    print(f"Mutation rate: {args.mutation_rate}")
    
    mutated_twins = batch_mutate(
        twins,
        mutation_rate=args.mutation_rate,
        maintain_correlations=True,
        seed=args.seed
    )
    print(f"Applied mutations to {len(mutated_twins)} twins")
    print()
    
    # Step 4: Validate twins
    print("STEP 4: Validating digital twins...")
    print("-" * 80)
    
    validation_results = validate_twin_cohort(
        mutated_twins,
        real_profiles,
        min_validation_score=0.7,
        verbose=True
    )
    print()
    
    # Step 5: Calculate diversity metrics
    print("STEP 5: Calculating diversity metrics...")
    print("-" * 80)
    
    diversity_metrics = calculate_twin_diversity(
        mutated_twins,
        real_profiles,
        verbose=True
    )
    print()
    
    # Step 6: Generate statistics
    print("STEP 6: Generating statistics...")
    print("-" * 80)
    
    twin_stats = get_twin_statistics(mutated_twins, real_profiles)
    
    print("\nTwin vs Real Statistics:")
    for feature, stats in twin_stats.get("distribution_comparisons", {}).items():
        if isinstance(stats, dict) and "twin_mean" in stats:
            print(f"  {feature}:")
            print(f"    Real: {stats['real_mean']:.2f} ± {stats['real_std']:.2f}")
            print(f"    Twin: {stats['twin_mean']:.2f} ± {stats['twin_std']:.2f}")
            print(f"    Difference: {stats['mean_diff']:.2f}")
    print()
    
    # Step 7: Save outputs
    print("STEP 7: Saving outputs...")
    print("-" * 80)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save twin cohort
    twin_file = output_dir / f"digital_twins_n{args.n_twins}.csv"
    mutated_twins.to_csv(twin_file, index=False)
    print(f"Saved twins to: {twin_file}")
    
    # Save diversity report
    report_file = output_dir / f"twin_diversity_report_n{args.n_twins}.txt"
    export_diversity_report(diversity_metrics, str(report_file))
    
    # Save validation summary
    import json
    validation_file = output_dir / f"twin_validation_n{args.n_twins}.json"
    with open(validation_file, 'w') as f:
        json.dump(validation_results, f, indent=2)
    print(f"Saved validation results to: {validation_file}")
    
    # Generate comparison plots
    plot_file = output_dir / f"twin_vs_real_comparison_n{args.n_twins}.png"
    plot_twin_vs_real_comparison(
        mutated_twins,
        real_profiles,
        output_path=str(plot_file)
    )
    print()
    
    # Step 8: Summary
    print("=" * 80)
    print("DIGITAL TWIN GENERATION COMPLETE!")
    print("=" * 80)
    print()
    print(f"Generated: {len(mutated_twins)} digital twins")
    print(f"Strategy: {args.strategy}")
    print(f"Mutation Rate: {args.mutation_rate}")
    print()
    print("Quality Metrics:")
    print(f"  Validation Pass Rate: {validation_results['passing_percentage']:.1f}%")
    print(f"  Mean Validation Score: {validation_results['mean_validation_score']:.3f}")
    print(f"  Overall Diversity Score: {diversity_metrics['overall_diversity_score']:.3f}")
    print()
    print("Output Files:")
    print(f"  - Twin Cohort: {twin_file}")
    print(f"  - Diversity Report: {report_file}")
    print(f"  - Validation Results: {validation_file}")
    print(f"  - Comparison Plot: {plot_file}")
    print()


if __name__ == "__main__":
    main()
