#!/usr/bin/env python
"""
Build Complete Patient Profile Database for PDS310.

This script creates the comprehensive digital profiles for all 370 patients
by integrating data from all available ADaM tables.
"""

import sys
from pathlib import Path
import yaml

# Add parent directory to path to import pds310
sys.path.insert(0, str(Path(__file__).parent.parent))

from pds310.io import load_adam_tables
from pds310.profile_database import create_profile_database, export_database_summary
from pds310.profile_viz import (
    plot_profile_distribution,
    plot_correlation_heatmap,
    plot_subgroup_comparison,
    export_profile_summary_table
)


def main():
    """Build profile database and generate visualizations."""
    
    # Load configuration
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    data_root = config["data_root"]
    outputs_root = config["outputs_root"]
    
    print("=" * 80)
    print("PDS310 DIGITAL PROFILE DATABASE BUILDER")
    print("=" * 80)
    print()
    
    # Step 1: Load all ADaM tables
    print("STEP 1: Loading ADaM tables...")
    print("-" * 80)
    tables = load_adam_tables(data_root, verbose=True)
    print()
    
    # Step 2: Create profile database
    print("STEP 2: Creating digital profile database...")
    print("-" * 80)
    output_path = Path(outputs_root) / "patient_profiles.csv"
    
    profile_db = create_profile_database(
        tables=tables,
        output_path=str(output_path),
        include_outcomes=True,
        verbose=True
    )
    print()
    
    # Step 3: Generate database summary
    print("STEP 3: Generating database summary...")
    print("-" * 80)
    summary_json_path = Path(outputs_root) / "profile_database_summary.json"
    summary_txt_path = Path(outputs_root) / "profile_database_summary.txt"
    
    export_database_summary(profile_db, str(summary_json_path), format="json")
    export_database_summary(profile_db, str(summary_txt_path), format="txt")
    print()
    
    # Step 4: Generate visualizations
    print("STEP 4: Generating visualizations...")
    print("-" * 80)
    
    # Profile distributions
    print("  - Creating profile distribution plots...")
    plot_profile_distribution(
        profile_db,
        output_dir=str(outputs_root)
    )
    
    # Correlation heatmap
    print("  - Creating correlation heatmap...")
    plot_correlation_heatmap(
        profile_db,
        output_path=str(Path(outputs_root) / "profile_correlations.png")
    )
    
    # Subgroup comparisons
    if "RAS_status" in profile_db.columns and "composite_risk_score" in profile_db.columns:
        print("  - Creating RAS status vs risk score comparison...")
        plot_subgroup_comparison(
            profile_db,
            subgroup_col="RAS_status",
            metric_col="composite_risk_score",
            output_path=str(Path(outputs_root) / "ras_vs_risk_score.png")
        )
    
    if "ATRT" in profile_db.columns and "DTHDYX" in profile_db.columns:
        print("  - Creating treatment arm vs survival comparison...")
        plot_subgroup_comparison(
            profile_db,
            subgroup_col="ATRT",
            metric_col="DTHDYX",
            output_path=str(Path(outputs_root) / "treatment_vs_survival.png")
        )
    
    # Export summary table
    print("  - Exporting profile summary table...")
    export_profile_summary_table(
        profile_db,
        output_path=str(Path(outputs_root) / "profile_summary_table.csv"),
        format="csv"
    )
    print()
    
    # Step 5: Summary statistics
    print("=" * 80)
    print("PROFILE DATABASE CREATED SUCCESSFULLY!")
    print("=" * 80)
    print()
    print(f"Database Location: {output_path}")
    print(f"Number of Patients: {len(profile_db)}")
    print(f"Number of Features: {len(profile_db.columns)}")
    print()
    print("Output Files:")
    print(f"  - Profile Database: {output_path}")
    print(f"  - Summary (JSON): {summary_json_path}")
    print(f"  - Summary (TXT): {summary_txt_path}")
    print(f"  - Distributions Plot: {Path(outputs_root) / 'profile_distributions.png'}")
    print(f"  - Correlations Plot: {Path(outputs_root) / 'profile_correlations.png'}")
    print(f"  - Summary Table: {Path(outputs_root) / 'profile_summary_table.csv'}")
    print()
    
    # Display key statistics
    print("Key Statistics:")
    print(f"  - Age (mean ± SD): {profile_db['AGE'].mean():.1f} ± {profile_db['AGE'].std():.1f} years")
    
    if "SEX" in profile_db.columns:
        sex_counts = profile_db["SEX"].value_counts()
        print(f"  - Sex: {sex_counts.to_dict()}")
    
    if "ATRT" in profile_db.columns:
        trt_counts = profile_db["ATRT"].value_counts()
        print(f"  - Treatment Arms:")
        for trt, count in trt_counts.items():
            print(f"      {trt}: {count}")
    
    if "RAS_status" in profile_db.columns:
        ras_counts = profile_db["RAS_status"].value_counts()
        print(f"  - RAS Status:")
        for status, count in ras_counts.items():
            print(f"      {status}: {count}")
    
    if "DTHX" in profile_db.columns:
        os_events = profile_db["DTHX"].sum()
        print(f"  - OS Events: {os_events}/{len(profile_db)} ({os_events/len(profile_db)*100:.1f}%)")
    
    if "best_response" in profile_db.columns:
        resp_counts = profile_db["best_response"].value_counts()
        print(f"  - Best Response:")
        for resp, count in resp_counts.items():
            print(f"      {resp}: {count}")
    
    print()
    print("Profile database build complete!")
    print()


if __name__ == "__main__":
    main()
