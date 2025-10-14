#!/usr/bin/env python
"""
Train Adverse Event and End-of-Treatment Models for PDS310.

This script trains models to predict:
1. Adverse events (AE) during treatment
2. End of treatment (EOT) timing/discontinuation

These models can be used to simulate treatment interruptions in virtual trials.
"""

import sys
from pathlib import Path
import yaml
import joblib
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pds310.io import load_adam_tables
from pds310.endpoints import prepare_eot_competing_from_adam
from pds310.profile_database import load_profile_database
from pds310.model_ae import fit_cause_specific_cox_ae, fit_eot_allcause
from pds310.simulate import simulate_patients
from pds310.reporting import write_report
from pds310.plotting import plot_ae_incidence, plot_eot_distributions_from_tables

ID_COL = "SUBJID"
STUDY_COL = "STUDYID"


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def main():
    """Train AE/EOT models and generate simulation outputs."""
    
    print("=" * 80)
    print("PDS310 ADVERSE EVENT & END-OF-TREATMENT MODEL TRAINING")
    print("=" * 80)
    print()
    
    # Load configuration
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    data_root = cfg["data_root"]
    out_root = cfg["outputs_root"]
    ensure_dir(out_root)
    
    # Step 1: Load data
    print("STEP 1: Loading ADaM tables...")
    print("-" * 80)
    tables = load_adam_tables(data_root, verbose=True)
    print()
    
    # Step 2: Check if profiles exist
    print("STEP 2: Loading patient profiles...")
    print("-" * 80)
    profile_path = Path(out_root) / "patient_profiles.csv"
    
    if not profile_path.exists():
        print("❌ ERROR: Digital profiles not found!")
        print(f"Expected location: {profile_path}")
        print("\nPlease build profiles first:")
        print("  uv run python pds310/build_profiles.py")
        print()
        return 1
    
    profiles = load_profile_database(str(profile_path))
    print(f"✅ Loaded {len(profiles)} patient profiles")
    print()
    
    # Step 3: Prepare EOT data
    print("STEP 3: Preparing end-of-treatment data...")
    print("-" * 80)
    eot = prepare_eot_competing_from_adam(
        tables.get("adsl"),
        tables.get("adlb"),
        tables.get("adae")
    )
    
    if eot is None or eot.empty:
        print("❌ ERROR: Unable to derive EOT from ADLB")
        print("Ensure ADLB contains VISITDY column with longitudinal data")
        return 1
    
    # Merge profiles with EOT data
    df = profiles.merge(eot, on=[ID_COL, STUDY_COL], how="inner")
    df = df[df["time"] > 0].reset_index(drop=True)
    print(f"✅ Prepared {len(df)} patients with EOT data")
    print()
    
    # Step 4: Train AE model
    print("STEP 4: Training adverse event model (Cox PH)...")
    print("-" * 80)
    ae_res = fit_cause_specific_cox_ae(df)
    
    ae_path = Path(out_root) / "ae_cox.joblib"
    joblib.dump(ae_res, ae_path)
    print(f"✅ AE model saved to: {ae_path}")
    print(f"   Features: {len(ae_res['feature_cols'])}")
    print()
    
    # Step 5: Train EOT model
    print("STEP 5: Training end-of-treatment model (AFT)...")
    print("-" * 80)
    eot_res = fit_eot_allcause(df)
    
    eot_path = Path(out_root) / "eot_model.joblib"
    joblib.dump(eot_res, eot_path)
    model_type = eot_res.get("model_type", "unknown")
    print(f"✅ EOT model saved to: {eot_path}")
    print(f"   Model type: {model_type}")
    print()
    
    # Step 6: Simulate patients
    print("STEP 6: Simulating patient AE/EOT outcomes...")
    print("-" * 80)
    n_sim = cfg.get("simulation", {}).get("n_simulations", 100)
    seed = cfg.get("simulation", {}).get("seed", 42)
    
    sim_df = simulate_patients(
        df_features=profiles,
        aft_allcause=eot_res,
        cox_ae=ae_res["model"],
        ohe=ae_res["ohe"],
        feature_cols=ae_res["feature_cols"],
        adsl=tables["adsl"],
        n_sim=n_sim,
        seed=seed,
    )
    
    sim_path = Path(out_root) / "sim_ae.csv"
    sim_df.to_csv(sim_path, index=False)
    print(f"✅ Simulated data saved to: {sim_path}")
    print(f"   Simulations: {n_sim}")
    print(f"   Total records: {len(sim_df)}")
    print()
    
    # Step 7: Generate report
    print("STEP 7: Generating AE comparison report...")
    print("-" * 80)
    report_path = write_report(
        tables.get("adsl"),
        tables.get("adlb"),
        tables.get("adae"),
        str(sim_path),
        out_root
    )
    print(f"✅ AE report written to: {report_path}")
    print()
    
    # Step 8: Generate plots
    print("STEP 8: Generating visualizations...")
    print("-" * 80)
    
    # AE incidence plots
    ae_plots = plot_ae_incidence(str(report_path), out_root)
    print("AE incidence plots:")
    for p in ae_plots:
        print(f"  - {p}")
    
    # EOT distribution plot
    eot_plot = plot_eot_distributions_from_tables(
        tables.get("adlb"),
        str(sim_path),
        out_root
    )
    print(f"EOT distribution plot:")
    print(f"  - {eot_plot}")
    print()
    
    # Summary
    print("=" * 80)
    print("AE/EOT MODEL TRAINING COMPLETE!")
    print("=" * 80)
    print()
    print("Output files:")
    print(f"  - AE Model: {ae_path}")
    print(f"  - EOT Model: {eot_path}")
    print(f"  - Simulated data: {sim_path}")
    print(f"  - Report: {report_path}")
    print(f"  - Plots: {Path(out_root) / 'plot_ae_incidence_*.png'}")
    print(f"           {Path(out_root) / 'plot_eot_distributions.png'}")
    print()
    print("These models can be used to simulate treatment discontinuation")
    print("in virtual trials (feature not yet integrated).")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
