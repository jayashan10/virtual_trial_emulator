import os
import json
import yaml
import joblib
import numpy as np
import pandas as pd

from lifelines import KaplanMeierFitter

from .io import load_adam_tables
from .profile_database import load_profile_database
from .model_survival import prepare_os_data, train_os_model


ID_COL = "SUBJID"
STUDY_COL = "STUDYID"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_baseline_os(config_path: str) -> None:
    """Run OS modeling using pre-built digital profiles."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    out_root = cfg["outputs_root"]
    ensure_dir(out_root)
    models_dir = os.path.join(out_root, "models")
    ensure_dir(models_dir)

    # Load digital profiles
    profile_path = os.path.join(out_root, "patient_profiles.csv")
    if not os.path.exists(profile_path):
        print("=" * 80)
        print("ERROR: Digital profiles not found!")
        print("=" * 80)
        print(f"Expected location: {profile_path}")
        print("\nPlease build profiles first:")
        print("  uv run python pds310/build_profiles.py")
        print()
        return

    print("Loading digital profiles...")
    profiles = load_profile_database(profile_path)
    print(f"Loaded {len(profiles)} patient profiles with {len(profiles.columns)} features")

    # Prepare OS data (includes rare category collapsing!)
    print("\nPreparing OS data...")
    X, durations, events = prepare_os_data(
        profiles,
        duration_col="DTHDYX",
        event_col="DTHX",
        min_duration=1.0,
        exclude_outcomes=True
    )
    print(f"Training set: {len(X)} patients, {len(X.columns)} features")
    print(f"Events: {events.sum()}/{len(events)} ({events.sum()/len(events)*100:.1f}%)")

    # Train Cox model (includes cross-validation)
    print("\nTraining Cox proportional hazards model...")
    cox_model = train_os_model(X, durations, events, penalizer=0.1, random_state=42)

    # Save model
    model_path = os.path.join(models_dir, "os_model.joblib")
    joblib.dump(cox_model, model_path)
    print(f"✅ OS model saved to: {model_path}")

    # Extract metrics (CV is done inside train_os_model)
    metrics = {
        "train_concordance": float(cox_model["train_concordance"]),
        "cv_concordance_mean": float(cox_model["cv_cindex_mean"]),
        "cv_concordance_std": float(cox_model["cv_cindex_std"]),
        "n_patients": int(len(X)),
        "n_features": int(len(X.columns)),
        "n_events": int(events.sum()),
        "event_rate": float(events.sum() / len(events)),
    }
    
    metrics_path = os.path.join(out_root, "metrics_os.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nOS Model Performance:")
    print(f"  Train concordance: {metrics['train_concordance']:.3f}")
    print(f"  CV concordance: {metrics['cv_concordance_mean']:.3f} ± {metrics['cv_concordance_std']:.3f}")
    print(f"\n✅ Metrics saved to: {metrics_path}")


def run_os_overlay(config_path: str, sims: int = 50) -> None:
    """Generate OS simulation overlays using trained model."""
    print("=" * 80)
    print("NOTE: OS overlay functionality has been moved to:")
    print("  uv run python pds310/run_virtual_trial.py --effect_source learned")
    print()
    print("The virtual trial script provides:")
    print("  - KM overlays with observed vs simulated survival")
    print("  - Treatment arm comparisons")
    print("  - Trial statistics and power analysis")
    print("=" * 80)


def run_ae_and_sim(config_path: str) -> None:
    """Deprecated: Use standalone train_ae_models.py script instead."""
    print("=" * 80)
    print("DEPRECATED: CLI AE command")
    print("=" * 80)
    print()
    print("AE/EOT modeling has been moved to a standalone script.")
    print()
    print("Use instead:")
    print("  uv run python pds310/train_ae_models.py")
    print()
    print("This script will:")
    print("  - Train adverse event (AE) models")
    print("  - Train end-of-treatment (EOT) models")
    print("  - Generate simulations and reports")
    print("  - Create visualization plots")
    print("=" * 80)


def run_report_ae(config_path: str) -> None:
    """Deprecated: Use train_ae_models.py which includes reporting."""
    print("=" * 80)
    print("DEPRECATED: CLI report-ae command")
    print("=" * 80)
    print()
    print("AE reporting is now integrated into:")
    print("  uv run python pds310/train_ae_models.py")
    print()
    print("This generates all AE reports and visualizations in one run.")
    print("=" * 80)


def run_plots(config_path: str) -> None:
    """Deprecated: Use train_ae_models.py which includes plots."""
    print("=" * 80)
    print("DEPRECATED: CLI plots command")
    print("=" * 80)
    print()
    print("AE/EOT plotting is now integrated into:")
    print("  uv run python pds310/train_ae_models.py")
    print()
    print("This generates all visualizations automatically.")
    print("=" * 80)


def run_calibrate_ae(config_path: str) -> None:
    """Deprecated: Use train_ae_models.py instead."""
    print("=" * 80)
    print("DEPRECATED: CLI calibrate-ae command")
    print("=" * 80)
    print()
    print("AE calibration is no longer supported.")
    print()
    print("Use the standalone script for AE modeling:")
    print("  uv run python pds310/train_ae_models.py")
    print("=" * 80)


def run_os_advanced(config_path: str) -> None:
    """Deprecated: RSF and GB models had poor performance (C-index ~0.33)."""
    print("=" * 80)
    print("DEPRECATED: Advanced OS models (RSF, GB) command")
    print("=" * 80)
    print()
    print("Random Survival Forest and Gradient Boosting models")
    print("underperformed Cox PH (C-index ~0.33 vs 0.666).")
    print()
    print("Use the standard Cox model instead:")
    print("  uv run -m pds310.cli os --config pds310/config.yaml")
    print()
    print("Or use the comprehensive training pipeline:")
    print("  uv run python pds310/train_models.py --seed 42")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PDS310 pipelines")
    sub = parser.add_subparsers(dest="cmd", required=False)

    p1 = sub.add_parser("os", help="Run OS baseline pipeline for PDS310")
    p1.add_argument("--config", type=str, default="pds310/config.yaml")

    p2 = sub.add_parser("os-overlay", help="Simulated vs observed OS overlays for PDS310")
    p2.add_argument("--config", type=str, default="pds310/config.yaml")
    p2.add_argument("--sims", type=int, default=50)

    p3 = sub.add_parser("ae", help="Run AE competing risks + EOT simulation for PDS310")
    p3.add_argument("--config", type=str, default="pds310/config.yaml")

    p4 = sub.add_parser("report-ae", help="Generate AE observed vs simulated report for PDS310")
    p4.add_argument("--config", type=str, default="pds310/config.yaml")

    p5 = sub.add_parser("calibrate-ae", help="Apply EOT time scaling and regenerate sim_ae_calibrated.csv for PDS310")
    p5.add_argument("--config", type=str, default="pds310/config.yaml")

    p6 = sub.add_parser("plots", help="Generate AE incidence and EOT distribution plots for PDS310")
    p6.add_argument("--config", type=str, default="pds310/config.yaml")

    p7 = sub.add_parser("os-advanced", help="Train RSF and GB survival models for OS (PDS310)")
    p7.add_argument("--config", type=str, default="pds310/config.yaml")

    args = parser.parse_args()

    if args.cmd == "os-overlay":
        run_os_overlay(args.config, sims=args.sims)
    elif args.cmd == "ae":
        run_ae_and_sim(args.config)
    elif args.cmd == "report-ae":
        run_report_ae(args.config)
    elif args.cmd == "calibrate-ae":
        run_calibrate_ae(args.config)
    elif args.cmd == "plots":
        run_plots(args.config)
    elif args.cmd == "os-advanced":
        run_os_advanced(args.config)
    else:
        run_baseline_os(args.config)


