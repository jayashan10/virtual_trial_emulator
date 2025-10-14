#!/usr/bin/env python
"""
Run Complete Virtual Clinical Trial for PDS310.

End-to-end simulation with enrollment, randomization, outcome prediction,
and statistical analysis.
"""

import sys
from pathlib import Path
import argparse
import warnings
from typing import Any, Dict, List
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from pds310.trial_design import TrialDesign, create_pds310_original_design
from pds310.virtual_trial import VirtualTrial
from pds310.predict_outcomes import OutcomePredictor
from pds310.profile_database import load_profile_database
from pds310.trial_statistics import (
    calculate_survival_statistics,
    plot_kaplan_meier,
    calculate_response_rate_comparison,
    calculate_sample_size_survival,
    forest_plot
)
import pandas as pd
import numpy as np
import json


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _sanitize_nan(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nan(v) for v in obj]
    if isinstance(obj, (float, np.floating)) and (np.isnan(obj) or not np.isfinite(obj)):
        return None
    if isinstance(obj, (np.floating, np.float32, np.float64)) and (np.isnan(float(obj)) or not np.isfinite(float(obj))):
        return None
    return obj


def main():
    parser = argparse.ArgumentParser(description="Run virtual clinical trial")
    parser.add_argument(
        "--profiles",
        type=str,
        default="outputs/pds310/patient_profiles.csv",
        help="Path to patient profiles"
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="outputs/pds310/models",
        help="Directory with trained models"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/pds310/virtual_trial",
        help="Output directory"
    )
    parser.add_argument(
        "--n_patients",
        type=int,
        default=300,
        help="Number of patients to enroll"
    )
    parser.add_argument(
        "--design",
        type=str,
        default="original",
        choices=["original", "custom"],
        help="Trial design to use"
    )
    parser.add_argument(
        "--effect_source",
        type=str,
        default="learned",
        choices=["learned", "assumed"],
        help="Use learned treatment effects or assumed design modifiers"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PDS310 VIRTUAL CLINICAL TRIAL SIMULATION")
    print("=" * 80)
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load patient profiles
    print("STEP 1: Loading patient profiles...")
    print("-" * 80)
    
    profiles = load_profile_database(args.profiles)
    print(f"Loaded {len(profiles)} patient profiles")
    print()
    
    # Step 2: Load trained models
    print("STEP 2: Loading prediction models...")
    print("-" * 80)
    
    models_dir = Path(args.models_dir)
    predictor = OutcomePredictor(
        response_model_path=str(models_dir / "response_model.joblib"),
        ttr_model_path=str(models_dir / "ttr_model.joblib"),
        survival_model_path=str(models_dir / "os_model.joblib"),
        random_seed=args.seed,
    )
    print(f"✅ Models loaded from: {args.models_dir}")
    print()
    
    # Step 3: Create trial design
    print("STEP 3: Creating trial design...")
    print("-" * 80)
    
    if args.design == "original":
        design = create_pds310_original_design()
        # Override n_patients
        design.n_patients = args.n_patients
        design.random_seed = args.seed
    else:
        raise ValueError(f"Design {args.design} not implemented")
    
    print(f"Trial: {design.trial_name}")
    print(f"Description: {design.description}")
    print(f"Sample size: {design.n_patients}")
    print(f"Arms: {len(design.treatment_arms)}")
    for arm in design.treatment_arms:
        print(f"  - {arm.name}: {arm.description}")
    print()
    
    # Save design
    design.save(str(output_dir / "trial_design.json"))
    
    # Step 4: Run virtual trial
    print("STEP 4: Running virtual trial simulation...")
    print("-" * 80)
    
    trial = VirtualTrial(
        design=design,
        source_profiles=profiles,
        outcome_predictor=predictor,
        effect_source=args.effect_source,
        verbose=True
    )
    
    trial_results = trial.run()
    
    # Save results
    trial.save_results(str(output_dir))
    print()
    
    # Step 5: Summary statistics
    print("STEP 5: Calculating summary statistics...")
    print("-" * 80)
    
    summary = trial.summary()
    print("\nTrial Summary by Treatment Arm:")
    print(summary.to_string(index=False))
    
    summary.to_csv(output_dir / "trial_summary.csv", index=False)
    print()
    
    # Step 6: Survival analysis (if OS/PFS data available)
    print("STEP 6: Statistical analysis...")
    print("-" * 80)
    
    enrolled = trial.enrolled_patients
    survival_results: List[Dict[str, Any]] = []
    response_results: List[Dict[str, Any]] = []
    effect_source = trial.trial_results.get('effect_source', 'learned')

    control_arm = next((arm.name for arm in design.treatment_arms if arm.is_control), design.treatment_arms[0].name)
    treatment_arms = [arm.name for arm in design.treatment_arms if arm.name != control_arm]

    if 'os_time' in enrolled.columns and 'os_event' in enrolled.columns:
        print("\nOverall Survival Analysis:")
        print("-" * 80)
        
        enrolled_survival = enrolled.copy()
        time_col = 'os_time'
        if effect_source == 'assumed':
            time_col = 'os_time_adjusted'
            for arm_obj in design.treatment_arms:
                mask = enrolled_survival['treatment_arm'] == arm_obj.name
                if arm_obj.is_control:
                    enrolled_survival.loc[mask, 'os_time_adjusted'] = enrolled_survival.loc[mask, 'os_time']
                else:
                    enrolled_survival.loc[mask, 'os_time_adjusted'] = (
                        enrolled_survival.loc[mask, 'os_time'] * (1 / max(arm_obj.os_hr, 1e-6))
                    )
        
        trt_col = "ATRT" if "ATRT" in profiles.columns else "TRT"
        observed_survival = profiles[[trt_col, "DTHDYX", "DTHX"]].rename(columns={
            trt_col: "treatment_arm",
            "DTHDYX": "os_time",
            "DTHX": "os_event",
        })

        canonical_names = {arm.name.lower(): arm.name for arm in design.treatment_arms}
        arm_aliases = {
            "best supportive care": "BSC",
            "bsc": "BSC",
            "panitumumab + bsc": "Panitumumab + BSC",
            "panitumumab plus best supportive care": "Panitumumab + BSC",
            "panit. plus best supportive care": "Panitumumab + BSC",
            "panitumumab+bsc": "Panitumumab + BSC",
        }

        def _normalize_arm(name):
            if not isinstance(name, str):
                return name
            key = name.strip().lower()
            if key in arm_aliases:
                return arm_aliases[key]
            if key in canonical_names:
                return canonical_names[key]
            if "panit" in key:
                return "Panitumumab + BSC"
            if "best supportive care" in key:
                return "BSC"
            return name

        observed_survival["treatment_arm"] = observed_survival["treatment_arm"].apply(_normalize_arm)
        observed_survival = observed_survival.dropna(subset=["os_time", "os_event"])
        observed_survival = observed_survival[
            observed_survival["treatment_arm"].isin([control_arm] + treatment_arms)
        ]

        for treatment_arm in treatment_arms:
            try:
                stats = calculate_survival_statistics(
                    data=enrolled_survival,
                    time_col=time_col,
                    event_col='os_event',
                    arm_col='treatment_arm',
                    control_arm=control_arm,
                    treatment_arm=treatment_arm
                )
                
                survival_results.append(stats)
                
                print(f"\n{treatment_arm} vs {control_arm}:")
                print(f"  Median OS: {stats['median_treatment']:.1f} vs {stats['median_control']:.1f} days")
                print(f"  Hazard Ratio: {stats['hazard_ratio']:.3f} "
                      f"(95% CI: {stats['hr_95ci_lower']:.3f}-{stats['hr_95ci_upper']:.3f})")
                print(f"  Log-rank p-value: {stats['logrank_p_value']:.4f}")
                print(f"  Significant: {'✅ YES' if stats['significant'] else '❌ NO'}")
                
            except Exception as e:
                print(f"Warning: Could not analyze {treatment_arm} vs {control_arm}: {e}")
        
        # Save survival results
        if survival_results:
            cleaned_survival = [_sanitize_nan(stats) for stats in survival_results]
            with open(output_dir / "survival_analysis.json", 'w') as f:
                json.dump(cleaned_survival, f, indent=2, default=_json_default, allow_nan=False)
            
            # Plot Kaplan-Meier curves
            try:
                plot_kaplan_meier(
                    data=enrolled_survival,
                    time_col=time_col,
                    event_col='os_event',
                    arm_col='treatment_arm',
                    title=f"{design.trial_name}: Overall Survival",
                    output_path=str(output_dir / "kaplan_meier_os.png"),
                    baseline_data=observed_survival,
                    baseline_label_prefix="Observed",
                    baseline_time_col='os_time',
                    baseline_event_col='os_event',
                    baseline_arm_col='treatment_arm'
                )
            except Exception as e:
                print(f"Warning: Could not create KM plot: {e}")
    
    # Response rate analysis
    if 'predicted_response' in enrolled.columns:
        print("\nResponse Rate Analysis:")
        print("-" * 80)
        
        response_results = []
        
        for treatment_arm in treatment_arms:
            try:
                stats = calculate_response_rate_comparison(
                    data=enrolled,
                    response_col='predicted_response',
                    arm_col='treatment_arm',
                    control_arm=control_arm,
                    treatment_arm=treatment_arm,
                    response_categories=['CR', 'PR']
                )
                
                response_results.append(stats)
                
                print(f"\n{treatment_arm} vs {control_arm}:")
                print(f"  Response Rate: {stats['rate_treatment']*100:.1f}% vs "
                      f"{stats['rate_control']*100:.1f}%")
                print(f"  Risk Difference: {stats['risk_difference']*100:.1f}% "
                      f"(95% CI: {stats['rd_95ci_lower']*100:.1f}%-{stats['rd_95ci_upper']*100:.1f}%)")
                print(f"  Odds Ratio: {stats['odds_ratio']:.2f}")
                print(f"  p-value: {stats['p_value']:.4f}")
                print(f"  Significant: {'✅ YES' if stats['significant'] else '❌ NO'}")
                
            except Exception as e:
                print(f"Warning: Could not analyze response rates: {e}")
        
        if response_results:
            cleaned_response = [_sanitize_nan(stats) for stats in response_results]
            with open(output_dir / "response_analysis.json", 'w') as f:
                json.dump(cleaned_response, f, indent=2, default=_json_default, allow_nan=False)
    
    print()
    
    # Step 7: Sample size calculations
    print("STEP 7: Sample size calculations for future trials...")
    print("-" * 80)
    
    # Example: What sample size would we need for 90% power?
    if survival_results:
        for stats in survival_results:
            hr = stats.get('hazard_ratio')
            if hr is None or not np.isfinite(hr) or hr <= 0:
                print(f"\nSkipping sample size calculation for {stats['treatment_arm']} vs {stats['control_arm']} (invalid HR)")
                continue

            ss = calculate_sample_size_survival(
                median_control=stats['median_control'],
                target_hr=hr,
                alpha=0.05,
                power=0.90,
                allocation_ratio=1.0
            )
            
            print(f"\nSample size for {stats['treatment_arm']} vs {stats['control_arm']}:")
            print(f"  Target HR: {hr:.2f}")
            print(f"  Required N: {ss['n_total']} ({ss['n_control']} control, {ss['n_treatment']} treatment)")
            print(f"  Required events: {ss['n_events_required']}")
    
    print()
    
    # Final summary
    print("=" * 80)
    print("VIRTUAL TRIAL COMPLETE!")
    print("=" * 80)
    print()
    print(f"Output files saved to: {output_dir}")
    print(f"  - Trial design: trial_design.json")
    print(f"  - Patient data: {design.trial_name}_patients.csv")
    print(f"  - Trial summary: trial_summary.csv")
    if survival_results:
        print(f"  - Survival analysis: survival_analysis.json")
        print(f"  - Kaplan-Meier plot: kaplan_meier_os.png")
    if response_results:
        print(f"  - Response analysis: response_analysis.json")
    print()
    print("✅ Phase 5.1-5.3: Trial simulation and analysis complete!")
    print()


if __name__ == "__main__":
    main()
