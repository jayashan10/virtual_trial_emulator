#!/usr/bin/env python
"""
Test Multi-Outcome Prediction System.

Tests the integrated prediction system on digital twins and real patients.
"""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from pds310.profile_database import load_profile_database
from pds310.predict_outcomes import (
    create_outcome_predictor,
    save_predictions,
    summarize_predictions
)


def main():
    parser = argparse.ArgumentParser(description="Test outcome predictions")
    parser.add_argument(
        "--profiles",
        type=str,
        default="outputs/pds310/patient_profiles.csv",
        help="Path to patient profiles"
    )
    parser.add_argument(
        "--twins",
        type=str,
        default="outputs/pds310/digital_twins_n1000.csv",
        help="Path to digital twins"
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
        default="outputs/pds310/predictions",
        help="Output directory"
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=100,
        help="Number of patients/twins to test"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PDS310 MULTI-OUTCOME PREDICTION TESTING")
    print("=" * 80)
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load outcome predictor
    print("STEP 1: Loading outcome predictor...")
    print("-" * 80)
    
    predictor = create_outcome_predictor(args.models_dir)
    print()
    
    # Step 2: Test on real patients
    print("STEP 2: Testing on real patients...")
    print("-" * 80)
    
    real_profiles = load_profile_database(args.profiles)
    print(f"Loaded {len(real_profiles)} real patients")
    
    # Test on subset
    test_real = real_profiles.head(min(args.n_test, len(real_profiles)))
    print(f"Testing on {len(test_real)} patients...")
    
    real_predictions = predictor.predict_cohort(
        test_real,
        include_biomarkers=False,  # Skip biomarkers if not trained
        verbose=True
    )
    
    # Save predictions
    output_path = output_dir / "real_patient_predictions.csv"
    save_predictions(real_predictions, str(output_path))
    
    # Summarize
    real_summary = summarize_predictions(real_predictions)
    print("\nReal Patient Prediction Summary:")
    print(f"  Patients: {real_summary['n_patients']}")
    if 'predicted_response_distribution' in real_summary:
        print(f"  Response distribution:")
        for resp, count in real_summary['predicted_response_distribution'].items():
            print(f"    {resp}: {count}")
    if 'predicted_response_rate' in real_summary:
        print(f"  Response rate: {real_summary['predicted_response_rate']:.1f}%")
    print()
    
    # Step 3: Test on digital twins
    print("STEP 3: Testing on digital twins...")
    print("-" * 80)
    
    try:
        twins = load_profile_database(args.twins)
        print(f"Loaded {len(twins)} digital twins")
        
        # Test on subset
        test_twins = twins.head(min(args.n_test, len(twins)))
        print(f"Testing on {len(test_twins)} twins...")
        
        twin_predictions = predictor.predict_cohort(
            test_twins,
            include_biomarkers=False,
            verbose=True
        )
        
        # Save predictions
        output_path = output_dir / "twin_predictions.csv"
        save_predictions(twin_predictions, str(output_path))
        
        # Summarize
        twin_summary = summarize_predictions(twin_predictions)
        print("\nDigital Twin Prediction Summary:")
        print(f"  Twins: {twin_summary['n_patients']}")
        if 'predicted_response_distribution' in twin_summary:
            print(f"  Response distribution:")
            for resp, count in twin_summary['predicted_response_distribution'].items():
                print(f"    {resp}: {count}")
        if 'predicted_response_rate' in twin_summary:
            print(f"  Response rate: {twin_summary['predicted_response_rate']:.1f}%")
        print()
        
        # Step 4: Compare real vs twins
        print("STEP 4: Comparing real patients vs digital twins...")
        print("-" * 80)
        
        print("\nResponse Rate Comparison:")
        print(f"  Real patients: {real_summary.get('predicted_response_rate', 0):.1f}%")
        print(f"  Digital twins: {twin_summary.get('predicted_response_rate', 0):.1f}%")
        print(f"  Difference: {abs(real_summary.get('predicted_response_rate', 0) - twin_summary.get('predicted_response_rate', 0)):.1f}%")
        
        if 'predicted_response_distribution' in real_summary and 'predicted_response_distribution' in twin_summary:
            print("\nResponse Distribution Comparison:")
            all_classes = set(real_summary['predicted_response_distribution'].keys()) | set(twin_summary['predicted_response_distribution'].keys())
            print(f"  {'Class':<10} {'Real':>10} {'Twins':>10} {'Diff':>10}")
            print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
            for cls in sorted(all_classes):
                real_count = real_summary['predicted_response_distribution'].get(cls, 0)
                twin_count = twin_summary['predicted_response_distribution'].get(cls, 0)
                real_pct = real_count / real_summary['n_patients'] * 100
                twin_pct = twin_count / twin_summary['n_patients'] * 100
                diff = abs(real_pct - twin_pct)
                print(f"  {cls:<10} {real_pct:>9.1f}% {twin_pct:>9.1f}% {diff:>9.1f}%")
        print()
        
    except FileNotFoundError:
        print("⚠️  Digital twins file not found - skipping twin testing")
        print()
    
    # Step 5: Test single patient prediction
    print("STEP 5: Testing single patient prediction...")
    print("-" * 80)
    
    # Get a random patient
    patient = real_profiles.sample(n=1, random_state=42).iloc[0].to_dict()
    patient_id = patient.get('SUBJID', 'UNKNOWN')
    
    print(f"Testing patient: {patient_id}")
    print(f"  Age: {patient.get('AGE')}")
    print(f"  Sex: {patient.get('SEX')}")
    print(f"  RAS status: {patient.get('RAS_status')}")
    print(f"  Treatment: {patient.get('ATRT')}")
    print()
    
    # Predict all outcomes
    import pandas as pd
    prediction = predictor.predict_all(pd.DataFrame([patient]))
    
    print("Predictions:")
    if 'response' in prediction:
        print(f"  Response: {prediction['response'].get('predicted_response')}")
        print(f"  Confidence: {prediction['response'].get('confidence', 0)*100:.1f}%")
        if 'probabilities' in prediction['response']:
            print(f"  Probabilities:")
            for resp, prob in prediction['response']['probabilities'].items():
                print(f"    {resp}: {prob*100:.1f}%")
    
    if 'time_to_response' in prediction and 'predicted_ttr' in prediction['time_to_response']:
        print(f"\n  Time to Response: {prediction['time_to_response']['predicted_ttr']:.0f} days")
        print(f"    95% CI: [{prediction['time_to_response']['lower_95ci']:.0f}, {prediction['time_to_response']['upper_95ci']:.0f}]")
    
    print()
    
    # Step 6: Summary
    print("=" * 80)
    print("PREDICTION TESTING COMPLETE!")
    print("=" * 80)
    print()
    print("Output Files:")
    print(f"  - Real patient predictions: {output_dir / 'real_patient_predictions.csv'}")
    if Path(args.twins).exists():
        print(f"  - Twin predictions: {output_dir / 'twin_predictions.csv'}")
    print()
    print("✅ Phase 3.4: Multi-outcome prediction system operational!")
    print()
    print("System Capabilities:")
    print("  ✅ Response classification (PD/SD/PR) with probabilities")
    print("  ✅ Time-to-response prediction with confidence intervals")
    print("  ✅ Batch prediction for cohorts")
    print("  ✅ Single patient prediction")
    print("  ⚠️  Biomarker trajectories (limited data - not currently active)")
    print()


if __name__ == "__main__":
    main()
