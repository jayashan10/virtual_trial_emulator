#!/usr/bin/env python
"""
Comprehensive Validation for PDS310 Models.

Runs holdout validation, calibration assessment, and uncertainty quantification.
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from pds310.profile_database import load_profile_database
from pds310.validation import (
    create_holdout_split,
    validate_response_model,
    validate_ttr_model,
    save_validation_results,
    plot_validation_results,
    compare_train_test_performance
)
from pds310.calibration import (
    assess_response_calibration,
    assess_regression_calibration,
    plot_calibration_curve,
    calculate_calibration_metrics_summary
)
from pds310.uncertainty import bootstrap_model_performance
from pds310.model_response import (
    prepare_response_data,
    train_response_classifier,
    prepare_features_for_model as prepare_response_features,
)
from pds310.model_ttr import (
    prepare_ttr_data,
    train_ttr_model,
    prepare_features_for_model as prepare_ttr_features,
)


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive validation")
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
        default="outputs/pds310/validation",
        help="Output directory"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test set fraction"
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=100,
        help="Number of bootstrap samples"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="rf",
        choices=["rf", "gb", "xgboost"],
        help="Estimator family for response/TTR models",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PDS310 COMPREHENSIVE MODEL VALIDATION")
    print("=" * 80)
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load data and create holdout split
    print("STEP 1: Creating holdout train/test split...")
    print("-" * 80)
    
    profiles = load_profile_database(args.profiles)
    train_profiles, test_profiles = create_holdout_split(
        profiles,
        test_size=args.test_size,
        stratify_col="RAS_status",
        random_state=args.seed
    )
    
    # Save splits
    train_profiles.to_csv(output_dir / "train_profiles.csv", index=False)
    test_profiles.to_csv(output_dir / "test_profiles.csv", index=False)
    print()
    
    # Step 2: Validate Response Model
    print("STEP 2: Validating Response Classification Model...")
    print("-" * 80)
    
    try:
        X_train_resp, y_train_resp, y_train_resp_simple = prepare_response_data(train_profiles)
        if len(y_train_resp.unique()) >= 3:
            response_model = train_response_classifier(
                X_train_resp,
                y_train_resp,
                model_type=args.model_type,
                random_state=args.seed,
            )
        else:
            response_model = train_response_classifier(
                X_train_resp,
                y_train_resp_simple,
                model_type=args.model_type,
                random_state=args.seed,
            )
        
        response_validation = validate_response_model(
            response_model,
            test_profiles,
            model_type="response",
        )
        
        # Save results
        save_validation_results(
            response_validation,
            str(output_dir / "response_validation.json")
        )
        
        # Plot results
        plot_validation_results(
            response_validation,
            model_type="response",
            output_path=str(output_dir / "response_validation.png")
        )
        
        # Performance comparison
        comparison = compare_train_test_performance(
            response_model,
            response_validation,
            model_type="response"
        )
        print("\nPerformance Comparison:")
        print(comparison.to_string(index=False))
        comparison.to_csv(output_dir / "response_performance_comparison.csv", index=False)
        
        # Calibration assessment
        print("\n" + "-" * 80)
        print("Assessing Response Model Calibration...")
        print("-" * 80)
        
        # Get predictions for calibration
        X_test, y_test, _ = prepare_response_data(test_profiles)
        X_test_features = prepare_response_features(response_model, X_test)
        y_pred_proba = response_model["model"].predict_proba(X_test_features)
        
        # Assess calibration
        response_calibration = assess_response_calibration(
            y_test.values,
            y_pred_proba,
            n_bins=5  # Use fewer bins for small sample
        )
        
        # Plot calibration
        plot_calibration_curve(
            response_calibration,
            model_type="response",
            output_path=str(output_dir / "response_calibration.png")
        )
        
        # Calibration summary
        cal_summary = calculate_calibration_metrics_summary(
            response_calibration,
            model_type="response"
        )
        print("\nCalibration Metrics:")
        print(cal_summary.to_string(index=False))
        cal_summary.to_csv(output_dir / "response_calibration_metrics.csv", index=False)

        preds_meta = response_validation.get("predictions", {})
        indices = preds_meta.get("indices")
        arm_col = None
        if 'ATRT' in test_profiles.columns:
            arm_col = 'ATRT'
        elif 'TRT' in test_profiles.columns:
            arm_col = 'TRT'

        if indices is not None and arm_col is not None:
            arms = test_profiles.loc[indices, arm_col].fillna('UNKNOWN').to_numpy()
            y_true = np.array(preds_meta.get('y_true', []))
            y_pred = np.array(preds_meta.get('y_pred', []))
            y_pred_proba = np.array(preds_meta.get('y_pred_proba', []))
            class_labels = response_validation.get('classes', [])
            class_to_idx = {cls: idx for idx, cls in enumerate(class_labels)}
            responder_idx = [class_to_idx[c] for c in ('CR', 'PR') if c in class_to_idx]

            per_arm_rows = []
            for arm in np.unique(arms):
                mask = arms == arm
                if mask.sum() == 0:
                    continue
                actual_rate = float(np.isin(y_true[mask], ['CR', 'PR']).mean()) if len(y_true) else float('nan')
                predicted_class_rate = float(np.isin(y_pred[mask], ['CR', 'PR']).mean()) if len(y_pred) else float('nan')
                if responder_idx and y_pred_proba.size:
                    responder_probs = y_pred_proba[mask][:, responder_idx].sum(axis=1)
                    predicted_prob_rate = float(responder_probs.mean())
                else:
                    predicted_prob_rate = float('nan')
                per_arm_rows.append({
                    'arm': arm,
                    'n': int(mask.sum()),
                    'actual_responder_rate_pct': actual_rate * 100 if not np.isnan(actual_rate) else float('nan'),
                    'predicted_class_responder_rate_pct': predicted_class_rate * 100 if not np.isnan(predicted_class_rate) else float('nan'),
                    'predicted_responder_probability_pct': predicted_prob_rate * 100 if not np.isnan(predicted_prob_rate) else float('nan'),
                })

            if per_arm_rows:
                per_arm_df = pd.DataFrame(per_arm_rows)
                print("\nPer-Arm Response Metrics (%):")
                print(per_arm_df.to_string(index=False, float_format=lambda v: f"{v:.1f}" if isinstance(v, float) and not np.isnan(v) else str(v)))
                per_arm_path = output_dir / "response_per_arm_metrics.csv"
                per_arm_df.to_csv(per_arm_path, index=False)
                print(f"Per-arm metrics saved to: {per_arm_path}")

                if len(per_arm_df) >= 2:
                    ref, other = per_arm_df.iloc[0], per_arm_df.iloc[1]
                    uplift_pred = ref['predicted_responder_probability_pct'] - other['predicted_responder_probability_pct']
                    uplift_actual = ref['actual_responder_rate_pct'] - other['actual_responder_rate_pct']
                    print(
                        f"  Predicted responder probability uplift ({ref['arm']} vs {other['arm']}): {uplift_pred:.1f} pp"
                    )
                    print(
                        f"  Observed responder uplift ({ref['arm']} vs {other['arm']}): {uplift_actual:.1f} pp"
                    )
                print()
            else:
                print("\nPer-arm metrics not available (single arm present).")
        else:
            print("\nPer-arm metrics skipped (treatment labels unavailable).")
        
        print("\n✅ Response model validation complete!")
        
    except Exception as e:
        print(f"❌ Response model validation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # Step 3: Validate TTR Model
    print("STEP 3: Validating Time-to-Response Model...")
    print("-" * 80)
    
    try:
        X_train_ttr, y_train_ttr = prepare_ttr_data(train_profiles)
        ttr_model = train_ttr_model(
            X_train_ttr,
            y_train_ttr,
            model_type=args.model_type,
            cv_folds=min(3, max(2, len(X_train_ttr))),
            random_state=args.seed,
        )
        
        ttr_validation = validate_ttr_model(
            ttr_model,
            test_profiles
        )
        
        if 'error' not in ttr_validation:
            # Save results
            save_validation_results(
                ttr_validation,
                str(output_dir / "ttr_validation.json")
            )
            
            # Plot results
            plot_validation_results(
                ttr_validation,
                model_type="ttr",
                output_path=str(output_dir / "ttr_validation.png")
            )
            
            # Performance comparison
            comparison = compare_train_test_performance(
                ttr_model,
                ttr_validation,
                model_type="ttr"
            )
            print("\nPerformance Comparison:")
            print(comparison.to_string(index=False))
            comparison.to_csv(output_dir / "ttr_performance_comparison.csv", index=False)
            
            # Calibration assessment
            print("\n" + "-" * 80)
            print("Assessing TTR Model Calibration...")
            print("-" * 80)
            
            y_true = np.array(ttr_validation['predictions']['y_true'])
            y_pred = np.array(ttr_validation['predictions']['y_pred'])
            
            ttr_calibration = assess_regression_calibration(
                y_true,
                y_pred,
                n_bins=5
            )
            
            plot_calibration_curve(
                ttr_calibration,
                model_type="regression",
                output_path=str(output_dir / "ttr_calibration.png")
            )
            
            print(f"\nCalibration Slope: {ttr_calibration['calibration_slope']:.3f}")
            print(f"Calibration R²: {ttr_calibration['calibration_r2']:.3f}")
            
            # Bootstrap uncertainty
            print("\n" + "-" * 80)
            print("Bootstrap Uncertainty Quantification...")
            print("-" * 80)
            
            from sklearn.metrics import mean_absolute_error
            
            X_test, y_test = prepare_ttr_data(test_profiles)

            bootstrap_mae = bootstrap_model_performance(
                ttr_model,
                X_test,
                y_test.values,
                mean_absolute_error,
                n_bootstrap=args.n_bootstrap,
                random_state=args.seed,
                prepare_fn=prepare_ttr_features,
            )
            
            print(f"\nBootstrap MAE:")
            print(f"  Mean: {bootstrap_mae['metric_mean']:.1f} days")
            print(f"  95% CI: [{bootstrap_mae['lower_ci']:.1f}, {bootstrap_mae['upper_ci']:.1f}]")
            print(f"  Std Error: {bootstrap_mae['std_err']:.1f}")
            
            print("\n✅ TTR model validation complete!")
        else:
            print(f"⚠️  {ttr_validation['error']}")
        
    except Exception as e:
        print(f"❌ TTR model validation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # Step 4: Summary
    print("=" * 80)
    print("VALIDATION COMPLETE!")
    print("=" * 80)
    print()
    print("Output Files:")
    print(f"  - Train/Test Splits: {output_dir}")
    print(f"  - Validation Results: response_validation.json, ttr_validation.json")
    print(f"  - Calibration Plots: *_calibration.png")
    print(f"  - Performance Comparisons: *_performance_comparison.csv")
    print()
    print("✅ Phase 4.1: Holdout validation complete")
    print("✅ Phase 4.2: Bootstrap uncertainty complete")
    print("✅ Phase 4.3: Calibration assessment complete")
    print()


if __name__ == "__main__":
    main()
