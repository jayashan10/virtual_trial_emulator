#!/usr/bin/env python
"""
Train All Outcome Prediction Models for PDS310.

This script trains response classification, time-to-response regression,
and prepares for biomarker trajectory modeling.
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pds310.profile_database import load_profile_database
from pds310.model_response import (
    prepare_response_data,
    train_response_classifier,
    save_response_model,
    plot_feature_importance as plot_response_importance,
    plot_confusion_matrix
)
from pds310.model_ttr import (
    prepare_ttr_data,
    train_ttr_model,
    save_ttr_model,
    plot_ttr_predictions,
    plot_ttr_feature_importance,
    prepare_features_for_model as prepare_ttr_features_for_model,
)
from pds310.model_survival import (
    prepare_os_data,
    train_os_model,
    save_os_model,
)


def main():
    parser = argparse.ArgumentParser(description="Train outcome prediction models")
    parser.add_argument(
        "--profiles",
        type=str,
        default="outputs/pds310/patient_profiles.csv",
        help="Path to patient profiles database"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/pds310/models",
        help="Output directory for trained models"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        choices=["xgboost", "rf", "gb", "logreg", "elasticnet"],
        help="Legacy flag to apply the same model type across tasks"
    )
    parser.add_argument(
        "--response_model",
        type=str,
        default=None,
        choices=["logreg", "rf", "gb", "xgboost"],
        help="Response classifier model type (default: rf)"
    )
    parser.add_argument(
        "--ttr_model",
        type=str,
        default=None,
        choices=["elasticnet", "rf", "gb", "xgboost"],
        help="Time-to-response model type (default: elasticnet)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()

    response_model_type = args.response_model or args.model_type or "rf"
    ttr_model_type = args.ttr_model or args.model_type or "rf"
    biomarker_model_type = args.model_type or "rf"
    
    print("=" * 80)
    print("PDS310 OUTCOME PREDICTION MODEL TRAINING")
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
    
    # Step 2: Train Response Classification Model
    print("STEP 2: Training Response Classification Model...")
    print("-" * 80)
    
    try:
        X_resp, y_resp, y_resp_simplified = prepare_response_data(profiles)
        print(f"Response data: {len(X_resp)} patients with response data")
        print(f"Response distribution:")
        print(y_resp.value_counts())
        print()
        print(f"Using response model type: {response_model_type}")
        print()
        
        # Try original response categories first
        if len(y_resp.unique()) >= 3:
            print("Training multi-class response model...")
            response_results = train_response_classifier(
                X_resp,
                y_resp,
                model_type=response_model_type,
                random_state=args.seed
            )
            training_labels = y_resp
        else:
            print("Training binary response model (Responder/Non-responder)...")
            response_results = train_response_classifier(
                X_resp,
                y_resp_simplified,
                model_type=response_model_type,
                random_state=args.seed
            )
            training_labels = y_resp_simplified
        
        print()
        print("Response Model Performance:")
        print(f"  CV Accuracy: {response_results['cv_mean']:.3f} ± {response_results['cv_std']:.3f}")
        print(f"  CV Balanced Accuracy: {response_results['cv_balanced_accuracy_mean']:.3f} ± {response_results['cv_balanced_accuracy_std']:.3f}")
        print(f"  CV Macro F1: {response_results['cv_macro_f1_mean']:.3f} ± {response_results['cv_macro_f1_std']:.3f}")
        print(f"  Train Accuracy: {response_results['train_accuracy']:.3f}")
        print(f"  Train Balanced Accuracy: {response_results['train_balanced_accuracy']:.3f}")
        print(f"  Train Macro F1: {response_results['train_macro_f1']:.3f}")
        print(f"  Train Weighted F1: {response_results['train_f1']:.3f}")
        if 'roc_auc' in response_results:
            print(f"  ROC-AUC: {response_results['roc_auc']:.3f}")
        baseline_accuracy = training_labels.value_counts(normalize=True).max()
        print(f"  Majority-class baseline accuracy: {baseline_accuracy:.3f}")
        if response_results['cv_mean'] <= baseline_accuracy:
            print("⚠️  CV accuracy does not exceed majority-class baseline.")
        print()
        
        # Save model
        model_path = output_dir / "response_model.joblib"
        save_response_model(response_results, str(model_path))
        
        # Plot feature importance
        if response_results['feature_importance'] is not None:
            plot_path = output_dir / "response_feature_importance.png"
            plot_response_importance(
                response_results['feature_importance'],
                top_n=20,
                output_path=str(plot_path)
            )
        
        # Plot confusion matrix
        conf_matrix_path = output_dir / "response_confusion_matrix.png"
        plot_confusion_matrix(
            response_results['confusion_matrix'],
            response_results['classes'],
            output_path=str(conf_matrix_path)
        )
        
        print("✅ Response model training complete!")
        print()
        
    except Exception as e:
        print(f"❌ Response model training failed: {e}")
        print()
    
    # Step 3: Train Time-to-Response Model
    print("STEP 3: Training Time-to-Response Regression Model...")
    print("-" * 80)
    
    try:
        X_ttr, y_ttr = prepare_ttr_data(profiles)
        print(f"TTR data: {len(X_ttr)} responders")
        print(f"TTR statistics:")
        print(f"  Mean: {y_ttr.mean():.1f} days")
        print(f"  Median: {y_ttr.median():.1f} days")
        print(f"  Range: {y_ttr.min():.0f} - {y_ttr.max():.0f} days")
        print()
        print(f"Using TTR model type: {ttr_model_type}")
        print()
        
        ttr_results = train_ttr_model(
            X_ttr,
            y_ttr,
            model_type=ttr_model_type,
            cv_folds=min(3, max(2, len(X_ttr))),
            random_state=args.seed
        )
        
        print()
        print("TTR Model Performance:")
        print(f"  CV R²: {ttr_results['cv_r2_mean']:.3f} ± {ttr_results['cv_r2_std']:.3f}")
        print(f"  CV MAE: {ttr_results['cv_mae_mean']:.1f} ± {ttr_results['cv_mae_std']:.1f} days")
        print(f"  Train R²: {ttr_results['train_r2']:.3f}")
        print(f"  Train MAE: {ttr_results['train_mae']:.1f} days")
        print(f"  Train RMSE: {ttr_results['train_rmse']:.1f} days")
        print(f"  Train MAPE: {ttr_results['train_mape']:.1f}%")
        print()
        
        # Check CAMP target (R² ≥ 0.80)
        if ttr_results['cv_r2_mean'] >= 0.80:
            print("🎯 CAMP performance target achieved (R² ≥ 0.80)!")
        elif ttr_results['cv_r2_mean'] >= 0.60:
            print("⚠️  Good performance but below CAMP target (R² ≥ 0.80)")
        else:
            print("⚠️  Performance below target - consider more features or samples")
        print()
        
        # Save model
        model_path = output_dir / "ttr_model.joblib"
        save_ttr_model(ttr_results, str(model_path))
        
        # Plot predictions
        X_ttr_prepared = prepare_ttr_features_for_model(ttr_results, X_ttr)
        y_pred = ttr_results['model'].predict(X_ttr_prepared)
        plot_path = output_dir / "ttr_predictions.png"
        plot_ttr_predictions(
            y_ttr.values,
            y_pred,
            output_path=str(plot_path)
        )
        
        # Plot feature importance
        if ttr_results['feature_importance'] is not None:
            plot_path = output_dir / "ttr_feature_importance.png"
            plot_ttr_feature_importance(
                ttr_results['feature_importance'],
                top_n=20,
                output_path=str(plot_path)
            )
        
        print("✅ TTR model training complete!")
        print()
        
    except Exception as e:
        print(f"❌ TTR model training failed: {e}")
        import traceback
        traceback.print_exc()
        print()
    
    # Step 4: Train Overall Survival Model
    print("STEP 4: Training Overall Survival Model...")
    print("-" * 80)

    try:
        X_os, durations, events = prepare_os_data(profiles)
        print(f"OS data: {len(X_os)} patients with survival outcomes")
        print(f"  Median duration: {durations.median():.1f} days")
        print(f"  Event rate: {events.mean() * 100:.1f}%")
        print()

        os_results = train_os_model(
            X_os,
            durations,
            events,
        )

        print("OS Model Performance:")
        print(f"  Train concordance: {os_results['train_concordance']:.3f}")
        print(f"  CV concordance: {os_results['cv_cindex_mean']:.3f} ± {os_results['cv_cindex_std']:.3f}")
        print()

        os_path = output_dir / "os_model.joblib"
        save_os_model(os_results, str(os_path))
        print(f"✅ OS model saved to {os_path}")
        print()

    except Exception as e:
        print(f"❌ OS model training failed: {e}")
        import traceback
        traceback.print_exc()
        print()
    
    # Step 5: Train Biomarker Trajectory Models
    print("STEP 5: Training Biomarker Trajectory Models...")
    print("-" * 80)
    
    try:
        from pds310.model_biomarkers import (
            train_biomarker_trajectory_models,
            save_biomarker_models,
            plot_biomarker_predictions
        )
        from pds310.io import load_adam_tables
        import yaml
        
        # Load ADaM tables for longitudinal data
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        tables = load_adam_tables(config["data_root"], verbose=False)
        adlb = tables.get("adlb")
        
        if adlb is not None:
            # Train models for key biomarkers (use full names from ADLB)
            biomarker_models = train_biomarker_trajectory_models(
                profile_db=profiles,
                adlb=adlb,
                biomarkers=["Lactate Dehydrogenase", "Hemoglobin"],  # Full names
                timepoints=[56, 112],  # Weeks 8 and 16
                model_type=biomarker_model_type,
                random_state=args.seed
            )
            
            if biomarker_models:
                # Save models
                model_path = output_dir / "biomarker_models.joblib"
                save_biomarker_models(biomarker_models, str(model_path))
                
                # Plot predictions for each biomarker/timepoint
                for biomarker, timepoint_models in biomarker_models.items():
                    for timepoint, model_data in timepoint_models.items():
                        # Get predictions
                        model = model_data['model']
                        feature_names = model_data['feature_names']
                        
                        # Prepare features for all patients
                        from pds310.model_biomarkers import prepare_biomarker_trajectories
                        X, y, _ = prepare_biomarker_trajectories(
                            profiles, adlb, biomarker, [timepoint]
                        )
                        
                        # Get predictions
                        X_features = X[feature_names].copy()
                        categorical_cols = X_features.select_dtypes(include=['object', 'string']).columns
                        for col in categorical_cols:
                            X_features[col] = pd.Categorical(X_features[col]).codes
                        numeric_cols = X_features.select_dtypes(include=[np.number]).columns
                        X_features[numeric_cols] = X_features[numeric_cols].fillna(X_features[numeric_cols].median())
                        
                        y_pred = model.predict(X_features)
                        y_true = y[f"{biomarker}_day{timepoint}"].values
                        
                        # Plot
                        plot_path = output_dir / f"{biomarker}_day{timepoint}_predictions.png"
                        plot_biomarker_predictions(
                            y_true, y_pred, biomarker, timepoint,
                            output_path=str(plot_path)
                        )
                
                print("✅ Biomarker models training complete!")
            else:
                print("⚠️  No biomarker models trained")
        else:
            print("⚠️  ADLB table not available - skipping biomarker models")
        print()
        
    except Exception as e:
        print(f"❌ Biomarker model training failed: {e}")
        import traceback
        traceback.print_exc()
        print()
    
    # Step 5: Summary
    print("=" * 80)
    print("MODEL TRAINING COMPLETE!")
    print("=" * 80)
    print()
    print("Trained Models:")
    print(f"  1. Response Classification: {output_dir / 'response_model.joblib'}")
    print(f"  2. Time-to-Response Regression: {output_dir / 'ttr_model.joblib'}")
    print(f"  3. Biomarker Trajectories: {output_dir / 'biomarker_models.joblib'}")
    print()
    print("Visualizations:")
    print(f"  - Response feature importance & confusion matrix")
    print(f"  - TTR predictions & feature importance")
    print(f"  - Biomarker predictions (LDH, HGB at days 56, 112)")
    print()
    print("Next Steps:")
    print("  - Phase 3.4: Test multi-outcome prediction system")
    print("  - Phase 4: Add uncertainty quantification & validation")
    print("  - Phase 5: Virtual trial simulation")
    print()


if __name__ == "__main__":
    main()
