"""
Holdout Validation Framework for PDS310.

Implements rigorous validation with train/test splits, cross-validation,
and performance metrics for all outcome models.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    r2_score, mean_absolute_error, mean_squared_error,
    roc_auc_score, roc_curve
)
from pathlib import Path
import json


def create_holdout_split(
    profile_db: pd.DataFrame,
    test_size: float = 0.2,
    stratify_col: Optional[str] = "RAS_status",
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/test split with stratification.
    
    Args:
        profile_db: Patient profile database
        test_size: Fraction for test set (default: 0.2)
        stratify_col: Column to stratify on (default: RAS_status)
        random_state: Random seed
    
    Returns:
        (train_df, test_df)
    """
    if stratify_col and stratify_col in profile_db.columns:
        stratify = profile_db[stratify_col]
    else:
        stratify = None
    
    train_df, test_df = train_test_split(
        profile_db,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state
    )
    
    print(f"Created holdout split:")
    print(f"  Training set: {len(train_df)} patients ({len(train_df)/len(profile_db)*100:.1f}%)")
    print(f"  Test set: {len(test_df)} patients ({len(test_df)/len(profile_db)*100:.1f}%)")
    
    if stratify_col and stratify_col in profile_db.columns:
        print(f"\n  Stratification by {stratify_col}:")
        print(f"    Train: {train_df[stratify_col].value_counts().to_dict()}")
        print(f"    Test: {test_df[stratify_col].value_counts().to_dict()}")
    
    return train_df, test_df


def validate_response_model(
    model_results: Dict[str, Any],
    test_profiles: pd.DataFrame,
    model_type: str = "response"
) -> Dict[str, Any]:
    """
    Validate response classification model on holdout test set.
    
    Args:
        model_results: Trained model results from train_response_classifier
        test_profiles: Test set patient profiles
        model_type: Type of model being validated
    
    Returns:
        Dictionary with validation metrics
    """
    from .model_response import (
        prepare_response_data,
        prepare_features_for_model as prepare_response_features,
    )
    
    # Prepare test data
    X_test, y_test, y_test_simplified = prepare_response_data(
        test_profiles,
        exclude_outcomes=True
    )
    
    if len(X_test) == 0:
        return {'error': 'No test data with response outcomes'}
    
    print(f"\nValidating {model_type} model on {len(X_test)} test patients...")
    print(f"Test response distribution: {y_test.value_counts().to_dict()}")
    
    model = model_results["model"]
    classes = model.named_steps["model"].classes_

    X_test_features = prepare_response_features(model_results, X_test)
    
    # Predict
    y_pred = model.predict(X_test_features)
    y_pred_proba = model.predict_proba(X_test_features)
    
    # Calculate metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=classes)
    
    # Classification report
    class_report = classification_report(
        y_test, y_pred, labels=classes, output_dict=True, zero_division=0
    )
    
    # ROC-AUC if possible
    try:
        if len(classes) == 2:
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            from sklearn.preprocessing import label_binarize
            y_test_bin = label_binarize(y_test, classes=classes)
            if y_test_bin.shape[1] > 1:
                roc_auc = roc_auc_score(
                    y_test_bin, y_pred_proba, 
                    average='weighted', multi_class='ovr'
                )
            else:
                roc_auc = None
    except Exception as e:
        print(f"Could not calculate ROC-AUC: {e}")
        roc_auc = None
    
    validation_results = {
        'n_test': len(X_test),
        'test_accuracy': float(test_accuracy),
        'test_precision': float(precision),
        'test_recall': float(recall),
        'test_f1': float(f1),
        'test_roc_auc': float(roc_auc) if roc_auc is not None else None,
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report,
        'train_accuracy': model_results.get('train_accuracy'),
        'cv_accuracy': model_results.get('cv_mean'),
        'classes': classes.tolist(),
        'predictions': {
            'y_true': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'y_pred_proba': y_pred_proba.tolist(),
            'indices': X_test.index.tolist(),
        }
    }
    
    # Print summary
    print(f"\nValidation Results:")
    print(f"  Test Accuracy: {test_accuracy:.3f}")
    print(f"  Test Precision: {precision:.3f}")
    print(f"  Test Recall: {recall:.3f}")
    print(f"  Test F1: {f1:.3f}")
    if roc_auc:
        print(f"  Test ROC-AUC: {roc_auc:.3f}")
    print(f"\n  Comparison:")
    print(f"    Train Accuracy: {model_results.get('train_accuracy', 0):.3f}")
    print(f"    CV Accuracy: {model_results.get('cv_mean', 0):.3f}")
    print(f"    Test Accuracy: {test_accuracy:.3f}")
    
    return validation_results


def validate_ttr_model(
    model_results: Dict[str, Any],
    test_profiles: pd.DataFrame
) -> Dict[str, Any]:
    """
    Validate time-to-response regression model on holdout test set.
    
    Args:
        model_results: Trained model results from train_ttr_model
        test_profiles: Test set patient profiles
    
    Returns:
        Dictionary with validation metrics
    """
    from .model_ttr import (
        prepare_ttr_data,
        prepare_features_for_model as prepare_ttr_features,
    )
    
    # Prepare test data
    try:
        X_test, y_test = prepare_ttr_data(test_profiles, exclude_outcomes=True)
    except ValueError as e:
        return {'error': str(e)}
    
    if len(X_test) == 0:
        return {'error': 'No test data with TTR outcomes'}
    
    print(f"\nValidating TTR model on {len(X_test)} test responders...")
    print(f"Test TTR: mean={y_test.mean():.1f}, median={y_test.median():.1f} days")
    
    model = model_results["model"]
    X_test_features = prepare_ttr_features(model_results, X_test)
    y_pred = model.predict(X_test_features)
    
    # Calculate metrics
    test_r2 = r2_score(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    validation_results = {
        'n_test': len(X_test),
        'test_r2': float(test_r2),
        'test_mae': float(test_mae),
        'test_rmse': float(test_rmse),
        'test_mape': float(test_mape),
        'train_r2': model_results.get('train_r2'),
        'cv_r2': model_results.get('cv_r2_mean'),
        'predictions': {
            'y_true': y_test.tolist(),
            'y_pred': y_pred.tolist(),
        }
    }
    
    # Print summary
    print(f"\nValidation Results:")
    print(f"  Test R²: {test_r2:.3f}")
    print(f"  Test MAE: {test_mae:.1f} days")
    print(f"  Test RMSE: {test_rmse:.1f} days")
    print(f"  Test MAPE: {test_mape:.1f}%")
    print(f"\n  Comparison:")
    print(f"    Train R²: {model_results.get('train_r2', 0):.3f}")
    print(f"    CV R²: {model_results.get('cv_r2_mean', 0):.3f}")
    print(f"    Test R²: {test_r2:.3f}")
    
    return validation_results


def save_validation_results(
    results: Dict[str, Any],
    output_path: str
):
    """Save validation results to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    results_json = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results_json[key] = value.tolist()
        elif isinstance(value, dict):
            results_json[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in value.items()
            }
        else:
            results_json[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\nValidation results saved to: {output_path}")


def plot_validation_results(
    validation_results: Dict[str, Any],
    model_type: str = "response",
    output_path: Optional[str] = None
):
    """
    Plot validation results.
    
    Args:
        validation_results: Results from validate_*_model
        model_type: "response" or "ttr"
        output_path: Path to save plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if model_type == "response":
        # Plot confusion matrix
        conf_matrix = np.array(validation_results['confusion_matrix'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=validation_results.get('classes', ['PD', 'PR', 'SD']),
            yticklabels=validation_results.get('classes', ['PD', 'PR', 'SD'])
        )
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Holdout Test Set Confusion Matrix\n(n={validation_results["n_test"]})')
        
    elif model_type == "ttr":
        # Plot actual vs predicted
        y_true = np.array(validation_results['predictions']['y_true'])
        y_pred = np.array(validation_results['predictions']['y_pred'])
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Scatter plot
        ax = axes[0]
        ax.scatter(y_true, y_pred, alpha=0.6, color='steelblue')
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
        
        ax.set_xlabel('Actual TTR (days)')
        ax.set_ylabel('Predicted TTR (days)')
        ax.set_title(f'Holdout Test Set: TTR Predictions\n'
                    f'R² = {validation_results["test_r2"]:.3f}, '
                    f'MAE = {validation_results["test_mae"]:.1f} days')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Residuals
        ax = axes[1]
        residuals = y_true - y_pred
        ax.scatter(y_pred, residuals, alpha=0.6, color='coral')
        ax.axhline(0, color='r', linestyle='--')
        ax.set_xlabel('Predicted TTR (days)')
        ax.set_ylabel('Residuals (days)')
        ax.set_title('Residual Plot')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Validation plot saved to: {output_path}")
    
    plt.close()


def compare_train_test_performance(
    train_results: Dict[str, Any],
    validation_results: Dict[str, Any],
    model_type: str = "response"
) -> pd.DataFrame:
    """
    Compare training vs test performance.
    
    Returns:
        DataFrame with metrics comparison
    """
    if model_type == "response":
        comparison = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Training': [
                train_results.get('train_accuracy', np.nan),
                train_results.get('train_precision', np.nan),
                train_results.get('train_recall', np.nan),
                train_results.get('train_f1', np.nan),
            ],
            'CV': [
                train_results.get('cv_mean', np.nan),
                np.nan, np.nan, np.nan  # CV only has accuracy
            ],
            'Test': [
                validation_results.get('test_accuracy', np.nan),
                validation_results.get('test_precision', np.nan),
                validation_results.get('test_recall', np.nan),
                validation_results.get('test_f1', np.nan),
            ]
        })
    
    elif model_type == "ttr":
        comparison = pd.DataFrame({
            'Metric': ['R²', 'MAE (days)', 'RMSE (days)', 'MAPE (%)'],
            'Training': [
                train_results.get('train_r2', np.nan),
                train_results.get('train_mae', np.nan),
                train_results.get('train_rmse', np.nan),
                train_results.get('train_mape', np.nan),
            ],
            'CV': [
                train_results.get('cv_r2_mean', np.nan),
                train_results.get('cv_mae_mean', np.nan),
                np.nan, np.nan
            ],
            'Test': [
                validation_results.get('test_r2', np.nan),
                validation_results.get('test_mae', np.nan),
                validation_results.get('test_rmse', np.nan),
                validation_results.get('test_mape', np.nan),
            ]
        })
    
    return comparison
