"""
Uncertainty Quantification via Bootstrap for PDS310.

Provides confidence intervals and prediction intervals for all models.
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
import pandas as pd
import numpy as np
from sklearn.utils import resample
from pathlib import Path
import joblib


def bootstrap_predictions(
    model_bundle: Dict[str, Any],
    X: pd.DataFrame,
    prepare_fn: Optional[Callable[[Dict[str, Any], pd.DataFrame], pd.DataFrame]] = None,
    feature_names: Optional[List[str]] = None,
    n_bootstrap: int = 100,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Generate bootstrap confidence intervals for predictions.
    
    Args:
        model_bundle: Trained model results bundle
        X: Raw feature matrix
        prepare_fn: Optional callable to align raw features with the trained pipeline
        feature_names: Optional list of feature names (used when prepare_fn is None)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default: 0.95)
        random_state: Random seed
    
    Returns:
        Dictionary with point estimates and confidence intervals
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    pipeline = model_bundle["model"]

    if prepare_fn is not None:
        X_features = prepare_fn(model_bundle, X)
    else:
        if feature_names is None:
            feature_names = model_bundle.get("feature_names")
        if feature_names is None:
            raise ValueError("feature_names must be provided when prepare_fn is None.")
        X_features = X[feature_names].copy()
        categorical_cols = X_features.select_dtypes(include=["object", "string"]).columns
        for col in categorical_cols:
            X_features[col] = pd.Categorical(X_features[col]).codes
        numeric_cols = X_features.select_dtypes(include=[np.number]).columns
        X_features[numeric_cols] = X_features[numeric_cols].fillna(X_features[numeric_cols].median())
    
    # Store bootstrap predictions
    bootstrap_preds = []
    
    for i in range(n_bootstrap):
        # Resample features (sample with replacement)
        X_boot = resample(X_features, random_state=random_state + i if random_state else None)
        
        # Predict
        pred = pipeline.predict(X_boot)
        bootstrap_preds.append(pred)
    
    # Convert to array
    bootstrap_preds = np.array(bootstrap_preds)
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    point_estimate = np.median(bootstrap_preds, axis=0)
    lower_ci = np.percentile(bootstrap_preds, lower_percentile, axis=0)
    upper_ci = np.percentile(bootstrap_preds, upper_percentile, axis=0)
    std_err = np.std(bootstrap_preds, axis=0)
    
    return {
        'point_estimate': point_estimate,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci,
        'std_err': std_err,
        'bootstrap_samples': bootstrap_preds,
    }


def bootstrap_model_performance(
    model_bundle: Dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    metric_func: Callable,
    n_bootstrap: int = 100,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None,
    prepare_fn: Optional[Callable[[Dict[str, Any], pd.DataFrame], pd.DataFrame]] = None,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Bootstrap confidence interval for model performance metric.
    
    Args:
        model_bundle: Trained model bundle containing the fitted pipeline
        X: Feature matrix
        y: True labels/values
        metric_func: Function to calculate metric (e.g., accuracy_score)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level
        random_state: Random seed
        prepare_fn: Optional callable to align raw features with pipeline
        feature_names: Optional list of raw feature names (used when prepare_fn is None)
    
    Returns:
        Dictionary with metric estimate and confidence interval
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    pipeline = model_bundle["model"]

    if prepare_fn is not None:
        X_features = prepare_fn(model_bundle, X)
    else:
        if feature_names is None:
            feature_names = model_bundle.get("feature_names")
        if feature_names is None:
            raise ValueError("feature_names must be provided when prepare_fn is None.")
        X_features = X[feature_names].copy()
        categorical_cols = X_features.select_dtypes(include=["object", "string"]).columns
        for col in categorical_cols:
            X_features[col] = pd.Categorical(X_features[col]).codes
        numeric_cols = X_features.select_dtypes(include=[np.number]).columns
        X_features[numeric_cols] = X_features[numeric_cols].fillna(X_features[numeric_cols].median())
    
    # Store bootstrap metrics
    bootstrap_metrics = []
    
    for i in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(len(X_features), size=len(X_features), replace=True)
        X_boot = X_features.iloc[indices]
        y_boot = y[indices]
        
        # Predict
        y_pred = pipeline.predict(X_boot)
        
        # Calculate metric
        try:
            metric = metric_func(y_boot, y_pred)
            bootstrap_metrics.append(metric)
        except Exception as e:
            # Skip if metric calculation fails
            continue
    
    bootstrap_metrics = np.array(bootstrap_metrics)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    return {
        'metric_mean': float(np.mean(bootstrap_metrics)),
        'metric_median': float(np.median(bootstrap_metrics)),
        'lower_ci': float(np.percentile(bootstrap_metrics, lower_percentile)),
        'upper_ci': float(np.percentile(bootstrap_metrics, upper_percentile)),
        'std_err': float(np.std(bootstrap_metrics)),
    }


def predict_with_uncertainty(
    model_bundle: Dict[str, Any],
    profile: pd.DataFrame,
    prepare_fn: Optional[Callable[[Dict[str, Any], pd.DataFrame], pd.DataFrame]] = None,
    feature_names: Optional[List[str]] = None,
    n_bootstrap: int = 100,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Make prediction with uncertainty estimates for single patient.
    
    Args:
        model_bundle: Trained model bundle
        profile: Patient profile (single row DataFrame)
        prepare_fn: Optional callable to align raw features with pipeline
        feature_names: Optional list of feature names (used when prepare_fn is None)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level
        random_state: Random seed
    
    Returns:
        Dictionary with prediction and uncertainty
    """
    pipeline = model_bundle["model"]

    if prepare_fn is not None:
        X = prepare_fn(model_bundle, profile)
    else:
        if feature_names is None:
            feature_names = model_bundle.get("feature_names")
        if feature_names is None:
            raise ValueError("feature_names must be provided when prepare_fn is None.")
        X = profile[feature_names].copy()
        categorical_cols = X.select_dtypes(include=["object", "string"]).columns
        for col in categorical_cols:
            X[col] = pd.Categorical(X[col]).codes
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    
    # Point prediction
    point_pred = pipeline.predict(X)[0]
    
    # Bootstrap for uncertainty (resample training data if available)
    # For now, use feature perturbation as proxy
    if random_state is not None:
        np.random.seed(random_state)
    
    bootstrap_preds = []
    
    for i in range(n_bootstrap):
        # Add small random noise to numeric features
        X_perturbed = X.copy()
        for col in numeric_cols:
            if col in X_perturbed.columns:
                noise = np.random.normal(0, X_perturbed[col].std() * 0.05)
                X_perturbed[col] = X_perturbed[col] + noise
        
        # Predict
        try:
            pred = pipeline.predict(X_perturbed)[0]
            bootstrap_preds.append(pred)
        except:
            bootstrap_preds.append(point_pred)
    
    bootstrap_preds = np.array(bootstrap_preds)
    
    # Calculate intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    return {
        'prediction': float(point_pred),
        'lower_ci': float(np.percentile(bootstrap_preds, lower_percentile)),
        'upper_ci': float(np.percentile(bootstrap_preds, upper_percentile)),
        'std_err': float(np.std(bootstrap_preds)),
        'confidence_level': confidence_level,
    }


def plot_prediction_intervals(
    y_true: np.ndarray,
    predictions_with_ci: Dict[str, np.ndarray],
    output_path: Optional[str] = None
):
    """
    Plot predictions with confidence intervals.
    
    Args:
        y_true: True values
        predictions_with_ci: Dict with 'point_estimate', 'lower_ci', 'upper_ci'
        output_path: Path to save plot
    """
    import matplotlib.pyplot as plt
    
    point_est = predictions_with_ci['point_estimate']
    lower_ci = predictions_with_ci['lower_ci']
    upper_ci = predictions_with_ci['upper_ci']
    
    # Sort by predicted value for better visualization
    sort_idx = np.argsort(point_est)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Prediction intervals plot
    ax = axes[0]
    x_idx = np.arange(len(point_est))
    
    ax.fill_between(x_idx, lower_ci[sort_idx], upper_ci[sort_idx], 
                    alpha=0.3, color='steelblue', label='95% CI')
    ax.plot(x_idx, point_est[sort_idx], 'b-', label='Prediction')
    ax.scatter(x_idx, y_true[sort_idx], color='red', s=20, alpha=0.5, label='Actual')
    
    ax.set_xlabel('Sample Index (sorted)')
    ax.set_ylabel('Value')
    ax.set_title('Predictions with 95% Confidence Intervals')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Coverage plot
    ax = axes[1]
    coverage = (y_true >= lower_ci) & (y_true <= upper_ci)
    coverage_rate = coverage.mean()
    
    ax.bar(['Within CI', 'Outside CI'], 
          [coverage.sum(), (~coverage).sum()],
          color=['green', 'red'], alpha=0.6)
    ax.set_ylabel('Count')
    ax.set_title(f'CI Coverage: {coverage_rate*100:.1f}%\n(Target: 95%)')
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Prediction intervals plot saved to: {output_path}")
    
    plt.close()


def calculate_coverage(
    y_true: np.ndarray,
    lower_ci: np.ndarray,
    upper_ci: np.ndarray,
    expected_coverage: float = 0.95
) -> Dict[str, float]:
    """
    Calculate actual coverage of confidence intervals.
    
    Args:
        y_true: True values
        lower_ci: Lower confidence bounds
        upper_ci: Upper confidence bounds
        expected_coverage: Expected coverage (e.g., 0.95 for 95% CI)
    
    Returns:
        Dictionary with coverage statistics
    """
    coverage = (y_true >= lower_ci) & (y_true <= upper_ci)
    actual_coverage = coverage.mean()
    
    # Binomial test for whether coverage is significantly different from expected
    from scipy import stats
    n = len(coverage)
    k = coverage.sum()
    p_value = stats.binom_test(k, n, expected_coverage, alternative='two-sided')
    
    return {
        'actual_coverage': float(actual_coverage),
        'expected_coverage': expected_coverage,
        'coverage_difference': float(actual_coverage - expected_coverage),
        'n_samples': int(n),
        'n_within_ci': int(k),
        'p_value': float(p_value),
        'is_well_calibrated': p_value > 0.05,
    }
