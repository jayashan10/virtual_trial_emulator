"""
Calibration Assessment for PDS310 Models.

Evaluates how well predicted probabilities match observed frequencies.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
import matplotlib.pyplot as plt
from pathlib import Path


def assess_response_calibration(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
    strategy: str = 'uniform'
) -> Dict[str, Any]:
    """
    Assess calibration for response classification model.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities (n_samples, n_classes)
        n_bins: Number of bins for calibration curve
        strategy: Binning strategy ('uniform' or 'quantile')
    
    Returns:
        Dictionary with calibration metrics
    """
    results = {}
    
    # Get unique classes
    classes = np.unique(y_true)
    
    # For each class, calculate calibration
    for i, cls in enumerate(classes):
        # Binary indicators
        y_binary = (y_true == cls).astype(int)
        
        # Get probabilities for this class
        if y_pred_proba.ndim == 1:
            probs = y_pred_proba
        else:
            probs = y_pred_proba[:, i] if y_pred_proba.shape[1] > i else y_pred_proba[:, 0]
        
        # Calculate calibration curve
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_binary, probs,
                n_bins=n_bins,
                strategy=strategy
            )
            
            # Brier score (lower is better, 0 = perfect)
            brier = brier_score_loss(y_binary, probs)
            
            # Expected Calibration Error (ECE)
            ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            
            # Maximum Calibration Error (MCE)
            mce = np.max(np.abs(fraction_of_positives - mean_predicted_value))
            
            results[str(cls)] = {
                'fraction_of_positives': fraction_of_positives.tolist(),
                'mean_predicted_value': mean_predicted_value.tolist(),
                'brier_score': float(brier),
                'ece': float(ece),
                'mce': float(mce),
                'n_samples': int(y_binary.sum()),
            }
            
        except Exception as e:
            print(f"Warning: Could not calculate calibration for class {cls}: {e}")
            continue
    
    # Overall metrics
    try:
        # Multi-class Brier score
        if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
            # One-hot encode true labels
            y_true_onehot = np.zeros_like(y_pred_proba)
            for i, cls in enumerate(classes):
                y_true_onehot[y_true == cls, i] = 1
            
            overall_brier = np.mean((y_pred_proba - y_true_onehot) ** 2)
            results['overall_brier_score'] = float(overall_brier)
    except Exception as e:
        print(f"Warning: Could not calculate overall Brier score: {e}")
    
    return results


def assess_regression_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Assess calibration for regression model (TTR).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        n_bins: Number of bins
    
    Returns:
        Dictionary with calibration metrics
    """
    # Create bins based on predicted values
    bins = np.percentile(y_pred, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)  # Remove duplicates
    
    bin_indices = np.digitize(y_pred, bins[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
    
    calibration_data = []
    
    for i in range(len(bins) - 1):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_true_mean = y_true[mask].mean()
            bin_pred_mean = y_pred[mask].mean()
            bin_count = mask.sum()
            
            calibration_data.append({
                'bin': i,
                'bin_range': f'[{bins[i]:.1f}, {bins[i+1]:.1f})',
                'n_samples': int(bin_count),
                'mean_predicted': float(bin_pred_mean),
                'mean_actual': float(bin_true_mean),
                'difference': float(abs(bin_true_mean - bin_pred_mean)),
            })
    
    # Overall calibration slope (should be close to 1.0)
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_pred, y_true)
    
    return {
        'bins': calibration_data,
        'calibration_slope': float(slope),
        'calibration_intercept': float(intercept),
        'calibration_r2': float(r_value ** 2),
        'n_bins': len(calibration_data),
    }


def plot_calibration_curve(
    calibration_results: Dict[str, Any],
    model_type: str = "response",
    output_path: Optional[str] = None
):
    """
    Plot calibration curve.
    
    Args:
        calibration_results: Results from assess_*_calibration
        model_type: "response" or "regression"
        output_path: Path to save plot
    """
    if model_type == "response":
        # Plot calibration curves for each class
        n_classes = len([k for k in calibration_results.keys() if k != 'overall_brier_score'])
        
        fig, axes = plt.subplots(1, min(n_classes, 3), figsize=(5*min(n_classes, 3), 5))
        if n_classes == 1:
            axes = [axes]
        
        for idx, (cls, cal_data) in enumerate(calibration_results.items()):
            if cls == 'overall_brier_score' or idx >= 3:
                continue
            
            ax = axes[idx] if n_classes > 1 else axes[0]
            
            # Plot calibration curve
            frac_pos = cal_data['fraction_of_positives']
            mean_pred = cal_data['mean_predicted_value']
            
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
            ax.plot(mean_pred, frac_pos, 's-', label=f'Class: {cls}')
            
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title(f'Calibration Curve: {cls}\n'
                        f'Brier={cal_data["brier_score"]:.3f}, '
                        f'ECE={cal_data["ece"]:.3f}')
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
        
        plt.tight_layout()
    
    elif model_type == "regression":
        # Plot predicted vs actual by bin
        bins_data = calibration_results['bins']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Calibration plot
        ax = axes[0]
        predicted = [b['mean_predicted'] for b in bins_data]
        actual = [b['mean_actual'] for b in bins_data]
        
        ax.plot([min(predicted), max(predicted)], 
               [min(predicted), max(predicted)], 
               'k--', label='Perfect Calibration')
        ax.scatter(predicted, actual, s=100, alpha=0.6, label='Bin Means')
        
        ax.set_xlabel('Mean Predicted Value')
        ax.set_ylabel('Mean Actual Value')
        ax.set_title(f'Regression Calibration\n'
                    f'Slope={calibration_results["calibration_slope"]:.3f}, '
                    f'R²={calibration_results["calibration_r2"]:.3f}')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Difference plot
        ax = axes[1]
        differences = [b['difference'] for b in bins_data]
        bin_centers = [(predicted[i] + predicted[min(i+1, len(predicted)-1)]) / 2 
                      for i in range(len(predicted))]
        
        ax.bar(range(len(bins_data)), differences, alpha=0.6, color='coral')
        ax.set_xlabel('Bin Index')
        ax.set_ylabel('|Mean Predicted - Mean Actual|')
        ax.set_title('Calibration Error by Bin')
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Calibration plot saved to: {output_path}")
    
    plt.close()


def hosmer_lemeshow_test(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Hosmer-Lemeshow goodness-of-fit test for calibration.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins
    
    Returns:
        Dictionary with test statistic and p-value
    """
    from scipy import stats
    
    # Create bins
    bins = np.percentile(y_pred_proba, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    
    bin_indices = np.digitize(y_pred_proba, bins[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
    
    # Calculate observed and expected
    chi_square = 0
    
    for i in range(len(bins) - 1):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue
        
        observed = y_true[mask].sum()
        expected = y_pred_proba[mask].sum()
        n = mask.sum()
        
        # Add to chi-square statistic
        if expected > 0 and expected < n:
            chi_square += ((observed - expected) ** 2) / (expected * (1 - expected / n))
    
    # Degrees of freedom
    df = len(bins) - 2  # n_bins - 2
    
    # P-value
    p_value = 1 - stats.chi2.cdf(chi_square, df)
    
    return {
        'chi_square': float(chi_square),
        'df': int(df),
        'p_value': float(p_value),
        'interpretation': 'Good calibration' if p_value > 0.05 else 'Poor calibration'
    }


def calculate_calibration_metrics_summary(
    calibration_results: Dict[str, Any],
    model_type: str = "response"
) -> pd.DataFrame:
    """
    Create summary table of calibration metrics.
    
    Returns:
        DataFrame with calibration metrics
    """
    if model_type == "response":
        rows = []
        for cls, metrics in calibration_results.items():
            if cls == 'overall_brier_score':
                continue
            rows.append({
                'Class': cls,
                'Brier Score': metrics['brier_score'],
                'ECE': metrics['ece'],
                'MCE': metrics['mce'],
                'N Samples': metrics['n_samples'],
            })
        
        summary = pd.DataFrame(rows)
        
        # Add overall row if available
        if 'overall_brier_score' in calibration_results:
            overall_row = pd.DataFrame([{
                'Class': 'Overall',
                'Brier Score': calibration_results['overall_brier_score'],
                'ECE': np.nan,
                'MCE': np.nan,
                'N Samples': sum(r['N Samples'] for r in rows),
            }])
            summary = pd.concat([summary, overall_row], ignore_index=True)
    
    elif model_type == "regression":
        summary = pd.DataFrame([{
            'Metric': 'Calibration Slope',
            'Value': calibration_results['calibration_slope'],
        }, {
            'Metric': 'Calibration Intercept',
            'Value': calibration_results['calibration_intercept'],
        }, {
            'Metric': 'Calibration R²',
            'Value': calibration_results['calibration_r2'],
        }, {
            'Metric': 'Number of Bins',
            'Value': calibration_results['n_bins'],
        }])
    
    return summary
