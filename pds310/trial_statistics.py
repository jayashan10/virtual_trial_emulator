"""
Statistical Analysis for Virtual Clinical Trials.

Provides survival analysis, hypothesis testing, and power calculations.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_survival_statistics(
    data: pd.DataFrame,
    time_col: str,
    event_col: str,
    arm_col: str,
    control_arm: str,
    treatment_arm: str
) -> Dict[str, Any]:
    """
    Calculate survival statistics comparing two arms.
    
    Args:
        data: Patient data with survival outcomes
        time_col: Column with survival time
        event_col: Column with event indicator (1=event, 0=censored)
        arm_col: Column with treatment arm
        control_arm: Name of control arm
        treatment_arm: Name of treatment arm
    
    Returns:
        Dictionary with survival statistics
    """
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test, multivariate_logrank_test
    
    # Get arm data
    control_data = data[data[arm_col] == control_arm]
    treatment_data = data[data[arm_col] == treatment_arm]
    
    # Fit Kaplan-Meier curves
    kmf_control = KaplanMeierFitter()
    kmf_treatment = KaplanMeierFitter()
    
    kmf_control.fit(
        control_data[time_col],
        control_data[event_col],
        label=control_arm
    )
    
    kmf_treatment.fit(
        treatment_data[time_col],
        treatment_data[event_col],
        label=treatment_arm
    )
    
    # Median survival
    median_control = kmf_control.median_survival_time_
    median_treatment = kmf_treatment.median_survival_time_
    
    # Log-rank test
    results = logrank_test(
        control_data[time_col],
        treatment_data[time_col],
        control_data[event_col],
        treatment_data[event_col]
    )
    
    logrank_p = results.p_value
    logrank_stat = results.test_statistic
    
    # Cox proportional hazards for HR
    from lifelines import CoxPHFitter
    
    cox_data = data[[time_col, event_col, arm_col]].copy()
    cox_data['treatment'] = (cox_data[arm_col] == treatment_arm).astype(int)
    cox_data = cox_data.drop(columns=[arm_col])
    
    cph = CoxPHFitter()
    try:
        cph.fit(cox_data, duration_col=time_col, event_col=event_col)
        
        hr = float(cph.hazard_ratios_.loc['treatment'])
        hr_ci_lower = float(np.exp(cph.confidence_intervals_.loc['treatment', '95% lower-bound']))
        hr_ci_upper = float(np.exp(cph.confidence_intervals_.loc['treatment', '95% upper-bound']))
        hr_p = float(cph.summary.loc['treatment', 'p'])
    except Exception as e:
        print(f"Warning: Cox regression failed: {e}")
        hr = np.nan
        hr_ci_lower = np.nan
        hr_ci_upper = np.nan
        hr_p = np.nan
    
    # Event rates
    n_events_control = control_data[event_col].sum()
    n_events_treatment = treatment_data[event_col].sum()
    
    return {
        'control_arm': control_arm,
        'treatment_arm': treatment_arm,
        'n_control': len(control_data),
        'n_treatment': len(treatment_data),
        'events_control': int(n_events_control),
        'events_treatment': int(n_events_treatment),
        'median_control': float(median_control),
        'median_treatment': float(median_treatment),
        'median_difference': float(median_treatment - median_control),
        'hazard_ratio': float(hr),
        'hr_95ci_lower': float(hr_ci_lower),
        'hr_95ci_upper': float(hr_ci_upper),
        'hr_p_value': float(hr_p),
        'logrank_statistic': float(logrank_stat),
        'logrank_p_value': float(logrank_p),
        'significant': logrank_p < 0.05,
    }


def plot_kaplan_meier(
    data: pd.DataFrame,
    time_col: str,
    event_col: str,
    arm_col: str,
    title: str = "Kaplan-Meier Survival Curves",
    output_path: Optional[str] = None,
    baseline_data: Optional[pd.DataFrame] = None,
    baseline_label_prefix: str = "Observed",
    baseline_time_col: Optional[str] = None,
    baseline_event_col: Optional[str] = None,
    baseline_arm_col: Optional[str] = None,
):
    """
    Plot Kaplan-Meier survival curves.
    
    Args:
        data: Patient data
        time_col: Time column
        event_col: Event indicator column
        arm_col: Treatment arm column
        title: Plot title
        output_path: Path to save plot
    """
    from lifelines import KaplanMeierFitter
    
    plt.figure(figsize=(10, 6))
    
    kmf = KaplanMeierFitter()
    arms = data[arm_col].unique()

    # Fix colour cycle so simulated and observed share colours
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', None)
    if color_cycle is None or len(color_cycle) == 0:
        color_cycle = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    color_map = {arm: color_cycle[idx % len(color_cycle)] for idx, arm in enumerate(arms)}

    for arm in arms:
        arm_data = data[data[arm_col] == arm]
        
        kmf.fit(
            arm_data[time_col],
            arm_data[event_col],
            label=arm
        )
        
        kmf.plot_survival_function(ci_show=False, color=color_map[arm], linewidth=2)

    if baseline_data is not None and not baseline_data.empty:
        b_time = baseline_time_col or time_col
        b_event = baseline_event_col or event_col
        b_arm = baseline_arm_col or arm_col

        kmf_obs = KaplanMeierFitter()
        for arm in baseline_data[b_arm].unique():
            arm_obs = baseline_data[baseline_data[b_arm] == arm]
            label = f"{baseline_label_prefix} {arm}"
            colour = color_map.get(arm)

            kmf_obs.fit(
                arm_obs[b_time],
                arm_obs[b_event],
                label=label
            )
            kmf_obs.plot_survival_function(ci_show=False, color=colour, linestyle='--', linewidth=1.5)
    
    plt.xlabel('Time (days)')
    plt.ylabel('Survival Probability')
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Kaplan-Meier plot saved to: {output_path}")
    
    plt.close()


def calculate_response_rate_comparison(
    data: pd.DataFrame,
    response_col: str,
    arm_col: str,
    control_arm: str,
    treatment_arm: str,
    response_categories: List[str] = ['CR', 'PR']
) -> Dict[str, Any]:
    """
    Compare response rates between arms.
    
    Args:
        data: Patient data
        response_col: Column with response categories
        arm_col: Treatment arm column
        control_arm: Control arm name
        treatment_arm: Treatment arm name
        response_categories: Categories considered as response
    
    Returns:
        Dictionary with comparison statistics
    """
    # Get arm data
    control_data = data[data[arm_col] == control_arm]
    treatment_data = data[data[arm_col] == treatment_arm]
    
    # Calculate response rates
    n_control = len(control_data)
    n_treatment = len(treatment_data)
    
    responders_control = control_data[response_col].isin(response_categories).sum()
    responders_treatment = treatment_data[response_col].isin(response_categories).sum()
    
    rate_control = responders_control / n_control
    rate_treatment = responders_treatment / n_treatment
    
    # Fisher's exact test
    contingency_table = np.array([
        [responders_treatment, n_treatment - responders_treatment],
        [responders_control, n_control - responders_control]
    ])
    
    odds_ratio, p_value = stats.fisher_exact(contingency_table)
    
    # Risk difference and relative risk
    risk_difference = rate_treatment - rate_control
    relative_risk = rate_treatment / rate_control if rate_control > 0 else np.inf
    
    # Confidence intervals (Wald method)
    se_diff = np.sqrt(
        rate_control * (1 - rate_control) / n_control +
        rate_treatment * (1 - rate_treatment) / n_treatment
    )
    
    ci_lower = risk_difference - 1.96 * se_diff
    ci_upper = risk_difference + 1.96 * se_diff
    
    return {
        'control_arm': control_arm,
        'treatment_arm': treatment_arm,
        'n_control': n_control,
        'n_treatment': n_treatment,
        'responders_control': int(responders_control),
        'responders_treatment': int(responders_treatment),
        'rate_control': float(rate_control),
        'rate_treatment': float(rate_treatment),
        'risk_difference': float(risk_difference),
        'rd_95ci_lower': float(ci_lower),
        'rd_95ci_upper': float(ci_upper),
        'relative_risk': float(relative_risk),
        'odds_ratio': float(odds_ratio),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
    }


def calculate_sample_size_survival(
    median_control: float,
    target_hr: float,
    alpha: float = 0.05,
    power: float = 0.80,
    allocation_ratio: float = 1.0,
    dropout_rate: float = 0.10,
    accrual_time: float = 365,
    followup_time: float = 365
) -> Dict[str, Any]:
    """
    Calculate required sample size for survival endpoint.
    
    Uses Schoenfeld's method.
    
    Args:
        median_control: Median survival in control arm (days)
        target_hr: Target hazard ratio
        alpha: Type I error rate
        power: Statistical power
        allocation_ratio: Treatment:control ratio
        dropout_rate: Expected dropout rate
        accrual_time: Patient accrual period (days)
        followup_time: Additional follow-up after accrual (days)
    
    Returns:
        Dictionary with sample size calculations
    """
    from scipy.stats import norm
    
    # Z-scores
    z_alpha = norm.ppf(1 - alpha / 2)  # Two-sided
    z_beta = norm.ppf(power)
    
    # Convert median to hazard rate
    lambda_control = np.log(2) / median_control
    lambda_treatment = lambda_control * target_hr
    
    # Probability of event
    total_time = accrual_time + followup_time
    p_event_control = 1 - np.exp(-lambda_control * total_time / 2)  # Average
    p_event_treatment = 1 - np.exp(-lambda_treatment * total_time / 2)
    
    # Number of events required (Schoenfeld)
    d = 4 * ((z_alpha + z_beta) ** 2) / ((np.log(target_hr)) ** 2)
    
    # Adjust for allocation ratio
    r = allocation_ratio
    d_adjusted = d * (1 + r) ** 2 / (4 * r)
    
    # Total sample size
    p_event_avg = (p_event_control + p_event_treatment * r) / (1 + r)
    n_total = d_adjusted / p_event_avg
    
    # Adjust for dropout
    n_total_dropout = n_total / (1 - dropout_rate)
    
    # Per arm
    n_control = n_total_dropout / (1 + r)
    n_treatment = n_total_dropout * r / (1 + r)
    
    return {
        'n_total': int(np.ceil(n_total_dropout)),
        'n_control': int(np.ceil(n_control)),
        'n_treatment': int(np.ceil(n_treatment)),
        'n_events_required': int(np.ceil(d_adjusted)),
        'assumptions': {
            'median_control': median_control,
            'target_hr': target_hr,
            'alpha': alpha,
            'power': power,
            'allocation_ratio': allocation_ratio,
            'dropout_rate': dropout_rate,
            'accrual_time': accrual_time,
            'followup_time': followup_time,
        }
    }


def calculate_sample_size_proportion(
    rate_control: float,
    rate_treatment: float,
    alpha: float = 0.05,
    power: float = 0.80,
    allocation_ratio: float = 1.0
) -> Dict[str, Any]:
    """
    Calculate required sample size for proportion comparison.
    
    Args:
        rate_control: Expected rate in control
        rate_treatment: Expected rate in treatment
        alpha: Type I error
        power: Statistical power
        allocation_ratio: Treatment:control ratio
    
    Returns:
        Dictionary with sample size
    """
    from scipy.stats import norm
    
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    
    p1 = rate_control
    p2 = rate_treatment
    p_avg = (p1 + p2) / 2
    
    r = allocation_ratio
    
    # Sample size for control arm
    n1 = ((z_alpha * np.sqrt((1 + r) * p_avg * (1 - p_avg)) +
           z_beta * np.sqrt(r * p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / \
         (r * (p2 - p1) ** 2)
    
    n2 = n1 * r
    
    return {
        'n_total': int(np.ceil(n1 + n2)),
        'n_control': int(np.ceil(n1)),
        'n_treatment': int(np.ceil(n2)),
        'assumptions': {
            'rate_control': rate_control,
            'rate_treatment': rate_treatment,
            'alpha': alpha,
            'power': power,
            'allocation_ratio': allocation_ratio,
        }
    }


def forest_plot(
    results: List[Dict[str, Any]],
    metric: str = "hazard_ratio",
    ci_lower: str = "hr_95ci_lower",
    ci_upper: str = "hr_95ci_upper",
    labels: Optional[List[str]] = None,
    title: str = "Forest Plot",
    output_path: Optional[str] = None
):
    """
    Create forest plot of effect estimates.
    
    Args:
        results: List of analysis results
        metric: Column with point estimate
        ci_lower: Column with lower CI
        ci_upper: Column with upper CI
        labels: Labels for each result
        title: Plot title
        output_path: Path to save plot
    """
    if labels is None:
        labels = [f"Analysis {i+1}" for i in range(len(results))]
    
    estimates = [r[metric] for r in results]
    lower = [r[ci_lower] for r in results]
    upper = [r[ci_upper] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, len(results) * 0.5 + 2))
    
    y_pos = np.arange(len(labels))
    
    # Plot estimates and CIs
    ax.scatter(estimates, y_pos, s=100, color='steelblue', zorder=3)
    
    for i, (est, lo, hi) in enumerate(zip(estimates, lower, upper)):
        ax.plot([lo, hi], [i, i], 'steelblue', linewidth=2)
    
    # Reference line
    if metric == "hazard_ratio":
        ax.axvline(1.0, color='red', linestyle='--', linewidth=1, label='No effect')
    else:
        ax.axvline(0.0, color='red', linestyle='--', linewidth=1, label='No difference')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.grid(alpha=0.3, axis='x')
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Forest plot saved to: {output_path}")
    
    plt.close()
