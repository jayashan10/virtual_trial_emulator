"""
Twin Validation Framework for PDS310.

Validates that generated digital twins are biologically plausible and
representative of the real patient population.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.covariance import EmpiricalCovariance


def validate_twin(
    twin: Dict[str, Any],
    real_profiles: pd.DataFrame,
    strict: bool = False
) -> Dict[str, Any]:
    """
    Check if digital twin is biologically plausible.
    
    Validation checks:
    1. Feature ranges: All values within observed min-max
    2. Correlations: Key relationships preserved
    3. Multivariate: Mahalanobis distance from real data
    4. Clinical logic: No contradictions
    
    Args:
        twin: Digital twin profile dictionary
        real_profiles: DataFrame of real patient profiles
        strict: If True, apply stricter validation criteria
    
    Returns:
        Dictionary with validation results:
        - is_valid: bool
        - validation_score: 0-1 (higher = more valid)
        - issues: List of problems if invalid
        - checks: Dict of individual check results
    """
    issues = []
    checks = {}
    
    # Check 1: Feature ranges
    range_check = _check_feature_ranges(twin, real_profiles)
    checks["feature_ranges"] = range_check
    if not range_check["passed"]:
        issues.extend(range_check["issues"])
    
    # Check 2: Required features present
    required_check = _check_required_features(twin)
    checks["required_features"] = required_check
    if not required_check["passed"]:
        issues.extend(required_check["issues"])
    
    # Check 3: Correlations
    correlation_check = _check_correlations(twin, real_profiles)
    checks["correlations"] = correlation_check
    if not correlation_check["passed"] and strict:
        issues.extend(correlation_check["issues"])
    
    # Check 4: Clinical logic
    logic_check = _check_clinical_logic(twin)
    checks["clinical_logic"] = logic_check
    if not logic_check["passed"]:
        issues.extend(logic_check["issues"])
    
    # Check 5: Multivariate distance
    distance_check = _check_multivariate_distance(twin, real_profiles)
    checks["multivariate_distance"] = distance_check
    if not distance_check["passed"] and strict:
        issues.append("Twin is multivariate outlier")
    
    # Calculate overall validation score
    check_scores = [
        checks["feature_ranges"]["score"],
        checks["required_features"]["score"],
        checks["correlations"]["score"],
        checks["clinical_logic"]["score"],
        checks["multivariate_distance"]["score"],
    ]
    
    validation_score = np.mean(check_scores)
    
    # Determine if valid
    if strict:
        is_valid = all(c["passed"] for c in checks.values())
    else:
        # Lenient: only fail on critical issues
        is_valid = (
            checks["feature_ranges"]["score"] >= 0.8 and
            checks["required_features"]["passed"] and
            checks["clinical_logic"]["passed"]
        )
    
    return {
        "is_valid": is_valid,
        "validation_score": validation_score,
        "issues": issues,
        "checks": checks,
    }


def _check_feature_ranges(
    twin: Dict[str, Any],
    real_profiles: pd.DataFrame
) -> Dict[str, Any]:
    """
    Check if all numeric features are within observed ranges.
    """
    issues = []
    out_of_range = 0
    total_numeric = 0
    
    for feature, value in twin.items():
        if feature not in real_profiles.columns:
            continue
        
        if isinstance(value, (int, float)) and not pd.isna(value):
            total_numeric += 1
            
            # Get observed range
            real_values = real_profiles[feature].dropna()
            if len(real_values) > 0:
                min_val = real_values.min()
                max_val = real_values.max()
                
                # Allow small buffer (±5%)
                buffer = (max_val - min_val) * 0.05
                if value < (min_val - buffer) or value > (max_val + buffer):
                    out_of_range += 1
                    issues.append(
                        f"{feature} = {value:.2f} outside range [{min_val:.2f}, {max_val:.2f}]"
                    )
    
    score = 1.0 - (out_of_range / total_numeric) if total_numeric > 0 else 1.0
    passed = out_of_range == 0
    
    return {
        "passed": passed,
        "score": score,
        "out_of_range_count": out_of_range,
        "total_numeric": total_numeric,
        "issues": issues,
    }


def _check_required_features(twin: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if required features are present.
    """
    required_features = [
        "AGE", "SEX", "RAS_status", "ATRT", "composite_risk_score"
    ]
    
    missing = [f for f in required_features if f not in twin or twin[f] is None]
    
    score = 1.0 - (len(missing) / len(required_features))
    passed = len(missing) == 0
    
    issues = [f"Missing required feature: {f}" for f in missing]
    
    return {
        "passed": passed,
        "score": score,
        "missing_features": missing,
        "issues": issues,
    }


def _check_correlations(
    twin: Dict[str, Any],
    real_profiles: pd.DataFrame
) -> Dict[str, Any]:
    """
    Check if key feature correlations are preserved.
    """
    issues = []
    
    # Define expected correlations
    correlations_to_check = [
        ("AGE", "performance_risk", "positive"),  # Older → worse performance
        ("baseline_HGB", "composite_risk_score", "negative"),  # Low HGB → higher risk
        ("composite_risk_score", "predicted_good_prognosis_flag", "negative"),  # High risk → poor prognosis
    ]
    
    violation_count = 0
    total_checks = 0
    
    for feat1, feat2, expected_direction in correlations_to_check:
        if feat1 in twin and feat2 in twin and feat1 in real_profiles.columns and feat2 in real_profiles.columns:
            total_checks += 1
            
            # Get real correlation
            real_corr = real_profiles[[feat1, feat2]].corr().iloc[0, 1]
            
            # Check if twin values are consistent with correlation
            val1 = twin[feat1]
            val2 = twin[feat2]
            
            if not pd.isna(val1) and not pd.isna(val2) and not pd.isna(real_corr):
                # Get percentile ranks
                pct1 = stats.percentileofscore(real_profiles[feat1].dropna(), val1) / 100
                pct2 = stats.percentileofscore(real_profiles[feat2].dropna(), val2) / 100
                
                # Check if correlation direction is preserved
                if expected_direction == "positive":
                    # Both should be high or both low
                    if abs(pct1 - pct2) > 0.5:  # More than 50 percentile points apart
                        violation_count += 1
                        issues.append(
                            f"Correlation violation: {feat1} and {feat2} should be positively correlated"
                        )
                elif expected_direction == "negative":
                    # One high, one low
                    if (pct1 + pct2) < 0.5 or (pct1 + pct2) > 1.5:  # Not opposite
                        violation_count += 1
                        issues.append(
                            f"Correlation violation: {feat1} and {feat2} should be negatively correlated"
                        )
    
    score = 1.0 - (violation_count / total_checks) if total_checks > 0 else 1.0
    passed = violation_count == 0
    
    return {
        "passed": passed,
        "score": score,
        "violations": violation_count,
        "total_checks": total_checks,
        "issues": issues,
    }


def _check_clinical_logic(twin: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check for clinical contradictions.
    """
    issues = []
    
    # Check 1: Age range
    age = twin.get("AGE")
    if age is not None and not pd.isna(age):
        if age < 18:
            issues.append(f"Age {age} is below adult threshold")
        elif age > 100:
            issues.append(f"Age {age} is unrealistic")
    
    # Check 2: Hemoglobin and anemia
    hgb = twin.get("baseline_HGB")
    if hgb is not None and not pd.isna(hgb):
        if hgb < 7:
            issues.append(f"Hemoglobin {hgb:.1f} is dangerously low")
        elif hgb > 20:
            issues.append(f"Hemoglobin {hgb:.1f} is unrealistically high")
    
    # Check 3: ECOG and performance risk consistency
    ecog = twin.get("B_ECOG")
    perf_risk = twin.get("performance_risk")
    if ecog is not None and perf_risk is not None and not pd.isna(perf_risk):
        if isinstance(ecog, str):
            ecog_map = {
                "Asymptomatic": 0,
                "Symptoms but ambulatory": 1,
                "In bed <50% of day": 2,
                "In bed >50% of day": 3,
                "Bedridden": 4,
            }
            ecog_num = ecog_map.get(ecog)
        else:
            ecog_num = ecog
        
        if ecog_num is not None:
            if ecog_num == 0 and perf_risk > 0.2:
                issues.append("ECOG 0 but high performance risk")
            elif ecog_num >= 3 and perf_risk < 0.8:
                issues.append("Poor ECOG but low performance risk")
    
    # Check 4: Early weight change within plausible bounds
    weight_pct_42d = twin.get("weight_change_pct_42d")
    if weight_pct_42d is not None and not pd.isna(weight_pct_42d):
        if weight_pct_42d < -50 or weight_pct_42d > 50:
            issues.append(
                f"weight_change_pct_42d {weight_pct_42d:.1f}% outside plausible range (-50%, 50%)"
            )
    
    # Check 5: Treatment and RAS status (if EGFR inhibitor trial)
    trt = twin.get("ATRT", "")
    ras_status = twin.get("RAS_status")
    if "panit" in str(trt).lower() and ras_status == "MUTANT":
        # Note: This is actually realistic - trial included KRAS mutants
        # So this is not a violation, just a note
        pass
    
    score = 1.0 if len(issues) == 0 else max(0.0, 1.0 - len(issues) * 0.2)
    passed = len(issues) == 0
    
    return {
        "passed": passed,
        "score": score,
        "issues": issues,
    }


def _check_multivariate_distance(
    twin: Dict[str, Any],
    real_profiles: pd.DataFrame,
    threshold_percentile: float = 99
) -> Dict[str, Any]:
    """
    Check if twin is a multivariate outlier using Mahalanobis distance.
    """
    # Select numeric features present in both twin and real profiles
    numeric_features = []
    for feat in real_profiles.select_dtypes(include=[np.number]).columns:
        if feat in twin and twin[feat] is not None and not pd.isna(twin[feat]):
            numeric_features.append(feat)
    
    if len(numeric_features) < 2:
        # Not enough features for multivariate analysis
        return {
            "passed": True,
            "score": 1.0,
            "distance": None,
            "threshold": None,
        }
    
    # Prepare data
    X_real = real_profiles[numeric_features].dropna()
    if len(X_real) < 10:
        # Insufficient data
        return {"passed": True, "score": 1.0, "distance": None, "threshold": None}
    
    # Create twin vector
    x_twin = np.array([twin[f] for f in numeric_features]).reshape(1, -1)
    
    # Compute Mahalanobis distance
    try:
        cov = EmpiricalCovariance().fit(X_real)
        distance = cov.mahalanobis(x_twin)[0]
        
        # Compute distances for all real patients
        real_distances = cov.mahalanobis(X_real)
        threshold = np.percentile(real_distances, threshold_percentile)
        
        passed = distance <= threshold
        
        # Score based on percentile
        percentile = stats.percentileofscore(real_distances, distance)
        score = 1.0 - (max(0, percentile - 95) / 5)  # Penalize if > 95th percentile
        
    except Exception:
        # Covariance estimation failed (e.g., singular matrix)
        return {"passed": True, "score": 1.0, "distance": None, "threshold": None}
    
    return {
        "passed": passed,
        "score": score,
        "distance": float(distance),
        "threshold": float(threshold),
        "percentile": float(percentile),
    }


def validate_twin_cohort(
    twins_df: pd.DataFrame,
    real_profiles: pd.DataFrame,
    min_validation_score: float = 0.7,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Validate entire cohort of digital twins.
    
    Args:
        twins_df: DataFrame of digital twins
        real_profiles: DataFrame of real patients
        min_validation_score: Minimum acceptable validation score
        verbose: Print summary
    
    Returns:
        Dictionary with cohort validation results
    """
    if verbose:
        print(f"Validating {len(twins_df)} digital twins...")
    
    results = []
    for idx, row in twins_df.iterrows():
        twin = row.to_dict()
        validation = validate_twin(twin, real_profiles, strict=False)
        results.append(validation)
    
    # Aggregate results
    valid_count = sum(1 for r in results if r["is_valid"])
    mean_score = np.mean([r["validation_score"] for r in results])
    
    passing_count = sum(1 for r in results if r["validation_score"] >= min_validation_score)
    
    # Collect common issues
    all_issues = []
    for r in results:
        all_issues.extend(r["issues"])
    
    issue_counts = pd.Series(all_issues).value_counts()
    
    summary = {
        "total_twins": len(twins_df),
        "valid_twins": valid_count,
        "valid_percentage": valid_count / len(twins_df) * 100,
        "passing_twins": passing_count,
        "passing_percentage": passing_count / len(twins_df) * 100,
        "mean_validation_score": mean_score,
        "min_validation_score": float(np.min([r["validation_score"] for r in results])),
        "max_validation_score": float(np.max([r["validation_score"] for r in results])),
        "common_issues": issue_counts.head(10).to_dict() if len(issue_counts) > 0 else {},
    }
    
    if verbose:
        print(f"\nValidation Summary:")
        print(f"  Valid twins: {valid_count}/{len(twins_df)} ({summary['valid_percentage']:.1f}%)")
        print(f"  Passing twins (score ≥ {min_validation_score}): {passing_count}/{len(twins_df)} ({summary['passing_percentage']:.1f}%)")
        print(f"  Mean validation score: {mean_score:.3f}")
        
        if len(summary["common_issues"]) > 0:
            print(f"\n  Most common issues:")
            for issue, count in list(summary["common_issues"].items())[:5]:
                print(f"    - {issue}: {count} twins")
    
    return summary
