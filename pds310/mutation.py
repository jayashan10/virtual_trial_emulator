"""
Mutation Engine for PDS310 Digital Twins.

Applies realistic variations to synthetic patient profiles to increase diversity
while maintaining biological plausibility.
"""

from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np


def apply_mutation(
    profile: Dict[str, Any],
    mutation_rate: float = 0.05,
    feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Apply realistic mutations to a digital twin profile.
    
    Mutation types:
    - Numeric features: Add Gaussian noise (±5% default)
    - Categorical features: Occasionally flip (5% probability default)
    - Correlated features: Maintain relationships (e.g., HGB correlates with RBC)
    
    Args:
        profile: Patient profile dictionary
        mutation_rate: Base probability of mutation (0-1)
        feature_ranges: Dictionary mapping features to (min, max) valid ranges
        seed: Random seed
    
    Returns:
        Mutated profile dictionary
    
    Examples:
        - AGE: 65 → 67 (±2 years with some probability)
        - LDH: 250 → 245 (±5 U/L)
        - SEX: Keep unchanged (biological constraint)
        - RAS_status: Keep unchanged (genetics fixed at diagnosis)
    """
    if seed is not None:
        np.random.seed(seed)
    
    mutated = profile.copy()
    
    # Get feature ranges if not provided
    if feature_ranges is None:
        feature_ranges = get_default_feature_ranges()
    
    # Define features that should NEVER be mutated
    immutable_features = {
        "SUBJID", "STUDYID",  # Identifiers
        "SEX",  # Biological sex (fixed)
        "RACE",  # Race (fixed)
        # Molecular biomarkers (genetics fixed at diagnosis)
        "KRAS_exon2", "KRAS_exon3", "KRAS_exon4",
        "NRAS_exon2", "NRAS_exon3", "NRAS_exon4",
        "BRAF_exon15", "RAS_status",
        # Treatment assignment (study design)
        "TRT", "ATRT",
    }
    
    # Define weakly mutable features (lower mutation rate)
    weak_mutation_features = {
        "AGE",  # Age changes slowly
        "DIAGMONS",  # Disease history (fixed)
        "HISSUBTY", "DIAGTYPE",  # Tumor characteristics (fixed)
        "num_prior_therapies",  # Past history (fixed)
    }
    
    # Mutate each feature
    for feature, value in profile.items():
        # Skip immutable features
        if feature in immutable_features:
            continue
        
        # Skip None/NaN values
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        
        # Determine mutation probability
        if feature in weak_mutation_features:
            feat_mutation_rate = mutation_rate * 0.3  # 30% of base rate
        else:
            feat_mutation_rate = mutation_rate
        
        # Roll for mutation
        if np.random.random() > feat_mutation_rate:
            continue  # No mutation
        
        # Apply mutation based on type
        if isinstance(value, (int, float)):
            # Numeric mutation
            mutated[feature] = _mutate_numeric(
                value, feature, feature_ranges.get(feature)
            )
        elif isinstance(value, str):
            # Categorical mutation
            mutated[feature] = _mutate_categorical(
                value, feature, profile
            )
    
    return mutated


def _mutate_numeric(
    value: float,
    feature_name: str,
    valid_range: Optional[Tuple[float, float]] = None
) -> float:
    """
    Apply Gaussian noise to numeric feature.
    
    Default: ±5% of value
    """
    # Determine noise magnitude based on feature type
    if "AGE" in feature_name.upper():
        # Age: ±2 years
        noise_std = 2.0
    elif any(x in feature_name.upper() for x in ["LAB_", "BASELINE_"]):
        # Lab values: ±10% to account for measurement variability
        noise_std = abs(value) * 0.10
    elif "WEIGHT" in feature_name.upper():
        # Weight: ±2 kg
        noise_std = 2.0
    elif "RISK_SCORE" in feature_name.upper():
        # Risk scores: ±0.05 (bounded 0-1)
        noise_std = 0.05
    elif "COUNT" in feature_name:
        # Counts: ±1 (round to nearest integer)
        noise_std = 1.0
    else:
        # Default: ±5%
        noise_std = abs(value) * 0.05
    
    # Add Gaussian noise
    mutated = value + np.random.normal(0, noise_std)
    
    # Apply valid range constraints
    if valid_range:
        min_val, max_val = valid_range
        mutated = np.clip(mutated, min_val, max_val)
    
    # Round counts to integers
    if "COUNT" in feature_name or "lesion" in feature_name.lower():
        mutated = round(mutated)
    
    # Ensure risk scores stay in [0, 1]
    if "risk_score" in feature_name.lower() or "prognosis_flag" in feature_name.lower():
        mutated = np.clip(mutated, 0, 1)
    
    return mutated


def _mutate_categorical(
    value: str,
    feature_name: str,
    profile: Dict[str, Any]
) -> str:
    """
    Occasionally flip categorical features.
    
    Uses domain knowledge to make plausible changes.
    """
    # Most categorical features should not be mutated randomly
    # Only allow specific transitions
    
    if "ECOG" in feature_name.upper():
        # ECOG can shift by ±1 level
        ecog_levels = ["Asymptomatic", "Symptoms but ambulatory", 
                      "In bed <50% of day", "In bed >50% of day", "Bedridden"]
        if value in ecog_levels:
            idx = ecog_levels.index(value)
            # Move up or down with equal probability
            if np.random.random() < 0.5 and idx > 0:
                return ecog_levels[idx - 1]  # Improve
            elif idx < len(ecog_levels) - 1:
                return ecog_levels[idx + 1]  # Worsen
    
    elif "tumor_burden" in feature_name.lower():
        # Tumor burden can shift
        burdens = ["low", "medium", "high"]
        if value in burdens:
            idx = burdens.index(value)
            if np.random.random() < 0.5 and idx > 0:
                return burdens[idx - 1]
            elif idx < len(burdens) - 1:
                return burdens[idx + 1]
    
    elif "trajectory" in feature_name.lower():
        # Weight trajectory can change
        trajectories = ["declining", "stable", "increasing"]
        if value in trajectories:
            idx = trajectories.index(value)
            if np.random.random() < 0.5 and idx > 0:
                return trajectories[idx - 1]
            elif idx < len(trajectories) - 1:
                return trajectories[idx + 1]
    
    # Default: no mutation for other categorical features
    return value


def get_default_feature_ranges() -> Dict[str, Tuple[float, float]]:
    """
    Return default valid ranges for numeric features.
    
    Based on clinical reference ranges and observed data.
    """
    return {
        # Demographics
        "AGE": (18, 100),
        "B_WEIGHT": (30, 150),  # kg
        
        # ECOG (if numeric)
        "B_ECOG": (0, 4),
        
        # Baseline labs
        "baseline_ALB": (2.0, 6.0),      # g/dL
        "baseline_ALP": (20, 500),        # U/L
        "baseline_CEA": (0, 1000),        # ng/mL
        "baseline_CREAT": (0.3, 3.0),     # mg/dL
        "baseline_HGB": (7, 20),          # g/dL
        "baseline_LDH": (50, 2000),       # U/L
        "baseline_PLT": (20, 600),        # K/uL
        "baseline_WBC": (0.5, 30),        # K/uL
        
        # Tumor characteristics
        "target_lesion_count": (0, 10),
        "nontarget_lesion_count": (0, 20),
        "sum_target_diameters": (0, 500),  # mm
        "max_lesion_size": (0, 200),       # mm
        "mean_lesion_size": (0, 100),      # mm
        "lesion_sites_count": (0, 10),
        
        # Physical measurements
        "weight_baseline": (30, 150),      # kg
        "weight_change_abs": (-30, 30),    # kg
        "weight_change_pct": (-50, 50),    # %
        
        # History
        "prior_ae_count": (0, 50),
        "prior_severe_ae_count": (0, 20),
        "num_prior_therapies": (0, 10),
        "time_since_diagnosis": (0, 300),  # months
        
        # Risk scores
        "lab_risk_score": (0, 1),
        "performance_risk": (0, 1),
        "tumor_burden_risk": (0, 1),
        "molecular_risk": (0, 1),
        "composite_risk_score": (0, 1),
        "predicted_good_prognosis_flag": (0, 1),
    }


def apply_correlated_mutations(
    profile: Dict[str, Any],
    mutation_rate: float = 0.05,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Apply mutations while maintaining feature correlations.
    
    For example:
    - If HGB decreases, also decrease HCT
    - If weight decreases significantly, adjust weight trajectory
    - If LDH increases, may increase tumor burden
    
    Args:
        profile: Patient profile
        mutation_rate: Mutation probability
        seed: Random seed
    
    Returns:
        Mutated profile with maintained correlations
    """
    if seed is not None:
        np.random.seed(seed)
    
    # First apply standard mutations
    mutated = apply_mutation(profile, mutation_rate=mutation_rate, seed=seed)
    
    # Then adjust correlated features
    
    # Weight and weight_change correlation
    if "B_WEIGHT" in mutated and "weight_baseline" in mutated:
        if "B_WEIGHT" in profile and "weight_baseline" in profile:
            weight_diff = mutated["B_WEIGHT"] - profile["B_WEIGHT"]
            if abs(weight_diff) > 1:  # Significant change
                # Adjust weight_change
                if "weight_change_abs" in mutated:
                    mutated["weight_change_abs"] = mutated.get("weight_change_abs", 0) + weight_diff
                if "weight_change_pct" in mutated and mutated.get("weight_baseline"):
                    mutated["weight_change_pct"] = (
                        mutated.get("weight_change_abs", 0) / mutated["weight_baseline"] * 100
                    )
                # Update trajectory
                if "weight_change_pct" in mutated:
                    pct = mutated["weight_change_pct"]
                    if pct < -5:
                        mutated["weight_trajectory"] = "declining"
                    elif pct > 5:
                        mutated["weight_trajectory"] = "increasing"
                    else:
                        mutated["weight_trajectory"] = "stable"
    
    # Performance status and composite risk correlation
    if "B_ECOG" in mutated and "performance_risk" in mutated:
        # Update performance risk based on ECOG
        ecog = mutated["B_ECOG"]
        if isinstance(ecog, str):
            ecog_map = {
                "Asymptomatic": 0,
                "Symptoms but ambulatory": 1,
                "In bed <50% of day": 2,
                "In bed >50% of day": 3,
                "Bedridden": 4,
            }
            ecog = ecog_map.get(ecog, 2)
        
        if ecog == 0:
            mutated["performance_risk"] = 0.0
        elif ecog in [1, 2]:
            mutated["performance_risk"] = 0.5
        else:
            mutated["performance_risk"] = 1.0
        
        # Recalculate composite risk score
        if all(k in mutated for k in ["lab_risk_score", "performance_risk", 
                                      "tumor_burden_risk", "molecular_risk"]):
            weights = {
                "lab_risk_score": 0.25,
                "performance_risk": 0.30,
                "tumor_burden_risk": 0.25,
                "molecular_risk": 0.20,
            }
            composite = sum(
                mutated.get(k, 0.5) * weights[k] 
                for k in weights.keys()
            )
            mutated["composite_risk_score"] = composite
            mutated["predicted_good_prognosis_flag"] = int(composite < 0.4)
    
    return mutated


def batch_mutate(
    profiles: pd.DataFrame,
    mutation_rate: float = 0.05,
    maintain_correlations: bool = True,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Apply mutations to a batch of profiles.
    
    Args:
        profiles: DataFrame of patient profiles
        mutation_rate: Mutation probability
        maintain_correlations: Whether to maintain feature correlations
        seed: Random seed
    
    Returns:
        DataFrame with mutated profiles
    """
    if seed is not None:
        np.random.seed(seed)
    
    mutated_profiles = []
    
    for idx, row in profiles.iterrows():
        profile = row.to_dict()
        
        # Apply mutations
        if maintain_correlations:
            mutated = apply_correlated_mutations(
                profile, 
                mutation_rate=mutation_rate,
                seed=seed + idx if seed else None
            )
        else:
            mutated = apply_mutation(
                profile,
                mutation_rate=mutation_rate,
                seed=seed + idx if seed else None
            )
        
        mutated_profiles.append(mutated)
    
    return pd.DataFrame(mutated_profiles)
