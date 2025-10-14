"""
Digital Twin Generator for PDS310.

This module creates synthetic patients through intelligent recombination of real
patient profiles, following the CAMP methodology for digital twin generation.
"""

from typing import Dict, Any, List, Optional, Literal
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from .io import ID_COL, STUDY_COL
from .digital_profile import get_profile_feature_groups


def generate_digital_twin(
    profile_db: pd.DataFrame,
    strategy: Literal["random", "cluster", "arm_specific", "subgroup"] = "random",
    constraints: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate single digital twin by recombining real patient profiles.
    
    Strategies:
    - 'random': Randomly select features from different patients
    - 'cluster': Select from similar patients (k-means clustering)
    - 'arm_specific': Generate twins matching a specific treatment arm
    - 'subgroup': Generate twins matching specific criteria (e.g., KRAS_WT)
    
    Args:
        profile_db: DataFrame with patient profiles
        strategy: Recombination strategy
        constraints: Dictionary of constraints (e.g., {"RAS_status": "WILD-TYPE"})
        seed: Random seed for reproducibility
        **kwargs: Strategy-specific parameters
    
    Returns:
        Dictionary containing synthetic patient profile
    
    Example:
        # Generate random twin
        twin = generate_digital_twin(profile_db, strategy="random", seed=42)
        
        # Generate RAS wild-type twin
        twin = generate_digital_twin(
            profile_db, 
            strategy="subgroup",
            constraints={"RAS_status": "WILD-TYPE", "B_ECOG": [0, 1]}
        )
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Apply constraints to filter donor pool
    donor_pool = profile_db.copy()
    if constraints:
        for key, value in constraints.items():
            if key in donor_pool.columns:
                if isinstance(value, list):
                    donor_pool = donor_pool[donor_pool[key].isin(value)]
                else:
                    donor_pool = donor_pool[donor_pool[key] == value]
    
    if len(donor_pool) < 2:
        raise ValueError(f"Insufficient donors after applying constraints. Found {len(donor_pool)} patients.")
    
    # Select strategy
    if strategy == "random":
        return _generate_random_twin(donor_pool, seed=seed)
    elif strategy == "cluster":
        n_clusters = kwargs.get("n_clusters", 5)
        return _generate_cluster_twin(donor_pool, n_clusters=n_clusters, seed=seed)
    elif strategy == "arm_specific":
        treatment_arm = kwargs.get("treatment_arm", None)
        if not treatment_arm:
            raise ValueError("Must specify 'treatment_arm' for arm_specific strategy")
        return _generate_arm_specific_twin(donor_pool, treatment_arm=treatment_arm, seed=seed)
    elif strategy == "subgroup":
        return _generate_subgroup_twin(donor_pool, constraints=constraints, seed=seed)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _generate_random_twin(
    donor_pool: pd.DataFrame,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate twin by randomly mixing features from different patients.
    
    Strategy:
    1. Select base patient randomly
    2. For correlated features (demographics), keep from base
    3. For independent features, randomly select from 2-3 donors
    4. For molecular features, keep from base (genetics fixed)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Select base patient
    base_idx = np.random.randint(0, len(donor_pool))
    base_patient = donor_pool.iloc[base_idx].to_dict()
    
    # Select 2-3 additional donors for mixing
    n_donors = min(3, len(donor_pool) - 1)
    donor_indices = np.random.choice(
        [i for i in range(len(donor_pool)) if i != base_idx],
        size=n_donors,
        replace=False
    )
    donors = [donor_pool.iloc[idx].to_dict() for idx in donor_indices]
    
    # Create twin profile
    twin = {}
    feature_groups = get_profile_feature_groups()
    
    # Always keep from base: identifiers (but generate new ID)
    twin[STUDY_COL] = "PDS310_TWIN"
    twin[ID_COL] = f"TWIN_{np.random.randint(100000, 999999)}"
    
    # Correlated features: keep from base patient
    correlated_groups = ["demographics", "disease", "treatment", "molecular"]
    for group in correlated_groups:
        for feature in feature_groups.get(group, []):
            if feature in base_patient and feature not in [ID_COL, STUDY_COL]:
                twin[feature] = base_patient[feature]
    
    # Independent features: mix from donors
    independent_groups = ["baseline_labs", "history"]
    for group in independent_groups:
        for feature in feature_groups.get(group, []):
            # Randomly pick from base or donors
            all_sources = [base_patient] + donors
            source = np.random.choice(all_sources)
            if feature in source:
                twin[feature] = source[feature]
    
    # Tumor characteristics: mix from patients with similar burden
    if "tumor_burden_category" in base_patient:
        base_burden = base_patient["tumor_burden_category"]
        # Find donors with similar burden
        similar_donors = [
            d for d in donors 
            if d.get("tumor_burden_category") == base_burden
        ]
        if similar_donors:
            tumor_source = np.random.choice(similar_donors)
        else:
            tumor_source = base_patient
        
        for feature in feature_groups.get("tumor", []):
            if feature in tumor_source:
                twin[feature] = tumor_source[feature]
    
    # Risk scores: will be recomputed later
    for feature in feature_groups.get("risk_scores", []):
        if feature in base_patient:
            twin[feature] = base_patient[feature]
    
    # Longitudinal labs: mix from donors
    # Get all longitudinal lab features (not in named groups)
    all_group_features = set()
    for group_features in feature_groups.values():
        all_group_features.update(group_features)
    
    longitudinal_features = [
        col for col in donor_pool.columns 
        if col.startswith("lab_") and col not in all_group_features
    ]
    
    for feature in longitudinal_features:
        all_sources = [base_patient] + donors
        source = np.random.choice(all_sources)
        if feature in source:
            twin[feature] = source[feature]
    
    # Outcomes: DO NOT include (will be predicted)
    # Skip outcome features
    
    return twin


def _generate_cluster_twin(
    donor_pool: pd.DataFrame,
    n_clusters: int = 5,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate twin by selecting from similar patients (cluster-based).
    
    Strategy:
    1. Cluster donor pool using k-means on numeric features
    2. Select random cluster
    3. Mix features from patients within that cluster
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Select numeric features for clustering
    numeric_cols = donor_pool.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in [ID_COL]]
    
    if len(numeric_cols) < 2:
        # Fallback to random if insufficient numeric features
        return _generate_random_twin(donor_pool, seed=seed)
    
    # Prepare data for clustering
    X = donor_pool[numeric_cols].fillna(donor_pool[numeric_cols].median())
    
    # Perform k-means clustering
    n_clusters = min(n_clusters, len(donor_pool) // 2)  # Ensure sufficient samples per cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # Select random cluster
    selected_cluster = np.random.randint(0, n_clusters)
    cluster_indices = np.where(cluster_labels == selected_cluster)[0]
    
    # Select patients from this cluster
    cluster_pool = donor_pool.iloc[cluster_indices]
    
    # Generate twin from cluster members
    return _generate_random_twin(cluster_pool, seed=seed)


def _generate_arm_specific_twin(
    donor_pool: pd.DataFrame,
    treatment_arm: str,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate twin matching specific treatment arm.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Filter to treatment arm
    if "ATRT" in donor_pool.columns:
        arm_pool = donor_pool[donor_pool["ATRT"] == treatment_arm]
        if len(arm_pool) < 2:
            raise ValueError(f"Insufficient patients in treatment arm '{treatment_arm}'")
    else:
        arm_pool = donor_pool
    
    # Generate twin from arm-specific pool
    twin = _generate_random_twin(arm_pool, seed=seed)
    
    # Ensure treatment arm is set correctly
    twin["ATRT"] = treatment_arm
    
    return twin


def _generate_subgroup_twin(
    donor_pool: pd.DataFrame,
    constraints: Dict[str, Any],
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate twin matching subgroup criteria.
    (Already filtered by constraints in main function)
    """
    return _generate_random_twin(donor_pool, seed=seed)


def generate_twin_cohort(
    profile_db: pd.DataFrame,
    n_twins: int = 1000,
    strategy: str = "random",
    constraints: Optional[Dict[str, Any]] = None,
    validation_threshold: float = 0.0,
    seed: Optional[int] = None,
    verbose: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Generate cohort of digital twins.
    
    Args:
        profile_db: DataFrame with real patient profiles
        n_twins: Number of twins to generate
        strategy: Recombination strategy
        constraints: Constraints for donor selection
        validation_threshold: Minimum validation score (0-1) to accept twin
        seed: Random seed
        verbose: Print progress
        **kwargs: Strategy-specific parameters
    
    Returns:
        DataFrame with synthetic twin cohort
    """
    if seed is not None:
        np.random.seed(seed)
    
    if verbose:
        print(f"Generating {n_twins} digital twins...")
        print(f"Strategy: {strategy}")
        if constraints:
            print(f"Constraints: {constraints}")
    
    twins = []
    n_rejected = 0
    
    for i in range(n_twins):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{n_twins} twins...")
        
        # Generate twin
        twin_seed = seed + i if seed is not None else None
        
        try:
            twin = generate_digital_twin(
                profile_db=profile_db,
                strategy=strategy,
                constraints=constraints,
                seed=twin_seed,
                **kwargs
            )
            
            # Validate if threshold > 0
            if validation_threshold > 0:
                from .twin_validator import validate_twin
                validation = validate_twin(twin, profile_db)
                
                if validation["validation_score"] < validation_threshold:
                    n_rejected += 1
                    # Retry once
                    twin = generate_digital_twin(
                        profile_db=profile_db,
                        strategy=strategy,
                        constraints=constraints,
                        seed=twin_seed + 1000 if twin_seed else None,
                        **kwargs
                    )
            
            twins.append(twin)
            
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to generate twin {i+1}: {e}")
            n_rejected += 1
            continue
    
    if verbose:
        print(f"\nGenerated {len(twins)} twins successfully")
        if n_rejected > 0:
            print(f"Rejected/failed: {n_rejected} twins")
    
    # Convert to DataFrame
    twins_df = pd.DataFrame(twins)
    
    # Ensure consistent column order with real profiles
    common_cols = [c for c in profile_db.columns if c in twins_df.columns]
    twins_df = twins_df[common_cols]
    
    return twins_df


def get_twin_statistics(
    twins_df: pd.DataFrame,
    real_profiles: pd.DataFrame
) -> Dict[str, Any]:
    """
    Calculate statistics comparing twin cohort to real patients.
    
    Returns:
        Dictionary with comparison statistics
    """
    stats = {
        "n_twins": len(twins_df),
        "n_real": len(real_profiles),
        "feature_counts": {},
        "distribution_comparisons": {},
    }
    
    # Compare feature counts
    stats["feature_counts"]["twins"] = len(twins_df.columns)
    stats["feature_counts"]["real"] = len(real_profiles.columns)
    stats["feature_counts"]["shared"] = len(
        set(twins_df.columns).intersection(set(real_profiles.columns))
    )
    
    # Compare distributions for key numeric features
    numeric_features = ["AGE", "B_WEIGHT", "baseline_HGB", "composite_risk_score"]
    
    for feature in numeric_features:
        if feature in twins_df.columns and feature in real_profiles.columns:
            twins_values = twins_df[feature].dropna()
            real_values = real_profiles[feature].dropna()
            
            if len(twins_values) > 0 and len(real_values) > 0:
                stats["distribution_comparisons"][feature] = {
                    "twin_mean": float(twins_values.mean()),
                    "real_mean": float(real_values.mean()),
                    "twin_std": float(twins_values.std()),
                    "real_std": float(real_values.std()),
                    "mean_diff": float(abs(twins_values.mean() - real_values.mean())),
                }
    
    # Compare categorical distributions
    categorical_features = ["SEX", "RAS_status", "ATRT", "B_ECOG"]
    
    for feature in categorical_features:
        if feature in twins_df.columns and feature in real_profiles.columns:
            twin_counts = twins_df[feature].value_counts(normalize=True).to_dict()
            real_counts = real_profiles[feature].value_counts(normalize=True).to_dict()
            
            stats["distribution_comparisons"][f"{feature}_twins"] = twin_counts
            stats["distribution_comparisons"][f"{feature}_real"] = real_counts
    
    return stats
