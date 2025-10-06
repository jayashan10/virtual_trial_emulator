"""
Profile Database Management for PDS310 Digital Twin System.

This module handles storage, retrieval, and querying of patient digital profiles.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import json
from .digital_profile import create_complete_digital_profile, get_profile_feature_groups
from .io import ID_COL, STUDY_COL, load_adam_tables


def create_profile_database(
    tables: Dict[str, pd.DataFrame],
    output_path: Optional[str] = None,
    include_outcomes: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create complete database of patient digital profiles.
    
    Args:
        tables: Dictionary of ADaM DataFrames
        output_path: Path to save CSV file (optional)
        include_outcomes: Whether to include outcome features
        verbose: Print progress messages
    
    Returns:
        DataFrame with one row per patient and ~100 feature columns
    """
    if "adsl" not in tables or tables["adsl"] is None or tables["adsl"].empty:
        raise ValueError("ADSL table is required to create profile database")
    
    # Get list of all subjects
    subject_ids = tables["adsl"][ID_COL].unique()
    n_subjects = len(subject_ids)
    
    if verbose:
        print(f"Creating digital profiles for {n_subjects} patients...")
    
    # Build profiles for all subjects
    profiles = []
    for i, subjid in enumerate(subject_ids, 1):
        if verbose and i % 50 == 0:
            print(f"  Processed {i}/{n_subjects} patients...")
        
        profile = create_complete_digital_profile(
            subjid=subjid,
            tables=tables,
            include_outcomes=include_outcomes
        )
        profiles.append(profile)
    
    # Convert to DataFrame
    df = pd.DataFrame(profiles)
    
    # Ensure consistent column order
    feature_groups = get_profile_feature_groups()
    ordered_cols = []
    for group_name in ["identifiers", "demographics", "disease", "treatment",
                       "baseline_labs", "tumor", "molecular", "physical",
                       "history", "risk_scores", "outcomes"]:
        for col in feature_groups[group_name]:
            if col in df.columns:
                ordered_cols.append(col)
    
    # Add any remaining columns (longitudinal features)
    for col in df.columns:
        if col not in ordered_cols:
            ordered_cols.append(col)
    
    df = df[ordered_cols]
    
    # Save if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        if verbose:
            print(f"\nProfile database saved to: {output_path}")
            print(f"  - {len(df)} patients")
            print(f"  - {len(df.columns)} features")
    
    return df


def load_profile_database(path: str) -> pd.DataFrame:
    """
    Load pre-computed profile database from CSV.
    
    Args:
        path: Path to profile database CSV file
    
    Returns:
        DataFrame with patient profiles
    """
    df = pd.read_csv(path)
    
    # Ensure ID columns are strings
    if ID_COL in df.columns:
        df[ID_COL] = df[ID_COL].astype(str)
    if STUDY_COL in df.columns:
        df[STUDY_COL] = df[STUDY_COL].astype(str)
    
    return df


def get_profile_by_id(
    db: pd.DataFrame,
    subjid: str,
    as_dict: bool = True
) -> Any:
    """
    Retrieve single patient profile by ID.
    
    Args:
        db: Profile database DataFrame
        subjid: Subject identifier
        as_dict: Return as dictionary (default) or Series
    
    Returns:
        Patient profile as dictionary or Series
    """
    profile = db[db[ID_COL] == subjid]
    
    if profile.empty:
        raise ValueError(f"Subject {subjid} not found in database")
    
    if as_dict:
        return profile.iloc[0].to_dict()
    else:
        return profile.iloc[0]


def get_profiles_by_criteria(
    db: pd.DataFrame,
    criteria: Dict[str, Any],
    match_all: bool = True
) -> pd.DataFrame:
    """
    Filter profiles by criteria.
    
    Args:
        db: Profile database DataFrame
        criteria: Dictionary of {column: value} or {column: (min, max)} or {column: [list]}
        match_all: If True, require all criteria (AND logic); if False, any criteria (OR logic)
    
    Returns:
        Filtered DataFrame
    
    Examples:
        # Single values
        get_profiles_by_criteria(db, {"RAS_status": "WILD-TYPE", "B_ECOG": 0})
        
        # Ranges
        get_profiles_by_criteria(db, {"AGE": (50, 70), "baseline_LDH": (None, 250)})
        
        # Lists of values
        get_profiles_by_criteria(db, {"TRT": ["Panitumumab+BSC", "BSC"]})
    """
    if not criteria:
        return db.copy()
    
    masks = []
    
    for column, value in criteria.items():
        if column not in db.columns:
            raise ValueError(f"Column '{column}' not found in database")
        
        # Handle different value types
        if isinstance(value, (list, tuple)) and len(value) == 2 and not isinstance(value, str):
            # Could be a range (min, max) or a list
            # Check if it's a range by seeing if elements are numeric or None
            if all(v is None or isinstance(v, (int, float)) for v in value):
                # It's a range
                min_val, max_val = value
                mask = pd.Series([True] * len(db), index=db.index)
                if min_val is not None:
                    mask &= db[column] >= min_val
                if max_val is not None:
                    mask &= db[column] <= max_val
            else:
                # It's a list of values
                mask = db[column].isin(value)
        elif isinstance(value, list):
            # List of values
            mask = db[column].isin(value)
        else:
            # Single value
            mask = db[column] == value
        
        masks.append(mask)
    
    # Combine masks
    if match_all:
        # AND logic
        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask &= mask
    else:
        # OR logic
        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask |= mask
    
    return db[combined_mask].copy()


def get_database_summary(db: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics for profile database.
    
    Returns:
        Dictionary with database statistics
    """
    summary = {
        "n_patients": len(db),
        "n_features": len(db.columns),
        "feature_groups": {},
    }
    
    feature_groups = get_profile_feature_groups()
    
    for group_name, features in feature_groups.items():
        available_features = [f for f in features if f in db.columns]
        if available_features:
            summary["feature_groups"][group_name] = {
                "n_features": len(available_features),
                "features": available_features,
            }
    
    # Demographic summary
    if "AGE" in db.columns:
        summary["demographics"] = {
            "age_mean": float(db["AGE"].mean()),
            "age_std": float(db["AGE"].std()),
            "age_range": (float(db["AGE"].min()), float(db["AGE"].max())),
        }
    
    if "SEX" in db.columns:
        summary["sex_distribution"] = db["SEX"].value_counts().to_dict()
    
    if "RACE" in db.columns:
        summary["race_distribution"] = db["RACE"].value_counts().to_dict()
    
    # Treatment arm distribution
    if "TRT" in db.columns:
        summary["treatment_distribution"] = db["TRT"].value_counts().to_dict()
    
    # RAS status distribution
    if "RAS_status" in db.columns:
        summary["ras_distribution"] = db["RAS_status"].value_counts().to_dict()
    
    # ECOG distribution
    if "B_ECOG" in db.columns:
        summary["ecog_distribution"] = db["B_ECOG"].value_counts().sort_index().to_dict()
    
    # Outcome statistics
    if "DTHX" in db.columns:
        summary["os_events"] = int(db["DTHX"].sum())
        summary["os_censored"] = int((db["DTHX"] == 0).sum())
    
    if "DTHDYX" in db.columns:
        summary["os_median_days"] = float(db["DTHDYX"].median())
    
    if "PFSCR" in db.columns:
        summary["pfs_events"] = int(db["PFSCR"].sum())
    
    if "best_response" in db.columns:
        summary["response_distribution"] = db["best_response"].value_counts().to_dict()
    
    # Missing data summary
    missing_counts = db.isnull().sum()
    high_missing = missing_counts[missing_counts > len(db) * 0.5]  # >50% missing
    if len(high_missing) > 0:
        summary["high_missing_features"] = high_missing.to_dict()
    
    return summary


def export_database_summary(
    db: pd.DataFrame,
    output_path: str,
    format: str = "json"
) -> None:
    """
    Export database summary to file.
    
    Args:
        db: Profile database
        output_path: Path to save summary
        format: Output format ("json" or "txt")
    """
    summary = get_database_summary(db)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
    elif format == "txt":
        with open(output_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("PATIENT PROFILE DATABASE SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total Patients: {summary['n_patients']}\n")
            f.write(f"Total Features: {summary['n_features']}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("FEATURE GROUPS\n")
            f.write("-" * 80 + "\n")
            for group_name, group_info in summary.get("feature_groups", {}).items():
                f.write(f"\n{group_name.upper()}: {group_info['n_features']} features\n")
                for feat in group_info['features']:
                    f.write(f"  - {feat}\n")
            
            if "demographics" in summary:
                f.write("\n" + "-" * 80 + "\n")
                f.write("DEMOGRAPHICS\n")
                f.write("-" * 80 + "\n")
                demo = summary["demographics"]
                f.write(f"Age: {demo['age_mean']:.1f} ± {demo['age_std']:.1f} years\n")
                f.write(f"Age Range: {demo['age_range'][0]:.0f} - {demo['age_range'][1]:.0f}\n")
            
            if "sex_distribution" in summary:
                f.write("\nSex Distribution:\n")
                for sex, count in summary["sex_distribution"].items():
                    f.write(f"  {sex}: {count}\n")
            
            if "treatment_distribution" in summary:
                f.write("\n" + "-" * 80 + "\n")
                f.write("TREATMENT ARMS\n")
                f.write("-" * 80 + "\n")
                for trt, count in summary["treatment_distribution"].items():
                    f.write(f"  {trt}: {count}\n")
            
            if "ras_distribution" in summary:
                f.write("\n" + "-" * 80 + "\n")
                f.write("RAS MUTATION STATUS\n")
                f.write("-" * 80 + "\n")
                for status, count in summary["ras_distribution"].items():
                    f.write(f"  {status}: {count}\n")
            
            if "os_events" in summary:
                f.write("\n" + "-" * 80 + "\n")
                f.write("SURVIVAL OUTCOMES\n")
                f.write("-" * 80 + "\n")
                f.write(f"OS Events: {summary['os_events']}\n")
                f.write(f"OS Censored: {summary['os_censored']}\n")
                if "os_median_days" in summary:
                    f.write(f"OS Median: {summary['os_median_days']:.1f} days\n")
            
            if "response_distribution" in summary:
                f.write("\n" + "-" * 80 + "\n")
                f.write("RESPONSE OUTCOMES\n")
                f.write("-" * 80 + "\n")
                for response, count in summary["response_distribution"].items():
                    f.write(f"  {response}: {count}\n")
    else:
        raise ValueError(f"Unknown format: {format}. Use 'json' or 'txt'")
    
    print(f"Database summary saved to: {output_path}")


def split_train_test(
    db: pd.DataFrame,
    test_size: float = 0.2,
    stratify_col: Optional[str] = None,
    random_state: int = 42
) -> tuple:
    """
    Split profile database into train/test sets.
    
    Args:
        db: Profile database
        test_size: Fraction for test set (default: 0.2)
        stratify_col: Column to use for stratified split (e.g., "TRT", "RAS_status")
        random_state: Random seed
    
    Returns:
        (train_df, test_df) tuple
    """
    from sklearn.model_selection import train_test_split
    
    if stratify_col and stratify_col in db.columns:
        stratify = db[stratify_col]
    else:
        stratify = None
    
    train_df, test_df = train_test_split(
        db,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state
    )
    
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
