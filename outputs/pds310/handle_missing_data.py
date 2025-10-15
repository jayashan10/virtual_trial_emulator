"""
Utilities for intelligent missing data handling in PDS310.

This module provides functions to handle sparse features more appropriately
than blind imputation, which can introduce noise for highly sparse features.
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


def analyze_missingness(X: pd.DataFrame, verbose: bool = True) -> pd.Series:
    """
    Analyze missingness patterns in the feature matrix.
    
    Args:
        X: Feature matrix
        verbose: If True, print summary statistics
        
    Returns:
        Series with missingness proportion for each feature
    """
    missingness = X.isnull().mean().sort_values(ascending=False)
    
    if verbose:
        print("\n=== Missingness Analysis ===")
        print(f"Total features: {len(X.columns)}")
        print(f"Total patients: {len(X)}")
        print(f"\nMissingness distribution:")
        print(f"  0-10% missing:   {(missingness <= 0.1).sum()} features")
        print(f"  10-30% missing:  {((missingness > 0.1) & (missingness <= 0.3)).sum()} features")
        print(f"  30-50% missing:  {((missingness > 0.3) & (missingness <= 0.5)).sum()} features")
        print(f"  50-70% missing:  {((missingness > 0.5) & (missingness <= 0.7)).sum()} features")
        print(f"  70-90% missing:  {((missingness > 0.7) & (missingness <= 0.9)).sum()} features")
        print(f"  >90% missing:    {(missingness > 0.9).sum()} features")
        
        if (missingness > 0.7).any():
            print(f"\n⚠️  Features with >70% missing:")
            for feat, miss in missingness[missingness > 0.7].items():
                print(f"    {feat}: {miss*100:.1f}%")
    
    return missingness


def drop_sparse_features(
    X: pd.DataFrame, 
    threshold: float = 0.7,
    verbose: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop features with missingness above threshold.
    
    Args:
        X: Feature matrix
        threshold: Drop features with missingness proportion > threshold
        verbose: If True, print dropped features
        
    Returns:
        X_filtered: Feature matrix with sparse features removed
        dropped_features: List of dropped feature names
    """
    missingness = X.isnull().mean()
    features_to_drop = missingness[missingness > threshold].index.tolist()
    
    if verbose and features_to_drop:
        print(f"\n=== Dropping {len(features_to_drop)} features with >{threshold*100}% missing ===")
        for feat in features_to_drop:
            print(f"  ❌ {feat}: {missingness[feat]*100:.1f}% missing")
    
    X_filtered = X.drop(columns=features_to_drop)
    
    if verbose:
        print(f"\nFeatures: {len(X.columns)} → {len(X_filtered.columns)}")
    
    return X_filtered, features_to_drop


def add_missingness_indicators(
    X: pd.DataFrame,
    features_to_indicate: List[str] = None,
    min_missingness: float = 0.1,
    max_missingness: float = 0.7,
    verbose: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add binary indicators for whether features were measured.
    
    This is valuable because missingness itself can be informative - e.g., 
    certain labs may only be measured for patients with specific characteristics.
    
    Args:
        X: Feature matrix
        features_to_indicate: Explicit list of features to add indicators for.
                             If None, auto-select based on missingness thresholds.
        min_missingness: Only add indicators if feature has >= this much missingness
        max_missingness: Only add indicators if feature has <= this much missingness
                        (very sparse features should be dropped instead)
        verbose: If True, print added indicators
        
    Returns:
        X_with_indicators: Feature matrix with indicator columns added
        indicators_added: List of indicator column names that were added
    """
    X_with_indicators = X.copy()
    indicators_added = []
    
    if features_to_indicate is None:
        # Auto-select features with moderate missingness
        missingness = X.isnull().mean()
        features_to_indicate = missingness[
            (missingness >= min_missingness) & (missingness <= max_missingness)
        ].index.tolist()
    
    for feat in features_to_indicate:
        if feat not in X.columns:
            continue
            
        indicator_name = f"{feat}_was_measured"
        X_with_indicators[indicator_name] = (~X[feat].isna()).astype(int)
        indicators_added.append(indicator_name)
    
    if verbose and indicators_added:
        print(f"\n=== Added {len(indicators_added)} missingness indicators ===")
        for indicator in indicators_added:
            original_feat = indicator.replace('_was_measured', '')
            miss_pct = X[original_feat].isnull().mean() * 100
            print(f"  ✅ {indicator} (original: {miss_pct:.1f}% missing)")
    
    return X_with_indicators, indicators_added


def get_feature_recommendations(
    X: pd.DataFrame,
    conservative: bool = True
) -> Dict[str, List[str]]:
    """
    Analyze features and recommend which to drop vs. add indicators for.
    
    Args:
        X: Feature matrix
        conservative: If True, use stricter thresholds (drop >70% missing)
                     If False, use looser thresholds (drop >80% missing)
        
    Returns:
        Dictionary with 'drop', 'add_indicator', and 'keep_as_is' lists
    """
    missingness = X.isnull().mean()
    
    if conservative:
        drop_threshold = 0.7
        indicator_min = 0.15
        indicator_max = 0.7
    else:
        drop_threshold = 0.8
        indicator_min = 0.2
        indicator_max = 0.8
    
    recommendations = {
        'drop': missingness[missingness > drop_threshold].index.tolist(),
        'add_indicator': missingness[
            (missingness >= indicator_min) & (missingness <= indicator_max)
        ].index.tolist(),
        'keep_as_is': missingness[missingness < indicator_min].index.tolist()
    }
    
    print(f"\n=== Feature Recommendations ({'Conservative' if conservative else 'Liberal'}) ===")
    print(f"📊 Features to keep as-is (<{indicator_min*100}% missing): {len(recommendations['keep_as_is'])}")
    print(f"🎯 Features to add indicators ({indicator_min*100}-{indicator_max*100}% missing): {len(recommendations['add_indicator'])}")
    print(f"❌ Features to drop (>{drop_threshold*100}% missing): {len(recommendations['drop'])}")
    
    return recommendations


def prepare_features_with_intelligent_missing_handling(
    X: pd.DataFrame,
    drop_threshold: float = 0.7,
    add_indicators: bool = True,
    indicator_min: float = 0.15,
    indicator_max: float = 0.7,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Complete preprocessing pipeline with intelligent missing data handling.
    
    This function:
    1. Analyzes missingness patterns
    2. Drops extremely sparse features (>threshold missing)
    3. Adds missingness indicators for moderately sparse features
    4. Returns cleaned feature matrix with metadata
    
    Args:
        X: Raw feature matrix
        drop_threshold: Drop features with more than this proportion missing
        add_indicators: Whether to add missingness indicators
        indicator_min: Minimum missingness to add indicator
        indicator_max: Maximum missingness to add indicator
        verbose: Whether to print progress
        
    Returns:
        X_processed: Cleaned feature matrix
        metadata: Dictionary with:
            - 'dropped_features': List of dropped feature names
            - 'indicators_added': List of indicator column names
            - 'features_remaining': Number of features after processing
            - 'missingness_summary': Missingness statistics
    """
    if verbose:
        print("\n" + "="*60)
        print("Intelligent Missing Data Handling Pipeline")
        print("="*60)
    
    # Step 1: Analyze missingness
    missingness = analyze_missingness(X, verbose=verbose)
    
    # Step 2: Drop sparse features
    X_processed, dropped = drop_sparse_features(X, threshold=drop_threshold, verbose=verbose)
    
    # Step 3: Add missingness indicators
    indicators_added = []
    if add_indicators:
        # Only add indicators for features that weren't dropped
        features_for_indicators = missingness[
            (missingness >= indicator_min) & 
            (missingness <= indicator_max) &
            (missingness.index.isin(X_processed.columns))
        ].index.tolist()
        
        X_processed, indicators_added = add_missingness_indicators(
            X_processed, 
            features_to_indicate=features_for_indicators,
            verbose=verbose
        )
    
    # Step 4: Final summary
    if verbose:
        print(f"\n=== Processing Complete ===")
        print(f"Initial features: {len(X.columns)}")
        print(f"Features dropped: {len(dropped)}")
        print(f"Indicators added: {len(indicators_added)}")
        print(f"Final features: {len(X_processed.columns)}")
        
        # Show remaining missingness
        remaining_missingness = X_processed.isnull().mean()
        max_miss = remaining_missingness.max()
        print(f"Maximum remaining missingness: {max_miss*100:.1f}%")
    
    metadata = {
        'dropped_features': dropped,
        'indicators_added': indicators_added,
        'features_remaining': len(X_processed.columns),
        'missingness_summary': {
            'before': {
                'n_features': len(X.columns),
                'max_missingness': float(missingness.max()),
                'mean_missingness': float(missingness.mean())
            },
            'after': {
                'n_features': len(X_processed.columns),
                'max_missingness': float(X_processed.isnull().mean().max()),
                'mean_missingness': float(X_processed.isnull().mean().mean())
            }
        }
    }
    
    return X_processed, metadata


# Convenience function for quick analysis
def quick_missingness_report(profile_path: str):
    """
    Quick command-line tool to analyze missingness in patient profiles.
    
    Usage:
        python -c "from pds310.handle_missing_data import quick_missingness_report; \
                   quick_missingness_report('outputs/pds310/patient_profiles.csv')"
    """
    import pandas as pd
    
    df = pd.read_csv(profile_path)
    
    # Exclude ID and outcome columns
    exclude_cols = ['SUBJID', 'STUDYID', 'best_response', 'response_at_week8', 
                   'response_at_week16', 'DTHDYX', 'DTHX', 'PFSDYCR', 'PFSCR']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols]
    
    print(f"\n{'='*60}")
    print(f"Missingness Report: {profile_path}")
    print(f"{'='*60}")
    
    # Run analysis
    analyze_missingness(X, verbose=True)
    
    # Get recommendations
    recommendations = get_feature_recommendations(X, conservative=True)
    
    print(f"\n{'='*60}")
    print("Recommendations")
    print(f"{'='*60}")
    
    if recommendations['drop']:
        print("\n❌ Recommend DROPPING (>70% missing):")
        missingness = X.isnull().mean()
        for feat in recommendations['drop']:
            print(f"   {feat}: {missingness[feat]*100:.1f}% missing")
    
    if recommendations['add_indicator']:
        print("\n🎯 Recommend ADDING INDICATORS (15-70% missing):")
        missingness = X.isnull().mean()
        for feat in recommendations['add_indicator'][:10]:  # Show first 10
            print(f"   {feat}: {missingness[feat]*100:.1f}% missing")
        if len(recommendations['add_indicator']) > 10:
            print(f"   ... and {len(recommendations['add_indicator']) - 10} more")
    
    return X, recommendations
