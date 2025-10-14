"""
Feature engineering for treatment effect modeling.
"""

import pandas as pd
import numpy as np


def add_treatment_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction features between treatment and key baseline characteristics.
    This helps models learn heterogeneous treatment effects.
    
    Args:
        df: DataFrame with TRT/ATRT and baseline features
        
    Returns:
        DataFrame with added interaction features
    """
    df_enhanced = df.copy()
    
    # Check if treatment column exists
    trt_col = None
    if 'TRT' in df.columns:
        trt_col = 'TRT'
    elif 'ATRT' in df.columns:
        trt_col = 'ATRT'
    else:
        return df_enhanced  # No treatment column, return as-is
    
    # Create binary treatment indicator (1 = Panitumumab, 0 = BSC)
    df_enhanced['treatment_indicator'] = df_enhanced[trt_col].apply(
        lambda x: 1 if 'panit' in str(x).lower() else 0
    )
    
    # Key features to interact with treatment
    interaction_features = {
        # Performance status
        'B_ECOG': 'trt_x_ecog',
        
        # Lab values (prognostic)
        'baseline_LDH': 'trt_x_ldh',
        'baseline_CEA': 'trt_x_cea',
        'baseline_HGB': 'trt_x_hgb',
        'baseline_ALB': 'trt_x_alb',
        
        # Tumor burden
        'sum_target_diameters': 'trt_x_tumor_burden',
        'target_lesion_count': 'trt_x_lesion_count',
        
        # Time factors
        'time_since_diagnosis': 'trt_x_time_since_dx',
        
        # Age
        'AGE': 'trt_x_age',
    }
    
    for base_feat, interaction_name in interaction_features.items():
        if base_feat in df_enhanced.columns:
            # Create interaction: treatment × baseline feature
            df_enhanced[interaction_name] = (
                df_enhanced['treatment_indicator'] * df_enhanced[base_feat].fillna(0)
            )
    
    # Remove the binary indicator (we keep the original TRT/ATRT)
    df_enhanced = df_enhanced.drop(columns=['treatment_indicator'])
    
    return df_enhanced


def stratify_by_treatment(df: pd.DataFrame) -> dict:
    """
    Split data by treatment arm for arm-specific modeling.
    
    Returns:
        Dictionary with 'treatment' and 'control' DataFrames
    """
    trt_col = 'TRT' if 'TRT' in df.columns else 'ATRT'
    
    treatment_patients = df[
        df[trt_col].str.lower().str.contains('panit', na=False)
    ].copy()
    
    control_patients = df[
        df[trt_col].str.lower().str.contains('bsc', na=False) & 
        ~df[trt_col].str.lower().str.contains('panit', na=False)
    ].copy()
    
    return {
        'treatment': treatment_patients,
        'control': control_patients
    }
