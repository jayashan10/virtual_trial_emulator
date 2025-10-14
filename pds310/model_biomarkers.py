"""
Biomarker Trajectory Prediction for PDS310.

Predicts longitudinal changes in key biomarkers (CEA, LDH, HGB, etc.)
over treatment course.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
from pathlib import Path


def prepare_biomarker_trajectories(
    profile_db: pd.DataFrame,
    adlb: pd.DataFrame,
    biomarker: str = "LDH",
    timepoints: List[int] = [28, 56, 84, 112]  # Days (weeks 4, 8, 12, 16)
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare data for biomarker trajectory prediction.
    
    Args:
        profile_db: Patient profiles with baseline features
        adlb: Longitudinal lab data
        biomarker: Lab test name (e.g., "LDH", "CEA", "HGB")
        timepoints: Days at which to predict values
    
    Returns:
        X: Feature matrix (baseline features)
        y: Target matrix (biomarker values at each timepoint)
    """
    from .io import ID_COL, STUDY_COL
    
    # Filter adlb to biomarker of interest
    biomarker_data = adlb[
        adlb["LBTEST"].astype(str).str.upper() == biomarker.upper()
    ].copy()
    
    if biomarker_data.empty:
        raise ValueError(f"No data found for biomarker: {biomarker}")
    
    biomarker_data["VISITDY"] = pd.to_numeric(biomarker_data["VISITDY"], errors="coerce")
    biomarker_data["VALUE"] = pd.to_numeric(biomarker_data["LBSTRESN"], errors="coerce")
    biomarker_data = biomarker_data.dropna(subset=["VISITDY", "VALUE"])
    
    # For each patient, get values at target timepoints (±7 days window)
    target_values = []
    
    for subjid in profile_db[ID_COL].unique():
        patient_data = biomarker_data[biomarker_data[ID_COL] == subjid]
        
        if patient_data.empty:
            continue
        
        row = {ID_COL: subjid}
        has_data = False
        
        for timepoint in timepoints:
            # Look for measurement within ±7 days
            window_data = patient_data[
                patient_data["VISITDY"].between(timepoint - 7, timepoint + 7)
            ]
            
            if not window_data.empty:
                # Use closest measurement
                closest = window_data.iloc[
                    (window_data["VISITDY"] - timepoint).abs().argmin()
                ]
                row[f"{biomarker}_day{timepoint}"] = closest["VALUE"]
                has_data = True
            else:
                row[f"{biomarker}_day{timepoint}"] = None
        
        if has_data:
            target_values.append(row)
    
    # Create target DataFrame
    y_df = pd.DataFrame(target_values)
    
    if y_df.empty:
        raise ValueError(f"No trajectory data found for {biomarker}")
    
    # Merge with baseline features
    X = profile_db.merge(y_df[[ID_COL]], on=ID_COL, how="inner")
    y = y_df
    
    print(f"Biomarker trajectory data for {biomarker}:")
    print(f"  Patients: {len(y)}")
    for tp in timepoints:
        col = f"{biomarker}_day{tp}"
        if col in y.columns:
            n_obs = y[col].notna().sum()
            print(f"  Day {tp}: {n_obs} observations")
    
    return X, y, timepoints


def train_biomarker_model(
    X: pd.DataFrame,
    y: pd.DataFrame,
    biomarker: str,
    timepoint: int,
    model_type: str = "rf",
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train model to predict biomarker at specific timepoint.
    
    Args:
        X: Feature matrix
        y: Target DataFrame with biomarker values
        biomarker: Biomarker name
        timepoint: Day to predict
        model_type: Model type ("rf", "gb", "linear")
        random_state: Random seed
    
    Returns:
        Dictionary with trained model and metrics
    """
    from .io import ID_COL, STUDY_COL
    
    # Get target column
    target_col = f"{biomarker}_day{timepoint}"
    
    if target_col not in y.columns:
        raise ValueError(f"Target {target_col} not found in y DataFrame")
    
    # Filter to patients with this timepoint
    valid_idx = y[target_col].notna()
    y_target = y.loc[valid_idx, target_col]
    X_subset = X.loc[valid_idx].copy()
    
    if len(y_target) < 10:
        raise ValueError(f"Insufficient data for {biomarker} day {timepoint}: only {len(y_target)} samples")
    
    print(f"\nTraining model for {biomarker} at day {timepoint}:")
    print(f"  Samples: {len(y_target)}")
    print(f"  Mean: {y_target.mean():.2f}")
    print(f"  Range: {y_target.min():.2f} - {y_target.max():.2f}")
    
    # Prepare features
    features_to_exclude = [
        ID_COL, STUDY_COL,
        'best_response', 'response_at_week8', 'response_at_week16', 'time_to_response',
        'DTHDYX', 'DTHX', 'PFSDYCR', 'PFSCR'
    ]
    
    feature_cols = [c for c in X_subset.columns if c not in features_to_exclude]
    X_features = X_subset[feature_cols].copy()
    
    # Handle categorical
    categorical_cols = X_features.select_dtypes(include=['object', 'string']).columns
    for col in categorical_cols:
        X_features[col] = pd.Categorical(X_features[col]).codes
    
    # Fill missing
    numeric_cols = X_features.select_dtypes(include=[np.number]).columns
    X_features[numeric_cols] = X_features[numeric_cols].fillna(X_features[numeric_cols].median())
    
    # Select model
    if model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        )
    elif model_type == "gb":
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
            model = HistGradientBoostingRegressor(
                max_iter=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=random_state
            )
        except ImportError:
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=random_state
            )
    elif model_type == "linear":
        model = LinearRegression()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Cross-validation (if enough samples)
    if len(y_target) >= 20:
        cv = KFold(n_splits=min(5, len(y_target) // 4), shuffle=True, random_state=random_state)
        cv_scores_r2 = cross_val_score(model, X_features, y_target, cv=cv, scoring='r2')
        cv_scores_mae = cross_val_score(model, X_features, y_target, cv=cv, scoring='neg_mean_absolute_error')
        
        print(f"  CV R²: {cv_scores_r2.mean():.3f} ± {cv_scores_r2.std():.3f}")
        print(f"  CV MAE: {-cv_scores_mae.mean():.2f} ± {cv_scores_mae.std():.2f}")
    else:
        cv_scores_r2 = None
        cv_scores_mae = None
        print(f"  Skipping CV (too few samples)")
    
    # Train final model
    model.fit(X_features, y_target)
    
    # Get predictions
    y_pred = model.predict(X_features)
    
    # Calculate metrics
    r2 = r2_score(y_target, y_pred)
    mae = mean_absolute_error(y_target, y_pred)
    rmse = np.sqrt(mean_squared_error(y_target, y_pred))
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_features.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        feature_importance = None
    
    results = {
        'model': model,
        'model_type': model_type,
        'biomarker': biomarker,
        'timepoint': timepoint,
        'cv_scores_r2': cv_scores_r2,
        'cv_r2_mean': float(cv_scores_r2.mean()) if cv_scores_r2 is not None else None,
        'cv_r2_std': float(cv_scores_r2.std()) if cv_scores_r2 is not None else None,
        'cv_mae_mean': float(-cv_scores_mae.mean()) if cv_scores_mae is not None else None,
        'train_r2': float(r2),
        'train_mae': float(mae),
        'train_rmse': float(rmse),
        'feature_importance': feature_importance,
        'feature_names': X_features.columns.tolist(),
        'n_samples': len(y_target),
        'target_mean': float(y_target.mean()),
        'target_std': float(y_target.std()),
    }
    
    return results


def train_biomarker_trajectory_models(
    profile_db: pd.DataFrame,
    adlb: pd.DataFrame,
    biomarkers: List[str] = ["LDH", "HGB", "CEA"],
    timepoints: List[int] = [56, 112],  # Weeks 8, 16
    model_type: str = "rf",
    random_state: int = 42
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Train models for multiple biomarkers at multiple timepoints.
    
    Args:
        profile_db: Patient profiles
        adlb: Longitudinal lab data
        biomarkers: List of biomarkers to model
        timepoints: Days to predict
        model_type: Model type
        random_state: Random seed
    
    Returns:
        Nested dict: {biomarker: {timepoint: model_results}}
    """
    all_models = {}
    
    for biomarker in biomarkers:
        print(f"\n{'='*80}")
        print(f"Training models for {biomarker}")
        print(f"{'='*80}")
        
        try:
            # Prepare data
            X, y, actual_timepoints = prepare_biomarker_trajectories(
                profile_db, adlb, biomarker, timepoints
            )
            
            biomarker_models = {}
            
            # Train model for each timepoint
            for tp in actual_timepoints:
                try:
                    results = train_biomarker_model(
                        X, y, biomarker, tp,
                        model_type=model_type,
                        random_state=random_state
                    )
                    biomarker_models[tp] = results
                    
                except Exception as e:
                    print(f"  Failed to train {biomarker} day {tp}: {e}")
                    continue
            
            if biomarker_models:
                all_models[biomarker] = biomarker_models
                print(f"\n✅ {biomarker}: Trained {len(biomarker_models)} timepoint models")
            else:
                print(f"\n❌ {biomarker}: No models trained")
        
        except Exception as e:
            print(f"❌ {biomarker}: Failed to prepare data - {e}")
            continue
    
    return all_models


def predict_biomarker_trajectory(
    models: Dict[int, Dict[str, Any]],
    profile: pd.DataFrame,
    biomarker: str
) -> Dict[int, Dict[str, float]]:
    """
    Predict biomarker trajectory for a patient.
    
    Args:
        models: Dictionary of {timepoint: model_results}
        profile: Patient profile
        biomarker: Biomarker name
    
    Returns:
        Dictionary of {timepoint: {'predicted': value, 'lower': ci_low, 'upper': ci_high}}
    """
    predictions = {}
    
    for timepoint, model_data in models.items():
        model = model_data['model']
        feature_names = model_data['feature_names']
        
        # Prepare features
        if isinstance(profile, dict):
            profile = pd.DataFrame([profile])
        
        X = profile[feature_names].copy()
        
        # Handle categorical
        categorical_cols = X.select_dtypes(include=['object', 'string']).columns
        for col in categorical_cols:
            X[col] = pd.Categorical(X[col]).codes
        
        # Fill missing
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        
        # Predict
        pred = model.predict(X)[0]
        
        # Estimate confidence interval (use training RMSE)
        rmse = model_data['train_rmse']
        
        predictions[timepoint] = {
            'predicted': float(pred),
            'lower_95ci': float(pred - 1.96 * rmse),
            'upper_95ci': float(pred + 1.96 * rmse),
            'uncertainty': float(rmse),
        }
    
    return predictions


def save_biomarker_models(
    all_models: Dict[str, Dict[int, Dict[str, Any]]],
    output_path: str
):
    """Save biomarker trajectory models."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(all_models, output_path)
    print(f"\nBiomarker models saved to: {output_path}")


def load_biomarker_models(model_path: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Load biomarker trajectory models."""
    models = joblib.load(model_path)
    print(f"Loaded biomarker models from: {model_path}")
    
    for biomarker, timepoint_models in models.items():
        print(f"  {biomarker}: {len(timepoint_models)} timepoints")
    
    return models


def plot_biomarker_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    biomarker: str,
    timepoint: int,
    output_path: Optional[str] = None
):
    """Plot biomarker predictions."""
    import matplotlib.pyplot as plt
    
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.5, color='steelblue')
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
    
    ax.set_xlabel(f'Actual {biomarker}')
    ax.set_ylabel(f'Predicted {biomarker}')
    ax.set_title(f'{biomarker} at Day {timepoint}\nR² = {r2:.3f}, MAE = {mae:.2f}')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Residuals
    ax = axes[1]
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, alpha=0.5, color='coral')
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel(f'Predicted {biomarker}')
    ax.set_ylabel('Residuals')
    ax.set_title('Residual Plot')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Biomarker plot saved to: {output_path}")
    
    plt.close()
