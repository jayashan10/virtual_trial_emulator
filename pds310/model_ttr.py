"""
Time-to-Response (TTR) Regression Model for PDS310.

Predicts days from treatment start to first CR or PR.
Following CAMP methodology target: R² ≥ 0.80.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


def prepare_ttr_data(
    profile_db: pd.DataFrame,
    exclude_outcomes: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for time-to-response regression (responders only).
    """
    df = profile_db[
        (profile_db["time_to_response"].notna()) & (profile_db["time_to_response"] > 0)
    ].copy()
    if df.empty:
        raise ValueError("No patients with time_to_response data found.")

    print(f"Found {len(df)} responders with TTR data")

    y = pd.to_numeric(df["time_to_response"], errors="coerce")

    exclude_cols = {
        "SUBJID",
        "STUDYID",
        "best_response",
        "response_at_week8",
        "response_at_week16",
        "time_to_response",
    }
    if exclude_outcomes:
        exclude_cols.update({"DTHDYX", "DTHX", "PFSDYCR", "PFSCR"})
    exclude_cols.update({
        "lab_risk_score",
        "performance_risk",
        "tumor_burden_risk",
        "molecular_risk",
        "composite_risk_score",
        "predicted_good_prognosis_flag",
    })

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].copy()

    all_missing = [col for col in X.columns if X[col].isna().all()]
    if all_missing:
        X = X.drop(columns=all_missing)

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        X[col] = X[col].astype("string")

    return X, y


def _split_feature_types(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    X_clean = X.copy()

    bool_cols = X_clean.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        X_clean[bool_cols] = X_clean[bool_cols].astype(float)

    numeric_cols = X_clean.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if X_clean[c].notna().any()]

    categorical_cols = [c for c in X_clean.columns if c not in numeric_cols]
    categorical_cols = [c for c in categorical_cols if X_clean[c].notna().any()]
    for col in categorical_cols:
        X_clean[col] = X_clean[col].astype("object")
        X_clean[col] = X_clean[col].where(X_clean[col].notna(), np.nan)

    ordered_cols = numeric_cols + categorical_cols
    return X_clean[ordered_cols], numeric_cols, categorical_cols


def _build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> Optional[ColumnTransformer]:
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            )
        )
    if not transformers:
        return None
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def _get_pipeline_feature_names(
    pipeline: Pipeline,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> np.ndarray:
    preprocess = pipeline.named_steps.get("preprocess")
    if preprocess is not None:
        feature_names = preprocess.get_feature_names_out()
    else:
        feature_names = np.array(numeric_cols + categorical_cols, dtype=object)

    variance = pipeline.named_steps.get("variance")
    if variance is not None:
        feature_names = np.asarray(feature_names)[variance.get_support()]
    return feature_names


def _prepare_prediction_frame(
    profile: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    ordered_cols: List[str],
) -> pd.DataFrame:
    X = profile.copy()
    for col in numeric_cols:
        if col not in X.columns:
            X[col] = np.nan
    for col in categorical_cols:
        if col not in X.columns:
            X[col] = pd.Series([np.nan] * len(X), dtype="object")

    X = X.reindex(columns=ordered_cols)

    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        X[bool_cols] = X[bool_cols].astype(float)
    for col in categorical_cols:
        X[col] = X[col].astype("object")
        X[col] = X[col].where(X[col].notna(), np.nan)
    return X


def prepare_features_for_model(
    model_bundle: Dict[str, Any],
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare raw profile dataframe for the stored TTR model pipeline.
    """
    if isinstance(profiles, dict):
        profiles = pd.DataFrame([profiles])
    numeric_cols: List[str] = model_bundle.get("numeric_features", [])
    categorical_cols: List[str] = model_bundle.get("categorical_features", [])
    ordered_cols: List[str] = model_bundle.get("raw_feature_names", numeric_cols + categorical_cols)
    return _prepare_prediction_frame(profiles.copy(), numeric_cols, categorical_cols, ordered_cols)


def train_ttr_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "rf",
    cv_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train time-to-response regression model.
    """
    X_clean, numeric_cols, categorical_cols = _split_feature_types(X)
    preprocessor = _build_preprocessor(numeric_cols, categorical_cols)

    if model_type == "xgboost":
        try:
            from xgboost import XGBRegressor

            estimator = XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
            )
        except Exception:
            print("XGBoost not available, falling back to GradientBoostingRegressor.")
            estimator = GradientBoostingRegressor(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.05,
                random_state=random_state,
            )
    elif model_type == "gb":
        estimator = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            random_state=random_state,
        )
    elif model_type == "rf":
        estimator = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=4,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    steps = []
    if preprocessor is not None:
        steps.append(("preprocess", preprocessor))
    steps.extend(
        [
            ("variance", VarianceThreshold(threshold=0.0)),
            ("model", estimator),
        ]
    )
    pipeline = Pipeline(steps)

    n_splits = min(cv_folds, len(X_clean))
    if n_splits < 2:
        raise ValueError("Not enough samples for cross-validation.")
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_scores_r2 = cross_val_score(pipeline, X_clean, y, cv=cv, scoring="r2")
    cv_scores_mae = cross_val_score(pipeline, X_clean, y, cv=cv, scoring="neg_mean_absolute_error")

    print(f"Cross-validation R²: {cv_scores_r2.mean():.3f} ± {cv_scores_r2.std():.3f}")
    print(f"Cross-validation MAE: {-cv_scores_mae.mean():.1f} ± {cv_scores_mae.std():.1f} days")

    pipeline.fit(X_clean, y)
    estimator_fitted = pipeline.named_steps["model"]

    y_pred = pipeline.predict(X_clean)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mape = float(np.mean(np.abs((y - y_pred) / y)) * 100.0)

    feature_names_out = _get_pipeline_feature_names(pipeline, numeric_cols, categorical_cols)
    if hasattr(estimator_fitted, "feature_importances_"):
        feature_importance = (
            pd.DataFrame(
                {
                    "feature": feature_names_out,
                    "importance": estimator_fitted.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    else:
        feature_importance = None

    results: Dict[str, Any] = {
        "model": pipeline,
        "model_type": model_type,
        "cv_scores_r2": cv_scores_r2,
        "cv_r2_mean": float(cv_scores_r2.mean()),
        "cv_r2_std": float(cv_scores_r2.std()),
        "cv_mae_mean": float(-cv_scores_mae.mean()),
        "cv_mae_std": float(cv_scores_mae.std()),
        "train_r2": float(r2),
        "train_mae": float(mae),
        "train_rmse": float(rmse),
        "train_mape": mape,
        "feature_importance": feature_importance,
        "feature_names": feature_names_out.tolist(),
        "raw_feature_names": (numeric_cols + categorical_cols),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "n_samples": int(len(X_clean)),
        "y_mean": float(y.mean()),
        "y_std": float(y.std()),
        "y_min": float(y.min()),
        "y_max": float(y.max()),
    }

    return results


def predict_time_to_response(
    model_bundle: Dict[str, Any],
    profile: pd.DataFrame,
    return_bounds: bool = True,
) -> Dict[str, float]:
    """
    Predict time to response for a patient profile.
    """
    if isinstance(profile, dict):
        profile_df = pd.DataFrame([profile])
    else:
        profile_df = profile.copy()

    pipeline: Pipeline = model_bundle["model"]
    numeric_cols: List[str] = model_bundle.get("numeric_features", [])
    categorical_cols: List[str] = model_bundle.get("categorical_features", [])
    ordered_cols: List[str] = model_bundle.get("raw_feature_names", numeric_cols + categorical_cols)

    X = _prepare_prediction_frame(profile_df, numeric_cols, categorical_cols, ordered_cols)
    ttr_pred = float(pipeline.predict(X)[0])

    result = {"predicted_ttr": ttr_pred}

    if return_bounds:
        estimator = pipeline.named_steps["model"]
        if hasattr(estimator, "estimators_"):
            preds = np.array([tree.predict(X)[0] for tree in estimator.estimators_])
            pred_std = float(np.std(preds))
            result["prediction_std"] = pred_std
            result["lower_95ci"] = float(ttr_pred - 1.96 * pred_std)
            result["upper_95ci"] = float(ttr_pred + 1.96 * pred_std)

    return result


def save_ttr_model(results: Dict[str, Any], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(results, output_path)
    print(f"TTR model saved to: {output_path}")


def load_ttr_model(model_path: str) -> Dict[str, Any]:
    results = joblib.load(model_path)
    print(f"Loaded TTR model from: {model_path}")
    print(f"  Model type: {results.get('model_type')}")
    print(f"  CV R²: {results.get('cv_r2_mean', float('nan')):.3f}")
    print(f"  CV MAE: {results.get('cv_mae_mean', float('nan')):.1f} days")
    return results


def plot_ttr_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Optional[str] = None,
):
    import matplotlib.pyplot as plt

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.5, color="steelblue")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction")
    ax.set_xlabel("Actual TTR (days)")
    ax.set_ylabel("Predicted TTR (days)")
    ax.set_title(f"Time-to-Response Predictions\nR² = {r2:.3f}, MAE = {mae:.1f} days")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, alpha=0.5, color="coral")
    ax.axhline(0, color="r", linestyle="--")
    ax.set_xlabel("Predicted TTR (days)")
    ax.set_ylabel("Residuals (days)")
    ax.set_title("Residuals Plot")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"TTR prediction plot saved to: {output_path}")

    plt.close(fig)


def plot_ttr_feature_importance(
    feature_importance: pd.DataFrame,
    top_n: int = 20,
    output_path: Optional[str] = None,
):
    import matplotlib.pyplot as plt

    if feature_importance is None or feature_importance.empty:
        print("No feature importance data available")
        return

    top_features = feature_importance.head(top_n)

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features["importance"].values, color="coral")
    plt.yticks(range(len(top_features)), top_features["feature"].values)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Features for Time-to-Response Prediction")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"TTR feature importance plot saved to: {output_path}")

    plt.close()
